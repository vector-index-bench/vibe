import argparse
import json
import logging
import multiprocessing
import os
import queue as pyqueue
import shutil
import signal
import subprocess
import threading
import time
from pathlib import Path
from typing import List, Iterable, Tuple, Optional, Set, Dict
from collections import defaultdict, deque

from .definitions import Definition

try:
    from . import numa

    HAS_NUMA = numa.available()
except Exception:
    HAS_NUMA = False


logger = logging.getLogger("vibe")


def _expand_ranges(spec: str) -> Set[int]:
    out = set()
    for tok in spec.split(","):
        tok = tok.strip()
        if not tok:
            continue
        if "-" in tok:
            a_str, b_str = tok.split("-", 1)
            a, b = int(a_str), int(b_str)
            if a > b:
                a, b = b, a
            out.update(range(a, b + 1))
        else:
            out.add(int(tok))
    return out


_numa_node_cache: Dict[int, Optional[int]] = {}


def get_numa_node_for_cpu(cpu: int) -> Optional[int]:
    if cpu in _numa_node_cache:
        return _numa_node_cache[cpu]

    if HAS_NUMA:
        max_node = numa.get_max_node()
        for node in range(max_node + 1):
            cpus_on_node = numa.node_to_cpus(node)
            if cpu in cpus_on_node:
                _numa_node_cache[cpu] = node
                return node

    nodes_root = Path("/sys/devices/system/node")
    if nodes_root.exists():
        for node_dir in sorted(nodes_root.glob("node[0-9]*"), key=lambda p: int(p.name[4:])):
            cpulist = node_dir / "cpulist"
            try:
                cpus = _expand_ranges(cpulist.read_text().strip())
            except Exception:
                continue
            if cpu in cpus:
                n = int(node_dir.name[4:])
                _numa_node_cache[cpu] = n
                return n

    _numa_node_cache[cpu] = None
    return None


def allowed_mems() -> Optional[Set[int]]:
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("Mems_allowed_list:"):
                    return _expand_ranges(line.split(":", 1)[1].strip())
    except Exception:
        pass

    return None


def get_physical_cores() -> List[int]:
    def read_cpu_list(path: Path) -> Optional[Set[int]]:
        return _expand_ranges(path.read_text().strip()) if path.exists() else None

    allowed = set(os.sched_getaffinity(0))
    online_path = Path("/sys/devices/system/cpu/online")
    if online_path.exists():
        allowed &= _expand_ranges(online_path.read_text().strip())

    reps, seen = [], set()
    for cpu in sorted(allowed):
        topo = Path(f"/sys/devices/system/cpu/cpu{cpu}/topology")

        sibs = read_cpu_list(topo / "core_cpus_list")
        if sibs is None:
            sibs = read_cpu_list(topo / "thread_siblings_list")
        if sibs is None:
            sibs = {cpu}

        sibs &= allowed
        if not sibs:
            continue

        rep = min(sibs)
        if rep not in seen:
            seen.add(rep)
            reps.append(rep)

    return reps


def filter_cores_by_allowed_mems(cores: List[int]) -> List[int]:
    return [c for c in cores if get_numa_node_for_cpu(c) in allowed_mems()]


def _numa_sort_key(node: Optional[int]) -> int:
    return node if node is not None else (1 << 30)


def interleave_by_numa(cpus: Iterable[int]) -> List[int]:
    by_node: Dict[Optional[int], deque[int]] = defaultdict(deque)
    for cpu in sorted(cpus):
        node = get_numa_node_for_cpu(cpu)
        by_node[node].append(cpu)

    if not by_node:
        return []

    ordered_nodes = sorted(by_node.keys(), key=_numa_sort_key)

    out: List[int] = []
    remaining = sum(len(q) for q in by_node.values())

    while remaining:
        for node in ordered_nodes:
            q = by_node[node]
            if q:
                out.append(q.popleft())
                remaining -= 1

    return out


def reserve_one_core_per_numa(cores: List[int]) -> Tuple[List[int], List[int]]:
    if not cores:
        return [], []

    by_node: Dict[Optional[int], List[int]] = defaultdict(list)
    for c in sorted(cores):
        by_node[get_numa_node_for_cpu(c)].append(c)

    num_nodes = len(by_node)
    if len(cores) <= num_nodes:
        reserved = [cores[-1]]
        workers = [c for c in cores if c != reserved[0]]
        return reserved, interleave_by_numa(workers)

    reserved: List[int] = []
    for node in sorted(by_node.keys(), key=_numa_sort_key):
        lst = by_node[node]
        reserved.append(lst[-1])

    workers = [c for c in cores if c not in set(reserved)]
    workers = interleave_by_numa(workers)

    return reserved, workers


def kill_process(p):
    if hasattr(p, "kill"):
        p.kill()
    else:
        try:
            os.kill(p.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass


def run_singularity(
    definition: Definition,
    dataset: str,
    count: int,
    runs: int,
    timeout: int,
    cpu: int,
    local: bool,
    terminate_evt,
) -> None:
    """Runs `run_from_cmdline` within a Singularity container with specified parameters and logs the output.

    See `run_from_cmdline` for details on the args.
    """
    if terminate_evt.is_set():
        return

    logger = logging.getLogger(f"vibe.worker.{cpu}")

    gov_path = Path(f"/sys/devices/system/cpu/cpu{cpu}/cpufreq/scaling_governor")
    if gov_path.exists() and gov_path.read_text().strip() != "performance":
        logger.warning(f"Core {cpu} not in performance mode")

    cmd = []

    node = get_numa_node_for_cpu(cpu)
    allowed = allowed_mems()

    if allowed is None:
        logger.error("No allowed memory nodes")
        return

    def _nearest_allowed_node(src_node: int, allowed_nodes: Set[int]) -> int:
        dist_path = Path(f"/sys/devices/system/node/node{src_node}/distance")
        if not allowed_nodes:
            raise RuntimeError("nearest_allowed_node called with empty allowed_nodes")
        if not dist_path.exists():
            return min(allowed_nodes)
        distances = [int(x) for x in dist_path.read_text().split()]
        valid_allowed = {n for n in allowed_nodes if 0 <= n < len(distances)}
        if not valid_allowed:
            return min(allowed_nodes)
        return min(valid_allowed, key=lambda n: (distances[n], n))

    if shutil.which("numactl"):
        if node is not None and (allowed is None or node in allowed):
            cmd += ["numactl", f"--membind={node}"]
        elif allowed:
            base = node if node is not None else 0
            nearest = _nearest_allowed_node(base, allowed)
            cmd += ["numactl", f"--membind={nearest}"]
        cmd += [f"--physcpubind={cpu}"]
    else:
        logger.warning("numactl not found; falling back to taskset (no NUMA policy).")
        cmd += ["taskset", "-c", str(cpu)]

    if not local:
        cmd += ["singularity", "exec"]
        if definition.gpu:
            cmd += ["--nv"]
        cmd += [f"images/{definition.singularity_image}.sif"]

    cmd += [
        "python3",
        "-u",
        "run_algorithm.py",
        "--dataset",
        dataset,
        "--algorithm",
        definition.algorithm,
        "--module",
        definition.module,
        "--constructor",
        definition.constructor,
        "--runs",
        str(runs),
        "--count",
        str(count),
    ]
    if definition.gpu:
        cmd += ["--gpu"]
    if definition.ood:
        cmd += ["--ood"]
    cmd.append(json.dumps(definition.arguments))
    cmd += [json.dumps(qag) for qag in definition.query_argument_groups]

    env = dict(
        os.environ,
        OMP_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
        OPENBLAS_NUM_THREADS="1",
        TBB_NUM_THREADS="1",
        NUMEXPR_NUM_THREADS="1",
        VECLIB_NUM_THREADS="1",
        BLIS_NUM_THREADS="1",
        GOTO_NUM_THREADS="1",
        NUMBA_NUM_THREADS="1",
        JULIA_NUM_THREADS="1",
        RAYON_NUM_THREADS="1",
        OMP_PROC_BIND="true",
        OMP_DYNAMIC="false",
        MKL_DYNAMIC="false",
    )

    def set_affinity_before_exec():
        os.sched_setaffinity(0, {cpu})

    try:
        with subprocess.Popen(
            cmd,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            bufsize=1,
            start_new_session=True,
            close_fds=True,
            preexec_fn=set_affinity_before_exec,
        ) as p:
            pgid = p.pid

            logger.info(f"Started process PID={p.pid}")

            reader_done = threading.Event()

            def stream_lines():
                try:
                    for line in p.stdout:
                        logger.info(line.rstrip())
                except Exception:
                    pass
                finally:
                    reader_done.set()

            reader = threading.Thread(target=stream_lines, daemon=False)
            reader.start()

            if terminate_evt.is_set():
                try:
                    os.killpg(pgid, signal.SIGTERM)
                except ProcessLookupError:
                    pass

            def stopper():
                while True:
                    if p.poll() is not None:
                        return

                    if terminate_evt.is_set():
                        if p.poll() is None:
                            logger.info(f"Sending SIGTERM to PGID {pgid}")
                            try:
                                os.killpg(pgid, signal.SIGTERM)
                            except ProcessLookupError:
                                pass
                            except Exception as e:
                                logger.warning(f"Failed to kill PGID {pgid}: {e}")
                        return

                    time.sleep(1.0)

            stopper_thread = threading.Thread(target=stopper, daemon=True)
            stopper_thread.start()

            timeout = timeout if timeout > 0 else None
            killed_by_timeout = False

            try:
                rc = p.wait(timeout=timeout)
                logger.info(f"Process completed with return code: {rc}")

            except subprocess.TimeoutExpired:
                killed_by_timeout = True

                logger.warning(f"Process {p.pid} exceeded timeout of {timeout}s")
                try:
                    os.killpg(pgid, signal.SIGTERM)
                except ProcessLookupError:
                    pass

                try:
                    p.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    logger.warning(f"Process {p.pid} unresponsive to SIGTERM, sending SIGKILL")
                    try:
                        os.killpg(pgid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                    try:
                        p.wait(timeout=1)
                    except subprocess.TimeoutExpired:
                        logger.error(f"Process {p.pid} could not be killed")

            finally:
                stopper_thread.join(timeout=0.5)

                if terminate_evt.is_set() and not killed_by_timeout:
                    try:
                        p.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        try:
                            os.killpg(pgid, signal.SIGKILL)
                            p.wait(timeout=1)
                        except (ProcessLookupError, subprocess.TimeoutExpired):
                            pass

                try:
                    if p.stdout:
                        p.stdout.close()
                except Exception:
                    pass

                reader_done.wait(timeout=2.0)

                if reader.is_alive():
                    try:
                        if p.stdout:
                            p.stdout.close()
                    except Exception:
                        pass
                    reader.join(timeout=1.0)
                    if reader.is_alive():
                        logger.warning("Output streaming thread did not terminate cleanly")
    except FileNotFoundError as e:
        logger.error(f"Failed to launch command: {e}")
        return


def run_worker(cpu, args, queue, terminate_evt) -> None:
    logger = logging.getLogger("vibe")

    try:
        while not terminate_evt.is_set():
            try:
                definition = queue.get(timeout=1.0)
            except pyqueue.Empty:
                continue

            if definition is None or terminate_evt.is_set():
                break

            try:
                run_singularity(
                    definition,
                    args.dataset,
                    args.count,
                    args.runs,
                    args.timeout,
                    cpu,
                    args.local,
                    terminate_evt=terminate_evt,
                )
            except Exception as e:
                logger.warning(f"Worker failed on CPU {cpu}: {e}")
    finally:
        pass


def backgroundize_orchestrator() -> None:
    try:
        if hasattr(os, "SCHED_BATCH"):
            os.sched_setscheduler(0, os.SCHED_BATCH, os.sched_param(0))
            logger.info("Orchestrator: set scheduling policy to SCHED_BATCH.")
        else:
            logger.info("Orchestrator: SCHED_BATCH not available on this Python/OS.")
    except PermissionError as e:
        logger.warning(f"Orchestrator: could not set SCHED_BATCH (permission): {e}")
    except Exception as e:
        logger.warning(f"Orchestrator: failed to set SCHED_BATCH: {e}")

    ionice = shutil.which("ionice")
    if ionice:
        try:
            subprocess.run(
                [ionice, "-c3", "-p", str(os.getpid())],
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            logger.info("Orchestrator: IO priority set to idle (ionice -c3).")
        except subprocess.CalledProcessError:
            subprocess.run(
                [ionice, "-c2", "-n", "7", "-p", str(os.getpid())],
                check=False,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            logger.info("Orchestrator: IO priority set to best-effort level 7 (ionice -c2 -n7).")
        except Exception as e:
            logger.warning(f"Orchestrator: ionice failed: {e}")
    else:
        logger.info("Orchestrator: 'ionice' not found; skipping IO priority change.")

    try:
        os.nice(10)
        logger.info("Orchestrator: CPU niceness increased by +10.")
    except Exception as e:
        logger.warning(f"Orchestrator: could not renice: {e}")


def create_workers_and_execute(definitions: List[Definition], args: argparse.Namespace):
    """
    Manages the creation, execution, and termination of worker processes based on provided arguments.

    Args:
        definitions (List[Definition]): List of algorithm definitions to be processed.
        args (argparse.Namespace): User provided arguments for running workers.

    Raises:
        Exception: If the level of parallelism exceeds the available CPU count or if batch mode is on with more than
                   one worker.
    """
    cores = get_physical_cores()
    cores = filter_cores_by_allowed_mems(cores)

    if not cores:
        raise RuntimeError("No available CPU cores")

    reserved_cpus, worker_cpus = reserve_one_core_per_numa(cores)

    os.sched_setaffinity(0, set(reserved_cpus))

    if args.parallelism > len(worker_cpus):
        raise Exception(
            f"Parallelism ({args.parallelism}) > usable physical cores ({len(worker_cpus)}); "
            f"reserved {len(reserved_cpus)} core(s), one per NUMA node"
        )

    num_workers = min(args.parallelism, len(worker_cpus), len(definitions))

    terminate_evt = multiprocessing.Event()
    task_queue = multiprocessing.Queue()

    procs = []
    for cpu in worker_cpus[:num_workers]:
        p = multiprocessing.Process(target=run_worker, args=(cpu, args, task_queue, terminate_evt))
        p.start()
        procs.append(p)

    for definition in definitions:
        task_queue.put(definition)
    for _ in range(num_workers):
        task_queue.put(None)

    backgroundize_orchestrator()

    try:
        for p in procs:
            p.join()
    except KeyboardInterrupt:
        logger.info("Interrupt received, cleaning up workers...")

        terminate_evt.set()

        try:
            while not task_queue.empty():
                task_queue.get_nowait()
        except Exception:
            pass

        for _ in range(num_workers):
            try:
                task_queue.put_nowait(None)
            except Exception:
                pass

        deadline = time.time() + 5
        for p in procs:
            timeout = max(0.1, deadline - time.time())
            p.join(timeout=timeout)

        for p in procs:
            if p.is_alive():
                logger.warning(f"Force terminating worker {p.pid}")
                p.terminate()
                p.join(timeout=1)
                if p.is_alive():
                    kill_process(p)
                    p.join(timeout=0.5)

        raise

    finally:
        task_queue.close()
        task_queue.join_thread()

        for p in procs:
            if p.is_alive():
                kill_process(p)
            p.join(timeout=0.1)
