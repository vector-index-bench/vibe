import platform

from ctypes import *
from ctypes.util import find_library


def __setup__():
    arch = platform.machine().lower()
    if arch in {"x86_64", "amd64", "i386", "i686", "x86"}:
        numa_num_nodes = 128
    else:
        numa_num_nodes = 2048

    ncpubits = sizeof(c_ulong) * 8
    cpu_setsize = 1024

    return {
        "NUMA_NUM_NODES": numa_num_nodes,
        "CPU_SETSIZE": cpu_setsize,
        "NCPUBITS": ncpubits,
    }


globals().update(__setup__())


class bitmask_t(Structure):
    _fields_ = [
        ("size", c_ulong),
        ("maskp", POINTER(c_ulong)),
    ]


class nodemask_t(Structure):
    _fields_ = [("n", c_ulong * (NUMA_NUM_NODES // (sizeof(c_ulong) * 8)))]


class cpu_set_t(Structure):
    _fields_ = [("__bits", c_ulong * (CPU_SETSIZE // NCPUBITS))]


libname = "libnuma"
if platform.system() == "Darwin":
    libname = find_library("numa") or "libnuma.dylib"
else:
    libname = find_library("numa") or "libnuma.so"

libnuma = CDLL(libname, use_errno=True)

libnuma.numa_available.argtypes = []
libnuma.numa_available.restype = c_int

libnuma.numa_max_node.argtypes = []
libnuma.numa_max_node.restype = c_int

libnuma.numa_node_size64.argtypes = [c_int, POINTER(c_longlong)]
libnuma.numa_node_size64.restype = c_longlong

libnuma.numa_preferred.argtypes = []
libnuma.numa_preferred.restype = c_int

libnuma.numa_node_to_cpus.argtypes = [c_int, POINTER(bitmask_t)]
libnuma.numa_node_to_cpus.restype = c_int

libnuma.numa_set_interleave_mask.argtypes = [POINTER(bitmask_t)]
libnuma.numa_set_interleave_mask.restype = c_void_p

libnuma.numa_get_interleave_mask.argtypes = []
libnuma.numa_get_interleave_mask.restype = POINTER(bitmask_t)

libnuma.numa_bitmask_clearall.argtypes = [POINTER(bitmask_t)]
libnuma.numa_bitmask_clearall.restype = POINTER(bitmask_t)

libnuma.copy_bitmask_to_nodemask.argtypes = [POINTER(bitmask_t), POINTER(nodemask_t)]
libnuma.copy_bitmask_to_nodemask.restype = c_void_p

libnuma.copy_nodemask_to_bitmask.argtypes = [POINTER(nodemask_t), POINTER(bitmask_t)]
libnuma.copy_nodemask_to_bitmask.restype = c_void_p

libnuma.numa_bitmask_free.argtypes = [POINTER(bitmask_t)]
libnuma.numa_bitmask_free.restype = c_void_p

libnuma.numa_allocate_nodemask.argtypes = []
libnuma.numa_allocate_nodemask.restype = POINTER(bitmask_t)

libnuma.numa_bind.argtypes = [POINTER(bitmask_t)]
libnuma.numa_bind.restype = c_void_p

libnuma.numa_set_membind.argtypes = [POINTER(bitmask_t)]
libnuma.numa_set_membind.restype = c_void_p

libnuma.numa_get_membind.argtypes = []
libnuma.numa_get_membind.restype = POINTER(bitmask_t)

libnuma.numa_set_preferred.argtypes = [c_int]
libnuma.numa_set_preferred.restype = c_void_p

libnuma.numa_set_localalloc.argtypes = []
libnuma.numa_set_localalloc.restype = c_void_p

libnuma.numa_get_run_node_mask.argtypes = []
libnuma.numa_get_run_node_mask.restype = POINTER(bitmask_t)

libnuma.numa_run_on_node_mask.argtypes = [POINTER(bitmask_t)]
libnuma.numa_run_on_node_mask.restype = c_int

libnuma.numa_distance.argtypes = [c_int, c_int]
libnuma.numa_distance.restype = c_int


def available():
    """
    Is numa(3) available?

    @rtype: C{bool}
    """
    return libnuma.numa_available() != -1


def get_max_node():
    """
    Maximum number of NUMA node.

    @rtype: C{int}
    """
    return libnuma.numa_max_node()


def get_node_size(node):
    """
    Get size of memory on C{node}.

    @param node: node idx
    @type node: C{int}
    @return: free memory/total memory
    @rtype: C{tuple}(C{int}, C{int})
    """

    free = c_longlong()

    if node < 0 or node > get_max_node():
        raise ValueError(node)

    size = libnuma.numa_node_size64(node, byref(free))

    return (free.value, size)


def get_preferred():
    """
    Return preferred node for this process.

    @rtype: C{int}
    """
    return libnuma.numa_preferred()


def node_to_cpus(node):
    """
    Get CPUs available on C{node}.

    @return: set of CPU ids
    @rtype: C{set}
    """

    result = set()

    if node < 0 or node > get_max_node():
        raise ValueError(node)

    mask = bitmask_t()
    mask.maskp = (c_ulong * (NUMA_NUM_NODES // (sizeof(c_ulong) * 8)))()
    mask.size = NUMA_NUM_NODES

    if libnuma.numa_node_to_cpus(node, byref(mask)) < 0:
        raise RuntimeError(node)

    for i in range(0, NUMA_NUM_NODES // (sizeof(c_ulong) * 8)):
        for j in range(0, sizeof(c_ulong) * 8):
            if mask.maskp[i] & (1 << j) == (1 << j):
                result.add(i * sizeof(c_ulong) * 8 + j)

    return result


def __nodemask_isset(mask, node):
    if node >= NUMA_NUM_NODES:
        return 0

    if mask.n[node // (8 * sizeof(c_ulong))] & (1 << (node % (8 * sizeof(c_ulong)))):
        return 1

    return 0


def __nodemask_set(mask, node):
    mask.n[node // (8 * sizeof(c_ulong))] |= 1 << (node % (8 * sizeof(c_ulong)))


def numa_nodemask_to_set(mask):
    """
    Convert NUMA nodemask to Python set.
    """
    result = set()

    for i in range(0, get_max_node() + 1):
        if __nodemask_isset(mask, i):
            result.add(i)

    return result


def __nodemask_zero(mask):
    tmp = bitmask_t()
    tmp.maskp = cast(byref(mask), POINTER(c_ulong))
    tmp.size = sizeof(nodemask_t) * 8
    libnuma.numa_bitmask_clearall(byref(tmp))


def set_to_numa_nodemask(mask):
    """
    Conver Python set to NUMA nodemask.
    """
    result = nodemask_t()
    __nodemask_zero(result)

    for i in range(0, get_max_node() + 1):
        if i in mask:
            __nodemask_set(result, i)

    return result


def set_interleave_mask(nodemask):
    """
    Sets the memory interleave mask for the current thread to C{nodemask}.

    @param nodemask: node mask
    @type nodemask: C{set}
    """

    mask = set_to_numa_nodemask(nodemask)
    tmp = bitmask_t()
    tmp.maskp = cast(byref(mask), POINTER(c_ulong))
    tmp.size = sizeof(nodemask_t) * 8

    libnuma.numa_set_interleave_mask(byref(tmp))


def get_interleave_mask():
    """
    Get interleave mask for current thread.

    @return: node mask
    @rtype: C{set}
    """

    nodemask = nodemask_t()
    bitmask = libnuma.numa_get_interleave_mask()
    libnuma.copy_bitmask_to_nodemask(bitmask, byref(nodemask))
    libnuma.numa_bitmask_free(bitmask)

    return numa_nodemask_to_set(nodemask)


def bind(nodemask):
    """
    Binds the current thread and its children to the nodes specified in nodemask.
    They will only run on the CPUs of the specified nodes and only be able to allocate memory from them.

    @param nodemask: node mask
    @type nodemask: C{set}
    """
    mask = set_to_numa_nodemask(nodemask)
    bitmask = libnuma.numa_allocate_nodemask()
    libnuma.copy_nodemask_to_bitmask(byref(mask), bitmask)
    libnuma.numa_bind(bitmask)
    libnuma.numa_bitmask_free(bitmask)


def set_preferred(node):
    """
    Sets  the preferred node for the current thread to node.

    The preferred node is the node on which memory is preferably allocated before falling back to other
    nodes. The default is to use the node on which the process is currently running (local policy).

    @param node: node idx
    @type node: C{int}
    """
    if node < 0 or node > get_max_node():
        raise ValueError(node)

    libnuma.numa_set_preferred(node)


def set_localalloc():
    """
    Sets a local memory allocation policy for the calling thread.

    Memory is preferably allocated on the node on which the thread is currently running.
    """
    libnuma.numa_set_localalloc()


def set_membind(nodemask):
    """
    Sets the memory allocation mask.

    The thread will only allocate memory from the nodes set in nodemask.

    @param nodemask: node mask
    @type nodemask: C{set}
    """
    mask = set_to_numa_nodemask(nodemask)

    tmp = bitmask_t()
    tmp.maskp = cast(byref(mask), POINTER(c_ulong))
    tmp.size = sizeof(nodemask_t) * 8

    libnuma.numa_set_membind(byref(tmp))


def get_membind():
    """
    Returns  the  mask of nodes from which memory can currently be allocated.

    @return: node mask
    @rtype: C{set}
    """
    bitmask = libnuma.numa_get_membind()
    nodemask = nodemask_t()
    libnuma.copy_bitmask_to_nodemask(bitmask, byref(nodemask))
    libnuma.numa_bitmask_free(bitmask)
    return numa_nodemask_to_set(nodemask)


def set_run_on_node_mask(nodemask):
    """
    Runs the  current thread and its children only on nodes specified in nodemask.

    They will not migrate to CPUs of other nodes until the node affinity is
    reset with a new call to L{set_run_on_node_mask}.

    @param nodemask: node mask
    @type nodemask: C{set}
    """
    mask = set_to_numa_nodemask(nodemask)

    tmp = bitmask_t()
    tmp.maskp = cast(byref(mask), POINTER(c_ulong))
    tmp.size = sizeof(nodemask_t) * 8

    if libnuma.numa_run_on_node_mask(byref(tmp)) < 0:
        raise RuntimeError()


def get_run_on_node_mask():
    """
    Returns the mask of nodes that the current thread is allowed to run on.

    @return: node mask
    @rtype: C{set}
    """
    bitmask = libnuma.numa_get_run_node_mask()
    nodemask = nodemask_t()
    libnuma.copy_bitmask_to_nodemask(bitmask, byref(nodemask))
    libnuma.numa_bitmask_free(bitmask)

    return numa_nodemask_to_set(nodemask)


def get_distance(node1, node2):
    """
    Reports the distance in the machine topology between two nodes.

    The factors are a multiple of 10. It returns 0 when the distance cannot be determined. A node  has
    distance 10 to itself.  Reporting the distance requires a Linux kernel version of 2.6.10 or newer.

    @param node1: node idx
    @type node1: C{int}
    @param node2: node idx
    @type node2: C{int}
    @rtype: C{int}
    """
    if node1 < 0 or node1 > get_max_node():
        raise ValueError(node1)
    if node2 < 0 or node2 > get_max_node():
        raise ValueError(node2)

    return libnuma.numa_distance(node1, node2)


def get_affinity(pid):
    """
    Returns the affinity mask of the process whose ID is pid.

    @param pid: process PID (0 == current process)
    @type pid: C{int}
    @return: set of CPU ids
    @rtype: C{set}
    """

    cpuset = cpu_set_t()
    result = set()

    libnuma.sched_getaffinity(pid, sizeof(cpu_set_t), byref(cpuset))

    for i in range(0, sizeof(cpu_set_t) * 8):
        if __CPU_ISSET(i, cpuset):
            result.add(i)

    return result


def __CPU_ZERO(cpuset):
    memset(byref(cpuset), 0, sizeof(cpu_set_t))


def __CPU_SET(cpu, cpuset):
    if cpu < sizeof(cpu_set_t) * 8:
        cpuset.__bits[cpu // (sizeof(c_ulong) * 8)] |= 1 << (cpu % (sizeof(c_ulong) * 8))


def __CPU_ISSET(cpu, cpuset):
    if cpu < sizeof(cpu_set_t) * 8:
        return cpuset.__bits[cpu // (sizeof(c_ulong) * 8)] & (1 << (cpu % (sizeof(c_ulong) * 8)))
    else:
        return 0


def set_affinity(pid, cpuset):
    """
    Sets  the  CPU  affinity  mask of the process whose ID is pid to the value specified by mask.

    If pid is zero, then the calling process is used.

    @param pid: process PID (0 == current process)
    @type pid: C{int}
    @param cpuset: set of CPU ids
    @type cpuset: C{set}
    """
    _cpuset = cpu_set_t()
    __CPU_ZERO(_cpuset)

    for i in cpuset:
        if i in range(0, sizeof(cpu_set_t) * 8):
            __CPU_SET(i, _cpuset)

    if libnuma.sched_setaffinity(pid, sizeof(cpu_set_t), byref(_cpuset)) < 0:
        raise RuntimeError()
