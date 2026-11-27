"""VIBE module for kci_vec: the partitioned 4-bit table-lookup index (KciVecIvf) and the int8 HNSW graph with the
quantized layer 0 (KciVecQg). Same ctypes bindings as kci_vec's ann-benchmarks module.

VIBE rules that differ from ANN-Benchmarks: every build is single-threaded (all kci_vec build calls get threads = 1), and
k = 100 by default (re-rank shortlists are at least 100). Metrics: cosine / normalized, euclidean (kci_vec_new_l2) and ip (kci_vec_new_ip).
The library is bound in-process with ctypes: KCI_VEC_LIB if set, else the library bundled in the installed kci_vec wheel
(libkci_vec_avx512.so or libkci_vec_avx2.so, picked by the CPU), else libkci_vec.so on the loader path.
"""
import ctypes as C
import os

import numpy as np

from ..base.module import BaseANN


def _lib_path():
    if os.environ.get("KCI_VEC_LIB"):
        return os.environ["KCI_VEC_LIB"]
    try:
        from kci_vec import _find_lib
        return _find_lib()
    except Exception:
        return "libkci_vec.so"


_LIB = _lib_path()
THREADS = 1   # VIBE: "All builds are single-threaded"


class _IvfParams(C.Structure):   # KciVecIvfParams
    _fields_ = [("nlist", C.c_int), ("threads", C.c_int), ("soar", C.c_float), ("code_dims", C.c_int), ("lowrank", C.c_int)]


def _load():
    lib = C.CDLL(_LIB)
    lib.kci_vec_new.restype = C.c_void_p
    lib.kci_vec_new.argtypes = [C.c_int, C.c_int, C.c_int]
    lib.kci_vec_new_l2.restype = C.c_void_p
    lib.kci_vec_new_l2.argtypes = [C.c_int, C.c_int, C.c_int, C.c_float]   # last: max length, 0 = found from the data
    if hasattr(lib, "kci_vec_learn"):                                       # wheels after 0.2.1
        lib.kci_vec_learn.argtypes = [C.c_void_p, C.c_void_p, C.c_uint32, C.c_void_p, C.c_int, C.c_int, C.c_int, C.c_int, C.c_void_p]
        lib.kci_vec_search_learned.argtypes = [C.c_void_p, C.c_void_p, C.c_int, C.c_int, C.c_int, C.c_int, C.c_void_p, C.c_void_p]
        lib.kci_vec_learn_pack.argtypes = [C.c_void_p]
        lib.kci_vec_searcher_set_learned_seeds.argtypes = [C.c_void_p, C.c_int]
        if hasattr(lib, "kci_vec_searcher_set_learned_pages"):
            lib.kci_vec_searcher_set_learned_pages.argtypes = [C.c_void_p, C.c_int]
        if hasattr(lib, "kci_vec_searcher_set_learned_hot"):
            lib.kci_vec_searcher_set_learned_hot.argtypes = [C.c_void_p, C.c_int]
    if hasattr(lib, "kci_vec_new_ip"):                                      # wheels after 0.2.1
        lib.kci_vec_new_ip.restype = C.c_void_p
        lib.kci_vec_new_ip.argtypes = [C.c_int, C.c_int, C.c_int, C.c_float]
    lib.kci_vec_free.argtypes = [C.c_void_p]
    lib.kci_vec_keep_float.argtypes = [C.c_void_p, C.c_int]
    lib.kci_vec_reserve.argtypes = [C.c_void_p, C.c_uint32]
    lib.kci_vec_add.argtypes = [C.c_void_p, C.c_uint64, C.c_void_p]
    lib.kci_vec_add.restype = C.c_int64
    lib.kci_vec_build.argtypes = [C.c_void_p, C.c_int, C.POINTER(C.c_double)]
    lib.kci_vec_build_ivf.argtypes = [C.c_void_p, C.c_int, C.c_int]
    lib.kci_vec_build_ivf_p.argtypes = [C.c_void_p, C.POINTER(_IvfParams)]
    lib.kci_vec_searcher_set_rerank_int8.argtypes = [C.c_void_p, C.c_int]
    lib.kci_vec_set_ivf_training.argtypes = [C.c_void_p, C.c_int, C.c_uint32]
    lib.kci_vec_searcher_new.restype = C.c_void_p
    lib.kci_vec_searcher_new.argtypes = [C.c_void_p]
    lib.kci_vec_searcher_free.argtypes = [C.c_void_p]
    lib.kci_vec_searcher_set_probe_gap.argtypes = [C.c_void_p, C.c_float, C.c_int]
    lib.kci_vec_searcher_set_eps.argtypes = [C.c_void_p, C.c_float]
    lib.kci_vec_build_qg.argtypes = [C.c_void_p, C.c_int, C.c_int]
    lib.kci_vec_build_qg_ex.argtypes = [C.c_void_p, C.c_int, C.c_int, C.c_int]              # 0.2.3
    lib.kci_vec_half_floats.argtypes = [C.c_void_p]
    lib.kci_vec_searcher_set_qg_exact.argtypes = [C.c_void_p, C.c_int]
    lib.kci_vec_searcher_set_qg.argtypes = [C.c_void_p, C.c_int]
    lib.kci_vec_search_r.argtypes = [C.c_void_p, C.c_void_p, C.c_int, C.c_int, C.c_void_p, C.c_void_p]
    lib.kci_vec_search_rerank.argtypes = [C.c_void_p, C.c_void_p, C.c_int, C.c_int, C.c_int, C.c_void_p, C.c_void_p]
    lib.kci_vec_search_ivf.argtypes = [C.c_void_p, C.c_void_p, C.c_int, C.c_int, C.c_int, C.c_void_p, C.c_void_p]
    return lib


class _Base(BaseANN):
    def _check_metric(self, metric):
        if metric not in ("cosine", "normalized", "euclidean", "ip"):
            raise NotImplementedError("kci_vec: cosine / normalized / euclidean / ip only")
        self._l2 = metric == "euclidean"
        self._ip = metric == "ip"

    def _new_index(self, X, M, efc):
        X = np.ascontiguousarray(X, dtype=np.float32)   # a copy when needed; X itself is not modified
        n, dim = X.shape
        if self._ip:
            if not hasattr(self._lib, "kci_vec_new_ip"):
                raise NotImplementedError("kci_vec: this library has no inner product (wheel 0.2.1 or older)")
            v = self._lib.kci_vec_new_ip(dim, M, efc, 0.0)
        else:
            v = self._lib.kci_vec_new_l2(dim, M, efc, 0.0) if self._l2 else self._lib.kci_vec_new(dim, M, efc)
        if not v:
            raise RuntimeError("kci_vec_new failed")
        self._lib.kci_vec_keep_float(v, 1)
        self._lib.kci_vec_reserve(v, n + 1)
        base, row = X.ctypes.data, dim * 4
        for i in range(n):
            self._lib.kci_vec_add(v, i, base + i * row)
        self._out_id = (C.c_uint64 * 4096)()
        self._out_sc = (C.c_float * 4096)()
        return v

    def _ids(self, got):
        return np.frombuffer(self._out_id, dtype=np.uint64, count=got).astype(np.int64)

    def __del__(self):
        try:
            if getattr(self, "_s", None):
                self._lib.kci_vec_searcher_free(self._s)
            if getattr(self, "_v", None):
                self._lib.kci_vec_free(self._v)
        except Exception:
            pass


class KciVecIvf(_Base):
    """Partitioned lists + 4-bit table lookup, re-rank of the best `rerank`.
    soar: 0 = one list per vector, > 0 = every vector also in a second list (SOAR; the value is its weight).
    cdiv: 1 = the codes cover all dims; n > 1 = the codes cover the top dim / n principal directions of the residuals
    (fewer bytes to scan per vector; the re-rank restores the precision). For high dimensions.
    train_sample: 0 = default sampled centroid training; -1 = all vectors, 30 k-means iterations; n > 0 = n samples, 30 it.
    i8 (query time): 0 = re-rank on the float vectors, 1 = on the int8 rows (faster, less exact)."""

    def __init__(self, metric, nlist, soar, cdiv, train_sample):
        self._check_metric(metric)
        self._lib = _load()
        self.nlist, self.soar, self.cdiv, self.train_sample = int(nlist), float(soar), int(cdiv), int(train_sample)
        self.nprobe, self.rerank, self.gap, self.i8 = 1, 100, 0.0, 0
        self._v = self._s = None

    def fit(self, X):
        self._v = self._new_index(X, 16, 200)
        if self.train_sample != 0:
            self._lib.kci_vec_set_ivf_training(self._v, 30, 0xFFFFFFFF if self.train_sample < 0 else self.train_sample)
        dim = X.shape[1]
        cd = (dim // self.cdiv) // 8 * 8 if self.cdiv > 1 else 0   # code_dims: even, at least 8, below dim; 0 = all dims
        p = _IvfParams(self.nlist, THREADS, self.soar, cd if cd >= 8 else 0, 0)
        if self._lib.kci_vec_build_ivf_p(self._v, C.byref(p)) != 0:
            raise RuntimeError("kci_vec_build_ivf_p failed")
        self._s = self._lib.kci_vec_searcher_new(self._v)

    def set_query_arguments(self, nprobe, rerank, gap, i8):
        self.nprobe, self.rerank, self.gap, self.i8 = int(nprobe), int(rerank), float(gap), int(i8)
        self._lib.kci_vec_searcher_set_probe_gap(self._s, self.gap, 8)
        self._lib.kci_vec_searcher_set_rerank_int8(self._s, self.i8)

    def query(self, v, n):
        q = np.ascontiguousarray(v, dtype=np.float32)
        got = self._lib.kci_vec_search_ivf(self._s, q.ctypes.data, n, self.nprobe, max(self.rerank, n), self._out_id, self._out_sc)
        return self._ids(got)

    def __str__(self):
        return (f"KciVecIvf(nlist={self.nlist},soar={self.soar},cdiv={self.cdiv},train_sample={self.train_sample},"
                f"nprobe={self.nprobe},rerank={self.rerank},gap={self.gap},i8={self.i8})")


class KciVecQg(_Base):
    """int8 HNSW graph whose layer 0 is stored as 4-bit neighbour blocks (kci_vec_build_qg), exploration radius eps,
    re-rank of the result set. dpc: dimensions per 4-bit code (1, or 2 = blocks half the size, coarser).
    The vectors for the exact pass are kept as 16-bit halves (kci_vec_half_floats). From 1,000 dimensions up
    (cosine / euclidean) the codes cover the top 3/8 of the principal directions (kci_vec_build_qg_ex), so a block is
    smaller, and r8 = 2 scores the candidates that pass the estimate with short int8 rows.
    r8 (query time): 0 = exact pass over the whole result set (ef), 1 = int8 scores only, 2 = exact pass over the best
    2k of the set."""

    def __init__(self, metric, M, efConstruction, dpc):
        self._check_metric(metric)
        self._lib = _load()
        self.M, self.efc, self.dpc, self.hd = int(M), int(efConstruction), int(dpc), 0
        self.ef, self.eps, self.r8 = 100, 0.0, 0
        self._v = self._s = None

    def fit(self, X):
        self._v = self._new_index(X, self.M, self.efc)
        secs = C.c_double(0.0)
        if self._lib.kci_vec_build(self._v, THREADS, C.byref(secs)) != 0:
            raise RuntimeError("kci_vec_build failed")
        self.hd = int(X.shape[1] >= 1000 and not self._ip)
        if self.hd:
            cd = (X.shape[1] * 3 // 8) // 2 * 2
            if self._lib.kci_vec_build_qg_ex(self._v, self.dpc, cd, THREADS) != 0:
                raise RuntimeError("kci_vec_build_qg_ex failed")
        elif self._lib.kci_vec_build_qg(self._v, self.dpc, THREADS) != 0:
            raise RuntimeError("kci_vec_build_qg failed")
        if self._lib.kci_vec_half_floats(self._v) != 0:
            raise RuntimeError("kci_vec_half_floats failed")
        self._s = self._lib.kci_vec_searcher_new(self._v)
        if self._lib.kci_vec_searcher_set_qg(self._s, 1) != 0:
            raise RuntimeError("kci_vec_searcher_set_qg failed")

    def set_query_arguments(self, ef, eps, r8):
        self.ef, self.eps, self.r8 = int(ef), float(eps), int(r8)
        self._lib.kci_vec_searcher_set_eps(self._s, self.eps)
        # the short rows only where their ranking is followed by a narrow exact pass; else the estimates as in 0.2.2
        self._lib.kci_vec_searcher_set_qg_exact(self._s, 2 if (self.hd and self.r8 == 2) else 0)

    def query(self, v, n):
        q = np.ascontiguousarray(v, dtype=np.float32)
        ef = max(self.ef, n)
        if self.r8 == 1:
            got = self._lib.kci_vec_search_r(self._s, q.ctypes.data, n, ef, self._out_id, self._out_sc)
        else:
            rr = ef if self.r8 == 0 else min(ef, 2 * n)
            got = self._lib.kci_vec_search_rerank(self._s, q.ctypes.data, n, ef, rr, self._out_id, self._out_sc)
        return self._ids(got)

    def __str__(self):
        return f"KciVecQg(M={self.M},efC={self.efc},dpc={self.dpc},ef={self.ef},eps={self.eps},r8={self.r8})"


class KciVecQl(_Base):
    """Lists learned from VIBE's query sample (fit_ood: X_learn and its neighbours), for the out-of-distribution datasets.
    The sample is clustered in query space (nclusters); each cluster keeps the vectors most often an answer of its
    queries, with an int8 copy of them in list order. A query takes its `ncl` nearest clusters and the first `per`
    vectors of each list; with ef < 0 the int8 copies are scanned and the -ef best scored exactly, with ef = 0 all are
    scored exactly, with ef > 0 they start the graph search (M, efConstruction), which completes what the lists miss
    (seeds > 0: only the `seeds` best of the int8 scan start it; 0: every vector of the list prefixes).
    Without a sample (fit) it is the plain graph search with float re-rank."""

    def __init__(self, metric, M, efConstruction, nclusters, cap=4000):
        self._check_metric(metric)
        self._lib = _load()
        self.cap = int(cap)   # most vectors kept per list (each costs dim bytes once packed)
        if not hasattr(self._lib, "kci_vec_learn"):
            raise NotImplementedError("kci_vec: this library has no kci_vec_learn (wheel 0.2.1 or older)")
        self.M, self.efc, self.nclusters = int(M), int(efConstruction), int(nclusters)
        self.ncl, self.per, self.ef, self.seeds, self.pages, self.hot = 1, 1000, 0, 0, 0, 0
        self._v = self._s = None
        self._learned = False

    def _build(self, X):
        self._v = self._new_index(X, self.M, self.efc)
        secs = C.c_double(0.0)
        if self._lib.kci_vec_build(self._v, THREADS, C.byref(secs)) != 0:
            raise RuntimeError("kci_vec_build failed")

    def fit(self, X):
        self._build(X)
        self._s = self._lib.kci_vec_searcher_new(self._v)

    def fit_ood(self, X_train, X_learn, X_learn_neighbors):
        self._build(X_train)
        L = np.ascontiguousarray(X_learn, dtype=np.float32)
        A = np.ascontiguousarray(X_learn_neighbors, dtype=np.uint64)
        if self._lib.kci_vec_learn(self._v, L.ctypes.data, len(L), A.ctypes.data, A.shape[1], self.nclusters, self.cap, THREADS, None) != 0:
            raise RuntimeError("kci_vec_learn failed")
        if self._lib.kci_vec_learn_pack(self._v) != 0:
            raise RuntimeError("kci_vec_learn_pack failed")
        try:  # how much of this process is on 2 MB pages (the packed rows ask for them): explains run-to-run differences
            hp = [l for l in open("/proc/self/smaps_rollup") if l.startswith("AnonHugePages")]
            print("kci_vec: after learn_pack", " ".join(hp[0].split()) if hp else "AnonHugePages unknown", flush=True)
        except OSError:
            pass
        self._learned = True
        self._s = self._lib.kci_vec_searcher_new(self._v)

    def set_query_arguments(self, ncl, per, ef, seeds, pages=0, hot=0):
        # hot > 0: also the first `hot` of the rows most often an answer over the whole sample (kci_vec_searcher_set_learned_hot)
        self.hot = int(hot)
        # pages > 0: the clusters are chosen in two stages (kci_vec_searcher_set_learned_pages)
        self.ncl, self.per, self.ef, self.seeds, self.pages = int(ncl), int(per), int(ef), int(seeds), int(pages)
        if self._learned:
            self._lib.kci_vec_searcher_set_learned_seeds(self._s, self.seeds)
            self._lib.kci_vec_searcher_set_learned_pages(self._s, self.pages)
            self._lib.kci_vec_searcher_set_learned_hot(self._s, self.hot)

    def query(self, v, n):
        q = np.ascontiguousarray(v, dtype=np.float32)
        if self._learned:
            got = self._lib.kci_vec_search_learned(self._s, q.ctypes.data, n, self.ncl, self.per, self.ef, self._out_id, self._out_sc)
        else:
            ef = max(abs(self.ef), n)
            got = self._lib.kci_vec_search_rerank(self._s, q.ctypes.data, n, ef, ef, self._out_id, self._out_sc)
        return self._ids(max(got, 0))

    def __str__(self):
        return f"KciVecQl(M={self.M},efC={self.efc},nclusters={self.nclusters},cap={self.cap},ncl={self.ncl},per={self.per},ef={self.ef},seeds={self.seeds},pages={self.pages},hot={self.hot})"


class KciVecQv(_Base):
    """Candidates voted by the nearest sample queries (fit_ood), for out-of-distribution data whose answers are NOT
    concentrated (VIBE cqadupstack-lemur / muvera: every vector answers some sample query). The sample queries, projected
    to `qdim` dims by a fixed random matrix, go into a cosine graph (M, efConstruction). A query finds its `voters`
    nearest sample queries (ef = voters * efx); the union of their answers is ranked by a short int8 code (`sdim` dims, a
    linear map fitted on the sample: the directions the queries use, weighted by the data, as in LeanVec-OOD), the best
    keepx * rerank go through the full int8 rows, and the best `rerank` are scored exactly. sdim = 0: no short code.
    The same family as MLANN (candidates voted by a model of the query sample); nothing here is a new method.
    Without a sample (fit) it is the plain graph search with float re-rank."""

    VMAX = 16384   # most voters per query

    def __init__(self, metric, M, efConstruction, qdim, sdim, qproj="rand"):
        self._check_metric(metric)
        self.qproj = str(qproj)   # rand: fixed random matrix; pca: the sample queries' own top qdim directions
        self._lib = _load()
        if not hasattr(self._lib, "kci_vec_vote_set"):
            raise NotImplementedError("kci_vec: this library has no kci_vec_vote_set")
        lib = self._lib
        lib.kci_vec_vote_set.argtypes = [C.c_void_p, C.c_void_p, C.c_uint32, C.c_int]
        lib.kci_vec_search_voted.argtypes = [C.c_void_p, C.c_void_p, C.c_void_p, C.c_int, C.c_int, C.c_int, C.c_void_p, C.c_void_p]
        lib.kci_vec_vote_short.argtypes = [C.c_void_p, C.c_void_p, C.c_void_p, C.c_uint32, C.c_int]
        lib.kci_vec_search_voted_short.argtypes = [C.c_void_p, C.c_void_p, C.c_void_p, C.c_int, C.c_void_p, C.c_int, C.c_int, C.c_int, C.c_void_p, C.c_void_p]
        self.M, self.efc, self.qdim, self.sdim = int(M), int(efConstruction), int(qdim), int(sdim)
        self.voters, self.efx, self.keepx, self.rerank = 100, 1.3, 5, 200
        self._v = self._s = self._vq = self._sq = None
        self._voted = False
        self._short = None

    def fit(self, X):
        self._v = self._new_index(X, 16, 200)
        if self._lib.kci_vec_build(self._v, THREADS, None) != 0:
            raise RuntimeError("kci_vec_build failed")
        self._s = self._lib.kci_vec_searcher_new(self._v)

    def fit_ood(self, X_train, X_learn, X_learn_neighbors):
        import time
        lib = self._lib
        n, dim = X_train.shape
        nq = len(X_learn)
        step = 20000
        t0 = time.time()
        # the data: only the vectors are used (exact scores, int8 copies); the smallest graph finishes the index
        self._v = self._new_index(X_train, 4, 16)
        if lib.kci_vec_build(self._v, THREADS, None) != 0:
            raise RuntimeError("kci_vec_build failed")
        print("kci_vec-qv: data index %.0fs" % (time.time() - t0), flush=True); t0 = time.time()
        # second moments of a sample of the queries and of the data, accumulated in blocks (no large copies)
        qd = min(self.qdim, dim)
        need_c = self.qproj == "pca" or (self._ip and 0 < self.sdim < dim)
        if need_c:
            Cq = np.zeros((dim, dim), np.float64); m = 0
            for a in range(0, nq, step):
                b = np.asarray(X_learn[a:a + step:16], dtype=np.float32); Cq += b.T @ b; m += len(b)
            Cq /= m
            wq, Uq = np.linalg.eigh(Cq)
            del Cq
        # the sample queries: projection to qdim dims (rand: a fixed random matrix; pca: their own top directions), cosine graph
        if self.qproj == "pca":
            self._U = np.ascontiguousarray(Uq[:, ::-1][:, :qd].astype(np.float32))
        else:
            self._U = np.ascontiguousarray((np.random.default_rng(7).standard_normal((dim, qd)) / np.sqrt(qd)).astype(np.float32))
        Lp = np.empty((nq, qd), np.float32)
        for a in range(0, nq, step):
            Lp[a:a + step] = np.asarray(X_learn[a:a + step], dtype=np.float32) @ self._U
        self._vq = lib.kci_vec_new(qd, self.M, self.efc)
        if not self._vq:
            raise RuntimeError("kci_vec_new failed")
        lib.kci_vec_reserve(self._vq, nq + 1)
        base, row = Lp.ctypes.data, qd * 4
        for i in range(nq):
            lib.kci_vec_add(self._vq, i, base + i * row)
        if lib.kci_vec_build(self._vq, THREADS, None) != 0:
            raise RuntimeError("kci_vec_build (sample queries) failed")
        del Lp
        print("kci_vec-qv: sample query graph %.0fs" % (time.time() - t0), flush=True); t0 = time.time()
        A = np.ascontiguousarray(X_learn_neighbors, dtype=np.uint64)
        if lib.kci_vec_vote_set(self._v, A.ctypes.data, nq, A.shape[1]) != 0:
            raise RuntimeError("kci_vec_vote_set failed")
        del A
        print("kci_vec-qv: vote_set %.0fs" % (time.time() - t0), flush=True); t0 = time.time()
        # the short code: q.x ~ (q A)(x B)^T with A B^T = identity on the kept directions (inner product only)
        if self._ip and 0 < self.sdim < dim:
            Cx = np.zeros((dim, dim), np.float64); m = 0
            for a in range(0, n, step):
                b = np.asarray(X_train[a:a + step:4], dtype=np.float32); Cx += b.T @ b; m += len(b)
            Cx /= m
            wq = np.maximum(wq, wq.max() * 1e-9)
            Hq = (Uq * np.sqrt(wq)) @ Uq.T
            iHq = (Uq / np.sqrt(wq)) @ Uq.T
            wm, Um = np.linalg.eigh(Hq @ Cx @ Hq)
            del Cx
            Um = Um[:, ::-1][:, :self.sdim]
            self._short = np.ascontiguousarray((iHq @ Um).astype(np.float32))
            B = (Hq @ Um).astype(np.float32)
            del Hq, iHq, Um
            Xs = np.empty((n, self.sdim), np.float32)
            for a in range(0, n, step):
                Xs[a:a + step] = np.asarray(X_train[a:a + step], dtype=np.float32) @ B
            ids = np.arange(n, dtype=np.uint64)
            if lib.kci_vec_vote_short(self._v, ids.ctypes.data, Xs.ctypes.data, n, self.sdim) != 0:
                raise RuntimeError("kci_vec_vote_short failed")
            del Xs
            print("kci_vec-qv: short code %.0fs" % (time.time() - t0), flush=True)
        self._voted = True
        self._s = lib.kci_vec_searcher_new(self._v)
        self._sq = lib.kci_vec_searcher_new(self._vq)
        self._vid = (C.c_uint64 * self.VMAX)()
        self._vsc = (C.c_float * self.VMAX)()
        self._vnp = np.frombuffer(self._vid, dtype=np.uint64)

    def set_query_arguments(self, voters, efx, keepx, rerank):
        self.voters, self.efx, self.keepx, self.rerank = min(int(voters), self.VMAX), float(efx), int(keepx), int(rerank)

    def query(self, v, n):
        q = np.ascontiguousarray(v, dtype=np.float32)
        lib = self._lib
        if not self._voted:
            ef = max(self.rerank, n)
            return self._ids(max(lib.kci_vec_search_rerank(self._s, q.ctypes.data, n, ef, ef, self._out_id, self._out_sc), 0))
        qp = q @ self._U
        T = self.voters
        g = lib.kci_vec_search_r(self._sq, qp.ctypes.data, T, max(int(T * self.efx), T), self._vid, self._vsc)
        if g <= 0:
            return self._ids(0)
        vo = self._vnp[:g].astype(np.uint32)
        R = max(self.rerank, n)
        if self._short is not None and self.keepx > 0:
            qs = q @ self._short
            got = lib.kci_vec_search_voted_short(self._s, q.ctypes.data, qs.ctypes.data, self.keepx * R, vo.ctypes.data, g, n, R, self._out_id, self._out_sc)
        else:
            got = lib.kci_vec_search_voted(self._s, q.ctypes.data, vo.ctypes.data, g, n, R, self._out_id, self._out_sc)
        return self._ids(max(got, 0))

    def __del__(self):
        try:
            if getattr(self, "_sq", None):
                self._lib.kci_vec_searcher_free(self._sq)
            if getattr(self, "_vq", None):
                self._lib.kci_vec_free(self._vq)
        except Exception:
            pass
        _Base.__del__(self)

    def __str__(self):
        return (f"KciVecQv(M={self.M},efC={self.efc},qdim={self.qdim},sdim={self.sdim},qproj={self.qproj},voters={self.voters},efx={self.efx},"
                f"keepx={self.keepx},rerank={self.rerank})")
