"""Native HingeZero C++20 adapter for VIBE float32 vectors."""
from __future__ import annotations
import ctypes as ct
import json
import os
from pathlib import Path
import tempfile
import numpy as np
from ..base.module import BaseANN

LIBPATH = "/opt/hingezero_cpp20/libhz20.so"


def _native():
    path = Path(os.environ.get("HINGEZERO_CPP20_LIB", LIBPATH))
    if not path.is_file():
        raise RuntimeError("C++20 HingeZero library missing: " + str(path))
    lib = ct.CDLL(str(path))
    lib.hz20_abi.argtypes = []
    lib.hz20_abi.restype = ct.c_int
    lib.hz20_error.argtypes = []
    lib.hz20_error.restype = ct.c_char_p
    lib.hz20_open.argtypes = [ct.c_char_p, ct.c_uint64, ct.c_int, ct.c_int, ct.c_int]
    lib.hz20_open.restype = ct.c_void_p
    lib.hz20_query.argtypes = [ct.c_void_p, ct.c_void_p, ct.c_int32, ct.c_int,
                                ct.c_void_p, ct.c_void_p]
    lib.hz20_query.restype = ct.c_int
    lib.hz20_refine.argtypes = [ct.c_void_p, ct.c_void_p, ct.c_int, ct.c_void_p]
    lib.hz20_refine.restype = ct.c_int
    lib.hz20_verify.argtypes = [ct.c_void_p, ct.c_int64, ct.c_void_p]
    lib.hz20_verify.restype = ct.c_int
    lib.hz20_append.argtypes = [ct.c_char_p, ct.c_uint64, ct.c_int,
                                ct.c_void_p, ct.c_uint64]
    lib.hz20_append.restype = ct.c_int
    lib.hz20_set_budget.argtypes = [ct.c_void_p, ct.c_int]
    lib.hz20_set_budget.restype = ct.c_int
    lib.hz20_close.argtypes = [ct.c_void_p]
    lib.hz20_close.restype = None
    if lib.hz20_abi() != 1:
        raise RuntimeError("HingeZero native ABI mismatch")
    return lib


class HingeZeroCpp20(BaseANN):
    """Zero-refinement memory search with exact ranking and witness checks."""

    def __init__(self, metric, field_size=256, storage_path=None):
        if metric not in ("ip", "normalized", "cosine", "euclidean"):
            raise ValueError("Unsupported VIBE distance metric")
        if not 1 <= int(field_size) <= 8192:
            raise ValueError("Invalid field size")
        self.metric = metric
        self.field_size, self.storage_path = int(field_size), storage_path
        self._tmp = None
        self.lib = None
        self.handle = None
        self.count = self.dimension = 0
        self.root = None
        self.steps = 0
        self.shortlist_budget = 256
        self.refinement_ms = self.scan_ms = self.witness_count = self.queries = 0
        self.source_map = {}

    def _err(self):
        v = self.lib.hz20_error() if self.lib else None
        return v.decode("utf8", "replace") if v else "unknown native error"

    def _check(self, code):
        if code:
            raise RuntimeError("HingeZero C++20: " + self._err())

    def _close(self):
        if self.handle and self.lib:
            self.lib.hz20_close(self.handle)
            self.handle = None

    def __del__(self):
        try:
            self._close()
        except Exception:
            pass

    def _mount(self):
        self._close()
        if self.count <= 0:
            return
        if self.root is None:
            raise RuntimeError("Missing bank root")
        file = self.root / "original-f32.bin"
        if not file.is_file() or file.stat().st_size != self.count*self.dimension*4:
            raise RuntimeError("Stored bank size mismatch; refuse to load")
        self.lib = self.lib or _native()
        ptr = self.lib.hz20_open(
            os.fsencode(file), self.count, self.dimension,
            {"euclidean":0,"ip":1,"normalized":1,"cosine":2}[self.metric],
            self.field_size)
        if not ptr:
            raise RuntimeError("HingeZero C++20 open: " + self._err())
        self.handle = ptr
        self._check(self.lib.hz20_set_budget(self.handle,self.shortlist_budget))

    def _manifest(self):
        filename = self.root / "manifest.json"
        info = {
            "format": "hingezero-cpp20-vibe-float32-v1",
            "metric": self.metric, "count": self.count,
            "dimension": self.dimension, "field_size": self.field_size,
            "provenance": "vibe-corpus",
            "lossless_float32": True,
        }
        tmp = filename.with_suffix(".tmp")
        tmp.write_text(json.dumps(info, sort_keys=True), encoding="utf8")
        os.replace(tmp, filename)

    def fit(self, X):
        X = np.asarray(X)
        if X.ndim != 2 or not 1 <= X.shape[0] <= np.iinfo(np.int32).max:
            raise ValueError("Expected nonempty NxD training bank")
        if not np.issubdtype(X.dtype, np.floating) or not np.isfinite(X).all():
            raise ValueError("VIBE bank must be finite floating vectors")
        if self.storage_path is None:
            self._tmp = tempfile.TemporaryDirectory(prefix="hingezero_cpp20_")
            self.root = Path(self._tmp.name)
        else:
            self.root = Path(self.storage_path)
            self.root.mkdir(parents=True, exist_ok=True)
        if (self.root / "manifest.json").exists() or (self.root / "original-f32.bin").exists():
            raise FileExistsError("Won't overwrite a pre-existing memory bank")
        self.lib = _native()
        self.dimension = int(X.shape[1])
        self.count = 0
        for start in range(0, len(X), 8192):
            self.append_memories(X[start:start+8192], _from_fit=True)
        self._mount()

    def append_memories(self, values, sources=None, *, _from_fit=False):
        if self.root is None:
            raise RuntimeError("fit/load_memory must be called before append")
        x = np.asarray(values, dtype=np.float32)
        if x.ndim == 1:
            x = x[None, :]
        if x.ndim != 2 or len(x) < 1 or x.shape[1] != self.dimension:
            raise ValueError("Wrong shape for appended records")
        if not np.isfinite(x).all():
            raise ValueError("Cannot store nonfinite values")
        if sources is not None and len(sources) != len(x):
            raise ValueError("Mismatched source metadata")
        if self.count + len(x) > np.iinfo(np.int32).max:
            raise ValueError("VIBE int32 ID limit")
        x = np.ascontiguousarray(x, dtype="<f4")
        self.lib = self.lib or _native()
        self._close()
        first = self.count
        self._check(self.lib.hz20_append(os.fsencode(self.root / "original-f32.bin"),
                                         first, self.dimension,
                                         ct.c_void_p(x.ctypes.data), len(x)))
        self.count += len(x)
        if sources is not None:
            with (self.root / "sources.jsonl").open("a", encoding="utf8") as f:
                for offset, source in enumerate(sources):
                    i = first + offset
                    self.source_map[i] = str(source)
                    f.write(json.dumps({"id":i,"source":str(source)},ensure_ascii=False)+"\n")
        self._manifest()
        if not _from_fit:
            self._mount()
        return np.arange(first, self.count, dtype=np.int64)

    def load_memory(self, folder):
        self._close()
        self.root = Path(folder)
        info = json.loads((self.root / "manifest.json").read_text(encoding="utf8"))
        if info.get("format") != "hingezero-cpp20-vibe-float32-v1":
            raise ValueError("Invalid HingeZero bank manifest")
        if self.metric != info["metric"]:
            raise ValueError("Metric differs from saved bank")
        self.count, self.dimension = int(info["count"]), int(info["dimension"])
        self.field_size = int(info["field_size"])
        self.lib = self.lib or _native()
        sources = self.root / "sources.jsonl"
        self.source_map = {}
        if sources.exists():
            for line in sources.read_text(encoding="utf8").splitlines():
                row = json.loads(line)
                self.source_map[int(row["id"])] = row["source"]
        self._mount()
        return self

    def set_query_arguments(self, steps, budget=256):
        steps,budget = int(steps),int(budget)
        if steps != 0:
            raise ValueError("Experimental HingeZero VIBE runs exactly zero refinement steps")
        if not 100 <= budget <= 65536:
            raise ValueError("Budget out of bounds")
        self.steps = 0
        self.shortlist_budget = budget
        if self.handle:
            self._check(self.lib.hz20_set_budget(self.handle,budget))

    def query(self, vector, n):
        if not self.handle:
            raise RuntimeError("HingeZero memory not fitted or opened")
        q = np.asarray(vector,dtype=np.float32)
        if q.shape != (self.dimension,) or not np.isfinite(q).all():
            raise ValueError("Invalid test query")
        q = np.ascontiguousarray(q)
        n = int(n)
        if not 1 <= n <= self.count:
            raise ValueError("Requested number of neighbours invalid")
        ids = np.empty(n,dtype=np.int32)
        timing = np.zeros(3,dtype=np.float64)
        self._check(self.lib.hz20_query(self.handle,ct.c_void_p(q.ctypes.data),
                  n,self.steps,
                  ct.c_void_p(ids.ctypes.data),ct.c_void_p(timing.ctypes.data)))
        if len(set(ids.tolist())) != n or np.any(ids<0) or np.any(ids>=self.count):
            raise RuntimeError("Native memory produced invalid IDs")
        self.refinement_ms += float(timing[0])
        self.scan_ms += float(timing[1])
        self.witness_count += n
        self.queries += 1
        return ids

    def refined_vector(self, vector, steps=8):
        if not self.handle: raise RuntimeError("Memory not opened")
        q = np.ascontiguousarray(vector,dtype=np.float32)
        if q.shape != (self.dimension,): raise ValueError("Wrong dimensions")
        out = np.empty_like(q)
        self._check(self.lib.hz20_refine(self.handle,ct.c_void_p(q.ctypes.data),
                         int(steps),ct.c_void_p(out.ctypes.data)))
        return out

    def read_record(self, record_id):
        i = int(record_id)
        if i < 0 or i >= self.count: raise IndexError("Unknown record")
        with (self.root/"original-f32.bin").open("rb") as f:
            f.seek(i*self.dimension*4)
            buf=f.read(self.dimension*4)
        if len(buf) != self.dimension*4:raise IOError("Bank truncated")
        return np.frombuffer(buf,dtype="<f4").copy()

    def witness(self, record_id, state):
        if not self.handle:return False
        v = np.asarray(state,dtype=np.float32)
        if v.shape != (self.dimension,):return False
        v = np.ascontiguousarray(v)
        return self.lib.hz20_verify(self.handle,int(record_id),
                 ct.c_void_p(v.ctypes.data)) == 1

    def memory_recall(self, vector, min_margin=0.1, min_similarity=0.35):
        """Single-record abstaining retrieval; not used in VIBE evaluation."""
        take=min(self.count,2)
        ids=self.query(vector,take)
        q=self.refined_vector(vector,0)
        vectors=np.stack([self.read_record(i) for i in ids]).astype(np.float64)
        q=q.astype(np.float64)
        if self.metric=="euclidean":
            score=-np.sum((vectors-q)**2,axis=1)
        elif self.metric=="cosine":
            den=np.linalg.norm(vectors,axis=1)*np.linalg.norm(q)
            score=np.divide(vectors@q,den,out=np.zeros(take),where=den>1e-12)
        else:
            score=vectors@q
        margin=float(score[0]-score[1]) if take>1 else float("inf")
        if float(score[0])<min_similarity:
            return {"status":"no_match","score":float(score[0]),"margin":margin}
        if margin<min_margin:
            return {"status":"ambiguous","score":float(score[0]),"margin":margin}
        state=self.read_record(int(ids[0]))
        if not self.witness(int(ids[0]),state):
            raise RuntimeError("Witness refused record")
        return {"status":"found","id":int(ids[0]),"score":float(score[0]),
            "margin":margin,"source":self.source_map.get(int(ids[0]),f"vibe:{int(ids[0])}"),
            "verified":True,"vector":state}

    def get_additional(self):
        return {"engine":"C++20 full-bank HingeZero witness/recall",
            "steps":0,"binary_source_included":True,
            "float32_bank":True,"bipolar_1b_data":False,
            "witness_checks":int(self.witness_count),
            "scan_ms_total":float(self.scan_ms),
            "refine_ms_total":float(self.refinement_ms)}
    def __str__(self):
        return f"HingeZeroCpp20(metric={self.metric},field={self.field_size},steps=0)"
