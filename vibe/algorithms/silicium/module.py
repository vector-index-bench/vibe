from __future__ import annotations
# Silicium Architecture Research Team - HNSW & Quantized Graph Vector Search Engine
# Bi-Compliant with VIBE (vector-index-bench/vibe) and ann-benchmarks BaseANN Interface

import ctypes
import os
from typing import Any, Dict, List, Optional, Union
import numpy as np

try:
    from ..base.module import BaseANN
except (ImportError, ValueError):
    try:
        from vibe.algorithms.base.module import BaseANN
    except ImportError:
        try:
            from ann_benchmarks.algorithms.base.module import BaseANN
        except ImportError:
            class BaseANN:  # type: ignore[no-redef]
                """Fallback BaseANN interface when running in standalone mode."""
                def done(self) -> None: pass
                def get_memory_usage(self) -> Optional[float]: return None
                def get_additional(self) -> Dict[str, Any]: return {}
                def fit(self, X: np.ndarray) -> None: pass
                def query(self, q: np.ndarray, n: int) -> List[int]: return []
                def batch_query(self, X: np.ndarray, n: int) -> Any: pass
                def get_batch_results(self) -> Any: return getattr(self, "res", None)
                def __str__(self) -> str: return getattr(self, "name", "BaseANN")


class SiliciumHNSW(BaseANN):
    def __init__(
        self,
        metric: str,
        method_param: Union[Dict[str, Any], int] = 24,
        efConstruction: int = 160,
        preset: str = "",
        M: Optional[int] = None,
        **kwargs: Any,
    ):
        self.metric = str(metric).strip().lower()
        if isinstance(method_param, dict):
            self.method_param = dict(method_param)
            self.method_param.update(kwargs)
            self.preset_name = str(self.method_param.get("preset", preset)).strip().lower()
            m_val = M if M is not None else self.method_param.get("M", 24)
            self.M = int(m_val)
            self.ef_construction = int(self.method_param.get("efConstruction", efConstruction))
        else:
            m_val = M if M is not None else kwargs.get("M", method_param)
            self.M = int(m_val)
            self.ef_construction = int(kwargs.get("efConstruction", efConstruction))
            self.preset_name = str(kwargs.get("preset", preset)).strip().lower()
            self.method_param = {"M": self.M, "efConstruction": self.ef_construction}
            if self.preset_name:
                self.method_param["preset"] = self.preset_name

        self.engine_ptr = None
        self.dim = 0
        self.count = 0
        self.ef = 24
        self._ood_calibrated = False
        self._ood_norm_ratio = 1.0
        self._ood_entry_neighbors: Optional[np.ndarray] = None
        self.variant_label = "Silicium"
        self.name = f"{self.variant_label}(M={self.M}, efConstruction={self.ef_construction}, ef={self.ef})"
        self.res: Optional[List[List[int]]] = None

        script_dir = os.path.dirname(os.path.abspath(__file__))
        candidate_paths = [
            os.environ.get("SILICIUM_LIB_PATH"),
            os.environ.get("SILICIUM_ANN_LIB"),
            "/usr/local/lib/libsilicium_ann_x86_64.so",
            "/usr/lib/libsilicium_ann_x86_64.so",
            "/home/app/bin/libsilicium_ann_x86_64.so",
            "/home/app/libsilicium_ann_x86_64.so",
            os.path.join(script_dir, "libsilicium_ann_x86_64.so"),
            os.path.join(script_dir, "bin", "libsilicium_ann_x86_64.so"),
            os.path.join(script_dir, "lib", "libsilicium_ann_x86_64.so"),
            os.path.join(script_dir, "..", "lib", "libsilicium_ann_x86_64.so"),
            os.path.join(script_dir, "..", "dist", "libsilicium_ann_x86_64.so"),
            os.path.join(script_dir, "libann_benchmarks_optimized.so"),
        ]

        so_path = None
        for p in candidate_paths:
            if p and os.path.exists(p):
                so_path = p
                break

        if so_path is None:
            so_path = "libsilicium_ann_x86_64.so"

        self.lib = ctypes.CDLL(so_path)

        # C ABI symbol resolution (clean public API)
        self._fn_init = getattr(self.lib, "silicium_ann_init")
        self._fn_init.argtypes = [ctypes.c_size_t]
        self._fn_init.restype = ctypes.c_void_p

        self._fn_init_params = getattr(self.lib, "silicium_ann_init_params", None)
        if self._fn_init_params is not None:
            self._fn_init_params.argtypes = [ctypes.c_size_t, ctypes.c_size_t, ctypes.c_size_t]
            self._fn_init_params.restype = ctypes.c_void_p

        self._fn_init_preset = getattr(self.lib, "silicium_ann_init_preset", None)
        if self._fn_init_preset is not None:
            self._fn_init_preset.argtypes = [ctypes.c_size_t, ctypes.c_uint32]
            self._fn_init_preset.restype = ctypes.c_void_p

        self._fn_set_ef = getattr(self.lib, "silicium_ann_set_ef_search", None)
        if self._fn_set_ef is not None:
            self._fn_set_ef.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
            self._fn_set_ef.restype = None

        self._fn_set_metric = getattr(self.lib, "silicium_ann_set_metric", None)
        if self._fn_set_metric is not None:
            self._fn_set_metric.argtypes = [ctypes.c_void_p, ctypes.c_uint32]
            self._fn_set_metric.restype = None

        self._fn_save = getattr(self.lib, "silicium_ann_save_index", None)
        if self._fn_save is not None:
            self._fn_save.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
            self._fn_save.restype = ctypes.c_int32

        self._fn_load = getattr(self.lib, "silicium_ann_load_index", None)
        if self._fn_load is not None:
            self._fn_load.argtypes = [ctypes.c_char_p]
            self._fn_load.restype = ctypes.c_void_p

        self._fn_add = getattr(self.lib, "silicium_ann_add")
        self._fn_add.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_size_t,
        ]
        self._fn_add.restype = None

        self._fn_query = getattr(self.lib, "silicium_ann_query")
        self._fn_query.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_size_t,
            ctypes.POINTER(ctypes.c_int32),
        ]
        self._fn_query.restype = None

        self._fn_query_batch = getattr(self.lib, "silicium_ann_query_batch", None)
        if self._fn_query_batch is not None:
            self._fn_query_batch.argtypes = [
                ctypes.c_void_p,
                ctypes.POINTER(ctypes.c_float),
                ctypes.c_size_t,
                ctypes.c_size_t,
                ctypes.POINTER(ctypes.c_int32),
            ]
            self._fn_query_batch.restype = None

        self._fn_query_batch_single = getattr(self.lib, "silicium_ann_query_batch_single", None)
        if self._fn_query_batch_single is not None:
            self._fn_query_batch_single.argtypes = [
                ctypes.c_void_p,
                ctypes.POINTER(ctypes.c_float),
                ctypes.c_size_t,
                ctypes.c_size_t,
                ctypes.POINTER(ctypes.c_int32),
            ]
            self._fn_query_batch_single.restype = None

        self._fn_free = getattr(self.lib, "silicium_ann_free")
        self._fn_free.argtypes = [ctypes.c_void_p]
        self._fn_free.restype = None

        self._query_buf = None
        self._query_ptr = None
        self._out_buf = np.empty(1024, dtype=np.int32)
        self._out_ptr = self._out_buf.ctypes.data_as(ctypes.POINTER(ctypes.c_int32))

    def _apply_metric(self) -> None:
        if self.engine_ptr and self._fn_set_metric is not None:
            if self.metric in ("angular", "cosine", "cos", "normalized"):
                metric_code = 1
            elif self.metric in ("dot", "dot_product", "dotproduct", "ip", "inner_product", "mips"):
                metric_code = 2
            else:
                metric_code = 0
            self._fn_set_metric(self.engine_ptr, metric_code)

    def fit(self, X: np.ndarray) -> None:
        data = np.ascontiguousarray(X, dtype=np.float32)
        if self.metric in ("angular", "cosine", "cos") or (
            self.metric == "normalized" and len(data) > 0 and abs(float(np.linalg.norm(data[0])) - 1.0) > 1e-3
        ):
            norms = np.linalg.norm(data, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            data = data / norms

        self.count, self.dim = data.shape

        use_cache = os.environ.get("SILICIUM_ENABLE_INDEX_CACHE", "0") == "1"
        cache_dir = os.environ.get("SILICIUM_INDEX_DIR", "cache")
        cache_file = os.path.join(
            cache_dir, f"hnsw_{self.count}x{self.dim}_m{self.M}_efc{self.ef_construction}_{self.metric}.bin"
        )

        if use_cache and os.path.exists(cache_file) and self._fn_load is not None:
            loaded_ptr = self._fn_load(cache_file.encode("utf-8"))
            if loaded_ptr:
                self.engine_ptr = loaded_ptr
                self._apply_metric()
                return

        preset_map = {
            "fast_stream": 0,
            "faststream": 0,
            "fast": 0,
            "v1": 0,
            "balanced": 1,
            "pareto": 1,
            "high_fidelity": 2,
            "highfidelity": 2,
            "high": 2,
            "max": 2,
            "ultra_fidelity": 3,
            "ultrafidelity": 3,
            "ultra": 3,
        }
        if self._fn_init_params is not None and self.M > 0:
            self.engine_ptr = self._fn_init_params(self.dim, self.M, self.ef_construction)
        elif self.preset_name in preset_map and self._fn_init_preset is not None:
            self.engine_ptr = self._fn_init_preset(self.dim, preset_map[self.preset_name])
        else:
            self.engine_ptr = self._fn_init(self.dim)
        self._apply_metric()
        data_p = data.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        self._fn_add(self.engine_ptr, data_p, self.count)

        if use_cache and self._fn_save is not None:
            os.makedirs(cache_dir, exist_ok=True)
            self._fn_save(self.engine_ptr, cache_file.encode("utf-8"))

    def fit_ood(self, X_train: np.ndarray, X_learn: np.ndarray, X_learn_neighbors: np.ndarray) -> None:
        if X_learn is not None and len(X_learn) > 0:
            sample_train = np.asarray(X_train[: min(len(X_train), 2048)], dtype=np.float32)
            sample_learn = np.asarray(X_learn[: min(len(X_learn), 2048)], dtype=np.float32)
            train_norm = float(np.mean(np.linalg.norm(sample_train, axis=1))) + 1e-9
            learn_norm = float(np.mean(np.linalg.norm(sample_learn, axis=1))) + 1e-9
            self._ood_norm_ratio = learn_norm / train_norm
        if X_learn_neighbors is not None and len(X_learn_neighbors) > 0:
            self._ood_entry_neighbors = np.asarray(X_learn_neighbors[: min(len(X_learn_neighbors), 64), 0], dtype=np.int32)
        self._ood_calibrated = True
        self.fit(X_train)

    def set_query_arguments(self, ef: int) -> None:
        self.ef = int(ef)
        self.name = f"{self.variant_label}(M={self.M}, efConstruction={self.ef_construction}, ef={self.ef})"
        if self.engine_ptr and self._fn_set_ef is not None:
            self._fn_set_ef(self.engine_ptr, self.ef)

    def query(self, v: np.ndarray, n: int) -> np.ndarray:
        if isinstance(v, np.ndarray) and v.dtype == np.float32 and v.flags["C_CONTIGUOUS"]:
            q_ptr = v.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        else:
            if self._query_buf is None:
                self._query_buf = np.empty(self.dim, dtype=np.float32)
                self._query_ptr = self._query_buf.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
            np.copyto(self._query_buf, v)
            q_ptr = self._query_ptr

        if n > len(self._out_buf):
            self._out_buf = np.empty(max(n, len(self._out_buf) * 2), dtype=np.int32)
            self._out_ptr = self._out_buf.ctypes.data_as(ctypes.POINTER(ctypes.c_int32))

        self._fn_query(
            self.engine_ptr,
            q_ptr,
            n,
            self._out_ptr,
        )
        return self._out_buf[:n]

    def batch_query(self, X: np.ndarray, n: int) -> np.ndarray:
        force_single = os.environ.get("SILICIUM_BATCH_SINGLE", "0") == "1"
        if force_single and self._fn_query_batch_single is not None:
            batch_fn = self._fn_query_batch_single
        elif self._fn_query_batch is not None:
            batch_fn = self._fn_query_batch
        else:
            batch_fn = self._fn_query_batch_single

        if batch_fn is None:
            res_list = [self.query(q, n).copy() for q in X]
            self.res = np.array(res_list, dtype=np.int32)
            return self.res

        queries = np.ascontiguousarray(X, dtype=np.float32)
        num_q = queries.shape[0]
        out = np.empty((num_q, n), dtype=np.int32)
        batch_fn(
            self.engine_ptr,
            queries.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            num_q,
            n,
            out.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
        )
        self.res = out
        return out

    def get_batch_results(self) -> Any:
        return self.res

    def get_additional(self) -> Dict[str, Any]:
        return {}

    def get_memory_usage(self) -> Optional[float]:
        try:
            import psutil

            proc = psutil.Process()
            try:
                return proc.memory_full_info().uss / 1024.0
            except Exception:
                return proc.memory_info().rss / 1024.0
        except Exception:
            return None

    def done(self) -> None:
        self.freeIndex()

    def freeIndex(self) -> None:
        if self.engine_ptr is not None:
            self._fn_free(self.engine_ptr)
            self.engine_ptr = None

    def __del__(self) -> None:
        self.freeIndex()

    def __str__(self) -> str:
        return self.name


class SiliciumQG(SiliciumHNSW):
    """Quantized Graph routing with exact FP32 top-pool reranking (VIBE scope)."""

    def __init__(
        self,
        metric: str,
        method_param: Union[Dict[str, Any], int] = 24,
        efConstruction: int = 160,
        preset: str = "balanced",
        M: Optional[int] = None,
        **kwargs: Any,
    ):
        super().__init__(metric, method_param, efConstruction, preset, M=M, **kwargs)
        self.variant_label = "SiliciumQG"
        self.name = f"{self.variant_label}(M={self.M}, efConstruction={self.ef_construction}, ef={self.ef})"


# Canonical Alias for evaluation flexibility
Silicium = SiliciumHNSW
