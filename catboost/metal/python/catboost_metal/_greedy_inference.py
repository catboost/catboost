"""Resident Metal cursors and binned prediction for variable-node trees."""
import ctypes as ct
import functools
import threading

import numpy as np

from . import _greedy
from ._native import TrainStats, _stats


class EvaluationStats(ct.Structure):
    _fields_ = [("compute", TrainStats)] + [(name, ct.c_uint64) for name in (
        "dataset_uploads", "bins_upload_bytes", "tree_upload_bytes", "resident_bytes")]


@functools.lru_cache(maxsize=4)
def _load(path):
    lib = ct.CDLL(str(path))
    u8, f32 = ct.POINTER(ct.c_uint8), ct.POINTER(ct.c_float)
    tail = [ct.c_char_p, ct.c_size_t]
    lib.cbm_greedy_evaluation_create.argtypes = [ct.c_uint32, ct.c_uint32, ct.c_uint32,
        u8, ct.c_uint64, ct.c_float, f32, ct.c_uint64, ct.POINTER(ct.c_void_p)] + tail
    lib.cbm_greedy_evaluation_create_vector.argtypes = [ct.c_uint32]*4 + [u8, ct.c_uint64,
        f32, ct.c_uint64, ct.POINTER(ct.c_void_p)] + tail
    lib.cbm_greedy_evaluation_create_vector.restype = ct.c_int
    lib.cbm_greedy_evaluation_add_vector_tree.argtypes = [ct.c_void_p, ct.POINTER(_greedy.Node),
        ct.c_uint64, f32, ct.c_uint64, f32, ct.c_uint64] + tail
    lib.cbm_greedy_evaluation_add_vector_tree.restype = ct.c_int
    lib.cbm_greedy_evaluation_add_tree.argtypes = [ct.c_void_p, ct.POINTER(_greedy.Node),
        ct.c_uint64, f32, ct.c_uint64, f32, ct.c_uint64] + tail
    lib.cbm_greedy_evaluation_predictions.argtypes = [ct.c_void_p, f32, ct.c_uint64] + tail
    lib.cbm_greedy_evaluation_stats.argtypes = [ct.c_void_p, ct.POINTER(EvaluationStats)] + tail
    for name in ("create", "add_tree", "predictions", "stats"):
        getattr(lib, "cbm_greedy_evaluation_" + name).restype = ct.c_int
    lib.cbm_greedy_evaluation_close.argtypes = [ct.c_void_p]
    lib.cbm_greedy_evaluation_close.restype = None
    return lib


def tree_depth(tree, features=None, max_depth=65535):
    """Validate flat node topology and return the deepest terminal path."""
    nodes, values = np.asarray(tree.nodes), np.asarray(tree.leaf_values)
    if (nodes.ndim != 2 or nodes.shape[1:] != (6,) or nodes.dtype.kind not in "iu" or
            (nodes < 0).any() or (nodes > 2**32 - 1).any() or values.ndim not in (1, 2) or
            (values.ndim == 2 and not 2 <= values.shape[1] <= 64) or
            values.dtype.kind not in "iuf" or not len(values) or len(values) > 65536 or
            len(nodes) != 2 * len(values) - 1 or not np.isfinite(values).all()):
        raise ValueError("Invalid greedy tree node or leaf arrays.")
    pending, seen, used, deepest = [(0, 0)], set(), set(), 0
    while pending:
        index, depth = pending.pop()
        if index >= len(nodes) or index in seen or depth > max_depth:
            raise ValueError("Invalid greedy tree graph, shared node, cycle or excessive depth.")
        seen.add(index)
        feature, border, kind, left, right, leaf = map(int, nodes[index])
        if leaf != 2**32 - 1:
            if leaf >= len(values) or leaf in used:
                raise ValueError("Invalid greedy leaf references.")
            used.add(leaf); deepest = max(deepest, depth)
        else:
            if border > 255 or kind > 1 or (features is not None and feature >= features):
                raise ValueError("Invalid greedy tree split.")
            pending.extend(((left, depth + 1), (right, depth + 1)))
    if len(seen) != len(nodes) or len(used) != len(values):
        raise ValueError("Greedy tree has unreachable nodes or missing leaves.")
    return deepest


class EvaluationCursor:
    def __init__(self, bins, *, bias=0., max_depth=65535, initial_predictions=None, dimensions=None):
        self._lock = threading.RLock(); self._handle = ct.c_void_p(); self._lib = None
        self._final_stats = None
        bins = np.asarray(bins)
        if (bins.ndim != 2 or bins.dtype.kind not in "iu" or not bins.shape[0] or
                bins.shape[0] > 65536 or bins.shape[1] > 1 << 27 or
                (bins < 0).any() or (bins > 255).any()):
            raise ValueError("bins must be a feature-major byte-valued integer matrix.")
        self._features, self._rows = bins.shape
        self._max_depth = _greedy._integer("max_depth", max_depth, 0, 65535)
        bias = np.asarray(bias)
        if dimensions is None:
            initial_shape = np.shape(initial_predictions)
            dimensions = len(bias) if bias.ndim == 1 else initial_shape[1] if len(initial_shape) == 2 else 1
        self._dimensions = _greedy._integer("dimensions", dimensions, 1, 64)
        self._shape = (self._rows,) if dimensions == 1 else (self._rows, dimensions)
        bias = _greedy._finite_array("bias", bias, () if bias.ndim == 0 else (dimensions,))
        if dimensions == 1 and bias.ndim: bias = bias.reshape(())
        initial = None if initial_predictions is None else _greedy._finite_array(
            "initial_predictions", initial_predictions, self._shape)
        if dimensions > 1 and initial is None:
            initial = np.broadcast_to(bias, self._shape).astype(np.float32).copy()
        bins = np.ascontiguousarray(bins, np.uint8)
        self._lib = _load(_greedy.build_library())
        if dimensions > 1:
            self._call(self._lib.cbm_greedy_evaluation_create_vector, self._rows, self._features, max_depth, dimensions,
                bins.ctypes.data_as(ct.POINTER(ct.c_uint8)), bins.size,
                initial.ctypes.data_as(ct.POINTER(ct.c_float)), initial.size, ct.byref(self._handle))
        else:
            self._call(self._lib.cbm_greedy_evaluation_create, self._rows, self._features, max_depth,
                bins.ctypes.data_as(ct.POINTER(ct.c_uint8)), bins.size, float(bias),
                None if initial is None else initial.ctypes.data_as(ct.POINTER(ct.c_float)),
                0 if initial is None else initial.size, ct.byref(self._handle))

    @staticmethod
    def _call(function, *args):
        error = ct.create_string_buffer(2048)
        if function(*args, error, len(error)):
            raise RuntimeError(error.value.decode("utf-8", errors="replace"))

    def _open(self):
        if not self._handle.value:
            raise RuntimeError("Greedy evaluation cursor is closed.")

    def predictions(self):
        with self._lock:
            self._open()
            output = np.empty(self._shape, np.float32)
            self._call(self._lib.cbm_greedy_evaluation_predictions, self._handle,
                output.ctypes.data_as(ct.POINTER(ct.c_float)), output.size)
            return output

    def add_tree(self, tree):
        with self._lock:
            self._open()
            tree_depth(tree, self._features, self._max_depth)
            nodes = np.ascontiguousarray(tree.nodes, np.uint32)
            shape = (len(tree.leaf_values),) if self._dimensions == 1 else (len(tree.leaf_values), self._dimensions)
            values = _greedy._finite_array("leaf_values", tree.leaf_values, shape)
            output = np.empty(self._shape, np.float32)
            add = self._lib.cbm_greedy_evaluation_add_tree if self._dimensions == 1 else self._lib.cbm_greedy_evaluation_add_vector_tree
            self._call(add, self._handle,
                nodes.ctypes.data_as(ct.POINTER(_greedy.Node)), len(nodes),
                values.ctypes.data_as(ct.POINTER(ct.c_float)), values.size,
                output.ctypes.data_as(ct.POINTER(ct.c_float)), output.size)
            return output

    @property
    def stats(self):
        with self._lock:
            if self._final_stats is not None:
                return self._final_stats.copy()
            self._open()
            output = EvaluationStats()
            self._call(self._lib.cbm_greedy_evaluation_stats, self._handle, ct.byref(output))
            return dict(_stats(output.compute), **{name: int(getattr(output, name)) for name in (
                "dataset_uploads", "bins_upload_bytes", "tree_upload_bytes", "resident_bytes")})

    def close(self):
        with self._lock:
            if self._handle.value and self._lib is not None:
                self._final_stats = self.stats
                self._lib.cbm_greedy_evaluation_close(self._handle)
                self._handle = ct.c_void_p()

    def __enter__(self): self._open(); return self
    def __exit__(self, *_): self.close()
    def __del__(self):
        try: self.close()
        except Exception: pass


def predict_bins(bins, trees, bias=0., tree_start=0, tree_end=None):
    trees = tuple(trees)
    start = _greedy._integer("tree_start", tree_start, 0, len(trees))
    end = len(trees) if tree_end is None else _greedy._integer("tree_end", tree_end, start, len(trees))
    dimensions = None
    if trees:
        shape = np.shape(trees[0].leaf_values)
        dimensions = shape[1] if len(shape) == 2 else 1
    elif np.ndim(bias) == 1:
        dimensions = len(bias)
    with EvaluationCursor(bins, bias=bias if start == 0 else 0., dimensions=dimensions) as cursor:
        for tree in trees[start:end]:
            cursor.add_tree(tree)
        return cursor.predictions()
