"""Private full-matrix PairLogitPairwise trainer; numeric/one-hot, Plain P1.

Edges are sampled for split search only. Leaf estimation uses original edges
and original document weights. This is the CUDA non-diagonal target algorithm,
with the existing Metal edge-index bootstrap stream.
"""
import ctypes as ct
import threading
import numpy as np
from . import _native
from ._query_data import validate_offsets, prepare_pair_arrays


class PairOptions(ct.Structure):
    _fields_ = [(name, ct.c_uint32) for name in ('pair_count', 'group_count', 'reserved0', 'reserved1')]


class Session(_native.Session):
    def __init__(self, bins, candidate_features, candidate_bins, *, pair_winners, pair_losers,
                 pair_weights=None, group_offsets=None, non_diagonal_regularization=.1, **options):
        self._handle=ct.c_void_p(); self._lock=threading.RLock(); self._completed=0
        self._permutation_count=1; self._permutations_configured=False
        self._feature_penalties_configured=False; self._lib=None
        config=dict(iterations=1, depth=6, learning_rate=.03, l2_leaf_reg=5., bias=0.,
            score_function='NewtonL2', sample_weight=None, leaf_estimation_iterations=1,
            leaf_estimation_backtracking='No', initial_predictions=None, candidate_types=None,
            objective_param=None, leaf_estimation_method='Newton', bootstrap_type='No', random_seed=0,
            iteration_offset=0, bagging_temperature=1., subsample=1., mvs_reg=None,
            initial_mvs_lambda=None, random_strength=0.)
        config.update(options)
        if 'objective' in config: raise ValueError('This session implements PairLogitPairwise only.')
        if config['leaf_estimation_method'] not in ('Newton','Gradient','Simple'):
            raise ValueError('PairLogitPairwise requires Newton, Gradient or Simple leaves.')
        if config['bootstrap_type']=='MVS': raise ValueError('PairLogitPairwise does not support MVS.')
        values=np.asarray(bins)
        if values.ndim!=2: raise ValueError('Quantized bins must be a feature-major matrix.')
        self._params, objective, bootstrap, arrays = _native._prepare(
            bins,np.zeros(values.shape[1],np.float32),candidate_features,candidate_bins,objective='RMSE',_full_matrix=True,**config)
        if self._params.train.depth>8: raise ValueError('PairLogitPairwise supports depth <= 8 like CUDA.')
        with np.errstate(over='ignore',invalid='ignore'):
            non_diag=np.float32(non_diagonal_regularization)
        if not np.isfinite(non_diag) or non_diag<0: raise ValueError('Pairwise non-diagonal regularization must be finite and nonnegative.')
        offsets=None if group_offsets is None else validate_offsets(group_offsets,self._params.train.rows)
        winners,losers,weights=prepare_pair_arrays(pair_winners,pair_losers,pair_weights,self._params.train.rows,offsets)
        self._params.objective=objective.objective=15; self.objective='PairLogitPairwise'
        self._lib=_native._load(_native.build_library())
        f32,u32,u8=ct.POINTER(ct.c_float),ct.POINTER(ct.c_uint32),ct.POINTER(ct.c_uint8)
        self._lib.cbm_session_create_pair_matrix.argtypes=[ct.POINTER(_native.SessionParams),ct.POINTER(_native.ObjectiveOptions),
            ct.POINTER(PairOptions),ct.c_float,u32,u32,f32,u32,u8,f32,f32,u32,u32,u8,
            ct.POINTER(ct.c_void_p),ct.c_char_p,ct.c_size_t]
        self._lib.cbm_session_create_pair_matrix.restype=ct.c_int
        bins,_,objects,initial,features,borders,types,*_=arrays
        pair=PairOptions(len(winners),0 if offsets is None else len(offsets)-1,0,0)
        error=ct.create_string_buffer(2048)
        self._check(self._lib.cbm_session_create_pair_matrix(ct.byref(self._params),ct.byref(objective),ct.byref(pair),non_diag,
            _native._u32(winners),_native._u32(losers),_native._f32(weights),_native._u32(offsets),_native._u8(bins),
            _native._f32(objects),_native._f32(initial),_native._u32(features),_native._u32(borders),_native._u8(types),
            ct.byref(self._handle),error,len(error)),error)
        try:
            self._check(self._lib.cbm_session_set_bootstrap(self._handle,ct.byref(bootstrap),error,len(error)),error)
            noise=_native.ScoreNoiseOptions(config['random_strength'],0,0,0)
            self._check(self._lib.cbm_session_set_score_noise(self._handle,ct.byref(noise),error,len(error)),error)
        except Exception:
            self.close(); raise


class TrainingSession(Session):
    """Signature adapter for the common lifecycle controller; labels are metadata."""
    def __init__(self,bins,targets,candidate_features,candidate_bins,*,objective='PairLogitPairwise',**options):
        if objective!='PairLogitPairwise':raise ValueError('This controller implements PairLogitPairwise only.')
        super().__init__(bins,candidate_features,candidate_bins,**options)
