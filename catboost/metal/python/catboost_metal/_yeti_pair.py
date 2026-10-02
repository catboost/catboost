"""Resident YetiRankPairwise: generated weak pairs and fixed Bayesian leaves.

Metal uses explicit per-tree RNG domains so replay does not depend on mutable
CUDA GPU seed buffers. PFound permutation arithmetic itself is CUDA-derived.
"""
import ctypes as ct
import numbers
import threading
import numpy as np
from . import _native
from ._query_data import validate_offsets

class YetiPairOptions(ct.Structure):
    _fields_=[('groups',ct.c_uint32),('permutations',ct.c_uint32),('decay',ct.c_float),('sampling_unit',ct.c_uint32)]

class Session(_native.Session):
    def __init__(self, bins, targets, candidate_features, candidate_bins, *, group_offsets,
                 permutations=10, decay=.85, sampling_unit="Object", non_diagonal_regularization=.1, **options):
        self._handle=ct.c_void_p(); self._lock=threading.RLock(); self._completed=0
        self._permutation_count=1; self._permutations_configured=False
        self._feature_penalties_configured=False; self._lib=None
        config=dict(iterations=1, depth=6, learning_rate=.03, l2_leaf_reg=0., bias=0.,
            score_function='NewtonL2', sample_weight=None, leaf_estimation_iterations=1,
            leaf_estimation_backtracking='No', initial_predictions=None, candidate_types=None,
            objective_param=None, leaf_estimation_method='Simple', bootstrap_type='No', random_seed=0,
            iteration_offset=0, bagging_temperature=1., subsample=1., mvs_reg=None,
            initial_mvs_lambda=None, random_strength=0.)
        config.update(options)
        self._random_seed=config['random_seed'];self._iteration_offset=config['iteration_offset']
        if 'objective' in config: raise ValueError('This session implements YetiRankPairwise only.')
        if config['leaf_estimation_method'] not in ('Newton','Gradient','Simple'):
            raise ValueError('YetiRankPairwise requires Newton, Gradient or Simple leaves.')
        if config['bootstrap_type'] not in ('No','Bayesian','Bernoulli'):
            raise ValueError('YetiRankPairwise supports No, Bayesian and Bernoulli bootstrap.')
        if config['leaf_estimation_backtracking']!='No':
            raise ValueError('YetiRankPairwise does not support leaf backtracking.')
        if sampling_unit not in ('Object','Group'):raise ValueError('sampling_unit must be Object or Group.')
        if isinstance(permutations,bool) or not isinstance(permutations,numbers.Integral) or not 1<=permutations<=10000:
            raise ValueError('YetiRankPairwise permutations must be in [1,10000].')
        if not np.isfinite(decay) or not 0<=decay<=1:raise ValueError('YetiRankPairwise decay must be in [0,1].')
        values=np.asarray(bins)
        if values.ndim!=2: raise ValueError('Quantized bins must be a feature-major matrix.')
        self._params, objective, bootstrap, arrays = _native._prepare(
            bins,targets,candidate_features,candidate_bins,objective='RMSE',_full_matrix=True,**config)
        if self._params.train.depth>8: raise ValueError('YetiRankPairwise supports depth <= 8 like CUDA.')
        with np.errstate(over='ignore',invalid='ignore'):
            non_diag=np.float32(non_diagonal_regularization)
        if not np.isfinite(non_diag) or non_diag<0: raise ValueError('Pairwise non-diagonal regularization must be finite and nonnegative.')
        offsets=validate_offsets(group_offsets,self._params.train.rows)
        self._params.objective=objective.objective=18; self.objective='YetiRankPairwise'
        self._lib=_native._load(_native.build_library())
        f32,u32,u8=ct.POINTER(ct.c_float),ct.POINTER(ct.c_uint32),ct.POINTER(ct.c_uint8)
        self._lib.cbm_session_create_yeti_pairwise.argtypes=[ct.POINTER(_native.SessionParams),ct.POINTER(_native.ObjectiveOptions),
            ct.POINTER(YetiPairOptions),ct.c_float,u32,u8,f32,f32,f32,u32,u32,u8,
            ct.POINTER(ct.c_void_p),ct.c_char_p,ct.c_size_t]
        self._lib.cbm_session_create_yeti_pairwise.restype=ct.c_int
        bins,targets,objects,initial,features,borders,types,*_=arrays
        pair=YetiPairOptions(len(offsets)-1,permutations,decay,sampling_unit=="Group")
        error=ct.create_string_buffer(2048)
        self._check(self._lib.cbm_session_create_yeti_pairwise(ct.byref(self._params),ct.byref(objective),ct.byref(pair),non_diag,
            _native._u32(offsets),_native._u8(bins),_native._f32(targets),
            _native._f32(objects),_native._f32(initial),_native._u32(features),_native._u32(borders),_native._u8(types),
            ct.byref(self._handle),error,len(error)),error)
        try:
            self._check(self._lib.cbm_session_set_bootstrap(self._handle,ct.byref(bootstrap),error,len(error)),error)
            noise=_native.ScoreNoiseOptions(config['random_strength'],0,0,0)
            self._check(self._lib.cbm_session_set_score_noise(self._handle,ct.byref(noise),error,len(error)),error)
        except Exception:
            self.close(); raise



class TrainingSession(Session):
    def __init__(self,bins,targets,candidate_features,candidate_bins,*,objective='YetiRankPairwise',**options):
        if objective!='YetiRankPairwise':raise ValueError('This controller implements YetiRankPairwise only.')
        self._targets=np.asarray(targets,np.float32).copy();self._weights=options.get('sample_weight')
        self._offsets=validate_offsets(options['group_offsets'],len(targets))
        from ._query_data import validate_subgroup_hashes
        self._subgroups=validate_subgroup_hashes(options.pop('subgroup_hashes',None),len(targets))
        if (self._targets.ndim!=1 or not np.isfinite(self._targets).all() or (self._targets<0).any() or (self._targets>1).any()):
            raise ValueError('YetiRankPairwise with PFound requires relevance labels in [0,1].')
        super().__init__(bins,targets,candidate_features,candidate_bins,**options)
        try:self._metric_history=[self._metric()]
        except Exception:self.close();raise
    def _metric(self):
        from ._training import _shared_metric
        return _shared_metric('PFound',self.predictions(),self._targets,self._weights,self._offsets,subgroup_hashes=self._subgroups)
    def step(self):
        with self._lock:
            from ._data import cuda_search_permutation
            self.select_permutation(cuda_search_permutation(self._random_seed,self._iteration_offset+self._completed,self._permutation_count))
            tree=super().step();tree.loss=self._metric();self._metric_history.append(tree.loss);return tree
    def configure_permutations(self,bins_list,initial_predictions=None,mvs_lambdas=None,mvs_valid=None):
        with self._lock:
            super().configure_permutations(bins_list,initial_predictions,mvs_lambdas,mvs_valid)
            self._metric_history=[self._metric()]
    def result(self):
        result=super().result();result.rmse=np.asarray(self._metric_history,np.float32)
        result.stats['yeti_pair_rng']='item_iteration_dataset_domains_v2' if self._permutation_count>1 else 'item_iteration_domains_v1';return result
