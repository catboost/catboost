"""Private resident YetiRank session with explicit per-oracle seed schedules.

The low-level oracle value is zero, as in CUDA. TrainingSession calculates
PFound and persists the target RNG for the public lifecycle controller.
"""
import ctypes as ct
import numbers
import threading

import numpy as np
from . import _native
from ._query_data import validate_offsets


class YetiOptions(ct.Structure):
    _fields_ = [('groups', ct.c_uint32), ('permutations', ct.c_uint32),
                ('decay', ct.c_float), ('legacy_prefix_centering', ct.c_uint32)]


class Session(_native.Session):
    def __init__(self, bins, targets, candidate_features, candidate_bins, *, group_offsets,
                 permutations=10, decay=.85, legacy_prefix_centering=False, **options):
        self._handle=ct.c_void_p(); self._lock=threading.RLock(); self._completed=0
        self._permutation_count=1; self._permutations_configured=False
        self._feature_penalties_configured=False; self._lib=None
        config=dict(iterations=1, depth=6, learning_rate=.03, l2_leaf_reg=0., bias=0.,
            score_function='Cosine', sample_weight=None, leaf_estimation_iterations=1,
            leaf_estimation_backtracking='No', initial_predictions=None, candidate_types=None,
            objective_param=None, leaf_estimation_method='Newton', bootstrap_type='No', random_seed=0,
            iteration_offset=0, bagging_temperature=1., subsample=1., mvs_reg=None,
            initial_mvs_lambda=None, random_strength=0.)
        config.update(options)
        if config['leaf_estimation_method']!='Newton' or config['leaf_estimation_backtracking']!='No':
            raise ValueError('YetiRank requires Newton leaves and no backtracking like CUDA.')
        if 'objective' in config:
            raise ValueError('This session implements YetiRank only.')
        if isinstance(permutations,bool) or not isinstance(permutations,numbers.Integral) or not 1<=permutations<=10000:
            raise ValueError('YetiRank permutations must be in [1,10000].')
        if not isinstance(legacy_prefix_centering,bool):
            raise ValueError('YetiRank legacy_prefix_centering must be boolean.')
        # Reuse shape, numeric, split, sampling and workspace validation only.
        # The native session is created directly with YetiRank objective 17.
        self._params,objective,bootstrap,arrays=_native._prepare(
            bins,targets,candidate_features,candidate_bins,objective='RMSE',**config)
        offsets=validate_offsets(group_offsets,self._params.train.rows)
        self._params.objective=objective.objective=17
        self.objective='YetiRank'
        self._lib=_native._load(_native.build_library())
        f32,u32,u8=ct.POINTER(ct.c_float),ct.POINTER(ct.c_uint32),ct.POINTER(ct.c_uint8)
        self._lib.cbm_session_create_yeti.argtypes=[ct.POINTER(_native.SessionParams),ct.POINTER(_native.ObjectiveOptions),
            ct.POINTER(YetiOptions),u32,u8,f32,f32,f32,u32,u32,u8,ct.POINTER(ct.c_void_p),ct.c_char_p,ct.c_size_t]
        self._lib.cbm_session_create_yeti.restype=ct.c_int
        self._lib.cbm_session_set_yeti_oracle_seeds.argtypes=[ct.c_void_p,ct.c_uint32,ct.POINTER(ct.c_uint64),ct.c_char_p,ct.c_size_t]
        self._lib.cbm_session_set_yeti_oracle_seeds.restype=ct.c_int
        self._lib.cbm_session_set_yeti_leaf_seeds.argtypes=self._lib.cbm_session_set_yeti_oracle_seeds.argtypes
        self._lib.cbm_session_set_yeti_leaf_seeds.restype=ct.c_int
        bins,targets,weights,initial,features,borders,types,*_=arrays
        yeti=YetiOptions(len(offsets)-1,permutations,decay,legacy_prefix_centering)
        error=ct.create_string_buffer(2048)
        self._check(self._lib.cbm_session_create_yeti(ct.byref(self._params),ct.byref(objective),ct.byref(yeti),
            _native._u32(offsets),_native._u8(bins),_native._f32(targets),_native._f32(weights),_native._f32(initial),
            _native._u32(features),_native._u32(borders),_native._u8(types),ct.byref(self._handle),error,len(error)),error)
        try:
            self._check(self._lib.cbm_session_set_bootstrap(self._handle,ct.byref(bootstrap),error,len(error)),error)
            noise=_native.ScoreNoiseOptions(config['random_strength'],0,0,0)
            self._check(self._lib.cbm_session_set_score_noise(self._handle,ct.byref(noise),error,len(error)),error)
        except Exception:
            self.close(); raise

    @property
    def leaf_seed_count(self):
        count=self._params.leaf_estimation_iterations
        return self._permutation_count*(count+int(count>1))

    def set_oracle_seeds(self, seeds, *, leaf_only=False):
        with self._lock:
            self._require_open()
            values=list(seeds)
            counts=(self.leaf_seed_count,) if leaf_only else (1,self.leaf_seed_count+1)
            if (len(values) not in counts or any(
                    isinstance(v,(bool,np.bool_)) or not isinstance(v,numbers.Integral) or not 0<=v<2**64 for v in values)):
                raise ValueError('Supply the expected uint64 weak/leaf seeds, including the final evaluation draw when I>1.')
            values=np.ascontiguousarray(values,np.uint64);error=ct.create_string_buffer(2048)
            operation=self._lib.cbm_session_set_yeti_leaf_seeds if leaf_only else self._lib.cbm_session_set_yeti_oracle_seeds
            self._check(operation(self._handle,len(values),
                values.ctypes.data_as(ct.POINTER(ct.c_uint64)),error,len(error)),error)

    def step(self, oracle_seeds):
        with self._lock:
            self.set_oracle_seeds(oracle_seeds)
            return super().step()

    def begin_tree(self, oracle_seeds):
        with self._lock:
            self.set_oracle_seeds(oracle_seeds)
            return super().begin_tree()

    def finish_tree(self, leaf_seeds=None):
        with self._lock:
            if leaf_seeds is not None:self.set_oracle_seeds(leaf_seeds,leaf_only=True)
            return super().finish_tree()


class TrainingSession(Session):
    """DocParallel controller with persisted target RNG and real PFound history."""
    def __init__(self, bins, targets, candidate_features, candidate_bins, *,
                 objective='YetiRank', initial_rng_state=None, dataset_permutations=1, **options):
        from ._yeti_rng import YetiRankRng
        if objective != 'YetiRank':
            raise ValueError('This controller implements classic YetiRank only.')
        self.rng = YetiRankRng(options.get('random_seed', 0), options.get('bootstrap_type', 'No'),
                              options.get('leaf_estimation_iterations', 1),
                              initial_state=initial_rng_state, iteration_offset=options.get('iteration_offset', 0),
                              dataset_permutations=dataset_permutations)
        self._targets = np.asarray(targets, np.float32).copy()
        self._weights = options.get('sample_weight')
        self._offsets = validate_offsets(options['group_offsets'], len(targets))
        from ._query_data import validate_subgroup_hashes
        self._subgroups = validate_subgroup_hashes(options.pop('subgroup_hashes', None), len(targets))
        if (self._targets.ndim != 1 or not np.isfinite(self._targets).all()
                or (self._targets < 0).any() or (self._targets > 1).any()):
            raise ValueError('Classic YetiRank with PFound requires relevance labels in [0, 1].')
        if not (np.diff(self._offsets) > 1).any():
            raise ValueError('YetiRank requires at least one query containing multiple rows.')
        self._candidate_count = len(candidate_features)
        self._legacy_centering = options.get('legacy_prefix_centering', False)
        super().__init__(bins, targets, candidate_features, candidate_bins, **options)
        try:
            self._metric_history = [self._metric()]
        except Exception:
            self.close()
            raise

    def _metric(self):
        from ._training import _shared_metric
        return _shared_metric('PFound', self.predictions(), self._targets, self._weights, self._offsets, subgroup_hashes=self._subgroups)

    def configure_permutations(self, bins_list, initial_predictions=None, mvs_lambdas=None, mvs_valid=None):
        with self._lock:
            if len(bins_list) != self.rng.dataset_permutations:
                raise ValueError('Permutation matrices must match the YetiRank RNG dataset_permutations.')
            super().configure_permutations(bins_list, initial_predictions, mvs_lambdas, mvs_valid)
            self._metric_history = [self._metric()]

    def step(self):
        with self._lock:
            if self._permutation_count != self.rng.dataset_permutations:
                raise ValueError('Configure every YetiRank permutation matrix before training.')
            from ._data import cuda_search_permutation
            self.select_permutation(cuda_search_permutation(self.rng.seed, self.rng.completed_iterations,
                                                           self._permutation_count))
            self.begin_tree([self.rng.begin()])
            attempts = 0
            while True:
                status = self.grow_tree()
                if self._params.train.depth and self._candidate_count:
                    attempts += 1
                if status['finished']:
                    break
            tree = self.finish_tree(self.rng.leaves(attempts))
            self.rng.complete()
            tree.loss = self._metric()
            self._metric_history.append(tree.loss)
            return tree

    def result(self):
        result = super().result()
        result.rmse = np.asarray(self._metric_history, np.float32)
        result.stats['yeti_rng'] = self.rng.state()
        result.stats['yeti_centering'] = ('legacy_prefix' if self._legacy_centering else 'all_rows')
        return result
