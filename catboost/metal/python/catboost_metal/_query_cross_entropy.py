"""Resident CUDA QueryCrossEntropy target, full matrix search and Newton leaves.

Query scales are selected before entering this private quantized controller.
Only whole-query Bernoulli sampling affects the weak target. Original queries
and document weights drive every leaf step and GPU objective measurement.
"""
import ctypes as ct
import threading
import numpy as np
from . import _native
from ._query_data import validate_offsets


class QueryOptions(ct.Structure):
    _fields_ = [(name, ct.c_uint32) for name in ('group_count', 'reserved0', 'reserved1', 'reserved2')]


class Session(_native.Session):
    def __init__(self, bins, targets, candidate_features, candidate_bins, *, group_offsets,
                 query_scales=None, alpha=.95, non_diagonal_regularization=.1, **options):
        self._handle=ct.c_void_p(); self._lock=threading.RLock(); self._completed=0
        self._permutation_count=1; self._permutations_configured=False
        self._feature_penalties_configured=False; self._lib=None
        config=dict(iterations=1, depth=6, learning_rate=.03, l2_leaf_reg=1., bias=0.,
            score_function='NewtonL2', sample_weight=None, leaf_estimation_iterations=10,
            leaf_estimation_backtracking='No', initial_predictions=None, candidate_types=None,
            objective_param=None, leaf_estimation_method='Newton', bootstrap_type='No', random_seed=0,
            iteration_offset=0, bagging_temperature=1., subsample=1., mvs_reg=None,
            initial_mvs_lambda=None, random_strength=0.)
        config.update(options)
        if config['leaf_estimation_method']=='Simple' and 'leaf_estimation_iterations' not in options:
            config['leaf_estimation_iterations']=1
        if 'objective' in config: raise ValueError('This session implements QueryCrossEntropy only.')
        if config['leaf_estimation_method'] not in ('Newton','Simple'):
            raise ValueError('QueryCrossEntropy requires Newton or Simple leaves like CUDA.')
        if config['score_function']=='L2': raise ValueError('QueryCrossEntropy does not support L2 structure score like CUDA.')
        if config['bootstrap_type'] not in ('No','Bernoulli'):
            raise ValueError('QueryCrossEntropy supports No or Bernoulli query bootstrap only.')
        self._params, objective, bootstrap, arrays = _native._prepare(
            bins,targets,candidate_features,candidate_bins,objective='RMSE',_full_matrix=True,**config)
        if self._params.train.depth>8: raise ValueError('QueryCrossEntropy supports depth <= 8 like CUDA.')
        with np.errstate(over='ignore',invalid='ignore'):
            non_diag=np.float32(non_diagonal_regularization); alpha=np.float32(alpha)
        if not np.isfinite(non_diag) or non_diag<0: raise ValueError('Non-diagonal regularization must be finite and nonnegative.')
        if not np.isfinite(alpha) or not 0<=alpha<=1: raise ValueError('QueryCrossEntropy alpha must be in [0, 1].')
        offsets=validate_offsets(group_offsets,self._params.train.rows)
        if np.diff(offsets).max()>256: raise ValueError('QueryCrossEntropy supports queries of 1..256 rows like CUDA.')
        with np.errstate(over='ignore',invalid='ignore'):
            scales=np.ones(len(offsets)-1,np.float32) if query_scales is None else np.ascontiguousarray(query_scales,np.float32)
        if scales.shape!=(len(offsets)-1,) or not np.isfinite(scales).all():
            raise ValueError('query_scales must contain one finite float32 scale per query.')
        bins,targets,objects,initial,features,borders,types,*_=arrays
        if (targets<0).any() or (targets>1).any(): raise ValueError('QueryCrossEntropy targets must be in [0, 1].')
        self._params.objective=objective.objective=16;objective.objective_param=alpha;self.objective='QueryCrossEntropy'
        self._lib=_native._load(_native.build_library())
        f32,u32,u8=ct.POINTER(ct.c_float),ct.POINTER(ct.c_uint32),ct.POINTER(ct.c_uint8)
        self._lib.cbm_session_create_query_cross_entropy.argtypes=[ct.POINTER(_native.SessionParams),ct.POINTER(_native.ObjectiveOptions),
            ct.POINTER(QueryOptions),ct.c_float,u32,f32,u8,f32,f32,f32,u32,u32,u8,
            ct.POINTER(ct.c_void_p),ct.c_char_p,ct.c_size_t]
        self._lib.cbm_session_create_query_cross_entropy.restype=ct.c_int
        query=QueryOptions(len(offsets)-1,0,0,0);error=ct.create_string_buffer(2048)
        self._check(self._lib.cbm_session_create_query_cross_entropy(ct.byref(self._params),ct.byref(objective),ct.byref(query),non_diag,
            _native._u32(offsets),_native._f32(scales),_native._u8(bins),_native._f32(targets),_native._f32(objects),
            _native._f32(initial),_native._u32(features),_native._u32(borders),_native._u8(types),ct.byref(self._handle),error,len(error)),error)
        try:
            self._check(self._lib.cbm_session_set_bootstrap(self._handle,ct.byref(bootstrap),error,len(error)),error)
            noise=_native.ScoreNoiseOptions(config['random_strength'],0,0,0)
            self._check(self._lib.cbm_session_set_score_noise(self._handle,ct.byref(noise),error,len(error)),error)
        except Exception:
            self.close(); raise


class TrainingSession(Session):
    def __init__(self,bins,targets,candidate_features,candidate_bins,*,objective='QueryCrossEntropy',**options):
        if objective!='QueryCrossEntropy':raise ValueError('This controller implements QueryCrossEntropy only.')
        super().__init__(bins,targets,candidate_features,candidate_bins,**options)


def select_scales(description, targets, offsets):
    """CUDA raw_values_scale lookup with bounded storage independent of sizes."""
    offsets=validate_offsets(offsets,len(targets));targets=np.asarray(targets,np.float32)
    if targets.ndim!=1 or not np.isfinite(targets).all() or (targets<0).any() or (targets>1).any():
        raise ValueError('QueryCrossEntropy targets must be finite and in [0, 1].')
    if np.diff(offsets).max()>256:raise ValueError('QueryCrossEntropy supports queries of 1..256 rows like CUDA.')
    if not isinstance(description,str) or len(description)>2**20:raise ValueError('raw_values_scale must be a string of at most 1 MiB.')
    entries={};default=1.;has_default=False
    if description:
        for token in description.split(' '):
            try:
                key,value=token.split(':');size,count=key.split(',')
                # CUDA TryFromString<uint32> accepts decimal digits, with no signs.
                if not size.isascii() or not count.isascii() or not size.isdecimal() or not count.isdecimal():raise ValueError
                size,count=int(size),int(count)
                with np.errstate(over='ignore',invalid='ignore'):scale=np.float32(value)
                if not 0<=count<=size<=2**32-1 or not np.isfinite(scale):raise ValueError
            except (ValueError,TypeError,OverflowError):
                raise ValueError('raw_values_scale requires group_size,true_count:scale entries with finite float32 scales.') from None
            if size==0 and not has_default:default=scale;has_default=True
            if size<=256:entries[size,count]=scale
    return np.array([entries.get((int(b-a),int(np.count_nonzero(targets[a:b]>.5))),default)
        for a,b in zip(offsets[:-1],offsets[1:])],np.float32)


def _metric_metadata(targets,sample_weight,group_offsets,query_scales):
    y=np.ascontiguousarray(targets,np.float32)
    if y.ndim!=1 or not len(y) or not np.isfinite(y).all():
        raise ValueError('QueryCrossEntropy metric needs finite target vectors.')
    offsets=validate_offsets(group_offsets,len(y));groups=len(offsets)-1
    weights=None if sample_weight is None else np.ascontiguousarray(sample_weight,np.float32)
    if weights is not None and (weights.shape!=y.shape or not np.isfinite(weights).all()):
        raise ValueError('QueryCrossEntropy metric needs matching finite weights.')
    scales=np.ones(groups,np.float32) if query_scales is None else np.ascontiguousarray(query_scales,np.float32)
    if scales.shape!=(groups,) or not np.isfinite(scales).all():raise ValueError('QueryCrossEntropy metric needs one finite scale per query.')
    return y,weights,offsets,scales


def metric(predictions,targets,sample_weight,group_offsets,*,alpha=.95,query_scales=None):
    """GPU target metric, including scales and CUDA single-class tolerance."""
    y,weights,offsets,scales=_metric_metadata(targets,sample_weight,group_offsets,query_scales)
    point=np.ascontiguousarray(predictions,np.float32);groups=len(offsets)-1
    if point.shape!=y.shape or not np.isfinite(point).all():
        raise ValueError('QueryCrossEntropy metric needs matching finite prediction and target vectors.')
    lib=_native._load(_native.build_library());f32=ct.POINTER(ct.c_float);u32=ct.POINTER(ct.c_uint32)
    lib.cbm_query_cross_entropy_metric.argtypes=[ct.c_uint32,ct.c_uint32,f32,f32,f32,u32,f32,ct.c_float,
        ct.POINTER(ct.c_double),ct.c_char_p,ct.c_size_t]
    lib.cbm_query_cross_entropy_metric.restype=ct.c_int
    result=ct.c_double();error=ct.create_string_buffer(2048)
    code=lib.cbm_query_cross_entropy_metric(len(y),groups,_native._f32(y),_native._f32(weights),_native._f32(point),
        _native._u32(offsets),_native._f32(scales),alpha,ct.byref(result),error,len(error))
    if code:raise ValueError(error.value.decode())
    return result.value


class MetricSession:
    """Immutable metric metadata with bounded, reusable GPU evaluation buffers."""
    def __init__(self,targets,sample_weight,group_offsets,query_scales=None,*,budget=1<<30):
        self._handle=ct.c_void_p();self._lock=threading.RLock();self.evaluations=0
        y,weights,offsets,scales=_metric_metadata(targets,sample_weight,group_offsets,query_scales)
        if isinstance(budget,bool) or not isinstance(budget,(int,np.integer)) or not 0<budget<=1<<30:
            raise ValueError('QueryCrossEntropy metric budget must be in 1..2^30 bytes.')
        self._rows=len(y);self._lib=_native._load(_native.build_library())
        f32=ct.POINTER(ct.c_float);u32=ct.POINTER(ct.c_uint32);u64=ct.POINTER(ct.c_uint64)
        self._lib.cbm_query_cross_entropy_metric_create.argtypes=[ct.c_uint32,ct.c_uint32,f32,f32,u32,f32,ct.c_uint64,ct.POINTER(ct.c_void_p),u64,ct.c_char_p,ct.c_size_t]
        self._lib.cbm_query_cross_entropy_metric_create.restype=ct.c_int
        self._lib.cbm_query_cross_entropy_metric_evaluate.argtypes=[ct.c_void_p,ct.c_uint32,f32,ct.c_float,ct.POINTER(ct.c_double),u64,ct.c_char_p,ct.c_size_t]
        self._lib.cbm_query_cross_entropy_metric_evaluate.restype=ct.c_int
        self._lib.cbm_query_cross_entropy_metric_destroy.argtypes=[ct.c_void_p];self._lib.cbm_query_cross_entropy_metric_destroy.restype=None
        allocated=ct.c_uint64();error=ct.create_string_buffer(2048)
        code=self._lib.cbm_query_cross_entropy_metric_create(self._rows,len(offsets)-1,_native._f32(y),_native._f32(weights),
            _native._u32(offsets),_native._f32(scales),budget,ct.byref(self._handle),ct.byref(allocated),error,len(error))
        if code:raise ValueError(error.value.decode())
        self.allocated_bytes=allocated.value

    def evaluate(self,predictions,alpha=.95):
        with self._lock:
            if not self._handle.value:raise ValueError('QueryCrossEntropy metric session is closed.')
            point=np.ascontiguousarray(predictions,np.float32)
            if point.shape!=(self._rows,) or not np.isfinite(point).all():
                raise ValueError('QueryCrossEntropy metric needs matching finite predictions.')
            result=ct.c_double();count=ct.c_uint64();error=ct.create_string_buffer(2048)
            code=self._lib.cbm_query_cross_entropy_metric_evaluate(self._handle,self._rows,_native._f32(point),alpha,
                ct.byref(result),ct.byref(count),error,len(error))
            if code:raise ValueError(error.value.decode())
            self.evaluations=count.value;return result.value

    def close(self):
        with self._lock:
            if self._handle.value:self._lib.cbm_query_cross_entropy_metric_destroy(self._handle);self._handle=ct.c_void_p()
    def __enter__(self):return self
    def __exit__(self,*args):self.close()
    def __del__(self):
        if hasattr(self,'_lock'):self.close()


def parse_description(description, *, metric=False):
    if not isinstance(description,str):raise ValueError('QueryCrossEntropy description must be a string.')
    base,separator,arguments=description.partition(':')
    if base!='QueryCrossEntropy':raise ValueError('Expected QueryCrossEntropy loss description.')
    result={};allowed={'alpha','raw_values_scale'}|({'use_weights','hints'} if metric else set())
    for token in arguments.split(';') if separator else []:
        key,equals,value=token.partition('=')
        if not equals or key in result or key not in allowed:raise ValueError('Invalid or duplicate QueryCrossEntropy parameter.')
        if key=='alpha':
            try:value=float(value)
            except ValueError:raise ValueError('QueryCrossEntropy alpha must be in [0, 1].') from None
            if not np.isfinite(value) or not 0<=value<=1:raise ValueError('QueryCrossEntropy alpha must be in [0, 1].')
        elif key=='raw_values_scale':select_scales(value,[0],[0,1])
        elif key=='use_weights':
            if value.lower() not in ('true','false','1','0','yes','no','on','off'):raise ValueError('Metric use_weights must be a valid boolean.')
        elif key=='hints':
            if value not in ('skip_train~true','skip_train~false'):raise ValueError('Unsupported QueryCrossEntropy metric hint.')
        result[key]=value
    return result
