"""Scalar Ordered group boundaries and effective weights for raw input."""
import numpy as np
from ._data import unpack_pool
from ._query_data import unpack_query_pool


def _has_groups(value):
    from catboost import Pool
    return isinstance(value, Pool) and value.get_group_id_hash() is not None


def unpack_ordered_data(X,y,weight,cats,eval_set,group_id,group_weight,eval_group_id,eval_group_weight):
    grouped=group_id is not None or group_weight is not None or _has_groups(X)
    sizes=None
    if grouped:
        raw,y,offsets,weight,cats,names=unpack_query_pool(X,y,weight,group_id,group_weight,cats)
        if len(offsets)<5:
            raise ValueError('Ordered requires at least four groups or documents.')
        sizes=np.diff(offsets).tolist()
    else:
        raw,y,weight,cats,names=unpack_pool(X,y,weight,cats)
    if eval_set is None:
        if eval_group_id is not None or eval_group_weight is not None:
            raise ValueError('Evaluation group metadata requires eval_set.')
    else:
        from catboost import Pool
        if isinstance(eval_set,Pool):
            eval_X,eval_y,eval_weight=eval_set,None,None
        elif isinstance(eval_set,tuple) and len(eval_set) in (2,3):
            eval_X,eval_y=eval_set[:2];eval_weight=eval_set[2] if len(eval_set)==3 else None
        else:
            raise ValueError('eval_set must be one Pool or an (X, y[, weight]) tuple.')
        if eval_group_id is not None or eval_group_weight is not None or _has_groups(eval_X):
            eval_raw,eval_y,_,eval_weight,eval_cats,eval_names=unpack_query_pool(
                eval_X,eval_y,eval_weight,eval_group_id,eval_group_weight,cats)
            named=hasattr(eval_X,'columns') or (isinstance(eval_X,Pool) and any(eval_X.get_feature_names()))
            if named and eval_names!=names:
                raise ValueError('Validation feature names/order must match training data.')
            if eval_cats!=cats or eval_raw.shape[1]!=raw.shape[1]:
                raise ValueError('Validation features and categorical indices must match training data.')
            eval_set=(eval_raw,eval_y,eval_weight)
    return raw,y,weight,cats,names,sizes,eval_set
