from .core import Pool, CatBoostError, get_catboost_bin_module, ARRAY_TYPES
from collections import defaultdict
import numpy as np

_catboost = get_catboost_bin_module()
_eval_metric_util = _catboost._eval_metric_util
_get_roc_curve = _catboost._get_roc_curve
_select_threshold = _catboost._select_threshold

compute_wx_test = _catboost.compute_wx_test


def create_cd(
    label=None,
    cat_features=None,
    weight=None,
    baseline=None,
    doc_id=None,
    group_id=None,
    subgroup_id=None,
    timestamp=None,
    auxiliary_columns=None,
    feature_names=None,
    output_path='train.cd'
):
    _from_param_to_cd = {
        'label': 'Label',
        'weight': 'Weight',
        'baseline': 'Baseline',
        'doc_id': 'DocId',
        'group_id': 'GroupId',
        'subgroup_id': 'SubgroupId',
        'timestamp': 'Timestamp'
    }
    _column_description = defaultdict(lambda: ['Num', ''])
    for key, value in locals().copy().items():
        if not (key.startswith('_') or value is None):
            if key in ('cat_features', 'auxiliary_columns'):
                if isinstance(value, int):
                    value = [value]
                for index in value:
                    if not isinstance(index, int):
                        raise CatBoostError('Unsupported index type. Expected int, got {}'.format(type(index)))
                    if index in _column_description:
                        raise CatBoostError('The index {} occurs more than once'.format(index))
                    _column_description[index] = ['Categ', ''] if key == 'cat_features' else ['Auxiliary', '']
            elif key not in ('feature_names', 'output_path'):
                if not isinstance(value, int):
                    raise CatBoostError('Unsupported index type. Expected int, got {}'.format(type(value)))
                if value in _column_description:
                    raise CatBoostError('The index {} occurs more than once'.format(value))
                _column_description[value] = [_from_param_to_cd[key], '']
    if feature_names is not None:
        for feature_index, name in feature_names.items():
            real_feature_index = feature_index
            for column_index, (title, _) in sorted(_column_description.items()):
                if column_index > real_feature_index:
                    break
                if title not in ('Num', 'Categ'):
                    real_feature_index += 1
            _column_description[real_feature_index][1] = name
    with open(output_path, 'w') as f:
        for index, (title, name) in sorted(_column_description.items()):
            f.write('{}\t{}\t{}\n'.format(index, title, name))


def eval_metric(label, approx, metric, weight=None, group_id=None, thread_count=-1):
    """
    Evaluate metrics with raw approxes and labels.

    Parameters
    ----------
    label : list or numpy.arrays or pandas.DataFrame or pandas.Series
        Object labels.

    approx : list or numpy.arrays or pandas.DataFrame or pandas.Series
        Object approxes.

    metric : string
        Metric name.

    weight : list or numpy.array or pandas.DataFrame or pandas.Series, optional (default=None)
        Object weights.

    group_id : list or numpy.array or pandas.DataFrame or pandas.Series, optional (default=None)
        Object group ids.

    thread_count : int, optional (default=-1)
        Number of threads to work with.
        If -1, then the number of threads is set to the number of CPU cores.

    Returns
    -------
    metric results : list with metric values.
    """
    if len(approx) == 0:
        approx = [[]]
    if not isinstance(approx[0], ARRAY_TYPES):
        approx = [approx]
    return _eval_metric_util(label, approx, metric, weight, group_id, thread_count)


def get_gpu_device_count():
    return get_catboost_bin_module()._get_gpu_device_count()


def reset_trace_backend(filename):
    get_catboost_bin_module()._reset_trace_backend(filename)


def get_roc_curve(model, data, thread_count=-1):
    """
    Build points of ROC curve.

    Parameters
    ----------
    model : catboost.CatBoost
        The trained model.

    data : catboost.Pool or list of catboost.Pool
        A set of samples to build ROC curve with.

    thread_count : int (default=-1)
        Number of threads to work with.
        If -1, then the number of threads is set to the number of CPU cores.

    Returns
    -------
    curve points : tuple of three arrays (fpr, tpr, thresholds)
    """
    if type(data) == Pool:
        data = [data]
    if not isinstance(data, list):
        raise CatBoostError('data must be a catboost.Pool or list of pools.')
    for pool in data:
        if not isinstance(pool, Pool):
            raise CatBoostError('one of data pools is not catboost.Pool')

    return _get_roc_curve(model._object, data, thread_count)


def get_fpr_curve(model=None, data=None, curve=None, thread_count=-1):
    """
    Build points of FPR curve.

    Parameters
    ----------
    model : catboost.CatBoost
        The trained model.

    data : catboost.Pool or list of catboost.Pool
        A set of samples to build ROC curve with.

    curve : tuple of three arrays (fpr, tpr, thresholds)
        ROC curve points in format of get_roc_curve returned value.
        If set, data parameter must not be set.

    thread_count : int (default=-1)
        Number of threads to work with.
        If -1, then the number of threads is set to the number of CPU cores.

    Returns
    -------
    curve points : tuple of two arrays (thresholds, fpr)
    """
    if curve is not None:
        if data is not None:
            raise CatBoostError('Only one of the parameters data and curve should be set.')
        if not (isinstance(curve, list) or isinstance(curve, tuple)) or len(curve) != 3:
            raise CatBoostError('curve must be list or tuple of three arrays (fpr, tpr, thresholds).')
        fpr, thresholds = curve[0][:], curve[2][:]
    else:
        if model is None or data is None:
            raise CatBoostError('model and data parameters should be set when curve parameter is None.')
        fpr, _, thresholds = get_roc_curve(model, data, thread_count)
    return thresholds, fpr


def get_fnr_curve(model=None, data=None, curve=None, thread_count=-1):
    """
    Build points of FNR curve.

    Parameters
    ----------
    model : catboost.CatBoost
        The trained model.

    data : catboost.Pool or list of catboost.Pool
        A set of samples to build ROC curve with.

    curve : tuple of three arrays (fpr, tpr, thresholds)
        ROC curve points in format of get_roc_curve returned value.
        If set, data parameter must not be set.

    thread_count : int (default=-1)
        Number of threads to work with.
        If -1, then the number of threads is set to the number of CPU cores.

    Returns
    -------
    curve points : tuple of two arrays (thresholds, fnr)
    """
    if curve is not None:
        if data is not None:
            raise CatBoostError('Only one of the parameters data and curve should be set.')
        if not (isinstance(curve, list) or isinstance(curve, tuple)) or len(curve) != 3:
            raise CatBoostError('curve must be list or tuple of three arrays (fpr, tpr, thresholds).')
        tpr, thresholds = curve[1], curve[2][:]
    else:
        if model is None or data is None:
            raise CatBoostError('model and data parameters should be set when curve parameter is None.')
        _, tpr, thresholds = get_roc_curve(model, data, thread_count)
    fnr = np.array([1 - x for x in tpr])
    return thresholds, fnr


def select_threshold(model=None, data=None, curve=None, FPR=None, FNR=None, thread_count=-1):
    """
    Selects a threshold for prediction.

    Parameters
    ----------
    model : catboost.CatBoost
        The trained model.

    data : catboost.Pool or list of catboost.Pool
        Set of samples to build ROC curve with.
        If set, curve parameter must not be set.

    curve : tuple of three arrays (fpr, tpr, thresholds)
        ROC curve points in format of get_roc_curve returned value.
        If set, data parameter must not be set.

    FPR : desired false-positive rate

    FNR : desired false-negative rate (only one of FPR and FNR should be chosen)

    thread_count : int (default=-1)
        Number of threads to work with.
        If -1, then the number of threads is set to the number of CPU cores.

    Returns
    -------
    threshold : double
    """
    if data is not None:
        if curve is not None:
            raise CatBoostError('Only one of the parameters data and curve should be set.')
        if model is None:
            raise CatBoostError('model and data parameters should be set when curve parameter is None.')
        if type(data) == Pool:
            data = [data]
        if not isinstance(data, list):
            raise CatBoostError('data must be a catboost.Pool or list of pools.')
        for pool in data:
            if not isinstance(pool, Pool):
                raise CatBoostError('one of data pools is not catboost.Pool')
    elif curve is not None:
        if not (isinstance(curve, list) or isinstance(curve, tuple)) or len(curve) != 3:
            raise CatBoostError('curve must be list or tuple of three arrays (fpr, tpr, thresholds).')
    else:
        raise CatBoostError('One of the parameters data and curve should be set.')

    return _select_threshold(model._object, data, curve, FPR, FNR, thread_count)
