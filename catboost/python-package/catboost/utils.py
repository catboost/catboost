from .core import Pool, DataFrame, CatboostError, get_catboost_bin_module, ARRAY_TYPES
from collections import defaultdict

_catboost = get_catboost_bin_module()
_eval_metric_util = _catboost._eval_metric_util
_get_roc_curve = _catboost._get_roc_curve
_select_decision_boundary = _catboost._select_decision_boundary


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
                        raise CatboostError('Unsupported index type. Expected int, got {}'.format(type(index)))
                    if index in _column_description:
                        raise CatboostError('The index {} occurs more than once'.format(index))
                    _column_description[index] = ['Categ', ''] if key == 'cat_features' else ['Auxiliary', '']
            elif key not in ('feature_names', 'output_path'):
                if not isinstance(value, int):
                    raise CatboostError('Unsupported index type. Expected int, got {}'.format(type(value)))
                if value in _column_description:
                    raise CatboostError('The index {} occurs more than once'.format(value))
                _column_description[value] = [_from_param_to_cd[key], '']
    if feature_names is not None:
        for index, name in feature_names.items():
            _column_description[index][1] = name
    with open(output_path, 'w') as f:
        for index, (title, name) in sorted(_column_description.items()):
            f.write('{}\t{}\t{}\n'.format(index, title, name))


def eval_metric(label, approx, metric, weight=None, group_id=None, thread_count=-1):
    if len(approx) == 0:
        approx = [[]]
    if not isinstance(approx[0], ARRAY_TYPES):
        approx = [approx]
    return _eval_metric_util(label, approx, metric, weight, group_id, thread_count)


def get_gpu_device_count():
    return get_catboost_bin_module()._get_gpu_device_count()


def get_roc_curve(model, data, thread_count=-1, as_pandas=True):
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
        If -1, then the number of threads is set to the number of cores.

    as_pandas : bool, optional (default=True)
        Return pandas.DataFrame when pandas is installed.
        If False or pandas is not installed, return dict.

    Returns
    -------
    curve points : pandas.DataFrame or dict
        columns: boundary, fnr, fpr
    """
    if type(data) == Pool:
        data = [data]
    if not isinstance(data, list):
        raise CatboostError('data must be a catboost.Pool or list of pools.')
    for pool in data:
        if not isinstance(pool, Pool):
            raise CatboostError('one of data pools is not catboost.Pool')

    return _get_roc_curve(model._object, data, thread_count, as_pandas)


def select_decision_boundary(model, data=None, curve=None, FPR=None, FNR=None, thread_count=-1):
    """
    Selects a probability boundary for prediction.

    Parameters
    ----------
    model : catboost.CatBoost
        The trained model.

    data : catboost.Pool or list of catboost.Pool
        Set of samples to build ROC curve with.
        If set, curve parameter must not be set.

    curve : pandas.DataFrame or dict
        ROC curve points in format of get_roc_curve returned value.
        If set, data parameter must not be set.

    FPR : desired false-positive rate

    FNR : desired false-negative rate (only one of FPR and FNR should be chosen)

    thread_count : int (default=-1)
        Number of threads to work with.
        If -1, then the number of threads is set to the number of cores.

    Returns
    -------
    boundary : double
    """
    if data is not None:
        if curve is not None:
            raise CatboostError('Only one of the parameters data and curve should be set.')
        if type(data) == Pool:
            data = [data]
        if not isinstance(data, list):
            raise CatboostError('data must be a catboost.Pool or list of pools.')
        for pool in data:
            if not isinstance(pool, Pool):
                raise CatboostError('one of data pools is not catboost.Pool')
    elif curve is not None:
        if not isinstance(curve, list) and not isinstance(curve, DataFrame):
            raise CatboostError('curve must be list or pandas.DataFrame.')
    else:
        raise CatboostError('One of the parameters data and curve should be set.')

    return _select_decision_boundary(model._object, data, curve, FPR, FNR, thread_count)
