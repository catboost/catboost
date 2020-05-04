from .core import Pool, CatBoostError, get_catboost_bin_module, ARRAY_TYPES, STRING_TYPES, _update_params_quantize_part, _process_synonyms
from collections import defaultdict
from contextlib import contextmanager
import numpy as np
import warnings

_catboost = get_catboost_bin_module()
_eval_metric_util = _catboost._eval_metric_util
_get_roc_curve = _catboost._get_roc_curve
_get_confusion_matrix = _catboost._get_confusion_matrix
_select_threshold = _catboost._select_threshold

compute_wx_test = _catboost.compute_wx_test
TargetStats = _catboost.TargetStats
DataMetaInfo = _catboost.DataMetaInfo
compute_training_options = _catboost.compute_training_options


@contextmanager
def _import_matplotlib():
    try:
        import matplotlib.pyplot as plt
    except ImportError as e:
        warnings.warn("To draw plots you should install matplotlib.")
        raise ImportError(str(e))
    yield plt


def _draw(plt, x, y, x_label, y_label, title):
    plt.figure(figsize=(16, 8))

    plt.plot(x, y, alpha=0.5, lw=2)

    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.grid(True)
    plt.xlabel(x_label, fontsize=16)
    plt.ylabel(y_label, fontsize=16)
    plt.title(title, fontsize=20)
    plt.show()


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


def read_cd(cd_file, column_count=None, data_file=None, canonize_column_types=False):
    """
    Reads CatBoost column description file
    (see https://catboost.ai/docs/concepts/input-data_column-descfile.html#input-data_column-descfile)

    Parameters
    ----------
    column_count : integer
    data_file : path to dataset file in CatBoost format
        specify either column_count directly or data_file to detect it

    canonize_column_types : bool
        if set to True types for columns with synonyms are renamed to canonical type.

    Returns
    -------
    dict with keys:
        "column_type_to_indices" :
            dict of column_type -> column_indices list, column_type is 'Label', 'Categ' etc.

        "column_dtypes" : dict of column_name -> numpy.dtype or 'category'

        "cat_feature_indices" : list of integers
            indices of categorical features in array of all features.
            Note: indices in array of features, not indices in array of all columns!

        "column_names" : list of strings

        "non_feature_column_indices" : list of integers
    """

    column_type_synonyms_map = {
        'Target': 'Label',
        'DocId': 'SampleId',
        'QueryId': 'GroupId'
    }

    if column_count is None:
        if data_file is None:
            raise Exception(
                'Cannot obtain column count: either specify column_count parameter or specify data_file '
                + 'parameter to get it'
            )
        with open(data_file) as f:
            column_count = len(f.readline()[:-1].split('\t'))

    column_type_to_indices = {}
    column_dtypes = {}
    cat_feature_indices = []
    text_feature_indices = []
    column_names = []
    non_feature_column_indices = []


    def add_missed_columns(start_column_idx, end_column_idx, non_feature_column_count):
        for missed_column_idx in range(start_column_idx, end_column_idx):
            column_name = 'feature_%i' % (missed_column_idx - non_feature_column_count)
            column_names.append(column_name)
            column_type_to_indices.setdefault('Num', []).append(missed_column_idx)
            column_dtypes[column_name] = np.float32

    last_column_idx = -1
    with open(cd_file) as f:
        for line_idx, line in enumerate(f):
            # some cd files in the wild contain empty lines
            if len(line.strip()) == 0:
                continue

            line_columns = line[:-1].split('\t')
            if len(line_columns) not in [2, 3]:
                raise Exception('Wrong number of columns in cd file')

            column_idx = int(line_columns[0])
            if column_idx <= last_column_idx:
                raise Exception('Non-increasing column indices in cd file')

            add_missed_columns(last_column_idx + 1, column_idx, len(non_feature_column_indices))

            column_type = line_columns[1]
            if canonize_column_types:
                column_type = column_type_synonyms_map.get(column_type, column_type)

            column_type_to_indices.setdefault(column_type, []).append(column_idx)

            column_name = None
            if len(line_columns) == 3:
                column_name = line_columns[2]

            if column_type in ['Num', 'Categ', 'Text']:
                feature_idx = column_idx - len(non_feature_column_indices)
                if column_name is None:
                    column_name = 'feature_%i' % feature_idx
                if column_type == 'Categ':
                    cat_feature_indices.append(feature_idx)
                    column_dtypes[column_name] = 'category'
                elif column_type == 'Text':
                    text_feature_indices.append(feature_idx)
                    column_dtypes[column_name] = object
                else:
                    column_dtypes[column_name] = np.float32
            else:
                non_feature_column_indices.append(column_idx)
                if column_name is None:
                    column_name = column_type

            column_names.append(column_name)

            last_column_idx = column_idx

    add_missed_columns(last_column_idx + 1, column_count, len(non_feature_column_indices))

    return {
        'column_type_to_indices' : column_type_to_indices,
        'column_dtypes' : column_dtypes,
        'cat_feature_indices' : cat_feature_indices,
        'text_feature_indices' : text_feature_indices,
        'column_names' : column_names,
        'non_feature_column_indices' : non_feature_column_indices
    }


def eval_metric(label, approx, metric, weight=None, group_id=None, subgroup_id=None, pairs=None, thread_count=-1):
    """
    Evaluate metrics with raw approxes and labels.

    Parameters
    ----------
    label : list or numpy.ndarrays or pandas.DataFrame or pandas.Series
        Object labels.

    approx : list or numpy.ndarrays or pandas.DataFrame or pandas.Series
        Object approxes.

    metric : string
        Metric name.

    weight : list or numpy.ndarray or pandas.DataFrame or pandas.Series, optional (default=None)
        Object weights.

    group_id : list or numpy.ndarray or pandas.DataFrame or pandas.Series, optional (default=None)
        Object group ids.

    subgroup_id : list or numpy.ndarray, optional (default=None)
        subgroup id for each instance.
        If not None, giving 1 dimensional array like data.

    pairs : list or numpy.ndarray or pandas.DataFrame or string
        The pairs description.
        If list or numpy.ndarrays or pandas.DataFrame, giving 2 dimensional.
        The shape should be Nx2, where N is the pairs' count. The first element of the pair is
        the index of winner object in the training set. The second element of the pair is
        the index of loser object in the training set.
        If string, giving the path to the file with pairs description.

    thread_count : int, optional (default=-1)
        Number of threads to work with.
        If -1, then the number of threads is set to the number of CPU cores.

    Returns
    -------
    metric results : list with metric values.
    """
    if len(label) > 0 and not isinstance(label[0], ARRAY_TYPES):
        label = [label]
    if len(approx) == 0:
        approx = [[]]
    if not isinstance(approx[0], ARRAY_TYPES):
        approx = [approx]
    return _eval_metric_util(label, approx, metric, weight, group_id, subgroup_id, pairs, thread_count)


def get_gpu_device_count():
    return get_catboost_bin_module()._get_gpu_device_count()


def reset_trace_backend(filename):
    get_catboost_bin_module()._reset_trace_backend(filename)


def get_confusion_matrix(model, data, thread_count=-1):
    """
    Build confusion matrix.

    Parameters
    ----------
    model : catboost.CatBoost
        The trained model.

    data : catboost.Pool
        A set of samples to build confusion matrix with.

    thread_count : int (default=-1)
        Number of threads to work with.
        If -1, then the number of threads is set to the number of CPU cores.

    Returns
    -------
    confusion matrix : array, shape = [n_classes, n_classes]
    """
    if not isinstance(data, Pool):
        raise CatBoostError('data must be a catboost.Pool')

    return _get_confusion_matrix(model._object, data, thread_count);


def get_roc_curve(model, data, thread_count=-1, plot=False):
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

    plot : bool, optional (default=False)
        If True, draw curve.

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

    roc_curve = _get_roc_curve(model._object, data, thread_count)

    if plot:
        with _import_matplotlib() as plt:
            _draw(plt, roc_curve[0], roc_curve[1], 'False Positive Rate', 'True Positive Rate', 'ROC Curve')

    return roc_curve


def get_fpr_curve(model=None, data=None, curve=None, thread_count=-1, plot=False):
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

    plot : bool, optional (default=False)
        If True, draw curve.

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

    if plot:
        with _import_matplotlib() as plt:
            _draw(plt, thresholds, fpr, 'Thresholds', 'False Positive Rate', 'FPR Curve')

    return thresholds, fpr


def get_fnr_curve(model=None, data=None, curve=None, thread_count=-1, plot=False):
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

    plot : bool, optional (default=False)
        If True, draw curve.

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

    if plot:
        with _import_matplotlib() as plt:
            _draw(plt, thresholds, fnr, 'Thresholds', 'False Negative Rate', 'FNR Curve')

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
        return _select_threshold(model._object, data, None, FPR, FNR, thread_count)
    elif curve is not None:
        if not (isinstance(curve, list) or isinstance(curve, tuple)) or len(curve) != 3:
            raise CatBoostError('curve must be list or tuple of three arrays (fpr, tpr, thresholds).')
        return _select_threshold(None, None, curve, FPR, FNR, thread_count)
    else:
        raise CatBoostError('One of the parameters data and curve should be set.')


def quantize(
    data_path,
    column_description=None,
    pairs=None,
    delimiter='\t',
    has_header=False,
    feature_names=None,
    thread_count=-1,
    ignored_features=None,
    per_float_feature_quantization=None,
    border_count=None,
    max_bin=None,
    feature_border_type=None,
    nan_mode=None,
    input_borders=None,
    task_type=None,
    used_ram_limit=None,
    random_seed=None,
    **kwargs
):
    """
    Construct quantized Pool from non-quantized pool stored in file.
    This method does not load whole non-quantized source dataset into memory
    so it can be used for huge datasets that fit in memory only after quantization.

    Parameters
    ----------
    data_path : string
        Path (with optional scheme) to non-quantized dataset.

    column_description : string, [default=None]
        ColumnsDescription parameter.
        There are several columns description types: Label, Categ, Num, Auxiliary, DocId, Weight, Baseline, GroupId, Timestamp.
        All columns are Num as default, it's not necessary to specify
        this type of columns. Default Label column index is 0 (zero).
        If None, Label column is 0 (zero) as default, all data columns are Num as default.
        If string, giving the path to the file with ColumnsDescription in column_description format.

    pairs : string, [default=None]
        Path to the file with pairs description.

    has_header : bool, [default=False]
        If True, read column names from first line.

    feature_names : string, [default=None]
        Path with scheme for feature names data to load.

    thread_count : int, [default=-1]
        Thread count for data processing.
        If -1, then the number of threads is set to the number of CPU cores.

    ignored_features : list, [default=None]
        Indices or names of features that should be excluded when training.

    per_float_feature_quantization : list of strings, [default=None]
        List of float binarization descriptions.
        Format : described in documentation on catboost.ai
        Example 1: ['0:1024'] means that feature 0 will have 1024 borders.
        Example 2: ['0:border_count=1024', '1:border_count=1024', ...] means that two first features have 1024 borders.
        Example 3: ['0:nan_mode=Forbidden,border_count=32,border_type=GreedyLogSum',
                    '1:nan_mode=Forbidden,border_count=32,border_type=GreedyLogSum'] - defines more quantization properties for first two features.

    border_count : int, [default = 254 for training on CPU or 128 for training on GPU]
        The number of partitions in numeric features binarization. Used in the preliminary calculation.
        range: [1,65535] on CPU, [1,255] on GPU

    max_bin : float, synonym for border_count.

    feature_border_type : string, [default='GreedyLogSum']
        The binarization mode in numeric features binarization. Used in the preliminary calculation.
        Possible values:
            - 'Median'
            - 'Uniform'
            - 'UniformAndQuantiles'
            - 'GreedyLogSum'
            - 'MaxLogSum'
            - 'MinEntropy'

    nan_mode : string, [default=None]
        Way to process missing values for numeric features.
        Possible values:
            - 'Forbidden' - raises an exception if there is a missing value for a numeric feature in a dataset.
            - 'Min' - each missing value will be processed as the minimum numerical value.
            - 'Max' - each missing value will be processed as the maximum numerical value.
        If None, then nan_mode=Min.

    input_borders : string, [default=None]
        input file with borders used in numeric features binarization.

    task_type : string, [default=None]
        The calcer type used to train the model.
        Possible values:
            - 'CPU'
            - 'GPU'

    used_ram_limit=None

    random_seed : int, [default=None]
        The random seed used for data sampling.
        If None, 0 is used.

    Returns
    -------
    pool : Pool
        Constructed and quantized pool.
    """
    if not data_path:
        raise CatBoostError("Data filename is empty.")
    if not isinstance(data_path, STRING_TYPES):
        raise CatBoostError("Data filename should be string type.")

    if pairs is not None and not isinstance(pairs, STRING_TYPES):
        raise CatBoostError("pairs should have None or string type when the pool is read from the file.")
    if column_description is not None and not isinstance(column_description, STRING_TYPES):
        raise CatBoostError("column_description should have None or string type when the pool is read from the file.")
    if feature_names is not None and not isinstance(feature_names, STRING_TYPES):
        raise CatBoostError("feature_names should have None or string type when the pool is read from the file.")

    params = {}
    _process_synonyms(params)

    if border_count is None:
        border_count = max_bin

    if 'dev_block_size' in kwargs:
        params['dev_block_size'] = kwargs.pop('dev_block_size')

    dev_max_subset_size_for_build_borders = kwargs.pop('dev_max_subset_size_for_build_borders', None)

    if kwargs:
        raise CatBoostError("got an unexpected keyword arguments: {}".format(kwargs.keys()))

    _update_params_quantize_part(
        params,
        ignored_features,
        per_float_feature_quantization,
        border_count,
        feature_border_type,
        None, # sparse_features_conflict_fraction
        None, # dev_efb_max_buckets
        nan_mode,
        input_borders,
        task_type,
        used_ram_limit,
        random_seed,
        dev_max_subset_size_for_build_borders)

    result = Pool(None)
    result._read(data_path, column_description, pairs, feature_names, delimiter, has_header, thread_count, params)

    return result
