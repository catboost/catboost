
import sys
from copy import deepcopy
from six import iteritems, string_types, integer_types
import os
import imp
from collections import Iterable, Sequence, Mapping, MutableMapping
import warnings
import numpy as np
import ctypes
import platform
import tempfile
from enum import Enum
from operator import itemgetter

if platform.system() == 'Linux':
    try:
        ctypes.CDLL('librt.so')
    except Exception:
        pass

try:
    from pandas import DataFrame, Series
except ImportError:
    class DataFrame(object):
        pass

    class Series(object):
        pass


def get_so_paths(dir_name):
    dir_name = os.path.join(os.path.dirname(__file__), dir_name)
    list_dir = os.listdir(dir_name) if os.path.isdir(dir_name) else []
    return [os.path.join(dir_name, so_name) for so_name in list_dir if so_name.split('.')[-1] in ['so', 'pyd']]


def get_catboost_bin_module():
    if '_catboost' in sys.modules:
        return sys.modules['_catboost']
    so_paths = get_so_paths('./')
    for so_path in so_paths:
        try:
            loaded_catboost = imp.load_dynamic('_catboost', so_path)
            sys.modules['catboost._catboost'] = loaded_catboost
            return loaded_catboost
        except ImportError:
            pass
    import _catboost
    return _catboost


_catboost = get_catboost_bin_module()
_PoolBase = _catboost._PoolBase
_CatBoost = _catboost._CatBoost
_MetricCalcerBase = _catboost._MetricCalcerBase
_cv = _catboost._cv
_set_logger = _catboost._set_logger
_reset_logger = _catboost._reset_logger
_configure_malloc = _catboost._configure_malloc
CatBoostError = _catboost.CatBoostError
_metric_description_or_str_to_str = _catboost._metric_description_or_str_to_str
is_classification_objective = _catboost.is_classification_objective
is_regression_objective = _catboost.is_regression_objective
_PreprocessParams = _catboost._PreprocessParams
_check_train_params = _catboost._check_train_params
_MetadataHashProxy = _catboost._MetadataHashProxy
_NumpyAwareEncoder = _catboost._NumpyAwareEncoder
FeaturesData = _catboost.FeaturesData


from contextlib import contextmanager  # noqa E402


_configure_malloc()
_catboost._library_init()

INTEGER_TYPES = (integer_types, np.integer)
FLOAT_TYPES = (float, np.floating)
STRING_TYPES = (string_types,)
ARRAY_TYPES = (list, np.ndarray, DataFrame, Series)


@contextmanager
def log_fixup():
    _set_logger(sys.stdout, sys.stderr)
    yield
    _reset_logger()


def _cast_to_base_types(value):
    # NOTE: Special case, avoiding new list creation.
    if isinstance(value, list):
        for index, element in enumerate(value):
            value[index] = _cast_to_base_types(element)
        return value
    if isinstance(value, ARRAY_TYPES[1:]):
        new_value = []
        for element in value:
            new_value.append(_cast_to_base_types(element))
        return new_value
    if isinstance(value, (Mapping, MutableMapping)):
        for key in list(value):
            value[key] = _cast_to_base_types(value[key])
        return value
    if isinstance(value, bool):
        return value
    if isinstance(value, INTEGER_TYPES):
        return int(value)
    if isinstance(value, FLOAT_TYPES):
        return float(value)
    return value


def metric_description_or_str_to_str(description):
    return _metric_description_or_str_to_str(description)


def _check_param_type(value, name, types, or_none=True):
    if not isinstance(value, types + ((type(None),) if or_none else ())):
        raise CatBoostError('Parameter {} should have a type of {}, got {}'.format(name, types, type(value)))


def _process_verbose(metric_period=None, verbose=None, logging_level=None, verbose_eval=None, silent=None):
    _check_param_type(metric_period, 'metric_period', (int,))
    _check_param_type(verbose, 'verbose', (bool, int))
    _check_param_type(logging_level, 'logging_level', (string_types,))
    _check_param_type(verbose_eval, 'verbose_eval', (bool, int))
    _check_param_type(silent, 'silent', (bool,))

    params = locals()
    exclusive_params = ['verbose', 'logging_level', 'verbose_eval', 'silent']
    at_most_one = sum(params.get(exclusive) is not None for exclusive in exclusive_params)
    if at_most_one > 1:
        raise CatBoostError('Only one of parameters {} should be set'.format(exclusive_params))

    if verbose is None:
        if silent is not None:
            verbose = not silent
        elif verbose_eval is not None:
            verbose = verbose_eval
    if verbose is not None:
        logging_level = 'Verbose' if verbose else 'Silent'
        verbose = int(verbose)

    if isinstance(metric_period, int):
        if metric_period <= 0:
            raise CatBoostError('metric_period should be positive.')
        if verbose is not None:
            if verbose % metric_period != 0:
                raise CatBoostError('verbose should be a multiple of metric_period')

    return (metric_period, verbose, logging_level)


def enum_from_enum_or_str(enum_type, arg):
    if isinstance(arg, enum_type):
        return arg
    elif isinstance(arg, str):
        return enum_type[arg]
    else:
        raise Exception("can't create enum " + str(enum_type) + " from type " + str(type(arg)))


class EFstrType(Enum):
    """Calculate score for every feature by values change."""
    PredictionValuesChange = 0
    """Calculate score for every feature by loss change"""
    LossFunctionChange = 1
    """Use LossFunctionChange for ranking models and PredictionValuesChange otherwise"""
    FeatureImportance = 2
    """Calculate pairwise score between every feature."""
    Interaction = 3
    """Calculate SHAP Values for every object."""
    ShapValues = 4


def _get_cat_features_indices(cat_features, feature_names):
    """
        Parameters
        ----------
        cat_features :
            must be a sequence of either integers or strings
            if it contains strings 'feature_names' parameter must be defined and string ids from 'cat_features'
            must represent a subset of in 'feature_names'

        feature_names :
            A sequence of string ids for features or None.
            Used to get feature indices for string ids in 'cat_features' parameter
    """
    if feature_names is not None:
        return [
            feature_names.index(cf) if isinstance(cf, STRING_TYPES) else cf
            for cf in cat_features
        ]
    else:
        for cf in cat_features:
            if isinstance(cf, STRING_TYPES):
                raise CatBoostError("cat_features parameter contains string value '{}' but feature names "
                                    "for a dataset are not specified".format(cf))
    return cat_features


class Pool(_PoolBase):
    """
    Pool used in CatBoost as a data structure to train model from.
    """

    def __init__(self, data, label=None, cat_features=None, column_description=None, pairs=None, delimiter='\t',
                 has_header=False, weight=None, group_id=None, group_weight=None, subgroup_id=None, pairs_weight=None, baseline=None,
                 feature_names=None, thread_count=-1):
        """
        Pool is an internal data structure that is used by CatBoost.
        You can construct Pool from list, numpy.array, pandas.DataFrame, pandas.Series.

        Parameters
        ----------
        data : list or numpy.array or pandas.DataFrame or pandas.Series or FeaturesData or string
            Data source of Pool.
            If list or numpy.arrays or pandas.DataFrame or pandas.Series, giving 2 dimensional array like data.
            If FeaturesData - see FeaturesData description for details, 'cat_features' and 'feature_names'
              parameters must be equal to None in this case
            If string, giving the path to the file with data in catboost format.

        label : list or numpy.arrays or pandas.DataFrame or pandas.Series, optional (default=None)
            Label of the training data.
            If not None, giving 1 dimensional array like data with floats.

        cat_features : list or numpy.array, optional (default=None)
            If not None, giving the list of Categ features indices or names.
            If it contains feature names, Pool's feature names must be defined: either by passing 'feature_names'
              parameter or if data is pandas.DataFrame (feature names are initialized from it's column names)
            Must be None if 'data' parameter has FeaturesData type

        column_description : string, optional (default=None)
            ColumnsDescription parameter.
            There are several columns description types: Label, Categ, Num, Auxiliary, DocId, Weight, Baseline, GroupId, Timestamp.
            All columns are Num as default, it's not necessary to specify
            this type of columns. Default Label column index is 0 (zero).
            If None, Label column is 0 (zero) as default, all data columns are Num as default.
            If string, giving the path to the file with ColumnsDescription in column_description format.

        pairs : list or numpy.array or pandas.DataFrame or string
            The pairs description.
            If list or numpy.arrays or pandas.DataFrame, giving 2 dimensional.
            The shape should be Nx2, where N is the pairs' count. The first element of the pair is
            the index of winner object in the training set. The second element of the pair is
            the index of loser object in the training set.
            If string, giving the path to the file with pairs description.

        delimiter : string, optional (default='\t')
            Delimiter to use for separate features in file.
            Should be only one symbol, otherwise would be taken only the first character of the string.

        has_header : bool optional (default=False)
            If True, read column names from first line.

        weight : list or numpy.array, optional (default=None)
            Weight for each instance.
            If not None, giving 1 dimensional array like data.

        group_id : list or numpy.array, optional (default=None)
            group id for each instance.
            If not None, giving 1 dimensional array like data.

        group_weight : list or numpy.array, optional (default=None)
            Group weight for each instance.
            If not None, giving 1 dimensional array like data.

        subgroup_id : list or numpy.array, optional (default=None)
            subgroup id for each instance.
            If not None, giving 1 dimensional array like data.

        pairs_weight : list or numpy.array, optional (default=None)
            Weight for each pair.
            If not None, giving 1 dimensional array like pairs.

        baseline : list or numpy.array, optional (default=None)
            Baseline for each instance.
            If not None, giving 2 dimensional array like data.

        feature_names : list, optional (default=None)
            Names for each given data_feature.
              If this parameter is None and 'data' is pandas.DataFrame feature names will be initialized
              from DataFrame's column names.
            Must be None if 'data' parameter has FeaturesData type

        thread_count : int, optional (default=-1)
            Thread count to read data from file.
            Use only with reading data from file.
            If -1, then the number of threads is set to the number of CPU cores.

        """
        if data is not None:
            self._check_data_type(data, cat_features)
            self._check_data_empty(data)
            if pairs is not None and isinstance(data, STRING_TYPES) != isinstance(pairs, STRING_TYPES):
                raise CatBoostError("data and pairs parameters should be the same types.")
            if column_description is not None and not isinstance(data, STRING_TYPES):
                raise CatBoostError("data should be the string type if column_description parameter is specified.")
            if isinstance(data, STRING_TYPES):
                if any(v is not None for v in [cat_features, weight, group_id, group_weight, subgroup_id, pairs_weight, baseline, feature_names]):
                    raise CatBoostError("cat_features, weight, group_id, group_weight, subgroup_id, pairs_weight, \
                                        baseline, feature_names should have the None type when the pool is read from the file.")
                self._read(data, column_description, pairs, delimiter, has_header, thread_count)
            else:
                if isinstance(data, FeaturesData):
                    if any(v is not None for v in [cat_features, feature_names]):
                        raise CatBoostError(
                            "cat_features, feature_names should have the None type when 'data' parameter "
                            " has FeaturesData type"
                        )
                elif isinstance(data, np.ndarray):
                    if (data.dtype == np.float32) and (cat_features is not None) and (len(cat_features) > 0):
                        raise CatBoostError(
                            "'data' is numpy array of np.float32, it means no categorical features,"
                            " but 'cat_features' parameter specifies nonzero number of categorical features"
                        )

                self._init(data, label, cat_features, pairs, weight, group_id, group_weight, subgroup_id, pairs_weight, baseline, feature_names)
        super(Pool, self).__init__()

    def _check_files(self, data, column_description, pairs):
        """
        Check files existence.
        """
        if not os.path.isfile(data):
            raise CatBoostError("Invalid data path='{}': file does not exist.".format(data))
        if column_description is not None and not os.path.isfile(column_description):
            raise CatBoostError("Invalid column_description path='{}': file does not exist.".format(column_description))
        if pairs is not None and not os.path.isfile(pairs):
            raise CatBoostError("Invalid pairs path='{}': file does not exist.".format(pairs))

    def _check_delimiter(self, delimiter):
        if not isinstance(delimiter, STRING_TYPES):
            raise CatBoostError("Invalid delimiter type={} : must be str().".format(type(delimiter)))
        if len(delimiter) < 1:
            raise CatBoostError("Invalid delimiter length={} : must be > 0.".format(len(delimiter)))

    def _check_column_description_type(self, column_description):
        """
        Check type of column_description parameter.
        """
        if not isinstance(column_description, STRING_TYPES):
            raise CatBoostError("Invalid column_description type={}: must be str().".format(type(column_description)))

    def _check_cf_type(self, cat_features):
        """
        Check type of cat_feature parameter.
        """
        if not isinstance(cat_features, (list, np.ndarray)):
            raise CatBoostError("Invalid cat_features type={}: must be list() or np.ndarray().".format(type(cat_features)))

    def _check_cf_value(self, cat_features, features_count):
        """
        Check values in cat_feature parameter. Must be int indices.
        """
        for indx, feature in enumerate(cat_features):
            if not isinstance(feature, INTEGER_TYPES):
                raise CatBoostError("Invalid cat_features[{}] = {} value type={}: must be int().".format(indx, feature, type(feature)))
            if feature >= features_count:
                raise CatBoostError("Invalid cat_features[{}] = {} value: must be < {}.".format(indx, feature, features_count))

    def _check_pairs_type(self, pairs):
        """
        Check type of pairs parameter.
        """
        if not isinstance(pairs, (list, np.ndarray, DataFrame)):
            raise CatBoostError("Invalid pairs type={}: must be list(), np.ndarray() or pd.DataFrame.".format(type(pairs)))

    def _check_pairs_value(self, pairs):
        """
        Check values in pairs parameter. Must be int indices.
        """
        for pair_id, pair in enumerate(pairs):
            if (len(pair) != 2):
                raise CatBoostError("Length of pairs[{}] isn't equal to 2.".format(pair_id))
            for i, index in enumerate(pair):
                if not isinstance(index, INTEGER_TYPES):
                    raise CatBoostError("Invalid pairs[{}][{}] = {} value type={}: must be int().".format(pair_id, i, index, type(index)))

    def _check_data_type(self, data, cat_features):
        """
        Check type of data.
        """
        if not isinstance(data, (STRING_TYPES, ARRAY_TYPES, FeaturesData)):
            raise CatBoostError("Invalid data type={}: data must be list(), np.ndarray(), DataFrame(), Series(), FeaturesData or filename str().".format(type(data)))

    def _check_data_empty(self, data):
        """
        Check that data is not empty (0 objects is ok).
        note: already checked if data is FeatureType, so no need to check again
        """

        if isinstance(data, STRING_TYPES):
            if not data:
                raise CatBoostError("Features filename is empty.")
        elif isinstance(data, ARRAY_TYPES):
            data_shape = np.shape(data)
            if len(data_shape) == 1 and data_shape[0] > 0:
                if isinstance(data[0], Iterable):
                    data_shape = tuple(data_shape + tuple([len(data[0])]))
                else:
                    data_shape = tuple(data_shape + tuple([1]))
            if not len(data_shape) == 2:
                raise CatBoostError("Input data has invalid shape: {}. Must be 2 dimensional".format(data_shape))
            if data_shape[1] == 0:
                raise CatBoostError("Input data must have at least one feature")

    def _check_label_type(self, label):
        """
        Check type of label.
        """
        if not isinstance(label, ARRAY_TYPES):
            raise CatBoostError("Invalid label type={}: must be array like.".format(type(label)))

    def _check_label_empty(self, label):
        """
        Check label is not empty.
        """
        if len(label) == 0:
            raise CatBoostError("Labels variable is empty.")

    def _check_label_shape(self, label, samples_count):
        """
        Check label length and dimension.
        """
        if len(label) != samples_count:
            raise CatBoostError("Length of label={} and length of data={} is different.".format(len(label), samples_count))
        if isinstance(label[0], Iterable) and not isinstance(label[0], STRING_TYPES):
            if len(label[0]) > 1:
                raise CatBoostError("Input label cannot have multiple values per row.")

    def _check_baseline_type(self, baseline):
        """
        Check type of baseline parameter.
        """
        if not isinstance(baseline, ARRAY_TYPES):
            raise CatBoostError("Invalid baseline type={}: must be array like.".format(type(baseline)))

    def _check_baseline_shape(self, baseline, samples_count):
        """
        Check baseline length and dimension.
        """
        if len(baseline) != samples_count:
            raise CatBoostError("Length of baseline={} and length of data={} are different.".format(len(baseline), samples_count))
        if not isinstance(baseline[0], Iterable) or isinstance(baseline[0], STRING_TYPES):
            raise CatBoostError("Baseline must be 2 dimensional data, 1 column for each class.")
        try:
            if np.array(baseline).dtype not in (np.dtype('float'), np.dtype('float32'), np.dtype('int')):
                raise CatBoostError()
        except CatBoostError:
            raise CatBoostError("Invalid baseline value type={}: must be float or int.".format(np.array(baseline).dtype))

    def _check_weight_type(self, weight):
        """
        Check type of weight parameter.
        """
        if not isinstance(weight, ARRAY_TYPES):
            raise CatBoostError("Invalid weight type={}: must be array like.".format(type(weight)))

    def _check_weight_shape(self, weight, samples_count):
        """
        Check weight length.
        """
        if len(weight) != samples_count:
            raise CatBoostError("Length of weight={} and length of data={} are different.".format(len(weight), samples_count))
        if not isinstance(weight[0], (INTEGER_TYPES, FLOAT_TYPES)):
            raise CatBoostError("Invalid weight value type={}: must be 1 dimensional data with int, float or long types.".format(type(weight[0])))

    def _check_group_id_type(self, group_id):
        """
        Check type of group_id parameter.
        """
        if not isinstance(group_id, ARRAY_TYPES):
            raise CatBoostError("Invalid group_id type={}: must be array like.".format(type(group_id)))

    def _check_group_id_shape(self, group_id, samples_count):
        """
        Check group_id length.
        """
        if len(group_id) != samples_count:
            raise CatBoostError("Length of group_id={} and length of data={} are different.".format(len(group_id), samples_count))

    def _check_group_weight_type(self, group_weight):
        """
        Check type of group_weight parameter.
        """
        if not isinstance(group_weight, ARRAY_TYPES):
            raise CatBoostError("Invalid group_weight type={}: must be array like.".format(type(group_weight)))

    def _check_group_weight_shape(self, group_weight, samples_count):
        """
        Check group_weight length.
        """
        if len(group_weight) != samples_count:
            raise CatBoostError("Length of group_weight={} and length of data={} are different.".format(len(group_weight), samples_count))
        if not isinstance(group_weight[0], (FLOAT_TYPES)):
            raise CatBoostError("Invalid group_weight value type={}: must be 1 dimensional data with float types.".format(type(group_weight[0])))

    def _check_subgroup_id_type(self, subgroup_id):
        """
        Check type of subgroup_id parameter.
        """
        if not isinstance(subgroup_id, ARRAY_TYPES):
            raise CatBoostError("Invalid subgroup_id type={}: must be array like.".format(type(subgroup_id)))

    def _check_subgroup_id_shape(self, subgroup_id, samples_count):
        """
        Check subgroup_id length.
        """
        if len(subgroup_id) != samples_count:
            raise CatBoostError("Length of subgroup_id={} and length of data={} are different.".format(len(subgroup_id), samples_count))

    def _check_feature_names(self, feature_names, num_col=None):
        if num_col is None:
            num_col = self.num_col()
        if not isinstance(feature_names, Sequence):
            raise CatBoostError("Invalid feature_names type={} : must be list".format(type(feature_names)))
        if len(feature_names) != num_col:
            raise CatBoostError("Invalid length of feature_names={} : must be equal to the number of columns in data={}".format(len(feature_names), num_col))

    def _check_thread_count(self, thread_count):
        if not isinstance(thread_count, INTEGER_TYPES):
            raise CatBoostError("Invalid thread_count type={} : must be int".format(type(thread_count)))

    def slice(self, rindex):
        if not isinstance(rindex, ARRAY_TYPES):
            raise CatBoostError("Invalid rindex type={} : must be list or numpy.array".format(type(rindex)))
        slicedPool = Pool(None)
        slicedPool._take_slice(self, rindex)
        return slicedPool

    def set_pairs(self, pairs):
        self._check_pairs_type(pairs)
        if isinstance(pairs, DataFrame):
            pairs = pairs.values
        self._check_pairs_value(pairs)
        self._set_pairs(pairs)
        return self

    def set_feature_names(self, feature_names):
        self._check_feature_names(feature_names)
        self._set_feature_names(feature_names)
        return self

    def set_baseline(self, baseline):
        self._check_baseline_type(baseline)
        baseline = self._if_pandas_to_numpy(baseline)
        baseline = np.reshape(baseline, (self.num_row(), -1))
        self._check_baseline_shape(baseline, self.num_row())
        self._set_baseline(baseline)
        return self

    def set_weight(self, weight):
        self._check_weight_type(weight)
        weight = self._if_pandas_to_numpy(weight)
        self._check_weight_shape(weight, self.num_row())
        self._set_weight(weight)
        return self

    def set_group_id(self, group_id):
        self._check_group_id_type(group_id)
        group_id = self._if_pandas_to_numpy(group_id)
        self._check_group_id_shape(group_id, self.num_row())
        self._set_group_id(group_id)
        return self

    def set_group_weight(self, group_weight):
        self._check_group_weight_type(group_weight)
        group_weight = self._if_pandas_to_numpy(group_weight)
        self._check_group_weight_shape(group_weight, self.num_row())
        self._set_group_weight(group_weight)
        return self

    def set_subgroup_id(self, subgroup_id):
        self._check_subgroup_id_type(subgroup_id)
        subgroup_id = self._if_pandas_to_numpy(subgroup_id)
        self._check_subgroup_id_shape(subgroup_id, self.num_row())
        self._set_subgroup_id(subgroup_id)
        return self

    def set_pairs_weight(self, pairs_weight):
        self._check_weight_type(pairs_weight)
        pairs_weight = self._if_pandas_to_numpy(pairs_weight)
        self._check_weight_shape(pairs_weight, self.num_pairs())
        self._set_pairs_weight(pairs_weight)
        return self

    def _if_pandas_to_numpy(self, array):
        if isinstance(array, Series):
            array = array.values
        if isinstance(array, DataFrame):
            array = np.transpose(array.values)[0]
        return array

    def _read(self, pool_file, column_description, pairs, delimiter, has_header, thread_count):
        """
        Read Pool from file.
        """
        with log_fixup():
            self._check_files(pool_file, column_description, pairs)
            self._check_delimiter(delimiter)
            if column_description is None:
                column_description = ''
            else:
                self._check_column_description_type(column_description)
            if pairs is None:
                pairs = ''
            self._check_thread_count(thread_count)
            self._read_pool(pool_file, column_description, pairs, delimiter[0], has_header, thread_count)

    def _init(self, data, label, cat_features, pairs, weight, group_id, group_weight, subgroup_id, pairs_weight, baseline, feature_names):
        """
        Initialize Pool from array like data.
        """
        if isinstance(data, DataFrame):
            if feature_names is None:
                feature_names = list(data.columns)
        if isinstance(data, Series):
            data = data.values.tolist()
        if isinstance(data, FeaturesData):
            samples_count = data.get_object_count()
            features_count = data.get_feature_count()
        else:
            if len(np.shape(data)) == 1:
                data = np.expand_dims(data, 1)
            samples_count, features_count = np.shape(data)
        pairs_len = 0
        if label is not None:
            self._check_label_type(label)
            self._check_label_empty(label)
            label = self._if_pandas_to_numpy(label)
            self._check_label_shape(label, samples_count)
        if feature_names is not None:
            self._check_feature_names(feature_names, features_count)
        if cat_features is not None:
            cat_features = _get_cat_features_indices(cat_features, feature_names)
            self._check_cf_type(cat_features)
            self._check_cf_value(cat_features, features_count)
        if pairs is not None:
            self._check_pairs_type(pairs)
            if isinstance(pairs, DataFrame):
                pairs = pairs.values
            self._check_pairs_value(pairs)
            pairs_len = np.shape(pairs)[0]
        if weight is not None:
            self._check_weight_type(weight)
            weight = self._if_pandas_to_numpy(weight)
            self._check_weight_shape(weight, samples_count)
        if group_id is not None:
            self._check_group_id_type(group_id)
            group_id = self._if_pandas_to_numpy(group_id)
            self._check_group_id_shape(group_id, samples_count)
        if group_weight is not None:
            self._check_group_weight_type(group_weight)
            group_weight = self._if_pandas_to_numpy(group_weight)
            self._check_group_weight_shape(group_weight, samples_count)
        if subgroup_id is not None:
            self._check_subgroup_id_type(subgroup_id)
            subgroup_id = self._if_pandas_to_numpy(subgroup_id)
            self._check_subgroup_id_shape(subgroup_id, samples_count)
        if pairs_weight is not None:
            self._check_weight_type(pairs_weight)
            pairs_weight = self._if_pandas_to_numpy(pairs_weight)
            self._check_weight_shape(pairs_weight, pairs_len)
        if baseline is not None:
            self._check_baseline_type(baseline)
            baseline = self._if_pandas_to_numpy(baseline)
            baseline = np.reshape(baseline, (samples_count, -1))
            self._check_baseline_shape(baseline, samples_count)
        self._init_pool(data, label, cat_features, pairs, weight, group_id, group_weight, subgroup_id, pairs_weight, baseline, feature_names)


def _build_train_pool(X, y, cat_features, pairs, sample_weight, group_id, group_weight, subgroup_id, pairs_weight, baseline, column_description):
    train_pool = None
    if isinstance(X, Pool):
        train_pool = X
        if any(v is not None for v in [cat_features, sample_weight, group_id, group_weight, subgroup_id, pairs_weight, baseline]):
            raise CatBoostError("cat_features, sample_weight, group_id, group_weight, subgroup_id, pairs_weight, baseline should have the None type when X has catboost.Pool type.")
        if X.get_label() is None and X.num_pairs() == 0:
            raise CatBoostError("Label in X has not been initialized.")
        if y is not None:
            raise CatBoostError("Incorrect value of y: X is catboost.Pool object, y must be initialized inside catboost.Pool.")
    elif isinstance(X, STRING_TYPES):
            train_pool = Pool(data=X, pairs=pairs, column_description=column_description)
    else:
        if y is None:
            raise CatBoostError("y has not initialized in fit(): X is not catboost.Pool object, y must be not None in fit().")
        train_pool = Pool(X, y, cat_features=cat_features, pairs=pairs, weight=sample_weight, group_id=group_id,
                          group_weight=group_weight, subgroup_id=subgroup_id, pairs_weight=pairs_weight, baseline=baseline)
    return train_pool


def _clear_training_files(train_dir):
    for filename in ['catboost_training.json']:
        path = os.path.join(train_dir, filename)
        if os.path.exists(path):
            os.remove(path)


def _get_train_dir(params):
    return params.get('train_dir', 'catboost_info')


def _get_catboost_widget(train_dir):
    _clear_training_files(train_dir)
    try:
        from .widget import MetricVisualizer
        return MetricVisualizer(train_dir)
    except ImportError as e:
        warnings.warn("To draw plots in fit() method you should install ipywidgets and ipython")
        raise ImportError(str(e))


@contextmanager
def plot_wrapper(plot, params):
    if plot:
        widget = _get_catboost_widget(_get_train_dir(params))
        widget._run_update()
    try:
        yield
    finally:
        if plot:
            widget._stop_update()


# the first element of the synonyms list is the canonical name
def _process_synonyms_group(synonyms, params):
    assert len(synonyms) > 1, 'there should be more than one synonym'

    value = None
    for synonym in synonyms:
        if synonym in params:
            if value is not None:
                raise CatBoostError('only one of the parameters ' + (', '.join(synonyms)) + ' should be initialized.')
            value = params[synonym]
            del params[synonym]

    if value is not None:
        params[synonyms[0]] = value


def _process_synonyms(params):
    if 'objective' in params:
        params['loss_function'] = params['objective']
        del params['objective']

    if 'scale_pos_weight' in params:
        if 'loss_function' in params and params['loss_function'] != 'Logloss':
                raise CatBoostError('scale_pos_weight is supported only for binary classification Logloss loss')
        if 'class_weights' in params:
            raise CatBoostError('only one of the parameters scale_pos_weight, class_weights should be initialized.')
        params['class_weights'] = [1.0, params['scale_pos_weight']]
        del params['scale_pos_weight']

    _process_synonyms_group(['learning_rate', 'eta'], params)
    _process_synonyms_group(['border_count', 'max_bin'], params)
    _process_synonyms_group(['depth', 'max_depth'], params)
    _process_synonyms_group(['rsm', 'colsample_bylevel'], params)
    _process_synonyms_group(['random_seed', 'random_state'], params)
    _process_synonyms_group(['l2_leaf_reg', 'reg_lambda'], params)
    _process_synonyms_group(['iterations', 'n_estimators', 'num_boost_round', 'num_trees'], params)
    _process_synonyms_group(['od_wait', 'early_stopping_rounds'], params)
    _process_synonyms_group(['custom_metric', 'custom_loss'], params)

    metric_period = None
    if 'metric_period' in params:
        metric_period = params['metric_period']
        del params['metric_period']

    verbose = None
    if 'verbose' in params:
        verbose = params['verbose']
        del params['verbose']

    logging_level = None
    if 'logging_level' in params:
        logging_level = params['logging_level']
        del params['logging_level']

    verbose_eval = None
    if 'verbose_eval' in params:
        verbose_eval = params['verbose_eval']
        del params['verbose_eval']

    silent = None
    if 'silent' in params:
        silent = params['silent']
        del params['silent']

    metric_period, verbose, logging_level = _process_verbose(metric_period, verbose, logging_level, verbose_eval, silent)

    if metric_period is not None:
        params['metric_period'] = metric_period
    if verbose is not None:
        params['verbose'] = verbose
    if logging_level is not None:
        params['logging_level'] = logging_level

    if 'used_ram_limit' in params:
        params['used_ram_limit'] = str(params['used_ram_limit'])


def _get_loss_function(params):
    if params is None:
        return None

    # check 'objective' first because it can be overridden as a CatBoost* param with default 'loss_function'
    objective_param = params.get('objective')
    if objective_param is not None:
        return objective_param

    return params.get('loss_function')


class _CatBoostBase(object):
    def __init__(self, params):
        self._init_params = params.copy() if params is not None else {}
        if 'thread_count' in self._init_params and self._init_params['thread_count'] == -1:
            self._init_params.pop('thread_count')
        self._object = _CatBoost()

    def __getstate__(self):
        params = self._init_params.copy()
        test_evals = self._object._get_test_evals()
        if test_evals:
            params['_test_evals'] = test_evals
        if self.is_fitted():
            params['__model'] = self._serialize_model()
        for attr in ['_classes', '_prediction_values_change', '_loss_value_change']:
            if getattr(self, attr, None) is not None:
                params[attr] = getattr(self, attr, None)
        return params

    def __setstate__(self, state):
        if '_object' not in dict(self.__dict__.items()):
            self._object = _CatBoost()
        if '_init_params' not in dict(self.__dict__.items()):
            self._init_params = {}
        if '__model' in state:
            self._deserialize_model(state['__model'])
            self._set_trained_model_attributes()
            del state['__model']
        if '_test_eval' in state:
            self._set_test_evals([state['_test_eval']])
            del state['_test_eval']
        if '_test_evals' in state:
            self._set_test_evals(state['_test_evals'])
            del state['_test_evals']
        for attr in ['_classes', '_prediction_values_change', '_loss_value_change']:
            if attr in state:
                setattr(self, attr, state[attr])
                del state[attr]
        self._init_params.update(state)

    def __copy__(self):
        return self.__deepcopy__(None)

    def __deepcopy__(self, _):
        state = self.__getstate__()
        model = self.__class__()
        model.__setstate__(state)
        return model

    def __eq__(self, other):
        return self._is_comparable_to(other) and self._object == other._object

    def __neq__(self, other):
        return not self._is_comparable_to(other) or self._object != other._object

    def copy(self):
        return self.__copy__()

    def is_fitted(self):
        return getattr(self, '_random_seed', None) is not None

    def _is_comparable_to(self, rhs):
        if not isinstance(rhs, _CatBoostBase):
            return False
        for side, estimator in [('left', self), ('right', rhs)]:
            if not estimator.is_fitted():
                message = 'The {} argument is not fitted, only fitted models' \
                          ' could be compared.'
                raise CatBoostError(message.format(side))
        return True

    def _set_trained_model_attributes(self):
        setattr(self, '_random_seed', self._object._get_random_seed())
        setattr(self, '_learning_rate', self._object._get_learning_rate())
        setattr(self, '_tree_count', self._object._get_tree_count())

    def _train(self, train_pool, test_pool, params, allow_clear_pool):
        self._object._train(train_pool, test_pool, params, allow_clear_pool)
        self._set_trained_model_attributes()

    def _set_test_evals(self, test_evals):
        self._object._set_test_evals(test_evals)

    def get_test_eval(self):
        test_evals = self._object._get_test_evals()
        if len(test_evals) == 0:
            if self.is_fitted():
                raise CatBoostError('The model has been trained without an eval set.')
            else:
                raise CatBoostError('You should train the model first.')
        if len(test_evals) > 1:
            raise CatBoostError("With multiple eval sets use 'get_test_evals()'")
        test_eval = test_evals[0]
        return test_eval[0] if len(test_eval) == 1 else test_eval

    def get_test_evals(self):
        test_evals = self._object._get_test_evals()
        if len(test_evals) == 0:
            if self.is_fitted():
                raise CatBoostError('The model has been trained without an eval set.')
            else:
                raise CatBoostError('You should train the model first.')
        return test_evals

    def get_evals_result(self):
        return self._object._get_metrics_evals()

    def get_best_score(self):
        return self._object._get_best_score()

    def get_best_iteration(self):
        return self._object._get_best_iteration()

    def _get_float_feature_indices(self):
        return self._object._get_float_feature_indices()

    def _get_cat_feature_indices(self):
        return self._object._get_cat_feature_indices()

    def _base_predict(self, pool, prediction_type, ntree_start, ntree_end, thread_count, verbose):
        return self._object._base_predict(pool, prediction_type, ntree_start, ntree_end, thread_count, verbose)

    def _base_predict_multi(self, pool, prediction_type, ntree_start, ntree_end, thread_count, verbose):
        return self._object._base_predict_multi(pool, prediction_type, ntree_start, ntree_end, thread_count, verbose)

    def _staged_predict_iterator(self, pool, prediction_type, ntree_start, ntree_end, eval_period, thread_count, verbose):
        return self._object._staged_predict_iterator(pool, prediction_type, ntree_start, ntree_end, eval_period, thread_count, verbose)

    def _base_eval_metrics(self, pool, metrics_description, ntree_start, ntree_end, eval_period, thread_count, result_dir, tmp_dir):
        metrics_description_list = metrics_description if isinstance(metrics_description, list) else [metrics_description]
        return self._object._base_eval_metrics(pool, metrics_description_list, ntree_start, ntree_end, eval_period, thread_count, result_dir, tmp_dir)

    def _calc_fstr(self, type, pool, thread_count, verbose, shap_mode):
        """returns (fstr_values, feature_ids)."""
        return self._object._calc_fstr(type.name, pool, thread_count, verbose, shap_mode)

    def _calc_ostr(self, train_pool, test_pool, top_size, ostr_type, update_method, importance_values_sign, thread_count, verbose):
        return self._object._calc_ostr(train_pool, test_pool, top_size, ostr_type, update_method, importance_values_sign, thread_count, verbose)

    def _base_shrink(self, ntree_start, ntree_end):
        self._object._base_shrink(ntree_start, ntree_end)
        self._set_trained_model_attributes()

    def _base_drop_unused_features(self):
        self._object._base_drop_unused_features()

    def _save_model(self, output_file, format, export_parameters, pool):
        import json
        if self.is_fitted():
            params_string = ""
            if export_parameters:
                params_string = json.dumps(export_parameters, cls=_NumpyAwareEncoder)

            self._object._save_model(output_file, format, params_string, pool)

    def _load_model(self, model_file, format):
        self._object._load_model(model_file, format)
        self._set_trained_model_attributes()
        for key, value in iteritems(self._get_params()):
            self._init_params[key] = value

    def _serialize_model(self):
        return self._object._serialize_model()

    def _deserialize_model(self, dump_model_str):
        self._object._deserialize_model(dump_model_str)

    def _sum_models(self, models_base, weights=None, ctr_merge_policy='IntersectingCountersAverage'):
        if weights is None:
            weights = [1.0 for _ in models_base]
        models_inner = [model._object for model in models_base]
        self._object._sum_models(models_inner, weights, ctr_merge_policy)
        setattr(self, '_random_seed', 0)
        setattr(self, '_learning_rate', 0)
        setattr(self, '_tree_count', self._object._get_tree_count())

    def _save_borders(self, output_file):
        if not self.is_fitted():
            raise CatBoostError("There is no trained model to use save_borders(). Use fit() to train model. Then use save_borders().")
        self._object._save_borders(output_file)

    def _get_params(self):
        params = self._object._get_params()
        init_params = self._init_params.copy()
        for key, value in iteritems(init_params):
            if key not in params:
                params[key] = value
        return params

    def _is_classification_objective(self, loss_function):
        return isinstance(loss_function, str) and is_classification_objective(loss_function)

    def _is_regression_objective(self, loss_function):
        return isinstance(loss_function, str) and is_regression_objective(loss_function)

    def get_metadata(self):
        return self._object._get_metadata_wrapper()

    @property
    def tree_count_(self):
        return getattr(self, '_tree_count') if self.is_fitted() else None

    @property
    def random_seed_(self):
        return getattr(self, '_random_seed') if self.is_fitted() else None

    @property
    def learning_rate_(self):
        return getattr(self, '_learning_rate') if self.is_fitted() else None

    @property
    def feature_names_(self):
        return self._object._get_feature_names() if self.is_fitted() else None

    @property
    def classes_(self):
        return self._object._get_class_names() if self.is_fitted() else None

    @property
    def evals_result_(self):
        return self.get_evals_result()

    @property
    def best_score_(self):
        return self.get_best_score()

    @property
    def best_iteration_(self):
        return self.get_best_iteration()


def _check_param_types(params):
    if not isinstance(params, (Mapping, MutableMapping)):
        raise CatBoostError("Invalid params type={}: must be dict().".format(type(params)))
    if 'ctr_description' in params:
        if not isinstance(params['ctr_description'], Sequence):
            raise CatBoostError("Invalid ctr_description type={} : must be list of strings".format(type(params['ctr_description'])))
    if 'ctr_target_border_count' in params:
        if not isinstance(params['ctr_target_border_count'], INTEGER_TYPES):
            raise CatBoostError('Invalid ctr_target_border_count type={} : must be integer type'.format(type(params['ctr_target_border_count'])))
    if 'custom_loss' in params:
        if isinstance(params['custom_loss'], STRING_TYPES):
            params['custom_loss'] = [params['custom_loss']]
        if not isinstance(params['custom_loss'], Sequence):
            raise CatBoostError("Invalid `custom_loss` type={} : must be string or list of strings.".format(type(params['custom_loss'])))
    if 'custom_metric' in params:
        if isinstance(params['custom_metric'], STRING_TYPES):
            params['custom_metric'] = [params['custom_metric']]
        if not isinstance(params['custom_metric'], Sequence):
            raise CatBoostError("Invalid `custom_metric` type={} : must be string or list of strings.".format(type(params['custom_metric'])))


def _params_type_cast(params):
    casted_params = {}
    for key, value in iteritems(params):
        value = _cast_to_base_types(value)
        casted_params[key] = value
    return casted_params


def _is_data_single_object(data):
    if isinstance(data, (Pool, FeaturesData, Series, DataFrame)):
        return False
    if not isinstance(data, ARRAY_TYPES):
        raise CatBoostError(
            "Invalid data type={} : must be list, numpy.ndarray, pandas.Series, pandas.DataFrame,"
            " catboost.FeaturesData or catboost.Pool".format(type(data))
        )
    return len(np.shape(data)) == 1


class CatBoost(_CatBoostBase):
    """
    CatBoost model. Contains training, prediction and evaluation methods.
    """

    def __init__(self, params=None):
        """
        Initialize the CatBoost.

        Parameters
        ----------
        params : dict
            Parameters for CatBoost.
            If  None, all params are set to their defaults.
            If  dict, overriding parameters present in dict.
        """
        super(CatBoost, self).__init__(params)

    def _fit(self, X, y, cat_features, pairs, sample_weight, group_id, group_weight, subgroup_id,
             pairs_weight, baseline, use_best_model, eval_set, verbose, logging_level, plot,
             column_description, verbose_eval, metric_period, silent, early_stopping_rounds,
             save_snapshot, snapshot_file, snapshot_interval):

        params = deepcopy(self._init_params)
        if params is None:
            params = {}

        _process_synonyms(params)

        if 'cat_features' in params:
            if isinstance(X, Pool):
                cat_feature_indices_from_params = _get_cat_features_indices(params['cat_features'], X.get_feature_names())
                if set(X.get_cat_feature_indices()) != set(cat_feature_indices_from_params):
                    raise CatBoostError("categorical features indices in the model are set to "
                                        + str(cat_feature_indices_from_params) +
                                        " and train dataset categorical features indices are set to " +
                                        str(X.get_cat_feature_indices()))
            elif isinstance(X, FeaturesData):
                raise CatBoostError("Categorical features are set in the model. It is not allowed to use FeaturesData type for training dataset.")
            else:
                if cat_features is not None and set(cat_features) != set(params['cat_features']):
                    raise CatBoostError("categorical features in the model are set to " + str(params['cat_features']) +
                                        ". categorical features passed to fit function are set to " + str(cat_features))
                cat_features = params['cat_features']
            del params['cat_features']

        metric_period, verbose, logging_level = _process_verbose(metric_period, verbose, logging_level, verbose_eval, silent)

        if metric_period is not None:
            params['metric_period'] = metric_period
        if logging_level is not None:
            params['logging_level'] = logging_level
        if verbose is not None:
            params['verbose'] = verbose
        if use_best_model is not None:
            params['use_best_model'] = use_best_model

        if early_stopping_rounds is not None:
            params['od_type'] = 'Iter'
            params['od_wait'] = early_stopping_rounds
            if 'od_pval' in params:
                del params['od_pval']

        if save_snapshot is not None:
            params['save_snapshot'] = save_snapshot

        if snapshot_file is not None:
            params['snapshot_file'] = snapshot_file

        if snapshot_interval is not None:
            params['snapshot_interval'] = snapshot_interval

        _check_param_types(params)
        params = _params_type_cast(params)
        _check_train_params(params)

        train_pool = _build_train_pool(X, y, cat_features, pairs, sample_weight, group_id, group_weight, subgroup_id, pairs_weight, baseline, column_description)
        if train_pool.is_empty_:
            raise CatBoostError("X is empty.")

        allow_clear_pool = not isinstance(X, Pool)

        eval_set_list = eval_set if isinstance(eval_set, list) else [eval_set]
        eval_sets = []
        eval_total_row_count = 0
        for eval_set in eval_set_list:
            if isinstance(eval_set, Pool):
                eval_sets.append(eval_set)
                eval_total_row_count += eval_sets[-1].num_row()
                if eval_sets[-1].num_row() == 0:
                    raise CatBoostError("Empty 'eval_set' in Pool")
            elif isinstance(eval_set, STRING_TYPES):
                eval_sets.append(Pool(eval_set, column_description=column_description))
                eval_total_row_count += eval_sets[-1].num_row()
                if eval_sets[-1].num_row() == 0:
                    raise CatBoostError("Empty 'eval_set' in file {}".format(eval_set))
            elif isinstance(eval_set, tuple):
                if len(eval_set) != 2:
                    raise CatBoostError("Invalid shape of 'eval_set': {}, must be (X, y).".format(str(tuple(type(_) for _ in eval_set))))
                eval_sets.append(Pool(eval_set[0], eval_set[1], cat_features=train_pool.get_cat_feature_indices()))
                eval_total_row_count += eval_sets[-1].num_row()
                if eval_sets[-1].num_row() == 0:
                    raise CatBoostError("Empty 'eval_set' in tuple")
            elif eval_set is None:
                if len(eval_set_list) > 1:
                    raise CatBoostError("Multiple eval set shall not contain None")
            else:
                raise CatBoostError("Invalid type of 'eval_set': {}, while expected Pool or (X, y) or filename, or list thereof.".format(type(eval_set)))

        if self.get_param('use_best_model') and eval_total_row_count == 0:
            raise CatBoostError("To employ param {'use_best_model': True} provide non-empty 'eval_set'.")

        with log_fixup(), plot_wrapper(plot, self.get_params()):
            self._train(train_pool, eval_sets, params, allow_clear_pool)

        if (not self._object._has_leaf_weights_in_model()) and allow_clear_pool:
            train_pool = _build_train_pool(X, y, cat_features, pairs, sample_weight, group_id, group_weight, subgroup_id, pairs_weight, baseline, column_description)
        if self._object._is_oblivious() and not self._object._is_groupwise_learned_model():
            self.get_feature_importance(type=EFstrType.PredictionValuesChange)

        if 'loss_function' in params and self._is_classification_objective(params['loss_function']):
            setattr(self, "_classes", np.unique(train_pool.get_label()))
        return self

    def fit(self, X, y=None, cat_features=None, pairs=None, sample_weight=None, group_id=None,
            group_weight=None, subgroup_id=None, pairs_weight=None, baseline=None, use_best_model=None,
            eval_set=None, verbose=None, logging_level=None, plot=False, column_description=None,
            verbose_eval=None, metric_period=None, silent=None, early_stopping_rounds=None,
            save_snapshot=None, snapshot_file=None, snapshot_interval=None):
        """
        Fit the CatBoost model.

        Parameters
        ----------
        X : catboost.Pool or list or numpy.array or pandas.DataFrame or pandas.Series or catboost.FeaturesData
             or string.
            If not catboost.Pool or catboost.FeaturesData it must be 2 dimensional Feature matrix
             or string - file with dataset.

             Must be non-empty (contain > 0 objects)

        y : list or numpy.array or pandas.DataFrame or pandas.Series, optional (default=None)
            Labels, 1 dimensional array like.
            Use only if X is not catboost.Pool.

        cat_features : list or numpy.array, optional (default=None)
            If not None, giving the list of Categ columns indices.
            Use only if X is not catboost.Pool and not catboost.FeaturesData

        pairs : list or numpy.array or pandas.DataFrame
            The pairs description.
            If list or numpy.arrays or pandas.DataFrame, giving 2 dimensional.
            The shape should be Nx2, where N is the pairs' count. The first element of the pair is
            the index of the winner object in the training set. The second element of the pair is
            the index of the loser object in the training set.

        sample_weight : list or numpy.array or pandas.DataFrame or pandas.Series, optional (default=None)
            Instance weights, 1 dimensional array like.

        group_id : list or numpy.array, optional (default=None)
            group id for each instance.
            If not None, giving 1 dimensional array like data.
            Use only if X is not catboost.Pool.

        group_weight : list or numpy.array, optional (default=None)
            Group weight for each instance.
            If not None, giving 1 dimensional array like data.

        subgroup_id : list or numpy.array, optional (default=None)
            subgroup id for each instance.
            If not None, giving 1 dimensional array like data.
            Use only if X is not catboost.Pool.

        pairs_weight : list or numpy.array, optional (default=None)
            Weight for each pair.
            If not None, giving 1 dimensional array like pairs.

        baseline : list or numpy.array, optional (default=None)
            If not None, giving 2 dimensional array like data.
            Use only if X is not catboost.Pool.

        use_best_model : bool, optional (default=None)
            Flag to use best model

        eval_set : catboost.Pool, or list of catboost.Pool, or list of (X, y) tuples, optional (default=None)
            Used as a validation set for early-stopping.

        logging_level : string, optional (default=None)
            Possible values:
                - 'Silent'
                - 'Verbose'
                - 'Info'
                - 'Debug'

        metric_period : int
            Frequency of evaluating metrics.

        verbose : bool or int
            If verbose is bool, then if set to True, logging_level is set to Verbose,
            if set to False, logging_level is set to Silent.
            If verbose is int, it determines the frequency of writing metrics to output and
            logging_level is set to Verbose.

        silent : bool
            If silent is True, logging_level is set to Silent.
            If silent is False, logging_level is set to Verbose.

        verbose_eval : bool or int
            Synonym for verbose. Only one of these parameters should be set.

        plot : bool, optional (default=False)
            If True, draw train and eval error in Jupyter notebook

        early_stopping_rounds : int
            Activates Iter overfitting detector with od_wait parameter set to early_stopping_rounds.

        save_snapshot : bool, [default=None]
            Enable progress snapshotting for restoring progress after crashes or interruptions

        snapshot_file : string, [default=None]
            Learn progress snapshot file path, if None will use default filename

        snapshot_interval: int, [default=600]
            Interval between saving snapshots (seconds)

        Returns
        -------
        model : CatBoost
        """
        return self._fit(X, y, cat_features, pairs, sample_weight, group_id, group_weight, subgroup_id,
                         pairs_weight, baseline, use_best_model, eval_set, verbose, logging_level, plot,
                         column_description, verbose_eval, metric_period, silent, early_stopping_rounds,
                         save_snapshot, snapshot_file, snapshot_interval)

    def _predict(self, data, prediction_type, ntree_start, ntree_end, thread_count, verbose, parent_method_name):
        verbose = verbose or self.get_param('verbose')
        if verbose is None:
            verbose = False
        if not self.is_fitted():
            raise CatBoostError("There is no trained model to use {}(). Use fit() to train model. Then use this method.".format(parent_method_name))

        data_is_single_object = _is_data_single_object(data)
        if not isinstance(data, Pool):
            data = Pool(
                data=[data] if data_is_single_object else data,
                cat_features=self._get_cat_feature_indices() if not isinstance(data, FeaturesData) else None
            )
        if not isinstance(prediction_type, STRING_TYPES):
            raise CatBoostError("Invalid prediction_type type={}: must be str().".format(type(prediction_type)))
        if prediction_type not in ('Class', 'RawFormulaVal', 'Probability'):
            raise CatBoostError("Invalid value of prediction_type={}: must be Class, RawFormulaVal or Probability.".format(prediction_type))

        loss_function_type = _get_loss_function(self._get_params())

        # TODO(kirillovs): very bad solution. user should be able to use custom multiclass losses
        if loss_function_type is not None and (loss_function_type == 'MultiClass' or loss_function_type == 'MultiClassOneVsAll'):
            return np.transpose(self._base_predict_multi(data, prediction_type, ntree_start, ntree_end, thread_count, verbose))
        predictions = np.array(self._base_predict(data, prediction_type, ntree_start, ntree_end, thread_count, verbose))
        if prediction_type == 'Probability':
            predictions = np.transpose([1 - predictions, predictions])
        return predictions[0] if data_is_single_object else predictions

    def predict(self, data, prediction_type='RawFormulaVal', ntree_start=0, ntree_end=0, thread_count=-1, verbose=None):
        """
        Predict with data.

        Parameters
        ----------
        data : catboost.Pool or list of features or list of lists or numpy.array or pandas.DataFrame or pandas.Series
                or catboost.FeaturesData
            Data to apply model on.
            If data is a simple list (not list of lists) or a one-dimensional numpy.ndarray it is interpreted
            as a list of features for a single object.

        prediction_type : string, optional (default='RawFormulaVal')
            Can be:
            - 'RawFormulaVal' : return raw value.
            - 'Class' : return majority vote class.
            - 'Probability' : return probability for every class.

        ntree_start: int, optional (default=0)
            Model is applied on the interval [ntree_start, ntree_end) (zero-based indexing).

        ntree_end: int, optional (default=0)
            Model is applied on the interval [ntree_start, ntree_end) (zero-based indexing).
            If value equals to 0 this parameter is ignored and ntree_end equal to tree_count_.

        thread_count : int (default=-1)
            The number of threads to use when applying the model.
            Allows you to optimize the speed of execution. This parameter doesn't affect results.
            If -1, then the number of threads is set to the number of CPU cores.

        verbose : bool, optional (default=False)
            If True, writes the evaluation metric measured set to stderr.

        Returns
        -------
        prediction :
            If data is for a single object, the return value depends on prediction_type value:
                - 'RawFormulaVal' : return raw formula value.
                - 'Class' : return majority vote class.
                - 'Probability' : return one-dimensional numpy.ndarray with probability for every class.
            otherwise numpy.ndarray, with values that depend on prediction_type value:
                - 'RawFormulaVal' : one-dimensional array of raw formula value for each object.
                - 'Class' : one-dimensional array of majority vote class for each object.
                - 'Probability' : two-dimensional numpy.ndarray with shape (number_of_objects x number_of_classes)
                  with probability for every class for each object.
        """
        return self._predict(data, prediction_type, ntree_start, ntree_end, thread_count, verbose, 'predict')

    def _staged_predict(self, data, prediction_type, ntree_start, ntree_end, eval_period, thread_count, verbose, parent_method_name):
        verbose = verbose or self.get_param('verbose')
        if verbose is None:
            verbose = False
        if not self.is_fitted() or self.tree_count_ is None:
            raise CatBoostError("There is no trained model to use {}(). Use fit() to train model. Then use this method.".format(parent_method_name))

        data_is_single_object = _is_data_single_object(data)
        if not isinstance(data, Pool):
            data = Pool(
                data=[data] if data_is_single_object else data,
                cat_features=self._get_cat_feature_indices() if not isinstance(data, FeaturesData) else None
            )
        if not isinstance(prediction_type, STRING_TYPES):
            raise CatBoostError("Invalid prediction_type type={}: must be str().".format(type(prediction_type)))
        if prediction_type not in ('Class', 'RawFormulaVal', 'Probability'):
            raise CatBoostError("Invalid value of prediction_type={}: must be Class, RawFormulaVal or Probability.".format(prediction_type))
        if ntree_end == 0:
            ntree_end = self.tree_count_
        staged_predict_iterator = self._staged_predict_iterator(data, prediction_type, ntree_start, ntree_end, eval_period, thread_count, verbose)
        loss_function = _get_loss_function(self._get_params())
        while True:
            predictions = staged_predict_iterator.next()
            if loss_function is not None and (loss_function == 'MultiClass' or loss_function == 'MultiClassOneVsAll'):
                predictions = np.transpose(predictions)
            else:
                predictions = np.array(predictions[0])
                if prediction_type == 'Probability':
                    predictions = np.transpose([1 - predictions, predictions])
            yield predictions[0] if data_is_single_object else predictions

    def staged_predict(self, data, prediction_type='RawFormulaVal', ntree_start=0, ntree_end=0, eval_period=1, thread_count=-1, verbose=None):
        """
        Predict target at each stage for data.

        Parameters
        ----------
        data : catboost.Pool or list of features or list of lists or numpy.array or pandas.DataFrame or pandas.Series
                or catboost.FeaturesData
            Data to apply model on.
            If data is a simple list (not list of lists) or a one-dimensional numpy.ndarray it is interpreted
            as a list of features for a single object.

        prediction_type : string, optional (default='RawFormulaVal')
            Can be:
            - 'RawFormulaVal' : return raw formula value.
            - 'Class' : return majority vote class.
            - 'Probability' : return probability for every class.

        ntree_start: int, optional (default=0)
            Model is applied on the interval [ntree_start, ntree_end) with the step eval_period (zero-based indexing).

        ntree_end: int, optional (default=0)
            Model is applied on the interval [ntree_start, ntree_end) with the step eval_period (zero-based indexing).
            If value equals to 0 this parameter is ignored and ntree_end equal to tree_count_.

        eval_period: int, optional (default=1)
            Model is applied on the interval [ntree_start, ntree_end) with the step eval_period (zero-based indexing).

        thread_count : int (default=-1)
            The number of threads to use when applying the model.
            Allows you to optimize the speed of execution. This parameter doesn't affect results.
            If -1, then the number of threads is set to the number of CPU cores.

        verbose : bool
            If True, writes the evaluation metric measured set to stderr.

        Returns
        -------
        prediction : generator for each iteration that generates:
            If data is for a single object, the return value depends on prediction_type value:
                - 'RawFormulaVal' : return raw formula value.
                - 'Class' : return majority vote class.
                - 'Probability' : return one-dimensional numpy.ndarray with probability for every class.
            otherwise numpy.ndarray, with values that depend on prediction_type value:
                - 'RawFormulaVal' : one-dimensional array of raw formula value for each object.
                - 'Class' : one-dimensional array of majority vote classe for each object.
                - 'Probability' : two-dimensional numpy.ndarray with shape (number_of_objects x number_of_classes)
                  with probability for every class for each object.
        """
        return self._staged_predict(data, prediction_type, ntree_start, ntree_end, eval_period, thread_count, verbose, 'staged_predict')

    def get_cat_feature_indices(self):
        if not self.is_fitted():
            raise CatBoostError("Model is not fitted")
        return self._get_cat_feature_indices()

    def _eval_metrics(self, data, metrics, ntree_start, ntree_end, eval_period, thread_count, tmp_dir, plot):
        if not self.is_fitted():
            raise CatBoostError("There is no trained model to evaluate metrics on. Use fit() to train model. Then call this method.")
        if not isinstance(data, Pool):
            raise CatBoostError("Invalid data type={}, must be catboost.Pool.".format(type(data)))
        if data.is_empty_:
            raise CatBoostError("Data is empty.")
        if not isinstance(metrics, ARRAY_TYPES) and not isinstance(metrics, STRING_TYPES):
            raise CatBoostError("Invalid metrics type={}, must be list() or str().".format(type(metrics)))
        if not all(map(lambda metric: isinstance(metric, string_types), metrics)):
            raise CatBoostError("Invalid metric type: must be string().")
        if tmp_dir is None:
            tmp_dir = tempfile.mkdtemp()

        with log_fixup(), plot_wrapper(plot, self.get_params()):
            metrics_score, metric_names = self._base_eval_metrics(data, metrics, ntree_start, ntree_end, eval_period, thread_count, _get_train_dir(self.get_params()), tmp_dir)

        return dict(zip(metric_names, metrics_score))

    def eval_metrics(self, data, metrics, ntree_start=0, ntree_end=0, eval_period=1, thread_count=-1, tmp_dir=None, plot=False):
        """
        Calculate metrics.

        Parameters
        ----------
        data : catboost.Pool
            Data to evaluate metrics on.

        metrics : list of strings
            List of evaluated metrics.

        ntree_start: int, optional (default=0)
            Model is applied on the interval [ntree_start, ntree_end) (zero-based indexing).

        ntree_end: int, optional (default=0)
            Model is applied on the interval [ntree_start, ntree_end) (zero-based indexing).
            If value equals to 0 this parameter is ignored and ntree_end equal to tree_count_.

        eval_period: int, optional (default=1)
            Model is applied on the interval [ntree_start, ntree_end) with the step eval_period (zero-based indexing).

        thread_count : int (default=-1)
            The number of threads to use when applying the model.
            Allows you to optimize the speed of execution. This parameter doesn't affect results.
            If -1, then the number of threads is set to the number of CPU cores.

        tmp_dir : string (default=None)
            The name of the temporary directory for intermediate results.
            If None, then the name will be generated.

        plot : bool, optional (default=False)
            If True, draw train and eval error in Jupyter notebook

        Returns
        -------
        prediction : dict: metric -> array of shape [(ntree_end - ntree_start) / eval_period]
        """
        return self._eval_metrics(data, metrics, ntree_start, ntree_end, eval_period, thread_count, tmp_dir, plot)

    def compare(self, second_model, data=None, metrics=None, ntree_start=0, ntree_end=0, eval_period=1, thread_count=-1, tmp_dir=None):
        """
        Draw train and eval errors in Jupyter notebook for both models

        Parameters
        ----------
        second_model: CatBoost model
            Another model to draw metrics

        data : catboost.Pool
            Data to evaluate metrics on.

        metrics : list of strings
            List of evaluated metrics.

        ntree_start: int, optional (default=0)
            Model is applied on the interval [ntree_start, ntree_end) (zero-based indexing).

        ntree_end: int, optional (default=0)
            Model is applied on the interval [ntree_start, ntree_end) (zero-based indexing).
            If value equals to 0 this parameter is ignored and ntree_end equal to tree_count_.

        eval_period: int, optional (default=1)
            Model is applied on the interval [ntree_start, ntree_end) with the step eval_period (zero-based indexing).

        thread_count : int (default=-1)
            The number of threads to use when applying the model.
            Allows you to optimize the speed of execution. This parameter doesn't affect results.
            If -1, then the number of threads is set to the number of CPU cores.

        tmp_dir : string (default=None)
            The name of the temporary directory for intermediate results.
            If None, then the name will be generated.

        plot : bool, optional (default=False)
            If True, draw train and eval error in Jupyter notebook
        """
        assert self is not second_model, "The models should be different"
        assert bool(metrics) == bool(data), "If you provide data, you should also provide metrics list"

        train_dir_first = _get_train_dir(self.get_params())
        train_dir_second = _get_train_dir(second_model.get_params())

        assert train_dir_first != train_dir_second, "Models' train dirs should be different"

        try:
            from .widget import MetricVisualizer
            widget = MetricVisualizer([train_dir_first, train_dir_second])
            widget._run_update()
            if data:
                _clear_training_files(train_dir_first)
                _clear_training_files(train_dir_second)
                self._eval_metrics(data, metrics, ntree_start, ntree_end, eval_period, thread_count, tmp_dir, plot=False)
                second_model._eval_metrics(data, metrics, ntree_start, ntree_end, eval_period, thread_count, tmp_dir, plot=False)
            widget._stop_update()
        except ImportError as e:
            warnings.warn("To draw plots in fit() method you should install ipywidgets and ipython")
            raise ImportError(str(e))

    def create_metric_calcer(self, metrics, ntree_start=0, ntree_end=0, eval_period=1, thread_count=-1, tmp_dir=None):
        """
        Create batch metric calcer. Could be used to aggregate metric on several pools
        Parameters
        ----------
            Same as in eval_metrics except data
        Returns
        -------
            BatchMetricCalcer object

        Usage example
        -------
        # Large dataset is partitioned into parts [part1, part2]
        model.fit(params)
        batch_calcer = model.create_metric_calcer(['Logloss'])
        batch_calcer.add_pool(part1)
        batch_calcer.add_pool(part2)
        metrics = batch_calcer.eval_metrics()
        """
        if not self.is_fitted():
            raise CatBoostError("There is no trained model to evaluate metrics on. Use fit() to train model. Then call this method.")
        return BatchMetricCalcer(self._object, metrics, ntree_start, ntree_end, eval_period, thread_count, tmp_dir)

    @property
    def feature_importances_(self):
        if self._object._is_groupwise_learned_model():
            return np.array(getattr(self, "_loss_function_change", None))
        else:
            return np.array(getattr(self, "_prediction_values_change", None))

    def get_feature_importance(self, data=None, type=EFstrType.FeatureImportance, prettified=False, thread_count=-1, verbose=False, fstr_type=None, shap_mode="Auto"):
        """
        Parameters
        ----------
        data : catboost.Pool or None
            Data to get feature importance.
            If type in ('Shap', 'PredictionValuesChange') data is a dataset. For every object in this dataset feature importances will be calculated.
            If type == 'PredictionValuesChange', data is None or train dataset (in case if model was explicitly trained with flag store no leaf weights).

        type : EFstrType or string (converted to EFstrType), optional
                    (default=EFstrType.FeatureImportance)
            Possible values:
                - PredictionValuesChange
                    Calculate score for every feature.
                - LossFunctionChange
                    Calculate score for every feature by loss.
                - FeatureImportance
                    PredictionValuesChange for non-ranking metrics and LossFunctionChange for ranking metrics
                - ShapValues
                    Calculate SHAP Values for every object.
                - Interaction
                    Calculate pairwise score between every feature.

        prettified : bool, optional (default=False)
            used only for PredictionValuesChange type
            change returned data format to the list of (feature_id, importance) pairs sorted by importance

        thread_count : int, optional (default=-1)
            Number of threads.
            If -1, then the number of threads is set to the number of CPU cores.

        verbose : bool or int
            If False, then evaluation is not logged. If True, then each possible iteration is logged.
            If a positive integer, then it stands for the size of batch N. After processing each batch, print progress
            and remaining time.

        fstr_type : string, deprecated, use type instead

        shap_mode : string, optional (default="Auto")
            used only for ShapValues type
            Possible values:
                - "Auto"
                    Use direct SHAP Values calculation only if data size is smaller than average leaves number
                    (the best of two strategies below is chosen).
                - "UsePreCalc"
                    Calculate SHAP Values for every leaf in preprocessing. Final complexity is
                    O(NT(D+F))+O(TL^2 D^2) where N is the number of documents(objects), T - number of trees,
                    D - average tree depth, F - average number of features in tree, L - average number of leaves in tree
                    This is much faster (because of a smaller constant) than direct calculation when N >> L
                - "NoPreCalc"
                    Use direct SHAP Values calculation calculation with complexity O(NTLD^2). Direct algorithm
                    is faster when N < L (algorithm from https://arxiv.org/abs/1802.03888)
        Returns
        -------
        depends on type:
            - PredictionValuesChange, LossFunctionChange with prettified=False (default)
                list of length [n_features] with feature_importance values (float) for feature
            - PredictionValuesChange, LossFunctionChange with prettified=True
                list of length [n_features] with (feature_id (string), feature_importance (float)) pairs, sorted by feature_importance in descending order
            - ShapValues
                np.array of shape (n_objects, n_features + 1) with Shap values (float) for (object, feature).
                In case of multiclass the returned value is np.array of shape
                (n_objects, classes_count, n_features + 1). For each object it contains Shap values (float).
                Values are calculated for RawFormulaVal predictions.
            - Interaction
                list of length [n_features] of 3-element lists of (first_feature_index, second_feature_index, interaction_score (float))
        """
        if self.is_fitted() and not self._object._is_oblivious():
            raise CatBoostError('Feature importance is not supported for non symmetric trees')

        if not isinstance(verbose, bool) and not isinstance(verbose, int):
            raise CatBoostError('verbose should be bool or int.')
        verbose = int(verbose)
        if verbose < 0:
            raise CatBoostError('verbose should be non-negative.')

        if fstr_type is not None:
            type = fstr_type
            warnings.warn("'fstr_type' parameter will be deprecated soon, use 'type' parameter instead")

        type = enum_from_enum_or_str(EFstrType, type)
        if type == EFstrType.FeatureImportance:
            if self._object._is_groupwise_learned_model():
                type = EFstrType.LossFunctionChange
            else:
                type = EFstrType.PredictionValuesChange

        if data is not None and not isinstance(data, Pool):
            from __builtin__ import type as typeof
            raise CatBoostError("Invalid data type={}, must be catboost.Pool.".format(typeof(data)))

        need_meta_info = type == EFstrType.PredictionValuesChange
        empty_data_is_ok = need_meta_info and self._object._has_leaf_weights_in_model() or type == EFstrType.Interaction
        if not empty_data_is_ok:
            if data is None:
                if need_meta_info:
                    raise CatBoostError(
                        "Model has no meta information needed to calculate feature importances. \
                        Pass training dataset to this function.")
                else:
                    raise CatBoostError(
                        "Feature importance type {} requires training dataset \
                        to be passed to this function.".format(type))
            if data.is_empty_:
                raise CatBoostError("data is empty.")

        with log_fixup():
            fstr, feature_names = self._calc_fstr(type, data, thread_count, verbose, shap_mode)
        if type in (EFstrType.PredictionValuesChange, EFstrType.LossFunctionChange):
            feature_importances = [value[0] for value in fstr]
            if prettified:
                feature_importances = sorted(zip(feature_names, feature_importances), key=itemgetter(1), reverse=True)
            attribute_name = "_prediction_values_change" if type == EFstrType.PredictionValuesChange else "_loss_value_change"
            setattr(
                self,
                attribute_name,
                feature_importances
            )
            return feature_importances
        if type == EFstrType.ShapValues:
            if isinstance(fstr[0][0], ARRAY_TYPES):
                return np.array([np.array([np.array([
                    value for value in dimension]) for dimension in doc]) for doc in fstr])
            else:
                return np.array([np.array([value for value in doc]) for doc in fstr])
        elif type == EFstrType.Interaction:
            return [[int(row[0]), int(row[1]), row[2]] for row in fstr]

    def get_object_importance(self, pool, train_pool, top_size=-1, ostr_type='Average', update_method='SinglePoint', importance_values_sign='All', thread_count=-1, verbose=False):
        """
        This is the implementation of the LeafInfluence algorithm from the following paper:
        https://arxiv.org/pdf/1802.06640.pdf

        Parameters
        ----------
        pool : Pool
            The pool for which you want to evaluate the object importances.

        train_pool : Pool
            The pool on which the model has been trained.

        top_size : int (default=-1)
            Method returns the result of the top_size most important train objects.
            If -1, then the top size is not limited.

        ostr_type : string, optional (default='Average')
            Possible values:
                - Average (Method returns the mean train objects scores for all input objects)
                - PerObject (Method returns the train objects scores for every input object)

        importance_values_sign : string, optional (default='All')
            Method returns only Positive, Negative or All values.
            Possible values:
                - Positive
                - Negative
                - All

        update_method : string, optional (default='SinglePoint')
            Possible values:
                - SinglePoint
                - TopKLeaves (It is posible to set top size : TopKLeaves:top=2)
                - AllPoints
            Description of the update set methods are given in section 3.1.3 of the paper.

        thread_count : int, optional (default=-1)
            Number of threads.
            If -1, then the number of threads is set to the number of CPU cores.

        verbose : bool or int
            If False, then evaluation is not logged. If True, then each possible iteration is logged.
            If a positive integer, then it stands for the size of batch N. After processing each batch, print progress
            and remaining time.

        Returns
        -------
        object_importances : tuple of two arrays (indices and scores) of shape = [top_size]
        """

        if self.is_fitted() and not self._object._is_oblivious():
            raise CatBoostError('Object importance is not supported for non symmetric trees')

        if not isinstance(verbose, bool) and not isinstance(verbose, int):
            raise CatBoostError('verbose should be bool or int.')
        verbose = int(verbose)
        if verbose < 0:
            raise CatBoostError('verbose should be non-negative.')

        with log_fixup():
            result = self._calc_ostr(train_pool, pool, top_size, ostr_type, update_method, importance_values_sign, thread_count, verbose)
        return result

    def shrink(self, ntree_end, ntree_start=0):
        """
        Shrink the model.

        Parameters
        ----------
        ntree_end: int
            Leave the trees with indices from the interval [ntree_start, ntree_end) (zero-based indexing).
        ntree_start: int, optional (default=0)
            Leave the trees with indices from the interval [ntree_start, ntree_end) (zero-based indexing).
        """
        if ntree_start > ntree_end:
            raise CatBoostError("ntree_start should be less than ntree_end.")
        self._base_shrink(ntree_start, ntree_end)

    def drop_unused_features(self):
        """
        Drop unused features information from model
        """
        self._base_drop_unused_features()

    def save_model(self, fname, format="cbm", export_parameters=None, pool=None):
        """
        Save the model to a file.

        Parameters
        ----------
        fname : string
            Output file name.
        format : string
            Possible values:
                * 'cbm' for catboost binary format,
                * 'coreml' to export into Apple CoreML format
                * 'onnx' to export into ONNX-ML format
                * 'cpp' to export as C++ code
                * 'python' to export as Python code.
        export_parameters : dict
            Parameters for CoreML export:
                * prediction_type : string - either 'probability' or 'raw'
                * coreml_description : string
                * coreml_model_version : string
                * coreml_model_author : string
                * coreml_model_license: string
        pool : catboost.Pool or list or numpy.array or pandas.DataFrame or pandas.Series or catboost.FeaturesData
            Training pool.
        """
        if not self.is_fitted():
            raise CatBoostError("There is no trained model to use save_model(). Use fit() to train model. Then use this method.")
        if not isinstance(fname, STRING_TYPES):
            raise CatBoostError("Invalid fname type={}: must be str().".format(type(fname)))
        if pool is not None and not isinstance(pool, Pool):
            pool = Pool(
                data=pool,
                cat_features=self._get_cat_feature_indices() if not isinstance(pool, FeaturesData) else None
            )
        self._save_model(fname, format, export_parameters, pool)

    def load_model(self, fname, format='catboost'):
        """
        Load model from a file.

        Parameters
        ----------
        fname : string
            Input file name.
        """
        if not isinstance(fname, STRING_TYPES):
            raise CatBoostError("Invalid fname type={}: must be str().".format(type(fname)))
        self._load_model(fname, format)
        return self

    def get_param(self, key):
        """
        Get param value from CatBoost model.

        Parameters
        ----------
        key : string
            The key to get param value from.

        Returns
        -------
        value :
            The param value of the key, returns None if param do not exist.
        """
        params = self.get_params()
        if params is None:
            return {}
        return params.get(key)

    def get_params(self, deep=True):
        """
        Get all params from CatBoost model.

        Returns
        -------
        result : dict
            Dictionary of {param_key: param_value}.
        """
        params = self._init_params.copy()
        if deep:
            return deepcopy(params)
        else:
            return params

    def save_borders(self, fname):
        """
        Save the model borders to a file.

        Parameters
        ----------
        fname : string
            Output file name.
        """
        if not isinstance(fname, STRING_TYPES):
            raise CatBoostError("Invalid fname type={}: must be str().".format(type(fname)))
        self._save_borders(fname)

    def set_params(self, **params):
        """
        Set parameters into CatBoost model.

        Parameters
        ----------
        **params : key=value format
            List of key=value paris. Example: model.set_params(iterations=500, thread_count=2).
        """
        for key, value in iteritems(params):
            self._init_params[key] = value
        return self


class CatBoostClassifier(CatBoost):

    _estimator_type = 'classifier'

    """
    Implementation of the scikit-learn API for CatBoost classification.

    Parameters
    ----------
    iterations : int, [default=500]
        Max count of trees.
        range: [1,+inf]
    learning_rate : float, [default value is selected automatically for binary classification with other parameters set to default. In all other cases default is 0.03]
        Step size shrinkage used in update to prevents overfitting.
        range: (0,1]
    depth : int, [default=6]
        Depth of a tree. All trees are the same depth.
        range: [1,+inf]
    l2_leaf_reg : float, [default=3.0]
        Coefficient at the L2 regularization term of the cost function.
        range: [0,+inf]
    model_size_reg : float, [default=None]
        Model size regularization coefficient.
        range: [0,+inf]
    rsm : float, [default=None]
        Subsample ratio of columns when constructing each tree.
        range: (0,1]
    loss_function : string or object, [default='Logloss']
        The metric to use in training and also selector of the machine learning
        problem to solve. If string, then the name of a supported metric,
        optionally suffixed with parameter description.
        If object, it shall provide methods 'calc_ders_range' or 'calc_ders_multi'.
    border_count : int, [default = 254 for training on CPU or 128 for training on GPU]
        The number of partitions in numeric features binarization. Used in the preliminary calculation.
        range: (0,+inf]
    feature_border_type : string, [default='GreedyLogSum']
        The binarization mode in numeric features binarization. Used in the preliminary calculation.
        Possible values:
            - 'Median'
            - 'Uniform'
            - 'UniformAndQuantiles'
            - 'GreedyLogSum'
            - 'MaxLogSum'
            - 'MinEntropy'
    input_borders : string, [default=None]
        input file with borders used in numeric features binarization.
    output_borders : string, [default=None]
        output file for borders that were used in numeric features binarization.
    fold_permutation_block : int, [default=1]
        To accelerate the learning.
        The recommended value is within [1, 256]. On small samples, must be set to 1.
        range: [1,+inf]
    od_pval : float, [default=None]
        Use overfitting detector to stop training when reaching a specified threshold.
        Can be used only with eval_set.
        range: [0,1]
    od_wait : int, [default=None]
        Number of iterations which overfitting detector will wait after new best error.
    od_type : string, [default=None]
        Type of overfitting detector which will be used in program.
        Posible values:
            - 'IncToDec'
            - 'Iter'
        For 'Iter' type od_pval must not be set.
        If None, then od_type=IncToDec.
    nan_mode : string, [default=None]
        Way to process missing values for numeric features.
        Possible values:
            - 'Forbidden' - raises an exception if there is a missing value for a numeric feature in a dataset.
            - 'Min' - each missing value will be processed as the minimum numerical value.
            - 'Max' - each missing value will be processed as the maximum numerical value.
        If None, then nan_mode=Min.
    counter_calc_method : string, [default=None]
        The method used to calculate counters for dataset with Counter type.
        Possible values:
            - 'PrefixTest' - only objects up to current in the test dataset are considered
            - 'FullTest' - all objects are considered in the test dataset
            - 'SkipTest' - Objects from test dataset are not considered
            - 'Full' - all objects are considered for both learn and test dataset
        If None, then counter_calc_method=PrefixTest.
    leaf_estimation_iterations : int, [default=None]
        The number of steps in the gradient when calculating the values in the leaves.
        If None, then leaf_estimation_iterations=1.
        range: [1,+inf]
    leaf_estimation_method : string, [default=None]
        The method used to calculate the values in the leaves.
        Possible values:
            - 'Newton'
            - 'Gradient'
    thread_count : int, [default=None]
        Number of parallel threads used to run CatBoost.
        If None or -1, then the number of threads is set to the number of CPU cores.
        range: [1,+inf]
    random_seed : int, [default=None]
        Random number seed.
        If None, 0 is used.
        range: [0,+inf]
    use_best_model : bool, [default=None]
        To limit the number of trees in predict() using information about the optimal value of the error function.
        Can be used only with eval_set.
    best_model_min_trees : int, [default=None]
        The minimal number of trees the best model should have.
    verbose: bool
        When set to True, logging_level is set to 'Verbose'.
        When set to False, logging_level is set to 'Silent'.
    silent: bool, synonym for verbose
    logging_level : string, [default='Verbose']
        Possible values:
            - 'Silent'
            - 'Verbose'
            - 'Info'
            - 'Debug'
    metric_period : int, [default=1]
        The frequency of iterations to print the information to stdout. The value should be a positive integer.
    simple_ctr: list of strings, [default=None]
        Binarization settings for categorical features.
            Format : see documentation
            Example: ['Borders:CtrBorderCount=5:Prior=0:Prior=0.5', 'BinarizedTargetMeanValue:TargetBorderCount=10:TargetBorderType=MinEntropy', ...]
            CTR types:
                CPU and GPU
                - 'Borders'
                - 'Buckets'
                CPU only
                - 'BinarizedTargetMeanValue'
                - 'Counter'
                GPU only
                - 'FloatTargetMeanValue'
                - 'FeatureFreq'
            Number_of_borders, binarization type, target borders and binarizations, priors are optional parametrs
    combinations_ctr: list of strings, [default=None]
    per_feature_ctr: list of strings, [default=None]
    ctr_target_border_count: int, [default=None]
        Maximum number of borders used in target binarization for categorical features that need it.
        If TargetBorderCount is specified in 'simple_ctr', 'combinations_ctr' or 'per_feature_ctr' option it
        overrides this value.
        range: [1, 255]
    ctr_leaf_count_limit : int, [default=None]
        The maximum number of leaves with categorical features.
        If the number of leaves exceeds the specified limit, some leaves are discarded.
        The leaves to be discarded are selected as follows:
            - The leaves are sorted by the frequency of the values.
            - The top N leaves are selected, where N is the value specified in the parameter.
            - All leaves starting from N+1 are discarded.
        This option reduces the resulting model size
        and the amount of memory required for training.
        Note that the resulting quality of the model can be affected.
        range: [1,+inf] (for zero limit use ignored_features)
    store_all_simple_ctr : bool, [default=None]
        Ignore categorical features, which are not used in feature combinations,
        when choosing candidates for exclusion.
        Use this parameter with ctr_leaf_count_limit only.
    max_ctr_complexity : int, [default=4]
        The maximum number of Categ features that can be combined.
        range: [0,+inf]
    has_time : bool, [default=False]
        To use the order in which objects are represented in the input data
        (do not perform a random permutation of the dataset at the preprocessing stage).
    allow_const_label : bool, [default=False]
        To allow the constant label value in dataset.
    classes_count : int, [default=None]
        The upper limit for the numeric class label.
        Defines the number of classes for multiclassification.
        Only non-negative integers can be specified.
        The given integer should be greater than any of the target values.
        If this parameter is specified the labels for all classes in the input dataset
        should be smaller than the given value.
        If several of 'classes_count', 'class_weights', 'class_names' parameters are defined
        the numbers of classes specified by each of them must be equal.
    class_weights : list of floats, [default=None]
        Classes weights. The values are used as multipliers for the object weights.
        If None, all classes are supposed to have weight one.
        If several of 'classes_count', 'class_weights', 'class_names' parameters are defined
        the numbers of classes specified by each of them must be equal.
    class_names: list of strings, [default=None]
        Class names. Allows to redefine the default values for class labels (integer numbers).
        If several of 'classes_count', 'class_weights', 'class_names' parameters are defined
        the numbers of classes specified by each of them must be equal.
    one_hot_max_size : int, [default=None]
        Convert the feature to float
        if the number of different values that it takes exceeds the specified value.
        Ctrs are not calculated for such features.
    random_strength : float, [default=1]
        Score standard deviation multiplier.
    name : string, [default='experiment']
        The name that should be displayed in the visualization tools.
    ignored_features : list, [default=None]
        Indices of features that should be excluded when training.
    train_dir : string, [default=None]
        The directory in which you want to record generated in the process of learning files.
    custom_metric : string or list of strings, [default=None]
        To use your own metric function.
    custom_loss: alias to custom_metric
    eval_metric : string or object, [default=None]
        To optimize your custom metric in loss.
    bagging_temperature : float, [default=None]
        Controls intensity of Bayesian bagging. The higher the temperature the more aggressive bagging is.
        Typical values are in range [0, 1] (0 - no bagging, 1 - default).
    save_snapshot : bool, [default=None]
        Enable progress snapshotting for restoring progress after crashes or interruptions
    snapshot_file : string, [default=None]
        Learn progress snapshot file path, if None will use default filename
    snapshot_interval: int, [default=600]
        Interval between saving snapshots (seconds)
    fold_len_multiplier : float, [default=None]
        Fold length multiplier. Should be greater than 1
    used_ram_limit : string or number, [default=None]
        Set a limit on memory consumption (value like '1.2gb' or 1.2e9).
        WARNING: Currently this option affects CTR memory usage only.
    gpu_ram_part : float, [default=0.95]
        Fraction of the GPU RAM to use for training, a value from (0, 1].
    pinned_memory_size: int [default=None]
        Size of additional CPU pinned memory used for GPU learning,
        usually is estimated automatically, thus usually should not be set.
    allow_writing_files : bool, [default=True]
        If this flag is set to False, no files with different diagnostic info will be created during training.
        With this flag no snapshotting can be done. Plus visualisation will not
        work, because visualisation uses files that are created and updated during training.
    final_ctr_computation_mode : string, [default='Default']
        Possible values:
            - 'Default' - Compute final ctrs for all pools.
            - 'Skip' - Skip final ctr computation. WARNING: model without ctrs can't be applied.
    approx_on_full_history : bool, [default=False]
        If this flag is set to True, each approximated value is calculated using all the preceeding rows in the fold (slower, more accurate).
        If this flag is set to False, each approximated value is calculated using only the beginning 1/fold_len_multiplier fraction of the fold (faster, slightly less accurate).
    boosting_type : string, default value depends on object count and feature count in train dataset and on learning mode.
        Boosting scheme.
        Possible values:
            - 'Ordered' - Gives better quality, but may slow down the training.
            - 'Plain' - The classic gradient boosting scheme. May result in quality degradation, but does not slow down the training.
    task_type : string, [default=None]
        The calcer type used to train the model.
        Possible values:
            - 'CPU'
            - 'GPU'
    device_config : string, [default=None], deprecated, use devices instead
    devices : list or string, [default=None], GPU devices to use.
        String format is: '0' for 1 device or '0:1:3' for multiple devices or '0-3' for range of devices.
        List format is : [0] for 1 device or [0,1,3] for multiple devices.

    bootstrap_type : string, Bayesian, Bernoulli, Poisson, MVS.
        Default bootstrap is Bayesian.
        Poisson bootstrap is supported only on GPU.

    subsample : float, [default=None]
        Sample rate for bagging. This parameter can be used Poisson or Bernoully bootstrap types.

    sampling_unit : string, [default='Object'].
        Possible values:
            - 'Object'
            - 'Group'
        The parameter allows to specify the sampling scheme:
        sample weights for each object individually or for an entire group of objects together.

    dev_score_calc_obj_block_size: int, [default=5000000]
        CPU only. Size of block of samples in score calculation. Should be > 0
        Used only for learning speed tuning.
        Changing this parameter can affect results due to numerical accuracy differences

    dev_efb_max_buckets : int, [default=1024]
        CPU only. Maximum bucket count in exclusive features bundle. Should be in an integer between 0 and 65536.
        Used only for learning speed tuning.

    efb_max_conflict_fraction : float, [default=0.0]
        CPU only. Maximum allowed fraction of conflicting non-default values for features in exclusive features bundle.
        Should be a real value in [0, 1) interval.

    grow_policy : string, [SymmetricTree,Lossguide,Depthwise], [default=SymmetricTree]
        GPU only. The tree growing policy. It describes how to perform greedy tree construction.

    min_data_in_leaf : int, [default=1].
        GPU only.
        The minimum training samples count in leaf.
        CatBoost will not search for new splits in leaves with samples count less than min_data_in_leaf.
        This parameter is used only for Depthwise and Lossguide growing policies.

    max_leaves : int, [default=31],
        GPU only. The maximum leaf count in resulting tree.
        This parameter is used only for Lossguide growing policy.

    score_function : string, possible values L2, Correlation, NewtonL2, NewtonCorrelation, [default=Correlation]
        For growing policy Lossguide default=NewtonL2.
        GPU only. Score that is used during tree construction to select the next tree split.

    max_depth : int, Synonym for depth.

    n_estimators : int, synonym for iterations.

    num_trees : int, synonym for iterations.

    num_boost_round : int, synonym for iterations.

    colsample_bylevel : float, synonym for rsm.

    random_state : int, synonym for random_seed.

    reg_lambda : float, synonym for l2_leaf_reg.

    objective : string, synonym for loss_function.

    eta : float, synonym for learning_rate.

    max_bin : float, synonym for border_count.

    scale_pos_weight : float, synonym for class_weights.
        Can be used only for binary classification. Sets weight multiplier for
        class 1 to scale_pos_weight value.

    metadata : dict, string to string key-value pairs to be stored in model metadata storage

    early_stopping_rounds : int
        Synonym for od_wait. Only one of these parameters should be set.

    cat_features : list or numpy.array, [default=None]
        If not None, giving the list of Categ features indices or names (names are represented as strings).
        If it contains feature names, feature names must be defined for the training dataset passed to 'fit'.

    leaf_estimation_backtracking : string, [default=None]
        Type of backtracking during gradient descent.
        Possible values:
            - 'No' - never backtrack; supported on CPU and GPU
            - 'AnyImprovement' - reduce the descent step until the value of loss function is less than before the step; supported on CPU and GPU
            - 'Armijo' - reduce the descent step until Armijo condition is satisfied; supported on GPU only
    """
    def __init__(
        self,
        iterations=None,
        learning_rate=None,
        depth=None,
        l2_leaf_reg=None,
        model_size_reg=None,
        rsm=None,
        loss_function='Logloss',
        border_count=None,
        feature_border_type=None,
        input_borders=None,
        output_borders=None,
        fold_permutation_block=None,
        od_pval=None,
        od_wait=None,
        od_type=None,
        nan_mode=None,
        counter_calc_method=None,
        leaf_estimation_iterations=None,
        leaf_estimation_method=None,
        thread_count=None,
        random_seed=None,
        use_best_model=None,
        best_model_min_trees=None,
        verbose=None,
        silent=None,
        logging_level=None,
        metric_period=None,
        ctr_leaf_count_limit=None,
        store_all_simple_ctr=None,
        max_ctr_complexity=None,
        has_time=None,
        allow_const_label=None,
        classes_count=None,
        class_weights=None,
        class_names=None,
        one_hot_max_size=None,
        random_strength=None,
        name=None,
        ignored_features=None,
        train_dir=None,
        custom_loss=None,
        custom_metric=None,
        eval_metric=None,
        bagging_temperature=None,
        save_snapshot=None,
        snapshot_file=None,
        snapshot_interval=None,
        fold_len_multiplier=None,
        used_ram_limit=None,
        gpu_ram_part=None,
        pinned_memory_size=None,
        allow_writing_files=None,
        final_ctr_computation_mode=None,
        approx_on_full_history=None,
        boosting_type=None,
        simple_ctr=None,
        combinations_ctr=None,
        per_feature_ctr=None,
        ctr_description=None,
        ctr_target_border_count=None,
        task_type=None,
        device_config=None,
        devices=None,
        bootstrap_type=None,
        subsample=None,
        sampling_unit=None,
        dev_score_calc_obj_block_size=None,
        dev_efb_max_buckets=None,
        efb_max_conflict_fraction=None,
        max_depth=None,
        n_estimators=None,
        num_boost_round=None,
        num_trees=None,
        colsample_bylevel=None,
        random_state=None,
        reg_lambda=None,
        objective=None,
        eta=None,
        max_bin=None,
        scale_pos_weight=None,
        gpu_cat_features_storage=None,
        data_partition=None,
        metadata=None,
        early_stopping_rounds=None,
        cat_features=None,
        grow_policy=None,
        min_data_in_leaf=None,
        max_leaves=None,
        score_function=None,
        leaf_estimation_backtracking=None,
        ctr_history_unit=None
    ):
        params = {}
        not_params = ["not_params", "self", "params", "__class__"]
        for key, value in iteritems(locals().copy()):
            if key not in not_params and value is not None:
                params[key] = value

        super(CatBoostClassifier, self).__init__(params)

    @property
    def classes_(self):
        return getattr(self, "_classes", None)

    def fit(self, X, y=None, cat_features=None, sample_weight=None, baseline=None, use_best_model=None,
            eval_set=None, verbose=None, logging_level=None, plot=False, column_description=None,
            verbose_eval=None, metric_period=None, silent=None, early_stopping_rounds=None,
            save_snapshot=None, snapshot_file=None, snapshot_interval=None):
        """
        Fit the CatBoostClassifier model.

        Parameters
        ----------
        X : catboost.Pool or list or numpy.array or pandas.DataFrame or pandas.Series
            If not catboost.Pool, 2 dimensional Feature matrix or string - file with dataset.

        y : list or numpy.array or pandas.DataFrame or pandas.Series, optional (default=None)
            Labels, 1 dimensional array like.
            Use only if X is not catboost.Pool.

        cat_features : list or numpy.array, optional (default=None)
            If not None, giving the list of Categ columns indices.
            Use only if X is not catboost.Pool.

        sample_weight : list or numpy.array or pandas.DataFrame or pandas.Series, optional (default=None)
            Instance weights, 1 dimensional array like.

        baseline : list or numpy.array, optional (default=None)
            If not None, giving 2 dimensional array like data.
            Use only if X is not catboost.Pool.

        use_best_model : bool, optional (default=None)
            Flag to use best model

        eval_set : catboost.Pool or list, optional (default=None)
            A list of (X, y) tuple pairs to use as a validation set for early-stopping

        metric_period : int
            Frequency of evaluating metrics.

        verbose : bool or int
            If verbose is bool, then if set to True, logging_level is set to Verbose,
            if set to False, logging_level is set to Silent.
            If verbose is int, it determines the frequency of writing metrics to output and
            logging_level is set to Verbose.

        silent : bool
            If silent is True, logging_level is set to Silent.
            If silent is False, logging_level is set to Verbose.

        logging_level : string, optional (default=None)
            Possible values:
                - 'Silent'
                - 'Verbose'
                - 'Info'
                - 'Debug'

        plot : bool, optional (default=False)
            If True, draw train and eval error in Jupyter notebook

        verbose_eval : bool or int
            Synonym for verbose. Only one of these parameters should be set.

        early_stopping_rounds : int
            Activates Iter overfitting detector with od_wait set to early_stopping_rounds.

        save_snapshot : bool, [default=None]
            Enable progress snapshotting for restoring progress after crashes or interruptions

        snapshot_file : string, [default=None]
            Learn progress snapshot file path, if None will use default filename

        snapshot_interval: int, [default=600]
            Interval between saving snapshots (seconds)

        Returns
        -------
        model : CatBoost
        """

        params = self._init_params.copy()
        _process_synonyms(params)
        if 'loss_function' in params:
            self._check_is_classification_objective(params['loss_function'])

        self._fit(X, y, cat_features, None, sample_weight, None, None, None, None, baseline, use_best_model,
                  eval_set, verbose, logging_level, plot, column_description, verbose_eval, metric_period,
                  silent, early_stopping_rounds, save_snapshot, snapshot_file, snapshot_interval)
        return self

    def predict(self, data, prediction_type='Class', ntree_start=0, ntree_end=0, thread_count=-1, verbose=None):
        """
        Predict with data.

        Parameters
        ----------
        data : catboost.Pool or list of features or list of lists or numpy.array or pandas.DataFrame or pandas.Series
                or catboost.FeaturesData
            Data to apply model on.
            If data is a simple list (not list of lists) or a one-dimensional numpy.ndarray it is interpreted
            as a list of features for a single object.

        prediction_type : string, optional (default='Class')
            Can be:
            - 'RawFormulaVal' : return raw formula value.
            - 'Class' : return majority vote class.
            - 'Probability' : return probability for every class.

        ntree_start: int, optional (default=0)
            Model is applied on the interval [ntree_start, ntree_end) (zero-based indexing).

        ntree_end: int, optional (default=0)
            Model is applied on the interval [ntree_start, ntree_end) (zero-based indexing).
            If value equals to 0 this parameter is ignored and ntree_end equal to tree_count_.

        thread_count : int (default=-1)
            The number of threads to use when applying the model.
            Allows you to optimize the speed of execution. This parameter doesn't affect results.
            If -1, then the number of threads is set to the number of CPU cores.

        verbose : bool, optional (default=False)
            If True, writes the evaluation metric measured set to stderr.

        Returns
        -------
        prediction:
            If data is for a single object, the return value depends on prediction_type value:
                - 'RawFormulaVal' : return raw formula value.
                - 'Class' : return majority vote class.
                - 'Probability' : return one-dimensional numpy.ndarray with probability for every class.
            otherwise numpy.ndarray, with values that depend on prediction_type value:
                - 'RawFormulaVal' : one-dimensional array of raw formula value for each object.
                - 'Class' : one-dimensional array of majority vote classe for each object.
                - 'Probability' : two-dimensional numpy.ndarray with shape (number_of_objects x number_of_classes)
                  with probability for every class for each object.
        """
        return self._predict(data, prediction_type, ntree_start, ntree_end, thread_count, verbose, 'predict')

    def predict_proba(self, data, ntree_start=0, ntree_end=0, thread_count=-1, verbose=None):
        """
        Predict class probability with data.

        Parameters
        ----------
        data : catboost.Pool or list of features or list of lists or numpy.array or pandas.DataFrame or pandas.Series
                or catboost.FeaturesData
            Data to apply model on.
            If data is a simple list (not list of lists) or a one-dimensional numpy.ndarray it is interpreted
            as a list of features for a single object.

        ntree_start: int, optional (default=0)
            Model is applied on the interval [ntree_start, ntree_end) (zero-based indexing).

        ntree_end: int, optional (default=0)
            Model is applied on the interval [ntree_start, ntree_end) (zero-based indexing).
            If value equals to 0 this parameter is ignored and ntree_end equal to tree_count_.

        thread_count : int (default=-1)
            The number of threads to use when applying the model.
            Allows you to optimize the speed of execution. This parameter doesn't affect results.
            If -1, then the number of threads is set to the number of CPU cores.

        verbose : bool
            If True, writes the evaluation metric measured set to stderr.

        Returns
        -------
        prediction :
            If data is for a single object
                return one-dimensional numpy.ndarray with probability for every class.
            otherwise
                return two-dimensional numpy.ndarray with shape (number_of_objects x number_of_classes)
                with probability for every class for each object.
        """
        return self._predict(data, 'Probability', ntree_start, ntree_end, thread_count, verbose, 'predict_proba')

    def staged_predict(self, data, prediction_type='Class', ntree_start=0, ntree_end=0, eval_period=1, thread_count=-1, verbose=None):
        """
        Predict target at each stage for data.

        Parameters
        ----------
        data : catboost.Pool or list of features or list of lists or numpy.array or pandas.DataFrame or pandas.Series
                or catboost.FeaturesData
            Data to apply model on.
            If data is a simple list (not list of lists) or a one-dimensional numpy.ndarray it is interpreted
            as a list of features for a single object.

        prediction_type : string, optional (default='Class')
            Can be:
            - 'RawFormulaVal' : return raw formula value.
            - 'Class' : return majority vote class.
            - 'Probability' : return probability for every class.

        ntree_start: int, optional (default=0)
            Model is applied on the interval [ntree_start, ntree_end) with the step eval_period (zero-based indexing).

        ntree_end: int, optional (default=0)
            Model is applied on the interval [ntree_start, ntree_end) with the step eval_period (zero-based indexing).
            If value equals to 0 this parameter is ignored and ntree_end equal to tree_count_.

        eval_period: int, optional (default=1)
            Model is applied on the interval [ntree_start, ntree_end) with the step eval_period (zero-based indexing).

        thread_count : int (default=-1)
            The number of threads to use when applying the model.
            Allows you to optimize the speed of execution. This parameter doesn't affect results.
            If -1, then the number of threads is set to the number of CPU cores.

        verbose : bool
            If True, writes the evaluation metric measured set to stderr.

        Returns
        -------
        prediction : generator for each iteration that generates:
            If data is for a single object, the return value depends on prediction_type value:
                - 'RawFormulaVal' : return raw formula value.
                - 'Class' : return majority vote class.
                - 'Probability' : return one-dimensional numpy.ndarray with probability for every class.
            otherwise numpy.ndarray, with values that depend on prediction_type value:
                - 'RawFormulaVal' : one-dimensional array of raw formula value for each object.
                - 'Class' : one-dimensional array of majority vote class for each object.
                - 'Probability' : two-dimensional numpy.ndarray with shape (number_of_objects x number_of_classes)
                  with probability for every class for each object.
        """
        return self._staged_predict(data, prediction_type, ntree_start, ntree_end, eval_period, thread_count, verbose, 'staged_predict')

    def staged_predict_proba(self, data, ntree_start=0, ntree_end=0, eval_period=1, thread_count=-1, verbose=None):
        """
        Predict classification target at each stage for data.

        Parameters
        ----------
        data : catboost.Pool or list of features or list of lists or numpy.array or pandas.DataFrame or pandas.Series
                or catboost.FeaturesData
            Data to apply model on.
            If data is a simple list (not list of lists) or a one-dimensional numpy.ndarray it is interpreted
            as a list of features for a single object.

        ntree_start: int, optional (default=0)
            Model is applied on the interval [ntree_start, ntree_end) with the step eval_period (zero-based indexing).

        ntree_end: int, optional (default=0)
            Model is applied on the interval [ntree_start, ntree_end) with the step eval_period (zero-based indexing).
            If value equals to 0 this parameter is ignored and ntree_end equal to tree_count_.

        eval_period: int, optional (default=1)
            Model is applied on the interval [ntree_start, ntree_end) with the step eval_period (zero-based indexing).

        thread_count : int (default=-1)
            The number of threads to use when applying the model.
            Allows you to optimize the speed of execution. This parameter doesn't affect results.
            If -1, then the number of threads is set to the number of CPU cores.

        verbose : bool
            If True, writes the evaluation metric measured set to stderr.

        Returns
        -------
        prediction : generator for each iteration that generates:
            If data is for a single object
                return one-dimensional numpy.ndarray with probability for every class.
            otherwise
                return two-dimensional numpy.ndarray with shape (number_of_objects x number_of_classes)
                with probability for every class for each object.
        """
        return self._staged_predict(data, 'Probability', ntree_start, ntree_end, eval_period, thread_count, verbose, 'staged_predict_proba')

    def score(self, X, y=None):
        """
        Calculate accuracy.

        Parameters
        ----------
        X : catboost.Pool or list or numpy.array or pandas.DataFrame or pandas.Series
            Data to apply model on.
        y : list or numpy.array
            True labels.

        Returns
        -------
        accuracy : float
        """
        if isinstance(X, Pool):
            if X.get_label() is None:
                raise CatBoostError("Label in X has not initialized.")
            if y is not None:
                raise CatBoostError("Wrong initializing y: X is catboost.Pool object, y must be initialized inside catboost.Pool.")
            y = X.get_label()
        elif y is None:
            raise CatBoostError("y should be specified.")
        correct = []
        y = np.array(y, dtype=np.int32)
        predicted_classes = self._predict(
            X,
            prediction_type='Class',
            ntree_start=0,
            ntree_end=0,
            thread_count=-1,
            verbose=None,
            parent_method_name='score'
        )
        for i, val in enumerate(predicted_classes):
            correct.append(1 * (y[i] == np.int32(val)))
        return np.mean(correct)

    def _check_is_classification_objective(self, loss_function):
        if isinstance(loss_function, str) and not self._is_classification_objective(loss_function):
            raise CatBoostError("Invalid loss_function='{}': for classifier use "
                                "Logloss, CrossEntropy, MultiClass, MultiClassOneVsAll or custom objective object".format(loss_function))


class CatBoostRegressor(CatBoost):

    _estimator_type = 'regressor'

    """
    Implementation of the scikit-learn API for CatBoost regression.

    Parameters
    ----------
    Like in CatBoostClassifier, except loss_function, classes_count, class_names and class_weights

    loss_function : string, [default='RMSE']
        'RMSE'
        'MAE'
        'Quantile:alpha=value'
        'LogLinQuantile:alpha=value'
        'Poisson'
        'MAPE'
        'Lq:q=value'
    """
    def __init__(
        self,
        iterations=None,
        learning_rate=None,
        depth=None,
        l2_leaf_reg=None,
        model_size_reg=None,
        rsm=None,
        loss_function='RMSE',
        border_count=None,
        feature_border_type=None,
        input_borders=None,
        output_borders=None,
        fold_permutation_block=None,
        od_pval=None,
        od_wait=None,
        od_type=None,
        nan_mode=None,
        counter_calc_method=None,
        leaf_estimation_iterations=None,
        leaf_estimation_method=None,
        thread_count=None,
        random_seed=None,
        use_best_model=None,
        best_model_min_trees=None,
        verbose=None,
        silent=None,
        logging_level=None,
        metric_period=None,
        ctr_leaf_count_limit=None,
        store_all_simple_ctr=None,
        max_ctr_complexity=None,
        has_time=None,
        allow_const_label=None,
        one_hot_max_size=None,
        random_strength=None,
        name=None,
        ignored_features=None,
        train_dir=None,
        custom_metric=None,
        eval_metric=None,
        bagging_temperature=None,
        save_snapshot=None,
        snapshot_file=None,
        snapshot_interval=None,
        fold_len_multiplier=None,
        used_ram_limit=None,
        gpu_ram_part=None,
        pinned_memory_size=None,
        allow_writing_files=None,
        final_ctr_computation_mode=None,
        approx_on_full_history=None,
        boosting_type=None,
        simple_ctr=None,
        combinations_ctr=None,
        per_feature_ctr=None,
        ctr_description=None,
        ctr_target_border_count=None,
        task_type=None,
        device_config=None,
        devices=None,
        bootstrap_type=None,
        subsample=None,
        sampling_unit=None,
        dev_score_calc_obj_block_size=None,
        dev_efb_max_buckets=None,
        efb_max_conflict_fraction=None,
        max_depth=None,
        n_estimators=None,
        num_boost_round=None,
        num_trees=None,
        colsample_bylevel=None,
        random_state=None,
        reg_lambda=None,
        objective=None,
        eta=None,
        max_bin=None,
        gpu_cat_features_storage=None,
        data_partition=None,
        metadata=None,
        early_stopping_rounds=None,
        cat_features=None,
        grow_policy=None,
        min_data_in_leaf=None,
        max_leaves=None,
        score_function=None,
        leaf_estimation_backtracking=None,
        ctr_history_unit=None
    ):
        params = {}
        not_params = ["not_params", "self", "params", "__class__"]
        for key, value in iteritems(locals().copy()):
            if key not in not_params and value is not None:
                params[key] = value

        super(CatBoostRegressor, self).__init__(params)

    def fit(self, X, y=None, cat_features=None, sample_weight=None, baseline=None, use_best_model=None,
            eval_set=None, verbose=None, logging_level=None, plot=False, column_description=None,
            verbose_eval=None, metric_period=None, silent=None, early_stopping_rounds=None,
            save_snapshot=None, snapshot_file=None, snapshot_interval=None):
        """
        Fit the CatBoost model.

        Parameters
        ----------
        X : catboost.Pool or list or numpy.array or pandas.DataFrame or pandas.Series
            If not catboost.Pool, 2 dimensional Feature matrix or string - file with dataset.

        y : list or numpy.array or pandas.DataFrame or pandas.Series, optional (default=None)
            Labels, 1 dimensional array like.
            Use only if X is not catboost.Pool.

        cat_features : list or numpy.array, optional (default=None)
            If not None, giving the list of Categ columns indices.
            Use only if X is not catboost.Pool.

        sample_weight : list or numpy.array or pandas.DataFrame or pandas.Series, optional (default=None)
            Instance weights, 1 dimensional array like.

        baseline : list or numpy.array, optional (default=None)
            If not None, giving 2 dimensional array like data.
            Use only if X is not catboost.Pool.

        use_best_model : bool, optional (default=None)
            Flag to use best model

        eval_set : catboost.Pool or list, optional (default=None)
            A list of (X, y) tuple pairs to use as a validation set for
            early-stopping

        metric_period : int
            Frequency of evaluating metrics.

        verbose : bool or int
            If verbose is bool, then if set to True, logging_level is set to Verbose,
            if set to False, logging_level is set to Silent.
            If verbose is int, it determines the frequency of writing metrics to output and
            logging_level is set to Verbose.

        silent : bool
            If silent is True, logging_level is set to Silent.
            If silent is False, logging_level is set to Verbose.

        logging_level : string, optional (default=None)
            Possible values:
                - 'Silent'
                - 'Verbose'
                - 'Info'
                - 'Debug'

        plot : bool, optional (default=False)
            If True, draw train and eval error in Jupyter notebook

        verbose_eval : bool or int
            Synonym for verbose. Only one of these parameters should be set.

        early_stopping_rounds : int
            Activates Iter overfitting detector with od_wait set to early_stopping_rounds.

        save_snapshot : bool, [default=None]
            Enable progress snapshotting for restoring progress after crashes or interruptions

        snapshot_file : string, [default=None]
            Learn progress snapshot file path, if None will use default filename

        snapshot_interval: int, [default=600]
            Interval between saving snapshots (seconds)

        Returns
        -------
        model : CatBoost
        """

        params = deepcopy(self._init_params)
        _process_synonyms(params)
        if 'loss_function' in params:
            self._check_is_regressor_loss(params['loss_function'])

        return self._fit(X, y, cat_features, None, sample_weight, None, None, None, None, baseline,
                         use_best_model, eval_set, verbose, logging_level, plot, column_description,
                         verbose_eval, metric_period, silent, early_stopping_rounds,
                         save_snapshot, snapshot_file, snapshot_interval)

    def predict(self, data, ntree_start=0, ntree_end=0, thread_count=-1, verbose=None):
        """
        Predict with data.

        Parameters
        ----------
        data : catboost.Pool or list of features or list of lists or numpy.array or pandas.DataFrame or pandas.Series
                or catboost.FeaturesData
            Data to apply model on.
            If data is a simple list (not list of lists) or a one-dimensional numpy.ndarray it is interpreted
            as a list of features for a single object.

        ntree_start: int, optional (default=0)
            Model is applied on the interval [ntree_start, ntree_end) (zero-based indexing).

        ntree_end: int, optional (default=0)
            Model is applied on the interval [ntree_start, ntree_end) (zero-based indexing).
            If value equals to 0 this parameter is ignored and ntree_end equal to tree_count_.

        thread_count : int (default=-1)
            The number of threads to use when applying the model.
            Allows you to optimize the speed of execution. This parameter doesn't affect results.
            If -1, then the number of threads is set to the number of CPU cores.

        verbose : bool
            If True, writes the evaluation metric measured set to stderr.

        Returns
        -------
        prediction :
            If data is for a single object, the return value is single float formula return value
            otherwise one-dimensional numpy.ndarray of formula return values for each object.
        """
        return self._predict(data, "RawFormulaVal", ntree_start, ntree_end, thread_count, verbose, 'predict')

    def staged_predict(self, data, ntree_start=0, ntree_end=0, eval_period=1, thread_count=-1, verbose=None):
        """
        Predict target at each stage for data.

        Parameters
        ----------
        data : catboost.Pool or list of features or list of lists or numpy.array or pandas.DataFrame or pandas.Series
                or catboost.FeaturesData
            Data to apply model on.
            If data is a simple list (not list of lists) or a one-dimensional numpy.ndarray it is interpreted
            as a list of features for a single object.

        ntree_start: int, optional (default=0)
            Model is applied on the interval [ntree_start, ntree_end) with the step eval_period (zero-based indexing).

        ntree_end: int, optional (default=0)
            Model is applied on the interval [ntree_start, ntree_end) with the step eval_period (zero-based indexing).
            If value equals to 0 this parameter is ignored and ntree_end equal to tree_count_.

        eval_period: int, optional (default=1)
            Model is applied on the interval [ntree_start, ntree_end) with the step eval_period (zero-based indexing).

        thread_count : int (default=-1)
            The number of threads to use when applying the model.
            Allows you to optimize the speed of execution. This parameter doesn't affect results.
            If -1, then the number of threads is set to the number of CPU cores.

        verbose : bool
            If True, writes the evaluation metric measured set to stderr.

        Returns
        -------
        prediction : generator for each iteration that generates:
            If data is for a single object, the return value is single float formula return value
            otherwise one-dimensional numpy.ndarray of formula return values for each object.
        """
        return self._staged_predict(data, "RawFormulaVal", ntree_start, ntree_end, eval_period, thread_count, verbose, 'staged_predict')

    def score(self, X, y=None):
        """
        Calculate RMSE.

        Parameters
        ----------
        X : catboost.Pool or list or numpy.array or pandas.DataFrame or pandas.Series
            Data to apply model on.
        y : list or numpy.array
            True labels.

        Returns
        -------
        RMSE : float
        """
        if isinstance(X, Pool):
            if X.get_label() is None:
                raise CatBoostError("Label in X has not initialized.")
            if y is not None:
                raise CatBoostError("Wrong initializing y: X is catboost.Pool object, y must be initialized inside catboost.Pool.")
            y = X.get_label()
        elif y is None:
            raise CatBoostError("y should be specified.")
        error = []
        y = np.array(y, dtype=np.float64)
        predictions = self._predict(
            X,
            prediction_type='RawFormulaVal',
            ntree_start=0,
            ntree_end=0,
            thread_count=-1,
            verbose=None,
            parent_method_name='score'
        )
        for i, val in enumerate(predictions):
            error.append(pow(y[i] - val, 2))
        return np.sqrt(np.mean(error))

    def _check_is_regressor_loss(self, loss_function):
        if isinstance(loss_function, str) and not self._is_regression_objective(loss_function):
            raise CatBoostError("Invalid loss_function='{}': for regressor use "
                                "RMSE, MAE, Quantile, LogLinQuantile, Poisson, MAPE, Lq or custom objective object".format(loss_function))


def train(pool=None, params=None, dtrain=None, logging_level=None, verbose=None, iterations=None,
          num_boost_round=None, evals=None, eval_set=None, plot=None, verbose_eval=None, metric_period=None,
          early_stopping_rounds=None, save_snapshot=None, snapshot_file=None, snapshot_interval=None):
    """
    Train CatBoost model.

    Parameters
    ----------
    params : dict
        Parameters for CatBoost.
        If  None, all params are set to their defaults.
        If  dict, overriding parameters present in the dict.

    pool : catboost.Pool or tuple (X, y)
        Data to train on.

    iterations : int
        Number of boosting iterations. Can be set in params dict.

    evals : catboost.Pool or tuple (X, y)
        Synonym for eval_set. Only one of these parameters should be set.

    dtrain : catboost.Pool or tuple (X, y)
        Synonym for pool parameter. Only one of these parameters should be set.

    logging_level : string, optional (default=None)
        Possible values:
            - 'Silent'
            - 'Verbose'
            - 'Info'
            - 'Debug'

    metric_period : int
        Frequency of evaluating metrics.

    verbose : bool or int
        If verbose is bool, then if set to True, logging_level is set to Verbose,
        if set to False, logging_level is set to Silent.
        If verbose is int, it determines the frequency of writing metrics to output and
        logging_level is set to Verbose.

    verbose_eval : bool or int
        Synonym for verbose. Only one of these parameters should be set.

    iterations : int
        Number of boosting iterations. Can be set in params dict.

    num_boost_round : int
        Synonym for iterations. Only one of these parameters should be set.

    eval_set : catboost.Pool or tuple (X, y) or list [(X, y)]
        Dataset for evaluation.

    plot : bool, optional (default=False)
        If True, draw train and eval error in Jupyter notebook

    early_stopping_rounds : int
        Activates Iter overfitting detector with od_wait set to early_stopping_rounds.

    save_snapshot : bool, [default=None]
        Enable progress snapshotting for restoring progress after crashes or interruptions

    snapshot_file : string, [default=None]
        Learn progress snapshot file path, if None will use default filename

    snapshot_interval: int, [default=600]
        Interval between saving snapshots (seconds)

    Returns
    -------
    model : CatBoost class
    """

    if params is None:
        raise CatBoostError("params should be set.")

    if dtrain is not None:
        if pool is None:
            pool = dtrain
        else:
            raise CatBoostError("Only one of the parameters pool and dtrain should be set.")

    if num_boost_round is not None:
        if iterations is None:
            iterations = num_boost_round
        else:
            raise CatBoostError("Only one of the parameters iterations and num_boost_round should be set.")
    if iterations is not None:
        params = deepcopy(params)
        params.update({
            'iterations': iterations
        })

    if early_stopping_rounds is not None:
        params.update({
            'od_type': 'Iter'
        })
        if 'od_pval' in params:
            del params['od_pval']
        params.update({
            'od_wait': early_stopping_rounds
        })

    if evals is not None:
        if eval_set is not None:
            raise CatBoostError('Only one of the parameters evals, eval_set should be set.')
        eval_set = evals

    model = CatBoost(params)
    model.fit(X=pool, eval_set=eval_set, logging_level=logging_level, plot=plot, verbose=verbose,
              verbose_eval=verbose_eval, metric_period=metric_period,
              early_stopping_rounds=early_stopping_rounds, save_snapshot=save_snapshot,
              snapshot_file=snapshot_file, snapshot_interval=snapshot_interval)
    return model


def cv(pool=None, params=None, dtrain=None, iterations=None, num_boost_round=None,
       fold_count=None, nfold=None, inverted=False, partition_random_seed=0, seed=None,
       shuffle=True, logging_level=None, stratified=False, as_pandas=True, metric_period=None,
       verbose=None, verbose_eval=None, plot=False, early_stopping_rounds=None,
       save_snapshot=None, snapshot_file=None, snapshot_interval=None, max_time_spent_on_fixed_cost_ratio=0.05,
       dev_max_iterations_batch_size=100000):
    """
    Cross-validate the CatBoost model.

    Parameters
    ----------
    pool : catboost.Pool
        Data to cross-validate on.

    params : dict
        Parameters for CatBoost.
        CatBoost has many of parameters, all have default values.
        If  None, all params still defaults.
        If  dict, overriding some (or all) params.

    dtrain : catboost.Pool or tuple (X, y)
        Synonym for pool parameter. Only one of these parameters should be set.

    iterations : int
        Number of boosting iterations. Can be set in params dict.

    num_boost_round : int
        Synonym for iterations. Only one of these parameters should be set.

    fold_count : int, optional (default=3)
        The number of folds to split the dataset into.

    nfold : int
        Synonym for fold_count.

    inverted : bool, optional (default=False)
        Train on the test fold and evaluate the model on the training folds.

    partition_random_seed : int, optional (default=0)
        Use this as the seed value for random permutation of the data.
        Permutation is performed before splitting the data for cross validation.
        Each seed generates unique data splits.

    seed : int, optional
        Synonym for partition_random_seed. This parameter is deprecated. Use
        partition_random_seed instead.
        If both parameters are initialised partition_random_seed parameter is
        ignored.

    shuffle : bool, optional (default=True)
        Shuffle the dataset objects before splitting into folds.

    logging_level : string, optional (default=None)
        Possible values:
            - 'Silent'
            - 'Verbose'
            - 'Info'
            - 'Debug'

    stratified : bool, optional (default=False)
        Perform stratified sampling.

    as_pandas : bool, optional (default=True)
        Return pd.DataFrame when pandas is installed.
        If False or pandas is not installed, return dict.

    metric_period : int, [default=1]
        The frequency of iterations to print the information to stdout. The value should be a positive integer.

    verbose : bool or int
        If verbose is bool, then if set to True, logging_level is set to Verbose,
        if set to False, logging_level is set to Silent.
        If verbose is int, it determines the frequency of writing metrics to output and
        logging_level is set to Verbose.

    verbose_eval : bool or int
        Synonym for verbose. Only one of these parameters should be set.

    plot : bool, optional (default=False)
        If True, draw train and eval error in Jupyter notebook

    early_stopping_rounds : int
        Activates Iter overfitting detector with od_wait set to early_stopping_rounds.

    save_snapshot : bool, [default=None]
        Enable progress snapshotting for restoring progress after crashes or interruptions

    snapshot_file : string, [default=None]
        Learn progress snapshot file path, if None will use default filename

    snapshot_interval: int, [default=600]
        Interval between saving snapshots (seconds)

    max_time_spent_on_fixed_cost_ratio: float [default:0.05]
        Iteration batch sizes are computed to keep time spent on fixed cost computations
        (spent on fold initialization for each batch) under this ratio.
        Increasing this parameter will decrease the batch sizes which could be useful to get first batch
        results sooner in exchange for greater total computation time.

    dev_max_iterations_batch_size: int [default:100000]
        Max number of iterations to compute for each fold before aggregating results.
        Should be used only for testing, max_time_spent_on_fixed_cost_ratio is the prefered parameter to be
        used in normal operation.

    Returns
    -------
    cv results : pandas.core.frame.DataFrame with cross-validation results
        columns are: test-error-mean  test-error-std  train-error-mean  train-error-std
    """
    if params is None:
        raise CatBoostError("params should be set.")

    params = deepcopy(params)
    _process_synonyms(params)

    metric_period, verbose, logging_level = _process_verbose(metric_period, verbose, logging_level, verbose_eval)

    if verbose is not None:
        params.update({
            'verbose': verbose
        })

    if logging_level is not None:
        params.update({
            'logging_level': logging_level
        })

    if metric_period is not None:
        params.update({
            'metric_period': metric_period
        })

    if early_stopping_rounds is not None:
        params.update({
            'od_type': 'Iter'
        })
        if 'od_pval' in params:
            del params['od_pval']
        params.update({
            'od_wait': early_stopping_rounds
        })

    if dtrain is not None:
        if pool is None:
            pool = dtrain
        else:
            raise CatBoostError("Only one of the parameters pool and dtrain should be set.")

    if num_boost_round is not None:
        if iterations is None:
            iterations = num_boost_round
        else:
            raise CatBoostError("Only one of the parameters iterations and num_boost_round should be set.")

    if iterations is not None:
        params.update({
            'iterations': iterations
        })

    if seed is not None:
        partition_random_seed = seed

    if save_snapshot is not None:
        params['save_snapshot'] = save_snapshot

    if snapshot_file is not None:
        params['snapshot_file'] = snapshot_file

    if snapshot_interval is not None:
        params['snapshot_interval'] = snapshot_interval

    if nfold is None and fold_count is None:
        fold_count = 3
    elif fold_count is None:
        fold_count = nfold
    else:
        assert nfold is None or nfold == fold_count

    if 'cat_features' in params:
        cat_feature_indices_from_params = _get_cat_features_indices(params['cat_features'], pool.get_feature_names())
        if set(pool.get_cat_feature_indices()) != set(cat_feature_indices_from_params):
            raise CatBoostError("categorical features indices in params are different from ones in pool "
                                + str(cat_feature_indices_from_params) +
                                " vs " + str(pool.get_cat_feature_indices()))
        del params['cat_features']

    with log_fixup(), plot_wrapper(plot, params):
        return _cv(params, pool, fold_count, inverted, partition_random_seed, shuffle, stratified,
                   as_pandas, max_time_spent_on_fixed_cost_ratio, dev_max_iterations_batch_size)


class BatchMetricCalcer(_MetricCalcerBase):

    def __init__(self, catboost, metrics, ntree_start, ntree_end, eval_period, thread_count, tmp_dir):
        super(BatchMetricCalcer, self).__init__(catboost)
        if tmp_dir is None:
            tmp_dir = tempfile.mkdtemp()
            delete_temp_dir_flag = True
        else:
            delete_temp_dir_flag = False

        if isinstance(metrics, str):
            metrics = [metrics]
        self._create_calcer(metrics, ntree_start, ntree_end, eval_period, thread_count, tmp_dir, delete_temp_dir_flag)


def sum_models(models, weights=None, ctr_merge_policy='IntersectingCountersAverage'):
    result = CatBoost()
    result._sum_models(models, weights, ctr_merge_policy)
    return result
