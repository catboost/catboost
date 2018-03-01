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

if platform.system() == 'Linux':
    try:
        ctypes.CDLL('librt.so')
    except:
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

so_paths = get_so_paths('./gpu') + get_so_paths('./')
for so_path in so_paths:
    try:
        _catboost = imp.load_dynamic('_catboost', so_path)
        _PoolBase = _catboost._PoolBase
        _CatBoostBase = _catboost._CatBoostBase
        _MetricCalcerBase = _catboost._MetricCalcerBase
        _cv = _catboost._cv
        _set_logger = _catboost._set_logger
        _reset_logger = _catboost._reset_logger
        _configure_malloc = _catboost._configure_malloc
        CatboostError = _catboost.CatboostError
        break
    except ImportError:
        pass
else:
    from _catboost import _PoolBase, _CatBoostBase, _MetricCalcerBase, CatboostError, _cv, _set_logger, _reset_logger, _configure_malloc

from contextlib import contextmanager


_configure_malloc()

INTEGER_TYPES = (integer_types, np.integer)
FLOAT_TYPES = (float, np.floating)
STRING_TYPES = (string_types,)
ARRAY_TYPES = (list, np.ndarray, DataFrame, Series)


@contextmanager
def log_fixup():
    _set_logger(sys.stdout)
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


class Pool(_PoolBase):
    """
    Pool used in CatBoost as data structure to train model from.
    """

    def __init__(self, data, label=None, cat_features=None, column_description=None, pairs=None, delimiter='\t',
                 has_header=False, weight=None, group_id=None, pairs_weight=None, baseline=None,
                 feature_names=None, thread_count=-1):
        """
        Pool is a internal data structure that used by CatBoost.
        You can construct Pool from list, numpy.array, pandas.DataFrame, pandas.Series.

        Parameters
        ----------
        data : list or numpy.array or pandas.DataFrame or pandas.Series or string
            Data source of Pool.
            If list or numpy.arrays or pandas.DataFrame or pandas.Series, giving 2 dimensional array like data.
            If string, giving the path to the file with data in catboost format.

        label : list or numpy.arrays or pandas.DataFrame or pandas.Series, optional (default=None)
            Label of the training data.
            If not  None, giving 1 dimensional array like data with floats.

        cat_features : list or numpy.array, optional (default=None)
            If not None, giving the list of Categ columns indices.

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
            The shape should be Nx2, where N is the pairs' count. The first element of pair is
            the index of winner document in training set. The second element of pair is
            the index of loser document in training set.
            If string, giving the path to the file with pairs description.

        delimiter : string, optional (default='\t')
            Delimiter to use for separate features in file.
            Should be only one symbol, otherwise would be taken only the first character of the string.

        has_header : boolm optional (default=False)
            If True, read column names from first line.

        weight : list or numpy.array, optional (default=None)
            Weight for each instance.
            If not None, giving 1 dimensional array like data.

        group_id : list or numpy.array, optional (default=None)
            group id for each instance.
            If not None, giving 1 dimensional array like data.

        pairs_weight : list or numpy.array, optional (default=None)
            Weight for each pair.
            If not None, giving 1 dimensional array like pairs.

        baseline : list or numpy.array, optional (default=None)
            Baseline for each instance.
            If not None, giving 2 dimensional array like data.

        feature_names : list, optional (default=None)
            Names for each given data_feature.

        thread_count : int, optional (default=-1)
            Thread count to read data from file.
            Use only with reading data from file.
            If -1, then the number of threads is set to the number of cores.
        """
        if data is not None:
            self._check_data_type(data)
            self._check_data_empty(data)
            if pairs is not None and isinstance(data, STRING_TYPES) != isinstance(pairs, STRING_TYPES):
                raise CatboostError("Data and pairs should be the same types.")
            if isinstance(data, STRING_TYPES):
                if any(v is not None for v in [cat_features, weight, group_id, pairs_weight, baseline, feature_names]):
                    raise CatboostError("cat_features, weight, group_id, pairs_weight, baseline, feature_names should have the None type when the pool is read from the file.")
                self._read(data, column_description, pairs, delimiter, has_header, thread_count)
            else:
                self._init(data, label, cat_features, pairs, weight, group_id, pairs_weight, baseline, feature_names)
        super(Pool, self).__init__()

    def _check_files(self, data, column_description, pairs):
        """
        Check files existence.
        """
        if not os.path.isfile(data):
            raise CatboostError("Invalid data path='{}': file does not exist.".format(data))
        if column_description is not None and not os.path.isfile(column_description):
            raise CatboostError("Invalid column_description path='{}': file does not exist.".format(column_description))
        if pairs is not None and not os.path.isfile(pairs):
            raise CatboostError("Invalid pairs path='{}': file does not exist.".format(pairs))

    def _check_delimiter(self, delimiter):
        if not isinstance(delimiter, STRING_TYPES):
            raise CatboostError("Invalid delimiter type={} : must be str().".format(type(delimiter)))
        if len(delimiter) < 1:
            raise CatboostError("Invalid delimiter length={} : must be > 0.".format(len(delimiter)))

    def _check_column_description_type(self, column_description):
        """
        Check type of column_description parameter.
        """
        if not isinstance(column_description, STRING_TYPES):
            raise CatboostError("Invalid column_description type={}: must be str().".format(type(column_description)))

    def _check_cf_type(self, cat_features):
        """
        Check type of cat_feature parameter.
        """
        if not isinstance(cat_features, (list, np.ndarray)):
            raise CatboostError("Invalid cat_features type={}: must be list() or np.ndarray().".format(type(cat_features)))

    def _check_cf_value(self, cat_features, features_count):
        """
        Check values in cat_feature parameter. Must be int indices.
        """
        for indx, feature in enumerate(cat_features):
            if not isinstance(feature, INTEGER_TYPES):
                raise CatboostError("Invalid cat_features[{}] = {} value type={}: must be int().".format(indx, feature, type(feature)))
            if feature >= features_count:
                raise CatboostError("Invalid cat_features[{}] = {} value: must be < {}.".format(indx, feature, features_count))

    def _check_pairs_type(self, pairs):
        """
        Check type of pairs parameter.
        """
        if not isinstance(pairs, (list, np.ndarray, DataFrame)):
            raise CatboostError("Invalid pairs type={}: must be list(), np.ndarray() or pd.DataFrame.".format(type(pairs)))

    def _check_pairs_value(self, pairs):
        """
        Check values in pairs parameter. Must be int indices.
        """
        for pair_id, pair in enumerate(pairs):
            if (len(pair) != 2):
                raise CatboostError("Length of pairs[{}] isn't equal to 2.".format(pair_id))
            for i, index in enumerate(pair):
                if not isinstance(index, INTEGER_TYPES):
                    raise CatboostError("Invalid pairs[{}][{}] = {} value type={}: must be int().".format(pair_id, i, index, type(index)))

    def _check_data_type(self, data):
        """
        Check type of data.
        """
        if not isinstance(data, (STRING_TYPES, ARRAY_TYPES)):
            raise CatboostError("Invalid data type={}: data must be list(), np.ndarray(), DataFrame(), Series() or filename str().".format(type(data)))

    def _check_data_empty(self, data):
        """
        Check data is not empty.
        """
        if isinstance(data, STRING_TYPES):
            if not data:
                raise CatboostError("Features filename is empty.")
        elif isinstance(data, ARRAY_TYPES):
            data_shape = np.shape(data)
            if isinstance(data, Series) and data_shape[0] > 0:
                if isinstance(data[0], Iterable):
                    data_shape = tuple(data_shape + tuple([len(data[0])]))
                else:
                    data_shape = tuple(data_shape + tuple([1]))
            if not len(data_shape) == 2 or not min(data_shape) > 0:
                raise CatboostError("Input data has invalid shape or empty: {}. Must be 2 dimensional".format(data_shape))

    def _check_label_type(self, label):
        """
        Check type of label.
        """
        if not isinstance(label, ARRAY_TYPES):
            raise CatboostError("Invalid label type={}: must be array like.".format(type(label)))

    def _check_label_empty(self, label):
        """
        Check label is not empty.
        """
        if len(label) == 0:
            raise CatboostError("Labels variable is empty.")

    def _check_label_shape(self, label, samples_count):
        """
        Check label length and dimension.
        """
        if len(label) != samples_count:
            raise CatboostError("Length of label={} and length of data={} is different.".format(len(label), samples_count))
        if isinstance(label[0], Iterable) and not isinstance(label[0], STRING_TYPES):
            if len(label[0]) > 1:
                raise CatboostError("Input label cannot have multiple values per row.")

    def _check_baseline_type(self, baseline):
        """
        Check type of baseline parameter.
        """
        if not isinstance(baseline, ARRAY_TYPES):
            raise CatboostError("Invalid baseline type={}: must be array like.".format(type(baseline)))

    def _check_baseline_shape(self, baseline, samples_count):
        """
        Check baseline length and dimension.
        """
        if len(baseline) != samples_count:
            raise CatboostError("Length of baseline={} and length of data={} are different.".format(len(baseline), samples_count))
        if not isinstance(baseline[0], Iterable) or isinstance(baseline[0], STRING_TYPES):
            raise CatboostError("Baseline must be 2 dimensional data, 1 column for each class.")
        try:
            if np.array(baseline).dtype not in (np.dtype('float'), np.dtype('int')):
                raise CatboostError()
        except CatboostError:
            raise CatboostError("Invalid baseline value type={}: must be float or int.".format(np.array(baseline).dtype))

    def _check_weight_type(self, weight):
        """
        Check type of weight parameter.
        """
        if not isinstance(weight, ARRAY_TYPES):
            raise CatboostError("Invalid weight type={}: must be array like.".format(type(weight)))

    def _check_weight_shape(self, weight, samples_count):
        """
        Check weight length.
        """
        if len(weight) != samples_count:
            raise CatboostError("Length of weight={} and length of data={} are different.".format(len(weight), samples_count))
        if not isinstance(weight[0], (INTEGER_TYPES, FLOAT_TYPES)):
            raise CatboostError("Invalid weight value type={}: must be 1 dimensional data with int, float or long types.".format(type(weight[0])))

    def _check_group_id_type(self, group_id):
        """
        Check type of group_id parameter.
        """
        if not isinstance(group_id, ARRAY_TYPES):
            raise CatboostError("Invalid group_id type={}: must be array like.".format(type(group_id)))

    def _check_group_id_shape(self, group_id, samples_count):
        """
        Check group_id length.
        """
        if len(group_id) != samples_count:
            raise CatboostError("Length of group_id={} and length of data={} are different.".format(len(group_id), samples_count))
        if not isinstance(group_id[0], (INTEGER_TYPES)):
            raise CatboostError("Invalid group_id value type={}: must be 1 dimensional data with int types.".format(type(group_id[0])))

    def _check_feature_names(self, feature_names, num_col=None):
        if num_col is None:
            num_col = self.num_col()
        if not isinstance(feature_names, Sequence):
            raise CatboostError("Invalid feature_names type={} : must be list".format(type(feature_names)))
        if len(feature_names) != num_col:
            raise CatboostError("Invalid length feature_names={} : must be equal to number of columns in data={}".format(len(feature_names), num_col))

    def _check_thread_count(self, thread_count):
        if not isinstance(thread_count, INTEGER_TYPES):
            raise CatboostError("Invalid thread_count type={} : must be int".format(type(thread_count)))

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

    def _init(self, data_matrix, label, cat_features, pairs, weight, group_id, pairs_weight, baseline, feature_names):
        """
        Initialize Pool from array like data.
        """
        if isinstance(data_matrix, DataFrame):
            feature_names = list(data_matrix.columns)
            data_matrix = data_matrix.values
        if isinstance(data_matrix, Series):
            data_matrix = data_matrix.values
        samples_count = len(data_matrix)
        features_count = len(data_matrix[0])
        pairs_len = 0
        if label is not None:
            self._check_label_type(label)
            self._check_label_empty(label)
            label = self._if_pandas_to_numpy(label)
            self._check_label_shape(label, samples_count)
        if cat_features is not None:
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
        if pairs_weight is not None:
            self._check_weight_type(pairs_weight)
            pairs_weight = self._if_pandas_to_numpy(pairs_weight)
            self._check_weight_shape(pairs_weight, pairs_len)
        if baseline is not None:
            self._check_baseline_type(baseline)
            baseline = self._if_pandas_to_numpy(baseline)
            baseline = np.reshape(baseline, (samples_count, -1))
            self._check_baseline_shape(baseline, samples_count)
        if feature_names is not None:
            self._check_feature_names(feature_names, features_count)
        self._init_pool(data_matrix, label, cat_features, pairs, weight, group_id, pairs_weight, baseline, feature_names)


def _build_train_pool(X, y, cat_features, pairs, sample_weight, group_id, pairs_weight, baseline, column_description):
    train_pool = None
    if isinstance(X, Pool):
        train_pool = X
        if any(v is not None for v in [cat_features, sample_weight, group_id, pairs_weight, baseline]):
            raise CatboostError("cat_features, sample_weight, group_id, pairs_weight, baseline should have the None type when X has Pool type.")
        if X.get_label() is None:
            raise CatboostError("Label in X has not initialized.")
        if y is not None:
            raise CatboostError("Wrong initializing y: X is Pool object, y must be initialized inside Pool.")
    elif isinstance(X, STRING_TYPES):
            train_pool = Pool(data=X, pairs=pairs, column_description=column_description)
    else:
        if y is None:
            raise CatboostError("y has not initialized in fit(): X is not Pool object, y must be not None in fit().")
        train_pool = Pool(X, y, cat_features=cat_features, pairs=pairs, weight=sample_weight, group_id=group_id,
                 pairs_weight=pairs_weight, baseline=baseline)
    return train_pool

class CatBoost(_CatBoostBase):
    """
    CatBoost model, that contains training, prediction and evaluation.
    """

    def __init__(self, params=None, model_file=None):
        """
        Initialize the CatBoost.

        Parameters
        ----------
        params : dict
            Parameters for CatBoost.
            If  None, all params are set to their defaults.
            If  dict, overriding parameters present in dict.

        model_file : string, optional (default=None)
            If string, giving the path to the file with input model.
        """
        params = deepcopy(params)
        if params is None:
            params = {}

        self._process_synonyms(params)

        self._additional_params = ['calc_feature_importance']
        kwargs = {}
        for param in self._additional_params:
            if param in params:
                kwargs.update({
                    param: params[param]
                })
                del params[param]
        kwargs.update(params.get('kwargs', {}))
        params['kwargs'] = kwargs

        if 'verbose' in params:
            if not isinstance(params['verbose'], bool):
                raise CatboostError('verbose should be bool')
            if 'logging_level' in params:
                raise CatboostError('only one of parameters logging_level, verbose should be set.')
            if params['verbose'] is True:
                params['logging_level'] = 'Verbose'
            else:
                params['logging_level'] = 'Silent'
            del params['verbose']

        self._check_params(params)
        params = self._params_type_cast(params)
        super(CatBoost, self).__init__(params)
        if model_file is not None:
            self.load_model(model_file)

    def _params_type_cast(self, params):
        casted_params = {}
        for key, value in iteritems(params):
            value = _cast_to_base_types(value)
            casted_params[key] = value
        return casted_params

    def _check_params(self, params):
        if not isinstance(params, (Mapping, MutableMapping)):
            raise CatboostError("Invalid params type={}: must be dict().".format(type(params)))
        if 'ctr_description' in params:
            if not isinstance(params['ctr_description'], Sequence):
                raise CatboostError("Invalid ctr_description type={} : must be list of strings".format(type(params['ctr_description'])))
        if 'custom_loss' in params:
            if isinstance(params['custom_loss'], STRING_TYPES):
                params['custom_loss'] = [params['custom_loss']]
            if not isinstance(params['custom_loss'], Sequence):
                raise CatboostError("Invalid `custom_loss` type={} : must be string or list of strings.".format(type(params['custom_loss'])))
        if 'custom_metric' in params:
            if isinstance(params['custom_metric'], STRING_TYPES):
                params['custom_metric'] = [params['custom_metric']]
            if not isinstance(params['custom_metric'], Sequence):
                raise CatboostError("Invalid `custom_metric` type={} : must be string or list of strings.".format(type(params['custom_metric'])))
        if 'kwargs' in params:
            for param in params['kwargs'].keys():
                if param not in self._additional_params:
                    raise CatboostError("Invalid param `{}`.".format(param))

    def _process_synonyms(self, params):
        if 'objective' in params:
            if 'loss_function' in params:
                raise CatboostError('only one of parameters loss_function, objective should be initialized.')
            params['loss_function'] = params['objective']
            del params['objective']

        if 'scale_pos_weight' in params:
            if 'loss_function' in params and params['loss_function'] != 'Logloss':
                    raise CatboostError('scale_pos_weight is only supported for binary classification Logloss loss')
            if 'class_weights' in params:
                raise CatboostError('only one of parameters scale_pos_weight, class_weights should be initialized.')
            params['class_weights'] = [1.0, params['scale_pos_weight']]
            del params['scale_pos_weight']

        if 'eta' in params:
            if 'learning_rate' in params:
                raise CatboostError('only one of parameters learning_rate, eta should be initialised.')
            params['learning_rate'] = params['eta']
            del params['eta']

        if 'max_bin' in params:
            if 'border_count' in params:
                raise CatboostError('only one of parameters max_bin, border_count, eta should be initialised.')
            params['border_count'] = params['max_bin']
            del params['max_bin']

        if 'max_depth' in params:
            if 'depth' in params:
                raise CatboostError('only one of parameters depth, max_depth should be initialised.')
            params['depth'] = params['max_depth']
            del params['max_depth']

        if 'colsample_bylevel' in params:
            if 'rsm' in params:
                raise CatboostError('only one of parameters colsample_bylevel, rsm should be initialised.')
            params['rsm'] = params['colsample_bylevel']
            del params['colsample_bylevel']

        if 'random_state' in params:
            if 'random_seed' in params:
                raise CatboostError('only one of parameters random_seed, random_state should be initialised.')
            params['random_seed'] = params['random_state']
            del params['random_state']

        if 'reg_lambda' in params:
            if 'l2_leaf_reg' in params:
                raise CatboostError('only one of parameters reg_lambda, l2_leaf_reg should be initialised.')
            params['l2_leaf_reg'] = params['reg_lambda']
            del params['reg_lambda']

        if 'n_estimators' in params:
            if 'iterations' in params or 'num_trees' in params or 'num_boost_round' in params:
                raise CatboostError('only one of parameters iterations, n_estimators, num_trees, num_boost_round should be initialised.')
            params['iterations'] = params['n_estimators']
            del params['n_estimators']

        if 'num_trees' in params:
            if 'iterations' in params or 'num_trees' in params or 'num_boost_round' in params:
                raise CatboostError('only one of parameters iterations, n_estimators, num_trees, num_boost_round should be initialised.')
            params['iterations'] = params['num_trees']
            del params['num_trees']

        if 'num_boost_round' in params:
            if 'iterations' in params or 'num_trees' in params or 'num_boost_round' in params:
                raise CatboostError('only one of parameters iterations, n_estimators, num_trees, num_boost_round should be initialised.')
            params['iterations'] = params['num_boost_round']
            del params['num_boost_round']

    def _clear_tsv_files(self, train_dir):
        for filename in ['learn_error.tsv', 'test_error.tsv', 'time_left.tsv', 'meta.tsv']:
            path = os.path.join(train_dir, filename)
            if os.path.exists(path):
                os.remove(path)


    def _fit(self, X, y, cat_features, pairs, sample_weight, group_id, pairs_weight, baseline, use_best_model, eval_set, verbose, logging_level, plot, column_description):
        params = self._get_init_train_params()
        init_params = self._get_init_params()
        calc_feature_importance = True
        if 'calc_feature_importance' in init_params:
            calc_feature_importance = init_params["calc_feature_importance"]
        if verbose is not None:
            if not isinstance(verbose, bool):
                raise CatboostError('verbose should be bool.')
            if logging_level is not None:
                raise CatboostError('only one of parameters logging_level, verbose should be set.')
            if logging_level is not None:
                raise CatboostError('only one of parameters logging_level, verbose should be set.')
            if verbose:
                logging_level = 'Verbose'
            else:
                logging_level = 'Silent'

        if logging_level is not None:
            params['logging_level'] = logging_level
        if use_best_model is not None:
            params['use_best_model'] = use_best_model

        train_pool = _build_train_pool(X, y, cat_features, pairs, sample_weight, group_id, pairs_weight, baseline, column_description)
        if train_pool.is_empty_:
            raise CatboostError("X is empty.")

        if column_description is not None and not isinstance(X, STRING_TYPES) and not isinstance(eval_set, STRING_TYPES):
            raise CatboostError("column_description should be None if X and eval_set are not strings.")

        allow_clear_pool = not isinstance(X, Pool)

        if eval_set is None:
            if self.get_param('use_best_model'):
                raise CatboostError("For use param {'use_best_model': True} need initialize 'eval_set'.")
            eval_set = Pool(None)
        elif not isinstance(eval_set, Pool):
            if isinstance(eval_set, STRING_TYPES):
                eval_set = Pool(eval_set, column_description=column_description)
            else:
                if len(eval_set) != 2:
                    raise CatboostError("Invalid eval_set shape={}: must be (X, y) or filename.".format(np.shape(eval_set)))
                eval_set = Pool(eval_set[0], eval_set[1], cat_features=cat_features)

        if plot:
            train_dir = self.get_param('train_dir') or '.'
            self._clear_tsv_files(train_dir)

            try:
                from .widget import CatboostIpythonWidget
                widget = CatboostIpythonWidget(train_dir)
                widget._run_update()
            except ImportError as e:
                warnings.warn("For drow plots in fit() method you should install ipywidgets and ipython")
                raise ImportError(str(e))
        with log_fixup():
            self._train(train_pool, eval_set, params, allow_clear_pool)
        if calc_feature_importance:
            if allow_clear_pool:
                train_pool = _build_train_pool(X, y, cat_features, pairs, sample_weight, group_id, pairs_weight, baseline, column_description)
            setattr(self, "_feature_importance", self.get_feature_importance(train_pool))

        if 'loss_function' in params and self._is_classification_loss(params['loss_function']):
            setattr(self, "_classes", np.unique(train_pool.get_label()))
        return self

    def fit(self, X, y=None, cat_features=None, pairs=None, sample_weight=None, group_id=None, pairs_weight=None,
            baseline=None, use_best_model=None, eval_set=None, verbose=None, logging_level=None, plot=False, column_description=None):
        """
        Fit the CatBoost model.

        Parameters
        ----------
        X : Pool or list or numpy.array or pandas.DataFrame or pandas.Series or string.
            If not Pool, 2 dimensional Feature matrix or string - file with dataset.

        y : list or numpy.array or pandas.DataFrame or pandas.Series, optional (default=None)
            Labels, 1 dimensional array like.
            Use only if X is not Pool.

        cat_features : list or numpy.array, optional (default=None)
            If not None, giving the list of Categ columns indices.
            Use only if X is not Pool.

        pairs : list or numpy.array or pandas.DataFrame
            The pairs description.
            If list or numpy.arrays or pandas.DataFrame, giving 2 dimensional.
            The shape should be Nx2, where N is the pairs' count. The first element of pair is
            the index of winner document in training set. The second element of pair is
            the index of loser document in training set.

        sample_weight : list or numpy.array or pandas.DataFrame or pandas.Series, optional (default=None)
            Instance weights, 1 dimensional array like.

        group_id : list or numpy.array, optional (default=None)
            group id for each instance.
            If not None, giving 1 dimensional array like data.
            Use only if X is not Pool.

        pairs_weight : list or numpy.array, optional (default=None)
            Weight for each pair.
            If not None, giving 1 dimensional array like pairs.

        baseline : list or numpy.array, optional (default=None)
            If not None, giving 2 dimensional array like data.
            Use only if X is not Pool.

        use_best_model : bool, optional (default=None)
            Flag to use best model

        eval_set : Pool or list, optional (default=None)
            A list of (X, y) tuple pairs to use as a validation set for
            early-stopping

        logging_level : string, optional (default=None)
            Possible values:
                - 'Silent'
                - 'Verbose'
                - 'Info'
                - 'Debug'

        verbose : bool, if set to True, logging_level is set to Verbose. If set
            to False, logging_level is set to Silent.

        plot : bool, optional (default=False)
            If True, drow train and eval error in Jupyter notebook

        Returns
        -------
        model : CatBoost
        """
        return self._fit(X, y, cat_features, pairs, sample_weight, group_id, pairs_weight, baseline, use_best_model, eval_set, verbose, logging_level, plot, column_description)

    def _predict(self, data, prediction_type, ntree_start, ntree_end, thread_count, verbose):
        verbose = verbose or self.get_param('verbose')
        if verbose is None:
            verbose = False
        if not self.is_fitted_:
            raise CatboostError("There is no trained model to use predict(). Use fit() to train model. Then use predict().")
        if not isinstance(data, Pool):
            data = Pool(data=data, cat_features=self._get_cat_feature_indices())
        elif not np.all(set(self._get_cat_feature_indices()).issubset(data.get_cat_feature_indices())):
            raise CatboostError("Data cat_features in predict()={} are not equal data cat_features in fit()={}.".format(data.get_cat_feature_indices(), self._get_cat_feature_indices()))
        if data.is_empty_:
            raise CatboostError("Data is empty.")
        if not isinstance(prediction_type, STRING_TYPES):
            raise CatboostError("Invalid prediction_type type={}: must be str().".format(type(prediction_type)))
        if prediction_type not in ('Class', 'RawFormulaVal', 'Probability'):
            raise CatboostError("Invalid value of prediction_type={}: must be Class, RawFormulaVal or Probability.".format(prediction_type))
        loss_function_type = self.get_param('loss_function')
        # TODO(kirillovs): very bad solution. user should be able to use custom multiclass losses
        if loss_function_type is not None and (loss_function_type == 'MultiClass' or loss_function_type == 'MultiClassOneVsAll'):
            return np.transpose(self._base_predict_multi(data, prediction_type, ntree_start, ntree_end, thread_count, verbose))
        predictions = np.array(self._base_predict(data, prediction_type, ntree_start, ntree_end, thread_count, verbose))
        if prediction_type == 'Probability':
            predictions = np.transpose([1 - predictions, predictions])
        return predictions

    def predict(self, data, prediction_type='RawFormulaVal', ntree_start=0, ntree_end=0, thread_count=-1, verbose=None):
        """
        Predict with data.

        Parameters
        ----------
        data : Pool or list or numpy.array or pandas.DataFrame or pandas.Series
            Data to predict.

        prediction_type : string, optional (default='RawFormulaVal')
            Can be:
            - 'RawFormulaVal' : return raw value.
            - 'Class' : return majority vote class.
            - 'Probability' : return probability for every class.

        ntree_start: int, optional (default=0)
            Model is applyed on the interval [ntree_start, ntree_end) (zero-based indexing).

        ntree_end: int, optional (default=0)
            Model is applyed on the interval [ntree_start, ntree_end) (zero-based indexing).
            If value equals to 0 this parameter is ignored and ntree_end equal to tree_count_.

        thread_count : int (default=-1)
            The number of threads to use when applying the model.
            Allows you to optimize the speed of execution. This parameter doesn't affect results.
            If -1, then the number of threads is set to the number of cores.

        verbose : bool, optional (default=False)
            If True, writes the evaluation metric measured set to stderr.

        Returns
        -------
        prediction : numpy.array
        """
        return self._predict(data, prediction_type, ntree_start, ntree_end, thread_count, verbose)

    def _staged_predict(self, data, prediction_type, ntree_start, ntree_end, eval_period, thread_count, verbose):
        verbose = verbose or self.get_param('verbose')
        if verbose is None:
            verbose = False
        if not self.is_fitted_ or self.tree_count_ is None:
            raise CatboostError("There is no trained model to use staged_predict(). Use fit() to train model. Then use staged_predict().")
        if not isinstance(data, Pool):
            data = Pool(data=data, cat_features=self._get_cat_feature_indices())
        elif not np.all(set(self._get_cat_feature_indices()).issubset(data.get_cat_feature_indices())):
            raise CatboostError("Data cat_features in predict()={} are not equal data cat_features in fit()={}.".format(data.get_cat_feature_indices(), self._get_cat_feature_indices()))
        if data.is_empty_:
            raise CatboostError("Data is empty.")
        if not isinstance(prediction_type, STRING_TYPES):
            raise CatboostError("Invalid prediction_type type={}: must be str().".format(type(prediction_type)))
        if prediction_type not in ('Class', 'RawFormulaVal', 'Probability'):
            raise CatboostError("Invalid value of prediction_type={}: must be Class, RawFormulaVal or Probability.".format(prediction_type))
        if ntree_end == 0:
            ntree_end = self.tree_count_
        staged_predict_iterator = self._staged_predict_iterator(data, prediction_type, ntree_start, ntree_end, eval_period, thread_count, verbose)
        loss_function = self.get_param('loss_function')
        while True:
            predictions = staged_predict_iterator.next()
            if loss_function is not None and (loss_function == 'MultiClass' or loss_function == 'MultiClassOneVsAll'):
                predictions = np.transpose(predictions)
            else:
                predictions = np.array(predictions[0])
                if prediction_type == 'Probability':
                    predictions = np.transpose([1 - predictions, predictions])
            yield predictions

    def staged_predict(self, data, prediction_type='RawFormulaVal', ntree_start=0, ntree_end=0, eval_period=1, thread_count=-1, verbose=None):
        """
        Predict target at each stage for data.

        Parameters
        ----------
        data : Pool or list or numpy.array or pandas.DataFrame or pandas.Series
            Data to predict.

        prediction_type : string, optional (default='RawFormulaVal')
            Can be:
            - 'RawFormulaVal' : return raw value.
            - 'Class' : return majority vote class.
            - 'Probability' : return probability for every class.

        ntree_start: int, optional (default=0)
            Model is applyed on the interval [ntree_start, ntree_end) with the step eval_period (zero-based indexing).

        ntree_end: int, optional (default=0)
            Model is applyed on the interval [ntree_start, ntree_end) with the step eval_period (zero-based indexing).
            If value equals to 0 this parameter is ignored and ntree_end equal to tree_count_.

        eval_period: int, optional (default=1)
            Model is applyed on the interval [ntree_start, ntree_end) with the step eval_period (zero-based indexing).

        thread_count : int (default=-1)
            The number of threads to use when applying the model.
            Allows you to optimize the speed of execution. This parameter doesn't affect results.
            If -1, then the number of threads is set to the number of cores.

        verbose : bool
            If True, writes the evaluation metric measured set to stderr.

        Returns
        -------
        prediction : generator numpy.array for each iteration
        """
        return self._staged_predict(data, prediction_type, ntree_start, ntree_end, eval_period, thread_count, verbose)

    def _eval_metrics(self, data, metrics, ntree_start, ntree_end, eval_period, thread_count, tmp_dir):
        if not self.is_fitted_:
            raise CatboostError("There is no trained model to use predict(). Use fit() to train model. Then use predict().")
        if not isinstance(data, Pool):
            data = Pool(data=data, cat_features=self._get_cat_feature_indices())
        elif not np.all(set(self._get_cat_feature_indices()).issubset(data.get_cat_feature_indices())):
            raise CatboostError("Data cat_features in predict()={} are not equal data cat_features in fit()={}.".format(data.get_cat_feature_indices(), self._get_cat_feature_indices()))
        if data.is_empty_:
            raise CatboostError("Data is empty.")
        if not isinstance(metrics, ARRAY_TYPES):
            raise CatboostError("Invalid metrics type={}: must be list().".format(type(metrics)))
        if not all(map(lambda metric: isinstance(metric, string_types), metrics)):
            raise CatboostError("Invalid metric type: must be string().")
        if tmp_dir is None:
            tmp_dir = tempfile.mkdtemp()
        metrics_score = self._base_eval_metrics(data, metrics, ntree_start, ntree_end, eval_period, thread_count, tmp_dir)
        return dict(zip(metrics, metrics_score))

    def eval_metrics(self, data, metrics, ntree_start=0, ntree_end=0, eval_period=1, thread_count=-1, tmp_dir=None):
        """
        Calculate metrics.

        Parameters
        ----------
        data : Pool or list or numpy.array or pandas.DataFrame or pandas.Series
            Data to predict.

        metrics : list of strings
            List of eval metrics.

        ntree_start: int, optional (default=0)
            Model is applyed on the interval [ntree_start, ntree_end) (zero-based indexing).

        ntree_end: int, optional (default=0)
            Model is applyed on the interval [ntree_start, ntree_end) (zero-based indexing).
            If value equals to 0 this parameter is ignored and ntree_end equal to tree_count_.

        eval_period: int, optional (default=1)
            Model is applyed on the interval [ntree_start, ntree_end) with the step eval_period (zero-based indexing).

        thread_count : int (default=-1)
            The number of threads to use when applying the model.
            Allows you to optimize the speed of execution. This parameter doesn't affect results.
            If -1, then the number of threads is set to the number of cores.

        tmp_dir : string (default=None)
            The name of the temporary directory for intermediate results.
            If None, then the name will be generated.

        Returns
        -------
        prediction : dict: metric -> array of shape [(ntree_end - ntree_start) / eval_period]
        """
        return self._eval_metrics(data, metrics, ntree_start, ntree_end, eval_period, thread_count, tmp_dir)

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
        if not self.is_fitted_:
            raise CatboostError("There is no trained model to use predict(). Use fit() to train model. Then use predict().")
        return BatchMetricCalcer(self._object, metrics, ntree_start, ntree_end, eval_period, thread_count, tmp_dir)

    @property
    def feature_importances_(self):
        feature_importances_ = getattr(self, "_feature_importance", None)
        if not self.is_fitted_:
            raise CatboostError("There is no trained model to use `feature_importances_`. Use fit() to train model with param `calc_feature_importance=True`. Then use `feature_importances_`.")
        if feature_importances_ is None:
            raise CatboostError("Invalid attribute `feature_importances_`: use calc_feature_importance=True in model params for use it")
        return feature_importances_

    def get_feature_importance(self, X, y=None, cat_features=None, thread_count=-1, fstr_type='FeatureImportance'):
        """
        Parameters
        ----------
        X : Pool or list or numpy.array or pandas.DataFrame or pandas.Series
            If not Pool, 2 dimensional Feature matrix.

        y : list or numpy.array or pandas.DataFrame or pandas.Series, optional (default=None)
            Labels, 1 dimensional array like.
            Use only if X is not Pool.

        cat_features : list or numpy.array, optional (default=None)
            If not None, giving the list of Categ columns indices.
            Use only if X is not Pool.

        thread_count : int, optional (default=-1)
            Number of threads.
            If -1, then the number of threads is set to the number of cores.

        fstr_type : string, optional (default='FeatureImportance')
            Possible values:
                - FeatureImportance
                    Calculate score for every feature.
                - Interaction
                    Calculate pairwise score between every feature.
                - Doc
                    Calculate score for every feature in every object.

        Returns
        -------
        feature_importances : array of shape = [n_features]
        """
        if fstr_type not in ('FeatureImportance', 'Interaction', 'Doc'):
            raise CatboostError("Invalid feature_importances type = {} : should be one of 'FeatureImportance', 'Interaction', 'Doc'".format(fstr_type))
        if isinstance(X, Pool):
            if X.get_label() is None:
                raise CatboostError("Label in X has not initialized.")
            if y is not None:
                raise CatboostError("Wrong initializing y in feature_importances(): X is Pool object, y must be initialized inside Pool.")
        else:
            if y is None:
                raise CatboostError("y has not initialized in feature_importances(): X is not Pool object, y must be not None in feature_importances().")
            if len(np.shape(X)) == 1 and len(np.shape(y)) == 0:
                X, y = [X], [y]
            if len(np.shape(X)) != 2:
                raise CatboostError("X has invalid shape or empty: {}. Must be 2 dimensional".format(np.shape(X)))
            if len(np.shape(y)) != 1:
                raise CatboostError("y has invalid shape or empty: {}. Must be 1 dimensional".format(np.shape(y)))
            X = Pool(X, y, cat_features=cat_features)
        if X.is_empty_:
            raise CatboostError("X is empty.")
        fstr = self._calc_fstr(X, fstr_type, thread_count)
        if fstr_type == 'FeatureImportance':
            return [value[0] for value in fstr]
        elif fstr_type == 'Doc':
            return np.transpose(fstr)
        return [[int(row[0]), int(row[1]), row[2]] for row in fstr]

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
            raise CatboostError("ntree_start should be less than ntree_end.")
        self._base_shrink(ntree_start, ntree_end)

    def save_model(self, fname, format="cbm", export_parameters=None):
        """
        Save the model to a file.

        Parameters
        ----------
        fname : string
            Output file name.
        format : string
            Either 'cbm' for catboost binary format, or 'coreml' to export into Apple CoreML format.
        export_parameters : dict
            Parameters for CoreML export:
                * prediction_type : string - either 'probability' or 'raw'
                * coreml_description : string
                * coreml_model_version : string
                * coreml_model_author : string
                * coreml_model_license: string
        """
        if not self.is_fitted_:
            raise CatboostError("There is no trained model to use save_model(). Use fit() to train model. Then use save_model().")
        if not isinstance(fname, STRING_TYPES):
            raise CatboostError("Invalid fname type={}: must be str().".format(type(fname)))
        self._save_model(fname, format, export_parameters)

    def load_model(self, fname, format='catboost'):
        """
        Load model from a file.

        Parameters
        ----------
        fname : string
            Input file name.
        """
        if not isinstance(fname, STRING_TYPES):
            raise CatboostError("Invalid fname type={}: must be str().".format(type(fname)))
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
        params = self._get_init_params()
        if deep:
            return deepcopy(params)
        else:
            return params

    def set_params(self, **params):
        """
        Set parameters into CatBoost model.

        Parameters
        ----------
        **params : key=value format
            List of key=value paris. Example: model.set_params(iterations=500, thread_count=2).
        """
        for key, value in iteritems(params):
            self._set_param(key, value)
        return self


class CatBoostClassifier(CatBoost):
    """
    Implementation of the scikit-learn API for CatBoost classification.

    Parameters
    ----------
    iterations : int, [default=500]
        Max count of trees.
        range: [1,+inf]
    learning_rate : float, [default=0.03]
        Step size shrinkage used in update to prevents overfitting.
        range: (0,1]
    depth : int, [default=6]
        Depth of a tree. All trees are the same depth.
        range: [1,+inf]
    l2_leaf_reg : int, [default=3]
        L2 regularization term on weights.
        range: [0,+inf]
    model_size_reg : int, [default=None]
        Model size regularization coefficient.
        range: [0,+inf]
    rsm : float, [default=None]
        Subsample ratio of columns when constructing each tree.
        range: (0,1]
    loss_function : string, [default='Logloss']
        Possible values:
            - 'Logloss'
            - 'CrossEntropy'
            - 'MultiClass'
            - 'MultiClassOneVsAll'
    border_count : int, [default=32]
        The number of partitions for Num features. Used in the preliminary calculation.
        range: (0,+inf]
    feature_border_type : string, [default='MinEntropy']
        Type of binarization target. Used only in Reggression tasks.
        Possible values:
            - 'Median'
            - 'UniformAndQuantiles'
            - 'GreedyLogSum'
            - 'MaxLogSum'
            - 'MinEntropy'
    fold_permutation_block_size : int, [default=1]
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
        Way to process nan-values.
        Possible values:
            - 'Forbidden' - raises an exception if there is nan value in dataset.
            - 'Min' - each nan float feature will be processed as minimum value from dataset.
            - 'Max' - each nan float feature will be processed as maximum value from dataset.
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
    leaf_estimation_method : string, [default='Gradient']
        The method used to calculate the values in the leaves.
        Possible values:
            - 'Newton'
            - 'Gradient'
    thread_count : int, [default=None]
        Number of parallel threads used to run CatBoost.
        If None, then the number of thread is set to the number of cores.
        range: [1,+inf]
    random_seed : int, [default=None]
        Random number seed.
        If None, used random number.
        range: [0,+inf]
    use_best_model : bool, [default=None]
        To limit the number of trees in predict() using information about the optimal value of the error function.
        Can be used only with eval_set.
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
    ctr_leaf_count_limit : int, [default=None]
        The maximum number of leafs with categorical features.
        If the quantity exceeds the specified value a part of leafs is discarded.
        The leafs to be discarded are selected as follows:
            - The leafs are sorted by the frequency of the values.
            - The top N leafs are selected, where N is the value specified in the parameter.
            - All leafs starting from N+1 are discarded.
        This option reduces the resulting model size
        and the amount of memory required for training.
        Note that the resulting quality of the model can be affected.
        range: [1,+inf]
    store_all_simple_ctr : bool, [default=None]
        Ignore categorical features, which are not used in feature combinations,
        when choosing candidates for exclusion.
        Use this parameter with ctr_leaf_count_limit only.
    max_ctr_complexity : int, [default=4]
        The maximum number of Categ features that can be combined.
        range: [0,+inf]
    has_time : bool, [default=False]
        To use the order in which objects are represented in the input data
        (do not perform a random permutation on the stages of converting
        the Categ features to Num and the choice of a tree structure).
    classes_count : int, [default=None]
        The upper limit for the numeric class label.
        Defines the number of classes for multiclassification.
        Only non-negative integers can be specified.
        The given integer should be greater than any of the target values.
        If this parameter is specified the labels for all classes in the input dataset
        should be smaller than the given value.
    class_weights : list of floats, [default=None]
        Classes weights. The values are used as multipliers for the object weights.
        If None, all classes are supposed to have weight one.
        Number of classes indicated by classes_count and class_weights should be the same.
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
    custom_metric : object, [default=None]
        To use your own metric function.
    custom_loss: alias to custom_metric, deprecated and will be removed in future
    eval_metric : string or object, [default=None]
        To optimize your custom metric in loss.
    bagging_temperature : float, [default=None]
        Controls intensity of Bayesian bagging. The higher the temperature the more aggressive bagging is.
        Typical values are in range [0, 1] (0 - no bagging, 1 - default).
    save_snapshot : bool, [default=None]
        Enable progress snapshoting for restoring progress after crashes or interruptions
    snapshot_file : string, [default=None]
        Learn progress snapshot file path, if None will use default filename
    fold_len_multiplier : float, [default=None]
        Fold length multiplier. Should be greater than 1
    used_ram_limit : int, [default=None]
        Try to limit used memory (limit value in bytes).
        WARNING: Currently this option affects CTR memory usage only.
    gpu_ram_part : int, [default=0.95]
        How much of the GPU RAM to use for training.
    allow_writing_files : bool, [default=True]
        If this flag is set to False, no files with different diagnostic info will be created during training.
        With this flag no snapshotting can be done. Plus visualisation will not
        work, because visualisation uses files that are created and updated during training.
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
    devices : string, [default=None], GPU devices to use.
        Format is: '0' for 1 device or '0:1:3' for multiple devices or '0-3' for range of devices.

    bootstrap_type : string, Bayesian, Bernoulli, Poisson.
        Default bootstrap is Bayesian.
        Poisson bootstrap is supported only on GPU.

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
        fold_permutation_block_size=None,
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
        verbose=None,
        logging_level=None,
        metric_period=None,
        ctr_leaf_count_limit=None,
        store_all_simple_ctr=None,
        max_ctr_complexity=None,
        has_time=None,
        classes_count=None,
        class_weights=None,
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
        fold_len_multiplier=None,
        used_ram_limit=None,
        gpu_ram_part=None,
        allow_writing_files=None,
        approx_on_full_history=None,
        boosting_type=None,
        simple_ctr=None,
        combinations_ctr=None,
        per_feature_ctr=None,
        ctr_description=None,
        task_type=None,
        device_config=None,
        devices=None,
        bootstrap_type=None,
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
        **kwargs
    ):
        if objective is not None:
            loss_function = objective
            objective = None

        if isinstance(loss_function, str) and not self._is_classification_loss(loss_function):
            raise CatboostError("Invalid loss_function='{}': for classifier use "
                                "Logloss, CrossEntropy, MultiClass, MultiClassOneVsAll, AUC, Accuracy, Precision, Recall, F1, TotalF1, MCC or custom objective object".format(loss_function))
        params = {}
        params["kwargs"] = kwargs
        not_params = ["not_params", "self", "params", "kwargs", "__class__"]
        for key, value in iteritems(locals().copy()):
            if key not in not_params and value is not None:
                params[key] = value

        self._process_synonyms(params)

        if custom_loss is not None and custom_metric is not None:
            raise CatboostError("Custom loss and custom metric can't be set at the same time. Use custom_metric instead of custom_loss (custom_loss is deprecated)")

        super(CatBoostClassifier, self).__init__(params)

    @property
    def classes_(self):
        return getattr(self, "_classes", None)

    def fit(self, X, y=None, cat_features=None, sample_weight=None, baseline=None, use_best_model=None, eval_set=None, verbose=None, logging_level=None, plot=False, column_description=None):
        """
        Fit the CatBoost model.

        Parameters
        ----------
        X : Pool or list or numpy.array or pandas.DataFrame or pandas.Series
            If not Pool, 2 dimensional Feature matrix or string - file with dataset.

        y : list or numpy.array or pandas.DataFrame or pandas.Series, optional (default=None)
            Labels, 1 dimensional array like.
            Use only if X is not Pool.

        cat_features : list or numpy.array, optional (default=None)
            If not None, giving the list of Categ columns indices.
            Use only if X is not Pool.

        sample_weight : list or numpy.array or pandas.DataFrame or pandas.Series, optional (default=None)
            Instance weights, 1 dimensional array like.

        baseline : list or numpy.array, optional (default=None)
            If not None, giving 2 dimensional array like data.
            Use only if X is not Pool.

        use_best_model : bool, optional (default=None)
            Flag to use best model

        eval_set : Pool or list, optional (default=None)
            A list of (X, y) tuple pairs to use as a validation set for
            early-stopping

        logging_level : string, optional (default=None)
            Possible values:
                - 'Silent'
                - 'Verbose'
                - 'Info'
                - 'Debug'

        plot : bool, optional (default=False)
            If True, drow train and eval error in Jupyter notebook

        Returns
        -------
        model : CatBoost
        """
        self._fit(X, y, cat_features, None, sample_weight, None, None, baseline, use_best_model, eval_set, verbose, logging_level, plot, column_description)
        return self

    def predict(self, data, prediction_type='Class', ntree_start=0, ntree_end=0, thread_count=-1, verbose=None):
        """
        Predict with data.

        Parameters
        ----------
        data : Pool or list or numpy.array or pandas.DataFrame or pandas.Series
            Data to predict.

        prediction_type : string, optional (default='Class')
            Can be:
            - 'RawFormulaVal' : return raw value.
            - 'Class' : return majority vote class.
            - 'Probability' : return probability for every class.

        ntree_start: int, optional (default=0)
            Model is applyed on the interval [ntree_start, ntree_end) (zero-based indexing).

        ntree_end: int, optional (default=0)
            Model is applyed on the interval [ntree_start, ntree_end) (zero-based indexing).
            If value equals to 0 this parameter is ignored and ntree_end equal to tree_count_.

        thread_count : int (default=-1)
            The number of threads to use when applying the model.
            Allows you to optimize the speed of execution. This parameter doesn't affect results.
            If -1, then the number of threads is set to the number of cores.

        verbose : bool, optional (default=False)
            If True, writes the evaluation metric measured set to stderr.

        Returns
        -------
        prediction : numpy.array
        """
        return self._predict(data, prediction_type, ntree_start, ntree_end, thread_count, verbose)

    def predict_proba(self, data, ntree_start=0, ntree_end=0, thread_count=-1, verbose=None):
        """
        Predict class probability with data.

        Parameters
        ----------
        data : Pool or list or numpy.array or pandas.DataFrame or pandas.Series
            Data to predict.

        ntree_start: int, optional (default=0)
            Model is applyed on the interval [ntree_start, ntree_end) (zero-based indexing).

        ntree_end: int, optional (default=0)
            Model is applyed on the interval [ntree_start, ntree_end) (zero-based indexing).
            If value equals to 0 this parameter is ignored and ntree_end equal to tree_count_.

        thread_count : int (default=-1)
            The number of threads to use when applying the model.
            Allows you to optimize the speed of execution. This parameter doesn't affect results.
            If -1, then the number of threads is set to the number of cores.

        verbose : bool
            If True, writes the evaluation metric measured set to stderr.

        Returns
        -------
        prediction : numpy.array
        """
        return self._predict(data, 'Probability', ntree_start, ntree_end, thread_count, verbose)

    def staged_predict(self, data, prediction_type='Class', ntree_start=0, ntree_end=0, eval_period=1, thread_count=-1, verbose=None):
        """
        Predict target at each stage for data.

        Parameters
        ----------
        data : Pool or list or numpy.array or pandas.DataFrame or pandas.Series
            Data to predict.

        prediction_type : string, optional (default='Class')
            Can be:
            - 'RawFormulaVal' : return raw value.
            - 'Class' : return majority vote class.
            - 'Probability' : return probability for every class.

        ntree_start: int, optional (default=0)
            Model is applyed on the interval [ntree_start, ntree_end) with the step eval_period (zero-based indexing).

        ntree_end: int, optional (default=0)
            Model is applyed on the interval [ntree_start, ntree_end) with the step eval_period (zero-based indexing).
            If value equals to 0 this parameter is ignored and ntree_end equal to tree_count_.

        eval_period: int, optional (default=1)
            Model is applyed on the interval [ntree_start, ntree_end) with the step eval_period (zero-based indexing).

        thread_count : int (default=-1)
            The number of threads to use when applying the model.
            Allows you to optimize the speed of execution. This parameter doesn't affect results.
            If -1, then the number of threads is set to the number of cores.

        verbose : bool
            If True, writes the evaluation metric measured set to stderr.

        Returns
        -------
        prediction : generator numpy.array for each iteration
        """
        return self._staged_predict(data, prediction_type, ntree_start, ntree_end, eval_period, thread_count, verbose)

    def staged_predict_proba(self, data, ntree_start=0, ntree_end=0, eval_period=1, thread_count=-1, verbose=None):
        """
        Predict classification target at each stage for data.

        Parameters
        ----------
        data : Pool or list or numpy.array or pandas.DataFrame or pandas.Series
            Data to predict.

        ntree_start: int, optional (default=0)
            Model is applyed on the interval [ntree_start, ntree_end) with the step eval_period (zero-based indexing).

        ntree_end: int, optional (default=0)
            Model is applyed on the interval [ntree_start, ntree_end) with the step eval_period (zero-based indexing).
            If value equals to 0 this parameter is ignored and ntree_end equal to tree_count_.

        eval_period: int, optional (default=1)
            Model is applyed on the interval [ntree_start, ntree_end) with the step eval_period (zero-based indexing).

        thread_count : int (default=-1)
            The number of threads to use when applying the model.
            Allows you to optimize the speed of execution. This parameter doesn't affect results.
            If -1, then the number of threads is set to the number of cores.

        verbose : bool
            If True, writes the evaluation metric measured set to stderr.

        Returns
        -------
        prediction : generator numpy.array for each iteration
        """
        return self._staged_predict(data, 'Probability', ntree_start, ntree_end, eval_period, thread_count, verbose)

    def score(self, X, y):
        """
        Calculate accuracy.

        Parameters
        ----------
        X : Pool or list or numpy.array or pandas.DataFrame or pandas.Series
            Data to predict.
        y : list or numpy.array
            True labels.

        Returns
        -------
        accuracy : float
        """
        correct = []
        y = np.array(y)
        for i, val in enumerate(self.predict(X)):
            correct.append(1 * (y[i] == val))
        return np.mean(correct)


class CatBoostRegressor(CatBoost):
    """
    Implementation of the scikit-learn API for CatBoost regression.

    Parameters
    ----------
    Like in CatBoostClassifier, except loss_function, class_weights and
    classes_count

    loss_function : string, [default='RMSE']
        'RMSE'
        'MAE'
        'Quantile:alpha=value'
        'LogLinQuantile:alpha=value'
        'Poisson'
        'MAPE'
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
        fold_permutation_block_size=None,
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
        verbose=None,
        logging_level=None,
        metric_period=None,
        ctr_leaf_count_limit=None,
        store_all_simple_ctr=None,
        max_ctr_complexity=None,
        has_time=None,
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
        fold_len_multiplier=None,
        used_ram_limit=None,
        gpu_ram_part=None,
        allow_writing_files=None,
        approx_on_full_history=None,
        boosting_type=None,
        simple_ctr=None,
        combinations_ctr=None,
        per_feature_ctr=None,
        ctr_description=None,
        task_type=None,
        device_config=None,
        devices=None,
        bootstrap_type=None,
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
        **kwargs
    ):
        if objective is not None:
            loss_function = objective
            objective = None

        if isinstance(loss_function, str) and self._is_classification_loss(loss_function):
            raise CatboostError("Invalid loss_function={}: for Regressor use RMSE, MAE, Quantile, LogLinQuantile, Poisson, MAPE, R2.".format(loss_function))
        params = {}
        params["kwargs"] = kwargs
        not_params = ["not_params", "self", "params", "kwargs", "__class__"]
        for key, value in iteritems(locals().copy()):
            if key not in not_params and value is not None:
                params[key] = value

        self._process_synonyms(params)

        super(CatBoostRegressor, self).__init__(params)

    def fit(self, X, y=None, cat_features=None, sample_weight=None, baseline=None, use_best_model=None, eval_set=None, verbose=None, logging_level=None, plot=False, column_description=None):
        """
        Fit the CatBoost model.

        Parameters
        ----------
        X : Pool or list or numpy.array or pandas.DataFrame or pandas.Series
            If not Pool, 2 dimensional Feature matrix or string - file with dataset.

        y : list or numpy.array or pandas.DataFrame or pandas.Series, optional (default=None)
            Labels, 1 dimensional array like.
            Use only if X is not Pool.

        cat_features : list or numpy.array, optional (default=None)
            If not None, giving the list of Categ columns indices.
            Use only if X is not Pool.

        sample_weight : list or numpy.array or pandas.DataFrame or pandas.Series, optional (default=None)
            Instance weights, 1 dimensional array like.

        baseline : list or numpy.array, optional (default=None)
            If not None, giving 2 dimensional array like data.
            Use only if X is not Pool.

        use_best_model : bool, optional (default=None)
            Flag to use best model

        eval_set : Pool or list, optional (default=None)
            A list of (X, y) tuple pairs to use as a validation set for
            early-stopping

        logging_level : string, optional (default=None)
            Possible values:
                - 'Silent'
                - 'Verbose'
                - 'Info'
                - 'Debug'

        plot : bool, optional (default=False)
            If True, drow train and eval error in Jupyter notebook

        Returns
        -------
        model : CatBoost
        """
        return self._fit(X, y, cat_features, None, sample_weight, None, None, baseline, use_best_model, eval_set, verbose, logging_level, plot, column_description)

    def predict(self, data, ntree_start=0, ntree_end=0, thread_count=-1, verbose=None):
        """
        Predict with data.

        Parameters
        ----------
        data : Pool or list or numpy.array or pandas.DataFrame or pandas.Series
            Data to predict.

        ntree_start: int, optional (default=0)
            Model is applyed on the interval [ntree_start, ntree_end) (zero-based indexing).

        ntree_end: int, optional (default=0)
            Model is applyed on the interval [ntree_start, ntree_end) (zero-based indexing).
            If value equals to 0 this parameter is ignored and ntree_end equal to tree_count_.

        thread_count : int (default=-1)
            The number of threads to use when applying the model.
            Allows you to optimize the speed of execution. This parameter doesn't affect results.
            If -1, then the number of threads is set to the number of cores.

        verbose : bool
            If True, writes the evaluation metric measured set to stderr.

        Returns
        -------
        prediction : numpy.array
        """
        return self._predict(data, "RawFormulaVal", ntree_start, ntree_end, thread_count, verbose)

    def staged_predict(self, data, ntree_start=0, ntree_end=0, eval_period=1, thread_count=-1, verbose=None):
        """
        Predict target at each stage for data.

        Parameters
        ----------
        data : Pool or list or numpy.array or pandas.DataFrame or pandas.Series
            Data to predict.

        ntree_start: int, optional (default=0)
            Model is applyed on the interval [ntree_start, ntree_end) with the step eval_period (zero-based indexing).

        ntree_end: int, optional (default=0)
            Model is applyed on the interval [ntree_start, ntree_end) with the step eval_period (zero-based indexing).
            If value equals to 0 this parameter is ignored and ntree_end equal to tree_count_.

        eval_period: int, optional (default=1)
            Model is applyed on the interval [ntree_start, ntree_end) with the step eval_period (zero-based indexing).

        thread_count : int (default=-1)
            The number of threads to use when applying the model.
            Allows you to optimize the speed of execution. This parameter doesn't affect results.
            If -1, then the number of threads is set to the number of cores.

        verbose : bool
            If True, writes the evaluation metric measured set to stderr.

        Returns
        -------
        prediction : generator numpy.array for each iteration
        """
        return self._staged_predict(data, "RawFormulaVal", ntree_start, ntree_end, eval_period, thread_count, verbose)

    def score(self, X, y):
        """
        Calculate RMSE.

        Parameters
        ----------
        X : Pool or list or numpy.array or pandas.DataFrame or pandas.Series
            Data to predict.
        y : list or numpy.array
            True labels.

        Returns
        -------
        RMSE : float
        """

        error = []
        y = np.array(y)
        for i, val in enumerate(self.predict(X)):
            error.append(pow(y[i] - val, 2))
        return np.sqrt(np.mean(error))


def train(pool=None, params=None, dtrain=None, logging_level=None, verbose=None, iterations=None, num_boost_round=None, evals=None, eval_set=None, plot=None):
    """
    Train CatBoost model.

    Parameters
    ----------
    pool : Pool or tuple (X, y)
        Data to train on.

    params : dict
        Parameters for CatBoost.
        If  None, all params are set to their defaults.
        If  dict, overriding parameters present in the dict.

    dtrain : Pool or tuple (X, y)
        Synonym for pool parameter. Only one of these parameters should be set.

    logging_level : string, optional (default=None)
        Possible values:
            - 'Silent'
            - 'Verbose'
            - 'Info'
            - 'Debug'

    verbose : bool
        If set to True, then logging_level is set to Verbose, otherwise
        logging_level is set to Silent.

    iterations : int
        Number of boosting iterations. Can be set in params dict.

    num_boost_round : int
        Synonym for iterations. Only one of these parameters should be set.

    eval_set : Pool or tuple (X, y)
        Dataset for evaluation.

    evals : Pool or tuple (X, y)
        Synonym for evals. Only one of these parameters should be set.

    plot : bool, optional (default=False)
        If True, drow train and eval error in Jupyter notebook

    Returns
    -------
    model : CatBoost class
    """

    if params is None:
        raise CatboostError("params should be set.")

    if dtrain is not None:
        if pool is None:
            pool = dtrain
        else:
            raise CatboostError("Only one of the parameters pool and dtrain should be set.")

    if verbose is not None and logging_level is not None:
        raise CatboostError("Only one of the parameters verbose and logging_level should be set.")

    if num_boost_round is not None:
        if iterations is None:
            iterations = num_boost_round
        else:
            raise CatboostError("Only one of the parameters iterations and num_boost_round should be set.")
    if iterations is not None:
        params = deepcopy(params)
        params.update({
            'iterations': iterations
        })

    model = CatBoost(params)

    model.fit(X=pool, eval_set=eval_set, verbose=verbose, logging_level=logging_level, plot=plot)
    return model


def cv(pool=None, params=None, dtrain=None, iterations=None, num_boost_round=None,
       fold_count=3, nfold=None, inverted=False, partition_random_seed=0, seed=None,
       shuffle=True, logging_level=None, stratified=False, as_pandas=True):
    """
    Cross-validate the CatBoost model.

    Parameters
    ----------
    pool : Pool
        Data to cross-validatte.

    params : dict
        Parameters for CatBoost.
        CatBoost has many of parameters, all have default values.
        If  None, all params still defaults.
        If  dict, overriding some (or all) params.

    dtrain : Pool or tuple (X, y)
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

    Returns
    -------
    cv results : pandas.core.frame.DataFrame with cross-validation results
        columns are: test-error-mean  test-error-std  train-error-mean  train-error-std
    """
    if params is None:
        raise CatboostError("params should be set.")

    if "use_best_model" in params:
        warnings.warn('Parameter "use_best_model" has no effect in cross-validation and is ignored')
    if logging_level:
        params = deepcopy(params)
        params.update({
            'logging_level': logging_level
        })

    if dtrain is not None:
        if pool is None:
            pool = dtrain
        else:
            raise CatboostError("Only one of the parameters pool and dtrain should be set.")

    if num_boost_round is not None:
        if iterations is None:
            iterations = num_boost_round
        else:
            raise CatboostError("Only one of the parameters iterations and num_boost_round should be set.")

    if iterations is not None:
        params = deepcopy(params)
        params.update({
            'iterations': iterations
        })

    if seed is not None:
        partition_random_seed = seed

    with log_fixup():
        return _cv(params, pool, fold_count, inverted, partition_random_seed, shuffle, stratified, as_pandas)


class BatchMetricCalcer(_MetricCalcerBase):

    def __init__(self, catboost, metrics, ntree_start, ntree_end, eval_period, thread_count, tmp_dir):
        super(_MetricCalcerBase, self).__init__(catboost)
        if tmp_dir is None:
            tmp_dir = tempfile.mkdtemp()
            delete_temp_dir_flag = True
        else:
            delete_temp_dir_flag = False

        if isinstance(metrics, str):
            metrics = [metrics]
        self._create_calcer(metrics, ntree_start, ntree_end, eval_period, thread_count, tmp_dir, delete_temp_dir_flag)
