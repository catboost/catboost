import sys
from six import iteritems, string_types, integer_types
import os.path
from collections import Iterable, Sequence, Mapping, MutableMapping
import warnings
import numpy as np

try:
    from pandas import DataFrame, Series
except ImportError:
    class DataFrame(object):
        pass

    class Series(object):
        pass

try:
    from _catboost import _PoolBase, _CatBoostBase, CatboostError, _cv, _set_logger, _reset_logger
except ImportError:
    from ._catboost import _PoolBase, _CatBoostBase, CatboostError, _cv, _set_logger, _reset_logger

from contextlib import contextmanager


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
    if isinstance(value, ARRAY_TYPES):
        new_value = []
        for val in value:
            val = _cast_to_base_types(val)
            new_value.append(val)
        return new_value
    if isinstance(value, (Mapping, MutableMapping)):
        new_value = {}
        for k, v in iteritems(value):
            v = _cast_to_base_types(v)
            new_value[k] = v
        return new_value
    if isinstance(value, bool):
        return value
    if isinstance(value, INTEGER_TYPES):
        return int(value)
    if isinstance(value, FLOAT_TYPES):
        return float(value)
    return value


def _is_classification_objective(loss_function):
    return loss_function in ('Logloss', 'CrossEntropy', 'MultiClass')


class Pool(_PoolBase):
    """
    Pool used in CatBoost as data structure to train model from.
    """

    def __init__(self, data, label=None, cat_features=None, column_description=None, delimiter='\t', has_header=False, weight=None, baseline=None, feature_names=None, thread_count=1):
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
            There are only three columns description types: Target, Categ, Num.
            All columns are Num as default, it's not necessary to specify
            this type of columns. Default Target column index is 0 (zero).
            If None, Target column is 0 (zero) as default, all data columns are Num as default.
            If string, giving the path to the file with ColumnsDescription in column_description format.

        delimiter : string, optional (default='\t')
            Delimiter to use for separate features in file.
            Should be only one symbol, otherwise would be taken only the first character of the string.

        has_header : boolm optional (default=False)
            If True, read column names from first line.

        weight : list or numpy.array, optional (default=None)
            Weight for each instance.
            If not None, giving 1 dimensional array like data.

        baseline : list or numpy.array, optional (default=None)
            Baseline for each instance.
            If not None, giving 2 dimensional array like data.
        feature_names : list, optional (default=None)
            Names for each given data_feature.
        thread_count : int
            Thread count to read data from file.
            Use only with reading data from file.
        """
        if data is not None:
            self._check_data_type(data)
            self._check_data_empty(data)
            if isinstance(data, STRING_TYPES):
                self._read(data, column_description, delimiter, has_header, thread_count)
            else:
                self._init(data, label, cat_features, weight, baseline, feature_names)
        super(Pool, self).__init__()

    def _check_files(self, data, column_description):
        """
        Check files existence.
        """
        if not os.path.isfile(data):
            raise CatboostError("Invalid data path='{}': file does not exist.".format(data))
        if column_description is not None and not os.path.isfile(column_description):
            raise CatboostError("Invalid column_description path='{}': file does not exist.".format(column_description))

    def _check_delimiter(self, delimiter):
        if not isinstance(delimiter, STRING_TYPES):
            raise CatboostError("Invalid delimiter type={} : must be str().".format(type(delimiter)))
        if len(delimiter) < 1:
            raise CatboostError("Invalid delimiter length={} : must be > 0.".format(len(delimiter)))

    def _check_column_description_type(self, column_description):
        """
        Check type of ColumnsDescription parameter.
        """
        if not isinstance(column_description, STRING_TYPES):
            raise CatboostError("Invalid column_description type={}: must be str().".format(type(column_description)))

    def _check_cf_type(self, cat_features):
        """
        Check type of cat_feature parameter.
        """
        if not isinstance(cat_features, (list, np.ndarray)):
            raise CatboostError("Invalid cat_features type={}: must be list() or np.ndarray().".format(type(cat_features)))

    def _check_cf_value(self, cat_features):
        """
        Check values in cat_feature parameter. Must be int indices.
        """
        for indx, feature in enumerate(cat_features):
            if not isinstance(feature, INTEGER_TYPES):
                raise CatboostError("Invalid cat_features[{}] = {} value type={}: must be int().".format(indx, feature, type(feature)))

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
            if not len(data_shape) == 2 or not sum(data_shape) > 2:
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

    def _check_label_unique_value(self, label):
        """
        Check count label unique values.
        """
        if len(np.unique(label)) < 2:
            raise CatboostError("Labels has only one unique value.")

    def _check_label_shape(self, label, data_len):
        """
        Check label length and dimension.
        """
        if len(label) != data_len:
            raise CatboostError("Length of label={} and length of data={} is different.".format(len(label), data_len))
        if isinstance(label[0], Iterable) and not isinstance(label[0], STRING_TYPES):
            if len(label[0]) > 1:
                raise CatboostError("Input label cannot have multiple values per row.")

    def _check_baseline_type(self, baseline):
        """
        Check type of baseline parameter.
        """
        if not isinstance(baseline, ARRAY_TYPES):
            raise CatboostError("Invalid baseline type={}: must be array like.".format(type(baseline)))

    def _check_baseline_shape(self, baseline, data_len):
        """
        Check baseline length and dimension.
        """
        if len(baseline) != data_len:
            raise CatboostError("Length of baseline={} and length of data={} are different.".format(len(baseline), data_len))
        if not isinstance(baseline[0], Iterable) or isinstance(baseline[0], STRING_TYPES):
            raise CatboostError("baseline must be 2 dimensional data, 1 column for each class.")
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

    def _check_weight_shape(self, weight, data_len):
        """
        Check weight length.
        """
        if len(weight) != data_len:
            raise CatboostError("Length of weight={} and length of data={} are different.".format(len(weight), data_len))
        if not isinstance(weight[0], (INTEGER_TYPES, FLOAT_TYPES)):
            raise CatboostError("Invalid weight value type={}: must be 1 dimensional data with int, float or long types.".format(type(weight[0])))

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
        if thread_count < 1:
            raise CatboostError("Invalid thread_count value={} : must be > 0".format(thread_count))

    def set_feature_names(self, feature_names):
        self._check_feature_names(feature_names)
        self._set_feature_names(feature_names)
        return self

    def set_baseline(self, baseline):
        self._check_baseline_type(baseline)
        self._check_baseline_shape(baseline, self.num_row())
        self._set_baseline(baseline)
        return self

    def _read(self, pool_file, column_description, delimiter, has_header, thread_count):
        """
        Read Pool from file.
        """
        with log_fixup():
            self._check_files(pool_file, column_description)
            self._check_delimiter(delimiter)
            if column_description is None:
                column_description = ''
            else:
                self._check_column_description_type(column_description)
            self._check_thread_count(thread_count)
            self._read_pool(pool_file, column_description, delimiter[0], has_header, thread_count)

    def _init(self, data_matrix, label, cat_features, weight, baseline, feature_names):
        """
        Initialize Pool from array like data.
        """
        if isinstance(data_matrix, DataFrame):
            feature_names = list(data_matrix.columns)
            data_matrix = data_matrix.values
        if isinstance(data_matrix, Series):
            data_matrix = data_matrix.values
        data_shape = np.shape(data_matrix)
        data_len = data_shape[0]
        if label is not None:
            self._check_label_type(label)
            self._check_label_empty(label)
            if isinstance(label, Series):
                label = label.values
            if isinstance(label, DataFrame):
                label = np.transpose(label.values)[0]
            self._check_label_unique_value(label)
            self._check_label_shape(label, data_len)
        if cat_features is not None:
            self._check_cf_type(cat_features)
            self._check_cf_value(cat_features)
            self._init_cat_features(cat_features)
        if weight is not None:
            self._check_weight_type(weight)
            if isinstance(weight, Series):
                weight = weight.values
            if isinstance(weight, DataFrame):
                weight = np.transpose(weight.values)[0]
            self._check_weight_shape(weight, data_len)
        if baseline is not None:
            self._check_baseline_type(baseline)
            if isinstance(baseline, Series):
                baseline = baseline.values
            if isinstance(baseline, DataFrame):
                baseline = np.transpose(baseline.values)[0]
            self._check_baseline_shape(baseline, data_len)
        if feature_names is not None:
            self._check_feature_names(feature_names, data_shape[1])
        self._init_pool(data_matrix, label, weight, baseline, feature_names)


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
            CatBoost has many of parameters, all have default values.
            If  None, all params still defaults.
            If  dict, overriding some (or all) params.

        model_file : string, optional (default=None)
            If string, giving the path to the file with input model.
        """
        if params is None:
            params = {}
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

    def _fit(self, X, y, cat_features, sample_weight, baseline, use_best_model, eval_set, verbose, plot):
        params = self._get_init_train_params()
        init_params = self.get_init_params()
        calc_feature_importance = False
        if 'calc_feature_importance' in init_params:
            calc_feature_importance = init_params["calc_feature_importance"]
        if verbose is not None:
            params['verbose'] = verbose
        if use_best_model is not None:
            params['use_best_model'] = use_best_model
        if isinstance(X, Pool):
            if X.get_label() is None:
                raise CatboostError("Label in X has not initialized.")
            if y is not None:
                raise CatboostError("Wrong initializing y in fit(): X is Pool object, y must be initialized inside Pool.")
        else:
            if y is None:
                raise CatboostError("y has not initialized in fit(): X is not Pool object, y must be not None in fit().")
            X = Pool(X, y, cat_features=cat_features, weight=sample_weight, baseline=baseline)
        if eval_set is None:
            if self.get_param('use_best_model'):
                raise CatboostError("For use param {'use_best_model': True} need initialize 'eval_set'.")
            eval_set = Pool(None)
        elif not isinstance(eval_set, Pool):
            if len(eval_set) != 2:
                raise CatboostError("Invalid eval_set shape={}: must be (X, y).".format(np.shape(eval_set)))
            eval_set = Pool(eval_set[0], eval_set[1], cat_features=cat_features)
        if X.is_empty():
            raise CatboostError("X is empty.")

        if plot:
            train_dir = self.get_param('train_dir') or '.'

            try:
                from .widget import CatboostIpythonWidget
                widget = CatboostIpythonWidget(train_dir)
                widget.run_update()
            except ImportError as e:
                warnings.warn("For drow plots in fit() method you should install ipywidgets and ipython")
                raise ImportError(str(e))
        with log_fixup():
            self._train(X, eval_set, params)
        if calc_feature_importance:
            setattr(self, "feature_importance", self.feature_importances(X))
        return self

    def fit(self, X, y=None, cat_features=None, sample_weight=None, baseline=None, use_best_model=None, eval_set=None, verbose=None, plot=False):
        """
        Fit the CatBoost model.

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

        sample_weight : list or numpy.array or pandas.DataFrame or pandas.Series, optional (default=None)
            Instance weights, 1 dimensional array like.

        baseline : list or numpy.array, optional (default=None)
            If not None, giving 2 dimensional array like data.
            Use only if X is not Pool.

        use_best_model : bool, optional (default=False)
            Flag to use best model

        eval_set : Pool or list, optional (default=None)
            A list of (X, y) tuple pairs to use as a validation set for
            early-stopping

        verbose : bool, optional (default=False)
            If True, writes the evaluation metric measured set to stderr.

        plot : bool, optional (default=False)
            If True, drow train and eval error in Jupyter notebook

        Returns
        -------
        model : CatBoost
        """
        return self._fit(X, y, cat_features, sample_weight, baseline, use_best_model, eval_set, verbose, plot)

    def _predict(self, data, weight, prediction_type, ntree_limit, verbose):
        verbose = verbose or self.get_param('verbose')
        if verbose is None:
            verbose = False
        if not self.is_fitted():
            raise CatboostError("There is no trained model to use predict(). Use fit() to train model. Then use predict().")
        if not isinstance(data, Pool):
            data = Pool(data=data, weight=weight, cat_features=self._get_cat_feature_indices())
        elif not np.all(data.get_cat_feature_indices() == self._get_cat_feature_indices()):
            raise CatboostError("data cat_features in predict()={} are not equal data cat_features in fit()={}.".format(data.get_cat_feature_indices(), self._get_cat_feature_indices()))
        if data.is_empty():
            raise CatboostError("data is empty.")
        if not isinstance(prediction_type, STRING_TYPES):
            raise CatboostError("Invalid prediction_type type={}: must be str().".format(type(prediction_type)))
        if prediction_type not in ('Class', 'RawFormulaVal', 'Probability'):
            raise CatboostError("Invalid value of prediction_type={}: must be Class, RawFormulaVal or Probability.".format(prediction_type))
        loss_function = self.get_param('loss_function')
        if loss_function is not None and loss_function == 'MultiClass':
            return np.transpose(self._base_predict_multi(data, prediction_type, ntree_limit, verbose))
        predictions = np.array(self._base_predict(data, prediction_type, ntree_limit, verbose))
        if prediction_type == 'Probability':
            predictions = np.transpose([1 - predictions, predictions])
        return predictions

    def predict(self, data, weight=None, prediction_type='RawFormulaVal', ntree_limit=0, verbose=None):
        """
        Predict with data.

        Parameters
        ----------
        data : Pool or list or numpy.array or pandas.DataFrame or pandas.Series
            Data to predict.

        weight : list or numpy.array or pandas.DataFrame or pandas.Series, optional (default=None)
            Instance weights, 1 dimensional array like.

        prediction_type : string, optional (default='RawFormulaVal')
            Can be:
            - 'RawFormulaVal' : return raw value.
            - 'Class' : return majority vote class.
            - 'Probability' : return probability for every class.

        ntree_limit: int, optional (default=0)
            Use first ntree_limit trees for prediction.
            If is set to 0 then all trees from the model are used.
            Defaults to 0.

        verbose : bool, optional (default=False)
            If True, writes the evaluation metric measured set to stderr.

        Returns
        -------
        prediction : numpy.array
        """
        return self._predict(data, weight, prediction_type, ntree_limit, verbose)

    def _staged_predict(self, data, weight, prediction_type, verbose):
        verbose = verbose or self.get_param('verbose')
        if verbose is None:
            verbose = False
        tree_count = self.get_tree_count()
        if not self.is_fitted() or tree_count is None:
            raise CatboostError("There is no trained model to use staged_predict(). Use fit() to train model. Then use staged_predict().")
        for ntree_limit in range(1, tree_count + 1):
            yield self._predict(data, weight, prediction_type, ntree_limit, verbose)

    def staged_predict(self, data, weight=None, prediction_type='RawFormulaVal', verbose=None):
        """
        Predict target at each stage for data.

        Parameters
        ----------
        data : Pool or list or numpy.array or pandas.DataFrame or pandas.Series
            Data to predict.

        weight : list or numpy.array or pandas.DataFrame or pandas.Series, optional (default=None)
            Instance weights, 1 dimensional array like.

        prediction_type : string, optional (default='RawFormulaVal')
            Can be:
            - 'RawFormulaVal' : return raw value.
            - 'Class' : return majority vote class.
            - 'Probability' : return probability for every class.

        verbose : bool
            If True, writes the evaluation metric measured set to stderr.

        Returns
        -------
        prediction : generator numpy.array for each iteration
        """
        return self._staged_predict(data, weight, prediction_type, verbose)

    @property
    def feature_importance_(self):
        feature_importance_ = getattr(self, "feature_importance", None)
        if not self.is_fitted():
            raise CatboostError("There is no trained model to use `feature_importance_`. Use fit() to train model with param `calc_feature_importance=True`. Then use `feature_importance_`.")
        if feature_importance_ is None:
            raise CatboostError("Invalid attribute `feature_importance_`: use calc_feature_importance=True in model params for use it")
        return feature_importance_

    def feature_importances(self, X, y=None, cat_features=None, weight=None, baseline=None, thread_count=1):
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

        weight : list or numpy.array or pandas.DataFrame or pandas.Series, optional (default=None)
            Instance weights, 1 dimensional array like.

        baseline : list or numpy.array, optional (default=None)
            If not None, giving 2 dimensional array like data.
            Use only if X is not Pool.

        thread_count : int, optional (default=1)
            Number of threads.

        Returns
        -------
        feature_importances : array of shape = [n_features]
        """
        if isinstance(X, Pool):
            if X.get_label() is None:
                raise CatboostError("Label in X has not initialized.")
            if y is not None:
                raise CatboostError("Wrong initializing y in feature_importances(): X is Pool object, y must be initialized inside Pool.")
        else:
            if y is None:
                raise CatboostError("y has not initialized in feature_importances(): X is not Pool object, y must be not None in feature_importances().")
            X = Pool(X, y, cat_features=cat_features, weight=weight, baseline=baseline)
        if X.is_empty():
            raise CatboostError("X is empty.")
        return self._calc_fstr(X, thread_count)

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
        if not self.is_fitted():
            raise CatboostError("There is no trained model to use save_model(). Use fit() to train model. Then use save_model().")
        if not isinstance(fname, STRING_TYPES):
            raise CatboostError("Invalid fname type={}: must be str().".format(type(fname)))
        self._save_model(fname, format, export_parameters)

    def load_model(self, fname):
        """
        Load model from a file.

        Parameters
        ----------
        fname : string
            Input file name.
        """
        if not isinstance(fname, STRING_TYPES):
            raise CatboostError("Invalid fname type={}: must be str().".format(type(fname)))
        self._load_model(fname)
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
        return self._get_params()

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
    rsm : float, [default=1]
        Subsample ratio of columns when constructing each tree.
        range: [0,1]
    loss_function : string, [default='Logloss']
        Can be:
        - 'Logloss'
        - 'CrossEntropy'
        - 'MultiClass'
    border : float, [default=None]
        Threshold of positive class.
        range: (0,1)
    border_count : int, [default=32]
        The number of partitions for Num features. Used in the preliminary calculation.
        range: (0,+inf]
    feature_border_type : string, [default='MinEntropy']
        Type of binarization target. Used only in Reggression tasks.
        Can be:
        - 'Median'
        - 'UniformAndQuantiles'
        - 'GreedyLogSum'
        - 'MaxSumLog'
        - 'MinEntropy'
    fold_permutation_block_size : int, [default=1]
        To accelerate the learning.
        The recommended value is within [1, 256]. On small samples, must be set to 1.
        range: [1,+inf]
    auto_stop_pval : float, [default=0]
        Use overfitting detector to stop training when reaching a specified threshold. Can be used only with eval_set.
        range: [0,1]
    gradient_iterations : int, [default=None]
        The number of steps in the gradient when calculating the values in the leaves.
        If None, then gradient_iterations=1.
        range: [1,+inf]
    leaf_estimation_method : string, [default='Gradient']
        The method used to calculate the values in the leaves.
        Can be:
        - 'Newton'
        - 'Gradient'
    thread_count : int, [default=None]
        Number of parallel threads used to run CatBoost.
        If None, then used maximum of the possible threads.
        range: [1,+inf]
    random_seed : int, [default=None]
        Random number seed.
        If None, used random number.
        range: [0,+inf]
    use_best_model : bool, [default=False]
        To limit the number of trees in predict() using information about the optimal value of the error function.
        Can be used only with eval_set.
    verbose : bool, [default=False]
        Whether to print messages while running boosting.
    ctr_description : list of strings, [default=None]
        Binarization settings for categorical features.
        Format :   ['<CTR_type_1>:[<number_of_borders_1>]:[<Binarization_type_1>]',
                    '<CTR_type_2>:[<number_of_borders_2>]:[<Binarization_type_2>]',
                    ... ]
        Example: ['Borders:5:Median', 'MeanValue:10:MinEntropy', ...]
        CTR types:
            - 'Borders'
            - 'Buckets'
            - 'MeanValue'
            - 'CounterTotal'
            - 'CounterMax'
        Number_of_borders and Binarization_type are optional parametrs
        that only used for regression.
            you can fit ctr_description like ['<CTR_type_1>', '<CTR_type_2>', ... ]
            in this case Number_of_borders and Binarization_type are still default.
        The number of borders for target binarization [default=1]:
            - integer values in scope [1, 255].
        The binarization type for the target [default='MinEntropy']:
            - 'Median'
            - 'Uniform'
            - 'UniformAndQuantiles'
            - 'MaxSumLog'
            - 'MinEntropy'
            - 'GreedyLogSum'
    ctr_border_count : int, [default=50]
        The number of partitions for Categ features.
        range: [1,255]
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
    store_all_simple_ctr : bool, [default=False]
        Ignore categorical features, which are not used in feature combinations,
        when choosing candidates for exclusion.
        Use this parameter with ctr_leaf_count_limit only.
    max_ctr_complexity : int, [default=4]
        The maximum number of Categ features that can be combined.
        range: [0,+inf]
    priors : list, [default=None]
        Use priors when training.
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
        Classes are indexed from 0 to classes count - 1.
        For example, in case of binary classification the classes are indexed 0 and 1.
        If None, all classes are supposed to have weight one.
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
    custom_loss : object, [default=None]
        To use your own error function.
    eval_metric : string or object, [default=None]
        To optimize your custom metric in loss.
    class_weights : dict, [default=None]
        Weights associated with classes in the form {class_label: weight}.
        If None, all classes are supposed to have weight one.
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
    feature_priors : list of strings, [default=None]
        You might provide custom per feature priors. They will be used instead of default ones.
        Format is: ['f1Idx:prior1:prior2:prior3', 'f2Idx:prior1']
    """
    def __init__(
        self,
        iterations=500,
        learning_rate=0.03,
        depth=6,
        l2_leaf_reg=3,
        rsm=1,
        loss_function='Logloss',
        border=None,
        border_count=None,
        feature_border_type='MinEntropy',
        fold_permutation_block_size=None,
        auto_stop_pval=0,
        gradient_iterations=None,
        leaf_estimation_method=None,
        thread_count=None,
        random_seed=None,
        use_best_model=False,
        verbose=False,
        ctr_description=None,
        ctr_border_count=None,
        ctr_leaf_count_limit=None,
        store_all_simple_ctr=False,
        max_ctr_complexity=None,
        priors=None,
        has_time=False,
        classes_count=None,
        class_weights=None,
        one_hot_max_size=None,
        random_strength=1,
        name='experiment',
        ignored_features=None,
        train_dir=None,
        custom_loss=None,
        eval_metric=None,
        bagging_temperature=None,
        save_snapshot=None,
        snapshot_file=None,
        fold_len_multiplier=None,
        used_ram_limit=None,
        feature_priors=None,
        **kwargs
    ):
        if isinstance(loss_function, str) and not _is_classification_objective(loss_function):
            raise CatboostError("Invalid loss_fanction='{}': for classifier use "
                                "Logloss, CrossEntropy, MultiClass or custom objective object".format(loss_function))
        params = {}
        params["kwargs"] = kwargs
        not_params = ["not_params", "self", "params", "kwargs", "__class__"]
        for key, value in iteritems(locals()):
            if key not in not_params and value is not None:
                params[key] = value
        super(CatBoostClassifier, self).__init__(params)

    @property
    def classes_(self):
        return getattr(self, "classes", None)

    def fit(self, X, y=None, cat_features=None, sample_weight=None, baseline=None, use_best_model=None, eval_set=None, verbose=None, plot=False):
        """
        Fit the CatBoost model.

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

        sample_weight : list or numpy.array or pandas.DataFrame or pandas.Series, optional (default=None)
            Instance weights, 1 dimensional array like.

        baseline : list or numpy.array, optional (default=None)
            If not None, giving 2 dimensional array like data.
            Use only if X is not Pool.

        use_best_model : bool, optional (default=False)
            Flag to use best model

        eval_set : Pool or list, optional (default=None)
            A list of (X, y) tuple pairs to use as a validation set for
            early-stopping

        verbose : bool, optional (default=False)
            If True, writes the evaluation metric measured set to stderr.

        plot : bool, optional (default=False)
            If True, drow train and eval error in Jupyter notebook

        Returns
        -------
        model : CatBoost
        """
        self._fit(X, y, cat_features, sample_weight, baseline, use_best_model, eval_set, verbose, plot)
        if y is not None:
            setattr(self, "classes", np.unique(y))
        else:
            setattr(self, "classes", np.unique(X.get_label()))
        return self

    def predict(self, data, weight=None, prediction_type='Class', ntree_limit=0, verbose=None):
        """
        Predict with data.

        Parameters
        ----------
        data : Pool or list or numpy.array or pandas.DataFrame or pandas.Series
            Data to predict.

        weight : list or numpy.array or pandas.DataFrame or pandas.Series, optional (default=None)
            Instance weights, 1 dimensional array like.

        prediction_type : string, optional (default='Class')
            Can be:
            - 'RawFormulaVal' : return raw value.
            - 'Class' : return majority vote class.
            - 'Probability' : return probability for every class.

        ntree_limit: int, optional (default=0)
            Use first ntree_limit trees for prediction.
            If is set to 0 then all trees from the model are used.
            Defaults to 0.

        verbose : bool, optional (default=False)
            If True, writes the evaluation metric measured set to stderr.

        Returns
        -------
        prediction : numpy.array
        """
        return self._predict(data, weight, prediction_type, ntree_limit, verbose)

    def predict_proba(self, data, weight=None,  ntree_limit=0, verbose=None):
        """
        Predict class probability with data.

        Parameters
        ----------
        data : Pool or list or numpy.array or pandas.DataFrame or pandas.Series
            Data to predict.

        weight : list or numpy.array or pandas.DataFrame or pandas.Series, optional (default=None)
            Instance weights, 1 dimensional array like.

        ntree_limit: int, optional (default=0)
            Use first ntree_limit trees for prediction.
            If is set to 0 then all trees from the model are used.
            Defaults to 0.

        verbose : bool
            If True, writes the evaluation metric measured set to stderr.

        Returns
        -------
        prediction : numpy.array
        """
        return self._predict(data, weight, 'Probability', ntree_limit, verbose)

    def staged_predict(self, data, weight=None, prediction_type='Class', verbose=None):
        """
        Predict target at each stage for data.

        Parameters
        ----------
        data : Pool or list or numpy.array or pandas.DataFrame or pandas.Series
            Data to predict.

        weight : list or numpy.array or pandas.DataFrame or pandas.Series, optional (default=None)
            Instance weights, 1 dimensional array like.

        prediction_type : string, optional (default='Class')
            Can be:
            - 'RawFormulaVal' : return raw value.
            - 'Class' : return majority vote class.
            - 'Probability' : return probability for every class.

        verbose : bool
            If True, writes the evaluation metric measured set to stderr.

        Returns
        -------
        prediction : generator numpy.array for each iteration
        """
        return self._staged_predict(data, weight, prediction_type, verbose)

    def staged_predict_proba(self, data, weight=None, verbose=None):
        """
        Predict classification target at each stage for data.

        Parameters
        ----------
        data : Pool or list or numpy.array or pandas.DataFrame or pandas.Series
            Data to predict.

        weight : list or numpy.array or pandas.DataFrame or pandas.Series, optional (default=None)
            Instance weights, 1 dimensional array like.

        verbose : bool
            If True, writes the evaluation metric measured set to stderr.

        Returns
        -------
        prediction : generator numpy.array for each iteration
        """
        return self._staged_predict(data, weight, 'Probability', verbose)

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
    Like in CatBoostClassifier, except loss_function.

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
        iterations=500,
        learning_rate=0.03,
        depth=6,
        l2_leaf_reg=3,
        rsm=1,
        loss_function='RMSE',
        border=None,
        border_count=None,
        feature_border_type='MinEntropy',
        fold_permutation_block_size=None,
        auto_stop_pval=0,
        gradient_iterations=None,
        leaf_estimation_method=None,
        thread_count=None,
        random_seed=None,
        use_best_model=False,
        verbose=False,
        ctr_description=None,
        ctr_border_count=None,
        ctr_leaf_count_limit=None,
        max_ctr_complexity=None,
        store_all_simple_ctr=False,
        priors=None,
        has_time=False,
        classes_count=None,
        class_weights=None,
        one_hot_max_size=None,
        random_strength=1,
        name='experiment',
        ignored_features=None,
        train_dir=None,
        custom_loss=None,
        eval_metric=None,
        bagging_temperature=None,
        save_snapshot=None,
        snapshot_file=None,
        fold_len_multiplier=None,
        used_ram_limit=None,
        feature_priors=None,
        **kwargs
    ):
        if _is_classification_objective(loss_function):
            raise CatboostError("Invalid loss_fanction={}: for Regressor use RMSE, MAE, Quantile, LogLinQuantile, Poisson, MAPE.".format(loss_function))
        params = {}
        params["kwargs"] = kwargs
        not_params = ["not_params", "self", "params", "kwargs", "__class__"]
        for key, value in iteritems(locals()):
            if key not in not_params and value is not None:
                params[key] = value
        super(CatBoostRegressor, self).__init__(params)

    def predict(self, data, weight=None, ntree_limit=0, verbose=None):
        """
        Predict with data.

        Parameters
        ----------
        data : Pool or list or numpy.array or pandas.DataFrame or pandas.Series
            Data to predict.

        weight : list or numpy.array or pandas.DataFrame or pandas.Series, optional (default=None)
            Instance weights, 1 dimensional array like.

        ntree_limit: int, optional (default=0)
            Limit number of trees in the prediction; defaults to 0 (use all trees).

        verbose : bool
            If True, writes the evaluation metric measured set to stderr.

        Returns
        -------
        prediction : numpy.array
        """
        return self._predict(data, weight, "RawFormulaVal", ntree_limit, verbose)

    def staged_predict(self, data, weight=None, verbose=None):
        """
        Predict target at each stage for data.

        Parameters
        ----------
        data : Pool or list or numpy.array or pandas.DataFrame or pandas.Series
            Data to predict.

        weight : list or numpy.array or pandas.DataFrame or pandas.Series, optional (default=None)
            Instance weights, 1 dimensional array like.

        verbose : bool
            If True, writes the evaluation metric measured set to stderr.

        Returns
        -------
        prediction : generator numpy.array for each iteration
        """
        return self._staged_predict(data, weight, "RawFormulaVal", verbose)

    def score(self, X, y):
        """
        Calculate MSE.

        Parameters
        ----------
        X : Pool or list or numpy.array or pandas.DataFrame or pandas.Series
            Data to predict.
        y : list or numpy.array
            True labels.

        Returns
        -------
        MSE : float
        """

        error = []
        y = np.array(y)
        for i, val in enumerate(self.predict(X)):
            error.append(pow(y[i] - val, 2))
        return np.mean(error)


def cv(params, pool, fold_count=3, inverted=False, random_seed=0, shuffle=True,
       enable_early_stopping=False, eval_period=1):
    with log_fixup():
        return _cv(params, pool, fold_count, inverted, random_seed, shuffle, enable_early_stopping, eval_period)
