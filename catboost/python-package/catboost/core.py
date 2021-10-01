import sys
from copy import deepcopy
from six import iteritems, string_types, integer_types
import os

if sys.version_info >= (3, 3):
    from collections.abc import Iterable, Sequence, Mapping, MutableMapping
else:
    from collections import Iterable, Sequence, Mapping, MutableMapping

from collections import OrderedDict, defaultdict

import warnings
import numpy as np
import ctypes
import platform
import tempfile
import shutil
from enum import Enum
from operator import itemgetter
from threading import Lock

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

import scipy.sparse


_typeof = type

from .plot_helpers import save_plot_file, try_plot_offline
from . import _catboost
from .metrics import BuiltinMetric

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
is_cv_stratified_objective = _catboost.is_cv_stratified_objective
is_regression_objective = _catboost.is_regression_objective
is_multiregression_objective = _catboost.is_multiregression_objective
is_multitarget_objective = _catboost.is_multitarget_objective
is_survivalregression_objective = _catboost.is_survivalregression_objective
is_groupwise_metric = _catboost.is_groupwise_metric
is_ranking_metric = _catboost.is_ranking_metric
_PreprocessParams = _catboost._PreprocessParams
_check_train_params = _catboost._check_train_params
_MetadataHashProxy = _catboost._MetadataHashProxy
_NumpyAwareEncoder = _catboost._NumpyAwareEncoder
FeaturesData = _catboost.FeaturesData
_have_equal_features = _catboost._have_equal_features
SPARSE_MATRIX_TYPES = _catboost.SPARSE_MATRIX_TYPES
MultiTargetCustomMetric = _catboost.MultiTargetCustomMetric
MultiTargetCustomObjective = _catboost.MultiTargetCustomObjective
MultiRegressionCustomMetric = _catboost.MultiTargetCustomMetric  # for compatibility
MultiRegressionCustomObjective = _catboost.MultiTargetCustomObjective  # for compatibility
fspath = _catboost.fspath
_eval_metric_util = _catboost._eval_metric_util


from contextlib import contextmanager  # noqa E402


_configure_malloc()
_catboost._library_init()

INTEGER_TYPES = (integer_types, np.integer)
FLOAT_TYPES = (float, np.floating)
STRING_TYPES = (string_types,)
ARRAY_TYPES = (list, np.ndarray, DataFrame, Series)

if sys.version_info >= (3, 6):
    PATH_TYPES = STRING_TYPES + (os.PathLike,)
elif sys.version_info >= (3, 4):
    from pathlib import Path
    PATH_TYPES = STRING_TYPES + (Path,)
else:
    PATH_TYPES = STRING_TYPES


class _StreamLikeWrapper:
    def __init__(self, callable_object):
        self.callable_object = callable_object

    def write(self, message):
        self.callable_object(message)


def _get_stream_like_object(obj):
    if hasattr(obj, 'write'):
        return obj
    if hasattr(obj, '__call__'):
        return _StreamLikeWrapper(obj)
    raise CatBoostError(
        'Expected callable object or stream-like object'
    )


@contextmanager
def log_fixup(log_cout=sys.stdout, log_cerr=sys.stderr):
    _set_logger(_get_stream_like_object(log_cout), _get_stream_like_object(log_cerr))
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
    if isinstance(value, tuple(set(PATH_TYPES) - set(STRING_TYPES))):
        return fspath(value)
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
        verbose = int(verbose)

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
    """Calculate most important features explaining difference in predictions for a pair of documents"""
    PredictionDiff = 5
    """Calculate SHAP Interaction Values pairwise between every feature for every object."""
    ShapInteractionValues = 6


class EShapCalcType(Enum):
    """Calculate regular SHAP values"""
    Regular = "Regular"
    """Calculate approximate SHAP values"""
    Approximate = "Approximate"
    """Calculate exact SHAP values"""
    Exact = "Exact"


class EFeaturesSelectionAlgorithm(Enum):
    """Use prediction values change as feature strength, eliminate batch of features at once"""
    RecursiveByPredictionValuesChange = "RecursiveByPredictionValuesChange"
    """Use loss function change as feature strength, eliminate batch of features at each step"""
    RecursiveByLossFunctionChange = "RecursiveByLossFunctionChange"
    """Use shap values to estimate loss function change, eliminate features one by one"""
    RecursiveByShapValues = "RecursiveByShapValues"


def _get_features_indices(features, feature_names):
    """
        Parameters
        ----------
        features :
            must be a sequence of either integers or strings
            if it contains strings 'feature_names' parameter must be defined and string ids from 'features'
            must represent a subset of in 'feature_names'

        feature_names :
            A sequence of string ids for features or None.
            Used to get feature indices for string ids in 'features' parameter
    """
    if (not isinstance(features, (Sequence, np.ndarray))) or isinstance(features, (str, bytes, bytearray)):
        raise CatBoostError("feature names should be a sequence, but got " + repr(features))
    if feature_names is not None:
        return [
            feature_names.index(f) if isinstance(f, STRING_TYPES) else f
            for f in features
        ]
    else:
        for f in features:
            if isinstance(f, STRING_TYPES):
                raise CatBoostError("features parameter contains string value '{}' but feature names "
                                    "for a dataset are not specified".format(f))
    return features


def _update_params_quantize_part(params, ignored_features, per_float_feature_quantization, border_count,
                                 feature_border_type, sparse_features_conflict_fraction, dev_efb_max_buckets,
                                 nan_mode, input_borders, task_type, used_ram_limit, random_seed,
                                 dev_max_subset_size_for_build_borders):
    if ignored_features is not None:
        params.update({
            'ignored_features': ignored_features
        })

    if per_float_feature_quantization is not None:
        params.update({
            'per_float_feature_quantization': per_float_feature_quantization
        })

    if border_count is not None:
        params.update({
            'border_count': border_count
        })

    if feature_border_type is not None:
        params.update({
            'feature_border_type': feature_border_type
        })

    if sparse_features_conflict_fraction is not None:
        params.update({
            'sparse_features_conflict_fraction': sparse_features_conflict_fraction
        })

    if dev_efb_max_buckets is not None:
        params.update({
            'dev_efb_max_buckets': dev_efb_max_buckets
        })

    if nan_mode is not None:
        params.update({
            'nan_mode': nan_mode
        })

    if input_borders is not None:
        params.update({
            'input_borders': input_borders
        })

    if task_type is not None:
        params.update({
            'task_type': task_type
        })

    if used_ram_limit is not None:
        params.update({
            'used_ram_limit': used_ram_limit
        })

    if random_seed is not None:
        params.update({
            'random_seed': random_seed
        })

    if dev_max_subset_size_for_build_borders is not None:
        params.update({
            'dev_max_subset_size_for_build_borders': dev_max_subset_size_for_build_borders
        })

    return params


def plot_features_selection_loss_graph(summary):
    warn_msg = "To draw plots you should install plotly."
    try:
        import plotly.graph_objs as go
    except ImportError as e:
        warnings.warn(warn_msg)
        raise ImportError(str(e))

    eliminated_features = summary['eliminated_features']
    eliminated_features_names = summary['eliminated_features_names']
    names_present = any(eliminated_features_names)
    names_or_indices = eliminated_features_names if names_present else list(map(str, eliminated_features))
    loss_values = summary['loss_graph']['loss_values']
    removed_features_cnt = summary['loss_graph']['removed_features_count']
    main_indices = summary['loss_graph']['main_indices']

    fig = go.Figure()
    color = 'rgb(51,160,44)'
    # line with all points
    fig.add_trace(go.Scatter(
        x=removed_features_cnt,
        y=loss_values,
        line=go.scatter.Line(color=color),
        mode='lines+markers',
        text=[''] + names_or_indices,
        name=''
    ))
    # red markers for main points
    fig.add_trace(go.Scatter(
        x=[removed_features_cnt[idx] for idx in main_indices],
        y=[loss_values[idx] for idx in main_indices],
        mode='markers',
        marker=go.scatter.Marker(size=10, symbol='square'),
        text=[names_or_indices[idx - 1] if idx > 0 else '' for idx in main_indices],
        name=''
    ))
    # labels with features indices
    fig.add_trace(go.Scatter(
        x=removed_features_cnt,
        y=loss_values,
        mode='text',
        text=[''] + list(map(str, eliminated_features)),
        textposition='bottom center',
        textfont=dict(family='sans serif', size=18, color=color),
        name='',
        visible=False
    ))
    if names_present:
        # labels with features names
        fig.add_trace(go.Scatter(
            x=removed_features_cnt,
            y=loss_values,
            mode='text',
            text=[''] + eliminated_features_names,
            textfont=dict(family='sans serif', size=18, color=color),
            textposition='bottom center',
            name='',
            visible=False
        ))
    axis_options = dict(
        gridcolor='rgb(255,255,255)', showgrid=True, showline=False,
        showticklabels=True, tickcolor='rgb(127,127,127)', ticks='outside', zeroline=False
    )
    fig['layout']['xaxis1'].update(title='number of removed features', **axis_options)
    fig['layout']['yaxis1'].update(title='loss value', **axis_options)

    buttons = []
    buttons.append(dict(
        label='Hide features',
        method='update',
        args=[{"visible": [True, True, False, False]}]
    ))
    buttons.append(dict(
        label='Show indices',
        method='update',
        args=[{"visible": [True, True, True, False]}]
    ))
    if names_present:
        buttons.append(dict(
            label='Show names',
            method='update',
            args=[{"visible": [True, True, False, True]}]
        ))

    fig.update_layout(
        updatemenus=[dict(
            active=0,
            buttons=buttons,
            pad={"r": 10, "t": 10},
            showactive=True,
            x=-0.25,
            xanchor="left",
            y=1.03,
            yanchor="top"
        )]
    )

    fig.update_layout(
        showlegend=False
    )

    return fig


class Pool(_PoolBase):
    """
    Pool used in CatBoost as a data structure to train model from.
    """

    def __init__(
        self,
        data,
        label=None,
        cat_features=None,
        text_features=None,
        embedding_features=None,
        column_description=None,
        pairs=None,
        delimiter='\t',
        has_header=False,
        ignore_csv_quoting=False,
        weight=None,
        group_id=None,
        group_weight=None,
        subgroup_id=None,
        pairs_weight=None,
        baseline=None,
        timestamp=None,
        feature_names=None,
        thread_count=-1,
        log_cout=sys.stdout,
        log_cerr=sys.stderr
    ):
        """
        Pool is an internal data structure that is used by CatBoost.
        You can construct Pool from list, numpy.ndarray, pandas.DataFrame, pandas.Series.

        Parameters
        ----------
        data : list or numpy.ndarray or pandas.DataFrame or pandas.Series or FeaturesData or string or pathlib.Path
            Data source of Pool.
            If list or numpy.ndarrays or pandas.DataFrame or pandas.Series, giving 2 dimensional array like data.
            If FeaturesData - see FeaturesData description for details, 'cat_features' and 'feature_names'
              parameters must be equal to None in this case
            If string or pathlib.Path, giving the path to the file with data in catboost format.
              If string starts with "quantized://", the file has to contain quantized dataset saved with Pool.save().

        label : list or numpy.ndarrays or pandas.DataFrame or pandas.Series, optional (default=None)
            Label of the training data.
            If not None, giving 1 or 2 dimensional array like data with floats.
            If data is a file, then label must be in the file, that is label must be equals to None

        cat_features : list or numpy.ndarray, optional (default=None)
            If not None, giving the list of Categ features indices or names.
            If it contains feature names, Pool's feature names must be defined: either by passing 'feature_names'
              parameter or if data is pandas.DataFrame (feature names are initialized from it's column names)
            Must be None if 'data' parameter has FeaturesData type

        text_features : list or numpy.ndarray, optional (default=None)
            If not None, giving the list of Text features indices or names.
            If it contains feature names, Pool's feature names must be defined: either by passing 'feature_names'
              parameter or if data is pandas.DataFrame (feature names are initialized from it's column names)
            Must be None if 'data' parameter has FeaturesData type

        embedding_features : list or numpy.ndarray, optional (default=None)
            If not None, giving the list of Embedding features indices or names.
            If it contains feature names, Pool's feature names must be defined: either by passing 'feature_names'
              parameter or if data is pandas.DataFrame (feature names are initialized from it's column names)
            Must be None if 'data' parameter has FeaturesData type

        column_description : string or pathlib.Path, optional (default=None)
            ColumnsDescription parameter.
            There are several columns description types: Label, Categ, Num, Auxiliary, DocId, Weight, Baseline, GroupId, Timestamp.
            All columns are Num as default, it's not necessary to specify
            this type of columns. Default Label column index is 0 (zero).
            If None, Label column is 0 (zero) as default, all data columns are Num as default.
            If string or pathlib.Path, giving the path to the file with ColumnsDescription in column_description format.

        pairs : list or numpy.ndarray or pandas.DataFrame or string or pathlib.Path
            The pairs description.
            If list or numpy.ndarrays or pandas.DataFrame, giving 2 dimensional.
            The shape should be Nx2, where N is the pairs' count. The first element of the pair is
            the index of winner object in the training set. The second element of the pair is
            the index of loser object in the training set.
            If string or pathlib.Path, giving the path to the file with pairs description.

        delimiter : string, optional (default='\t')
            Delimiter to use for separate features in file.
            Should be only one symbol, otherwise would be taken only the first character of the string.

        has_header : bool optional (default=False)
            If True, read column names from first line.

        ignore_csv_quoting : bool optional (default=False)
            If True ignore quoting '"'.

        weight : list or numpy.ndarray, optional (default=None)
            Weight for each instance.
            If not None, giving 1 dimensional array like data.

        group_id : list or numpy.ndarray, optional (default=None)
            group id for each instance.
            If not None, giving 1 dimensional array like data.

        group_weight : list or numpy.ndarray, optional (default=None)
            Group weight for each instance.
            If not None, giving 1 dimensional array like data.

        subgroup_id : list or numpy.ndarray, optional (default=None)
            subgroup id for each instance.
            If not None, giving 1 dimensional array like data.

        pairs_weight : list or numpy.ndarray, optional (default=None)
            Weight for each pair.
            If not None, giving 1 dimensional array like pairs.

        baseline : list or numpy.ndarray, optional (default=None)
            Baseline for each instance.
            If not None, giving 2 dimensional array like data.

        timestamp: list or numpy.ndarray, optional (default=None)
            Timestamp for each instance.
            Should be a non-negative integer.
            Useful for sorting a learning dataset by this field during training.

        feature_names : list or string or pathlib.Path, optional (default=None)
            If list - list of names for each given data_feature.
            If string or pathlib.Path - path with scheme for feature names data to load.
            If this parameter is None and 'data' is pandas.DataFrame feature names will be initialized
              from DataFrame's column names.
            Must be None if 'data' parameter has FeaturesData type

        thread_count : int, optional (default=-1)
            Thread count for data processing.
            If -1, then the number of threads is set to the number of CPU cores.

        log_cout: output stream or callback for logging

        log_cerr: error stream or callback for logging

        """
        if data is not None:
            self._check_data_type(data)
            self._check_data_empty(data)
            if pairs is not None and isinstance(data, PATH_TYPES) != isinstance(pairs, PATH_TYPES):
                raise CatBoostError("data and pairs parameters should be the same types.")
            if column_description is not None and not isinstance(data, PATH_TYPES):
                raise CatBoostError("data should be the string or pathlib.Path type if column_description parameter is specified.")
            if isinstance(data, PATH_TYPES):
                if any(v is not None for v in [cat_features, text_features, embedding_features, weight, group_id, group_weight,
                                               subgroup_id, pairs_weight, baseline, label]):
                    raise CatBoostError(
                        "cat_features, text_features, embedding_features, weight, group_id, group_weight, subgroup_id, pairs_weight, "
                        "baseline, label should have the None type when the pool is read from the file."
                    )
                if (feature_names is not None) and (not isinstance(feature_names, PATH_TYPES)):
                    raise CatBoostError(
                        "feature_names should have None or string or pathlib.Path type when the pool is read from the file."
                    )
                self._read(data, column_description, pairs, feature_names, delimiter, has_header, ignore_csv_quoting, thread_count,
                           log_cout=log_cout, log_cerr=log_cerr)
            else:
                if isinstance(data, FeaturesData):
                    if any(v is not None for v in [cat_features, text_features, embedding_features, feature_names]):
                        raise CatBoostError(
                            "cat_features, text_features, embedding_features, feature_names should have the None type"
                            " when 'data' parameter has FeaturesData type"
                        )
                elif isinstance(data, np.ndarray):
                    if (data.dtype.kind == 'f') and (cat_features is not None) and (len(cat_features) > 0):
                        raise CatBoostError(
                            "'data' is numpy array of floating point numerical type, it means no categorical features,"
                            " but 'cat_features' parameter specifies nonzero number of categorical features"
                        )
                    if (data.dtype.kind == 'f') and (text_features is not None) and (len(text_features) > 0):
                        raise CatBoostError(
                            "'data' is numpy array of floating point numerical type, it means no text features,"
                            " but 'text_features' parameter specifies nonzero number of text features"
                        )
                    if (data.dtype.kind != 'O') and (embedding_features is not None) and (len(embedding_features) > 0):
                        raise CatBoostError(
                            "'data' is numpy array of non-object type, it means no embedding features,"
                            " but 'embedding_features' parameter specifies nonzero number of embedding features"
                        )
                elif isinstance(data, scipy.sparse.spmatrix):
                    if (data.dtype.kind == 'f') and (cat_features is not None) and (len(cat_features) > 0):
                        raise CatBoostError(
                            "'data' is scipy.sparse.spmatrix of floating point numerical type, it means no categorical features,"
                            " but 'cat_features' parameter specifies nonzero number of categorical features"
                        )
                    if (text_features is not None) and (len(text_features) > 0):
                        raise CatBoostError(
                            "'data' is scipy.sparse.spmatrix, it means no text features,"
                            " but 'text_features' parameter specifies nonzero number of text features"
                        )
                    if (embedding_features is not None) and (len(embedding_features) > 0):
                        raise CatBoostError(
                            "'data' is scipy.sparse.spmatrix, it means no embedding features,"
                            " but 'embedding_features' parameter specifies nonzero number of embedding features"
                        )

                if isinstance(feature_names, PATH_TYPES):
                    raise CatBoostError(
                        "feature_names must be None or have non-string type when the pool is created from "
                        "python objects."
                    )

                self._init(data, label, cat_features, text_features, embedding_features, pairs, weight, group_id, group_weight, subgroup_id, pairs_weight, baseline, timestamp, feature_names, thread_count)
        super(Pool, self).__init__()

    def _check_files(self, data, column_description, pairs):
        """
        Check files existence.
        """
        data = fspath(data)
        column_description = fspath(column_description)
        pairs = fspath(pairs)
        if data.find('://') == -1 and not os.path.isfile(data):
            raise CatBoostError("Invalid data path='{}': file does not exist.".format(data))
        if column_description is not None and column_description.find('://') == -1 and not os.path.isfile(column_description):
            raise CatBoostError("Invalid column_description path='{}': file does not exist.".format(column_description))
        if pairs is not None and pairs.find('://') == -1 and not os.path.isfile(pairs):
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
        if not isinstance(column_description, PATH_TYPES):
            raise CatBoostError("Invalid column_description type={}: must be str() or pathlib.Path().".format(type(column_description)))

    def _check_string_feature_type(self, features, features_name):
        """
        Check type of cat_feature parameter.
        """
        if not isinstance(features, (list, np.ndarray)):
            raise CatBoostError("Invalid {} type={}: must be list() or np.ndarray().".format(features_name, type(features)))

    def _check_string_feature_value(self, features, features_count, features_name):
        """
        Check values in cat_feature parameter. Must be int indices.
        """
        for indx, feature in enumerate(features):
            if not isinstance(feature, INTEGER_TYPES):
                raise CatBoostError("Invalid {}[{}] = {} value type={}: must be int().".format(features_name, indx, feature, type(feature)))
            if feature >= features_count:
                raise CatBoostError("Invalid {}[{}] = {} value: index must be < {}.".format(features_name, indx, feature, features_count))

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

    def _check_data_type(self, data):
        """
        Check type of data.
        """
        if not isinstance(data, (PATH_TYPES, ARRAY_TYPES, SPARSE_MATRIX_TYPES, FeaturesData)):
            raise CatBoostError(
                "Invalid data type={}: data must be list(), np.ndarray(), DataFrame(), Series(), FeaturesData " +
                " scipy.sparse matrix or filename str() or pathlib.Path().".format(type(data))
            )

    def _check_data_empty(self, data):
        """
        Check that data is not empty (0 objects is ok).
        note: already checked if data is FeatureType, so no need to check again
        """

        if isinstance(data, PATH_TYPES):
            if not data:
                raise CatBoostError("Features filename is empty.")
        elif isinstance(data, (ARRAY_TYPES, SPARSE_MATRIX_TYPES)):
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

    def _check_timestamp_type(self, timestamp):
        """
        Check type of timestamp parameter.
        """
        if not isinstance(timestamp, ARRAY_TYPES):
            raise CatBoostError("Invalid timestamp type={}: must be array like.".format(type(timestamp)))

    def _check_timestamp_shape(self, timestamp, samples_count):
        """
        Check timestamp length.
        """
        if len(timestamp) != samples_count:
            raise CatBoostError("Length of timestamp={} and length of data={} are different.".format(len(timestamp), samples_count))


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
            raise CatBoostError("Invalid rindex type={} : must be list or numpy.ndarray".format(type(rindex)))
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

    def set_timestamp(self, timestamp):
        self._check_timestamp_type(timestamp)
        timestamp = self._if_pandas_to_numpy(timestamp)
        self._check_timestamp_shape(timestamp, self.num_row())
        self._set_timestamp(timestamp)
        return self

    def save(self, fname):
        """
        Save the quantized pool to a file.

        Parameters
        ----------
        fname : string or pathlib.Path
            Output file name.
        """
        if not self.is_quantized():
            raise CatBoostError('Pool is not quantized')

        if not isinstance(fname, PATH_TYPES):
            raise CatBoostError("Invalid fname type={}: must be str() or pathlib.Path().".format(type(fname)))

        self._save(fname)

    def quantize(self, ignored_features=None, per_float_feature_quantization=None, border_count=None,
                 max_bin=None, feature_border_type=None, sparse_features_conflict_fraction=None,
                 nan_mode=None, input_borders=None, task_type=None, used_ram_limit=None, random_seed=None, **kwargs):
        """
        Quantize this pool

        Parameters
        ----------
        pool : catboost.Pool
            Dataset to quantize.

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

        sparse_features_conflict_fraction : float, [default=0.0]
            CPU only. Maximum allowed fraction of conflicting non-default values for features in exclusive features bundle.
            Should be a real value in [0, 1) interval.

        nan_mode : string, [default=None]
            Way to process missing values for numeric features.
            Possible values:
                - 'Forbidden' - raises an exception if there is a missing value for a numeric feature in a dataset.
                - 'Min' - each missing value will be processed as the minimum numerical value.
                - 'Max' - each missing value will be processed as the maximum numerical value.
            If None, then nan_mode=Min.

        input_borders : string or pathlib.Path, [default=None]
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
        """
        if self.is_quantized():
            raise CatBoostError('Pool is already quantized')

        params = {}
        _process_synonyms(params)

        if border_count is None:
            border_count = max_bin

        dev_efb_max_buckets = kwargs.pop('dev_efb_max_buckets', None)
        dev_max_subset_size_for_build_borders = kwargs.pop('dev_max_subset_size_for_build_borders', None)

        if kwargs:
            raise CatBoostError("got an unexpected keyword arguments: {}".format(kwargs.keys()))

        _update_params_quantize_part(params, ignored_features, per_float_feature_quantization, border_count,
                                     feature_border_type, sparse_features_conflict_fraction, dev_efb_max_buckets,
                                     nan_mode, input_borders, task_type, used_ram_limit, random_seed,
                                     dev_max_subset_size_for_build_borders)

        self._quantize(params)

    def _if_pandas_to_numpy(self, array):
        if isinstance(array, Series):
            array = array.values
        if isinstance(array, DataFrame):
            array = np.transpose(array.values)[0]
        return array

    def _label_if_pandas_to_numpy(self, label):
        if isinstance(label, Series):
            label = label.values
        if isinstance(label, DataFrame):
            label = label.values
        return label

    def _read(
        self,
        pool_file,
        column_description,
        pairs,
        feature_names_path,
        delimiter,
        has_header,
        ignore_csv_quoting,
        thread_count,
        quantization_params=None,
        log_cout=sys.stdout,
        log_cerr=sys.stderr
    ):
        """
        Read Pool from file.
        """
        with log_fixup(log_cout, log_cerr):
            self._check_files(pool_file, column_description, pairs)
            self._check_delimiter(delimiter)
            if column_description is None:
                column_description = ''
            else:
                self._check_column_description_type(column_description)
            if pairs is None:
                pairs = ''
            if feature_names_path is None:
                feature_names_path = ''
            self._check_thread_count(thread_count)
            self._read_pool(
                pool_file,
                column_description,
                pairs,
                feature_names_path,
                delimiter[0],
                has_header,
                ignore_csv_quoting,
                thread_count,
                quantization_params
            )

    def _init(
        self,
        data,
        label,
        cat_features,
        text_features,
        embedding_features,
        pairs, weight,
        group_id,
        group_weight,
        subgroup_id,
        pairs_weight,
        baseline,
        timestamp,
        feature_names,
        thread_count
    ):
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
            label = self._label_if_pandas_to_numpy(label)
            if len(np.shape(label)) == 1:
                label = np.expand_dims(label, 1)
            self._check_label_shape(label, samples_count)
        if feature_names is not None:
            self._check_feature_names(feature_names, features_count)
        if cat_features is not None:
            cat_features = _get_features_indices(cat_features, feature_names)
            self._check_string_feature_type(cat_features, 'cat_features')
            self._check_string_feature_value(cat_features, features_count, 'cat_features')
        if text_features is not None:
            text_features = _get_features_indices(text_features, feature_names)
            self._check_string_feature_type(text_features, 'text_features')
            self._check_string_feature_value(text_features, features_count, 'text_features')
        if embedding_features is not None:
            embedding_features = _get_features_indices(embedding_features, feature_names)
            self._check_string_feature_type(embedding_features, 'embedding_features')
            self._check_string_feature_value(embedding_features, features_count, 'embedding_features')
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
        if timestamp is not None:
            self._check_timestamp_type(timestamp)
            timestamp = self._if_pandas_to_numpy(timestamp)
            self._check_timestamp_shape(timestamp, samples_count)
        self._init_pool(data, label, cat_features, text_features, embedding_features, pairs, weight, group_id, group_weight, subgroup_id, pairs_weight, baseline, timestamp, feature_names, thread_count)


def _build_train_pool(X, y, cat_features, text_features, embedding_features, pairs, sample_weight, group_id, group_weight, subgroup_id, pairs_weight, baseline, column_description):
    train_pool = None
    if isinstance(X, Pool):
        train_pool = X
        if any(v is not None for v in [cat_features, text_features, embedding_features, sample_weight, group_id, group_weight, subgroup_id, pairs_weight, baseline]):
            raise CatBoostError("cat_features, text_features, embedding_features, sample_weight, group_id, group_weight, subgroup_id, pairs_weight, baseline should have the None type when X has catboost.Pool type.")
        if (not X.has_label()) and X.num_pairs() == 0:
            raise CatBoostError("Label in X has not been initialized.")
        if y is not None:
            raise CatBoostError("Incorrect value of y: X is catboost.Pool object, y must be initialized inside catboost.Pool.")
    elif isinstance(X, PATH_TYPES):
        train_pool = Pool(data=X, pairs=pairs, column_description=column_description)
    else:
        if y is None:
            raise CatBoostError("y has not initialized in fit(): X is not catboost.Pool object, y must be not None in fit().")
        train_pool = Pool(X, y, cat_features=cat_features, text_features=text_features, embedding_features=embedding_features, pairs=pairs, weight=sample_weight, group_id=group_id,
                          group_weight=group_weight, subgroup_id=subgroup_id, pairs_weight=pairs_weight, baseline=baseline)
    return train_pool


def _clear_training_files(train_dir):
    for filename in ['catboost_training.json']:
        path = os.path.join(train_dir, filename)
        if os.path.exists(path):
            os.remove(path)


def _get_train_dir(params):
    return params.get('train_dir', 'catboost_info')


def _get_catboost_widget(train_dirs):
    for train_dir in train_dirs:
        _clear_training_files(train_dir)
    try:
        from .widget import MetricVisualizer
        return MetricVisualizer(train_dirs)
    except ImportError as e:
        warnings.warn("To draw plots in fit() method you should install ipywidgets and ipython")
        raise ImportError(str(e))


@contextmanager
def plot_wrapper(plot, train_dirs):
    if plot:
        widget = _get_catboost_widget(train_dirs)
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

def _process_synonyms_groups(params):
    _process_synonyms_group(['learning_rate', 'eta'], params)
    _process_synonyms_group(['border_count', 'max_bin'], params)
    _process_synonyms_group(['depth', 'max_depth'], params)
    _process_synonyms_group(['rsm', 'colsample_bylevel'], params)
    _process_synonyms_group(['random_seed', 'random_state'], params)
    _process_synonyms_group(['l2_leaf_reg', 'reg_lambda'], params)
    _process_synonyms_group(['iterations', 'n_estimators', 'num_boost_round', 'num_trees'], params)
    _process_synonyms_group(['od_wait', 'early_stopping_rounds'], params)
    _process_synonyms_group(['custom_metric', 'custom_loss'], params)
    _process_synonyms_group(['max_leaves', 'num_leaves'], params)
    _process_synonyms_group(['min_data_in_leaf', 'min_child_samples'], params)

def _process_synonyms(params):
    if 'objective' in params:
        params['loss_function'] = params['objective']
        del params['objective']

    if 'scale_pos_weight' in params:
        if 'loss_function' in params and params['loss_function'] != 'Logloss':
                raise CatBoostError('scale_pos_weight is supported only for binary classification Logloss loss')
        if 'class_weights' in params or 'auto_class_weights' in params:
            raise CatBoostError('only one of the parameters scale_pos_weight, class_weights, auto_class_weights should be initialized.')
        params['class_weights'] = [1.0, params['scale_pos_weight']]
        del params['scale_pos_weight']
    if ('class_weights' in params) and isinstance(params['class_weights'], (dict, OrderedDict)):
        class_weights_dict = params['class_weights']
        class_weights_list = []
        if ('class_names' in params) and (params['class_names'] is not None):
            if len(class_weights_dict) != len(params['class_names']):
                raise CatBoostError('Number of classes in class_names and class_weights differ')
            for class_label in params['class_names']:
                if class_label not in class_weights_dict:
                    raise CatBoostError(
                        'class "{}" is present in "class_names" but not in "class_weights" dictionary'.format(
                            class_label
                        )
                    )
                class_weights_list.append(class_weights_dict[class_label])
        else:
            class_labels_list = []
            for class_label, class_weight in class_weights_dict.items():
                class_labels_list.append(class_label)
                class_weights_list.append(class_weight)
            params['class_names'] = class_labels_list
        params['class_weights'] = class_weights_list

    _process_synonyms_groups(params)

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

    metric_period, verbose, logging_level = _process_verbose(
        metric_period, verbose, logging_level, verbose_eval, silent)

    if metric_period is not None:
        params['metric_period'] = metric_period
    if verbose is not None:
        params['verbose'] = verbose
    if logging_level is not None:
        params['logging_level'] = logging_level

    if 'used_ram_limit' in params:
        params['used_ram_limit'] = str(params['used_ram_limit'])

def stringify_builtin_metrics(params):
    """Replace all occurrences of BuiltinMetric with their string representations."""
    for f in [
            "loss_function",
            "objective",
            "eval_metric",
            "custom_metric",
            "custom_loss"
        ]:
        if f not in params:
            continue
        val = params[f]
        if isinstance(val, BuiltinMetric):
            params[f] = str(val)
        elif isinstance(val, STRING_TYPES):
            continue
        elif isinstance(val, Sequence):
            params[f] = stringify_builtin_metrics_list(val)
    return params


def stringify_builtin_metrics_list(metrics):
    return list(map(str, metrics))

def _get_loss_function_for_train(params, estimator_type, train_pool):
    """
        estimator_type must be 'classifier', 'regressor', 'ranker' or None
        train_pool must be Pool
    """

    loss_function_param = params.get('loss_function')
    if loss_function_param is not None:
        return loss_function_param

    if estimator_type == 'classifier':
        if not isinstance(train_pool, Pool):
            raise CatBoostError('train_pool param must have Pool type')

        label = train_pool.get_label()
        if label is None:
            raise CatBoostError('loss function has not been specified and cannot be deduced')

        """
            len(set) is faster than np.unique on Python lists:
             https://bbengfort.github.io/observations/2017/05/02/python-unique-benchmark.html
        """
        is_multiclass_task = len(set(label)) > 2 and 'target_border' not in params
        return 'MultiClass' if is_multiclass_task else 'Logloss'
    elif estimator_type == 'ranker':
        return 'YetiRank'
    else:
        return 'RMSE'


class _CatBoostBase(object):
    def __init__(self, params):
        init_params = params.copy() if params is not None else {}
        stringify_builtin_metrics(init_params)
        self._init_params = init_params
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
        for attr in ['_prediction_values_change', '_loss_value_change']:
            if getattr(self, attr, None) is not None:
                params[attr] = getattr(self, attr, None)
        return params

    def __setstate__(self, state):
        if '_object' not in dict(self.__dict__.items()):
            self._object = _CatBoost()
        if '_init_params' not in dict(self.__dict__.items()):
            self._init_params = {}
        if '__model' in state:
            self._load_from_string(state['__model'])
            del state['__model']
        if '_test_eval' in state:
            self._set_test_evals([state['_test_eval']])
            del state['_test_eval']
        if '_test_evals' in state:
            self._set_test_evals(state['_test_evals'])
            del state['_test_evals']
        for attr in ['_prediction_values_change', '_loss_value_change']:
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

    def __ne__(self, other):
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
        setattr(self, '_is_fitted_', True)  # compatibility with meta algorithms in sklearn
        setattr(self, '_random_seed', self._object._get_random_seed())
        setattr(self, '_learning_rate', self._object._get_learning_rate())
        setattr(self, '_tree_count', self._object._get_tree_count())
        setattr(self, '_n_features_in', self._object._get_n_features_in())

    def _train(self, train_pool, test_pool, params, allow_clear_pool, init_model):
        self._object._train(train_pool, test_pool, params, allow_clear_pool, init_model._object if init_model else None)
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

    def get_n_features_in(self):
        return self._object._get_n_features_in()

    def _get_float_feature_indices(self):
        return self._object._get_float_feature_indices()

    def _get_cat_feature_indices(self):
        return self._object._get_cat_feature_indices()

    def _get_text_feature_indices(self):
        return self._object._get_text_feature_indices()

    def _get_embedding_feature_indices(self):
        return self._object._get_embedding_feature_indices()

    def _base_predict(self, pool, prediction_type, ntree_start, ntree_end, thread_count, verbose, task_type):
        return self._object._base_predict(pool, prediction_type, ntree_start, ntree_end, thread_count, verbose, task_type)

    def _base_virtual_ensembles_predict(self, pool, prediction_type, ntree_end, virtual_ensembles_count, thread_count, verbose):
        return self._object._base_virtual_ensembles_predict(pool, prediction_type, ntree_end, virtual_ensembles_count, thread_count, verbose)

    def _staged_predict_iterator(self, pool, prediction_type, ntree_start, ntree_end, eval_period, thread_count, verbose):
        return self._object._staged_predict_iterator(pool, prediction_type, ntree_start, ntree_end, eval_period, thread_count, verbose)

    def _leaf_indexes_iterator(self, pool, ntree_start, ntree_end):
        return self._object._leaf_indexes_iterator(pool, ntree_start, ntree_end)

    def _base_calc_leaf_indexes(self, pool, ntree_start, ntree_end, thread_count, verbose):
        return self._object._base_calc_leaf_indexes(pool, ntree_start, ntree_end, thread_count, verbose)

    def _base_eval_metrics(self, pool, metrics_description, ntree_start, ntree_end, eval_period, thread_count, result_dir, tmp_dir):
        metrics_description_list = metrics_description if isinstance(metrics_description, list) else [metrics_description]
        return self._object._base_eval_metrics(pool, metrics_description_list, ntree_start, ntree_end, eval_period, thread_count, result_dir, tmp_dir)

    def _calc_fstr(self, type, pool, reference_data, thread_count, verbose, model_output, shap_mode, interaction_indices, shap_calc_type):
        return self._object._calc_fstr(type.name, pool, reference_data, thread_count, verbose, model_output, shap_mode, interaction_indices, shap_calc_type)

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
        if not isinstance(model_file, PATH_TYPES):
            raise CatBoostError("Invalid fname type={}: must be str() or pathlib.Path().".format(type(model_file)))

        self._object._load_model(model_file, format)
        self._set_trained_model_attributes()
        for key, value in iteritems(self._get_params()):
            self._init_params[key] = value

    def _serialize_model(self):
        return self._object._serialize_model()

    def _deserialize_model(self, dump_model_str):
        assert isinstance(dump_model_str, bytes), "Not bytes passed as argument"
        self._object._deserialize_model(dump_model_str)

    def _load_from_string(self, dump_model_str):
        self._deserialize_model(dump_model_str)
        self._set_trained_model_attributes()

    def _load_from_stream(self, stream):
        self._object._load_from_stream(stream)
        self._set_trained_model_attributes()

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

    def _get_borders(self):
        if not self.is_fitted():
            raise CatBoostError("There is no trained model to use get_feature_borders(). Use fit() to train model. Then use get_borders().")
        return self._object._get_borders()

    def _get_nan_treatments(self):
        assert self.is_fitted()
        return self._object._get_nan_treatments()

    def _get_params(self):
        params = self._object._get_params()
        init_params = self._init_params.copy()
        for key, value in iteritems(init_params):
            if key not in params:
                params[key] = value
        return params

    @staticmethod
    def _is_classification_objective(loss_function):
        return isinstance(loss_function, str) and is_classification_objective(loss_function)

    @staticmethod
    def _is_regression_objective(loss_function):
        return isinstance(loss_function, str) and is_regression_objective(loss_function)

    @staticmethod
    def _is_multiregression_objective(loss_function):
        return isinstance(loss_function, str) and is_multiregression_objective(loss_function)

    @staticmethod
    def _is_multitarget_objective(loss_function):
        return isinstance(loss_function, str) and is_multitarget_objective(loss_function)

    @staticmethod
    def _is_survivalregression_objective(loss_function):
        return isinstance(loss_function, str) and is_survivalregression_objective(loss_function)

    @staticmethod
    def _is_ranking_objective(loss_function):
        return isinstance(loss_function, str) and is_ranking_metric(loss_function)

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
    def n_features_in_(self):
        return getattr(self, '_n_features_in') if self.is_fitted() else None

    @property
    def feature_names_(self):
        return self._object._get_feature_names() if self.is_fitted() else None

    @property
    def classes_(self):
        return self._object._get_class_labels() if self.is_fitted() else None

    @property
    def evals_result_(self):
        return self.get_evals_result()

    @property
    def best_score_(self):
        return self.get_best_score()

    @property
    def best_iteration_(self):
        return self.get_best_iteration()

    def _get_tree_splits(self, tree_idx, pool):
        return self._object._get_tree_splits(tree_idx, pool)

    def _get_tree_leaf_values(self, tree_idx):
        return self._object._get_tree_leaf_values(tree_idx)

    def _get_tree_step_nodes(self, tree_idx):
        return self._object._get_tree_step_nodes(tree_idx)

    def _get_tree_node_to_leaf(self, tree_idx):
        return self._object._get_tree_node_to_leaf(tree_idx)

    def get_tree_leaf_counts(self):
        '''
        Returns
        -------
        tree_leaf_counts : 1d-array of numpy.uint32 of size tree_count_.
        tree_leaf_counts[i] equals to the number of leafs in i-th tree of the ensemble.
        '''
        return self._object._get_tree_leaf_counts()

    def get_leaf_values(self):
        '''
        Returns
        -------
        leaf_values : 1d-array of leaf values for all trees.
        Value corresponding to j-th leaf of i-th tree is at position
        sum(get_tree_leaf_counts()[:i]) + j (leaf and tree indexing starts from zero).
        '''
        return self._object._get_leaf_values()

    def get_leaf_weights(self):
        '''
        Returns
        -------
        leaf_weights : 1d-array of leaf weights for all trees.
        Weight of j-th leaf of i-th tree is at position
        sum(get_tree_leaf_counts()[:i]) + j (leaf and tree indexing starts from zero).
        '''
        return self._object._get_leaf_weights()

    def set_leaf_values(self, new_leaf_values):
        '''
        Sets values at tree leafs of ensemble equal to new_leaf_values.

        Parameters
        ----------
        new_leaf_values : 1d-array with new leaf values for all trees.
        It's size should be equal to sum(get_tree_leaf_counts()).
        Value corresponding to j-th leaf of i-th tree should be at position
        sum(get_tree_leaf_counts()[:i]) + j (leaf and tree indexing starts from zero).
        '''
        self._object._set_leaf_values(new_leaf_values)

    def set_feature_names(self, feature_names):
        '''
        Sets feature names equal to feature_names

        Parameters
        ----------
        feature_names: 1-d array of strings with new feature names in the same order as in pool
        '''
        self._object._set_feature_names(feature_names)


    def _get_tags(self):
        tags = {
            'requires_positive_X': False,
            'requires_positive_y': False,
            'requires_y': True,
            'poor_score': False,
            'no_validation': True,
            'stateless': False,
            'pairwise': False,
            'multilabel': False,
            '_skip_test': False,
            'multioutput_only': False,
            'binary_only': False,
            'requires_fit': True}

        params = deepcopy(self._init_params)
        if params is None:
            params = {}
        _process_synonyms(params)

        tags['non_deterministic'] = 'task_type' in params and params['task_type'] == 'GPU'
        loss_function = params.get('loss_function', '')
        tags['multioutput'] = (loss_function == 'MultiRMSE' or loss_function == 'RMSEWithUncertainty')
        tags['allow_nan'] = 'nan_mode' not in params or params['nan_mode'] != 'Forbidden'

        return tags

    def get_scale_and_bias(self):
        return self._object._get_scale_and_bias()

    def set_scale_and_bias(self, scale, bias):
        if isinstance(bias, FLOAT_TYPES):
            self._object._set_scale_and_bias(scale, [bias])
        else:
            self._object._set_scale_and_bias(scale, bias)


def _cast_value_to_list_of_strings(params, key):
    if key in params:
        if isinstance(params[key], STRING_TYPES):
            params[key] = [params[key]]
        if not isinstance(params[key], Sequence):
            raise CatBoostError("Invalid `" + key + "` type={} : must be string or list of strings.".format(type(params[key])))


def _check_param_types(params):
    if not isinstance(params, (Mapping, MutableMapping)):
        raise CatBoostError("Invalid params type={}: must be dict().".format(type(params)))
    if 'ctr_description' in params:
        if not isinstance(params['ctr_description'], Sequence):
            raise CatBoostError("Invalid ctr_description type={} : must be list of strings".format(type(params['ctr_description'])))
    if 'ctr_target_border_count' in params:
        if not isinstance(params['ctr_target_border_count'], INTEGER_TYPES):
            raise CatBoostError('Invalid ctr_target_border_count type={} : must be integer type'.format(type(params['ctr_target_border_count'])))
    _cast_value_to_list_of_strings(params, 'custom_loss')
    _cast_value_to_list_of_strings(params, 'custom_metric')
    _cast_value_to_list_of_strings(params, 'per_float_feature_quantization')
    if 'monotone_constraints' in params:
        if not isinstance(params['monotone_constraints'], STRING_TYPES + ARRAY_TYPES + (dict,)):
            raise CatBoostError("Invalid `monotone_constraints` type={} : must be string or list of ints in range {-1, 0, 1} or dict.".format(type(param)))
    if 'feature_weights' in params:
        if not isinstance(params['feature_weights'], STRING_TYPES + ARRAY_TYPES + (dict,)):
            raise CatBoostError("Invalid `feature_weights` type={} : must be string or list of floats or dict.".format(type(param)))
    if 'first_feature_use_penalties' in params:
        if not isinstance(params['first_feature_use_penalties'], STRING_TYPES + ARRAY_TYPES + (dict,)):
            raise CatBoostError("Invalid `first_feature_use_penalties` type={} : must be string or list of floats or dict.".format(type(param)))
    if 'per_object_feature_penalties' in params:
        if not isinstance(params['per_object_feature_penalties'], STRING_TYPES + ARRAY_TYPES + (dict,)):
            raise CatBoostError("Invalid `per_object_feature_penalties` type={} : must be string or list of floats or dict.".format(type(param)))


def _params_type_cast(params):
    casted_params = {}
    for key, value in iteritems(params):
        value = _cast_to_base_types(value)
        casted_params[key] = value
    return casted_params


def _is_data_single_object(data):
    if isinstance(data, (Pool, FeaturesData, DataFrame) + SPARSE_MATRIX_TYPES):
        return False
    if not isinstance(data, ARRAY_TYPES):
        raise CatBoostError(
            "Invalid data type={} : must be list, numpy.ndarray, pandas.Series, pandas.DataFrame,"
            " scipy.sparse matrix, catboost.FeaturesData or catboost.Pool".format(type(data))
        )
    return len(np.shape(data)) == 1


def _process_feature_indices(feature_indices, pool, params, param_name):
    if param_name not in params:
        return feature_indices

    if param_name == 'cat_features':
        feature_type_name = 'categorical'
    elif param_name == 'text_features':
        feature_type_name = 'text'
    elif param_name == 'embedding_features':
        feature_type_name = 'embedding'
    else:
        raise CatBoostError('Unknown params_name=' + param_name)

    if isinstance(pool, Pool):
        feature_indices_from_params = _get_features_indices(params[param_name], pool.get_feature_names())
        if param_name == 'cat_features':
            feature_indices_from_pool = pool.get_cat_feature_indices()
        elif param_name == 'text_features':
            feature_indices_from_pool = pool.get_text_feature_indices()
        else:
            feature_indices_from_pool = pool.get_embedding_feature_indices()

        if set(feature_indices_from_pool) != set(feature_indices_from_params):
            raise CatBoostError(feature_type_name + " features indices in the model are set to "
                                + str(feature_indices_from_params) +
                                " and train dataset " + feature_type_name + " features indices are set to " +
                                str(feature_indices_from_pool))
    elif isinstance(pool, FeaturesData):
        raise CatBoostError(
            "Categorical features are set in the model. It is not allowed to use FeaturesData type for training dataset.")
    else:
        if feature_indices is not None and set(feature_indices) != set(params[param_name]):
            raise CatBoostError(feature_type_name + " features in the model are set to " + str(params[param_name]) +
                                ". " + feature_type_name + " features passed to fit function are set to " + str(feature_indices))
        feature_indices = params[param_name]
    del params[param_name]
    return feature_indices


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

    def _prepare_train_params(self, X=None, y=None, cat_features=None, text_features=None, embedding_features=None,
                              pairs=None, sample_weight=None, group_id=None, group_weight=None, subgroup_id=None,
                              pairs_weight=None, baseline=None, use_best_model=None, eval_set=None, verbose=None,
                              logging_level=None, plot=None, column_description=None, verbose_eval=None,
                              metric_period=None, silent=None, early_stopping_rounds=None, save_snapshot=None,
                              snapshot_file=None, snapshot_interval=None, init_model=None, callbacks=None):
        params = deepcopy(self._init_params)
        if params is None:
            params = {}

        _process_synonyms(params)

        if isinstance(X, FeaturesData):
            warnings.warn("FeaturesData is deprecated for using in fit function "
                          "and soon will not be supported. If you want to use FeaturesData, "
                          "please pass it to Pool initialization and use Pool in fit")

        cat_features = _process_feature_indices(cat_features, X, params, 'cat_features')
        text_features = _process_feature_indices(text_features, X, params, 'text_features')
        embedding_features = _process_feature_indices(embedding_features, X, params, 'embedding_features')

        train_pool = _build_train_pool(X, y, cat_features, text_features, embedding_features, pairs,
                                       sample_weight, group_id, group_weight, subgroup_id, pairs_weight,
                                       baseline, column_description)
        if train_pool.is_empty_:
            raise CatBoostError("X is empty.")

        allow_clear_pool = not isinstance(X, Pool)

        params['loss_function'] = _get_loss_function_for_train(
            params,
            getattr(self, '_estimator_type', None),
            train_pool
        )

        metric_period, verbose, logging_level = _process_verbose(
            metric_period, verbose, logging_level, verbose_eval, silent)

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

        if callbacks is not None:
            params['callbacks'] = _TrainCallbacksWrapper(callbacks)

        _check_param_types(params)
        params = _params_type_cast(params)
        _check_train_params(params)

        eval_set_list = eval_set if isinstance(eval_set, list) else [eval_set]
        eval_sets = []
        eval_total_row_count = 0
        for eval_set in eval_set_list:
            if isinstance(eval_set, Pool):
                eval_sets.append(eval_set)
                eval_total_row_count += eval_sets[-1].num_row()
                if eval_sets[-1].num_row() == 0:
                    raise CatBoostError("Empty 'eval_set' in Pool")
            elif isinstance(eval_set, PATH_TYPES):
                eval_sets.append(Pool(eval_set, column_description=column_description))
                eval_total_row_count += eval_sets[-1].num_row()
                if eval_sets[-1].num_row() == 0:
                    raise CatBoostError("Empty 'eval_set' in file {}".format(eval_set))
            elif isinstance(eval_set, tuple):
                if len(eval_set) != 2:
                    raise CatBoostError("Invalid shape of 'eval_set': {}, must be (X, y).".format(str(tuple(type(_) for _ in eval_set))))
                if eval_set[0] is None or eval_set[1] is None:
                    raise CatBoostError("'eval_set' tuple contains at least one None value")
                eval_sets.append(
                    Pool(
                        eval_set[0],
                        eval_set[1],
                        cat_features=train_pool.get_cat_feature_indices(),
                        text_features=train_pool.get_text_feature_indices(),
                        embedding_features=train_pool.get_embedding_feature_indices()
                    )
                )

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

        if (init_model is not None) and isinstance(init_model, PATH_TYPES):
            try:
                init_model = CatBoost().load_model(init_model)
            except Exception as e:
                raise CatBoostError("Error while loading init_model: {}".format(e))

        return {
            "train_pool": train_pool,
            "eval_sets": eval_sets,
            "params": params,
            "allow_clear_pool": allow_clear_pool,
            "init_model": init_model
        }

    def _fit(self, X, y, cat_features, text_features, embedding_features, pairs, sample_weight, group_id, group_weight, subgroup_id,
             pairs_weight, baseline, use_best_model, eval_set, verbose, logging_level, plot,
             column_description, verbose_eval, metric_period, silent, early_stopping_rounds,
             save_snapshot, snapshot_file, snapshot_interval, init_model, callbacks, log_cout=sys.stdout, log_cerr=sys.stderr):

        if X is None:
            raise CatBoostError("X must not be None")

        if y is None and not isinstance(X, PATH_TYPES + (Pool,)):
            raise CatBoostError("y may be None only when X is an instance of catboost.Pool or string")

        train_params = self._prepare_train_params(
            X=X, y=y, cat_features=cat_features, text_features=text_features, embedding_features=embedding_features,
            pairs=pairs, sample_weight=sample_weight, group_id=group_id, group_weight=group_weight,
            subgroup_id=subgroup_id, pairs_weight=pairs_weight, baseline=baseline, use_best_model=use_best_model,
            eval_set=eval_set, verbose=verbose, logging_level=logging_level, plot=plot,
            column_description=column_description, verbose_eval=verbose_eval, metric_period=metric_period,
            silent=silent, early_stopping_rounds=early_stopping_rounds, save_snapshot=save_snapshot,
            snapshot_file=snapshot_file, snapshot_interval=snapshot_interval, init_model=init_model,
            callbacks=callbacks
        )
        params = train_params["params"]
        train_pool = train_params["train_pool"]
        allow_clear_pool = train_params["allow_clear_pool"]

        with log_fixup(log_cout, log_cerr), \
            plot_wrapper(plot, [_get_train_dir(self.get_params())]):
            self._train(
                train_pool,
                train_params["eval_sets"],
                params,
                allow_clear_pool,
                train_params["init_model"]
            )

        # Have property feature_importance possibly set
        loss = self._object._get_loss_function_name()
        if loss and is_groupwise_metric(loss):
            pass  # too expensive
        elif (len(self.get_embedding_feature_indices()) > 0):
            pass  # is not implemented yet
        else:
            if not self._object._has_leaf_weights_in_model():
                if allow_clear_pool:
                    train_pool = _build_train_pool(X, y, cat_features, text_features, embedding_features, pairs, sample_weight, group_id, group_weight, subgroup_id, pairs_weight, baseline, column_description)
                self.get_feature_importance(data=train_pool, type=EFstrType.PredictionValuesChange)
            else:
                self.get_feature_importance(type=EFstrType.PredictionValuesChange)

        return self

    def fit(self, X, y=None, cat_features=None, text_features=None, embedding_features=None, pairs=None, sample_weight=None, group_id=None,
            group_weight=None, subgroup_id=None, pairs_weight=None, baseline=None, use_best_model=None,
            eval_set=None, verbose=None, logging_level=None, plot=False, column_description=None,
            verbose_eval=None, metric_period=None, silent=None, early_stopping_rounds=None,
            save_snapshot=None, snapshot_file=None, snapshot_interval=None, init_model=None, callbacks=None,
            log_cout=sys.stdout, log_cerr=sys.stderr):
        """
        Fit the CatBoost model.

        Parameters
        ----------
        X : catboost.Pool or list or numpy.ndarray or pandas.DataFrame or pandas.Series
             or string.
            If not catboost.Pool or catboost.FeaturesData it must be 2 dimensional Feature matrix
             or string - file with dataset.

             Must be non-empty (contain > 0 objects)

        y : list or numpy.ndarray or pandas.DataFrame or pandas.Series, optional (default=None)
            Labels, 1 dimensional array like.
            Use only if X is not catboost.Pool.

        cat_features : list or numpy.ndarray, optional (default=None)
            If not None, giving the list of Categ columns indices.
            Use only if X is not catboost.Pool and not catboost.FeaturesData

        text_features: list or numpy.ndarray, optional (default=None)
            If not none, giving the list of Text columns indices.
            Use only if X is not catboost.Pool and not catboost.FeaturesData

        embedding_features: list or numpy.ndarray, optional (default=None)
            If not none, giving the list of Embedding columns indices.
            Use only if X is not catboost.Pool and not catboost.FeaturesData

        pairs : list or numpy.ndarray or pandas.DataFrame
            The pairs description.
            If list or numpy.ndarrays or pandas.DataFrame, giving 2 dimensional.
            The shape should be Nx2, where N is the pairs' count. The first element of the pair is
            the index of the winner object in the training set. The second element of the pair is
            the index of the loser object in the training set.

        sample_weight : list or numpy.ndarray or pandas.DataFrame or pandas.Series, optional (default=None)
            Instance weights, 1 dimensional array like.

        group_id : list or numpy.ndarray, optional (default=None)
            group id for each instance.
            If not None, giving 1 dimensional array like data.
            Use only if X is not catboost.Pool.

        group_weight : list or numpy.ndarray, optional (default=None)
            Group weight for each instance.
            If not None, giving 1 dimensional array like data.

        subgroup_id : list or numpy.ndarray, optional (default=None)
            subgroup id for each instance.
            If not None, giving 1 dimensional array like data.
            Use only if X is not catboost.Pool.

        pairs_weight : list or numpy.ndarray, optional (default=None)
            Weight for each pair.
            If not None, giving 1 dimensional array like pairs.

        baseline : list or numpy.ndarray, optional (default=None)
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

        snapshot_file : string or pathlib.Path, [default=None]
            Learn progress snapshot file path, if None will use default filename

        snapshot_interval: int, [default=600]
            Interval between saving snapshots (seconds)

        init_model : CatBoost class or string or pathlib.Path, [default=None]
            Continue training starting from the existing model.
            If this parameter is a string or pathlib.Path, load initial model from the path specified by this string.

        callbacks : list, optional (default=None)
            List of callback objects that are applied at end of each iteration.

        log_cout: output stream or callback for logging

        log_cerr: error stream or callback for logging

        Returns
        -------
        model : CatBoost
        """
        return self._fit(X, y, cat_features, text_features, embedding_features, pairs, sample_weight, group_id, group_weight, subgroup_id,
                         pairs_weight, baseline, use_best_model, eval_set, verbose, logging_level, plot,
                         column_description, verbose_eval, metric_period, silent, early_stopping_rounds,
                         save_snapshot, snapshot_file, snapshot_interval, init_model, callbacks, log_cout, log_cerr)

    def _process_predict_input_data(self, data, parent_method_name, thread_count, label=None):
        if not self.is_fitted() or self.tree_count_ is None:
            raise CatBoostError(("There is no trained model to use {}(). "
                                 "Use fit() to train model. Then use this method.").format(parent_method_name))
        is_single_object = _is_data_single_object(data)
        if not isinstance(data, Pool):
            data = Pool(
                data=[data] if is_single_object else data,
                label=label,
                cat_features=self._get_cat_feature_indices() if not isinstance(data, FeaturesData) else None,
                text_features=self._get_text_feature_indices() if not isinstance(data, FeaturesData) else None,
                embedding_features=self._get_embedding_feature_indices() if not isinstance(data, FeaturesData) else None,
                thread_count=thread_count
            )
        return data, is_single_object

    def _validate_prediction_type(self, prediction_type, valid_prediction_types=('Class', 'RawFormulaVal', 'Probability', 'LogProbability', 'Exponent', 'RMSEWithUncertainty')):
        if not isinstance(prediction_type, STRING_TYPES):
            raise CatBoostError("Invalid prediction_type type={}: must be str().".format(type(prediction_type)))
        if prediction_type not in valid_prediction_types:
            raise CatBoostError("Invalid value of prediction_type={}: must be {}.".format(prediction_type, ', '.join(valid_prediction_types)))

    def _predict(self, data, prediction_type, ntree_start, ntree_end, thread_count, verbose, parent_method_name, task_type="CPU"):
        verbose = verbose or self.get_param('verbose')
        if verbose is None:
            verbose = False
        data, data_is_single_object = self._process_predict_input_data(data, parent_method_name, thread_count)
        self._validate_prediction_type(prediction_type)

        predictions = self._base_predict(data, prediction_type, ntree_start, ntree_end, thread_count, verbose, task_type)
        return predictions[0] if data_is_single_object else predictions

    def predict(self, data, prediction_type='RawFormulaVal', ntree_start=0, ntree_end=0, thread_count=-1, verbose=None, task_type="CPU"):
        """
        Predict with data.

        Parameters
        ----------
        data : catboost.Pool or list of features or list of lists or numpy.ndarray or pandas.DataFrame or pandas.Series
                or catboost.FeaturesData
            Data to apply model on.
            If data is a simple list (not list of lists) or a one-dimensional numpy.ndarray it is interpreted
            as a list of features for a single object.

        prediction_type : string, optional (default='RawFormulaVal')
            Can be:
            - 'RawFormulaVal' : return raw value.
            - 'Class' : return class label.
            - 'Probability' : return probability for every class.
            - 'Exponent' : return Exponent of raw formula value.
            - 'RMSEWithUncertainty': return standard deviation for RMSEWithUncertainty loss function
              (logarithm of the standard deviation is returned by default).

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
                - 'Class' : return class label.
                - 'Probability' : return one-dimensional numpy.ndarray with probability for every class.
            otherwise numpy.ndarray, with values that depend on prediction_type value:
                - 'RawFormulaVal' : one-dimensional array of raw formula value for each object.
                - 'Class' : one-dimensional array of class label for each object.
                - 'Probability' : two-dimensional numpy.ndarray with shape (number_of_objects x number_of_classes)
                  with probability for every class for each object.
        """
        return self._predict(data, prediction_type, ntree_start, ntree_end, thread_count, verbose, 'predict', task_type)

    def _virtual_ensembles_predict(self, data, prediction_type, ntree_end, virtual_ensembles_count, thread_count, verbose, parent_method_name):
        verbose = verbose or self.get_param('verbose')
        if verbose is None:
            verbose = False
        data, data_is_single_object = self._process_predict_input_data(data, parent_method_name, thread_count)
        self._validate_prediction_type(prediction_type, ['VirtEnsembles', 'TotalUncertainty'])

        if ntree_end == 0:
            ntree_end = self.tree_count_

        predictions = self._base_virtual_ensembles_predict(data, prediction_type, ntree_end, virtual_ensembles_count, thread_count, verbose)
        if prediction_type == 'VirtEnsembles':
            shape = predictions.shape
            predictions = predictions.reshape(shape[0], virtual_ensembles_count, int(shape[1] / virtual_ensembles_count))
        return predictions[0] if data_is_single_object else predictions

    def virtual_ensembles_predict(self, data, prediction_type='VirtEnsembles', ntree_end=0, virtual_ensembles_count=10, thread_count=-1, verbose=None):
        """
        Predict with data.

        Parameters
        ----------
        data : catboost.Pool or list of features or list of lists or numpy.ndarray or pandas.DataFrame or pandas.Series
                or catboost.FeaturesData
            Data to apply model on.
            If data is a simple list (not list of lists) or a one-dimensional numpy.ndarray it is interpreted
            as a list of features for a single object.

        prediction_type : string, optional (default='RawFormulaVal')
            Can be:
            - 'VirtEnsembles': return V (virtual_ensembles_count) predictions.
                k-th virtEnsemle consists of trees [0, T/2] + [T/2 + T/(2V) * k, T/2 + T/(2V) * (k + 1)]  * constant.
            - 'TotalUncertainty': return mean predict, var (and knowledge uncertainty
                if model was trained with RMSEWithUncertainty loss function) for virtEnsembles

        ntree_end: int, optional (default=0)
            Model is applied on the interval [ntree_start, ntree_end) (zero-based indexing).
            If value equals to 0 this parameter is ignored and ntree_end equal to tree_count_.

        virtual_ensembles_count: int, optional (default=10)
            virtual ensembles count for 'TotalUncertainty' and 'VirtEnsembles' prediction types.

        thread_count : int (default=-1)
            The number of threads to use when applying the model.
            Allows you to optimize the speed of execution. This parameter doesn't affect results.
            If -1, then the number of threads is set to the number of CPU cores.

        verbose : bool, optional (default=False)
            If True, writes the evaluation metric measured set to stderr.

        Returns
        -------
        prediction :
            (with V as virtual_ensembles_count and T as trees count,
            k-th virtEnsemle consists of trees [0, T/2] + [T/2 + T/(2V) * k, T/2 + T/(2V) * (k + 1)]  * constant)
            If data is for a single object, return 1-dimensional array of predictions with size depends on prediction type,
            otherwise return 2-dimensional numpy.ndarray with shape (number_of_objects x size depends on prediction type);
            Returned predictions depends on prediction type:
            If loss-function was RMSEWithUncertainty:
                - 'VirtEnsembles': [mean0, var0, mean1, var1, ..., vark-1].
                - 'TotalUncertainty': [mean_predict, KnowledgeUnc, DataUnc].
            otherwise for regression:
                - 'VirtEnsembles':  [mean0, mean1, ...].
                - 'TotalUncertainty': [mean_predicts, KnowledgeUnc].
            otherwise for binary classification:
                - 'VirtEnsembles':  [ApproxRawFormulaVal0, ApproxRawFormulaVal1, ..., ApproxRawFormulaValk-1].
                - 'TotalUncertainty':  [DataUnc, TotalUnc].
        """
        return self._virtual_ensembles_predict(data, prediction_type, ntree_end, virtual_ensembles_count, thread_count, verbose, 'virtual_ensembles_predict')

    def _staged_predict(self, data, prediction_type, ntree_start, ntree_end, eval_period, thread_count, verbose, parent_method_name):
        verbose = verbose or self.get_param('verbose')
        if verbose is None:
            verbose = False
        data, data_is_single_object = self._process_predict_input_data(data, parent_method_name, thread_count)
        self._validate_prediction_type(prediction_type)

        if ntree_end == 0:
            ntree_end = self.tree_count_
        staged_predict_iterator = self._staged_predict_iterator(data, prediction_type, ntree_start, ntree_end, eval_period, thread_count, verbose)
        for predictions in staged_predict_iterator:
            yield predictions[0] if data_is_single_object else predictions

    def staged_predict(self, data, prediction_type='RawFormulaVal', ntree_start=0, ntree_end=0, eval_period=1, thread_count=-1, verbose=None):
        """
        Predict target at each stage for data.

        Parameters
        ----------
        data : catboost.Pool or list of features or list of lists or numpy.ndarray or pandas.DataFrame or pandas.Series
                or catboost.FeaturesData
            Data to apply model on.
            If data is a simple list (not list of lists) or a one-dimensional numpy.ndarray it is interpreted
            as a list of features for a single object.

        prediction_type : string, optional (default='RawFormulaVal')
            Can be:
            - 'RawFormulaVal' : return raw formula value.
            - 'Class' : return class label.
            - 'Probability' : return probability for every class.
            - 'RMSEWithUncertainty': return standard deviation for RMSEWithUncertainty loss function
              (logarithm of the standard deviation is returned by default).

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
                - 'Class' : return class label.
                - 'Probability' : return one-dimensional numpy.ndarray with probability for every class.
            otherwise numpy.ndarray, with values that depend on prediction_type value:
                - 'RawFormulaVal' : one-dimensional array of raw formula value for each object.
                - 'Class' : one-dimensional array of class label for each object.
                - 'Probability' : two-dimensional numpy.ndarray with shape (number_of_objects x number_of_classes)
                  with probability for every class for each object.
        """
        return self._staged_predict(data, prediction_type, ntree_start, ntree_end, eval_period, thread_count, verbose, 'staged_predict')

    def _iterate_leaf_indexes(self, data, ntree_start, ntree_end):
        if ntree_end == 0:
            ntree_end = self.tree_count_
        data, _ = self._process_predict_input_data(data, "iterate_leaf_indexes", thread_count=-1)
        leaf_indexes_iterator = self._leaf_indexes_iterator(data, ntree_start, ntree_end)
        for leaf_index in leaf_indexes_iterator:
            yield leaf_index

    def iterate_leaf_indexes(self, data, ntree_start=0, ntree_end=0):
        """
        Returns indexes of leafs to which objects from pool are mapped by model trees.

        Parameters
        ----------
        data : catboost.Pool or list of features or list of lists or numpy.ndarray or pandas.DataFrame or pandas.Series
                or catboost.FeaturesData
            Data to apply model on.
            If data is a simple list (not list of lists) or a one-dimensional numpy.ndarray it is interpreted
            as a list of features for a single object.

        ntree_start: int, optional (default=0)
            Index of first tree for which leaf indexes will be calculated (zero-based indexing).

        ntree_end: int, optional (default=0)
            Index of the tree after last tree for which leaf indexes will be calculated (zero-based indexing).
            If value equals to 0 this parameter is ignored and ntree_end equal to tree_count_.

        Returns
        -------
        leaf_indexes : generator. For each object in pool yields one-dimensional numpy.ndarray of leaf indexes.
        """
        return self._iterate_leaf_indexes(data, ntree_start, ntree_end)

    def _calc_leaf_indexes(self, data, ntree_start, ntree_end, thread_count, verbose):
        if ntree_end == 0:
            ntree_end = self.tree_count_
        data, _ = self._process_predict_input_data(data, "calc_leaf_indexes", thread_count)
        return self._base_calc_leaf_indexes(data, ntree_start, ntree_end, thread_count, verbose)

    def calc_leaf_indexes(self, data, ntree_start=0, ntree_end=0, thread_count=-1, verbose=False):
        """
        Returns indexes of leafs to which objects from pool are mapped by model trees.

        Parameters
        ----------
        data : catboost.Pool or list of features or list of lists or numpy.ndarray or pandas.DataFrame or pandas.Series
                or catboost.FeaturesData
            Data to apply model on.
            If data is a simple list (not list of lists) or a one-dimensional numpy.ndarray it is interpreted
            as a list of features for a single object.

        ntree_start: int, optional (default=0)
            Index of first tree for which leaf indexes will be calculated (zero-based indexing).

        ntree_end: int, optional (default=0)
            Index of the tree after last tree for which leaf indexes will be calculated (zero-based indexing).
            If value equals to 0 this parameter is ignored and ntree_end equal to tree_count_.

        thread_count : int (default=-1)
            The number of threads to use when applying the model.
            Allows you to optimize the speed of execution. This parameter doesn't affect results.
            If -1, then the number of threads is set to the number of CPU cores.

        verbose : bool (default=False)
            Enable debug logging level.

        Returns
        -------
        leaf_indexes : 2-dimensional numpy.ndarray of numpy.uint32 with shape (object count, ntree_end - ntree_start).
            i-th row is an array of leaf indexes for i-th object.
        """
        return self._calc_leaf_indexes(data, ntree_start, ntree_end, thread_count, verbose)

    def get_cat_feature_indices(self):
        if not self.is_fitted():
            raise CatBoostError("Model is not fitted")
        return self._get_cat_feature_indices()

    def get_text_feature_indices(self):
        if not self.is_fitted():
            raise CatBoostError("Model is not fitted")
        return self._get_text_feature_indices()

    def get_embedding_feature_indices(self):
        if not self.is_fitted():
            raise CatBoostError("Model is not fitted")
        return self._get_embedding_feature_indices()

    def _eval_metrics(self, data, metrics, ntree_start, ntree_end, eval_period, thread_count, res_dir, tmp_dir, plot, log_cout=sys.stdout, log_cerr=sys.stderr):
        if not self.is_fitted():
            raise CatBoostError("There is no trained model to evaluate metrics on. Use fit() to train model. Then call this method.")
        if not isinstance(data, Pool):
            raise CatBoostError("Invalid data type={}, must be catboost.Pool.".format(type(data)))
        if data.is_empty_:
            raise CatBoostError("Data is empty.")
        if not isinstance(metrics, ARRAY_TYPES) and not isinstance(metrics, STRING_TYPES) and not isinstance(metrics, BuiltinMetric):
            raise CatBoostError("Invalid metrics type={}, must be list(), str() or one of builtin catboost.metrics.* class instances.".format(type(metrics)))
        if not all(map(lambda metric: isinstance(metric, string_types) or isinstance(metric, BuiltinMetric), metrics)):
            raise CatBoostError("Invalid metric type: must be string() or one of builtin catboost.metrics.* class instances.")
        if tmp_dir is None:
            tmp_dir = tempfile.mkdtemp()

        if isinstance(metrics, STRING_TYPES) or isinstance(metrics, BuiltinMetric):
            metrics = [metrics]
        metrics = stringify_builtin_metrics_list(metrics)
        with log_fixup(log_cout, log_cerr), plot_wrapper(plot, [res_dir]):
            metrics_score, metric_names = self._base_eval_metrics(data, metrics, ntree_start, ntree_end, eval_period, thread_count, res_dir, tmp_dir)

        return dict(zip(metric_names, metrics_score))

    def eval_metrics(self, data, metrics, ntree_start=0, ntree_end=0, eval_period=1, thread_count=-1, tmp_dir=None, plot=False, log_cout=sys.stdout, log_cerr=sys.stderr):
        """
        Calculate metrics.

        Parameters
        ----------
        data : catboost.Pool
            Data to evaluate metrics on.

        metrics : list of strings or catboost.metrics.BuiltinMetric
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

        tmp_dir : string or pathlib.Path (default=None)
            The name of the temporary directory for intermediate results.
            If None, then the name will be generated.

        plot : bool, optional (default=False)
            If True, draw train and eval error in Jupyter notebook

        log_cout: output stream or callback for logging

        log_cerr: error stream or callback for logging

        Returns
        -------
        prediction : dict: metric -> array of shape [(ntree_end - ntree_start) / eval_period]
        """
        return self._eval_metrics(data, metrics, ntree_start, ntree_end, eval_period, thread_count, _get_train_dir(self.get_params()), tmp_dir, plot, log_cout, log_cerr)

    def compare(self, model, data, metrics, ntree_start=0, ntree_end=0, eval_period=1, thread_count=-1, tmp_dir=None, log_cout=sys.stdout, log_cerr=sys.stderr):
        """
        Draw train and eval errors in Jupyter notebook for both models

        Parameters
        ----------
        model: CatBoost model
            Another model to draw metrics

        data : catboost.Pool
            Data to evaluate metrics on.

        metrics : list of strings or catboost.metrics.BuiltinMetric
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

        tmp_dir : string or pathlib.Path (default=None)
            The name of the temporary directory for intermediate results.
            If None, then the name will be generated.

        log_cout: output stream or callback for logging

        log_cerr: error stream or callback for logging
        """

        if model is None:
            raise CatBoostError("You should provide model for comparison.")
        if data is None:
            raise CatBoostError("You should provide data for comparison.")
        if metrics is None:
            raise CatBoostError("You should provide metrics for comparison.")

        need_to_remove = False
        if tmp_dir is None:
            need_to_remove = True
            tmp_dir = tempfile.mkdtemp()
        first_dir = os.path.join(tmp_dir, 'first_model')
        second_dir = os.path.join(tmp_dir, 'second_model')

        create_if_not_exist = lambda path: os.mkdir(path) if not os.path.exists(path) else None
        create_if_not_exist(tmp_dir)
        create_if_not_exist(first_dir)
        create_if_not_exist(second_dir)

        with plot_wrapper(True, [first_dir, second_dir]):
            self._eval_metrics(data, metrics, ntree_start, ntree_end, eval_period, thread_count, first_dir, tmp_dir,
                               plot=False, log_cout=log_cout, log_cerr=log_cerr)
            model._eval_metrics(data, metrics, ntree_start, ntree_end, eval_period, thread_count, second_dir, tmp_dir,
                                plot=False, log_cout=log_cout, log_cerr=log_cerr)

        if need_to_remove:
            shutil.rmtree(tmp_dir)

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
        batch_calcer.add(part1)
        batch_calcer.add(part2)
        metrics = batch_calcer.eval_metrics()
        """
        if not self.is_fitted():
            raise CatBoostError("There is no trained model to evaluate metrics on. Use fit() to train model. Then call this method.")
        return BatchMetricCalcer(self._object, metrics, ntree_start, ntree_end, eval_period, thread_count, tmp_dir)

    @property
    def feature_importances_(self):
        loss = self._object._get_loss_function_name()
        if loss and is_groupwise_metric(loss):
            return np.array(getattr(self, "_loss_value_change", None))
        else:
            return np.array(getattr(self, "_prediction_values_change", None))


    def get_feature_importance(self, data=None, type=EFstrType.FeatureImportance, prettified=False, thread_count=-1, verbose=False, fstr_type=None, shap_mode="Auto", model_output="Raw", interaction_indices=None, shap_calc_type="Regular", reference_data=None, log_cout=sys.stdout, log_cerr=sys.stderr):
        """
        Parameters
        ----------
        data :
            Data to get feature importance.
            If type in ('LossFunctionChange', 'ShapValues', 'ShapInteractionValues') data must of Pool type.
                For every object in this dataset feature importances will be calculated.
            If type == 'PredictionValuesChange', data is None or a dataset of Pool type
                Dataset specification is needed only in case if the model does not contain leaf weight information (trained with CatBoost v < 0.9).
            If type == 'PredictionDiff' data must contain a matrix of feature values of shape (2, n_features).
                Possible types are catboost.Pool or list of lists or numpy.ndarray or pandas.DataFrame or pandas.Series
                or catboost.FeaturesData or pandas.SparseDataFrame or scipy.sparse.spmatrix
            If type == 'FeatureImportance'
                See 'PredictionValuesChange' for non-ranking metrics and 'LossFunctionChange' for ranking metrics.
            If type == 'Interaction'
                This parameter is not used.

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
                - ShapInteractionValues
                    Calculate SHAP Interaction Values between each pair of features for every object
                - Interaction
                    Calculate pairwise score between every feature.
                - PredictionDiff
                    Calculate most important features explaining difference in predictions for a pair of documents.

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

        shap_calc_type : EShapCalcType or string, optional (default="Regular")
            used only for ShapValues type
            Possible values:
                - "Regular"
                    Calculate regular SHAP values
                - "Approximate"
                    Calculate approximate SHAP values
                - "Exact"
                    Calculate exact SHAP values

        interaction_indices : list of int or string (feature_idx_1, feature_idx_2), optional (default=None)
            used only for ShapInteractionValues type
            Calculate SHAP Interaction Values between pair of features feature_idx_1 and feature_idx_2 for every object

        reference_data: catboost.Pool or None
            Reference data for Independent Tree SHAP values from https://arxiv.org/abs/1905.04610v1
            if type == 'ShapValues' and reference_data is not None, then Independent Tree SHAP values are calculated

        log_cout: output stream or callback for logging

        log_cerr: error stream or callback for logging

        Returns
        -------
        depends on type:
            - FeatureImportance
                See PredictionValuesChange for non-ranking metrics and LossFunctionChange for ranking metrics.
            - PredictionValuesChange, LossFunctionChange, PredictionDiff with prettified=False (default)
                list of length [n_features] with feature_importance values (float) for feature
            - PredictionValuesChange, LossFunctionChange, PredictionDiff with prettified=True
                list of length [n_features] with (feature_id (string), feature_importance (float)) pairs, sorted by feature_importance in descending order
            - ShapValues
                np.ndarray of shape (n_objects, n_features + 1) with Shap values (float) for (object, feature).
                In case of multiclass the returned value is np.ndarray of shape
                (n_objects, classes_count, n_features + 1). For each object it contains Shap values (float).
                Values are calculated for RawFormulaVal predictions.
            - ShapInteractionValues
                np.ndarray of shape (n_objects, n_features + 1, n_features + 1) with Shap interaction values (float) for (object, feature(i), feature(j)).
                In case of multiclass the returned value is np.ndarray of shape
                (n_objects, classes_count, n_features + 1, n_features + 1). For each object it contains Shap interaction values (float).
                Values are calculated for RawFormulaVal predictions.
            - Interaction
                list of length [n_features] of 3-element lists of (first_feature_index, second_feature_index, interaction_score (float))
        """

        if not isinstance(verbose, bool) and not isinstance(verbose, int):
            raise CatBoostError('verbose should be bool or int.')
        verbose = int(verbose)
        if verbose < 0:
            raise CatBoostError('verbose should be non-negative.')

        if fstr_type is not None:
            type = fstr_type

        type = enum_from_enum_or_str(EFstrType, type)
        if type == EFstrType.FeatureImportance:
            loss = self._object._get_loss_function_name()
            if loss and is_groupwise_metric(loss):
                type = EFstrType.LossFunctionChange
            else:
                type = EFstrType.PredictionValuesChange

        if type == EFstrType.PredictionDiff:
            data, _ = self._process_predict_input_data(data, "get_feature_importance", thread_count)
            if data.num_row() != 2:
                raise CatBoostError("{} requires a pair of documents, found {}".format(type, data.num_row()))
        else:
            if data is not None and not isinstance(data, Pool):
                raise CatBoostError("Invalid data type={}, must be catboost.Pool.".format(_typeof(data)))

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

        with log_fixup(log_cout, log_cerr):
            shap_calc_type = enum_from_enum_or_str(EShapCalcType, shap_calc_type).value
            fstr, feature_names = self._calc_fstr(type, data, reference_data, thread_count, verbose, model_output, shap_mode, interaction_indices,
                                                  shap_calc_type)
        if type in (EFstrType.PredictionValuesChange, EFstrType.LossFunctionChange, EFstrType.PredictionDiff):
            feature_importances = [value[0] for value in fstr]
            attribute_name = None
            if type == EFstrType.PredictionValuesChange:
                attribute_name = "_prediction_values_change"
            if type == EFstrType.LossFunctionChange:
                attribute_name = "_loss_value_change"
            if attribute_name:
                setattr(
                    self,
                    attribute_name,
                    feature_importances
                )

            if prettified:
                feature_importances = sorted(zip(feature_names, feature_importances), key=itemgetter(1), reverse=True)
                columns = ['Feature Id', 'Importances']
                return DataFrame(feature_importances, columns=columns)
            else:
                return np.array(feature_importances)
        elif type == EFstrType.ShapValues:
            if isinstance(fstr[0][0], ARRAY_TYPES):
                return np.array([np.array([np.array([
                    value for value in dimension]) for dimension in doc]) for doc in fstr])
            else:
                result = [[value for value in doc] for doc in fstr]
                if prettified:
                    return DataFrame(result)
                else:
                    return np.array(result)
        elif type == EFstrType.ShapInteractionValues:
            if isinstance(fstr[0][0], ARRAY_TYPES):
                return np.array([np.array([np.array([
                    feature2 for feature2 in feature1]) for feature1 in doc]) for doc in fstr])
            else:
                return np.array([np.array([np.array([np.array([
                    feature2 for feature2 in feature1]) for feature1 in dimension]) for dimension in doc]) for doc in fstr])
        elif type == EFstrType.Interaction:
            result = [[int(row[0]), int(row[1]), row[2]] for row in fstr]
            if prettified:
                columns = ['First Feature Index', 'Second Feature Index', 'Interaction']
                return DataFrame(result, columns=columns)
            else:
                return np.array(result)

    def get_object_importance(
            self, pool, train_pool, top_size=-1, type='Average', update_method='SinglePoint',
            importance_values_sign='All', thread_count=-1, verbose=False, ostr_type=None,
            log_cout=sys.stdout, log_cerr=sys.stderr):
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

        type : string, optional (default='Average')
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

        ostr_type : string, deprecated, use type instead

        log_cout: output stream or callback for logging

        log_cerr: error stream or callback for logging

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

        if ostr_type is not None:
            type = ostr_type
            warnings.warn("'ostr_type' parameter will be deprecated soon, use 'type' parameter instead")

        with log_fixup(log_cout, log_cerr):
            result = self._calc_ostr(train_pool, pool, top_size, type, update_method, importance_values_sign, thread_count, verbose)
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
                * 'pmml' to export into PMML format
                * 'cpp' to export as C++ code
                * 'python' to export as Python code.
        export_parameters : dict
            Parameters for CoreML export:
                * prediction_type : string - either 'probability' or 'raw'
                * coreml_description : string
                * coreml_model_version : string
                * coreml_model_author : string
                * coreml_model_license: string
            Parameters for PMML export:
                * pmml_copyright : string
                * pmml_description : string
                * pmml_model_version : string
        pool : catboost.Pool or list or numpy.ndarray or pandas.DataFrame or pandas.Series or catboost.FeaturesData
            Training pool.
        """
        if not self.is_fitted():
            raise CatBoostError("There is no trained model to use save_model(). Use fit() to train model. Then use this method.")
        if not isinstance(fname, PATH_TYPES):
            raise CatBoostError("Invalid fname type={}: must be str() or pathlib.Path().".format(type(fname)))
        if pool is not None and not isinstance(pool, Pool):
            pool = Pool(
                data=pool,
                cat_features=self._get_cat_feature_indices() if not isinstance(pool, FeaturesData) else None,
                text_features=self._get_text_feature_indices() if not isinstance(pool, FeaturesData) else None,
                embedding_features=self._get_embedding_feature_indices() if not isinstance(pool, FeaturesData) else None
            )
        self._save_model(fname, format, export_parameters, pool)

    def load_model(self, fname=None, format='cbm', stream=None, blob=None):
        """
        Load model from a file, stream or blob.

        Parameters
        ----------
        fname : string
            Input file name.
        """
        if (fname is None) + (stream is None) + (blob is None) != 2:
            raise CatBoostError("Exactly one of fname/stream/blob arguments mustn't be None")

        if fname is not None:
            self._load_model(fname, format)
        elif stream is not None:
            self._load_from_stream(stream)
        elif blob is not None:
            self._load_from_string(blob)
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

    def get_all_params(self):
        """
        Get all params (specified by user and default params) that were set in training from CatBoost model.
        Full parameters documentation could be found here: https://catboost.ai/docs/concepts/python-reference_parameters-list.html

        Returns
        -------
        result : dict
            Dictionary of {param_key: param_value}.
        """
        if not self.is_fitted():
            raise CatBoostError("There is no trained model to use get_all_params(). Use fit() to train model. Then use this method.")
        return self._object._get_plain_params()


    def save_borders(self, fname):
        """
        Save the model borders to a file.

        Parameters
        ----------
        fname : string or pathlib.Path
            Output file name.
        """
        if not isinstance(fname, PATH_TYPES):
            raise CatBoostError("Invalid fname type={}: must be str() or pathlib.Path().".format(type(fname)))
        self._save_borders(fname)

    def get_borders(self):
        """
        Return map feature_index: borders for float features.

        """
        return self._get_borders()

    def set_params(self, **params):
        """
        Set parameters into CatBoost model.

        Parameters
        ----------
        **params : key=value format
            List of key=value paris. Example: model.set_params(iterations=500, thread_count=2).
        """
        if self.is_fitted():
            raise CatBoostError("You can't change params of fitted model.")
        for key, value in iteritems(params):
            self._init_params[key] = value
        if 'thread_count' in self._init_params and self._init_params['thread_count'] == -1:
            self._init_params.pop('thread_count')
        return self

    def plot_predictions(self, data, features_to_change, plot=True, plot_file=None):
        """
        To use this function, you should install plotly.
        data: numpy.ndarray or pandas.DataFrame or catboost.Pool
        feature:
            Float features indexes in pd.DataFrame for which you want vary prediction value.
        plot: bool
            Plot predictions.
        plot_file: str
            Output file for plot predictions.
        Returns
        -------
            List of list of predictions for all buckets for all documents in data
        """

        def predict(doc, feature_idx, borders, nan_treatment):
            left_extend_border = min(2 * borders[0], -1)
            right_extend_border = max(2 * borders[-1], 1)
            extended_borders = [left_extend_border] + borders + [right_extend_border]
            points = []
            predictions = []
            border_idx = None
            if np.isnan(doc[feature_idx]):
                border_idx = len(borders) if nan_treatment == 'AsTrue' else 0
            for i in range(len(extended_borders) - 1):
                points += [(extended_borders[i] + extended_borders[i + 1]) / 2.]
                if border_idx is None and doc[feature_idx] < extended_borders[i + 1]:
                    border_idx = i
                buf = doc[feature_idx]
                doc[feature_idx] = points[-1]
                predictions += [self.predict(doc)]
                doc[feature_idx] = buf
            if border_idx is None:
                border_idx = len(borders)
            return predictions, border_idx

        def get_layout(go, feature, xaxis):
            return go.Layout(
                title="Prediction variation for feature '{}'".format(feature),
                yaxis={
                    'title': 'Prediction',
                    'side': 'left',
                    'overlaying': 'y2'
                },
                xaxis=xaxis
            )
        try:
            import plotly.graph_objs as go
        except ImportError as e:
            warnings.warn("To draw plots you should install plotly.")
            raise ImportError(str(e))

        model_borders = self._get_borders()

        data, _ = self._process_predict_input_data(data, "vary_feature_value_and_apply", thread_count=-1)
        figs = []
        all_predictions = [{}] * data.num_row()
        nan_treatments = self._get_nan_treatments()
        for feature in features_to_change:
            if not isinstance(feature, int):
                if self.feature_names_ is None or feature not in self.feature_names_:
                    raise CatBoostError('No feature named "{}" in model'.format(feature))
                feature_idx = self.feature_names_.index(feature)
            else:
                feature_idx = feature
                feature = self.feature_names_[feature_idx]
            assert feature_idx in model_borders, "only float features indexes are supported"
            borders = model_borders[feature_idx]

            if len(borders) == 0:
                xaxis = go.layout.XAxis(title='Bins', tickvals=[0])
                figs += go.Figure(data=[],
                                 layout=get_layout(go, feature_idx, xaxis))

            xaxis = go.layout.XAxis(
                title='Bins',
                tickmode='array',
                tickvals=list(range(len(borders) + 1)),
                ticktext=['(-inf, {:.4f}]'.format(borders[0])] +
                         ['({:.4f}, {:.4f}]'.format(val_1, val_2)
                          for val_1, val_2 in zip(borders[:-1], borders[1:])] +
                         ['({:.4f}, +inf)'.format(borders[-1])],
                showticklabels=False
            )

            trace = []
            for idx, features in enumerate(data.get_features()):
                predictions, border_idx = predict(features, feature_idx, borders, nan_treatments[feature_idx])
                all_predictions[idx][feature_idx] = predictions
                trace.append(go.Scatter(
                    y = predictions,
                    mode = 'lines+markers',
                    name = u'Document {} predictions'.format(idx)
                ))

                trace.append(go.Scatter(
                    x = [border_idx],
                    y = [predictions[border_idx]],
                    showlegend=False
                ))

            layout = get_layout(go, feature, xaxis)
            figs += [go.Figure(data=trace, layout=layout)]

        if plot:
            try_plot_offline(figs)

        if plot_file:
            save_plot_file(plot_file, 'Predictions for all buckets', figs)

        return all_predictions, figs

    def plot_partial_dependence(self, data, features, plot=True, plot_file=None, thread_count=-1):
        """
        To use this function, you should install plotly.
        data: numpy.ndarray or pandas.DataFrame or catboost.Pool
        features: int, str, list<int>, tuple<int>, list<string>, tuple<string>
            Float features to calculate partial dependence for. Number of features should be 1 or 2.
        plot: bool
            Plot predictions.
        plot_file: str
            Output file for plot predictions.
        thread_count: int
            Number of threads to use. If -1 use maximum available number of threads.
        Returns
        -------
            If number of features is one - 1d numpy array and figure with line plot.
            If number of features is two - 2d numpy array and figure with 2d heatmap.
        """

        try:
            import plotly.graph_objs as go
        except ImportError as e:
            warnings.warn("To draw plots you should install plotly.")
            raise ImportError(str(e))

        def getFeatureIdx(feature):
            if not isinstance(feature, int):
                if self.feature_names_ is None or feature not in self.feature_names_:
                    raise CatBoostError('No feature named "{}" in model'.format(feature))
                feature_idx = self.feature_names_.index(feature)
            else:
                feature_idx = feature
            assert feature_idx in self._get_borders(), "only float features indexes are supported"
            assert len(self._get_borders()[feature_idx]) > 0, "feature with idx {} is not used in model".format(feature_idx)
            return feature_idx

        def getFeatureIndices(features):
            if isinstance(features, list) or isinstance(features, tuple):
                features_idxs = [getFeatureIdx(feature) for feature in features]
            elif isinstance(features, int) or isinstance(features, str):
                features_idxs = [getFeatureIdx(features)]
            else:
                raise CatBoostError('Unsupported type for argument \'features\'. Must be one of: int, string, list<string>, list<int>, tuple<int>, tuple<string>')
            return features_idxs

        def getAxisParams(borders, feature_name=None):
            return {
                'title': 'Bins' if feature_name is None else 'Bins of feature \'{}\''.format(feature_name),
                'tickmode': 'array',
                'tickvals': list(range(len(borders) + 1)),
                'ticktext': ['(-inf, {:.4f}]'.format(borders[0])] +
                            ['({:.4f}, {:.4f}]'.format(val_1, val_2)
                            for val_1, val_2 in zip(borders[:-1], borders[1:])] +
                            ['({:.4f}, +inf)'.format(borders[-1])],
                'showticklabels': False}

        def plot2d(feature_names, borders, predictions):
            xaxis = go.layout.XAxis(**getAxisParams(borders[1], feature_name=feature_names[1]))
            yaxis = go.layout.YAxis(**getAxisParams(borders[0], feature_name=feature_names[0]))
            layout = go.Layout(
                title='Partial dependence plot for features {}'.format('\'{}\''.format('\', \''.join(map(str, feature_names)))),
                yaxis=yaxis,
                xaxis=xaxis
            )
            fig = go.Figure(data=go.Heatmap(z=predictions), layout=layout)
            return fig

        def plot1d(feature, borders, predictions):
            xaxis = go.layout.XAxis(**getAxisParams(borders))
            yaxis = {
                'title': 'Mean Prediction',
                'side': 'left'
            }
            layout = go.Layout(
                title="Partial dependence plot for feature '{}'".format(feature),
                yaxis=yaxis,
                xaxis=xaxis
            )
            fig = go.Figure(data=go.Scatter(y=predictions, mode='lines+markers'), layout=layout)
            return fig

        features_idx = getFeatureIndices(features)
        borders = [self._get_borders()[idx] for idx in features_idx]
        if len(features_idx) not in [1, 2]:
            raise CatBoostError('Number of \'features\' should be 1 or 2, got {}'.format(len(features_idx)))
        is_2d_plot = len(features_idx) == 2

        data, _ = self._process_predict_input_data(data, "plot_partial_dependence", thread_count=thread_count)
        all_predictions = np.array(self._object._calc_partial_dependence(data, features_idx, thread_count))

        if is_2d_plot:
            all_predictions = all_predictions.reshape([len(x) + 1 for x in borders])
            fig = plot2d(features_idx, borders,  all_predictions)
        else:
            fig = plot1d(features_idx[0], borders[0], all_predictions)

        if plot:
            try_plot_offline(fig)

        if plot_file:
            save_plot_file(plot_file, "Partial dependence plot for features '{}'".format(features), fig)

        return all_predictions, fig


    def calc_feature_statistics(self, data, target=None, feature=None, prediction_type=None,
                                cat_feature_values=None, plot=True, max_cat_features_on_plot=10,
                                thread_count=-1, plot_file=None):
        """
        Get statistics for the feature using the model, dataset and target.
        To use this function, you should install plotly.

        The catboost model has borders for the float features used in it. The borders divide
        feature values into bins, and the model's prediction depends on the number of the bin where the
        feature value falls in.

        For float features this function takes model's borders and computes
        1) Mean target value for every bin;
        2) Mean model prediction for every bin;
        3) The number of objects in dataset which fall into each bin;
        4) Predictions on varying feature. For every object, varies the feature value
        so that it falls into bin #0, bin #1, ... and counts model predictions.
        Then counts average prediction for each bin.

        For categorical features (only one-hot supported) does the same, but takes feature values
        provided in cat_feature_values instead of borders.

        Parameters
        ----------
        data: numpy.ndarray or pandas.DataFrame or catboost. Pool or dict {'pool_name': pool} if you want several pools
            Data to compute statistics on
        target: numpy.ndarray or pandas.Series or dict {'pool_name': target} if you want several pools or None
            Target corresponding to data
            Use only if data is not catboost.Pool.
        feature: None, int, string, or list of int or strings
            Features indexes or names in pd.DataFrame for which you want to get statistics.
            None, if you need statistics for all features.
        prediction_type: str
            Prediction type used for counting mean_prediction: 'Class', 'Probability' or 'RawFormulaVal'.
            If not specified, is derived from the model.
        cat_feature_values: list or numpy.ndarray or pandas.Series or
                            dict: int or string to list or numpy.ndarray or pandas.Series
            Contains categorical feature values you need to get statistics on.
            Use dict, when parameter 'feature' is a list to specify cat values for different features.
            When parameter 'feature' is int or str, you can just pass list of cat values.
        plot: bool
            Plot statistics.
        max_cat_features_on_plot: int
            If categorical feature takes more than max_cat_features_on_plot different unique values,
            output result on several plots, not more than max_cat_features_on_plot feature values on each.
            Used only if plot=True or plot_file is not None.
        thread_count: int
            Number of threads to use for getting statistics.
        plot_file: str
            Output file for plot statistics.

        Returns
        -------
        dict if parameter 'feature' is int or string, else dict of dicts:
            For each unique feature contain
            python dict with binarized feature statistics.
            For float feature, includes
                    'borders' -- borders for the specified feature in model
                    'binarized_feature' -- numbers of bins where feature values fall
                    'mean_target' -- mean value of target over each bin
                    'mean_prediction' -- mean value of model prediction over each bin
                    'objects_per_bin' -- number of objects per bin
                    'predictions_on_varying_feature' -- averaged over dataset predictions for
                    varying feature (see above)
            For one-hot feature, returns the same, but with 'cat_values' instead of 'borders'
        """
        target_is_none = target is None
        if not isinstance(data, dict):
            data = {'': data}
        if not isinstance(target, dict):
            target = {'': target}
        assert target_is_none or len(data) == len(target), 'inconsistent size of data and target'
        assert target_is_none or target.keys() == data.keys(), 'inconsistent pool_names of data and target'
        for key in data.keys():
            data[key], _ = self._process_predict_input_data(data[key], "get_binarized_statistics", thread_count, target.get(key, None))
        data, pool_names = list(data.values()), list(data.keys())

        if prediction_type is None:
            prediction_type = 'Probability' if self.get_param('loss_function') in ['CrossEntropy', 'Logloss'] \
                else 'RawFormulaVal'

        if prediction_type not in ['Class', 'Probability', 'RawFormulaVal', 'Exponent']:
            raise CatBoostError('Unknown prediction type "{}"'.format(prediction_type))

        if feature is None:
            feature = self.feature_names_

        if cat_feature_values is None:
            cat_feature_values = {}
        else:
            if not isinstance(cat_feature_values, dict):
                if isinstance(features, list):
                    raise CatBoostError('cat_feature_values should be dict when features is a list')
                else:
                    cat_feature_values = {features: cat_feature_values}

        if isinstance(feature, str) or isinstance(feature, int):
            features = [feature]
            is_for_one_feature = True
        else:
            features = feature
            is_for_one_feature = False

        cat_features_nums = []
        float_features_nums = []
        feature_type_mapper = []
        feature_names = []
        feature_name_to_num = {}
        for feature in features:
            if not isinstance(feature, int):
                if self.feature_names_ is None or feature not in self.feature_names_:
                    raise CatBoostError('No feature named "{}" in model'.format(feature))
                feature_num = self.feature_names_.index(feature)
            else:
                feature_num = feature
                feature = self.feature_names_[feature_num]
            if feature in feature_names:
                continue  # There is no reason to count statistics twice for the same feature
            if feature_num in cat_feature_values:
                cat_feature_values[feature] = cat_feature_values[feature_num]
            feature_names.append(feature)
            feature_name_to_num[feature] = feature_num
            feature_type, feature_internal_index = self._object._get_feature_type_and_internal_index(feature_num)
            if feature_type == 'categorical':
                cat_features_nums.append(feature_internal_index)
                feature_type_mapper.append('cat')
            else:
                float_features_nums.append(feature_internal_index)
                feature_type_mapper.append('float')
        results = [self._object._get_binarized_statistics(
            data_item,
            cat_features_nums,
            float_features_nums,
            prediction_type,
            thread_count
        ) for data_item in data]
        # res = [dict,   dict,   ...,   dict,  dict,   dict,   ...,   dict ]
        #        |  stat for cat features  |  |  stat for float features  |
        statistics_by_feature = defaultdict(list)
        to_float_offset = len(cat_features_nums)
        cat_index, float_index = 0, to_float_offset
        for i, type in enumerate(feature_type_mapper):
            feature_name = feature_names[i]
            feature_num = feature_name_to_num[feature_name]
            if type == 'cat':
                if feature_name not in cat_feature_values:
                    cat_feature_values_ = self._object._get_cat_feature_values(data[0], feature_num)
                    cat_feature_values_ = [val for val in cat_feature_values_]
                else:
                    cat_feature_values_ = cat_feature_values[feature_name]
                if not isinstance(cat_feature_values_, ARRAY_TYPES):
                    raise CatBoostError(
                        "Feature '{}' is categorical. "
                        "Please provide values for which you need statistics in cat_feature_values"
                        .format(feature)
                    )
                val_to_hash = dict()
                for val in cat_feature_values_:
                    val_to_hash[val] = self._object._calc_cat_feature_perfect_hash(val, cat_features_nums[cat_index])
                hash_to_val = {hash: val for val, hash in val_to_hash.items()}
                for i, res in enumerate(results):
                    res[cat_index]['cat_values'] = np.array([hash_to_val[i] for i in sorted(hash_to_val.keys())])
                    res[cat_index].pop('borders', None)
                    statistics_by_feature[feature_num].append(res[cat_index])
                cat_index += 1
            else:
                for res in results:
                    statistics_by_feature[feature_num].append(res[float_index])
                float_index += 1
        # now order in statistics_by_feature is the same as in features

        # draw only unique plots
        if plot or plot_file is not None:
            fig = _plot_feature_statistics(
                statistics_by_feature,
                pool_names,
                self.feature_names_,
                max_cat_features_on_plot)
            if plot:
                try_plot_offline(fig)

            if plot_file is not None:
                save_plot_file(plot_file, 'Catboost metrics graph', [fig])

        for key in statistics_by_feature.keys():
            if len(statistics_by_feature[key]) == 1:
                statistics_by_feature[key] = statistics_by_feature[key][0]

        if is_for_one_feature:
            return statistics_by_feature[feature_name_to_num[feature_names[0]]]

        # return dict with possible duplicates
        # (if features from input contains both str and int values appropriate one feature)
        return_stats = {}
        for feature in features:
            if isinstance(feature, int):
                return_stats[feature] = statistics_by_feature[feature]
            else:
                return_stats[feature] = statistics_by_feature[feature_name_to_num[feature]]

        return return_stats

    def _plot_oblivious_tree(self, splits, leaf_values):
        from graphviz import Digraph
        graph = Digraph()

        layer_size = 1
        current_size = 0

        for split_num in range(len(splits) - 1, -2, -1):
            for node_num in range(layer_size):
                if split_num >= 0:
                    node_label = splits[split_num].replace('bin=', 'value>', 1).replace('border=', 'value>', 1)
                    color = 'black'
                    shape = 'ellipse'
                else:
                    node_label = leaf_values[node_num]
                    color = 'red'
                    shape = 'rect'

                try:
                    node_label = node_label.decode("utf-8")
                except:
                    pass

                graph.node(str(current_size), node_label, color=color, shape=shape)

                if current_size > 0:
                    parent = (current_size - 1) // 2
                    edge_label = 'Yes' if current_size % 2 == 0 else 'No'
                    graph.edge(str(parent), str(current_size), edge_label)

                current_size += 1

            layer_size *= 2

        return graph

    def _plot_nonsymmetric_tree(self, splits, leaf_values, step_nodes, node_to_leaf):
        from graphviz import Digraph
        graph = Digraph()

        def plot_leaf(node_idx, graph):
            cur_id = 'leaf_{}'.format(node_to_leaf[node_idx])
            node_label = leaf_values[node_to_leaf[node_idx]]
            graph.node(cur_id, node_label, color='red', shape='rect')
            return cur_id

        def plot_subtree(node_idx, graph):
            if step_nodes[node_idx] == (0, 0):
                return plot_leaf(node_idx, graph)
            else:
                cur_id = 'node_{}'.format(node_idx)
                node_label = splits[node_idx].replace('bin=', 'value>', 1).replace('border=', 'value>', 1)
                graph.node(cur_id, node_label, color='black', shape='ellipse')

                if step_nodes[node_idx][0] == 0:
                    child_id = plot_leaf(node_idx, graph)
                else:
                    child_id = plot_subtree(node_idx + step_nodes[node_idx][0], graph)
                graph.edge(cur_id, child_id, 'No')

                if step_nodes[node_idx][1] == 0:
                    child_id = plot_leaf(node_idx, graph)
                else:
                    child_id = plot_subtree(node_idx + step_nodes[node_idx][1], graph)
                graph.edge(cur_id, child_id, 'Yes')
            return cur_id

        plot_subtree(0, graph)
        return graph

    def plot_tree(self, tree_idx, pool=None):
        pool, _ = self._process_predict_input_data(pool, "plot_tree", thread_count=-1) if pool is not None else (None, None)

        splits = self._get_tree_splits(tree_idx, pool)
        leaf_values = self._get_tree_leaf_values(tree_idx)
        if self._object._is_oblivious():
            return self._plot_oblivious_tree(splits, leaf_values)
        else:
            step_nodes = self._get_tree_step_nodes(tree_idx)
            node_to_leaf = self._get_tree_node_to_leaf(tree_idx)
            return self._plot_nonsymmetric_tree(splits, leaf_values, step_nodes, node_to_leaf)

    def _tune_hyperparams(self, param_grid, X, y=None, cv=3, n_iter=10, partition_random_seed=0,
                          calc_cv_statistics=True, search_by_train_test_split=True,
                          refit=True, shuffle=True, stratified=None, train_size=0.8, verbose=1, plot=False,
                          log_cout=sys.stdout, log_cerr=sys.stderr):

        if refit and self.is_fitted():
            raise CatBoostError("Model was fitted before hyperparameters tuning. You can't change hyperparameters of fitted model.")

        currently_not_supported_params = {
            'ignored_features',
            'input_borders',
            'loss_function',
            'eval_metric'
        }
        if isinstance(param_grid, Mapping):
            param_grid = [param_grid]

        for grid_num, grid in enumerate(param_grid):
            _process_synonyms_groups(grid)
            grid = _params_type_cast(grid)

            for param in currently_not_supported_params:
                if param in grid:
                    raise CatBoostError("Parameter '{}' currently is not supported in grid search".format(param))

            ignored_params = set()

        if X is None:
            raise CatBoostError("X must not be None")

        if y is None and not isinstance(X, PATH_TYPES + (Pool,)):
            raise CatBoostError("y may be None only when X is an instance of catboost.Pool, str or pathlib.Path")

        if not isinstance(param_grid, (Mapping, Iterable)):
            raise TypeError('Parameter grid is not a dict or a list ({!r})'.format(param_grid))

        train_params = self._prepare_train_params(X=X, y=y)
        params = train_params["params"]

        custom_folds = None
        fold_count = 0
        if isinstance(cv, INTEGER_TYPES):
            fold_count = cv
            loss_function = params.get('loss_function', None)
            if stratified is None:
                stratified = isinstance(loss_function, STRING_TYPES) and is_cv_stratified_objective(loss_function)
        else:
            if not hasattr(cv, '__iter__') and not hasattr(cv, 'split'):
                raise AttributeError("cv should be one of possible things:"
                    "\n- None, to use the default 3-fold cross validation,"
                    "\n- integer, to specify the number of folds in a (Stratified)KFold"
                    "\n- one of the scikit-learn splitter classes"
                    " (https://scikit-learn.org/stable/modules/classes.html#splitter-classes)"
                    "\n- An iterable yielding (train, test) splits as arrays of indices")
            custom_folds = cv
            shuffle = False

        if stratified is None:
            loss_function = params.get('loss_function', None)
            stratified = isinstance(loss_function, STRING_TYPES) and is_cv_stratified_objective(loss_function)

        with log_fixup(log_cout, log_cerr), plot_wrapper(plot, [_get_train_dir(params)]):
            cv_result = self._object._tune_hyperparams(
                param_grid, train_params["train_pool"], params, n_iter,
                fold_count, partition_random_seed, shuffle, stratified, train_size,
                search_by_train_test_split, calc_cv_statistics, custom_folds, verbose
            )

        if refit:
            assert not self.is_fitted()
            self.set_params(**cv_result['params'])
            self.fit(X, y, silent=True)
        return cv_result

    def grid_search(self, param_grid, X, y=None, cv=3, partition_random_seed=0,
                    calc_cv_statistics=True, search_by_train_test_split=True,
                    refit=True, shuffle=True, stratified=None, train_size=0.8, verbose=True, plot=False,
                    log_cout=sys.stdout, log_cerr=sys.stderr):
        """
        Exhaustive search over specified parameter values for a model.
        Aafter calling this method model is fitted and can be used, if not specified otherwise (refit=False).

        Parameters
        ----------
        param_grid: dict or list of dictionaries
            Dictionary with parameters names (string) as keys and lists of parameter settings
            to try as values, or a list of such dictionaries, in which case the grids spanned by each
            dictionary in the list are explored.
            This enables searching over any sequence of parameter settings.

        X: numpy.ndarray or pandas.DataFrame or catboost.Pool
            Data to compute statistics on

        y: numpy.ndarray or pandas.Series or None
            Target corresponding to data
            Use only if data is not catboost.Pool.

        cv: int, cross-validation generator or an iterable, optional (default=None)
            Determines the cross-validation splitting strategy. Possible inputs for cv are:
            - None, to use the default 3-fold cross validation,
            - integer, to specify the number of folds in a (Stratified)KFold
            - one of the scikit-learn splitter classes
                (https://scikit-learn.org/stable/modules/classes.html#splitter-classes)
            - An iterable yielding (train, test) splits as arrays of indices.

        partition_random_seed: int, optional (default=0)
            Use this as the seed value for random permutation of the data.
            Permutation is performed before splitting the data for cross validation.
            Each seed generates unique data splits.
            Used only when cv is None or int.

        search_by_train_test_split: bool, optional (default=True)
            If True, source dataset is splitted into train and test parts, models are trained
            on the train part and parameters are compared by loss function score on the test part.
            After that, if calc_cv_statistics=true, statistics on metrics are calculated
            using cross-validation using best parameters and the model is fitted with these parameters.

            If False, every iteration of grid search evaluates results on cross-validation.
            It is recommended to set parameter to True for large datasets, and to False for small datasets.

        calc_cv_statistics: bool, optional (default=True)
            The parameter determines whether quality should be estimated.
            using cross-validation with the found best parameters. Used only when search_by_train_test_split=True.

        refit: bool (default=True)
            Refit an estimator using the best found parameters on the whole dataset.

        shuffle: bool, optional (default=True)
            Shuffle the dataset objects before parameters searching.

        stratified: bool, optional (default=None)
            Perform stratified sampling. True for classification and False otherwise.
            Currently supported only for final cross-validation.

        train_size: float, optional (default=0.8)
            Should be between 0.0 and 1.0 and represent the proportion of the dataset to include in the train split.

        verbose: bool or int, optional (default=True)
            If verbose is int, it determines the frequency of writing metrics to output
            verbose==True is equal to verbose==1
            When verbose==False, there is no messages

        plot : bool, optional (default=False)
            If True, draw train and eval error for every set of parameters in Jupyter notebook

        log_cout: output stream or callback for logging

        log_cerr: error stream or callback for logging

        Returns
        -------
        dict with two fields:
            'params': dict of best found parameters
            'cv_results': dict or pandas.core.frame.DataFrame with cross-validation results
                columns are: test-error-mean  test-error-std  train-error-mean  train-error-std
        """
        if isinstance(param_grid, Mapping):
            param_grid = [param_grid]
        for grid in param_grid:
            if not isinstance(grid, Mapping):
                raise TypeError('Parameter grid is not a dict ({!r})'.format(grid))
            for key in grid:
                if not isinstance(grid[key], Iterable):
                    raise TypeError('Parameter grid value is not iterable (key={!r}, value={!r})'.format(key, grid[key]))

        return self._tune_hyperparams(
            param_grid=param_grid, X=X, y=y, cv=cv, n_iter=-1,
            partition_random_seed=partition_random_seed, calc_cv_statistics=calc_cv_statistics,
            search_by_train_test_split=search_by_train_test_split, refit=refit, shuffle=shuffle,
            stratified=stratified, train_size=train_size, verbose=verbose, plot=plot,
            log_cout=log_cout, log_cerr=log_cerr,
        )

    def randomized_search(self, param_distributions, X, y=None, cv=3, n_iter=10, partition_random_seed=0,
                          calc_cv_statistics=True, search_by_train_test_split=True, refit=True,
                          shuffle=True, stratified=None, train_size=0.8, verbose=True, plot=False,
                          log_cout=sys.stdout, log_cerr=sys.stderr):
        """
        Randomized search on hyper parameters.
        After calling this method model is fitted and can be used, if not specified otherwise (refit=False).

        In contrast to grid_search, not all parameter values are tried out,
        but rather a fixed number of parameter settings is sampled from the specified distributions.
        The number of parameter settings that are tried is given by n_iter.

        Parameters
        ----------
        param_distributions: dict
            Dictionary with parameters names (string) as keys and distributions or lists of parameters to try.
            Distributions must provide a rvs method for sampling (such as those from scipy.stats.distributions).
            If a list is given, it is sampled uniformly.

        X: numpy.ndarray or pandas.DataFrame or catboost.Pool
            Data to compute statistics on

        y: numpy.ndarray or pandas.Series or None
            Target corresponding to data
            Use only if data is not catboost.Pool.

        cv: int, cross-validation generator or an iterable, optional (default=None)
            Determines the cross-validation splitting strategy. Possible inputs for cv are:
            - None, to use the default 3-fold cross validation,
            - integer, to specify the number of folds in a (Stratified)KFold
            - one of the scikit-learn splitter classes
                (https://scikit-learn.org/stable/modules/classes.html#splitter-classes)
            - An iterable yielding (train, test) splits as arrays of indices.

        n_iter: int
            Number of parameter settings that are sampled.
            n_iter trades off runtime vs quality of the solution.

        partition_random_seed: int, optional (default=0)
            Use this as the seed value for random permutation of the data.
            Permutation is performed before splitting the data for cross validation.
            Each seed generates unique data splits.
            Used only when cv is None or int.

        search_by_train_test_split: bool, optional (default=True)
            If True, source dataset is splitted into train and test parts, models are trained
            on the train part and parameters are compared by loss function score on the test part.
            After that, if calc_cv_statistics=true, statistics on metrics are calculated
            using cross-validation using best parameters and the model is fitted with these parameters.

            If False, every iteration of grid search evaluates results on cross-validation.
            It is recommended to set parameter to True for large datasets, and to False for small datasets.

        calc_cv_statistics: bool, optional (default=True)
            The parameter determines whether quality should be estimated.
            using cross-validation with the found best parameters. Used only when search_by_train_test_split=True.

        refit: bool (default=True)
            Refit an estimator using the best found parameters on the whole dataset.

        shuffle: bool, optional (default=True)
            Shuffle the dataset objects before parameters searching.

        stratified: bool, optional (default=None)
            Perform stratified sampling. True for classification and False otherwise.
            Currently supported only for cross-validation.

        train_size: float, optional (default=0.8)
            Should be between 0.0 and 1.0 and represent the proportion of the dataset to include in the train split.

        verbose: bool or int, optional (default=True)
            If verbose is int, it determines the frequency of writing metrics to output
            verbose==True is equal to verbose==1
            When verbose==False, there is no messages

        plot : bool, optional (default=False)
            If True, draw train and eval error for every set of parameters in Jupyter notebook

        log_cout: output stream or callback for logging

        log_cerr: error stream or callback for logging

        Returns
        -------
        dict with two fields:
            'params': dict of best found parameters
            'cv_results': dict or pandas.core.frame.DataFrame with cross-validation results
                columns are: test-error-mean  test-error-std  train-error-mean  train-error-std
        """
        if n_iter <= 0:
            assert CatBoostError("n_iter should be a positive number")
        if not isinstance(param_distributions, Mapping):
            assert CatBoostError("param_distributions should be a dictionary")
        for key in param_distributions:
            if not isinstance(param_distributions[key], Iterable) and not hasattr(param_distributions[key], "rvs"):
                raise TypeError('Parameter grid value is not iterable and do not have \'rvs\' method (key={!r}, value={!r})'.format(key, param_distributions[key]))

        return self._tune_hyperparams(
            param_grid=param_distributions, X=X, y=y, cv=cv, n_iter=n_iter,
            partition_random_seed=partition_random_seed, calc_cv_statistics=calc_cv_statistics,
            search_by_train_test_split=search_by_train_test_split, refit=refit, shuffle=shuffle,
            stratified=stratified, train_size=train_size, verbose=verbose, plot=plot,
            log_cout=log_cout, log_cerr=log_cerr,
        )

    def select_features(self, X, y=None, eval_set=None, features_for_select=None, num_features_to_select=None,
                        algorithm=None, steps=None, shap_calc_type=None, train_final_model=True, verbose=None,
                        logging_level=None, plot=False, log_cout=sys.stdout, log_cerr=sys.stderr):
        """
        Select best features from pool according to loss value.

        Parameters
        ----------
        X : catboost.Pool or list or numpy.ndarray or pandas.DataFrame or pandas.Series
            If not catboost.Pool, 2 dimensional Feature matrix or string - file with dataset.

        y : list or numpy.ndarray or pandas.DataFrame or pandas.Series, optional (default=None)
            Labels, 1 dimensional array like.
            Use only if X is not catboost.Pool.

        eval_set : catboost.Pool or tuple (X, y) or list [(X, y)], optional (default=None)
            Dataset for evaluation.

        features_for_select : str or list of feature indices, names or ranges
            Which features should participate in the selection.
            Format examples:
                - [0, 2, 3, 4, 17]
                - [0, "2-4", 17] (both ends in ranges are inclusive)
                - "0,2-4,20"
                - ["Name0", "Name2", "Name3", "Name4", "Name20"]

        num_features_to_select : positive int
            How many features to select from features_for_select.

        algorithm : EFeaturesSelectionAlgorithm or string, optional (default=RecursiveByShapValues)
            Which algorithm to use for features selection.
            Possible values:
                - RecursiveByPredictionValuesChange
                    Use prediction values change as feature strength, eliminate batch of features at once.
                - RecursiveByLossFunctionChange
                    Use loss function change as feature strength, eliminate batch of features at each step.
                - RecursiveByShapValues
                    Use shap values to estimate loss function change, eliminate features one by one.

        steps : positive int, optional (default=1)
            How many steps should be performed. In other words, how many times a full model will be trained.
            More steps give more accurate results.

        shap_calc_type : EShapCalcType or string, optional (default=Regular)
            Which method to use for calculation of shap values.
            Possible values:
                - Regular
                    Calculate regular SHAP values
                - Approximate
                    Calculate approximate SHAP values
                - Exact
                    Calculate exact SHAP values

        train_final_model : bool, optional (default=True)
            Need to fit model with selected features.

        verbose : bool or int
            If verbose is bool, then if set to True, logging_level is set to Verbose,
            if set to False, logging_level is set to Silent.
            If verbose is int, it determines the frequency of writing metrics to output and
            logging_level is set to Verbose.

        logging_level : string, optional (default=None)
            Possible values:
                - 'Silent'
                - 'Verbose'
                - 'Info'
                - 'Debug'

        plot : bool, optional (default=False)
            If True, draw train and eval error in Jupyter notebook.

        log_cout: output stream or callback for logging

        log_cerr: error stream or callback for logging

        Returns
        -------
        dict with fields:
            'selected_features': list of selected features indices
            'eliminated_features': list of eliminated features indices
        """
        if train_final_model and self.is_fitted():
            raise CatBoostError("Model was already fitted. Set train_final_model to False or use not fitted model.")
        if X is None:
            raise CatBoostError("X must not be None")
        if y is None and not isinstance(X, PATH_TYPES + (Pool,)):
            raise CatBoostError("y may be None only when X is an instance of catboost.Pool, str or pathlib.Path.")
        if isinstance(features_for_select, Iterable) and not isinstance(features_for_select, STRING_TYPES):
            features_for_select = ",".join(map(str, features_for_select))
        if features_for_select is None:
            raise CatBoostError("You should specify features_for_select")
        if num_features_to_select is None:
            raise CatBoostError("You should specify num_features_to_select")

        train_params = self._prepare_train_params(X=X, y=y, eval_set=eval_set, verbose=verbose, logging_level=logging_level)
        params = train_params["params"]
        objective = params.get("loss_function")
        is_custom_objective = objective is not None and not isinstance(objective, string_types)
        if is_custom_objective:
            raise CatBoostError("Custom objective is not supported for features selection")
        params["features_for_select"] = features_for_select
        params["num_features_to_select"] = num_features_to_select
        if algorithm is not None:
            params["features_selection_algorithm"] = enum_from_enum_or_str(EFeaturesSelectionAlgorithm, algorithm).value
        if steps is not None:
            params["features_selection_steps"] = steps
        if shap_calc_type is not None:
            params["shap_calc_type"] = enum_from_enum_or_str(EShapCalcType, shap_calc_type).value
        if train_final_model:
            params["train_final_model"] = True

        train_pool = train_params["train_pool"]
        test_pool = None
        if len(train_params["eval_sets"]) > 1:
            raise CatBoostError("Multiple eval sets are not supported for features selection")
        elif len(train_params["eval_sets"]) == 1:
            test_pool = train_params["eval_sets"][0]

        create_if_not_exist = lambda path: os.mkdir(path) if not os.path.exists(path) else None
        train_dir = _get_train_dir(self.get_params())
        create_if_not_exist(train_dir)
        plot_dirs = []
        for step in range(steps or 1):
            plot_dirs.append(os.path.join(train_dir, 'model-{}'.format(step)))
        if train_final_model:
            plot_dirs.append(os.path.join(train_dir, 'model-final'))
        for plot_dir in plot_dirs:
            create_if_not_exist(plot_dir)

        with log_fixup(log_cout, log_cerr), plot_wrapper(plot, plot_dirs):
            summary = self._object._select_features(train_pool, test_pool, params)

        if train_final_model:
            self._set_trained_model_attributes()

        if plot:
            fig = plot_features_selection_loss_graph(summary)
            fig.show()

        return summary

    def _convert_to_asymmetric_representation(self):
        self._object._convert_oblivious_to_asymmetric()

class CatBoostClassifier(CatBoost):
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
        range: [1,65535] on CPU, [1,255] on GPU
    feature_border_type : string, [default='GreedyLogSum']
        The binarization mode in numeric features binarization. Used in the preliminary calculation.
        Possible values:
            - 'Median'
            - 'Uniform'
            - 'UniformAndQuantiles'
            - 'GreedyLogSum'
            - 'MaxLogSum'
            - 'MinEntropy'
    per_float_feature_quantization : list of strings, [default=None]
        List of float binarization descriptions.
        Format : described in documentation on catboost.ai
        Example 1: ['0:1024'] means that feature 0 will have 1024 borders.
        Example 2: ['0:border_count=1024', '1:border_count=1024', ...] means that two first features have 1024 borders.
        Example 3: ['0:nan_mode=Forbidden,border_count=32,border_type=GreedyLogSum',
                    '1:nan_mode=Forbidden,border_count=32,border_type=GreedyLogSum'] - defines more quantization properties for first two features.
    input_borders : string or pathlib.Path, [default=None]
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
    target_border: float, [default=None]
        Border for target binarization.
    classes_count : int, [default=None]
        The upper limit for the numeric class label.
        Defines the number of classes for multiclassification.
        Only non-negative integers can be specified.
        The given integer should be greater than any of the target values.
        If this parameter is specified the labels for all classes in the input dataset
        should be smaller than the given value.
        If several of 'classes_count', 'class_weights', 'class_names' parameters are defined
        the numbers of classes specified by each of them must be equal.
    class_weights : list or dict, [default=None]
        Classes weights. The values are used as multipliers for the object weights.
        If None, all classes are supposed to have weight one.
        If list - class weights in order of class_names or sequential classes if class_names is undefined
        If dict - dict of class_name -> class_weight.
        If several of 'classes_count', 'class_weights', 'class_names' parameters are defined
        the numbers of classes specified by each of them must be equal.
    auto_class_weights : string [default=None]
        Enables automatic class weights calculation. Possible values:
            - Balanced  # weight = maxSummaryClassWeight / summaryClassWeight, statistics determined from train pool
            - SqrtBalanced  # weight = sqrt(maxSummaryClassWeight / summaryClassWeight)
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
        Indices or names of features that should be excluded when training.
    train_dir : string or pathlib.Path, [default=None]
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
    snapshot_file : string or pathlib.Path, [default=None]
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
        Default bootstrap is Bayesian for GPU and MVS for CPU.
        Poisson bootstrap is supported only on GPU.
        MVS bootstrap is supported only on CPU.

    subsample : float, [default=None]
        Sample rate for bagging. This parameter can be used Poisson or Bernoully bootstrap types.

    mvs_reg : float, [default is set automatically at each iteration based on gradient distribution]
        Regularization parameter for MVS sampling algorithm

    monotone_constraints : list or numpy.ndarray or string or dict, [default=None]
        Monotone constraints for features.

    feature_weights : list or numpy.ndarray or string or dict, [default=None]
        Coefficient to multiply split gain with specific feature use. Should be non-negative.

    penalties_coefficient : float, [default=1]
        Common coefficient for all penalties. Should be non-negative.

    first_feature_use_penalties : list or numpy.ndarray or string or dict, [default=None]
        Penalties to first use of specific feature in model. Should be non-negative.

    per_object_feature_penalties : list or numpy.ndarray or string or dict, [default=None]
        Penalties for first use of feature for each object. Should be non-negative.

    sampling_frequency : string, [default=PerTree]
        Frequency to sample weights and objects when building trees.
        Possible values:
            - 'PerTree' - Before constructing each new tree
            - 'PerTreeLevel' - Before choosing each new split of a tree

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

    sparse_features_conflict_fraction : float, [default=0.0]
        CPU only. Maximum allowed fraction of conflicting non-default values for features in exclusive features bundle.
        Should be a real value in [0, 1) interval.

    grow_policy : string, [SymmetricTree,Lossguide,Depthwise], [default=SymmetricTree]
        The tree growing policy. It describes how to perform greedy tree construction.

    min_data_in_leaf : int, [default=1].
        The minimum training samples count in leaf.
        CatBoost will not search for new splits in leaves with samples count less than min_data_in_leaf.
        This parameter is used only for Depthwise and Lossguide growing policies.

    max_leaves : int, [default=31],
        The maximum leaf count in resulting tree.
        This parameter is used only for Lossguide growing policy.

    score_function : string, possible values L2, Cosine, NewtonL2, NewtonCosine, [default=Cosine]
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

    num_leaves : int, synonym for max_leaves.

    min_child_samples : int, synonym for min_data_in_leaf

    eta : float, synonym for learning_rate.

    max_bin : float, synonym for border_count.

    scale_pos_weight : float, synonym for class_weights.
        Can be used only for binary classification. Sets weight multiplier for
        class 1 to scale_pos_weight value.

    metadata : dict, string to string key-value pairs to be stored in model metadata storage

    early_stopping_rounds : int
        Synonym for od_wait. Only one of these parameters should be set.

    cat_features : list or numpy.ndarray, [default=None]
        If not None, giving the list of Categ features indices or names (names are represented as strings).
        If it contains feature names, feature names must be defined for the training dataset passed to 'fit'.

    text_features : list or numpy.ndarray, [default=None]
        If not None, giving the list of Text features indices or names (names are represented as strings).
        If it contains feature names, feature names must be defined for the training dataset passed to 'fit'.

    embedding_features : list or numpy.ndarray, [default=None]
        If not None, giving the list of Embedding features indices or names (names are represented as strings).
        If it contains feature names, feature names must be defined for the training dataset passed to 'fit'.

    leaf_estimation_backtracking : string, [default=None]
        Type of backtracking during gradient descent.
        Possible values:
            - 'No' - never backtrack; supported on CPU and GPU
            - 'AnyImprovement' - reduce the descent step until the value of loss function is less than before the step; supported on CPU and GPU
            - 'Armijo' - reduce the descent step until Armijo condition is satisfied; supported on GPU only

    model_shrink_rate : float, [default=0]
        This parameter enables shrinkage of model at the start of each iteration. CPU only.
        For Constant mode shrinkage coefficient is calculated as (1 - model_shrink_rate * learning_rate).
        For Decreasing mode shrinkage coefficient is calculated as (1 - model_shrink_rate / iteration).
        Shrinkage coefficient should be in [0, 1).

    model_shrink_mode : string, [default=None]
        Mode of shrinkage coefficient calculation. CPU only.
        Possible values:
            - 'Constant' - Shrinkage coefficient is constant at each iteration.
            - 'Decreasing' - Shrinkage coefficient decreases at each iteration.

    langevin : bool, [default=False]
        Enables the Stochastic Gradient Langevin Boosting. CPU only.

    diffusion_temperature : float, [default=0]
        Langevin boosting diffusion temperature. CPU only.

    posterior_sampling : bool, [default=False]
        Set group of parameters for further use Uncertainty prediction:
            - Langevin = True
            - Model Shrink Rate = 1/(2N), where N is dataset size
            - Model Shrink Mode = Constant
            - Diffusion-temperature = N, where N is dataset size. CPU only.

    boost_from_average : bool, [default=True for RMSE, False for other losses]
        Enables to initialize approx values by best constant value for specified loss function.
        Available for RMSE, Logloss, CrossEntropy, Quantile and MAE.

    tokenizers : list of dicts,
        Each dict is a tokenizer description. Example:
        ```
        [
            {
                'tokenizer_id': 'Tokenizer',  # Tokeinzer identifier.
                'lowercasing': 'false',  # Possible values: 'true', 'false'.
                'number_process_policy': 'LeaveAsIs',  # Possible values: 'Skip', 'LeaveAsIs', 'Replace'.
                'number_token': '%',  # Rarely used character. Used in conjunction with Replace NumberProcessPolicy.
                'separator_type': 'ByDelimiter',  # Possible values: 'ByDelimiter', 'BySense'.
                'delimiter': ' ',  # Used in conjunction with ByDelimiter SeparatorType.
                'split_by_set': 'false',  # Each single character in delimiter used as individual delimiter.
                'skip_empty': 'true',  # Possible values: 'true', 'false'.
                'token_types': ['Word', 'Number', 'Unknown'],  # Used in conjunction with BySense SeparatorType.
                    # Possible values: 'Word', 'Number', 'Punctuation', 'SentenceBreak', 'ParagraphBreak', 'Unknown'.
                'subtokens_policy': 'SingleToken',  # Possible values:
                    # 'SingleToken' - All subtokens are interpreted as single token).
                    # 'SeveralTokens' - All subtokens are interpreted as several token.
            },
            ...
        ]
        ```

    dictionaries : list of dicts,
        Each dict is a tokenizer description. Example:
        ```
        [
            {
                'dictionary_id': 'Dictionary',  # Dictionary identifier.
                'token_level_type': 'Word',  # Possible values: 'Word', 'Letter'.
                'gram_order': '1',  # 1 for Unigram, 2 for Bigram, ...
                'skip_step': '0',  # 1 for 1-skip-gram, ...
                'end_of_word_token_policy': 'Insert',  # Possible values: 'Insert', 'Skip'.
                'end_of_sentence_token_policy': 'Skip',  # Possible values: 'Insert', 'Skip'.
                'occurrence_lower_bound': '3',  # The lower bound of token occurrences in the text to include it in the dictionary.
                'max_dictionary_size': '50000',  # The max dictionary size.
            },
            ...
        ]
        ```

    feature_calcers : list of strings,
        Each string is a calcer description. Example:
        ```
        [
            'NaiveBayes',
            'BM25',
            'BoW:top_tokens_count=2000',
        ]
        ```

    text_processing : dict,
        Text processging description.
    """

    _estimator_type = 'classifier'

    def __init__(
        self,
        iterations=None,
        learning_rate=None,
        depth=None,
        l2_leaf_reg=None,
        model_size_reg=None,
        rsm=None,
        loss_function=None,
        border_count=None,
        feature_border_type=None,
        per_float_feature_quantization=None,
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
        target_border=None,
        classes_count=None,
        class_weights=None,
        auto_class_weights=None,
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
        mvs_reg=None,
        sampling_unit=None,
        sampling_frequency=None,
        dev_score_calc_obj_block_size=None,
        dev_efb_max_buckets=None,
        sparse_features_conflict_fraction=None,
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
        min_child_samples=None,
        max_leaves=None,
        num_leaves=None,
        score_function=None,
        leaf_estimation_backtracking=None,
        ctr_history_unit=None,
        monotone_constraints=None,
        feature_weights=None,
        penalties_coefficient=None,
        first_feature_use_penalties=None,
        per_object_feature_penalties=None,
        model_shrink_rate=None,
        model_shrink_mode=None,
        langevin=None,
        diffusion_temperature=None,
        posterior_sampling=None,
        boost_from_average=None,
        text_features=None,
        tokenizers=None,
        dictionaries=None,
        feature_calcers=None,
        text_processing=None,
        embedding_features=None,
        callback=None
    ):
        params = {}
        not_params = ["not_params", "self", "params", "__class__"]
        for key, value in iteritems(locals().copy()):
            if key not in not_params and value is not None:
                params[key] = value

        super(CatBoostClassifier, self).__init__(params)

    def fit(self, X, y=None, cat_features=None, text_features=None, embedding_features=None, sample_weight=None, baseline=None, use_best_model=None,
            eval_set=None, verbose=None, logging_level=None, plot=False, column_description=None,
            verbose_eval=None, metric_period=None, silent=None, early_stopping_rounds=None,
            save_snapshot=None, snapshot_file=None, snapshot_interval=None, init_model=None, callbacks=None,
            log_cout=sys.stdout, log_cerr=sys.stderr):
        """
        Fit the CatBoostClassifier model.

        Parameters
        ----------
        X : catboost.Pool or list or numpy.ndarray or pandas.DataFrame or pandas.Series
            If not catboost.Pool, 2 dimensional Feature matrix or string - file with dataset.

        y : list or numpy.ndarray or pandas.DataFrame or pandas.Series, optional (default=None)
            Labels, 1 dimensional array like.
            Use only if X is not catboost.Pool.

        cat_features : list or numpy.ndarray, optional (default=None)
            If not None, giving the list of Categ columns indices.
            Use only if X is not catboost.Pool.

        text_features : list or numpy.ndarray, optional (default=None)
            If not None, giving the list of Text columns indices.
            Use only if X is not catboost.Pool.

        embedding_features : list or numpy.ndarray, optional (default=None)
            If not None, giving the list of Embedding columns indices.
            Use only if X is not catboost.Pool.

        sample_weight : list or numpy.ndarray or pandas.DataFrame or pandas.Series, optional (default=None)
            Instance weights, 1 dimensional array like.

        baseline : list or numpy.ndarray, optional (default=None)
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

        snapshot_file : string or pathlib.Path, [default=None]
            Learn progress snapshot file path, if None will use default filename

        snapshot_interval: int, [default=600]
            Interval between saving snapshots (seconds)

        init_model : CatBoost class or string or pathlib.Path, [default=None]
            Continue training starting from the existing model.
            If this parameter is a string or pathlib.Path, load initial model from the path specified by this string.

        callbacks : list, optional (default=None)
            List of callback objects that are applied at end of each iteration.

        log_cout: output stream or callback for logging

        log_cerr: error stream or callback for logging

        Returns
        -------
        model : CatBoost
        """

        params = self._init_params.copy()
        _process_synonyms(params)
        if 'loss_function' in params:
            CatBoostClassifier._check_is_compatible_loss(params['loss_function'])

        self._fit(X, y, cat_features, text_features, embedding_features, None, sample_weight, None, None, None, None, baseline, use_best_model,
                  eval_set, verbose, logging_level, plot, column_description, verbose_eval, metric_period,
                  silent, early_stopping_rounds, save_snapshot, snapshot_file, snapshot_interval, init_model, callbacks, log_cout, log_cerr)
        return self

    def predict(self, data, prediction_type='Class', ntree_start=0, ntree_end=0, thread_count=-1, verbose=None, task_type="CPU"):
        """
        Predict with data.

        Parameters
        ----------
        data : catboost.Pool or list of features or list of lists or numpy.ndarray or pandas.DataFrame or pandas.Series
                or catboost.FeaturesData
            Data to apply model on.
            If data is a simple list (not list of lists) or a one-dimensional numpy.ndarray it is interpreted
            as a list of features for a single object.

        prediction_type : string, optional (default='Class')
            Can be:
            - 'RawFormulaVal' : return raw formula value.
            - 'Class' : return class label.
            - 'Probability' : return probability for every class.
            - 'LogProbability' : return log probability for every class.

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
                - 'Class' : return class label.
                - 'Probability' : return one-dimensional numpy.ndarray with probability for every class.
                - 'LogProbability' : return one-dimensional numpy.ndarray with
                  log probability for every class.
            otherwise numpy.ndarray, with values that depend on prediction_type value:
                - 'RawFormulaVal' : one-dimensional array of raw formula value for each object.
                - 'Class' : one-dimensional array of class label for each object.
                - 'Probability' : two-dimensional numpy.ndarray with shape (number_of_objects x number_of_classes)
                  with probability for every class for each object.
                - 'LogProbability' : two-dimensional numpy.ndarray with shape (number_of_objects x number_of_classes)
                  with log probability for every class for each object.
        """
        return self._predict(data, prediction_type, ntree_start, ntree_end, thread_count, verbose, 'predict', task_type)

    def predict_proba(self, X, ntree_start=0, ntree_end=0, thread_count=-1, verbose=None, task_type="CPU"):
        """
        Predict class probability with X.

        Parameters
        ----------
        X : catboost.Pool or list of features or list of lists or numpy.ndarray or pandas.DataFrame or pandas.Series
                or catboost.FeaturesData
            Data to apply model on.
            If X is a simple list (not list of lists) or a one-dimensional numpy.ndarray it is interpreted
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
            If X is for a single object
                return one-dimensional numpy.ndarray with probability for every class.
            otherwise
                return two-dimensional numpy.ndarray with shape (number_of_objects x number_of_classes)
                with probability for every class for each object.
        """
        return self._predict(X, 'Probability', ntree_start, ntree_end, thread_count, verbose, 'predict_proba', task_type)


    def predict_log_proba(self, data, ntree_start=0, ntree_end=0, thread_count=-1, verbose=None, task_type="CPU"):
        """
        Predict class log probability with data.

        Parameters
        ----------
        data : catboost.Pool or list of features or list of lists or numpy.ndarray or pandas.DataFrame or pandas.Series
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
                return one-dimensional numpy.ndarray with log probability for every class.
            otherwise
                return two-dimensional numpy.ndarray with shape (number_of_objects x number_of_classes)
                with log probability for every class for each object.
        """
        return self._predict(data, 'LogProbability', ntree_start, ntree_end, thread_count, verbose, 'predict_log_proba', task_type)

    def staged_predict(self, data, prediction_type='Class', ntree_start=0, ntree_end=0, eval_period=1, thread_count=-1, verbose=None):
        """
        Predict target at each stage for data.

        Parameters
        ----------
        data : catboost.Pool or list of features or list of lists or numpy.ndarray or pandas.DataFrame or pandas.Series
                or catboost.FeaturesData
            Data to apply model on.
            If data is a simple list (not list of lists) or a one-dimensional numpy.ndarray it is interpreted
            as a list of features for a single object.

        prediction_type : string, optional (default='Class')
            Can be:
            - 'RawFormulaVal' : return raw formula value.
            - 'Class' : return class label.
            - 'Probability' : return probability for every class.
            - 'LogProbability' : return log probability for every class.

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
                - 'LogProbability' : return one-dimensional numpy.ndarray with
                  log probability for every class.
            otherwise numpy.ndarray, with values that depend on prediction_type value:
                - 'RawFormulaVal' : one-dimensional array of raw formula value for each object.
                - 'Class' : one-dimensional array of class label for each object.
                - 'Probability' : two-dimensional numpy.ndarray with shape (number_of_objects x number_of_classes)
                  with probability for every class for each object.
                - 'LogProbability' : two-dimensional numpy.ndarray with shape (number_of_objects x number_of_classes)
                  with log probability for every class for each object.
        """
        return self._staged_predict(data, prediction_type, ntree_start, ntree_end, eval_period, thread_count, verbose, 'staged_predict')

    def staged_predict_proba(self, data, ntree_start=0, ntree_end=0, eval_period=1, thread_count=-1, verbose=None):
        """
        Predict classification target at each stage for data.

        Parameters
        ----------
        data : catboost.Pool or list of features or list of lists or numpy.ndarray or pandas.DataFrame or pandas.Series
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


    def staged_predict_log_proba(self, data, ntree_start=0, ntree_end=0, eval_period=1, thread_count=-1, verbose=None):
        """
        Predict classification target at each stage for data.

        Parameters
        ----------
        data : catboost.Pool or list of features or list of lists or numpy.ndarray or pandas.DataFrame or pandas.Series
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
                return one-dimensional numpy.ndarray with log probability for every class.
            otherwise
                return two-dimensional numpy.ndarray with shape (number_of_objects x number_of_classes)
                with log probability for every class for each object.
        """
        return self._staged_predict(data, 'LogProbability', ntree_start, ntree_end, eval_period, thread_count, verbose, 'staged_predict_log_proba')

    def score(self, X, y=None):
        """
        Calculate accuracy.

        Parameters
        ----------
        X : catboost.Pool or list or numpy.ndarray or pandas.DataFrame or pandas.Series
            Data to apply model on.
        y : list or numpy.ndarray
            True labels.

        Returns
        -------
        accuracy : float
        """
        if isinstance(X, Pool):
            if y is not None:
                raise CatBoostError("Wrong initializing y: X is catboost.Pool object, y must be initialized inside catboost.Pool.")
            y = X.get_label()
            if y is None:
                raise CatBoostError("Label in X has not initialized.")
        if isinstance(y, DataFrame):
            if len(y.columns) != 1:
                raise CatBoostError("y is DataFrame and has {} columns, but must have exactly one.".format(len(y.columns)))
            y = y[y.columns[0]]
        elif y is None:
            raise CatBoostError("y should be specified.")
        y = np.array(y)
        predicted_classes = self._predict(
            X,
            prediction_type='Class',
            ntree_start=0,
            ntree_end=0,
            thread_count=-1,
            verbose=None,
            parent_method_name='score'
        ).reshape(-1)
        if np.issubdtype(predicted_classes.dtype, np.number):
            if np.issubdtype(y.dtype, np.character):
                raise CatBoostError('predicted classes have numeric type but specified y contains strings')
        else:
            if np.issubdtype(y.dtype, np.number):
                raise CatBoostError('predicted classes have string type but specified y is numeric')
            elif np.issubdtype(y.dtype, np.bool_):
                raise CatBoostError('predicted classes have string type but specified y is boolean')
        return np.mean(np.array(predicted_classes) == np.array(y))

    def set_probability_threshold(self, binclass_probability_threshold=None):
        """
        Set a threshold for classes separation in binary classification task for a trained model.
        :param binclass_probability_threshold: float number in [0, 1] or None to discard it
        """
        if not self.is_fitted():
            raise CatBoostError("You can't set probability threshold for not fitted model.")
        metadata = self.get_metadata()
        if binclass_probability_threshold is None:
            if 'binclass_probability_threshold' in metadata.keys():
                del metadata['binclass_probability_threshold']
        else:
            if not isinstance(binclass_probability_threshold, FLOAT_TYPES):
                raise CatBoostError("binclass_probability_threshold must have float type")
            assert 0. <= binclass_probability_threshold <= 1.,\
                "Please provide correct probability for binclass_probability_threshold argument in [0, 1] range"
            self.get_metadata()['binclass_probability_threshold'] = str(binclass_probability_threshold)

    def get_probability_threshold(self):
        """
        Get a threshold for classes separation in binary classification task
        """
        if not self.is_fitted():
            raise CatBoostError("Not fitted models don't have a probability threshold.")
        return self._object._get_binclass_probability_threshold()

    @staticmethod
    def _check_is_compatible_loss(loss_function):
        if isinstance(loss_function, str) and not CatBoost._is_classification_objective(loss_function):
            raise CatBoostError("Invalid loss_function='{}': for classifier use "
                                "Logloss, CrossEntropy, MultiClass, MultiClassOneVsAll or custom objective object".format(loss_function))


class CatBoostRegressor(CatBoost):
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
        'SurvivalAft:dist=value;scale=value'
    """

    _estimator_type = 'regressor'

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
        per_float_feature_quantization=None,
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
        target_border=None,
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
        mvs_reg=None,
        sampling_frequency=None,
        sampling_unit=None,
        dev_score_calc_obj_block_size=None,
        dev_efb_max_buckets=None,
        sparse_features_conflict_fraction=None,
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
        min_child_samples=None,
        max_leaves=None,
        num_leaves=None,
        score_function=None,
        leaf_estimation_backtracking=None,
        ctr_history_unit=None,
        monotone_constraints=None,
        feature_weights=None,
        penalties_coefficient=None,
        first_feature_use_penalties=None,
        per_object_feature_penalties=None,
        model_shrink_rate=None,
        model_shrink_mode=None,
        langevin=None,
        diffusion_temperature=None,
        posterior_sampling=None,
        boost_from_average=None
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
            save_snapshot=None, snapshot_file=None, snapshot_interval=None, init_model=None, callbacks=None,
            log_cout=sys.stdout, log_cerr=sys.stderr):
        """
        Fit the CatBoost model.

        Parameters
        ----------
        X : catboost.Pool or list or numpy.ndarray or pandas.DataFrame or pandas.Series
            If not catboost.Pool, 2 dimensional Feature matrix or string - file with dataset.

        y : list or numpy.ndarray or pandas.DataFrame or pandas.Series, optional (default=None)
            Labels, 1 dimensional array like.
            Use only if X is not catboost.Pool.

        cat_features : list or numpy.ndarray, optional (default=None)
            If not None, giving the list of Categ columns indices.
            Use only if X is not catboost.Pool.

        sample_weight : list or numpy.ndarray or pandas.DataFrame or pandas.Series, optional (default=None)
            Instance weights, 1 dimensional array like.

        baseline : list or numpy.ndarray, optional (default=None)
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

        snapshot_file : string or pathlib.Path, [default=None]
            Learn progress snapshot file path, if None will use default filename

        snapshot_interval: int, [default=600]
            Interval between saving snapshots (seconds)

        init_model : CatBoost class or string or pathlib.Path, [default=None]
            Continue training starting from the existing model.
            If this parameter is a string or pathlib.Path, load initial model from the path specified by this string.

        callbacks : list, optional (default=None)
            List of callback objects that are applied at end of each iteration.

        log_cout: output stream or callback for logging

        log_cerr: error stream or callback for logging

        Returns
        -------
        model : CatBoost
        """

        params = deepcopy(self._init_params)
        _process_synonyms(params)
        if 'loss_function' in params:
            CatBoostRegressor._check_is_compatible_loss(params['loss_function'])

        return self._fit(X, y, cat_features, None, None, None, sample_weight, None, None, None, None, baseline,
                         use_best_model, eval_set, verbose, logging_level, plot, column_description,
                         verbose_eval, metric_period, silent, early_stopping_rounds,
                         save_snapshot, snapshot_file, snapshot_interval, init_model, callbacks, log_cout, log_cerr)

    def predict(self, data, prediction_type=None, ntree_start=0, ntree_end=0, thread_count=-1, verbose=None, task_type="CPU"):
        """
        Predict with data.

        Parameters
        ----------
        data : catboost.Pool or list of features or list of lists or numpy.ndarray or pandas.DataFrame or pandas.Series
                or catboost.FeaturesData
            Data to apply model on.
            If data is a simple list (not list of lists) or a one-dimensional numpy.ndarray it is interpreted
            as a list of features for a single object.

        prediction_type : string, optional (default='RawFormulaVal')
            Can be:
            - 'RawFormulaVal' : return raw formula value.
            - 'Exponent' : return Exponent of raw formula value.

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
        if prediction_type is None:
            prediction_type = self._get_default_prediction_type()
        return self._predict(data, prediction_type, ntree_start, ntree_end, thread_count, verbose, 'predict', task_type)

    def staged_predict(self, data, prediction_type='RawFormulaVal', ntree_start=0, ntree_end=0, eval_period=1, thread_count=-1, verbose=None):
        """
        Predict target at each stage for data.

        Parameters
        ----------
        data : catboost.Pool or list of features or list of lists or numpy.ndarray or pandas.DataFrame or pandas.Series
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
        return self._staged_predict(data, prediction_type, ntree_start, ntree_end, eval_period, thread_count, verbose, 'staged_predict')

    def score(self, X, y=None):
        """
        Calculate R^2.

        Parameters
        ----------
        X : catboost.Pool or list or numpy.ndarray or pandas.DataFrame or pandas.Series
            Data to apply model on.
        y : list or numpy.ndarray
            True labels.

        Returns
        -------
        R^2 : float
        """
        if isinstance(X, Pool):
            if y is not None:
                raise CatBoostError("Wrong initializing y: X is catboost.Pool object, y must be initialized inside catboost.Pool.")
            y = X.get_label()
            if y is None:
                raise CatBoostError("Label in X has not initialized.")
        elif y is None:
            raise CatBoostError("y should be specified.")
        y = np.array(y, dtype=np.float64)
        predictions = self._predict(
            X,
            prediction_type=self._get_default_prediction_type(),
            ntree_start=0,
            ntree_end=0,
            thread_count=-1,
            verbose=None,
            parent_method_name='score'
        )
        loss = self._object._get_loss_function_name()
        if loss == 'RMSEWithUncertainty':
            predictions = predictions[:, 0]
        total_sum_of_squares = np.sum((y - y.mean(axis=0)) ** 2)
        residual_sum_of_squares = np.sum((y - predictions) ** 2)
        return 1 - residual_sum_of_squares / total_sum_of_squares

    @staticmethod
    def _check_is_compatible_loss(loss_function):
        is_regression = CatBoost._is_regression_objective(loss_function) or CatBoost._is_multiregression_objective(loss_function) or CatBoost._is_survivalregression_objective(loss_function)
        if isinstance(loss_function, str) and not is_regression:
            raise CatBoostError("Invalid loss_function='{}': for regressor use "
                                "RMSE, MultiRMSE, SurvivalAft, MAE, Quantile, LogLinQuantile, Poisson, MAPE, Lq or custom objective object".format(loss_function))

    def _get_default_prediction_type(self):
        # TODO(ilyzhin) change on get_all_params after MLTOOLS-4758
        params = deepcopy(self._init_params)
        _process_synonyms(params)
        loss_function = params.get('loss_function')
        if loss_function and isinstance(loss_function, str):
            if loss_function.startswith('Poisson') or loss_function.startswith('Tweedie'):
                return 'Exponent'
            if loss_function == 'RMSEWithUncertainty':
                return 'RMSEWithUncertainty'
        return 'RawFormulaVal'

class CatBoostRanker(CatBoost):
    """
    Implementation of the scikit-learn API for CatBoost ranking.
    Parameters
    ----------
    Like in CatBoostClassifier, except loss_function, classes_count, class_names and class_weights
    loss_function : string, [default='YetiRank']
        'YetiRank'
        'YetiRankPairwise'
        'StochasticFilter'
        'StochasticRank'
        'QueryCrossEntropy'
        'QueryRMSE'
        'QuerySoftMax'
        'PairLogit'
        'PairLogitPairwise'
    """

    _estimator_type = 'ranker'

    def __init__(
        self,
        iterations=None,
        learning_rate=None,
        depth=None,
        l2_leaf_reg=None,
        model_size_reg=None,
        rsm=None,
        loss_function='YetiRank',
        border_count=None,
        feature_border_type=None,
        per_float_feature_quantization=None,
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
        target_border=None,
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
        mvs_reg=None,
        sampling_frequency=None,
        sampling_unit=None,
        dev_score_calc_obj_block_size=None,
        dev_efb_max_buckets=None,
        sparse_features_conflict_fraction=None,
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
        min_child_samples=None,
        max_leaves=None,
        num_leaves=None,
        score_function=None,
        leaf_estimation_backtracking=None,
        ctr_history_unit=None,
        monotone_constraints=None,
        feature_weights=None,
        penalties_coefficient=None,
        first_feature_use_penalties=None,
        per_object_feature_penalties=None,
        model_shrink_rate=None,
        model_shrink_mode=None,
        langevin=None,
        diffusion_temperature=None,
        posterior_sampling=None,
        boost_from_average=None,
        text_features=None,
        tokenizers=None,
        dictionaries=None,
        feature_calcers=None,
        text_processing=None,
        embedding_features=None
    ):
        params = {}
        not_params = ["not_params", "self", "params", "__class__"]
        for key, value in iteritems(locals().copy()):
            if key not in not_params and value is not None:
                params[key] = value

        super(CatBoostRanker, self).__init__(params)

    def fit(self, X, y=None, group_id=None, cat_features=None, text_features=None,
            embedding_features=None, pairs=None, sample_weight=None, group_weight=None,
            subgroup_id=None, pairs_weight=None, baseline=None, use_best_model=None,
            eval_set=None, verbose=None, logging_level=None, plot=False, column_description=None,
            verbose_eval=None, metric_period=None, silent=None, early_stopping_rounds=None,
            save_snapshot=None, snapshot_file=None, snapshot_interval=None, init_model=None, callbacks=None,
            log_cout=sys.stdout, log_cerr=sys.stderr):
        """
        Fit the CatBoostRanker model.
        Parameters
        ----------
        X : catboost.Pool or list or numpy.ndarray or pandas.DataFrame or pandas.Series
            If not catboost.Pool, 2 dimensional Feature matrix or string - file with dataset.
        y : list or numpy.ndarray or pandas.DataFrame or pandas.Series, optional (default=None)
            Labels, 1 dimensional array like.
            Use only if X is not catboost.Pool.
        group_id : numpy.ndarray or pandas.DataFrame or pandas.Series, optional (default=None)
            Ranking groups, 1 dimensional array like.
            Use only if X is not catboost.Pool.
        cat_features : list or numpy.ndarray, optional (default=None)
            If not None, giving the list of Categ columns indices.
            Use only if X is not catboost.Pool.
        text_features : list or numpy.ndarray, optional (default=None)
            If not None, giving the list of Text columns indices.
            Use only if X is not catboost.Pool.
        embedding_features : list or numpy.ndarray, optional (default=None)
            If not None, giving the list of Embedding columns indices.
            Use only if X is not catboost.Pool.
        pairs : list or numpy.ndarray or pandas.DataFrame, optional (default=None)
            The pairs description in the form of a two-dimensional matrix of shape N by 2:
            N is the number of pairs.
            The first element of the pair is the zero-based index of the winner object from the input dataset for pairwise comparison.
            The second element of the pair is the zero-based index of the loser object from the input dataset for pairwise comparison.
        sample_weight : list or numpy.ndarray or pandas.DataFrame or pandas.Series, optional (default=None)
            Instance weights, 1 dimensional array like.
        group_weight : list or numpy.ndarray (default=None)
            The weights of all objects within the defined groups from the input data in the form of one-dimensional array-like data.
            Used for calculating the final values of trees. By default, it is set to 1 for all objects in all groups.
            Only a weight or group_weight parameter can be used at a time
        subgroup_id : list or numpy.ndarray (default=None)
            Subgroup identifiers for all input objects. Supported identifier types are:
            int
            string types (string or unicode for Python 2 and bytes or string for Python 3).
        pairs_weight : list or numpy.ndarray (default=None)
            The weight of each input pair of objects in the form of one-dimensional array-like pairs.
            The number of given values must match the number of specified pairs.
            This information is used for calculation and optimization of Pairwise metrics .
            By default, it is set to 1 for all pairs.
        baseline : list or numpy.ndarray, optional (default=None)
            If not None, giving 2 dimensional array like data.
            Use only if X is not catboost.Pool.
        use_best_model : bool, optional (default=None)
            Flag to use best model
        eval_set : catboost.Pool or list, optional (default=None)
            A list of (X, y) tuple pairs to use as a validation set for early-stopping
        verbose : bool or int
            If verbose is bool, then if set to True, logging_level is set to Verbose,
            if set to False, logging_level is set to Silent.
            If verbose is int, it determines the frequency of writing metrics to output and
            logging_level is set to Verbose.
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
        metric_period : int
            Frequency of evaluating metrics.
        silent : bool
            If silent is True, logging_level is set to Silent.
            If silent is False, logging_level is set to Verbose.
        early_stopping_rounds : int
            Activates Iter overfitting detector with od_wait set to early_stopping_rounds.
        save_snapshot : bool, [default=None]
            Enable progress snapshotting for restoring progress after crashes or interruptions
        snapshot_file : string or pathlib.Path, [default=None]
            Learn progress snapshot file path, if None will use default filename
        snapshot_interval: int, [default=600]
            Interval between saving snapshots (seconds)
        init_model : CatBoost class or string or pathlib.Path, [default=None]
            Continue training starting from the existing model.
            If this parameter is a string or pathlib.Path, load initial model from the path specified by this string.
        callbacks : list, optional (default=None)
            List of callback objects that are applied at end of each iteration.

        log_cout: output stream or callback for logging

        log_cerr: error stream or callback for logging

        Returns
        -------
        model : CatBoost
        """

        params = deepcopy(self._init_params)
        _process_synonyms(params)
        if 'loss_function' in params:
            CatBoostRanker._check_is_compatible_loss(params['loss_function'])

        self._fit(X, y, cat_features, text_features, embedding_features, pairs,
                  sample_weight, group_id, group_weight, subgroup_id, pairs_weight, baseline, use_best_model,
                  eval_set, verbose, logging_level, plot, column_description, verbose_eval, metric_period,
                  silent, early_stopping_rounds, save_snapshot, snapshot_file, snapshot_interval, init_model, callbacks, log_cout, log_cerr)
        return self

    def predict(self, X, ntree_start=0, ntree_end=0, thread_count=-1, verbose=None):
        """
        Predict with data.
        Parameters
        ----------
        X : catboost.Pool or list of features or list of lists or numpy.ndarray or pandas.DataFrame or pandas.Series
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
        return self._predict(X, 'RawFormulaVal', ntree_start, ntree_end, thread_count, verbose, 'predict')

    def staged_predict(self, X, ntree_start=0, ntree_end=0, eval_period=1, thread_count=-1, verbose=None):
        """
        Predict target at each stage for data.
        Parameters
        ----------
        X : catboost.Pool or list of features or list of lists or numpy.ndarray or pandas.DataFrame or pandas.Series
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
        return self._staged_predict(X, 'RawFormulaVal', ntree_start, ntree_end, eval_period, thread_count, verbose, 'staged_predict')

    def score(self, X, y=None, group_id=None, top=None, type=None, denominator=None, group_weight=None, thread_count=-1):
        """
        Calculate NDCG@top
        Parameters
        ----------
        X : catboost.Pool or list or numpy.ndarray or pandas.DataFrame or pandas.Series
            Data to apply model on.
        y : list or numpy.ndarrays or pandas.DataFrame or pandas.Series
            True labels.
        group_id : list or numpy.ndarray or pandas.DataFrame or pandas.Series
            Ranking groups. If X is a Pool, group_id must be defined into X
        top : unsigned integer, up to `pow(2, 32) / 2 - 1`
            NDCG, Number of top-ranked objects to calculate NDCG
        type : str
            NDCG, Metric_type: 'Base' or 'Exp'
        denominator : str
            NDCG, Denominator type: 'LogPosition' or 'Position'
        group_weight : list or numpy.ndarray or pandas.DataFrame or pandas.Series
            Group weights.
        thread_count : int, optional (default=-1)
            Number of threads to work with.
        Returns
        -------
        NDCG@top : float
                   higher is better
        """
        def get_ndcg_metric_name(values, names):
            if not np.any(np.array(values) == None):
                return 'NDCG'
            return 'NDCG:' + ';'.join(['{}={}'.format(n, v) for v, n in zip(values, names) if v is not None])

        if isinstance(X, Pool):
            if y is not None:
                raise CatBoostError("Wrong initializing y: X is catboost.Pool object, y must be initialized inside catboost.Pool.")
            y = X.get_label()
            if group_id is not None:
                raise CatBoostError("Wrong initializing group_id: X is catboost.Pool object, group_id must be initialized inside catboost.Pool.")
            group_id = X.get_group_id_hash()

        if y is None:
            raise CatBoostError("y must be initialized.")
        if group_id is None:
            raise CatBoostError("group_id must be initialized. If groups are not expected, pass an array of zeros")

        predictions = self.predict(X)
        return _eval_metric_util([y], [predictions], get_ndcg_metric_name(), None, group_id, group_weight, None, None, thread_count)[0]


    @staticmethod
    def _check_is_compatible_loss(loss_function):
        is_ranking = CatBoost._is_ranking_objective(loss_function)
        is_regression = CatBoost._is_regression_objective(loss_function)

        if is_regression:
            warnings.warn("Regression loss ('{}') ignores an important ranking parameter 'group_id'".format(loss_function), RuntimeWarning)
        if not (is_ranking or is_regression):
            raise CatBoostError("Invalid loss_function='{}': for ranker use "
                                "YetiRank, YetiRankPairwise, StochasticFilter, StochasticRank, "
                                "QueryCrossEntropy, QueryRMSE, QuerySoftMax, PairLogit, PairLogitPairwise. "
                                "It's also possible to use a regression loss".format(loss_function))


def train(pool=None, params=None, dtrain=None, logging_level=None, verbose=None, iterations=None,
          num_boost_round=None, evals=None, eval_set=None, plot=None, verbose_eval=None, metric_period=None,
          early_stopping_rounds=None, save_snapshot=None, snapshot_file=None, snapshot_interval=None,
          init_model=None, log_cout=sys.stdout, log_cerr=sys.stderr):
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

    snapshot_file : string or pathlib.Path, [default=None]
        Learn progress snapshot file path, if None will use default filename

    snapshot_interval: int, [default=600]
        Interval between saving snapshots (seconds)

    init_model : CatBoost class or string or pathlib.Path, [default=None]
        Continue training starting from the existing model.
        If this parameter is a string or pathlib.Path, load initial model from the path specified by this string.

    log_cout: output stream or callback for logging

    log_cerr: error stream or callback for logging

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
              snapshot_file=snapshot_file, snapshot_interval=snapshot_interval, init_model=init_model,
              log_cout=log_cout, log_cerr=log_cerr)
    return model


def _convert_to_catboost(models):
    """
    Convert _Catboost instances to Catboost ones
    """
    output_models = []
    for model in models:
        cb_model = CatBoost()
        cb_model._object = model
        cb_model._set_trained_model_attributes()
        output_models.append(cb_model)
    return output_models


def cv(pool=None, params=None, dtrain=None, iterations=None, num_boost_round=None,
       fold_count=None, nfold=None, inverted=False, partition_random_seed=0, seed=None,
       shuffle=True, logging_level=None, stratified=None, as_pandas=True, metric_period=None,
       verbose=None, verbose_eval=None, plot=False, early_stopping_rounds=None,
       save_snapshot=None, snapshot_file=None, snapshot_interval=None, metric_update_interval=0.5,
       folds=None, type='Classical', return_models=False, log_cout=sys.stdout, log_cerr=sys.stderr):
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

    type : string, optional (default='Classical')
        Type of cross-validation
        Possible values:
            - 'Classical'
            - 'Inverted'
            - 'TimeSeries'

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

    stratified : bool, optional (default=None)
        Perform stratified sampling. True for classification and False otherwise.

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

    snapshot_file : string or pathlib.Path, [default=None]
        Learn progress snapshot file path, if None will use default filename

    snapshot_interval: int, [default=600]
        Interval between saving snapshots (seconds)

    metric_update_interval: float, [default=0.5]
        Interval between updating metrics (seconds)

    folds: generator or iterator of (train_idx, test_idx) tuples, scikit-learn splitter object or None, optional (default=None)
        If generator or iterator, it should yield the train and test indices for each fold.
        If object, it should be one of the scikit-learn splitter classes
        (https://scikit-learn.org/stable/modules/classes.html#splitter-classes)
        and have ``split`` method.
        if folds is not None, then all of fold_count, shuffle, partition_random_seed, inverted are None

    return_models: bool, optional (default=False)
        if True, return a list of models fitted for each CV fold

    log_cout: output stream or callback for logging

    log_cerr: error stream or callback for logging

    Returns
    -------
    cv results : pandas.core.frame.DataFrame with cross-validation results
        columns are: test-error-mean  test-error-std  train-error-mean  train-error-std
    cv models : list of trained models, if return_models=True
    """
    if params is None:
        raise CatBoostError("params should be set.")

    params = deepcopy(params)
    _process_synonyms(params)
    stringify_builtin_metrics(params)

    metric_period, verbose, logging_level = _process_verbose(
        metric_period, verbose, logging_level, verbose_eval)

    if 'loss_function' not in params:
        raise CatBoostError("Parameter loss_function should be specified for cross-validation")

    if any(v is not None for v in [fold_count,nfold]) and folds is not None:
        raise CatBoostError(
            "if folds is not None, then all of fold_count, shuffle, partition_random_seed, inverted are None"
        )

    if folds is not None or type == 'TimeSeries':
        shuffle = False
        inverted = False

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

    if folds is not None or type == 'TimeSeries':
        stratified = False
    elif stratified is None:
        loss_function = params.get('loss_function', None)
        stratified = isinstance(loss_function, STRING_TYPES) and is_cv_stratified_objective(loss_function)

    if 'cat_features' in params:
        cat_feature_indices_from_params = _get_features_indices(params['cat_features'], pool.get_feature_names())
        if set(pool.get_cat_feature_indices()) != set(cat_feature_indices_from_params):
            raise CatBoostError("categorical features indices in params are different from ones in pool "
                                + str(cat_feature_indices_from_params) +
                                " vs " + str(pool.get_cat_feature_indices()))
        del params['cat_features']

    if 'text_features' in params:
        text_feature_indices_from_params = _get_features_indices(params['text_features'], pool.get_feature_names())
        if set(pool.get_text_feature_indices()) != set(text_feature_indices_from_params):
            raise CatBoostError("text features indices in params are different from ones in pool "
                                + str(text_feature_indices_from_params) +
                                " vs " + str(pool.get_text_feature_indices()))
        del params['text_features']


    if 'embedding_features' in params:
        embedding_feature_indices_from_params = _get_features_indices(params['embedding_features'], pool.get_feature_names())
        if set(pool.get_embedding_feature_indices()) != set(embedding_feature_indices_from_params):
            raise CatBoostError("embedding features indices in params are different from ones in pool "
                                + str(embedding_feature_indices_from_params) +
                                " vs " + str(pool.get_embedding_feature_indices()))
        del params['embedding_features']

    create_if_not_exist = lambda path: os.mkdir(path) if not os.path.exists(path) else None
    train_dir = _get_train_dir(params)
    create_if_not_exist(train_dir)
    plot_dirs = []
    for step in range(fold_count):
        plot_dirs.append(os.path.join(train_dir, 'fold-{}'.format(step)))
    for plot_dir in plot_dirs:
        create_if_not_exist(plot_dir)

    with log_fixup(log_cout, log_cerr), plot_wrapper(plot, plot_dirs):
        if not return_models:
            return _cv(params, pool, fold_count, inverted, partition_random_seed, shuffle, stratified,
                    metric_update_interval, as_pandas, folds, type, return_models)
        else:
            results, cv_models = _cv(params, pool, fold_count, inverted, partition_random_seed, shuffle, stratified,
                                     metric_update_interval, as_pandas, folds, type, return_models)
            output_cv_models = _convert_to_catboost(cv_models)
            return results, output_cv_models


class BatchMetricCalcer(_MetricCalcerBase):

    def __init__(self, catboost, metrics, ntree_start, ntree_end, eval_period, thread_count, tmp_dir):
        super(BatchMetricCalcer, self).__init__(catboost)
        if tmp_dir is None:
            tmp_dir = tempfile.mkdtemp()
            delete_temp_dir_flag = True
        else:
            delete_temp_dir_flag = False

        if isinstance(metrics, STRING_TYPES) or isinstance(metrics, BuiltinMetric):
            metrics = [metrics]
        metrics = stringify_builtin_metrics_list(metrics)
        self._create_calcer(metrics, ntree_start, ntree_end, eval_period, thread_count, tmp_dir, delete_temp_dir_flag)


def sum_models(models, weights=None, ctr_merge_policy='IntersectingCountersAverage'):
    result = CatBoost()
    result._sum_models(models, weights, ctr_merge_policy)
    return result


def _calc_feature_statistics_layout(go, xaxis, single_pool):
    return go.Layout(
        yaxis={
            'title': 'Prediction and target',
            'side': 'left',
            'overlaying': 'y2'
        },
        yaxis2={
            'title': 'Objects per bin' if single_pool else '% pool objects in bin',
            'side': 'right',
            'position': 1.0
        },
        xaxis=xaxis,
        legend={
            'bgcolor': 'rgba(0,0,0,0)',
            'x': 1.07
        }
    )


def _build_binarized_feature_statistics_fig(statistics_list, pool_names):
    try:
        import plotly.graph_objs as go
        import plotly.colors as colors
    except ImportError as e:
        warnings.warn("To draw binarized feature statistics you should install plotly.")
        raise ImportError(str(e))

    pools_count = len(statistics_list)
    data = []
    statistics = statistics_list[0]
    if 'borders' in statistics.keys():
        if len(statistics['borders']) == 0:
            xaxis = go.layout.XAxis(title='Bins', tickvals=[0])
            return go.Figure(data=[], layout=_calc_feature_statistics_layout(go, xaxis, pools_count == 1))

        order = np.arange(len(statistics['objects_per_bin']))
        bar_width = 0.8
        xaxis = go.layout.XAxis(
            title='Bins',
            tickmode='array',
            tickvals=list(range(len(statistics['borders']) + 1)),
            ticktext=['(-inf, {:.4f}]'.format(statistics['borders'][0])] +
                     ['({:.4f}, {:.4f}]'.format(val_1, val_2)
                      for val_1, val_2 in zip(statistics['borders'][:-1], statistics['borders'][1:])] +
                     ['({:.4f}, +inf)'.format(statistics['borders'][-1])],
            showticklabels=False
        )
    elif 'cat_values' in statistics.keys():
        order = np.argsort(statistics['objects_per_bin'])[::-1]
        bar_width = 0.2
        xaxis = go.layout.XAxis(
            title='Cat values',
            tickmode='array',
            tickvals=list(range(len(statistics['cat_values']))),
            ticktext=statistics['cat_values'][order],
            showticklabels=True
        )
    else:
        raise CatBoostError('Expected field "borders" or "cat_values" in binarized feature statistics')

    for i, statistics in enumerate(statistics_list):
        if  pools_count == 1:
            name_suffix = ''
        else:
            name_suffix = ', {} pool'.format(pool_names[i])
        trace_1 = go.Scatter(
            y=statistics['mean_target'][order],
            mode='lines+markers',
            name='Mean target' + name_suffix,
            yaxis='y1',
            xaxis='x',
        )

        trace_2 = go.Scatter(
            y=statistics['mean_prediction'][order],
            mode='lines+markers',
            line={'dash' : 'dash'},
            name='Mean prediction on each segment of feature values' + name_suffix,
            yaxis='y1',
            xaxis='x'
        )
        if (len(statistics['mean_weighted_target']) != 0):
            trace_3 = go.Scatter(
                y=statistics['mean_weighted_target'][order],
                mode='lines+markers',
                line={'dash' : 'dot'},
                name='Mean weighted target' + name_suffix,
                yaxis='y1',
                xaxis='x'
            )

        if pools_count > 1:
            objects_in_pool = statistics['objects_per_bin'].sum()
            color_a = np.array([30, 150, 30])
            color_b = np.array([30, 30, 150])
            color = (color_a * i  + color_b * (pools_count - 1 - i)) / float(pools_count - 1)
            color = color.astype(int)
            trace_4 = go.Bar(
                y=statistics['objects_per_bin'][order] / float(objects_in_pool),
                width=bar_width / pools_count,
                name='% pool objects in bin (total {})'.format(objects_in_pool) + name_suffix,
                yaxis='y2',
                xaxis='x',
                marker={
                    'color': 'rgba({}, {}, {}, 0.4)'.format(*color)
                }
            )
        else:
            trace_4 = go.Bar(
                y=statistics['objects_per_bin'][order],
                width=bar_width,
                name='Objects per bin' + name_suffix,
                yaxis='y2',
                xaxis='x',
                marker={
                    'color': 'rgba(30, 150, 30, 0.4)'
                }
            )

        trace_5 = go.Scatter(
            y=statistics['predictions_on_varying_feature'][order],
            mode='lines+markers',
            line={'dash' : 'dashdot'},
            name='Mean prediction with substituted feature' + name_suffix,
            yaxis='y1',
            xaxis='x'
        )
        if (len(statistics['mean_weighted_target']) != 0):
            data += [trace_1, trace_2, trace_3, trace_4, trace_5]
        else:
            data += [trace_1, trace_2, trace_4, trace_5]

    layout = _calc_feature_statistics_layout(go, xaxis, pools_count == 1)
    fig = go.Figure(data=data, layout=layout)

    return fig


def _plot_feature_statistics_units(statistics, pool_names, feature_name, max_cat_features_on_plot):
    if 'cat_values' in statistics[0].keys() and len(statistics[0]['cat_values']) > max_cat_features_on_plot:
        figs = []
        for begin in range(0, len(statistics[0]['cat_values']), max_cat_features_on_plot):
            end = begin + max_cat_features_on_plot
            statistics_keys = ['cat_values', 'mean_target', 'mean_weighted_target', 'mean_prediction',
                               'objects_per_bin', 'predictions_on_varying_feature']
            sub_statistics = dict([(k, dict([(key, stats[key][begin : end]) for key in statistics_keys])) for k, stats in statistics])
            fig = _build_binarized_feature_statistics_fig(sub_statistics, pool_names)
            feature_name_with_part_suffix = '{}_parts[{}:{}]'.format(feature_name, begin, end)
            figs += [(fig, feature_name_with_part_suffix)]
        return figs
    else:
        fig = _build_binarized_feature_statistics_fig(statistics, pool_names)
        return [(fig, feature_name)]


def _plot_feature_statistics(statistics_by_feature, pool_names, feature_names, max_cat_features_on_plot):
    figs_with_names = []
    for feature_num in statistics_by_feature:
        feature_name = feature_names[feature_num]
        statistics = statistics_by_feature[feature_num]
        need_skip = True
        if 'borders' in statistics[0].keys():
            for stats in statistics:
                if len(stats['borders']) > 0:
                    need_skip = False
        if need_skip:
            continue
        figs_with_names += _plot_feature_statistics_units(statistics, pool_names, feature_name, max_cat_features_on_plot)

    main_fig = figs_with_names[0][0]
    buttons = []
    for fig, feature_name in figs_with_names:
        buttons.append(
            dict(
                label=feature_name,
                method='update',
                args=[{'y': [data.y for data in fig.data]}, {'xaxis': fig.layout.xaxis}],
            ),
        )
    main_fig.update_layout(
        updatemenus=[
            dict(
                direction='down',
                pad={'r': 10, 't': 10},
                showactive=True,
                x=0.25,
                xanchor='left',
                y=1.09,
                yanchor='top',
                buttons=buttons,
            )
        ],
        annotations=[
            dict(text='Statistics for feature', showarrow=False,
                 x=0, xref='paper', y=1.05, yref='paper', align='left')
        ]
    )
    return main_fig


def _to_subclass(model, subclass):
    """
    Convert a CatBoost model to a sklearn-compatible model.

    Parameters
    ----------
    model : CatBoost model
        a model to convert from

    subclass : an sklearn-compatible class
        a class to convert to : CatBoostClassifier, CatBoostRegressor or CatBoostRanker

    Returns
    -------
    a converted model : `subclass` type
        a model converted from the initial CatBoost `model` to a sklearn-compatible `subclass` model
    """
    if isinstance(model, subclass):
        return model
    if not isinstance(model, CatBoost):
        raise CatBoostError('model should be a subclass of CatBoost')

    converted_model = subclass.__new__(subclass)

    # TODO(ilyzhin) change on get_all_params after MLTOOLS-4758
    params = deepcopy(model._init_params)
    _process_synonyms(params)
    if 'loss_function' in params:
        subclass._check_is_compatible_loss(params['loss_function'])

    for attr in model.__dict__:
        setattr(converted_model, attr, getattr(model, attr))
    return converted_model


def to_regressor(model):
    return _to_subclass(model, CatBoostRegressor)


def to_classifier(model):
    return _to_subclass(model, CatBoostClassifier)


def to_ranker(model):
    return _to_subclass(model, CatBoostRanker)


class _TrainCallbacksWrapper(object):
    def __init__(self, callbacks):
        self._callbacks = callbacks

    def after_iteration(self, info):
        for cb in self._callbacks:
            if not cb.after_iteration(info):
                return False
        return True
