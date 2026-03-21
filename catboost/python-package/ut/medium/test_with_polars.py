import copy
import decimal
import math
import numbers
import os
import random
import sys
from typing import Dict, List, Union, Tuple, Optional

import pytest

import numpy as np

import polars as pl

from catboost import Pool, _have_equal_features, CatBoost, CatBoostClassifier, CatBoostRanker, CatBoostRegressor
import catboost.utils

try:
    import pyarrow  # noqa: F401
    has_pyarrow = True
except ImportError:
    has_pyarrow = False


try:
    import catboost_pytest_lib as lib
    pytest_plugins = "list_plugin"
except ImportError:
    sys.path.append(os.path.join(os.environ['CMAKE_SOURCE_DIR'], 'catboost', 'pytest'))
    import lib

data_file = lib.data_file


ROTTEN_TOMATOES_TRAIN_FILE = data_file('rotten_tomatoes', 'train')
ROTTEN_TOMATOES_TRAIN_SMALL_NO_QUOTES_FILE = data_file('rotten_tomatoes', 'train_small_no_quotes')
ROTTEN_TOMATOES_TEST_FILE = data_file('rotten_tomatoes', 'test')
ROTTEN_TOMATOES_CD_FILE = data_file('rotten_tomatoes', 'cd')
ROTTEN_TOMATOES_CD_BINCLASS_FILE = data_file('rotten_tomatoes', 'cd_binclass')
ROTTEN_TOMATOES_WITH_EMBEDDINGS_TRAIN_FILE = data_file('rotten_tomatoes_small_with_embeddings', 'train')
ROTTEN_TOMATOES_WITH_EMBEDDINGS_CD_BINCLASS_FILE = data_file(
    'rotten_tomatoes_small_with_embeddings',
    'cd_binclass'
)


common_data_types_specs = [
    pl.Boolean,
    pl.Decimal(8, 4),
    pl.Int8,
    pl.Int16,
    pl.Int32,
    pl.Int64,
    pl.UInt8,
    pl.UInt16,
    pl.UInt32,
    pl.UInt64,
    pl.Float32,
    pl.Float64,
    pl.String,
    pl.Categorical,
    pl.Enum(["cat0"]),
    pl.Enum(["cat0", "cat1", "cat2"]),
    (pl.Object, pl.Float32),  # DataFrame data type is pl.Object, generated data type is float32
    (pl.Object, pl.Int32),    # DataFrame data type is pl.Object, generated data type is int32
    (pl.Object, pl.String),   # DataFrame data type is pl.Object, generated data type is string
]

optional_types = ['Float16', 'Int128', 'UInt128']

for ot in optional_types:
    if hasattr(pl, ot):
        common_data_types_specs.append(getattr(pl, ot))


# testing all possible data types combinations for all features types together is expensive,
# so test exhaustively all data types for each feature type separately
# + test some subset of combinations


def ensure_nulls_after_offset(data: List[List[object]], sample_offset: int):
    # ensure that nulls are still present
    n_samples = len(data)
    for col_idx in range(len(data[0])):
        sample_idx = random.randrange(sample_offset, n_samples)
        data[sample_idx][col_idx] = None

    return data


def ensure_columns_have_all_values(
    data: List[List[object]],
    sample_offset: int,
    get_uniq_values
) -> List[List[object]]:
    n_samples = len(data)
    for col_idx in range(len(data[0])):
        uniq_values = get_uniq_values(col_idx)
        indices = random.sample(range(sample_offset, n_samples), len(uniq_values))
        for cat_idx, sample_idx in enumerate(indices):
            data[sample_idx][col_idx] = uniq_values[cat_idx]
    return data


def get_int_limits(data_type: pl.DataType) -> Tuple[int, int]:
    # data_type.max / data_type.min are not available in older polars versions
    map = {
        pl.Int8: (- 2 ** 7, 2 ** 7 - 1),
        pl.UInt8: (0, 2 ** 8 - 1),
        pl.Int16: (- 2 ** 15, 2 ** 15 - 1),
        pl.UInt16: (0, 2 ** 16 - 1),
        pl.Int32: (- 2 ** 31, 2 ** 31 - 1),
        pl.UInt32: (0, 2 ** 32 - 1),
        pl.Int64: (- 2 ** 63, 2 ** 63 - 1),
        pl.UInt64: (0, 2 ** 64 - 1),
    }
    if hasattr(pl, 'Int128'):
        map[pl.Int128] = (- 2 ** 127, 2 ** 127 - 1)
    if hasattr(pl, 'UInt128'):
        map[pl.UInt128] = (0, 2 ** 128 - 1)
    return map[data_type]


def generate_data(
    data_type: Union[pl.DataType, Tuple[pl.DataType, Tuple[int, int]]],
    has_nulls: bool,
    n_samples: int,
    n_columns: int,
    sample_offset: int = 0,
    seed: int = 42,
    for_cat_features: bool = False,
    non_negative_only: bool = False,
) -> Tuple[List[List[object]], List[str]]:
    """
    sample offset is used to ensure that at all categories and
        at least a single null (if necessary) are present after an offset

    The result will be (row-major data, List of feature_names)

    if data_type is pl.String a number of different strings will be small
        (i.e. useful for categorical, not text features)
    if data_type is Tuple then its elements:
        [0] - must be pl.List
        [1] - must be (min_size, max_size) of Lists
    """

    random.seed(seed)

    def get_int_rng(data_type):
        dt_min, dt_max = get_int_limits(data_type)
        if for_cat_features:
            # TODO: values outside of Int64 range are not supported for cat features
            i64_min, i64_max = get_int_limits(pl.Int64)
            min_value = max(dt_min, i64_min)
            max_value = min(dt_max, i64_max)
        else:
            min_value = dt_min
            max_value = dt_max

        if non_negative_only:
            min_value = max(0, min_value)

        def vg(col_idx: int):
            return random.randrange(min_value, max_value)
        return vg

    def get_float_rng(data_type):
        rng = np.random.default_rng(seed=seed)
        dtype_map = {
            pl.Float32: np.float32,
            pl.Float64: np.float64
        }
        if hasattr(pl, 'Float16'):
            dtype_map[pl.Float16] = np.float16

        np_dtype = dtype_map[data_type]

        def vg(col_idx: int):
            if non_negative_only:
                return np_dtype(rng.random())
            else:
                return np_dtype(rng.random() - 0.5)
        return vg

    if (data_type == pl.Array) or isinstance(data_type, Tuple):
        if isinstance(data_type, Tuple):
            assert data_type[0] == pl.List
            inner_data_type = data_type[0].inner
            min_size = data_type[1][0]
            max_size = data_type[1][1]
        else:
            inner_data_type = data_type.inner
            min_size = data_type.size
            max_size = data_type.size

        if inner_data_type.is_integer():
            inner_rng = get_int_rng(inner_data_type)
        elif inner_data_type.is_float():
            inner_rng = get_float_rng(inner_data_type)
        else:
            raise ValueError(f'unsupported pl.Array inner_type: {inner_data_type}')

        def vg(col_idx: int):
            size = random.randrange(min_size, max_size + 1)
            return [inner_rng(col_idx) for i in range(size)]
        value_generator = vg
    elif data_type == pl.Boolean:
        def vg(col_idx: int):
            return random.choice([False, True])
        value_generator = vg
    elif data_type == pl.Decimal:
        def vg(col_idx: int):
            min_int_part_value = 0 if non_negative_only else -10 ** 3
            return decimal.Decimal(f'{random.randrange(min_int_part_value, 10 ** 3)}.{random.randrange(10 ** 3)}')
        value_generator = vg
    elif data_type.is_integer():
        value_generator = get_int_rng(data_type)
    elif data_type.is_float():
        value_generator = get_float_rng(data_type)
    elif (data_type == pl.String) or (data_type == pl.Categorical):
        categories = [
            # ensure that there are strings with lengths <= 12 and > 12 for some test cases
            # to test both small and long string packing in Arrow StringView data
            [f'categId_f{fi}_{j}' for j in range(min(1, n_samples - 1), min(fi + 3, n_samples))]
            for fi in range(n_columns)
        ]

        def vg(col_idx: int):
            return random.choice(categories[col_idx])
        value_generator = vg
    elif data_type == pl.Enum:
        categories = [data_type.categories for fi in range(n_columns)]

        def vg(col_idx: int):
            return random.choice(categories[col_idx])
        value_generator = vg

    data = []
    for sample_idx in range(n_samples):
        sample_values = []
        for col_idx in range(n_columns):
            if has_nulls and random.randrange(4) == 0:
                sample_values.append(None)
            else:
                sample_values.append(value_generator(col_idx))
        data.append(sample_values)

    if for_cat_features:
        if data_type in (pl.Categorical, pl.Enum):
            ensure_columns_have_all_values(
                data,
                sample_offset,
                lambda col_idx: categories[col_idx],
            )

    if (data_type == pl.Boolean) and (n_samples > 1):
        ensure_columns_have_all_values(
            data,
            sample_offset,
            lambda col_idx: [False, True],
        )

    if has_nulls:
        data = ensure_nulls_after_offset(data, sample_offset)

    return (data, [f'f_{i}' for i in range(n_columns)])


def generate_simple_features_data(n_samples: int) -> Tuple[List[List[object]], List[str]]:
    return generate_data(
        pl.Float32,
        has_nulls=False,
        n_samples=n_samples,
        n_columns=2,
    )


def generate_text_features_data(
    has_nulls: bool,
    n_samples: int,
    n_features: int,
    sample_offset: int,
    seed: int = 42,
) -> Tuple[List[List[object]], List[str]]:
    """
    The result will be (row-major data, List of feature_names)

    if data_type is pl.String a number of different strings will be small
        (i.e. useful for categorical, not text features)
    if data_type is Tuple then its elements:
        [0] - must be pl.List
        [1] - must be (min_size, max_size) of Lists
    """

    text_columns = catboost.utils.read_cd(
        ROTTEN_TOMATOES_CD_FILE,
        data_file=ROTTEN_TOMATOES_TRAIN_FILE,
    )['column_type_to_indices']['Text']

    assert n_features <= len(text_columns)

    text_columns_from_file = [[] for i in range(n_features)]  # [feature_idx][sample_idx]
    with open(ROTTEN_TOMATOES_TRAIN_FILE, 'r') as f:
        for line in f:
            els = line[:-1].split('\t')
            for feature_idx in range(n_features):
                text_columns_from_file[feature_idx].append(els[text_columns[feature_idx]])

    random.seed(seed)

    data = []
    for sample_idx in range(n_samples):
        sample_features = []
        for feature_idx in range(n_features):
            if has_nulls and random.randrange(4) == 0:
                sample_features.append(None)
            else:
                sample_features.append(random.choice(text_columns_from_file[feature_idx]))
        data.append(sample_features)

    if has_nulls:
        data = ensure_nulls_after_offset(data, sample_offset)

    return data, [f'f_{i}' for i in range(n_features)]


def preprocess_data(
    X: List[object],
    Xpl: Union[pl.DataFrame, pl.Series],
    offset: int,
    multiple_chunks: bool
) -> Tuple[List[object], Union[pl.DataFrame, pl.Series]]:
    """
    :return: (X, Xpl)
    """

    if offset:
        X = X[offset:]
        Xpl = Xpl.slice(offset=offset)
        if has_pyarrow:
            if isinstance(Xpl, pl.DataFrame):
                for column in Xpl:
                    assert column.to_arrow().offset == offset
            elif isinstance(Xpl, pl.Series):
                assert Xpl.to_arrow().offset == offset

    n_samples_after_offset = len(X)

    if multiple_chunks:
        cut_pos = n_samples_after_offset // 3
        Xpl1 = Xpl.slice(offset=0, length=cut_pos)
        Xpl2 = Xpl.slice(offset=cut_pos)
        Xpl = pl.concat([Xpl1, Xpl2])
        if isinstance(Xpl, pl.DataFrame):
            for column in Xpl:
                assert column.n_chunks() == 2
        elif isinstance(Xpl, pl.Series):
            assert Xpl.n_chunks() == 2

    return X, Xpl


def generate_group_sizes(n_samples: int, min_group_size: int, max_group_size: int, seed: int = 42) -> List[int]:
    assert min_group_size > 0
    assert max_group_size > 0
    assert max_group_size >= min_group_size
    assert min_group_size <= n_samples

    random.seed(seed)

    min_elements = math.ceil(n_samples / max_group_size)
    max_elements = n_samples // min_group_size
    assert max_elements >= min_elements

    n_groups = random.randint(min_elements, max_elements)

    group_sizes = [min_group_size] * n_groups

    remainder = n_samples - min_group_size * n_groups

    max_extra_elements_per_group = max_group_size - min_group_size

    # distribute the remainder elements randomly between groups
    for i in range(n_groups):
        if remainder == 0:
            break
        max_after_remainder = (n_groups - 1 - i) * max_extra_elements_per_group
        min_add = max(0, remainder - max_after_remainder)
        max_add = min(remainder, max_extra_elements_per_group)
        add = random.randint(min_add, max_add)
        group_sizes[i] += add
        remainder -= add

    return group_sizes


def expand_per_group_data(group_sizes: List[int], per_group_data: List[object]) -> List[object]:
    result = []
    for i, group_size in enumerate(group_sizes):
        result += [per_group_data[i]] * group_size
    return result


def group_sizes_to_group_ids(group_sizes: List[int], unique_ids: Optional[object] = None) -> List[int]:
    return expand_per_group_data(group_sizes, unique_ids if unique_ids is not None else range(len(group_sizes)))


def generate_pairs(group_sizes: List[int], n_pairs: int, seed: int = 42) -> List[List[int]]:
    # elements of the returned List are Lists of 2 elements instead of Tuple because we might want to change it later
    # (insert None)

    assert n_pairs > 0

    random.seed(seed)

    result = []

    group_sample_offset = 0
    for group_size in group_sizes:
        n_group_pairs = random.randint(1, min(group_size - 1, n_pairs - len(result)))
        for i in range(n_group_pairs):
            pair = random.sample(range(group_sample_offset, group_sample_offset + group_size), 2)
            result.append(pair)
        if len(result) == n_pairs:
            break
        group_sample_offset += group_size

    if len(result) < n_pairs:
        raise Exception(f'Unable to generate {n_pairs} from {len(group_sizes)} groups')

    return result


def generate_class_label_data(n_samples: int, n_classes: int, seed: int = 42) -> List[int]:
    assert n_samples >= n_classes

    random.seed(seed)
    result = [random.randrange(0, n_classes) for _ in range(n_samples)]
    # ensure that there is at least one sample from each class
    sample_indices = random.sample(range(n_samples), n_classes)
    for class_idx, sample_idx in enumerate(sample_indices):
        result[sample_idx] = class_idx
    return result


def compare_feature_statistics(stats_from_lists: Dict[str, object], stats_from_pl: Dict[str, object]):
    keys = [
        'borders',
        'binarized_feature',
        'mean_target',
        'mean_prediction',
        'objects_per_bin',
        'predictions_on_varying_feature',
    ]
    for key in keys:
        assert np.allclose(stats_from_lists[key], stats_from_pl[key])


def check_features_with_model_methods(
    X: Union[List[object], Pool],
    Xdf: pl.DataFrame,
    y: Union[List[object], pl.Series, pl.DataFrame],
    feature_names: List[str],
    check_ranker: bool = True,
    **pool_ctor_kwargs
):
    n_samples = X.num_row() if isinstance(X, Pool) else len(X)

    model_from_lists = CatBoostClassifier(iterations=5, thread_count=2)
    model_from_lists.fit(
        X if isinstance(X, Pool) else Pool(X, y, feature_names=feature_names, **pool_ctor_kwargs)
    )

    model_from_df = CatBoostClassifier(iterations=5, thread_count=2)
    model_from_df.fit(Xdf, y, **pool_ctor_kwargs)

    prediction_from_lists = model_from_lists.predict(X)
    prediction_from_df = model_from_df.predict(Xdf)
    assert np.all(prediction_from_lists == prediction_from_df)

    pred_prob_from_lists = model_from_lists.predict_proba(X)
    pred_prob_from_df = model_from_df.predict_proba(Xdf)
    assert np.allclose(pred_prob_from_lists, pred_prob_from_df)

    pred_log_prob_from_lists = model_from_lists.predict_log_proba(X)
    pred_log_prob_from_df = model_from_df.predict_log_proba(Xdf)
    assert np.allclose(pred_log_prob_from_lists, pred_log_prob_from_df)

    score_from_lists = model_from_lists.score(X, None if isinstance(X, Pool) else y)
    score_from_df = model_from_df.score(Xdf, y)
    assert np.allclose(score_from_lists, score_from_df)

    non_num_features_indices = (
        model_from_lists.get_cat_feature_indices()
        + model_from_lists.get_text_feature_indices()
        + model_from_lists.get_embedding_feature_indices()
    )

    if len(non_num_features_indices) < len(feature_names):
        for feature_idx in range(len(feature_names)):
            if feature_idx in non_num_features_indices:
                continue
            stats_from_lists = model_from_lists.calc_feature_statistics(
                X,
                None if isinstance(X, Pool) else y,
                feature=feature_idx,
                plot=False,
            )
            stats_from_pl = model_from_df.calc_feature_statistics(Xdf, y, feature=feature_idx, plot=False)
            compare_feature_statistics(stats_from_lists, stats_from_pl)
            break

    if check_ranker and (n_samples > 5):
        group_sizes = generate_group_sizes(n_samples, min_group_size=2, max_group_size=21)
        group_ids = group_sizes_to_group_ids(group_sizes)

        model_from_lists = CatBoostRanker(iterations=5, thread_count=2)
        model_from_lists.fit(
            X if isinstance(X, Pool) else Pool(X, y, feature_names=feature_names, group_id=group_ids, **pool_ctor_kwargs)
        )

        model_from_df = CatBoostRanker(iterations=5, thread_count=2)
        model_from_df.fit(Xdf, y, group_id=group_ids, **pool_ctor_kwargs)

        score_from_lists = model_from_lists.score(X, None if isinstance(X, Pool) else y, group_id=group_ids)
        score_from_df = model_from_df.score(Xdf, y, group_id=group_ids)
        assert np.allclose(score_from_lists, score_from_df)

    model_from_lists = CatBoostRegressor(iterations=5, thread_count=2)
    model_from_lists.fit(X if isinstance(X, Pool) else Pool(X, y, feature_names=feature_names, **pool_ctor_kwargs))

    model_from_df = CatBoostRegressor(iterations=5, thread_count=2)
    model_from_df.fit(Xdf, y, **pool_ctor_kwargs)

    score_from_lists = model_from_lists.score(X, None if isinstance(X, Pool) else y)
    score_from_df = model_from_df.score(Xdf, y)
    assert np.allclose(score_from_lists, score_from_df)


def check_features_common(
    X: List[object],
    Xdf: pl.DataFrame,
    y: List[object],
    feature_names: List[str],
    should_fail: bool,
    check_model_methods: bool = True,
    check_ranker: bool = True,
    **pool_ctor_kwargs
):
    if should_fail:
        with pytest.raises(Exception):
            dataset_from_df = Pool(Xdf, y, **pool_ctor_kwargs)
    else:
        dataset_from_lists = Pool(X, y, feature_names=feature_names, **pool_ctor_kwargs)
        dataset_from_df = Pool(Xdf, y, **pool_ctor_kwargs)
        assert _have_equal_features(dataset_from_lists, dataset_from_df)

        dataset_from_lists.quantize()
        dataset_from_df.quantize()
        assert _have_equal_features(dataset_from_lists, dataset_from_df)

        if check_model_methods:
            check_features_with_model_methods(X, Xdf, y, feature_names, check_ranker, **pool_ctor_kwargs)


def check_extra_data_with_model(
    X: List[object],
    y: List[object],
    data_name: str,
    data_simple: object,
    data_pl: object,
    estimator_class: CatBoost,
    extra_estimator_params: Dict[str, object] = {},
    extra_data: Dict[str, object] = {},  # for some data we need extra params like group_id
):
    fit_simple_kwargs = copy.deepcopy(extra_data)
    fit_pl_kwargs = copy.deepcopy(extra_data)
    eval_metric_simple_kwargs = copy.deepcopy(extra_data)
    eval_metric_pl_kwargs = copy.deepcopy(extra_data)
    if data_name == 'label':
        y_for_pl = data_pl
    else:
        y_for_pl = y
        data_name_for_fit = 'sample_weight' if data_name == 'weight' else data_name
        fit_simple_kwargs[data_name_for_fit] = data_simple
        fit_pl_kwargs[data_name_for_fit] = data_pl
        eval_metric_simple_kwargs[data_name] = data_simple
        eval_metric_pl_kwargs[data_name] = data_pl

    model_simple = estimator_class(iterations=5, thread_count=2, **extra_estimator_params)
    model_simple.fit(X, y, **fit_simple_kwargs)

    model_pl = estimator_class(iterations=5, thread_count=2, **extra_estimator_params)
    model_pl.fit(X, y_for_pl, **fit_pl_kwargs)

    if data_name == 'graph':
        Xpred = Pool(X, graph=data_simple)
    else:
        Xpred = X

    prediction_simple = model_simple.predict(Xpred)
    prediction_pl = model_pl.predict(Xpred)

    if estimator_class == CatBoostClassifier:
        assert np.all(prediction_pl == prediction_simple)
    else:
        assert np.allclose(prediction_pl, prediction_simple)

    if data_name == 'label':
        if (estimator_class == CatBoostClassifier) and (len(np.shape(y)) > 1):
            pytest.xfail('TODO: fix/implement score for multilabel classification')
        dtypes = y_for_pl.dtypes if isinstance(y_for_pl, pl.DataFrame) else [y_for_pl.dtype]
        for dt in ['Int128', 'UInt128']:
            if hasattr(pl, dt) and (getattr(pl, dt) in dtypes):
                pytest.xfail('TODO: support Int128 and UInt128 for score labels')
        score_simple = model_simple.score(Xpred, y)
        score_pl = model_pl.score(Xpred, y_for_pl)
        assert np.allclose(score_pl, score_simple)
    elif data_name == 'group_id':
        assert estimator_class == CatBoostRanker
        score_simple = model_simple.score(Xpred, y, group_id=data_simple)
        score_pl = model_pl.score(Xpred, y, group_id=data_pl)
        assert np.allclose(score_pl, score_simple)
    elif data_name == 'group_weight':
        assert estimator_class == CatBoostRanker
        score_simple = model_simple.score(Xpred, y, group_id=extra_data['group_id'], group_weight=data_simple)
        score_pl = model_pl.score(Xpred, y, group_id=extra_data['group_id'], group_weight=data_pl)
        assert np.allclose(score_pl, score_simple)

    if data_name in ('graph', 'pairs_weight', 'baseline'):
        return

    if estimator_class == CatBoostClassifier:
        approx = model_simple.predict(Xpred, prediction_type='RawFormulaVal')
        metric = 'AUC'
    else:
        approx = prediction_simple
        metric = 'NDCG' if estimator_class == CatBoostRanker else ('MultiRMSE' if len(np.shape(y)) > 1 else 'RMSE')

    should_fail = False
    if data_name == 'label':
        if isinstance(data_pl, pl.DataFrame):
            columns = [data_pl.to_series(i) for i in range(data_pl.width)]
        else:
            columns = [data_pl]
        for column in columns:
            if any((not isinstance(e, (decimal.Decimal, numbers.Real))) for e in column):
                should_fail = True
                break

    if should_fail:
        with pytest.raises(Exception):
            catboost.utils.eval_metric(
                label=y_for_pl,
                approx=approx,
                metric=metric,
                thread_count=4,
                **eval_metric_pl_kwargs
            )
    else:
        metric_values_from_simple = catboost.utils.eval_metric(
            label=y,
            approx=approx,
            metric=metric,
            thread_count=4,
            **eval_metric_simple_kwargs
        )
        metric_values_from_pl = catboost.utils.eval_metric(
            label=y_for_pl,
            approx=approx,
            metric=metric,
            thread_count=4,
            **eval_metric_pl_kwargs
        )
        assert np.allclose(metric_values_from_simple, metric_values_from_pl)


@pytest.mark.parametrize(
    'data_type_spec',
    common_data_types_specs,
    ids=[f'data_type_spec={dt}' for dt in common_data_types_specs]
)
@pytest.mark.parametrize(
    'has_nulls',
    [False, True],
    ids=[f'has_nulls={e}' for e in [False, True]]
)
@pytest.mark.parametrize(
    'n_samples',
    [5, 92, 1200],
    ids=[f'n_samples={e}' for e in [5, 92, 1200]]
)
@pytest.mark.parametrize(
    'n_features',
    [7, 30],
    ids=[f'n_features={e}' for e in [7, 30]]
)
@pytest.mark.parametrize(
    'offset_in_pl',
    [False, True],
    ids=[f'offset_in_pl={e}' for e in [False, True]]
)
@pytest.mark.parametrize(
    'multiple_chunks',
    [False, True],
    ids=[f'multiple_chunks={e}' for e in [False, True]]
)
def test_features_numeric(data_type_spec, has_nulls, n_samples, n_features, offset_in_pl, multiple_chunks):
    sample_offset = n_samples // 5 if offset_in_pl else 0

    if isinstance(data_type_spec, Tuple):
        assert data_type_spec[0] == pl.Object
        X, feature_names = generate_data(data_type_spec[1], has_nulls, n_samples, n_features, sample_offset)
        Xdf = pl.DataFrame(X, schema=[(n, pl.Object) for n in feature_names], orient='row')
    else:
        X, feature_names = generate_data(data_type_spec, has_nulls, n_samples, n_features, sample_offset)
        Xdf = pl.DataFrame(X, schema=[(n, data_type_spec) for n in feature_names], orient='row')

    X, Xdf = preprocess_data(X, Xdf, sample_offset, multiple_chunks)

    random.seed(0)
    y = [random.choice([0, 1]) for i in range(len(X))]

    should_fail = False
    if isinstance(data_type_spec, Tuple):
        should_fail = (data_type_spec[1] == pl.String)
    else:
        should_fail = data_type_spec in (pl.String, pl.Categorical, pl.Enum)

    check_features_common(X, Xdf, y, feature_names, should_fail)


@pytest.mark.parametrize(
    'data_type_spec',
    common_data_types_specs,
    ids=[f'data_type_spec={dt}' for dt in common_data_types_specs]
)
@pytest.mark.parametrize(
    'has_nulls',
    [False, True],
    ids=[f'has_nulls={e}' for e in [False, True]]
)
@pytest.mark.parametrize(
    'n_samples',
    [5, 92, 1200],
    ids=[f'n_samples={e}' for e in [5, 92, 1200]]
)
@pytest.mark.parametrize(
    'n_features',
    [7, 30],
    ids=[f'n_features={e}' for e in [7, 30]]
)
@pytest.mark.parametrize(
    'offset_in_pl',
    [False, True],
    ids=[f'offset_in_pl={e}' for e in [False, True]]
)
@pytest.mark.parametrize(
    'multiple_chunks',
    [False, True],
    ids=[f'multiple_chunks={e}' for e in [False, True]]
)
def test_features_categorical(data_type_spec, has_nulls, n_samples, n_features, offset_in_pl, multiple_chunks):
    sample_offset = n_samples // 5 if offset_in_pl else 0

    if isinstance(data_type_spec, Tuple):
        assert data_type_spec[0] == pl.Object
        X, feature_names = generate_data(
            data_type_spec[1],
            has_nulls,
            n_samples,
            n_features,
            sample_offset,
            for_cat_features=True,
        )
        Xdf = pl.DataFrame(X, schema=[(n, pl.Object) for n in feature_names], orient='row')
    else:
        X, feature_names = generate_data(
            data_type_spec,
            has_nulls,
            n_samples,
            n_features,
            sample_offset,
            for_cat_features=True,
        )
        Xdf = pl.DataFrame(X, schema=[(n, data_type_spec) for n in feature_names], orient='row')

    X, Xdf = preprocess_data(X, Xdf, sample_offset, multiple_chunks)

    random.seed(0)
    y = [random.choice([0, 1]) for i in range(len(X))]

    should_fail = False
    if has_nulls:
        should_fail = True
    else:
        if isinstance(data_type_spec, Tuple):
            inner_data_type = data_type_spec[1]
        else:
            inner_data_type = data_type_spec
        should_fail = inner_data_type.is_float() or (inner_data_type in (pl.Decimal, pl))

    if should_fail:
        with pytest.raises(Exception):
            dataset_from_df = Pool(Xdf, y, cat_features=feature_names)
    else:
        dataset_from_lists = Pool(X, y, feature_names=feature_names, cat_features=feature_names)
        dataset_from_df = Pool(Xdf, y, cat_features=feature_names)
        assert _have_equal_features(dataset_from_lists, dataset_from_df, ignore_cat_features_hash_to_string=True)

        if (inner_data_type == pl.Enum) and (len(inner_data_type.categories) == 1):
            with pytest.raises(Exception):
                dataset_from_df.quantize()
        else:
            dataset_from_lists.quantize()
            dataset_from_df.quantize()
            assert _have_equal_features(dataset_from_lists, dataset_from_df, ignore_cat_features_hash_to_string=True)

            check_features_with_model_methods(X, Xdf, y, feature_names, cat_features=feature_names)


text_test_data_types_specs = [
    pl.Int32,
    pl.Float32,
    pl.String,
    (pl.Object, pl.Int32),    # DataFrame data type is pl.Object, generated data type is int32
    (pl.Object, pl.String),   # DataFrame data type is pl.Object, generated data type is string
]


@pytest.mark.parametrize(
    'data_type_spec',
    text_test_data_types_specs,
    ids=[f'data_type_spec={dt}' for dt in text_test_data_types_specs]
)
@pytest.mark.parametrize(
    'has_nulls',
    [False, True],
    ids=[f'has_nulls={e}' for e in [False, True]]
)
@pytest.mark.parametrize(
    'n_samples',
    [5, 92, 1200],
    ids=[f'n_samples={e}' for e in [5, 92, 1200]]
)
@pytest.mark.parametrize(
    'n_features',
    [1, 3],
    ids=[f'n_features={e}' for e in [1, 3]]
)
@pytest.mark.parametrize(
    'offset_in_pl',
    [False, True],
    ids=[f'offset_in_pl={e}' for e in [False, True]]
)
@pytest.mark.parametrize(
    'multiple_chunks',
    [False, True],
    ids=[f'multiple_chunks={e}' for e in [False, True]]
)
def test_features_text(data_type_spec, has_nulls, n_samples, n_features, offset_in_pl, multiple_chunks):
    sample_offset = n_samples // 5 if offset_in_pl else 0

    if isinstance(data_type_spec, Tuple):
        assert data_type_spec[0] == pl.Object
        column_data_type = data_type_spec[0]
        inner_data_type = data_type_spec[1]
    else:
        column_data_type = data_type_spec
        inner_data_type = data_type_spec

    if inner_data_type == pl.String:
        X, feature_names = generate_text_features_data(has_nulls, n_samples, n_features, sample_offset)
    else:
        X, feature_names = generate_data(
            inner_data_type,
            has_nulls,
            n_samples,
            n_features,
            sample_offset,
            for_cat_features=True,
        )

    Xdf = pl.DataFrame(X, schema=[(n, column_data_type) for n in feature_names], orient='row')

    X, Xdf = preprocess_data(X, Xdf, sample_offset, multiple_chunks)

    random.seed(0)
    y = [random.choice([0, 1]) for i in range(len(X))]

    if has_nulls:
        should_fail = True
    else:
        should_fail = inner_data_type != pl.String

    check_model_methods = n_samples != 5
    check_features_common(X, Xdf, y, feature_names, should_fail, check_model_methods, text_features=feature_names)


embedding_data_types_specs = [
    pl.Int32,
    pl.Float32,
    pl.String,
    pl.Array(pl.Int32, 5),
    pl.Array(pl.Float32, 10),
    pl.Array(pl.Float64, 256),
    (pl.List(pl.Float32), (64, 64)),  # 2nd tuple represents (min_size, max_size) of possible List lengths
    (pl.List(pl.Float64), (32, 64)),  # 2nd tuple represents (min_size, max_size) of possible List lengths
    (pl.Object, pl.Array(pl.Float32, 10)),
]


@pytest.mark.parametrize(
    'data_type_spec',
    embedding_data_types_specs,
    ids=[f'data_type_spec={dt}' for dt in embedding_data_types_specs]
)
@pytest.mark.parametrize(
    'has_nulls',
    [False, True],
    ids=[f'has_nulls={e}' for e in [False, True]]
)
@pytest.mark.parametrize(
    'n_samples',
    [5, 92, 1200],
    ids=[f'n_samples={e}' for e in [5, 92, 1200]]
)
@pytest.mark.parametrize(
    'n_features',
    [1, 3],
    ids=[f'n_features={e}' for e in [1, 3]]
)
@pytest.mark.parametrize(
    'offset_in_pl',
    [False, True],
    ids=[f'offset_in_pl={e}' for e in [False, True]]
)
@pytest.mark.parametrize(
    'multiple_chunks',
    [False, True],
    ids=[f'multiple_chunks={e}' for e in [False, True]]
)
def test_features_embeddings(data_type_spec, has_nulls, n_samples, n_features, offset_in_pl, multiple_chunks):
    sample_offset = n_samples // 5 if offset_in_pl else 0

    if isinstance(data_type_spec, Tuple):
        column_data_type = data_type_spec[0]
        if (data_type_spec[0] == pl.Object):
            inner_data_type = data_type_spec[1]
        else:
            inner_data_type = data_type_spec[0]
    else:
        column_data_type = data_type_spec
        inner_data_type = data_type_spec

    X, feature_names = generate_data(
        inner_data_type if column_data_type == pl.Object else data_type_spec,
        has_nulls,
        n_samples,
        n_features,
        sample_offset,
    )

    Xdf = pl.DataFrame(X, schema=[(n, column_data_type) for n in feature_names], orient='row')

    X, Xdf = preprocess_data(X, Xdf, sample_offset, multiple_chunks)

    random.seed(0)
    y = [random.choice([0, 1]) for i in range(len(X))]

    if has_nulls:
        should_fail = True
    elif isinstance(data_type_spec, Tuple) and (data_type_spec[0] == pl.List):
        should_fail = data_type_spec[1][0] != data_type_spec[1][1]
    else:
        should_fail = inner_data_type not in (pl.Array, pl.List)

    check_features_common(X, Xdf, y, feature_names, should_fail, embedding_features=feature_names)


def test_features_mixed_types():
    column_description = catboost.utils.read_cd(
        ROTTEN_TOMATOES_WITH_EMBEDDINGS_CD_BINCLASS_FILE,
        data_file=ROTTEN_TOMATOES_WITH_EMBEDDINGS_TRAIN_FILE,
    )

    embeddings_sizes = dict()   # column_idx -> size

    with open(ROTTEN_TOMATOES_WITH_EMBEDDINGS_TRAIN_FILE) as f:
        els = f.readline()[:-1].split('\t')
        for emb_column in column_description['column_type_to_indices']['NumVector']:
            embeddings_sizes[emb_column] = len(els[emb_column].split(';'))

    dataset_from_files = Pool(
        ROTTEN_TOMATOES_WITH_EMBEDDINGS_TRAIN_FILE,
        column_description=ROTTEN_TOMATOES_WITH_EMBEDDINGS_CD_BINCLASS_FILE
    )

    feature_names = [
        name for i, name in enumerate(column_description['column_names'])
        if i not in column_description['non_feature_column_indices']
    ]
    dataset_from_files.set_feature_names(feature_names)

    dataset_pl = pl.read_csv(
        ROTTEN_TOMATOES_WITH_EMBEDDINGS_TRAIN_FILE,
        separator='\t',
        has_header=False,
        missing_utf8_is_empty_string=True,
        new_columns=column_description['column_names'],
        schema_overrides={
            f'column_{i+1}': pl.String for i in (list(embeddings_sizes.keys()) + [14])
        }
    )

    for i, size in embeddings_sizes.items():
        dataset_pl = dataset_pl.with_columns(
            pl.col(column_description['column_names'][i])
            .str.split(";")
            .cast(pl.Array(pl.Float32, shape=size))
        )

    label_pl = dataset_pl.select('top_critic')
    features_pl = dataset_pl.select(pl.selectors.all().exclude("top_critic"))

    dataset_from_pl = Pool(
        features_pl,
        label=label_pl,
        cat_features=column_description['cat_feature_indices'],
        text_features=column_description['text_feature_indices'],
        embedding_features=column_description['embedding_feature_indices'],
    )

    assert _have_equal_features(dataset_from_files, dataset_from_pl, ignore_cat_features_hash_to_string=True)

    check_features_with_model_methods(
        dataset_from_files,
        features_pl,
        label_pl,
        feature_names=dataset_from_files.get_feature_names(),
        check_ranker=False,   # no group_id data
        cat_features=column_description['cat_feature_indices'],
        text_features=column_description['text_feature_indices'],
        embedding_features=column_description['embedding_feature_indices'],
    )

    dataset_from_files.quantize()
    dataset_from_pl.quantize()
    assert _have_equal_features(dataset_from_files, dataset_from_pl, ignore_cat_features_hash_to_string=True)


@pytest.mark.parametrize(
    'data_type_spec',
    common_data_types_specs,
    ids=[f'data_type_spec={dt}' for dt in common_data_types_specs]
)
@pytest.mark.parametrize(
    'has_nulls',
    [False, True],
    ids=[f'has_nulls={e}' for e in [False, True]]
)
@pytest.mark.parametrize(
    'n_samples',
    [5, 92, 1200],
    ids=[f'n_samples={e}' for e in [5, 92, 1200]]
)
@pytest.mark.parametrize(
    'dim_1',
    [1, 3],
    ids=[f'dim_1={d}' for d in [1, 3]]
)
@pytest.mark.parametrize(
    'offset_in_pl',
    [False, True],
    ids=[f'offset_in_pl={e}' for e in [False, True]]
)
@pytest.mark.parametrize(
    'multiple_chunks',
    [False, True],
    ids=[f'multiple_chunks={e}' for e in [False, True]]
)
def test_label(data_type_spec, has_nulls, n_samples, dim_1, offset_in_pl, multiple_chunks):
    if isinstance(data_type_spec, Tuple):
        column_data_type = data_type_spec[0]
        inner_data_type = data_type_spec[1]
    else:
        column_data_type = data_type_spec
        inner_data_type = data_type_spec

    if (dim_1 > 1) and (inner_data_type in (pl.Enum, pl.Categorical, pl.String)):
        pytest.skip(f'inner_data_type={inner_data_type} is not supported for dim_1={dim_1}')

    labels_data = generate_data(
        inner_data_type,
        has_nulls=has_nulls,
        n_samples=n_samples,
        n_columns=dim_1,
    )[0]

    labels_df = pl.DataFrame(labels_data, schema=[(f'label_{i}', column_data_type) for i in range(dim_1)], orient='row')

    labels_data, labels_df = preprocess_data(labels_data, labels_df, offset_in_pl, multiple_chunks)

    if dim_1 == 1:
        labels_data = [row[0] for row in labels_data]
        labels_pl = labels_df['label_0']
    else:
        labels_pl = labels_df

    features_data, feature_names = generate_simple_features_data(len(labels_data))

    features_data_pl = pl.DataFrame(features_data, schema=[(n, pl.Float32) for n in feature_names], orient='row')

    if has_nulls:
        # check objects order
        with pytest.raises(Exception):
            model = CatBoostClassifier(iterations=3, thread_count=2)
            model.fit(Pool(features_data, labels_pl))

        # check features order
        with pytest.raises(Exception):
            model = CatBoostClassifier(iterations=3, thread_count=2)
            model.fit(Pool(features_data_pl, labels_pl))
    else:
        dataset_from_lists = Pool(features_data, labels_data, feature_names=feature_names)

        # check objects order
        dataset_from_pl = Pool(features_data, labels_pl)
        assert np.all(dataset_from_lists.get_label() == dataset_from_pl.get_label())

        # check features order
        dataset_from_pl = Pool(features_data_pl, labels_pl)
        assert np.all(dataset_from_lists.get_label() == dataset_from_pl.get_label())

        if (inner_data_type == pl.Enum) and (len(inner_data_type.categories) == 1):
            return

        check_extra_data_with_model(
            features_data,
            labels_data,
            'label',
            labels_data,
            labels_pl,
            CatBoostRegressor if inner_data_type.is_numeric() else CatBoostClassifier,
            extra_estimator_params={
                'loss_function': (
                    (
                        ('MultiRMSE' if (dim_1 > 1) else 'RMSE') if inner_data_type.is_numeric()
                        else ('MultiLogloss' if (dim_1 > 1) else 'MultiClass')
                    )
                )
            }
        )


pairs_or_graph_data_types_specs = [
    pl.Int32,
    pl.Float32,
    pl.String,
    (pl.Object, pl.Int32),
    (pl.Object, pl.Float32),
    (pl.Object, pl.String),
]


@pytest.mark.parametrize(
    'data_type_spec',
    pairs_or_graph_data_types_specs,
    ids=[f'data_type_spec={dt}' for dt in pairs_or_graph_data_types_specs]
)
@pytest.mark.parametrize(
    'has_nulls',
    [False, True],
    ids=[f'has_nulls={e}' for e in [False, True]]
)
@pytest.mark.parametrize(
    'offset_in_pl',
    [False, True],
    ids=[f'offset_in_pl={e}' for e in [False, True]]
)
@pytest.mark.parametrize(
    'multiple_chunks',
    [False, True],
    ids=[f'multiple_chunks={e}' for e in [False, True]]
)
def test_pairs_and_graph(data_type_spec, has_nulls, offset_in_pl, multiple_chunks):
    n_samples = 120
    n_pairs = 15
    pairs_offset = 4 if offset_in_pl else 0

    group_sizes = generate_group_sizes(n_samples, min_group_size=2, max_group_size=21)
    pairs = generate_pairs(group_sizes, n_pairs=n_pairs)
    group_ids = group_sizes_to_group_ids(group_sizes)

    if isinstance(data_type_spec, Tuple):
        column_data_type = data_type_spec[0]
        inner_data_type = data_type_spec[1]
    else:
        column_data_type = data_type_spec
        inner_data_type = data_type_spec

    py_type = {
        pl.Int32 : int,
        pl.Float32: float,
        pl.String: str,
    }[inner_data_type]

    for i in range(len(pairs)):
        pairs[i] = [py_type(pairs[i][0]), py_type(pairs[i][1])]

    if has_nulls:
        for i in range(3):
            idx = random.randrange(pairs_offset, n_pairs)
            sub_idx = random.randint(0, 1)
            pairs[idx][sub_idx] = None

    pairs_as_df = pl.DataFrame(pairs, schema=[('winner', column_data_type), ('loser', column_data_type)], orient='row')

    pairs, pairs_as_df = preprocess_data(pairs, pairs_as_df, pairs_offset, multiple_chunks)

    features_data = generate_simple_features_data(n_samples)[0]

    random.seed(42)

    y = [random.randrange(0, 5) for _ in range(n_samples)]

    if has_nulls or (inner_data_type != pl.Int32):
        with pytest.raises(Exception):
            Pool(features_data, group_id=group_ids, pairs=pairs_as_df)
        with pytest.raises(Exception):
            Pool(features_data, group_id=group_ids, graph=pairs_as_df)
    else:
        dataset_with_pairs_from_lists = Pool(features_data, group_id=group_ids, pairs=pairs)
        dataset_with_pairs_from_df = Pool(features_data, group_id=group_ids, pairs=pairs_as_df)
        assert dataset_with_pairs_from_lists == dataset_with_pairs_from_df

        dataset_with_pairs_from_df = Pool(features_data, group_id=group_ids)
        dataset_with_pairs_from_df.set_pairs(pairs_as_df)
        assert dataset_with_pairs_from_lists == dataset_with_pairs_from_df

        check_extra_data_with_model(
            features_data,
            y,
            'pairs',
            pairs,
            pairs_as_df,
            CatBoostRanker,
            extra_data={'group_id': group_ids},
        )

        dataset_with_graph_from_lists = Pool(features_data, group_id=group_ids, graph=pairs)
        dataset_with_graph_from_df = Pool(features_data, group_id=group_ids, graph=pairs_as_df)
        assert dataset_with_graph_from_lists == dataset_with_graph_from_df

        check_extra_data_with_model(
            features_data,
            y,
            'graph',
            pairs,
            pairs_as_df,
            CatBoostRanker,
            extra_data={'group_id': group_ids},
        )


weight_or_timestamp_data_types_specs = [
    (pl.Boolean, True),
    (pl.Decimal(8, 4), False),
    (pl.Decimal(8, 4), True),
    (pl.Int32, False),
    (pl.Int32, True),
    (pl.UInt32, True),
    (pl.Float32, False),
    (pl.Float32, True),
    (pl.Float64, False),
    (pl.Float64, True),
    (pl.String, False),
    ((pl.Object, pl.Int32), True),
    ((pl.Object, pl.Float32), True),
    ((pl.Object, pl.String), False),
]


@pytest.mark.parametrize(
    'data_type_spec,non_negative_only',
    weight_or_timestamp_data_types_specs,
    ids=[f'data_type_spec={dt[0]},non_negative_only={dt[1]}' for dt in weight_or_timestamp_data_types_specs]
)
@pytest.mark.parametrize(
    'has_nulls',
    [False, True],
    ids=[f'has_nulls={e}' for e in [False, True]]
)
@pytest.mark.parametrize(
    'offset_in_pl',
    [False, True],
    ids=[f'offset_in_pl={e}' for e in [False, True]]
)
@pytest.mark.parametrize(
    'multiple_chunks',
    [False, True],
    ids=[f'multiple_chunks={e}' for e in [False, True]]
)
def test_weight(data_type_spec, non_negative_only, has_nulls, offset_in_pl, multiple_chunks):
    n_samples = 35
    sample_offset = 4 if offset_in_pl else 0

    if isinstance(data_type_spec, Tuple):
        column_data_type = data_type_spec[0]
        inner_data_type = data_type_spec[1]
    else:
        column_data_type = data_type_spec
        inner_data_type = data_type_spec

    if inner_data_type == pl.Decimal:
        pytest.xfail('TODO: Decimal type is not currently supported for weights')

    weights = generate_data(
        inner_data_type,
        has_nulls=has_nulls,
        n_samples=n_samples,
        n_columns=1,
        sample_offset=sample_offset,
        non_negative_only=non_negative_only,
    )[0]
    weights = [e[0] for e in weights]

    weights_pl = pl.Series(name='weight', values=weights, dtype=column_data_type)

    weights, weights_pl = preprocess_data(weights, weights_pl, sample_offset, multiple_chunks)

    features_data, feature_names = generate_simple_features_data(len(weights))

    features_data_pl = pl.DataFrame(features_data, schema=[(n, pl.Float32) for n in feature_names], orient='row')

    random.seed(0)
    y = [random.choice([0, 1]) for i in range(len(weights))]

    if has_nulls or (inner_data_type == pl.String) or (not non_negative_only):
        # check objects order
        with pytest.raises(Exception):
            model = CatBoostClassifier(iterations=3, thread_count=2)
            model.fit(Pool(features_data, y, weight=weights_pl))

        # check features order
        with pytest.raises(Exception):
            model = CatBoostClassifier(iterations=3, thread_count=2)
            model.fit(Pool(features_data_pl, y, weight=weights_pl))
    else:
        dataset_with_weight_from_list = Pool(features_data, y, feature_names=feature_names, weight=weights)

        # check objects order
        dataset_with_weight_from_pl = Pool(features_data, y, feature_names=feature_names, weight=weights_pl)
        assert dataset_with_weight_from_list == dataset_with_weight_from_pl

        # check features order
        dataset_with_weight_from_pl = Pool(features_data_pl, y, weight=weights_pl)
        assert dataset_with_weight_from_list == dataset_with_weight_from_pl

        # check set_weight
        dataset_with_weight_from_pl = Pool(features_data_pl, y)
        dataset_with_weight_from_pl.set_weight(weights_pl)
        assert dataset_with_weight_from_list == dataset_with_weight_from_pl

        check_extra_data_with_model(
            features_data,
            y,
            'weight',
            weights,
            weights_pl,
            CatBoostClassifier,
        )


group_id_data_types_specs = [
    pl.Boolean,
    pl.Decimal(8, 4),
    pl.Int32,
    pl.UInt32,
    pl.Float32,
    pl.Float64,
    pl.String,
    (pl.Object, pl.Int32),
    (pl.Object, pl.Float32),
    (pl.Object, pl.String),
]


@pytest.mark.parametrize(
    'data_type_spec',
    group_id_data_types_specs,
    ids=[f'data_type_spec={dt}' for dt in group_id_data_types_specs]
)
@pytest.mark.parametrize(
    'has_nulls',
    [False, True],
    ids=[f'has_nulls={e}' for e in [False, True]]
)
@pytest.mark.parametrize(
    'offset_in_pl',
    [False, True],
    ids=[f'offset_in_pl={e}' for e in [False, True]]
)
@pytest.mark.parametrize(
    'multiple_chunks',
    [False, True],
    ids=[f'multiple_chunks={e}' for e in [False, True]]
)
def test_group_id(data_type_spec, has_nulls, offset_in_pl, multiple_chunks):
    n_samples = 120
    groups_offset = 4 if offset_in_pl else 0

    if isinstance(data_type_spec, Tuple):
        column_data_type = data_type_spec[0]
        inner_data_type = data_type_spec[1]
    else:
        column_data_type = data_type_spec
        inner_data_type = data_type_spec

    group_sizes = generate_group_sizes(n_samples, min_group_size=2, max_group_size=21)

    random.seed(0)

    if inner_data_type == pl.String:
        # generate_data won't work because of duplicates
        unique_group_ids = [f'group_{i}' for i in range(len(group_sizes))]
        if has_nulls:
            unique_group_ids[random.randrange(groups_offset, len(group_sizes))] = None
    else:
        unique_group_ids = generate_data(
            inner_data_type,
            has_nulls=has_nulls,
            n_samples=len(group_sizes),
            n_columns=1,
            sample_offset=groups_offset,
        )[0]
        unique_group_ids = [e[0] for e in unique_group_ids]

    group_ids = group_sizes_to_group_ids(group_sizes, unique_group_ids)

    group_ids_pl = pl.Series(name='group_id', values=group_ids, dtype=column_data_type)

    sample_offset = sum(group_sizes[:groups_offset])

    group_ids, group_ids_pl = preprocess_data(group_ids, group_ids_pl, sample_offset, multiple_chunks)

    features_data = generate_simple_features_data(len(group_ids))[0]
    y = [random.choice(range(5)) for i in range(len(group_ids))]

    is_type_supported = inner_data_type.is_integer() or (inner_data_type == pl.String)
    if has_nulls or (not is_type_supported):
        with pytest.raises(Exception):
            model = CatBoostRanker(iterations=3, thread_count=2)
            model.fit(Pool(features_data, y, group_id=group_ids_pl))
    else:
        dataset_with_group_id_from_list = Pool(features_data, y, group_id=group_ids)
        dataset_with_group_id_from_pl = Pool(features_data, y, group_id=group_ids_pl)
        assert dataset_with_group_id_from_list == dataset_with_group_id_from_pl

        # check set_group_id
        dataset_with_group_id_from_pl = Pool(features_data, y)
        dataset_with_group_id_from_pl.set_group_id(group_ids_pl)
        assert dataset_with_group_id_from_list == dataset_with_group_id_from_pl

        check_extra_data_with_model(
            features_data,
            y,
            'group_id',
            group_ids,
            group_ids_pl,
            CatBoostRanker,
        )


@pytest.mark.parametrize(
    'data_type_spec,non_negative_only',
    weight_or_timestamp_data_types_specs,
    ids=[f'data_type_spec={dt[0]},non_negative_only={dt[1]}' for dt in weight_or_timestamp_data_types_specs]
)
@pytest.mark.parametrize(
    'has_nulls',
    [False, True],
    ids=[f'has_nulls={e}' for e in [False, True]]
)
@pytest.mark.parametrize(
    'offset_in_pl',
    [False, True],
    ids=[f'offset_in_pl={e}' for e in [False, True]]
)
@pytest.mark.parametrize(
    'multiple_chunks',
    [False, True],
    ids=[f'multiple_chunks={e}' for e in [False, True]]
)
def test_group_weight(data_type_spec, non_negative_only, has_nulls, offset_in_pl, multiple_chunks):
    n_samples = 120
    groups_offset = 4 if offset_in_pl else 0

    group_sizes = generate_group_sizes(n_samples, min_group_size=2, max_group_size=21)

    if isinstance(data_type_spec, Tuple):
        column_data_type = data_type_spec[0]
        inner_data_type = data_type_spec[1]
    else:
        column_data_type = data_type_spec
        inner_data_type = data_type_spec

    if inner_data_type == pl.Decimal:
        pytest.xfail('TODO: Decimal type is not currently supported for group_weights')
    if not inner_data_type.is_float():
        pytest.xfail('TODO: Non float types are not currently supported for group_weights')

    unique_group_weights = generate_data(
        inner_data_type,
        has_nulls=has_nulls,
        n_samples=len(group_sizes),
        n_columns=1,
        sample_offset=groups_offset,
        non_negative_only=non_negative_only,
    )[0]
    unique_group_weights = [e[0] for e in unique_group_weights]

    group_weights = expand_per_group_data(group_sizes, unique_group_weights)

    group_weights_pl = pl.Series(name='group_weight', values=group_weights, dtype=column_data_type)

    sample_offset = sum(group_sizes[:groups_offset])

    group_weights, group_weights_pl = preprocess_data(group_weights, group_weights_pl, sample_offset, multiple_chunks)

    group_ids = group_sizes_to_group_ids(group_sizes[groups_offset:])

    features_data, feature_names = generate_simple_features_data(len(group_weights))
    features_data_pl = pl.DataFrame(features_data, schema=[(n, pl.Float32) for n in feature_names], orient='row')

    random.seed(0)
    y = [random.choice([0, 1]) for i in range(len(group_weights))]

    if has_nulls or (inner_data_type == pl.String) or (not non_negative_only):
        # check objects order
        with pytest.raises(Exception):
            model = CatBoostClassifier(iterations=3, thread_count=2)
            model.fit(Pool(features_data, y, group_id=group_ids, group_weight=group_weights_pl))

        # check features order
        with pytest.raises(Exception):
            model = CatBoostClassifier(iterations=3, thread_count=2)
            model.fit(Pool(features_data_pl, y, group_id=group_ids, group_weight=group_weights_pl))
    else:
        dataset_with_group_weight_from_list = Pool(
            features_data,
            y,
            feature_names=feature_names,
            group_id=group_ids,
            group_weight=group_weights,
        )

        # check objects order
        dataset_with_group_weight_from_pl = Pool(
            features_data,
            y,
            feature_names=feature_names,
            group_id=group_ids,
            group_weight=group_weights_pl,
        )
        assert dataset_with_group_weight_from_list == dataset_with_group_weight_from_pl

        # check features order
        dataset_with_group_weight_from_pl = Pool(
            features_data_pl,
            y,
            group_id=group_ids,
            group_weight=group_weights_pl,
        )
        assert dataset_with_group_weight_from_list == dataset_with_group_weight_from_pl

        # check set_group_weight
        dataset_with_group_weight_from_pl = Pool(features_data, y, feature_names=feature_names, group_id=group_ids)
        dataset_with_group_weight_from_pl.set_group_weight(group_weights_pl)
        assert dataset_with_group_weight_from_list == dataset_with_group_weight_from_pl

        check_extra_data_with_model(
            features_data,
            y,
            'group_weight',
            group_weights,
            group_weights_pl,
            CatBoostRanker,
            extra_data={'group_id': group_ids},
        )


@pytest.mark.parametrize(
    'data_type_spec',
    group_id_data_types_specs,
    ids=[f'data_type_spec={dt}' for dt in group_id_data_types_specs]
)
@pytest.mark.parametrize(
    'has_nulls',
    [False, True],
    ids=[f'has_nulls={e}' for e in [False, True]]
)
@pytest.mark.parametrize(
    'offset_in_pl',
    [False, True],
    ids=[f'offset_in_pl={e}' for e in [False, True]]
)
@pytest.mark.parametrize(
    'multiple_chunks',
    [False, True],
    ids=[f'multiple_chunks={e}' for e in [False, True]]
)
def test_subgroup_id(data_type_spec, has_nulls, offset_in_pl, multiple_chunks):
    n_samples = 120
    groups_offset = 4 if offset_in_pl else 0

    if isinstance(data_type_spec, Tuple):
        column_data_type = data_type_spec[0]
        inner_data_type = data_type_spec[1]
    else:
        column_data_type = data_type_spec
        inner_data_type = data_type_spec

    group_sizes = generate_group_sizes(n_samples, min_group_size=2, max_group_size=21)

    subgroup_ids = []

    for group_size in group_sizes:
        subgroup_sizes = generate_group_sizes(group_size, min_group_size=1, max_group_size=3)
        unique_subgroup_ids = generate_data(
            inner_data_type,
            has_nulls=has_nulls,
            n_samples=len(subgroup_sizes),
            n_columns=1,
        )[0]
        subgroup_ids.extend(
            group_sizes_to_group_ids(subgroup_sizes, [e[0] for e in unique_subgroup_ids])
        )

    group_ids = group_sizes_to_group_ids(group_sizes[groups_offset:])

    subgroup_ids_pl = pl.Series(name='subgroup_id', values=subgroup_ids, dtype=column_data_type)

    sample_offset = sum(group_sizes[:groups_offset])

    subgroup_ids, subgroup_ids_pl = preprocess_data(subgroup_ids, subgroup_ids_pl, sample_offset, multiple_chunks)

    features_data = generate_simple_features_data(len(subgroup_ids))[0]

    random.seed(0)
    y = [random.choice(range(5)) for i in range(len(subgroup_ids))]

    # TODO: should boolean really be allowed?
    is_type_supported = inner_data_type.is_integer() or (inner_data_type in (pl.Boolean, pl.String))
    if has_nulls or (not is_type_supported):
        with pytest.raises(Exception):
            model = CatBoostRanker(iterations=3, thread_count=2)
            model.fit(Pool(features_data, y, group_id=group_ids, subgroup_id=subgroup_ids_pl))
    else:
        dataset_with_subgroup_id_from_list = Pool(features_data, y, group_id=group_ids, subgroup_id=subgroup_ids)
        dataset_with_subgroup_id_from_pl = Pool(features_data, y, group_id=group_ids, subgroup_id=subgroup_ids_pl)
        assert dataset_with_subgroup_id_from_list == dataset_with_subgroup_id_from_pl

        dataset_with_subgroup_id_from_pl = Pool(features_data, y, group_id=group_ids)
        dataset_with_subgroup_id_from_pl.set_subgroup_id(subgroup_ids_pl)
        assert dataset_with_subgroup_id_from_list == dataset_with_subgroup_id_from_pl

        check_extra_data_with_model(
            features_data,
            y,
            'subgroup_id',
            subgroup_ids,
            subgroup_ids_pl,
            CatBoostRanker,
            extra_data={'group_id': group_ids},
        )


@pytest.mark.parametrize(
    'data_type_spec,non_negative_only',
    weight_or_timestamp_data_types_specs,
    ids=[f'data_type_spec={dt[0]},non_negative_only={dt[1]}' for dt in weight_or_timestamp_data_types_specs]
)
@pytest.mark.parametrize(
    'has_nulls',
    [False, True],
    ids=[f'has_nulls={e}' for e in [False, True]]
)
@pytest.mark.parametrize(
    'offset_in_pl',
    [False, True],
    ids=[f'offset_in_pl={e}' for e in [False, True]]
)
@pytest.mark.parametrize(
    'multiple_chunks',
    [False, True],
    ids=[f'multiple_chunks={e}' for e in [False, True]]
)
def test_pairs_weight(data_type_spec, non_negative_only, has_nulls, offset_in_pl, multiple_chunks):
    n_samples = 120
    n_pairs = 15
    pairs_offset = 4 if offset_in_pl else 0

    group_sizes = generate_group_sizes(n_samples, min_group_size=2, max_group_size=21)
    pairs = generate_pairs(group_sizes, n_pairs=n_pairs)
    group_ids = group_sizes_to_group_ids(group_sizes)

    if isinstance(data_type_spec, Tuple):
        column_data_type = data_type_spec[0]
        inner_data_type = data_type_spec[1]
    else:
        column_data_type = data_type_spec
        inner_data_type = data_type_spec

    if inner_data_type == pl.Decimal:
        pytest.xfail('TODO: Decimal type is not currently supported for pairs_weight')

    pairs_weight = generate_data(
        inner_data_type,
        has_nulls=has_nulls,
        n_samples=len(pairs),
        n_columns=1,
        sample_offset=pairs_offset,
        non_negative_only=non_negative_only,
    )[0]
    pairs_weight = [e[0] for e in pairs_weight]

    pairs = pairs[pairs_offset:]

    pairs_weight_pl = pl.Series(name='pairs_weight', values=pairs_weight, dtype=column_data_type)

    pairs_weight, pairs_weight_pl = preprocess_data(pairs_weight, pairs_weight_pl, pairs_offset, multiple_chunks)

    features_data = generate_simple_features_data(n_samples)[0]

    if has_nulls or (inner_data_type == pl.String) or (not non_negative_only):
        with pytest.raises(Exception):
            Pool(features_data, group_id=group_ids, pairs=pairs, pairs_weight=pairs_weight_pl)
    else:
        dataset_with_pairs_weight_from_list = Pool(
            features_data,
            group_id=group_ids,
            pairs=pairs,
            pairs_weight=pairs_weight,
        )
        dataset_with_pairs_weight_from_pl = Pool(
            features_data,
            group_id=group_ids,
            pairs=pairs,
            pairs_weight=pairs_weight_pl,
        )
        assert dataset_with_pairs_weight_from_list == dataset_with_pairs_weight_from_pl

        dataset_with_pairs_weight_from_pl = Pool(
            features_data,
            group_id=group_ids,
            pairs=pairs,
        )
        dataset_with_pairs_weight_from_pl.set_pairs_weight(pairs_weight_pl)
        assert dataset_with_pairs_weight_from_list == dataset_with_pairs_weight_from_pl

        random.seed(42)

        check_extra_data_with_model(
            features_data,
            [random.randrange(0, 5) for _ in range(n_samples)],
            'pairs_weight',
            pairs_weight,
            pairs_weight_pl,
            CatBoostRanker,
            extra_data={'group_id': group_ids, 'pairs': pairs},
        )


@pytest.mark.parametrize(
    'data_type_spec',
    common_data_types_specs,
    ids=[f'data_type_spec={dt}' for dt in common_data_types_specs]
)
@pytest.mark.parametrize(
    'has_nulls',
    [False, True],
    ids=[f'has_nulls={e}' for e in [False, True]]
)
@pytest.mark.parametrize(
    'n_samples',
    [5, 92, 1200],
    ids=[f'n_samples={e}' for e in [5, 92, 1200]]
)
@pytest.mark.parametrize(
    'dim_1',
    [1, 3],
    ids=[f'dim_1={d}' for d in [1, 3]]
)
@pytest.mark.parametrize(
    'offset_in_pl',
    [False, True],
    ids=[f'offset_in_pl={e}' for e in [False, True]]
)
@pytest.mark.parametrize(
    'multiple_chunks',
    [False, True],
    ids=[f'multiple_chunks={e}' for e in [False, True]]
)
def test_baseline(data_type_spec, has_nulls, n_samples, dim_1, offset_in_pl, multiple_chunks):
    if isinstance(data_type_spec, Tuple):
        column_data_type = data_type_spec[0]
        inner_data_type = data_type_spec[1]
    else:
        column_data_type = data_type_spec
        inner_data_type = data_type_spec

    if inner_data_type == pl.Decimal:
        pytest.xfail('TODO: Decimal type is not currently supported for baseline')

    baseline_data = generate_data(
        inner_data_type,
        has_nulls=has_nulls,
        n_samples=n_samples,
        n_columns=dim_1,
    )[0]

    baseline_df = pl.DataFrame(
        baseline_data,
        schema=[(f'baseline_{i}', column_data_type) for i in range(dim_1)],
        orient='row',
    )

    baseline_data, baseline_df = preprocess_data(baseline_data, baseline_df, offset_in_pl, multiple_chunks)

    if dim_1 == 1:
        baseline_data = [row[0] for row in baseline_data]
        baseline_pl = baseline_df['baseline_0']
    else:
        baseline_pl = baseline_df

    features_data, feature_names = generate_simple_features_data(len(baseline_data))

    features_data_pl = pl.DataFrame(features_data, schema=[(n, pl.Float32) for n in feature_names], orient='row')

    y = generate_class_label_data(len(baseline_data), dim_1 if dim_1 > 1 else 2)

    if has_nulls or (not inner_data_type.is_numeric()):
        # check objects order
        with pytest.raises(Exception):
            model = CatBoostClassifier(iterations=3, thread_count=2)
            model.fit(Pool(features_data, y, baseline=baseline_data))

        # check features order
        with pytest.raises(Exception):
            model = CatBoostClassifier(iterations=3, thread_count=2)
            model.fit(Pool(features_data_pl, y, baseline=baseline_data))
    else:
        dataset_with_baseline_from_lists = Pool(
            features_data,
            y,
            feature_names=feature_names,
            baseline=baseline_data,
        )

        # check objects order
        dataset_with_baseline_from_pl = Pool(
            features_data,
            y,
            feature_names=feature_names,
            baseline=baseline_pl,
        )
        assert dataset_with_baseline_from_lists == dataset_with_baseline_from_pl

        # check features order
        dataset_with_baseline_from_pl = Pool(
            features_data_pl,
            y,
            baseline=baseline_pl,
        )
        assert dataset_with_baseline_from_lists == dataset_with_baseline_from_pl

        # check set_baseline
        dataset_with_baseline_from_pl = Pool(features_data, y, feature_names=feature_names)
        dataset_with_baseline_from_pl.set_baseline(baseline_pl)
        assert dataset_with_baseline_from_lists == dataset_with_baseline_from_pl

        check_extra_data_with_model(
            features_data,
            y,
            'baseline',
            baseline_data,
            baseline_pl,
            CatBoostClassifier,
        )


@pytest.mark.parametrize(
    'data_type_spec,non_negative_only',
    weight_or_timestamp_data_types_specs,
    ids=[f'data_type_spec={dt[0]},non_negative_only={dt[1]}' for dt in weight_or_timestamp_data_types_specs]
)
@pytest.mark.parametrize(
    'has_nulls',
    [False, True],
    ids=[f'has_nulls={e}' for e in [False, True]]
)
@pytest.mark.parametrize(
    'offset_in_pl',
    [False, True],
    ids=[f'offset_in_pl={e}' for e in [False, True]]
)
@pytest.mark.parametrize(
    'multiple_chunks',
    [False, True],
    ids=[f'multiple_chunks={e}' for e in [False, True]]
)
def test_timestamp(data_type_spec, non_negative_only, has_nulls, offset_in_pl, multiple_chunks):
    n_samples = 35
    sample_offset = 4 if offset_in_pl else 0

    if isinstance(data_type_spec, Tuple):
        column_data_type = data_type_spec[0]
        inner_data_type = data_type_spec[1]
    else:
        column_data_type = data_type_spec
        inner_data_type = data_type_spec

    timestamp_data = generate_data(
        inner_data_type,
        has_nulls=has_nulls,
        n_samples=n_samples,
        n_columns=1,
        sample_offset=sample_offset,
        non_negative_only=non_negative_only,
    )[0]
    timestamp_data = [e[0] for e in timestamp_data]

    timestamp_pl = pl.Series(name='timestamp', values=timestamp_data, dtype=column_data_type)

    timestamp_data, timestamp_pl = preprocess_data(timestamp_data, timestamp_pl, sample_offset, multiple_chunks)

    features_data = generate_simple_features_data(len(timestamp_data))[0]

    random.seed(0)
    y = [random.choice([0, 1]) for i in range(len(timestamp_data))]

    if has_nulls or (not inner_data_type.is_integer()) or (not non_negative_only):
        with pytest.raises(Exception):
            model = CatBoostClassifier(iterations=3, thread_count=2)
            model.fit(Pool(features_data, y, timestamp=timestamp_pl))
    else:
        dataset_with_timestamp_from_list = Pool(features_data, y, timestamp=timestamp_data)
        dataset_with_timestamp_from_pl = Pool(features_data, y, timestamp=timestamp_pl)
        assert dataset_with_timestamp_from_list == dataset_with_timestamp_from_pl

        dataset_with_timestamp_from_pl = Pool(features_data, y)
        dataset_with_timestamp_from_pl.set_timestamp(timestamp_pl)
        assert dataset_with_timestamp_from_list == dataset_with_timestamp_from_pl
