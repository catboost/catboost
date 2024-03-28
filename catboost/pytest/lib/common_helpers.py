import collections
import csv
import json
import os
import random
import shutil
import sys
from pandas import read_csv, DataFrame
from copy import deepcopy
import numpy as np
from catboost.utils import read_cd
__all__ = [
    'DelayedTee',
    'append_params_to_cmdline',
    'binary_path',
    'compare_evals',
    'compare_evals_with_precision',
    'compare_fit_evals_with_precision',
    'compare_metrics_with_diff',
    'format_crossvalidation',
    'get_limited_precision_dsv_diff_tool',
    'get_limited_precision_json_diff_tool',
    'get_limited_precision_numpy_diff_tool',
    'generate_random_labeled_dataset',
    'generate_concatenated_random_labeled_dataset',
    'generate_dataset_with_num_and_cat_features',
    'load_dataset_as_dataframe',
    'load_pool_features_as_df',
    'permute_dataset_columns',
    'remove_time_from_json',
    'test_output_path',
    'compare_with_limited_precision',
    'is_canonical_test_run',
]

try:
    from yatest.common import binary_path, test_output_path
except ImportError:
    sys.path += [
        os.environ['CMAKE_SOURCE_DIR'],
        os.path.join(os.environ['CMAKE_SOURCE_DIR'], 'library', 'python', 'testing', 'yatest_common')
    ]
    from yatest.common import binary_path, test_output_path


def remove_time_from_json(filename):
    with open(filename) as f:
        log = json.load(f)
    iterations = log['iterations']
    for i, iter_info in enumerate(iterations):
        for key in ['remaining_time', 'passed_time']:
            if key in iter_info.keys():
                del iter_info[key]
    with open(filename, 'w') as f:
        json.dump(log, f, sort_keys=True)
    return filename


# rewinds dst_stream to the start of the captured output so you can read it
class DelayedTee(object):

    def __init__(self, src_stream, dst_stream):
        self.src_stream = src_stream
        self.dst_stream = dst_stream

    def __enter__(self):
        self.src_stream.flush()
        self._old_src_stream = os.dup(self.src_stream.fileno())
        self._old_dst_stream_pos = self.dst_stream.tell()
        os.dup2(self.dst_stream.fileno(), self.src_stream.fileno())

    def __exit__(self, exc_type, exc_value, traceback):
        self.src_stream.flush()
        os.dup2(self._old_src_stream, self.src_stream.fileno())
        self.dst_stream.seek(self._old_dst_stream_pos)
        shutil.copyfileobj(self.dst_stream, self.src_stream)
        self.dst_stream.seek(self._old_dst_stream_pos)


def permute_dataset_columns(test_pool_path, cd_path, seed=123):
    permuted_test_path = test_output_path('permuted_test')
    permuted_cd_path = test_output_path('permuted_cd')
    generator = random.Random(seed)
    column_count = len(open(test_pool_path).readline().split('\t'))
    permutation = list(range(column_count))
    generator.shuffle(permutation)
    with open(cd_path) as original_cd, open(permuted_cd_path, 'w') as permuted_cd:
        for line in original_cd:
            line = line.strip()
            if not line:
                continue
            index, rest = line.split('\t', 1)
            permuted_cd.write('{}\t{}\n'.format(permutation.index(int(index)), rest))
    with open(test_pool_path) as test_pool, open(permuted_test_path, 'w') as permuted_test:
        for line in test_pool:
            splitted = line.strip().split('\t')
            permuted_test.write('\t'.join([splitted[i] for i in permutation]) + '\n')

    return permuted_test_path, permuted_cd_path


def generate_concatenated_random_labeled_dataset(nrows, nvals, labels, seed=20181219, prng=None):
    if prng is None:
        prng = np.random.RandomState(seed=seed)
    label = prng.choice(labels, [nrows, 1])
    feature = prng.random_sample([nrows, nvals])
    return np.concatenate([label, feature], axis=1)


def generate_patients_datasets(train_path, test_path):
    samples = 237

    for samples, path in zip([237, 154], [train_path, test_path]):
        data = DataFrame()
        data['age'] = np.random.randint(20, 71, size=samples)
        data['gender'] = np.where(np.random.binomial(1, 0.7, samples) == 1, 'male', 'female')
        data['diet'] = np.where(np.random.binomial(1, 0.1, samples) == 1, 'yes', 'no')
        data['glucose'] = np.random.uniform(4, 12, size=samples)
        data['platelets'] = np.random.randint(100, 500, size=samples)
        data['cholesterol'] = np.random.uniform(4.5, 6.5, size=samples)
        data['survival_in_days'] = np.random.randint(30, 500, size=samples)
        data['outcome'] = np.where(np.random.binomial(1, 0.8, size=samples) == 1, 'dead', 'alive')
        data['target'] = np.where(data['outcome'] == 'dead', data['survival_in_days'], - data['survival_in_days'])
        data = data.drop(['outcome', 'survival_in_days'], axis=1)
        data.to_csv(
            path,
            header=False,
            index=False,
            sep='\t'
        )


# returns (features : numpy.ndarray, labels : list) tuple
def generate_random_labeled_dataset(
    n_samples,
    n_features,
    labels,
    features_density=1.0,
    features_dtype=np.float32,
    features_range=(-1., 1.),
    features_order='C',
    seed=20191008
):
    assert features_density > 0.0

    random.seed(seed)

    features = np.empty((n_samples, n_features), dtype=features_dtype, order=features_order)
    for feature_idx in range(n_features):
        for sample_idx in range(n_samples):
            v1 = random.random()
            if v1 > features_density:
                value = 0
            else:
                value = features_range[0] + (features_range[1] - features_range[0]) * (v1 / features_density)
            features[sample_idx, feature_idx] = features_dtype(value)

    labels = [random.choice(tuple(labels)) for _ in range(n_samples)]

    return (features, labels)


# returns (features : pandas.DataFrame, labels : list) tuple
def generate_dataset_with_num_and_cat_features(
    n_samples,
    n_num_features,
    n_cat_features,
    labels,
    num_features_density=1.0,
    num_features_dtype=np.float32,
    num_features_range=(-1., 1.),
    cat_features_uniq_value_count=5,
    cat_features_dtype=np.int32,
    seed=20201015
):
    assert num_features_density > 0.0
    assert cat_features_uniq_value_count > 0

    random.seed(seed)

    # put num and categ features to the result DataFrame in random order but keep sequential names within each type
    feature_columns = collections.OrderedDict()

    num_feature_idx = 0
    cat_feature_idx = 0
    while (num_feature_idx < n_num_features) or (cat_feature_idx < n_cat_features):
        if (cat_feature_idx < n_cat_features) and random.randrange(2):
            values = []
            for sample_idx in range(n_samples):
                values.append(cat_features_dtype(random.randrange(cat_features_uniq_value_count)))
            feature_columns['c' + str(cat_feature_idx)] = values
            cat_feature_idx += 1
        elif num_feature_idx < n_num_features:
            values = []
            for sample_idx in range(n_samples):
                v1 = random.random()
                if v1 > num_features_density:
                    value = 0
                else:
                    value = (
                        num_features_range[0]
                        + (num_features_range[1] - num_features_range[0]) * (v1 / num_features_density)
                    )
                values.append(num_features_dtype(value))
            feature_columns['n' + str(num_feature_idx)] = values
            num_feature_idx += 1

    labels = [random.choice(labels) for i in range(n_samples)]

    return (DataFrame(feature_columns), labels)


def generate_survival_dataset(seed=20201015):
    np.random.seed(seed)

    X = np.random.rand(200, 20)*10

    mean_y = np.sin(X[:, 0])

    y = np.random.randn(200, 10) * 0.3 + mean_y[:, None]

    y_lower = np.min(y, axis=1)
    y_upper = np.max(y, axis=1)
    y_upper = np.where(y_upper >= 1.4, -1, y_upper+abs(np.min(y_lower)))
    y_lower += abs(np.min(y_lower))

    right_censored_ids = np.where(y_upper == -1)[0]
    interval_censored_ids = np.where(y_upper != -1)[0]

    train_ids = np.hstack(
        [right_censored_ids[::2], interval_censored_ids[:140]])
    test_ids = np.hstack(
        [right_censored_ids[1::2], interval_censored_ids[140:]])

    X_train, y_lower_train, y_upper_train = X[train_ids], y_lower[train_ids], y_upper[train_ids]
    X_test, y_lower_test, y_upper_test = X[test_ids], y_lower[test_ids], y_upper[test_ids]

    return [(X_train, y_lower_train, y_upper_train), (X_test, y_lower_test, y_upper_test)]


BY_CLASS_METRICS = ['AUC', 'Precision', 'Recall', 'F1']


def compare_metrics_with_diff(custom_metric, fit_eval, calc_eval, eps=1e-7):
    with open(fit_eval, "r") as fit_eval_file, open(calc_eval, "r") as calc_eval_file:
        csv_fit = csv.reader(fit_eval_file, dialect='excel-tab')
        csv_calc = csv.reader(calc_eval_file, dialect='excel-tab')

        head_fit = next(csv_fit)
        head_calc = next(csv_calc)

        if isinstance(custom_metric, str):
            custom_metric = [custom_metric]

        for metric_name in deepcopy(custom_metric):
            if metric_name in BY_CLASS_METRICS:
                custom_metric.remove(metric_name)

                for fit_metric_name in head_fit:
                    if fit_metric_name[:len(metric_name)] == metric_name:
                        custom_metric.append(fit_metric_name)

        col_idx_fit = {}
        col_idx_calc = {}

        for metric_name in custom_metric:
            col_idx_fit[metric_name] = head_fit.index(metric_name)
            col_idx_calc[metric_name] = head_calc.index(metric_name)

        while True:
            try:
                line_fit = next(csv_fit)
                line_calc = next(csv_calc)
                for metric_name in custom_metric:
                    fit_value = float(line_fit[col_idx_fit[metric_name]])
                    calc_value = float(line_calc[col_idx_calc[metric_name]])
                    max_abs = max(abs(fit_value), abs(calc_value))
                    err = abs(fit_value - calc_value) / max_abs if max_abs > 0 else 0
                    if err > eps:
                        raise Exception('{}, iter {}: fit vs calc = {} vs {}, err = {} > eps = {}'.format(
                            metric_name, line_fit[0], fit_value, calc_value, err, eps))
            except StopIteration:
                break


def compare_evals(fit_eval, calc_eval, skip_header=False):
    with open(fit_eval, "r") as fit_eval_file, open(calc_eval, "r") as calc_eval_file:
        csv_fit = csv.reader(fit_eval_file, dialect='excel-tab')
        csv_calc = csv.reader(calc_eval_file, dialect='excel-tab')
        if skip_header:
            next(csv_fit)
            next(csv_calc)
        while True:
            try:
                line_fit = next(csv_fit)
                line_calc = next(csv_calc)
                if line_fit[:-1] != line_calc:
                    return False
            except StopIteration:
                break
        return True


def compare_evals_with_precision(fit_eval, calc_eval, rtol=1e-6, atol=1e-8, skip_last_column_in_fit=True):
    df_fit = read_csv(fit_eval, sep='\t')
    if skip_last_column_in_fit:
        df_fit = df_fit.iloc[:, :-1]

    df_calc = read_csv(calc_eval, sep='\t')

    if np.any(df_fit.columns != df_calc.columns):
        sys.stderr.write('column sets differ: {}, {}'.format(df_fit.columns, df_calc.columns))
        return False

    def print_diff(column, row_idx):
        sys.stderr.write(
            "column: {}, index: {} {} != {}\n".format(
                column,
                row_idx,
                df_fit[column][row_idx],
                df_calc[column][row_idx]
            )
        )

    for column in df_fit.columns:
        if column in ['SampleId', 'Label']:
            if (df_fit[column] != df_calc[column]).any():
                print_diff(column, np.where(df_fit[column] != df_calc[column])[0])
                return False
        else:
            is_close = np.isclose(df_fit[column].to_numpy(), df_calc[column].to_numpy(), rtol=rtol, atol=atol)
            if np.any(is_close == 0):
                print_diff(column, np.where(is_close == 0)[0])
                return False
    return True


def compare_fit_evals_with_precision(fit_eval_1, fit_eval_2, rtol=1e-6, atol=1e-8):
    return compare_evals_with_precision(
        fit_eval_1,
        fit_eval_2,
        rtol=rtol,
        atol=atol,
        skip_last_column_in_fit=False
    )


def load_dataset_as_dataframe(data_file, columns_metadata, has_header=False):
    """
        returns dict with 'features', 'target' keys
    """

    if 'Label' not in columns_metadata['column_type_to_indices']:
        raise Exception('no target in dataset')

    df = read_csv(
        data_file,
        sep='\t',
        header=1 if has_header else None
    )

    df.columns = columns_metadata['column_names']
    df = df.astype(columns_metadata['column_dtypes'])

    result = {}
    result['target'] = df.iloc[:, columns_metadata['column_type_to_indices']['Label'][0]].values
    df.drop(columns=df.columns[columns_metadata['non_feature_column_indices']], inplace=True)
    result['features'] = df

    return result


# returns (features DataFrame, cat_feature_indices)
def load_pool_features_as_df(pool_file, cd_file):
    columns_metadata = read_cd(cd_file, data_file=pool_file, canonize_column_types=True)
    data = load_dataset_as_dataframe(pool_file, columns_metadata)
    return (data['features'], columns_metadata['cat_feature_indices'])


def append_params_to_cmdline(cmd, params):
    if isinstance(params, dict):
        for param in params.items():
            key = "{}".format(param[0])
            value = "{}".format(param[1])
            cmd.append(key)
            cmd.append(value)
    else:
        for param in params:
            cmd.append(param)


def format_crossvalidation(is_inverted, n, k):
    cv_type = 'Inverted' if is_inverted else 'Classical'
    return '{}:{};{}'.format(cv_type, n, k)


def is_canonical_test_run():
    return os.environ.get('IS_CANONICAL_TEST_RUN', '1').lower() in ('yes', 'true', '1')


def get_limited_precision_dsv_diff_tool(diff_limit, have_header=False):
    diff_tool = [
        binary_path("catboost/tools/limited_precision_dsv_diff/limited_precision_dsv_diff"),
    ]
    if diff_limit is not None:
        diff_tool += ['--diff-limit', str(diff_limit)]
    if have_header:
        diff_tool += ['--have-header']
    return diff_tool


def get_limited_precision_json_diff_tool(diff_limit):
    diff_tool = [
        binary_path("catboost/tools/limited_precision_json_diff/limited_precision_json_diff"),
    ]
    if diff_limit is not None:
        diff_tool += ['--diff-limit', str(diff_limit)]
    return diff_tool


def get_limited_precision_numpy_diff_tool(rtol=None, atol=None):
    diff_tool = [binary_path("catboost/tools/limited_precision_numpy_diff/limited_precision_numpy_diff")]
    if diff_tool[0] is None:
        diff_tool = [
            'python',
            os.path.join(os.environ['CMAKE_SOURCE_DIR'], 'catboost', 'tools', 'limited_precision_numpy_diff', 'main.py')
        ]

    if rtol is not None:
        diff_tool += ['--rtol', str(rtol)]
    if atol is not None:
        diff_tool += ['--atol', str(atol)]
    return diff_tool


# arguments can be JSON-like simple data structures
def compare_with_limited_precision(lhs, rhs, rtol=1e-6, atol=1e-8):
    if isinstance(lhs, dict):
        if not isinstance(rhs, dict):
            return False
        if len(lhs) != len(rhs):
            return False
        for k in lhs.keys():
            if k not in rhs:
                return False
            if not compare_with_limited_precision(lhs[k], rhs[k], rtol, atol):
                return False
        return True
    elif isinstance(lhs, list):
        if not isinstance(rhs, list):
            return False
        if len(lhs) != len(rhs):
            return False
        return all((compare_with_limited_precision(lhs[i], rhs[i], rtol, atol) for i in range(len(lhs))))
    elif isinstance(lhs, np.ndarray):
        if not isinstance(rhs, np.ndarray):
            return False
        return np.allclose(lhs, rhs, rtol=rtol, atol=atol, equal_nan=True)
    elif isinstance(lhs, (float, np.floating)):
        if not isinstance(rhs, (float, np.floating)):
            return False
        return abs(lhs - rhs) <= atol + rtol * abs(rhs)
    else:
        return lhs == rhs
