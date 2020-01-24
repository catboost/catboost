import csv
import json
import itertools
import os
import random
import shutil
import sys
from pandas import read_csv
from copy import deepcopy
import numpy as np
from catboost.utils import read_cd
__all__ = [
    'DelayedTee',
    'binary_path',
    'compare_evals',
    'compare_evals_with_precision',
    'compare_fit_evals_with_precision',
    'compare_metrics_with_diff',
    'generate_random_labeled_dataset',
    'generate_concatenated_random_labeled_dataset',
    'load_dataset_as_dataframe',
    'load_pool_features_as_df',
    'permute_dataset_columns',
    'remove_time_from_json',
    'test_output_path',
]

try:
    import yatest
    binary_path = yatest.common.binary_path
    test_output_path = yatest.common.test_output_path

except ImportError:
    def binary_path(*path):
        return os.path.join(os.environ["BINARY_PATH"], *path)

    def test_output_path(*path):
        return os.path.join(os.getcwd(), *path)


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

    labels = [random.choice(labels) for i in range(n_samples)]

    return (features, labels)


BY_CLASS_METRICS = ['AUC', 'Precision', 'Recall', 'F1']


def compare_metrics_with_diff(custom_metric, fit_eval, calc_eval, eps=1e-7):
    csv_fit = csv.reader(open(fit_eval, "r"), dialect='excel-tab')
    csv_calc = csv.reader(open(calc_eval, "r"), dialect='excel-tab')

    head_fit = next(csv_fit)
    head_calc = next(csv_calc)

    if isinstance(custom_metric, basestring):
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


def compare_evals(fit_eval, calc_eval):
    csv_fit = csv.reader(open(fit_eval, "r"), dialect='excel-tab')
    csv_calc = csv.reader(open(calc_eval, "r"), dialect='excel-tab')
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
    array_fit = np.loadtxt(fit_eval, delimiter='\t', skiprows=1, ndmin=2)
    array_calc = np.loadtxt(calc_eval, delimiter='\t', skiprows=1, ndmin=2)
    header_fit = open(fit_eval, "r").readline().split()
    header_calc = open(calc_eval, "r").readline().split()
    if skip_last_column_in_fit:
        array_fit = np.delete(array_fit, np.s_[-1], 1)
        header_fit = header_fit[:-1]
    if header_fit != header_calc:
        return False
    is_close = np.isclose(array_fit, array_calc, rtol=rtol, atol=atol)
    if np.all(is_close):
        return True
    for i, _ in itertools.islice(filter(lambda x: not np.all(x[1]), enumerate(is_close)), 100):
        sys.stderr.write("index: {} {} != {}\n".format(i, array_fit[i], array_calc[i]))
    return False


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
        names=columns_metadata['column_names'],
        dtype=columns_metadata['column_dtypes'],
        skiprows=1 if has_header else 0
    )

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
