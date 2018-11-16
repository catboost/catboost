import numpy as np
import os
import re

from collections import namedtuple
from experiments import EXPERIMENTS


ALGORITHMS = [method + '-' + device_type
              for device_type in ['CPU', 'GPU']
              for method in ['catboost', 'xgboost', 'lightgbm']]

TIME_REGEX = r'Time: \[\s*(\d+\.?\d*)\s*\]\t'
ELAPSED_REGEX = re.compile(r'Elapsed: (\d+\.?\d*)')
LOG_LINE_REGEX = {
    'lightgbm': re.compile(TIME_REGEX + r'\[(\d+)\]\tvalid_0\'s (\w+): (\d+\.?\d*)'),
    'xgboost': re.compile(TIME_REGEX + r'\[(\d+)\]\t([a-zA-Z\-]+):(\d+\.?\d*)'),
    'catboost': re.compile(TIME_REGEX + r'(\d+)'),
    'catboost-tsv': re.compile(r'(\d+)(\t(\d+\.?\d*))+\n')
}


class Track:
    param_regex = re.compile(r'(\w+)\[(\d+\.?\d*)\]')

    def __init__(self, algorithm_name, dataset, task_type, parameters_str, time_series, scores, duration):
        self.log_name = parameters_str
        self.owner_name = algorithm_name
        self.scores = scores
        self.dataset = dataset
        self.task_type = task_type
        self.duration = duration

        for i in range(1, time_series.shape[0]):
            if time_series[i] - time_series[i - 1] < 0.:
                time_series[i:] = time_series[i:] + 60.

        dur_series = time_series[-1] - time_series[0]
        diff_elapsed_time = np.abs(dur_series - duration)
        if diff_elapsed_time > 100:
            print(parameters_str)
            print('WARNING: difference ' + str(diff_elapsed_time) + ' in calculated duration may indicate broken log.')

        self.time_series = time_series
        assert(np.all(self.time_series - self.time_series[0] >= 0.))

        time_per_iter = time_series[1:] - time_series[:-1]
        # remove outliers
        ids = np.where(time_per_iter < np.quantile(time_per_iter, 0.99))
        self.time_per_iter = time_per_iter[ids]

        params = Track.param_regex.findall(parameters_str)

        param_keys = []
        param_values = []

        for param in sorted(params, key=lambda x: x[0]):
            param_keys.append(param[0])
            param_values.append(float(param[1]))

        self.params_type = namedtuple('Params', param_keys)
        self.params = self.params_type(*param_values)
        self.params_dict = {key: value for key, value in zip(param_keys, param_values)}

    def __str__(self):
        params_str = ''

        for i, field in enumerate(self.params._fields):
            if field == 'iterations':
                continue

            params_str += ', ' + field + ':' + str(self.params[i])

        return self.owner_name + params_str

    def __eq__(self, other):
        return self.owner_name == other.owner_name and self.params == other.params

    def __hash__(self):
        return hash(self.params)

    def get_series(self):
        return self.time_series, self.scores

    def get_time_per_iter(self):
        return self.time_per_iter

    def get_median_time_per_iter(self):
        return np.median(self.time_per_iter)

    def get_fit_iterations(self):
        return self.time_series.shape[0]

    def get_best_score(self):
        return np.min(self.scores)


TASK_TYPES_ACCURACY = ['binclass', 'multiclass']
METRIC_NAME = {
    'lightgbm': {'regression': 'rmse', 'binclass': 'binary_error', 'multiclass': 'multi_error'},
    'xgboost': {'regression': 'eval-rmse', 'binclass': 'eval-error', 'multiclass': 'eval-merror'},
    'catboost': {'regression': 'RMSE', 'binclass': 'Accuracy', 'multiclass': 'Accuracy'}
}


def parse_catboost_log(test_error_file, task_type, iterations):
    values = []
    with open(test_error_file) as metric_log:
        file_content = metric_log.read()

        first_line_idx = file_content.find('\n')
        first_line = file_content[:first_line_idx]
        header = first_line.split('\t')

        column_idx = header.index(METRIC_NAME['catboost'][task_type])
        regex = LOG_LINE_REGEX['catboost-tsv']
        matches = regex.findall(file_content)

        if len(matches) != int(iterations):
            print('WARNING: Broken log file (num matches not equal num iterations): ' + test_error_file)

        for match in matches:
            value = float(match[column_idx])

            if task_type in TASK_TYPES_ACCURACY:
                # Convert to error
                value = 1. - value

            values.append(value)

    return values


def parse_log(algorithm_name, task_type, file_name, iterations):
    time_series = []
    values = []
    algorithm = algorithm_name.rstrip('-CPU|GPU')

    if algorithm == 'catboost':
        catboost_train_dir = os.path.splitext(file_name)[0]
        test_error_file = os.path.join(catboost_train_dir, 'test_error.tsv')
        values = parse_catboost_log(test_error_file, task_type, iterations)

    with open(file_name, 'r') as log:
        file_content = log.read()

        regex = LOG_LINE_REGEX[algorithm]
        matches = regex.findall(file_content)

        if len(matches) != int(iterations):
            print('WARNING: Broken log file ' + file_name)

        for i, match in enumerate(matches):
            time_series.append(float(match[0]))

            if algorithm in ['lightgbm', 'xgboost']:
                metric = match[2]

                # Sanity check on parsed metric
                assert metric == METRIC_NAME[algorithm][task_type]

                values.append(float(match[3]))

        duration = ELAPSED_REGEX.findall(file_content)
        duration = float(duration[0]) if len(duration) > 0 else 0.

    return np.array(time_series), np.array(values), duration


def read_results(dir_name):
    experiment_name = os.path.basename(dir_name.rstrip(os.path.sep))
    task_type = filter(lambda experiment: experiment.name == experiment_name, EXPERIMENTS.itervalues())[0].task
    print(experiment_name + ' ' + task_type)

    results = {}

    for algorithm_name in os.listdir(dir_name):
        print(algorithm_name)
        if algorithm_name not in ALGORITHMS:
            continue

        results[algorithm_name] = []

        cur_dir = os.path.join(dir_name, algorithm_name)
        for log_name in os.listdir(cur_dir):
            path = os.path.join(cur_dir, log_name)

            iterations_str = re.findall(r'iterations\[(\d+)\]', log_name)

            if not os.path.isfile(path) or len(iterations_str) != 1:
                continue

            iterations = int(iterations_str[0])
            try:
                payload = parse_log(algorithm_name, task_type, path, iterations)
            except Exception as e:
                print('Log for ' + path + ' is broken: ' + repr(e))
                continue

            time_series, values, duration = payload
            track = Track(algorithm_name, experiment_name, task_type, log_name, time_series, values, duration)
            results[algorithm_name].append(track)

    return results
