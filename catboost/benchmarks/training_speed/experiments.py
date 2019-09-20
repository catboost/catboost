# This file is modified version of benchmark.py.
# benchmark.py was released by RAMitchell (Copyright (c) 2018 Rory Mitchell) under MIT License
# and available at https://github.com/RAMitchell/GBM-Benchmarks/blob/master/benchmark.py
# License text is available at https://github.com/RAMitchell/GBM-Benchmarks/blob/master/LICENSE

import json
import os

from sklearn.model_selection import ParameterGrid

from data_loader import get_dataset
from log_parser import parse_log, Track


def check_exists(hash_id, result_file):
    if not os.path.exists(result_file):
        with open(result_file, 'w') as f:
            json.dump({}, f)
        return False

    with open(result_file, 'r') as f:
        content = json.load(f)
        return str(hash_id) in content


def update_result_file(track, result_file):
    chunk = track.dump_to_json()

    with open(result_file, 'r') as f:
        results = json.load(f)

    results.update(chunk)

    backup_result_file = result_file + '.bkp'
    with open(backup_result_file, 'w') as f:
        json.dump(results, f, indent=4)

    os.rename(backup_result_file, result_file)


class Experiment:
    def __init__(self, name, task, metric):
        self.name = name
        self.task = task
        self.metric = metric

    def run(self, use_gpu, learners, params_grid, dataset_dir, result_file, out_dir):
        dataset = get_dataset(self.name, dataset_dir)

        device_type = 'GPU' if use_gpu else 'CPU'

        for LearnerType in learners:
            learner = LearnerType(dataset, self.task, self.metric, use_gpu)
            algorithm_name = learner.name() + '-' + device_type
            print('Started to train ' + algorithm_name)

            for params in ParameterGrid(params_grid):
                params_str = params_to_str(params)
                log_file = os.path.join(out_dir, self.name, algorithm_name, params_str + '.log')

                print(params_str)

                hash_id = Track.hash(self.name, algorithm_name, self.task, params_str)
                if check_exists(hash_id, result_file):
                    print('Skipped: already evaluated')
                    continue

                try:
                    elapsed = learner.run(params, log_file)
                    print('Timing: ' + str(elapsed) + ' sec')

                    track = parse_log(algorithm_name, self.name, self.task, params_str,
                                      log_file, params['iterations'])
                    update_result_file(track, result_file)
                except Exception as e:
                    print('Exception during training: ' + repr(e))


EXPERIMENT_TYPE = {
    "abalone":
        ["regression", "RMSE"],
    "airline":
        ["binclass", "Accuracy"],
    "airline-one-hot":
        ["binclass", "Accuracy"],
    "bosch":
        ["binclass", "Accuracy"],
    "cover-type":
        ["multiclass", "Accuracy"],
    "epsilon":
        ["binclass", "Accuracy"],
    "epsilon-sampled":
        ["binclass", "Accuracy"],
    "higgs":
        ["binclass", "Accuracy"],
    "higgs-sampled":
        ["binclass", "Accuracy"],
    "letters":
        ["multiclass", "Accuracy"],
    "msrank":
        ["regression", "RMSE"],
    "msrank-classification":
        ["multiclass", "Accuracy"],
    "synthetic-classification":
        ["binclass", "Accuracy"],
    "synthetic":
        ["regression", "RMSE"],
    "synthetic-5k-features":
        ["regression", "RMSE"],
    "year-msd":
        ["regression", "RMSE"],
    "yahoo":
        ["regression", "RMSE"],
    "yahoo-classification":
        ["multiclass", "Accuracy"]
}

EXPERIMENTS = {
    name: Experiment(name, experiment_type[0], experiment_type[1])
    for name, experiment_type in EXPERIMENT_TYPE.items()
}


def params_to_str(params):
    return ''.join(map(lambda item: '{}[{}]'.format(item[0], str(item[1])), params.items()))
