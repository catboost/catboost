import os

from sklearn.model_selection import ParameterGrid

from data_loader import get_dataset


class Experiment:
    def __init__(self, name, task, metric):
        self.name = name
        self.task = task
        self.metric = metric

    def run(self, use_gpu, learners, params_grid, dataset_dir, out_dir):
        dataset = get_dataset(self.name, dataset_dir)

        device_type = 'GPU' if use_gpu else 'CPU'

        for LearnerType in learners:
            learner = LearnerType(dataset, self.task, self.metric, use_gpu)
            algorithm_name = learner.name() + '-' + device_type
            print('Started to train ' + algorithm_name)

            for params in ParameterGrid(params_grid):
                print(params)

                log_dir_name = os.path.join(out_dir, self.name, algorithm_name)
                try:
                    elapsed = learner.run(params, log_dir_name)
                    print('Timing: ' + str(elapsed) + ' sec')
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
    for name, experiment_type in EXPERIMENT_TYPE.iteritems()
}
