import os
import numpy as np

from sklearn.model_selection import train_test_split, ParameterGrid

import dataset_loader.datasets as data_loader


class Data:
    def __init__(self, X, y, name, task, metric, train_size=0.8):
        assert 0. < train_size < 1.

        test_size = 1. - train_size
        self.name = name
        self.task = task
        self.metric = metric

        if 'MSRank' in name:
            self.X_train = np.vstack([X[0], X[1]])
            self.y_train = np.hstack([y[0], y[1]])

            self.X_test = X[2]
            self.y_test = y[2]
        elif 'CoverType' in name:
            self.X_train = X[0]
            self.y_train = y[0]

            self.X_test = X[1]
            self.y_test = y[1]
        else:
            self.X_train, self.X_test, self.y_train, self.y_test = \
                train_test_split(X, y, test_size=test_size, random_state=0)


class Experiment:
    def __init__(self, data_func, name, task, metric):
        self.data_func = data_func
        self.name = name
        self.task = task
        self.metric = metric

    def run(self, use_gpu, learners, params_grid, out_dir):
        X, y = self.data_func()
        data = Data(X, y, self.name, self.task, self.metric)

        device_type = 'GPU' if use_gpu else 'CPU'

        for LearnerType in learners:
            learner = LearnerType(data, use_gpu)
            algorithm_name = learner.name() + '-' + device_type
            print('Started to train ' + algorithm_name)

            for params in ParameterGrid(params_grid):
                print(params)

                log_dirname = os.path.join(out_dir, self.name, algorithm_name)
                try:
                    elapsed = learner.run(params, log_dirname)
                    print('Timing: ' + str(elapsed) + ' sec')
                except Exception as e:
                    print('Exception during training: ' + repr(e))


DATASETS = {
    "abalone": Experiment(data_loader.get_abalone, "Abalone", "Regression", "RMSE"),
    "letters": Experiment(data_loader.get_letters, "Letters", "Multiclass", "Accuracy"),
    "year-msd": Experiment(data_loader.get_year, "YearPredictionMSD", "Regression", "RMSE"),
    "synthetic": Experiment(data_loader.get_synthetic_regression, "Synthetic", "Regression", "RMSE"),
    "synthetic-5k-features": Experiment(data_loader.get_synthetic_regression_5k_features,
                                        "Synthetic5kFeatures", "Regression", "RMSE"),
    "cover-type": Experiment(data_loader.get_cover_type, "CoverType", "Multiclass", "Accuracy"),
    "epsilon": Experiment(data_loader.get_epsilon, "Epsilon", "Classification", "Accuracy"),
    "higgs": Experiment(data_loader.get_higgs, "Higgs", "Classification", "Accuracy"),
    "bosch": Experiment(data_loader.get_bosch, "Bosch", "Classification", "Accuracy"),
    "airline": Experiment(data_loader.get_airline, "Airline", "Classification", "Accuracy"),
    "higgs-sampled": Experiment(data_loader.get_higgs_sampled, "Higgs", "Classification", "Accuracy"),
    "epsilon-sampled": Experiment(data_loader.get_epsilon_sampled, "Epsilon", "Classification", "Accuracy"),
    "synthetic-classification": Experiment(data_loader.get_synthetic_classification,
                                           "Synthetic2", "Classification", "Accuracy"),
    "msrank": Experiment(data_loader.get_msrank, "MSRank-RMSE", "Regression", "RMSE"),
    "msrank-classification": Experiment(data_loader.get_msrank, "MSRank-MultiClass", "Multiclass", "Accuracy")
}
