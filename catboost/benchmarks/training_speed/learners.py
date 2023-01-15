# This file is modified version of benchmark.py.
# benchmark.py was released by RAMitchell (Copyright (c) 2018 Rory Mitchell) under MIT License
# and available at https://github.com/RAMitchell/GBM-Benchmarks/blob/master/benchmark.py
# License text is available at https://github.com/RAMitchell/GBM-Benchmarks/blob/master/LICENSE

import os
import sys
import time
from copy import deepcopy
from datetime import datetime

import catboost as cat
import lightgbm as lgb
import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_squared_error, accuracy_score

RANDOM_SEED = 0


class TimeAnnotatedFile:
    def __init__(self, file_descriptor):
        self.file_descriptor = file_descriptor

    def write(self, message):
        if message == '\n':
            self.file_descriptor.write('\n')
            return

        cur_time = datetime.now()
        new_message = "Time: [%d.%06d]\t%s" % (cur_time.second, cur_time.microsecond, message)
        self.file_descriptor.write(new_message)

    def flush(self):
        self.file_descriptor.flush()

    def close(self):
        self.file_descriptor.close()


class Logger:
    def __init__(self, filename):
        self.filename = filename
        self.stdout = sys.stdout

    def __enter__(self):
        self.file = TimeAnnotatedFile(open(self.filename, 'w'))
        sys.stdout = self.file

    def __exit__(self, exception_type, exception_value, traceback):
        if exception_type is not None:
            print(str(exception_value) + '\n' + str(traceback))

        sys.stdout = self.stdout
        self.file.close()


def eval_metric(data, prediction):
    if data.metric == "RMSE":
        return np.sqrt(mean_squared_error(data.y_test, prediction))
    elif data.metric == "Accuracy":
        if data.task == "binclass":
            prediction = prediction > 0.5
        elif data.task == "multiclass":
            if prediction.ndim > 1:
                prediction = np.argmax(prediction, axis=1)
        return accuracy_score(data.y_test, prediction)
    else:
        raise ValueError("Unknown metric: " + data.metric)


class Learner:
    def __init__(self):
        self.default_params = {}

    def _fit(self, tunable_params):
        params = deepcopy(self.default_params)
        params.update(tunable_params)

        print('Parameters:\n{}' + str(params))
        return params

    def eval(self, data, num_iterations, step=10):
        scores = []

        for n_tree in range(num_iterations, step=step):
            prediction = self.predict(n_tree)
            score = eval_metric(data, prediction)
            scores.append(score)

        return scores

    def predict(self, n_tree):
        raise Exception('Not implemented')

    def set_train_dir(self, params, path):
        pass

    def run(self, params, log_filename):
        log_dir_name = os.path.dirname(log_filename)

        if not os.path.exists(log_dir_name):
            os.makedirs(log_dir_name)

        self.set_train_dir(params, log_filename + 'dir')

        with Logger(log_filename):
            start = time.time()
            self._fit(params)
            elapsed = time.time() - start
            print('Elapsed: ' + str(elapsed))

        return elapsed


class XGBoostLearner(Learner):
    def __init__(self, data, task, metric, use_gpu):
        Learner.__init__(self)
        params = {
            'n_gpus': 1,
            'silent': 0,
            'seed': RANDOM_SEED
        }

        if use_gpu:
            params['tree_method'] = 'gpu_hist'
        else:
            params['tree_method'] = 'hist'

        if task == "regression":
            params["objective"] = "reg:linear"
            if use_gpu:
                params["objective"] = "gpu:" + params["objective"]
        elif task == "multiclass":
            params["objective"] = "multi:softmax"
            params["num_class"] = int(np.max(data.y_test)) + 1
        elif task == "binclass":
            params["objective"] = "binary:logistic"
            if use_gpu:
                params["objective"] = "gpu:" + params["objective"]
        else:
            raise ValueError("Unknown task: " + task)

        if metric == 'Accuracy':
            if task == 'binclass':
                params['eval_metric'] = 'error'
            elif task == 'multiclass':
                params['eval_metric'] = 'merror'

        self.train = xgb.DMatrix(data.X_train, data.y_train)
        self.test = xgb.DMatrix(data.X_test, data.y_test)

        self.default_params = params

    @staticmethod
    def name():
        return 'xgboost'

    def _fit(self, tunable_params):
        params = Learner._fit(self, tunable_params)
        self.learner = xgb.train(params, self.train, tunable_params['iterations'], evals=[(self.test, 'eval')])

    def predict(self, n_tree):
        return self.learner.predict(self.test, ntree_limit=n_tree)


class LightGBMLearner(Learner):
    def __init__(self, data, task, metric, use_gpu):
        Learner.__init__(self)

        params = {
            'task': 'train',
            'boosting_type': 'gbdt',
            'verbose': 0,
            'random_state': RANDOM_SEED,
            'bagging_freq': 1
        }

        if use_gpu:
            params["device"] = "gpu"

        if task == "regression":
            params["objective"] = "regression"
        elif task == "multiclass":
            params["objective"] = "multiclass"
            params["num_class"] = int(np.max(data.y_test)) + 1
        elif task == "binclass":
            params["objective"] = "binary"
        else:
            raise ValueError("Unknown task: " + task)

        if metric == 'Accuracy':
            if task == 'binclass':
                params['metric'] = 'binary_error'
            elif task == 'multiclass':
                params['metric'] = 'multi_error'
        elif metric == 'RMSE':
            params['metric'] = 'rmse'

        self.train = lgb.Dataset(data.X_train, data.y_train)
        self.test = lgb.Dataset(data.X_test, data.y_test, reference=self.train)

        self.default_params = params

    @staticmethod
    def name():
        return 'lightgbm'

    def _fit(self, tunable_params):
        params_copy = deepcopy(tunable_params)

        if 'max_depth' in params_copy:
            params_copy['num_leaves'] = 2 ** params_copy['max_depth']
            del params_copy['max_depth']

        num_iterations = params_copy['iterations']
        del params_copy['iterations']

        params = Learner._fit(self, params_copy)
        self.learner = lgb.train(
            params,
            self.train,
            num_boost_round=num_iterations,
            valid_sets=self.test
        )

    def predict(self, n_tree):
        return self.learner.predict(self.test, num_iteration=n_tree)


class CatBoostLearner(Learner):
    def __init__(self, data, task, metric, use_gpu):
        Learner.__init__(self)

        params = {
            'devices': [0],
            'logging_level': 'Info',
            'use_best_model': False,
            'bootstrap_type': 'Bernoulli',
            'random_seed': RANDOM_SEED
        }

        if use_gpu:
            params['task_type'] = 'GPU'

        if task == 'regression':
            params['loss_function'] = 'RMSE'
        elif task == 'binclass':
            params['loss_function'] = 'Logloss'
        elif task == 'multiclass':
            params['loss_function'] = 'MultiClass'

        if metric == 'Accuracy':
            params['custom_metric'] = 'Accuracy'

        self.train = cat.Pool(data.X_train, data.y_train)
        self.test = cat.Pool(data.X_test, data.y_test)

        self.default_params = params

    @staticmethod
    def name():
        return 'catboost'

    def _fit(self, tunable_params):
        params = Learner._fit(self, tunable_params)
        self.model = cat.CatBoost(params)
        self.model.fit(self.train, eval_set=self.test, verbose_eval=True)

    def set_train_dir(self, params, path):
        if not os.path.exists(path):
            os.makedirs(path)
        params["train_dir"] = path

    def predict(self, n_tree):
        if self.default_params['loss_function'] == "MultiClass":
            prediction = self.model.predict_proba(self.test, ntree_end=n_tree)
        else:
            prediction = self.model.predict(self.test, ntree_end=n_tree)

        return prediction
