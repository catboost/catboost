import catboost as cat
import lightgbm as lgb
import numpy as np
import os
import sys
import time
import xgboost as xgb

from copy import deepcopy
from datetime import datetime
from sklearn.metrics import mean_squared_error, accuracy_score


# Global parameter
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


def _params_to_str(params):
    return ''.join(map(lambda (key, value): '{}[{}]'.format(key, str(value)), params.items()))


def check_log(log_file):
    print('Checking log')
    if os.path.exists(log_file):
        with open(log_file, 'r') as f:
            lines = f.readlines()
            return len(lines) > 0 and 'Elapsed: ' in lines[-1]
    return False


def eval_metric(data, prediction):
    if data.metric == "RMSE":
        return np.sqrt(mean_squared_error(data.y_test, prediction))
    elif data.metric == "Accuracy":
        if data.task == "Classification":
            prediction = prediction > 0.5
        elif data.task == "Multiclass":
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

    def run(self, params, log_dir_name):
        if not os.path.exists(log_dir_name):
            os.makedirs(log_dir_name)

        path = os.path.join(log_dir_name, _params_to_str(params))
        filename = os.path.join(path + '.log')

        if check_log(filename):
            print('Skipping experiment, reason: log already exists and is consistent')
            return 0

        self.set_train_dir(params, path)

        with Logger(filename):
            start = time.time()
            self._fit(params)
            elapsed = time.time() - start
            print('Elapsed: ' + str(elapsed))

        return elapsed


class XGBoostLearner(Learner):
    def __init__(self, data, use_gpu):
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

        if data.task == "Regression":
            params["objective"] = "reg:linear"
            if use_gpu:
                params["objective"] = "gpu:" + params["objective"]
        elif data.task == "Multiclass":
            params["objective"] = "multi:softmax"
            params["num_class"] = np.max(data.y_test) + 1
        elif data.task == "Classification":
            params["objective"] = "binary:logistic"
            if use_gpu:
                params["objective"] = "gpu:" + params["objective"]
        else:
            raise ValueError("Unknown task: " + data.task)

        if data.metric == 'Accuracy':
            params['eval_metric'] = 'error' if data.task == 'Classification' else 'merror'

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
    def __init__(self, data, use_gpu):
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

        if data.task == "Regression":
            params["objective"] = "regression"
        elif data.task == "Multiclass":
            params["objective"] = "multiclass"
            params["num_class"] = np.max(data.y_test) + 1
        elif data.task == "Classification":
            params["objective"] = "binary"
        else:
            raise ValueError("Unknown task: " + data.task)

        if data.task == 'Classification':
            params['metric'] = 'binary_error'
        elif data.task == 'Multiclass':
            params['metric'] = 'multi_error'
        elif data.task == 'Regression':
            params['metric'] = 'rmse'

        self.train = lgb.Dataset(data.X_train, data.y_train)
        self.test = lgb.Dataset(data.X_test, data.y_test, reference=self.train)

        self.default_params = params

    @staticmethod
    def name():
        return 'lightgbm'

    def _fit(self, tunable_params):
        if 'max_depth' in tunable_params:
            tunable_params['num_leaves'] = 2 ** tunable_params['max_depth']
            del tunable_params['max_depth']

        num_iterations = tunable_params['iterations']
        del tunable_params['iterations']

        params = Learner._fit(self, tunable_params)
        self.learner = lgb.train(
            params,
            self.train,
            num_boost_round=num_iterations,
            valid_sets=self.test
        )

    def predict(self, n_tree):
        return self.learner.predict(self.test, num_iteration=n_tree)


class CatBoostLearner(Learner):
    def __init__(self, data, use_gpu):
        Learner.__init__(self)

        params = {
            'devices': [0],
            'logging_level': 'Info',
            'use_best_model': False,
            'bootstrap_type': 'Bernoulli'
        }

        if use_gpu:
            params['task_type'] = 'GPU'

        if data.task == 'Regression':
            params['loss_function'] = 'RMSE'
        elif data.task == 'Classification':
            params['loss_function'] = 'Logloss'
        elif data.task == 'Multiclass':
            params['loss_function'] = 'MultiClass'

        if data.metric == 'Accuracy':
            params['custom_metric'] = 'Accuracy'

        self.train = cat.Pool(data.X_train, data.y_train)
        self.test = cat.Pool(data.X_test, data.y_test)

        self.default_params = params

    @staticmethod
    def name():
        return 'catboost'

    def _fit(self, tunable_params):
        params = Learner._fit(self, tunable_params)
        if 'nthread' in params:
            params['thread_count'] = params['nthread']
            del params['nthread']

        self.model = cat.CatBoost(params)
        self.model.fit(self.train, eval_set=self.test, verbose_eval=True)

    def set_train_dir(self, params, path):
        params["train_dir"] = path

    def predict(self, n_tree):
        if self.default_params['loss_function'] == "Multiclass":
            prediction = self.model.predict_proba(self.test, ntree_end=n_tree)
        else:
            prediction = self.model.predict(self.test, ntree_end=n_tree)

        return prediction
