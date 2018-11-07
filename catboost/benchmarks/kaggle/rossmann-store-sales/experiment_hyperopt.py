#!/usr/bin/env python2

import argparse
import os
import pickle
import sys
import time

from hyperopt import hp, fmin, tpe, Trials, STATUS_OK, STATUS_FAIL
import numpy as np
import pandas as pd
from sklearn.model_selection._split import TimeSeriesSplit

import catboost as cb
import lightgbm as lgb
import xgboost as xgb

import config


class Experiment(object):

    def __init__(self, learning_task='classification', bst_name=None, gpu_id=0, max_n_estimators=1500,
                 hyperopt_evals=50, dataset_path='./', output_folder_path='./'):
        self.learning_task, self.bst_name = learning_task, bst_name
        self.gpu_id = gpu_id
        self.max_n_estimators = max_n_estimators
        self.best_loss = np.inf
        self.hyperopt_evals, self.hyperopt_eval_num = hyperopt_evals, 0
        self.dataset_path, self.output_folder_path = dataset_path, output_folder_path
        self.default_params, self.best_params = None, None
        if self.learning_task == 'classification':
            self.metric = 'logloss'
        elif self.learning_task == 'regression':
            self.metric = 'rmse'
        else:
            raise ValueError('Task type must be "classification" or "regression"')

        if not os.path.exists(self.output_folder_path):
            os.mkdir(self.output_folder_path)


    def read_file(self, file_name, target_col):
        X = pd.read_csv(file_name, sep='\t', header=None)
        if self.learning_task == 'classification':
            y = np.maximum(X[target_col].values, 0)
        else:
            y = X[target_col].values
        X.drop(target_col, axis=1, inplace=True)
        return X, y


    def read_data(self, dataset_path=None):
        dataset_path = dataset_path or self.dataset_path
        cols = pd.read_csv(os.path.join(dataset_path, 'cd'), sep='\t', header=None)
        target_col = cols[cols[1] == "Label"][0].values[0]
        cat_cols = cols[cols[1] == "Categ"][0].values

        X_train, y_train = self.read_file(os.path.join(dataset_path, 'train'), target_col)
        X_test, y_test = self.read_file(os.path.join(dataset_path, 'test'), target_col)
        data = pd.concat([X_train, X_test],ignore_index=True)
        data[cat_cols] = data[cat_cols].apply(lambda x: x.astype('category').cat.codes)
        data = np.array(data).astype('float')
        X_train, X_test = data[:X_train.shape[0]], data[X_train.shape[0]:]

        cat_cols[cat_cols > target_col] = cat_cols[cat_cols > target_col] - 1
        return X_train, y_train, X_test, y_test, cat_cols


    def convert_to_dataset(self, data, label, cat_cols=None):
        raise NotImplementedError('Method convert_to_dataset is not implemented.')


    def split_and_preprocess(self, X_train, y_train, X_test, y_test, cat_cols, n_splits=5, random_state=0):
        cv = TimeSeriesSplit(n_splits=n_splits)
        cv_pairs = []

        print 'np.shape(X_train)', np.shape(X_train), 'np.shape(y_train)', y_train

        for train_index, test_index in cv.split(X_train, y_train):
            print 'train fold', train_index[0], train_index[-1]
            print 'test fold', test_index[0], test_index[-1]

            train, test = X_train[train_index], X_train[test_index]
            dtrain = self.convert_to_dataset(train.astype(float), y_train[train_index], cat_cols)
            dtest = self.convert_to_dataset(test.astype(float), y_train[test_index], cat_cols)
            cv_pairs.append((dtrain, dtest))

        dtrain = self.convert_to_dataset(X_train.astype(float), y_train, cat_cols)
        dtest = self.convert_to_dataset(X_test.astype(float), y_test, cat_cols)

        return cv_pairs, (dtrain, dtest)


    # specify either n_estimators or early_stopping (bool)
    # returns (score, iteration, array of metrics)
    def fit(self, params, dtrain, dtest, max_n_estimators, n_estimators=None, early_stopping=False, seed=0):
        raise NotImplementedError('Method fit is not implemented.')


    def predict(self, bst, dtest, X_test):
        raise NotImplementedError('Method predict is not implemented.')


    def preprocess_params(self, params):
        raise NotImplementedError('Method preprocess_params is not implemented.')


    def run_cv(self, cv_pairs, params=None, verbose=False):
        params = params or self.default_params
        params = self.preprocess_params(params)

        # (best_score, best_iteration pairs)
        evals_results = []
        start_time = time.time()
        for dtrain, dtest in cv_pairs:
            best_score, best_iteration, _ = self.fit(
                params,
                dtrain,
                dtest,
                max_n_estimators=self.max_n_estimators,
                n_estimators=None,
                early_stopping=True
            )
            evals_results.append((best_score, best_iteration))
        eval_time = time.time() - start_time

        print 'evals_results', evals_results

        loss = np.mean(evals_results, axis=0)[0]

        cv_result = {'loss': loss,
                     'best_n_estimators': evals_results[-1][1],
                     'eval_time': eval_time,
                     'status': STATUS_FAIL if np.isnan(loss) else STATUS_OK,
                     'params': params.copy()}
        self.best_loss = min(self.best_loss, cv_result['loss'])
        self.hyperopt_eval_num += 1
        cv_result.update({'hyperopt_eval_num': self.hyperopt_eval_num, 'best_loss': self.best_loss})

        if verbose:
            print '[{0}/{1}]\teval_time={2:.2f} sec\tcurrent_{3}={4:.6f}\tmin_{3}={5:.6f}\tparams={6}'.format(
                        self.hyperopt_eval_num, self.hyperopt_evals, eval_time,
                        self.metric, cv_result['loss'], self.best_loss, params)
        return cv_result


    def calc_best_n_estimators(self, cv_pairs):
        if self.best_n_estimators > (self.max_n_estimators - 100):
            self.best_loss, self.best_n_estimators, _ = self.fit(
                self.best_params,
                cv_pairs[-1][0],
                cv_pairs[-1][1],
                n_estimators=None,
                max_n_estimators=20000,
                early_stopping=True
            )
            print 'calc_best_n_estimators:'
            print 'best loss %s = %s' % (self.metric, self.best_loss)
            print 'best_n_estimators = %s\n' % self.best_n_estimators


    def train_on_all_train_eval_on_test(self, dtrain, dtest, params=None, n_estimators=None):
        params = params or self.best_params or self.default_params
        n_estimators = n_estimators or self.best_n_estimators

        print 'n_estimators', n_estimators

        start_time = time.time()
        test_score, _, _ = self.fit(params, dtrain, dtest, n_estimators)
        eval_time = time.time() - start_time
        print 'fit on all train data took %s sec' % (time.time() - start_time)
        print 'loss on test: %s' % test_score


    def optimize_params(self, cv_pairs, max_evals=None, verbose=True):
        max_evals = max_evals or self.hyperopt_evals

        self.hyperopt_eval_num, self.best_loss = 0, np.inf

        rstate=np.random.RandomState(1)

        db_filename = os.path.join(self.output_folder_path, 'trials')

        if os.path.exists(db_filename):
            self.trials = pickle.load(open(db_filename, "rb"))
        else:
            self.trials = Trials()

        start_time = time.time()

        for iter in xrange(len(self.trials.trials), max_evals):
            _ = fmin(
                fn=lambda params: self.run_cv(cv_pairs, params, verbose=verbose),
                space=self.space,
                algo=tpe.suggest,
                max_evals=iter+1,
                trials=self.trials,
                rstate=rstate
            )
            pickle.dump(self.trials, open(db_filename, "wb"))

        print 'optimize_params took %s sec' % (time.time() - start_time)


        self.best_params = self.trials.best_trial['result']['params']
        self.best_n_estimators = self.trials.best_trial['result']['best_n_estimators']
        return self.trials.best_trial['result']


    def print_result(self, result, name=''):
        print '%s:\n' % name
        print '%s = %s' % (self.metric, result['loss'])
        if 'best_n_estimators' in result.keys():
            print 'best_n_estimators = %s' % result['best_n_estimators']
        elif 'n_estimators' in result.keys():
            print 'n_estimators = %s' % result['n_estimators']
        print 'params = %s' % result['params']



    def run(self):
        print 'Loading and preprocessing dataset...'
        X_train, y_train, X_test, y_test, cat_cols = self.read_data()
        cv_pairs, (dtrain, dtest) = self.split_and_preprocess(X_train, y_train, X_test, y_test, cat_cols)

        print 'Optimizing params...'
        cv_result = self.optimize_params(cv_pairs)
        self.print_result(cv_result, '\nBest result on cv')

        print 'Calc best N estimators...'
        self.calc_best_n_estimators(cv_pairs)

        print 'Train on all train dataset and eval on test dataset...'
        self.train_on_all_train_eval_on_test(dtrain, dtest)



class XGBExperiment(Experiment):

    def __init__(self, learning_task, gpu_id=0, max_n_estimators=1500,
                 max_hyperopt_evals=50, dataset_path='./', output_folder_path='./'):
        Experiment.__init__(self, learning_task, 'xgb', gpu_id, max_n_estimators, max_hyperopt_evals,
                            dataset_path, output_folder_path)
        self.space = {
            'learning_rate': hp.loguniform('learning_rate', -3, 0),
            'min_split_loss': hp.choice('min_split_loss', [0, hp.loguniform('min_split_loss_positive', -16, 0)]),
            'max_depth' : hp.quniform('max_depth', 5, 16, 1),
            # min_child_weight ?
            'subsample': hp.uniform('subsample', 0.5, 1),
            'colsample_bytree': hp.uniform('colsample_bytree', 0.5, 1),
            # does not work on hist
            #'colsample_bylevel': hp.uniform('colsample_bylevel', 0.5, 1),
            'reg_lambda': hp.choice('reg_lambda', [0, hp.loguniform('reg_lambda_positive', -8, 3)]),
            'reg_alpha': hp.choice('reg_alpha', [0, hp.loguniform('reg_alpha_positive', -8, 3)]),
            'grow_policy': hp.choice('grow_policy',['depthwise', 'lossguide']),
            # max_leaves?
            'max_bin': hp.quniform('max_bin', 256, 1024, 1),
        }

        self.default_params = {
            'learning_rate': 0.3,
            'min_split_loss': 0,
            'max_depth': 6,
            'subsample': 1.0,
            'colsample_bytree': 1.0,
             #'colsample_bylevel': 1.0,
            'min_child_weight': 1,
            'reg_alpha': 0,
            'reg_lambda': 1,
            'max_bin': 256
        }
        self.default_params = self.preprocess_params(self.default_params)


    def preprocess_params(self, params):
        if self.learning_task == "classification":
            params.update({'objective': 'binary:logistic', 'eval_metric': 'logloss', 'silent': 1})
        elif self.learning_task == "regression":
            params.update({'objective': 'reg:linear', 'eval_metric': 'rmse', 'silent': 1})
        if self.gpu_id:
            params['tree_method'] = 'gpu_hist'
            params['gpu_id'] = self.gpu_id
        else:
            params['tree_method'] = 'hist'
        params['max_depth'] = int(params['max_depth'])
        params['max_bin'] = int(params['max_bin'])

        return params


    def convert_to_dataset(self, data, label, cat_cols=None):
        return xgb.DMatrix(data, label)


    def fit(self, params, dtrain, dtest, max_n_estimators, n_estimators=None, early_stopping=False, seed=0):
        evals_result = {}
        #print 'fit params', params
        bst = xgb.train(params, dtrain, evals=[(dtest, 'test')], evals_result=evals_result,
                        early_stopping_rounds=100 if early_stopping else None,
                        num_boost_round=n_estimators if n_estimators else max_n_estimators,
                        verbose_eval=False)

        results = evals_result['test']['rmse'] if self.learning_task == 'regression' \
                  else evals_result['test']['logloss']
        return (
            bst.best_score if early_stopping else results[-1],
            bst.best_iteration if early_stopping else (len(results) - 1),
            results
        )


    def predict(self, bst, dtest, X_test):
        preds = bst.predict(dtest)
        return preds


class CABExperiment(Experiment):

    def __init__(self, learning_task, gpu_id=0,
                 max_n_estimators=1500, max_hyperopt_evals=50,
                 dataset_path='./', output_folder_path='./'):
        Experiment.__init__(self, learning_task, 'cab', gpu_id, max_n_estimators, max_hyperopt_evals,
                            dataset_path, output_folder_path)

        self.space = {
            'learning_rate': hp.loguniform('learning_rate', -3, 0),
            'depth': hp.quniform('depth', 5, 14, 1),
            'random_strength': hp.choice('random_strength', [1, 20]),
            'l2_leaf_reg': hp.loguniform('l2_leaf_reg', 0, np.log(10)),
            'bootstrap_type': hp.choice(
                'bootstrap_type',
                [
                    {'type':'Bayesian',
                     'bagging_temperature': hp.uniform('bagging_temperature', 0, 1)
                    },
                    {'type':'Bernoulli'}
                ]
            ),
            'one_hot_max_size': hp.quniform('one_hot_max_size', 2, 255, 1),
            #'colsample_bylevel': hp.uniform('colsample_bylevel', 0.5, 1),
            'boosting_type': hp.choice('boosting_type', ['Plain', 'Ordered']),
            'used_ram_limit': hp.choice('used_ram_limit', [100000000000]),
        }

        if learning_task == 'classification':
            self.space.update({
                'gradient_iterations' : hp.choice('gradient_iterations', [1, 10])
            })

        self.default_params  = {
            'learning_rate': 0.03,
            'depth': 6,
            'fold_len_multiplier': 2,

             # specify 254 to make results consistent between CPU and GPU (GPU has 128 borders by default)
            'border_count': 254,
            'l2_leaf_reg': 3,
            'leaf_estimation_method': 'Newton',
            'gradient_iterations': 10,
            'bagging_temperature': 1,
            'one_hot_max_size': 2,
            #'colsample_bylevel': 1.0, unsupported on GPU
            'boosting_type': 'Plain',
            'used_ram_limit': 100000000000,
        }
        self.default_params = self.preprocess_params(self.default_params)
        self.title = 'CatBoost'


    def convert_to_dataset(self, data, label, cat_cols):
        return cb.Pool(data, label, cat_features=cat_cols)


    def preprocess_params(self, params):
        if self.learning_task == 'classification':
            params.update({'loss_function': 'Logloss', 'verbose': False, 'thread_count': 32, 'random_seed': 0})
        elif self.learning_task == 'regression':
            params.update({'loss_function': 'RMSE', 'verbose': False, 'thread_count': 32, 'random_seed': 0})

        if 'bootstrap_type' in params:
            if params['bootstrap_type']['type'] == 'Bayesian':
                params['bagging_temperature'] = params['bootstrap_type']['bagging_temperature']
            params['bootstrap_type'] = params['bootstrap_type']['type']


        return params


    def fit(self, params, dtrain, dtest, max_n_estimators, n_estimators=None, early_stopping=False, seed=0):
        if self.gpu_id:
            params.update({"task_type": 'GPU'})
            params.update({'devices': self.gpu_id})
        if early_stopping:
            params.update({"od_wait": 100})
        params.update({"iterations": n_estimators if n_estimators else max_n_estimators})
        params.update({"random_seed": seed})
        bst = cb.CatBoost(params)
        bst.fit(dtrain, eval_set=dtest)

        results = bst.evals_result_['validation_0']['RMSE'] if self.learning_task == 'regression' \
                  else bst.evals_result_['validation_0']['Logloss']

        return (
            bst.best_score_['validation_0']['RMSE'] if early_stopping else results[-1],
            bst.best_iteration_ if early_stopping else (len(results) - 1),
            results
        )

    def predict(self, bst, dtest, X_test):
        preds = np.array(bst.predict(dtest))
        if self.learning_task == 'classification':
            preds = np.power(1 + np.exp(-preds), -1)
        return preds


class LGBExperiment(Experiment):

    def __init__(self, learning_task, gpu_id=0, max_n_estimators=1500,
                 max_hyperopt_evals=50, dataset_path='./', output_folder_path='./'):
        Experiment.__init__(self, learning_task, 'lgb', gpu_id, max_n_estimators, max_hyperopt_evals,
                            dataset_path, output_folder_path)
        self.space = {
            #'boosting': hp.choice('boosting', ['gbdt', 'rf', 'dart', 'goss']),
            'learning_rate': hp.loguniform('learning_rate', -7, 0),
            'num_leaves': hp.qloguniform('num_leaves', 0, 7, 1),
            'max_depth': hp.choice('max_depth', [-1, hp.qloguniform('max_depth_positive', 0, 4, 1)]),
            'min_data_in_leaf': hp.qloguniform('min_data_in_leaf', 0, 6, 1),
            'min_sum_hessian_in_leaf': hp.loguniform('min_sum_hessian_in_leaf', -16, 5),
            'bagging_fraction': hp.uniform('bagging_fraction', 0.5, 1),
            'bagging_freq': hp.choice('bagging_freq', [0, hp.qloguniform('bagging_freq_positive', 0, 4, 1)]),
            'feature_fraction': hp.uniform('feature_fraction', 0.5, 1),

            # tune?
            'max_delta_step': hp.choice('max_delta_step', [0.0, hp.loguniform('max_delta_step_positive', -7, 4)]),
            'lambda_l1': hp.choice('lambda_l1', [0, hp.loguniform('lambda_l1_positive', -16, 2)]),
            'lambda_l2': hp.choice('lambda_l2', [0, hp.loguniform('lambda_l2_positive', -16, 2)]),
            'min_gain_to_split': hp.choice('min_gain_to_split', [0.0, hp.loguniform('min_gain_to_split_positive', -16, -1)]),

            # used only in dart
            # drop_rate, max_drop, skip_drop, uniform_drop

            # used only in goss
            # top_rate, other_rate

            'min_data_per_group': hp.qloguniform('min_data_per_group', 0, 6, 1),
            'max_cat_threshold': hp.quniform('max_cat_threshold', 2, 128, 1),
            'cat_l2': hp.loguniform('cat_l2', 0, 3),
            'cat_smooth': hp.loguniform('cat_smooth', 0, 3),
            'max_cat_to_onehot': hp.qloguniform('max_cat_to_onehot', 0, 5, 1),
        }

        self.default_params = {
            'learning_rate': 0.1,
            'num_leaves': 127,
            'max_depth': -1,
            'feature_fraction': 1.0,
            'bagging_fraction': 1.0,
            'min_data_in_leaf': 100,
            'min_sum_hessian_in_leaf': 10,
            'lambda_l1': 0,
            'lambda_l2': 0,
            'min_data_per_group': 100,
            'max_cat_threshold': 32,
            'max_cat_to_onehot': 4
        }
        self.default_params = self.preprocess_params(self.default_params)
        self.title = 'LightGBM'


    def preprocess_params(self, params):
        params_ = params.copy()
        if self.learning_task == 'classification':
            params_.update({'objective': 'binary', 'metric': 'binary_logloss',
                            'bagging_freq': 1, 'verbose': -1})
        elif self.learning_task == "regression":
            params_.update({'objective': 'mean_squared_error', 'metric': 'l2',
                            'bagging_freq': 1, 'verbose': -1})
        params_['num_leaves'] = max(int(params_['num_leaves']), 2)

        for param in [
            'max_depth',
            'min_data_in_leaf',
            'min_data_per_group',
            'max_cat_threshold',
            'max_cat_to_onehot'
        ]:
            params_[param] = int(params_[param])
        return params_


    def convert_to_dataset(self, data, label, cat_cols=None):
        return lgb.Dataset(data, label)


    def fit(self, params, dtrain, dtest, max_n_estimators, n_estimators=None, early_stopping=False, seed=0):
        params.update({
            'data_random_seed': 1 + seed,
            'feature_fraction_seed': 2 + seed,
            'bagging_seed': 3 + seed,
            'drop_seed': 4 + seed,
        })
        evals_result = {}
        bst = lgb.train(params, dtrain, valid_sets=[dtest], valid_names=['test'], evals_result=evals_result,
                        num_boost_round=n_estimators if n_estimators else max_n_estimators,
                        early_stopping_rounds=100 if early_stopping else None,
                        verbose_eval=False)

        results = np.power(evals_result['test']['l2'], 0.5) if self.learning_task == 'regression' \
                  else evals_result['test']['binary_logloss']

        return (
            results[bst.best_iteration - 1] if early_stopping else results[-1],
            (bst.best_iteration - 1) if early_stopping else (bst.num_trees() - 1),
            results
        )


    def predict(self, bst, dtest, X_test):
        preds = bst.predict(X_test)
        return preds



def createParser():
    parser = argparse.ArgumentParser()
    parser.add_argument('bst', choices=['xgb', 'lgb', 'cab'])

    parser.add_argument('-t', '--max_n_estimators', type=int, default=1500)
    parser.add_argument('-n', '--hyperopt_evals', type=int, default=50)
    parser.add_argument('--gpu_id', type=int, default=None,
                        help='Set gpu_id to run on GPU (CPU is the default)')
    return parser

if __name__ == "__main__":
    parser = createParser()
    namespace = parser.parse_args(sys.argv[1:])

    if namespace.bst == 'xgb':
    	ExperimentClass = XGBExperiment
    elif namespace.bst == 'lgb':
    	ExperimentClass = LGBExperiment
    elif namespace.bst == 'cab':
        ExperimentClass = CABExperiment


    experiment = ExperimentClass(
        'regression',
        namespace.gpu_id,
        namespace.max_n_estimators,
        namespace.hyperopt_evals,
        dataset_path=config.preprocessed_dataset_path,
        output_folder_path=os.path.join(config.training_output_path, namespace.bst + 'ExperimentHyperopt')
    )
    experiment.run()
