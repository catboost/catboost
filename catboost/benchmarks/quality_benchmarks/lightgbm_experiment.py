import lightgbm as lgb
from hyperopt import hp
from experiment import Experiment
import numpy as np


class LGBExperiment(Experiment):

    def __init__(self, learning_task, n_estimators=5000, max_hyperopt_evals=50, 
                 counters_sort_col=None, holdout_size=0, 
                 train_path=None, test_path=None, cd_path=None, output_folder_path='./'):
        Experiment.__init__(self, learning_task, 'lgb', n_estimators, max_hyperopt_evals, 
                            True, counters_sort_col, holdout_size, 
                            train_path, test_path, cd_path, output_folder_path)

        self.space = {
            'learning_rate': hp.loguniform('learning_rate', -7, 0),
            'num_leaves' : hp.qloguniform('num_leaves', 0, 7, 1),
            'feature_fraction': hp.uniform('feature_fraction', 0.5, 1),
            'bagging_fraction': hp.uniform('bagging_fraction', 0.5, 1),
            'min_data_in_leaf': hp.qloguniform('min_data_in_leaf', 0, 6, 1),
            'min_sum_hessian_in_leaf': hp.loguniform('min_sum_hessian_in_leaf', -16, 5),
            'lambda_l1': hp.choice('lambda_l1', [0, hp.loguniform('lambda_l1_positive', -16, 2)]),
            'lambda_l2': hp.choice('lambda_l2', [0, hp.loguniform('lambda_l2_positive', -16, 2)]),
        }

        self.default_params = {'learning_rate': 0.1, 'num_leaves': 127, 'feature_fraction': 1.0, 
                               'bagging_fraction': 1.0, 'min_data_in_leaf': 100, 'min_sum_hessian_in_leaf': 10, 
                               'lambda_l1': 0, 'lambda_l2': 0}
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
        params_['min_data_in_leaf'] = int(params_['min_data_in_leaf'])
        return params_


    def convert_to_dataset(self, data, label, cat_cols=None):
        return lgb.Dataset(data, label)


    def fit(self, params, dtrain, dtest, n_estimators, seed=0):
        params.update({
            'data_random_seed': 1 + seed, 
            'feature_fraction_seed': 2 + seed, 
            'bagging_seed': 3 + seed, 
            'drop_seed': 4 + seed, 
        })
        evals_result = {}
        bst = lgb.train(params, dtrain, valid_sets=[dtest], valid_names=['test'], evals_result=evals_result,
                        num_boost_round=n_estimators, verbose_eval=False)
        
        results = np.power(evals_result['test']['l2'], 0.5) if self.learning_task == 'regression' \
                  else evals_result['test']['binary_logloss']
        return bst, results


    def predict(self, bst, dtest, X_test):
        preds = bst.predict(X_test)
        return preds

