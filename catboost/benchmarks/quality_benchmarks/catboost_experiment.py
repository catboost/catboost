from catboost import CatBoost, Pool
from hyperopt import hp
from experiment import Experiment
import os, numpy as np


class CABExperiment(Experiment):

    def __init__(self, learning_task, n_estimators=5000, max_hyperopt_evals=50, 
                 counters_sort_col=None, holdout_size=0, 
                 train_path=None, test_path=None, cd_path=None, output_folder_path='./'):
        assert holdout_size == 0, 'For Catboost holdout_size must be equal to 0'
        Experiment.__init__(self, learning_task, 'cab', n_estimators, max_hyperopt_evals, 
                            False, None, holdout_size, 
                            train_path, test_path, cd_path, output_folder_path)

        self.space = {
            'depth': hp.choice('depth', [6]),
            'ctr_border_count': hp.choice('ctr_border_count', [16]),
            'border_count': hp.choice('border_count', [128]),
            'ctr_description': hp.choice('ctr_description', [['Borders','CounterMax']]),
            'learning_rate': hp.loguniform('learning_rate', -5, 0),
            'random_strength': hp.choice('random_strength', [1, 20]),
            'one_hot_max_size': hp.choice('one_hot_max_size', [0, 25]),
            'l2_leaf_reg': hp.loguniform('l2_leaf_reg', 0, np.log(10)),
            'bagging_temperature': hp.uniform('bagging_temperature', 0, 1),
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
            'rsm': 1.0,
            'border_count': 128,
            'ctr_border_count': 16,
            'l2_leaf_reg': 3,
            'leaf_estimation_method': 'Newton',
            'gradient_iterations': 10,
            'ctr_description': ['Borders','CounterMax'],
            'used_ram_limit': 100000000000,
        }
        self.default_params = self.preprocess_params(self.default_params)
        self.title = 'CatBoost'

    
    def convert_to_dataset(self, data, label, cat_cols):
        return Pool(data, label, cat_features=cat_cols)


    def preprocess_params(self, params):
        if self.learning_task == 'classification':
            params.update({'loss_function': 'Logloss', 'verbose': False, 'thread_count': 16, 'random_seed': 0})
        elif self.learning_task == 'regression':
            params.update({'loss_function': 'RMSE', 'verbose': False, 'thread_count': 16, 'random_seed': 0})
        return params


    def fit(self, params, dtrain, dtest, n_estimators, seed=0):
        params.update({"iterations": n_estimators})
        params.update({"random_seed": seed})
        bst = CatBoost(params)
        bst.fit(dtrain, eval_set=dtest)
        with open("test_error.tsv", "r") as f:
            results = np.array(map(lambda x: float(x.strip().split()[-1]), f.readlines()[1:]))
        
        return bst, results

    def predict(self, bst, dtest, X_test):
        preds = np.array(bst.predict(dtest))
        if self.learning_task == 'classification':
            preds = np.power(1 + np.exp(-preds), -1)
        return preds

