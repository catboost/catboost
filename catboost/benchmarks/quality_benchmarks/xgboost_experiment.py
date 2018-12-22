import xgboost as xgb
from hyperopt import hp
from experiment import Experiment


class XGBExperiment(Experiment):

    def __init__(self, learning_task, n_estimators=5000, max_hyperopt_evals=50, 
                 counters_sort_col=None, holdout_size=0, 
                 train_path=None, test_path=None, cd_path=None, output_folder_path='./'):
        Experiment.__init__(self, learning_task, 'xgb', n_estimators, max_hyperopt_evals, 
                            True, counters_sort_col, holdout_size, 
                            train_path, test_path, cd_path, output_folder_path)

        self.space = {
            'eta': hp.loguniform('eta', -7, 0),
            'max_depth' : hp.quniform('max_depth', 2, 10, 1),
            'subsample': hp.uniform('subsample', 0.5, 1),
            'colsample_bytree': hp.uniform('colsample_bytree', 0.5, 1),
            'colsample_bylevel': hp.uniform('colsample_bylevel', 0.5, 1),
            'min_child_weight': hp.loguniform('min_child_weight', -16, 5),
            'alpha': hp.choice('alpha', [0, hp.loguniform('alpha_positive', -16, 2)]),
            'lambda': hp.choice('lambda', [0, hp.loguniform('lambda_positive', -16, 2)]),
            'gamma': hp.choice('gamma', [0, hp.loguniform('gamma_positive', -16, 2)])
        }

        self.default_params = {'eta': 0.3, 'max_depth': 6, 'subsample': 1.0, 
                               'colsample_bytree': 1.0, 'colsample_bylevel': 1.0,
                               'min_child_weight': 1, 'alpha': 0, 'lambda': 1, 'gamma': 0}
        self.default_params = self.preprocess_params(self.default_params)
        self.title = 'XGBoost'


    def preprocess_params(self, params):
        if self.learning_task == "classification":
            params.update({'objective': 'binary:logistic', 'eval_metric': 'logloss', 'silent': 1})
        elif self.learning_task == "regression":
            params.update({'objective': 'reg:linear', 'eval_metric': 'rmse', 'silent': 1})
        params['max_depth'] = int(params['max_depth'])
        return params


    def convert_to_dataset(self, data, label, cat_cols=None):
        return xgb.DMatrix(data, label)


    def fit(self, params, dtrain, dtest, n_estimators, seed=0):
        params.update({"seed": seed})
        evals_result = {}
        bst = xgb.train(params, dtrain, evals=[(dtest, 'test')], evals_result=evals_result,
                        num_boost_round=n_estimators, verbose_eval=False)
        
        results = evals_result['test']['rmse'] if self.learning_task == 'regression' \
                  else evals_result['test']['logloss']
        return bst, results


    def predict(self, bst, dtest, X_test):
        preds = bst.predict(dtest)
        return preds
