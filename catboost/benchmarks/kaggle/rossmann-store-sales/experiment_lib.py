import os.path
import time

import numpy as np
import pandas as pd
import scipy.stats

from sklearn.model_selection import (
    train_test_split,
    GridSearchCV,
    RandomizedSearchCV,
    TimeSeriesSplit
)



class ExperimentBase(object):

    def __init__(self, train_path, test_path, cd_path, output_folder_path='./',
                 learning_task='regression', scoring='neg_mean_squared_error',
                 header_in_data=True):
        self.train_path, self.test_path, self.cd_path = train_path, test_path, cd_path
        self.output_folder_path = os.path.join(output_folder_path, '')
        self.learning_task = learning_task
        self.scoring = scoring
        self.header_in_data = header_in_data

    def read_file(self, file_name, target_col, header_in_data):
        X = pd.read_csv(file_name, sep='\t', header=0 if header_in_data else None)
        if self.learning_task == 'classification':
            y = np.maximum(X[target_col].values, 0)
        else:
            y = X[target_col].values
        X.drop(target_col, axis=1, inplace=True)
        return X, y

    def read_data(self, header_in_data):
        cols = pd.read_csv(self.cd_path, sep='\t', header=None)
        target_col = np.where(cols[1] == 'Label')[0][0]
        cat_cols = cols[cols[1] == "Categ"][0].values

        X_train, y_train = self.read_file(self.train_path, target_col, header_in_data)
        X_test, y_test = self.read_file(self.test_path, target_col, header_in_data)
        data = pd.concat([X_train, X_test])
        data[cat_cols] = data[cat_cols].apply(lambda x: x.astype('category').cat.codes)
        data = np.array(data).astype('float')
        X_train, X_test = data[:X_train.shape[0]], data[X_train.shape[0]:]

        cat_cols[cat_cols > target_col] = cat_cols[cat_cols > target_col] - 1
        return X_train, y_train, X_test, y_test, cat_cols

    def optimize_params(X_train, y_train, cat_cols):
        raise NotImplementedError('Method optimize_params is not implemented.')

    # override if necessary
    def predict_with_best_estimator(self, X_test):
        return self.best_estimator.predict(X_test)

    def print_result(self, name=''):
        print ('%s:\n' % name)
        print ('best_params = %s' % self.best_params)
        print ('best_score = %s' % self.best_score)
        if hasattr(self, 'best_iteration'):
            print ('best_iteration = %s' % self.best_iteration)

    def eval_on_test(self, X_test, y_test):
        y_pred = self.predict_with_best_estimator(X_test)
        print ('best_predictor\'s score on test = %s ' % np.mean((y_test - y_pred) ** 2) ** 0.5)

    def run(self):
        print 'Loading dataset...'
        X_train, y_train, X_test, y_test, cat_cols = self.read_data(self.header_in_data)

        print 'Optimizing params...'
        self.optimize_params(X_train, y_train, cat_cols)
        self.print_result('\nBest result')

        self.eval_on_test(X_test, y_test)


class ExperimentEarlyStopping(ExperimentBase):

    def __init__(self, **kwargs):
        super(ExperimentEarlyStopping, self).__init__(**kwargs)

    def get_estimator(self, cat_cols):
        raise NotImplementedError('Method get_estimator is not implemented.')

    # override if necessary
    def fit_estimator(self, estimator, X_train, y_train, X_test, y_test, cat_cols, early_stopping_rounds):
        estimator.fit(
            X_train,
            y_train,
            eval_set=[(X_test, y_test)],
            early_stopping_rounds=early_stopping_rounds
        )

        self.best_estimator = estimator
        self.best_iteration = estimator.best_iteration_
        self.best_params = estimator.get_params()
        self.best_score = estimator.best_score_

    def optimize_params(self, X, y, cat_cols, verbose=True):
        X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=False)

        t_start = time.time()

        estimator = self.get_estimator(cat_cols)

        self.fit_estimator(
            estimator,
            X_train,
            y_train,
            X_test,
            y_test,
            cat_cols,
            early_stopping_rounds=100
        )

        print ('estimation took %s sec' % (time.time() - t_start))


class ExperimentGridSearchCV(ExperimentBase):

    def __init__(self, **kwargs):
        super(ExperimentGridSearchCV, self).__init__(**kwargs)

    def get_estimator(self, cat_cols):
        raise NotImplementedError('Method get_estimator is not implemented.')

    def get_param_grid(self):
        raise NotImplementedError('Method get_param_grid is not implemented.')

    # override if necessary
    def call_fit(self, grid_search_instance, X, y, cat_cols):
        grid_search_instance.fit(X, y)

    def optimize_params(self, X, y, cat_cols, verbose=True):
        t_start = time.time()

        gs = GridSearchCV(
            estimator=self.get_estimator(cat_cols),
            param_grid=self.get_param_grid(),
            n_jobs=-1,
            cv=TimeSeriesSplit(n_splits=3).split(X, y),
            scoring=self.scoring,
            refit=True,
            verbose=1,
            pre_dispatch=1,
        )

        self.call_fit(gs, X, y, cat_cols)

        print ('GridSearchCV took %s sec' % (time.time() - t_start))

        self.best_estimator = gs.best_estimator_
        self.best_params = gs.best_params_
        self.best_score = np.sqrt(-gs.best_score_)


class ExperimentRandomSearchCV(ExperimentBase):

    def __init__(self, **kwargs):
        super(ExperimentRandomSearchCV, self).__init__(**kwargs)

    def get_estimator(self, cat_cols):
        raise NotImplementedError('Method get_estimator is not implemented.')

    def get_param_distributions(self):
        raise NotImplementedError('Method get_param_distributions is not implemented.')

    # override if necessary
    def call_fit(self, randomized_search_instance, X, y, cat_cols):
        randomized_search_instance.fit(X, y)

    def optimize_params(self, X, y, cat_cols, verbose=True):
        t_start = time.time()

        rs = RandomizedSearchCV(
            estimator=self.get_estimator(cat_cols),
            param_distributions=self.get_param_distributions(),
            n_jobs=1,
            cv=TimeSeriesSplit(n_splits=3).split(X, y),
            scoring=self.scoring,
            refit=True,
            verbose=10,
            random_state=0,
            pre_dispatch=1,
        )

        self.call_fit(rs, X, y, cat_cols)

        print ('RandomizedSearchCV took %s sec' % (time.time() - t_start))

        self.best_estimator = rs.best_estimator_
        self.best_params = rs.best_params_
        self.best_score = np.sqrt(-rs.best_score_)


# helper class for param distribution
class LogUniform(object):

    def __init__(self, l_bound, r_bound, is_integral=False):
        self.l_bound = np.log(l_bound)
        self.width = np.log(r_bound) - self.l_bound
        self.uniform = scipy.stats.uniform()
        self.is_integral = is_integral

    def rvs(self, random_state=None):
        cont_sample = np.exp(self.l_bound + self.width * self.uniform.rvs(random_state=random_state))
        return int(cont_sample) if self.is_integral else cont_sample

