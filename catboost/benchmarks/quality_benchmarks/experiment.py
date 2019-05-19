import pandas as pd, numpy as np
import pickle, time
from sklearn.model_selection import StratifiedKFold, KFold, train_test_split
from hyperopt import fmin, tpe, Trials, STATUS_OK, STATUS_FAIL
from datetime import datetime
from cat_counter import CatCounter
#from pandas.io.common import EmptyDataError
import os


class Experiment(object):

    def __init__(self, learning_task='classification', bst_name=None, n_estimators=5000, hyperopt_evals=50,
                 compute_counters=True, counters_sort_col=None, holdout_size=0,
                 train_path=None, test_path=None, cd_path=None, output_folder_path='./'):
        self.learning_task, self.bst_name = learning_task, bst_name
        self.compute_counters = compute_counters
        self.holdout_size = holdout_size
        self.counters_sort_col = counters_sort_col
        self.n_estimators, self.best_loss = n_estimators, np.inf
        self.best_n_estimators = None
        self.hyperopt_evals, self.hyperopt_eval_num = hyperopt_evals, 0
        self.train_path, self.test_path, self.cd_path = train_path, test_path, cd_path
        self.output_folder_path = os.path.join(output_folder_path, '')
        self.default_params, self.best_params = None, None
        self.title = None
        if self.learning_task == 'classification':
            self.metric = 'logloss'
        elif self.learning_task == 'regression':
            self.metric = 'rmse'
        else:
            raise ValueError('Task type must be "classification" or "regression"')


    def read_file(self, file_name, target_col):
        X = pd.read_csv(file_name, sep='\t', header=None)
        if self.learning_task == 'classification':
            y = np.maximum(X[target_col].values, 0)
        else:
            y = X[target_col].values
        X.drop(target_col, axis=1, inplace=True)
        return X, y


    def read_data(self):
        cols = pd.read_csv(self.cd_path, sep='\t', header=None)
        target_col = cols[0][np.where(cols[1] == 'Target')[0][0]]
        cat_cols = cols[cols[1] == "Categ"][0].values

        X_train, y_train = self.read_file(self.train_path, target_col)
        X_test, y_test = self.read_file(self.test_path, target_col)
        data = pd.concat([X_train, X_test])
        data[cat_cols] = data[cat_cols].apply(lambda x: x.astype('category').cat.codes)
        data = np.array(data).astype('float')
        X_train, X_test = data[:X_train.shape[0]], data[X_train.shape[0]:]

        cat_cols[cat_cols > target_col] = cat_cols[cat_cols > target_col] - 1
        return X_train, y_train, X_test, y_test, cat_cols


    def convert_to_dataset(self, data, label, cat_cols=None):
        raise NotImplementedError('Method convert_to_dataset is not implemented.')


    def preprocess_cat_cols(self, X_train, y_train, cat_cols, X_test=None, cc=None):
        if self.compute_counters == False:
            return None
        if cc is None:
            sort_values = None if self.counters_sort_col is None else X_train[:, self.counters_sort_col]
            cc = CatCounter(self.learning_task, sort_values)
            X_train[:,cat_cols] = cc.fit(X_train[:,cat_cols], y_train)
        else:
            X_train[:,cat_cols] = cc.transform(X_train[:,cat_cols])
        if not X_test is None:
            X_test[:,cat_cols] = cc.transform(X_test[:,cat_cols])
        return cc


    def split_and_preprocess(self, X_train, y_train, X_test, y_test, cat_cols, n_splits=5, random_state=0):
        if self.holdout_size > 0:
            print('Holdout is used for counters.')
            X_train, X_hout, y_train, y_hout = train_test_split(X_train, y_train,
                                                                test_size=self.holdout_size,
                                                                random_state=random_state)
            cc = self.preprocess_cat_cols(X_hout, y_hout, cat_cols)
        else:
            cc = None

        CVSplit = KFold if self.learning_task == 'regression' else StratifiedKFold
        cv = CVSplit(n_splits=n_splits, shuffle=True, random_state=random_state)
        cv_pairs = []

        for train_index, test_index in cv.split(X_train, y_train):
            train, test = X_train[train_index], X_train[test_index]
            _ = self.preprocess_cat_cols(train, y_train[train_index], cat_cols, test, cc)
            dtrain = self.convert_to_dataset(train.astype(float), y_train[train_index], cat_cols)
            dtest = self.convert_to_dataset(test.astype(float), y_train[test_index], cat_cols)
            cv_pairs.append((dtrain, dtest))

        _ = self.preprocess_cat_cols(X_train, y_train, cat_cols, X_test, cc)
        dtrain = self.convert_to_dataset(X_train.astype(float), y_train, cat_cols)
        dtest = self.convert_to_dataset(X_test.astype(float), y_test, cat_cols)

        return cv_pairs, (dtrain, dtest)


    def fit(self, params, dtrain, dtest, n_estimators):
        raise NotImplementedError('Method train is not implemented.')


    def predict(self, bst, dtest, X_test):
        raise NotImplementedError('Method predict is not implemented.')


    def preprocess_params(self, params):
        raise NotImplementedError('Method preprocess_params is not implemented.')


    def run_cv(self, cv_pairs, params=None, n_estimators=None, verbose=False):
        params = params or self.default_params
        n_estimators = n_estimators or self.n_estimators
        params = self.preprocess_params(params)
        evals_results, start_time = [], time.time()
        for dtrain, dtest in cv_pairs:
            _, evals_result = self.fit(params, dtrain, dtest, n_estimators)
            evals_results.append(evals_result)
        mean_evals_results = np.mean(evals_results, axis=0)
        best_n_estimators = np.argmin(mean_evals_results) + 1
        eval_time = time.time() - start_time

        cv_result = {'loss': mean_evals_results[best_n_estimators - 1],
                     'best_n_estimators': best_n_estimators,
                     'eval_time': eval_time,
                     'status': STATUS_FAIL if np.isnan(mean_evals_results[best_n_estimators - 1]) else STATUS_OK,
                     'params': params.copy()}
        self.best_loss = min(self.best_loss, cv_result['loss'])
        self.hyperopt_eval_num += 1
        cv_result.update({'hyperopt_eval_num': self.hyperopt_eval_num, 'best_loss': self.best_loss})

        if verbose:
            print '[{0}/{1}]\teval_time={2:.2f} sec\tcurrent_{3}={4:.6f}\tmin_{3}={5:.6f}'.format(
                        self.hyperopt_eval_num, self.hyperopt_evals, eval_time,
                        self.metric, cv_result['loss'], self.best_loss)
        return cv_result


    def run_test(self, dtrain, dtest, X_test=None, params=None, n_estimators=None, custom_metric=None, seed=0):
        params = params or self.best_params or self.default_params
        n_estimators = n_estimators or self.best_n_estimators or self.n_estimators
        params = self.preprocess_params(params)
        start_time = time.time()
        bst, evals_result = self.fit(params, dtrain, dtest, n_estimators, seed=seed)
        eval_time = time.time() - start_time
        preds = self.predict(bst, dtest, X_test)

        result = {'loss': evals_result[-1], 'bst': bst, 'n_estimators': n_estimators,
                  'eval_time': eval_time, 'status': STATUS_OK,  'params': params.copy(),
                  'preds': preds}

        if custom_metric is not None:
            if type(custom_metric) is not dict:
                raise TypeError("custom_metric argument should be dict")
            pred = self.predict(bst, dtest, X_test)
            for title, func in custom_metric.iteritems():
                score = func(dtest.get_label(), pred, sample_weight=None) # TODO weights
                result[title] = score

        return result


    def optimize_params(self, cv_pairs, max_evals=None, verbose=True):
        max_evals = max_evals or self.hyperopt_evals
        self.trials = Trials()
        self.hyperopt_eval_num, self.best_loss = 0, np.inf

        _ = fmin(fn=lambda params: self.run_cv(cv_pairs, params, verbose=verbose),
                 space=self.space, algo=tpe.suggest, max_evals=max_evals, trials=self.trials, rseed=1)

        self.best_params = self.trials.best_trial['result']['params']
        self.best_n_estimators = self.trials.best_trial['result']['best_n_estimators']
        return self.trials.best_trial['result']


    def dump(self, preds, elementwise_losses, test_losses, file_name):
        results = {'trials': self.trials, 'best_params': self.best_params,
                   'best_n_estimators': self.best_n_estimators,
                   'preds': preds, 'elementwise_losses': elementwise_losses, 'test_losses': test_losses}
        with open(file_name, 'wb') as f:
            pickle.dump(results, f)


    def load(self, file_name):
        with open(file_name, 'r') as f:
            results = pickle.load(f)
        self.trials = results['trials']
        self.best_params = results['best_params']
        self.best_n_estimators = results['best_n_estimators']
        preds = results['preds']
        losses = results['losses']
        test_loss = results['test_loss']
        return preds, losses, test_loss


    def print_result(self, result, name='', extra_keys=None):
        print '%s:\n' % name
        print '%s = %s' % (self.metric, result['loss'])
        if 'best_n_estimators' in result.keys():
            print 'best_n_estimators = %s' % result['best_n_estimators']
        elif 'n_estimators' in result.keys():
            print 'n_estimators = %s' % result['n_estimators']
        print 'params = %s' % result['params']
        if extra_keys is not None:
            for k in extra_keys:
                if k in result:
                    print "%s = %f" % (k, result[k])


    def elementwise_loss(self, y, p):
        if self.learning_task == 'classification':
            p_ = np.clip(p, 1e-16, 1-1e-16)
            return - y * np.log(p_) - (1 - y) * np.log(1 - p_)
        return (y - p) ** 2


    def run(self):
        print 'Loading and preprocessing dataset...'
        X_train, y_train, X_test, y_test, cat_cols = self.read_data()
        cv_pairs, (dtrain, dtest) = self.split_and_preprocess(X_train, y_train, X_test, y_test, cat_cols)

        print 'Optimizing params...'
        cv_result = self.optimize_params(cv_pairs)
        self.print_result(cv_result, '\nBest result on cv')

        print '\nTraining algorithm with the tuned parameters for different seed...'
        preds, test_losses, elementwise_losses = [], [], []
        for seed in range(5):
            test_result = self.run_test(dtrain, dtest, X_test, seed=seed)
            preds.append(test_result['preds'])
            test_losses.append(test_result['loss'])
            elementwise_losses.append(self.elementwise_loss(y_test, preds[-1]))
            print 'For seed=%d Test\'s %s : %.5f' % (seed, self.metric, test_losses[-1])

        print '\nTest\'s %s mean: %.5f, Test\'s %s std: %.5f' % (self.metric, np.mean(test_losses), self.metric, np.std(test_losses))

        if not self.output_folder_path is None:
            date = datetime.now().strftime('%Y%m%d-%H%M%S')
            dataset_name = self.train_path.replace('/', ' ').strip().split()[-2]
            file_name = '{}{}_results_{}_{}.pkl'.format(self.output_folder_path, self.bst_name, dataset_name, date)
            self.dump(preds, elementwise_losses, test_losses, file_name)
            print 'Results are saved to %s' % file_name

