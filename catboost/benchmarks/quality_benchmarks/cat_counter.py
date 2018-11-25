import numpy as np
from collections import defaultdict


class CatCounter(object):

    def __init__(self, learning_task, sort_values=None, seed=0):
        self.learning_task = learning_task
        self.sort_values = sort_values
        self.seed = seed
        self.sum_dicts = defaultdict(lambda : defaultdict(float))
        self.count_dicts = defaultdict(lambda : defaultdict(float))


    def update(self, value, col, key):
        self.sum_dicts[col][key] += value
        self.count_dicts[col][key] += 1


    def counter(self, key, col):
        num, den = self.sum_dicts[col][key], self.count_dicts[col][key]
        if self.learning_task == 'classification':
            return (num + 1.) / (den + 2.)
        elif self.learning_task == 'regression':
            return num / den if den > 0 else 0
        else:
            raise ValueError('Task type must be "classification" or "regression"')
    

    def fit(self, X, y):
        self.sum_dicts = defaultdict(lambda : defaultdict(float))
        self.count_dicts = defaultdict(lambda : defaultdict(float))

        if self.sort_values is None:
            indices = np.arange(X.shape[0])
            np.random.seed(self.seed)
            np.random.shuffle(indices)
        else:
            indices = np.argsort(self.sort_values)

        results = [np.zeros((X.shape[0], 0))]
        for col in range(X.shape[1]):
            result = np.zeros(X.shape[0])
            for index in indices:
                key = X[index, col]
                result[index] = self.counter(key, col)
                self.update(y[index], col, key)
            results.append(result.reshape(-1, 1))

        return np.concatenate(results, axis=1)
        
        
    def transform(self, X):
        results = [np.zeros((X.shape[0], 0))]
        for col in range(X.shape[1]):
            result = np.zeros(X.shape[0])
            for index in range(X.shape[0]):
                key = X[index, col]
                result[index] = self.counter(key, col)
            results.append(result.reshape(-1, 1))
        return np.concatenate(results, axis=1)

