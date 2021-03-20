from catboost import CatBoostRanker, Pool
from collections import Counter
from utils import mean_ndcg
import lightgbm as lgb
import xgboost as xgb


class Data:
    def __init__(self, train, test, RankerType):
        self.X_train = train[0]
        self.y_train = train[1]
        self.queries_train = train[2]

        self.X_test = test[0]
        self.y_test = test[1]
        self.queries_test = test[2]

        if RankerType is CatBoostRanker:
            self.train_pool = Pool(data=self.X_train, label=self.y_train, group_id=self.queries_train)
            self.test_pool = Pool(data=self.X_test, label=self.y_test, group_id=self.queries_test)

        elif RankerType == XGBoostRanker:
            self.train_pool = xgb.DMatrix(self.X_train, label=self.y_train, silent=True)
            self.test_pool = xgb.DMatrix(self.X_test, label=self.y_test, silent=True)

            group_train = Counter(self.queries_train).values()
            group_test = Counter(self.queries_test).values()

            self.train_pool.set_group(group_train)
            self.test_pool.set_group(group_test)

        elif RankerType == LightGBMRanker:
            group_train = Counter(self.queries_train).values()
            self.group_test = Counter(self.queries_test).values()
            self.train_pool = lgb.Dataset(self.X_train, self.y_train, group=group_train)


class XGBoostRanker:
    def __init__(self, params):
        self.params = params

        self.iterations = self.params['iterations']
        del self.params['iterations']

    def fit(self, data):
        self.model = xgb.train(
            params=self.params,
            num_boost_round=self.iterations,
            dtrain=data.train_pool
        )

    def staged_predict(self, data, eval_period):
        staged_predictions = []
        for i in xrange(0, self.iterations, eval_period):
            prediction = self.model.predict(data.test_pool, ntree_limit=i + 1)
            staged_predictions.append(prediction)

        return staged_predictions


class LightGBMRanker:
    def __init__(self, params):
        self.params = params

        self.params['num_leaves'] = 2 ** self.params['max_depth']
        del self.params['max_depth']

        self.iterations = self.params['iterations']
        del self.params['iterations']

    def fit(self, data):
        self.model = lgb.train(
            self.params,
            data.train_pool,
            num_boost_round=self.iterations
        )

    def staged_predict(self, data, eval_period):
        staged_predictions = []
        for i in xrange(0, self.iterations, eval_period):
            prediction = self.model.predict(data.X_test, i + 1, group=data.group_test)
            staged_predictions.append(prediction)

        return staged_predictions
