from catboost import CatBoost, Pool
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


class Ranker:
    def eval_ndcg(self, data, eval_period=10):
        staged_predictions = self.staged_predict(data, eval_period)

        eval_log = []
        for y_pred in staged_predictions:
            value = mean_ndcg(y_pred, data.y_test, data.queries_test)
            eval_log.append(value)

        return eval_log

    def fit(self, train):
        raise Exception('call of interface function')

    def staged_predict(self, data, eval_period):
        raise Exception('call of interface function')


class CatBoostRanker(Ranker):
    def __init__(self, params):
        self.params = params
        if params['loss_function'] == 'PairLogitPairwise' and params['max_depth'] >= 8:
            raise Exception('max_depth for pair-logit-pairwise should be < 8')
        self.model = CatBoost(params)

    def fit(self, data):
        self.model.fit(X=data.train_pool)

    def staged_predict(self, data, eval_period):
        return list(self.model.staged_predict(data.test_pool, eval_period=eval_period))


class XGBoostRanker(Ranker):
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


class LightGBMRanker(Ranker):
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
