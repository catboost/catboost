import argparse
import datetime
import json
import os
from copy import deepcopy

from sklearn.model_selection import ParameterGrid

from models import *
from utils import read_dataset

RANDOM_SEED = 0


def argmin(fn, space):
    best_score = np.NINF
    best_params = {}

    for params in ParameterGrid(space):
        try:
            score = fn(params)
        except Exception as e:
            print('Exception during training: ' + repr(e))
            continue

        if score > best_score:
            best_score = score
            best_params = params

    return {'best_score': best_score, 'best_params': best_params}


def _params_to_str(params):
    return ''.join(map(lambda (key, value): '{}[{}]'.format(key, str(value)), params.items()))


def eval_params(ranker_name, RankerType, data, static_params, param_space, log_file, out_file):
    if not os.path.exists(log_file):
        with open(log_file, 'w') as f:
            json.dump({}, f)

    def objective(params):
        ranker_params = deepcopy(static_params)
        ranker_params.update(params)
        print('Fit with params: ' + _params_to_str(ranker_params))

        with open(log_file, 'r') as f:
            log = json.load(f)
            if ranker_name not in log:
                log[ranker_name] = {}
            params_str = _params_to_str(params)
            if params_str in log[ranker_name]:
                print('Return result from cache')
                return max(log[ranker_name][params_str]['ndcg'])

        ranker = RankerType(ranker_params)

        start = datetime.datetime.now()

        ranker.fit(data)

        train_time = datetime.datetime.now() - start
        train_time = train_time.total_seconds()

        eval_log = ranker.eval_ndcg(data)

        log[ranker_name][params_str] = {
            'time': train_time,
            'ndcg': eval_log
        }
        dump = log_file + '.dmp'

        with open(dump, 'w') as f:
            json.dump(log, f, indent=4, sort_keys=True)

        os.rename(dump, log_file)

        return max(eval_log)

    best_params = argmin(fn=objective, space=param_space)

    print('Best params:' + str(best_params))

    with open(out_file, 'w') as f:
        json.dump(best_params, f, indent=4, sort_keys=True)

    return best_params


def print_versions():
    import catboost
    import xgboost
    import lightgbm
    print('CatBoost: ' + catboost.__version__)
    print('XGBoost : ' + xgboost.__version__)
    print('LightGBM: ' + lightgbm.__version__)


if __name__ == "__main__":
    rankers = {
        'xgb-rmse': [XGBoostRanker, 'reg:linear'],
        'xgb-lmart-ndcg': [XGBoostRanker, 'rank:ndcg'],
        'xgb-pairwise': [XGBoostRanker, 'rank:pairwise'],
        'lgb-rmse': [LightGBMRanker, 'regression'],
        'lgb-pairwise': [LightGBMRanker, 'lambdarank'],
        'cat-rmse': [CatBoostRanker, 'RMSE'],
        'cat-query-rmse': [CatBoostRanker, 'QueryRMSE'],
        'cat-pair-logit': [CatBoostRanker, 'PairLogit'],
        'cat-pair-logit-pairwise': [CatBoostRanker, 'PairLogitPairwise'],
        'cat-yeti-rank': [CatBoostRanker, 'YetiRank'],
        'cat-yeti-rank-pairwise': [CatBoostRanker, 'YetiRankPairwise']
    }

    parser = argparse.ArgumentParser()
    parser.add_argument('--learner', choices=rankers.keys(), required=True)
    parser.add_argument('--train', required=True)
    parser.add_argument('--test', required=True)
    parser.add_argument('-s', '--param-space', required=True)
    parser.add_argument('-o', '--out-file', required=True)
    parser.add_argument('-l', '--log-file', required=True)
    parser.add_argument('-n', '--iterations', type=int, default=10000)
    parser.add_argument('--use-gpu', action='store_true')
    args = parser.parse_args()

    print_versions()

    RankerType = rankers[args.learner][0]
    loss_function = rankers[args.learner][1]

    if RankerType == CatBoostRanker:
        static_params = {
            'loss_function': loss_function,
            'logging_level': 'Silent',
            'use_best_model': False,
            'random_seed': RANDOM_SEED
        }

        if args.use_gpu:
            static_params['task_type'] = 'GPU'
            static_params['devices'] = [0]

    elif RankerType == XGBoostRanker:
        static_params = {
            'silent': 1,
            'seed': RANDOM_SEED,
            'tree_method': 'hist'
        }

        tree_method = args.learner

        if args.use_gpu:
            static_params['tree_method'] = 'gpu_hist'
            static_params['gpu_id'] = 0

    elif RankerType == LightGBMRanker:
        static_params = {
            'verbose': 0,
            'boosting_type': 'gbdt',
            'random_seed': RANDOM_SEED
        }

        if args.use_gpu:
            static_params['device'] = 'gpu'
            static_params['gpu_device_id'] = 0

    static_params['iterations'] = args.iterations

    with open(args.param_space) as f:
        param_space = json.load(f)

    train = read_dataset(args.train)
    test = read_dataset(args.test)
    data = Data(train, test, RankerType)

    result = eval_params(args.learner, RankerType, data, static_params, param_space, args.log_file, args.out_file)
    print('NDCG best value: ' + str(result))
