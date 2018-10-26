import argparse
import json

from experiments import *
from learners import *

LEARNERS = {
    "xgb": XGBoostLearner,
    "lgb": LightGBMLearner,
    "cat": CatBoostLearner
}


def _get_all_values_from_subset(items, subset):
    filtered_keys = filter(lambda x: x in subset, items.keys())
    return [items[key] for key in filtered_keys]


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--learners', nargs='+', choices=LEARNERS.keys(), required=True)
    parser.add_argument('--datasets', nargs='+', choices=DATASETS.keys(), required=True)
    parser.add_argument('--use-gpu', action='store_true')
    parser.add_argument('--iterations', type=int, required=True)
    parser.add_argument('--params-grid', default=None, help='path to json file, each key corresponds'
                                                            ' to learner parameter e.g. max_depth, and'
                                                            ' list of values to run in experiment')
    parser.add_argument('-o', '--out-dir', default='results')
    args = parser.parse_args()

    experiment_learners = _get_all_values_from_subset(LEARNERS, args.learners)
    experiments = _get_all_values_from_subset(DATASETS, args.datasets)

    params_grid = {
        "iterations": [args.iterations]
    }

    if args.params_grid:
        with open(args.params_grid) as f:
            grid = json.load(f)
        params_grid.update(grid)

    for experiment in experiments:
        print(experiment.name)
        experiment.run(args.use_gpu, experiment_learners, params_grid, args.out_dir)
