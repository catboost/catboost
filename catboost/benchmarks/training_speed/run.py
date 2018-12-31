import argparse

from data_loader import DATASET_CHARACTERISTIC
from experiments import *
from learners import *

LEARNERS = {
    "xgb": XGBoostLearner,
    "lgb": LightGBMLearner,
    "cat": CatBoostLearner
}


def default_num_iterations(experiment_name):
    num_samples = DATASET_CHARACTERISTIC[experiment_name][0]
    if num_samples > 10e6:
        return 8000
    elif num_samples > 50e3:
        return 5000
    else:
        return 2000


def _get_all_values_from_subset(items, subset):
    filtered_keys = filter(lambda x: x in subset, items.keys())
    return [items[key] for key in filtered_keys]


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--learners', nargs='+', choices=LEARNERS.keys(), required=True)
    parser.add_argument('--experiment', choices=EXPERIMENTS.keys(), required=True)
    parser.add_argument('--use-gpu', action='store_true')
    parser.add_argument('--iterations', type=int)
    parser.add_argument('--params-grid', default=None, help='path to json file, each key corresponds'
                                                            ' to learner parameter e.g. max_depth, and'
                                                            ' list of values to run in experiment')
    parser.add_argument('--dataset-dir', default='datasets')
    parser.add_argument('-l', '--log-dir', default='logs')
    parser.add_argument('-o', '--result', default='result.json')
    args = parser.parse_args()

    experiment_learners = _get_all_values_from_subset(LEARNERS, args.learners)
    experiment = EXPERIMENTS[args.experiment]

    iterations = default_num_iterations(args.experiment)
    if args.iterations is not None:
        iterations = args.iterations

    params_grid = {
        "iterations": [iterations]
    }

    if args.params_grid:
        with open(args.params_grid) as f:
            grid = json.load(f)
        params_grid.update(grid)

    print(experiment.name)
    experiment.run(args.use_gpu, experiment_learners, params_grid, args.dataset_dir, args.result, args.log_dir)
