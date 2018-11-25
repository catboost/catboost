import argparse
from experiments import EXPERIMENTS
from learners import *
from generate_report import get_experiment_stats, print_all_in_one_table


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--use-gpu', action='store_true')
    parser.add_argument('--datasets', default='datasets')
    parser.add_argument('--iterations', default=1000, type=int)
    parser.add_argument('--result', default='result.json')
    parser.add_argument('--table', default='common-table.txt')
    args = parser.parse_args()

    experiments_names = [
        'abalone',
        'airline',
        'epsilon',
        'higgs',
        'letters',
        'msrank',
        'msrank-classification',
        'synthetic',
        'synthetic-5k-features'
    ]

    learners = [
        XGBoostLearner,
        LightGBMLearner,
        CatBoostLearner
    ]

    iterations = args.iterations
    logs_dir = 'logs'

    params_grid = {
        'iterations': [iterations],
        'max_depth': [6],
        'learning_rate': [0.03, 0.07, 0.15]
    }

    for experiment_name in experiments_names:
        print(experiment_name)
        experiment = EXPERIMENTS[experiment_name]
        experiment.run(args.use_gpu, learners, params_grid, args.datasets, args.result, logs_dir)

    stats = get_experiment_stats(args.result, args.use_gpu, niter=iterations)
    print_all_in_one_table(stats, args.use_gpu, params=(6.0, 1.0), output=args.table)


if __name__ == "__main__":
    main()
