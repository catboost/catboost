import sys, argparse

def createParser():
    parser = argparse.ArgumentParser()
    parser.add_argument('bst', choices=['xgb', 'lgb', 'cab'])
    parser.add_argument('learning_task', choices=['classification', 'regression'])
    parser.add_argument('-t', '--n_estimators', type=int, default=5000)
    parser.add_argument('-n', '--hyperopt_evals', type=int, default=50)
    parser.add_argument('-s', '--time_sort', type=int, default=None)
    parser.add_argument('--train', required=True)
    parser.add_argument('--test', required=True)
    parser.add_argument('--cd', required=True)
    parser.add_argument('-o', '--output_folder_path', default=None)
    parser.add_argument('--holdout_size', type=float, default=0)
    return parser

if __name__ == "__main__":
    parser = createParser()
    namespace = parser.parse_args(sys.argv[1:])

    if namespace.bst == 'xgb':
    	from xgboost_experiment import XGBExperiment
        Experiment = XGBExperiment
    elif namespace.bst == 'lgb':
    	from lightgbm_experiment import LGBExperiment
        Experiment = LGBExperiment
    elif namespace.bst == 'cab':
        from catboost_experiment import CABExperiment
        Experiment = CABExperiment

    experiment = Experiment(namespace.learning_task, namespace.n_estimators, namespace.hyperopt_evals, 
                            namespace.time_sort, namespace.holdout_size, 
                            namespace.train, namespace.test, namespace.cd,
                            namespace.output_folder_path)
    experiment.run()
