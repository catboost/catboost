import sys, argparse
from experiment import Experiment
from datetime import datetime
import numpy as np
import pickle
import os

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
        BstExperiment = XGBExperiment
    elif namespace.bst == 'lgb':
    	from lightgbm_experiment import LGBExperiment
        BstExperiment = LGBExperiment
    elif namespace.bst == 'cab':
        from catboost_experiment import CABExperiment
        BstExperiment = CABExperiment

    learning_task = namespace.learning_task
    train_path = namespace.train
    test_path = namespace.test
    cd_path = namespace.cd
    n_estimators = namespace.n_estimators
    output_folder_path = os.path.join(namespace.output_folder_path, '')
    max_hyperopt_evals = namespace.hyperopt_evals

    print 'Loading and preprocessing dataset...'

    X_train, y_train, X_test, y_test, cat_cols = Experiment(learning_task, train_path=train_path,
                                                            test_path=test_path, cd_path=cd_path).read_data()

    bst_experiment = BstExperiment(learning_task, train_path=train_path,
                            test_path=test_path, cd_path=cd_path, 
                            max_hyperopt_evals=max_hyperopt_evals,
                            n_estimators=n_estimators)

    cv_pairs, (dtrain, dtest) = bst_experiment.split_and_preprocess(X_train.copy(), y_train, 
                                                                X_test.copy(), y_test, 
                                                                cat_cols, n_splits=5)
    
    default_cv_result = bst_experiment.run_cv(cv_pairs)
    bst_experiment.print_result(default_cv_result, '\nBest result on cv with the default parameters')
    print '\nTraining algorithm with the default parameters for different seed...'
    preds, test_losses = [], []
    for seed in range(5):
        default_test_result = bst_experiment.run_test(dtrain, dtest, X_test, params=default_cv_result['params'],
                                            n_estimators=default_cv_result['best_n_estimators'], seed=seed)
        preds.append(default_test_result['preds'])
        test_losses.append(default_test_result['loss'])
        print 'For seed=%d Test\'s %s : %.5f' % (seed, bst_experiment.metric, test_losses[-1])
    
    print '\nTest\'s %s mean: %.5f, Test\'s %s std: %.5f' % (bst_experiment.metric, np.mean(test_losses),
                                                             bst_experiment.metric, np.std(test_losses))

    default_test_result = {
        'losses': test_losses,
        'params': default_test_result['params'],
        'n_estimators': default_test_result['n_estimators']
    }

    default_result = {
        'cv': default_cv_result,
        'test': default_test_result,
        'preds': preds,
    }

    date = datetime.now().strftime('%Y%m%d-%H%M%S')
    dataset_name = train_path.replace('/', ' ').strip().split()[-2]
    file_name = '{}{}_results_default_{}_{}.pkl'.format(output_folder_path, namespace.bst, dataset_name, date)

    with open(file_name, 'wb') as f:
            pickle.dump(default_result, f)
