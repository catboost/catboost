import numpy as np
import pandas as pd
from functools import cmp_to_key
from matplotlib import pyplot as plt


def read_dataset(file_name):
    df = pd.read_csv(file_name, sep='\t', header=None)

    y = df[0].values
    queries = df[1].values
    X = df.iloc[:, 2:].values

    # assert np.all(queries == np.sort(queries))

    return X, y, queries


def plot_validate_curves(num_iterations, cb_log, xgb_log, lgb_log):
    plt.figure(figsize=(13, 8))
    plt.title('NDCG(iteration)', fontdict={'fontsize': 20})
    x_values = np.array(range(0, num_iterations, 10))
    cb_line, = plt.plot(x_values, np.mean(cb_log, axis=0))
    xgb_line, = plt.plot(x_values, np.mean(xgb_log, axis=0))
    lgb_line, = plt.plot(x_values, np.mean(lgb_log, axis=0))
    plt.grid(True)
    plt.legend([cb_line, xgb_line, lgb_line], ['CatBoost', 'XGBoost', 'LightGBM'], fontsize='x-large')
    plt.savefig('eval_results.png')
