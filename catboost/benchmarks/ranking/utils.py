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


def doc_comparator(doc1, doc2):
    if doc1[1] < doc2[1]:
        return 1
    elif doc1[1] == doc2[1]:
        return int(doc1[0] > doc2[0])
    else:
        return -1


def cumulative_gain(relevances):
    return np.sum((2 ** relevances - 1) / np.log2(np.arange(relevances.shape[0]) + 2))


def ndcg(y_pred, y_true, top):
    assert y_pred.shape[0] == y_true.shape[0]
    top = min(top, y_pred.shape[0])

    first_k_docs = sorted(zip(y_true, y_pred), key=cmp_to_key(doc_comparator))
    first_k_docs = np.array(first_k_docs)[:top,0]

    top_k_idxs = np.argsort(y_true)[::-1][:top]
    top_k_docs = y_true[top_k_idxs]

    dcg = cumulative_gain(first_k_docs)
    idcg = cumulative_gain(top_k_docs)

    return dcg / idcg if idcg > 0 else 1.


def mean_ndcg(y_pred, y_true, query_idxs, top=10):
    sum_ndcg = 0
    queries = np.unique(query_idxs)

    for query in queries:
        idxs = query_idxs == query
        value = ndcg(y_pred[idxs], y_true[idxs], top)
        sum_ndcg += value

    return sum_ndcg / float(queries.shape[0])


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
