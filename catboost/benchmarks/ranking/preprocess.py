import os
import numpy as np
import pandas as pd
from collections import Counter
from sklearn.datasets import load_svmlight_file


def sparsity(X):
    number_of_nan = np.count_nonzero(np.isnan(X))
    number_of_zeros = np.count_nonzero(np.abs(X) < 1e-6)
    return (number_of_nan + number_of_zeros) / float(X.shape[0] * X.shape[1]) * 100.


def print_dataset_statistics(X, y, queries, name):
    print('----------------------------------')
    print("Characteristics of dataset " + name)
    print("rows x columns " + str(X.shape))
    print("sparsity: " + str(sparsity(X)))
    print("y distribution")
    print(Counter(y))
    print("num samples in queries: minimum, median, maximum")
    num_queries = Counter(queries).values()
    print(np.min(num_queries), np.median(num_queries), np.max(num_queries))
    print('----------------------------------')


def process_libsvm_file(file_name):
    X, y, queries = load_svmlight_file(file_name, query_id=True)
    return X.todense(), y, queries


def dump_to_file(out_file_name, X, y, queries):
    all = np.hstack((y.reshape(-1, 1), queries.reshape(-1, 1), X))
    pd.DataFrame(all).sort_values(by=[1]).to_csv(out_file_name, sep='\t', header=False, index=False)


def mq2008(src_path, dst_path):
    """
    0 - label, 1 - qid, ...features...
    ----------------------------------
    Characteristics of dataset mq2008 train
    rows x columns (9630, 46)
    sparsity: 47.2267370987
    y distribution
    Counter({0.0: 7820, 1.0: 1223, 2.0: 587})
    num samples in queries: minimum, median, maximum
    (5, '8.0', 121)
    ----------------------------------
    ----------------------------------
    Characteristics of dataset mq2008 test
    rows x columns (2874, 46)
    sparsity: 46.1128256331
    y distribution
    Counter({0.0: 2319, 1.0: 378, 2.0: 177})
    num samples in queries: minimum, median, maximum
    (6, '14.5', 119)
    ----------------------------------
    """
    train_file = os.path.join(src_path, "train.txt")
    test_file = os.path.join(src_path, "test.txt")

    train_out_file = os.path.join(dst_path, "train.tsv")
    test_out_file = os.path.join(dst_path, "test.tsv")

    X, y, queries = process_libsvm_file(train_file)
    print_dataset_statistics(X, y, queries, "mq2008 train")
    dump_to_file(train_out_file, X, y, queries)

    X, y, queries = process_libsvm_file(test_file)
    print_dataset_statistics(X, y, queries, "mq2008 test")
    dump_to_file(test_out_file, X, y, queries)


def msrank(src_path, dst_path):
    """
    0 - label, 1 - qid, ...features...
    ----------------------------------
    Characteristics of dataset msrank train
    rows x columns (723412, 136)
    sparsity: 37.2279141802
    y distribution
    Counter({0.0: 377957, 1.0: 232569, 2.0: 95082, 3.0: 12658, 4.0: 5146})
    num samples in queries: minimum, median, maximum
    (1, '110.0', 809)
    ----------------------------------
    ----------------------------------
    Characteristics of dataset msrank test
    rows x columns (241521, 136)
    sparsity: 37.3672293019
    y distribution
    Counter({0.0: 124784, 1.0: 77896, 2.0: 32459, 3.0: 4450, 4.0: 1932})
    num samples in queries: minimum, median, maximum
    (1, '109.0', 514)
    ----------------------------------
    """

    train_file = os.path.join(src_path, "train.txt")
    test_file = os.path.join(src_path, "test.txt")

    train_out_file = os.path.join(dst_path, "train.tsv")
    test_out_file = os.path.join(dst_path, "test.tsv")

    X, y, queries = process_libsvm_file(train_file)
    print_dataset_statistics(X, y, queries, "msrank train")
    dump_to_file(train_out_file, X, y, queries)

    X, y, queries = process_libsvm_file(test_file)
    print_dataset_statistics(X, y, queries, "msrank test")
    dump_to_file(test_out_file, X, y, queries)


def yahoo(src_path, dst_path):
    """
    0 - label, 1 - qid, ...features...
    ----------------------------------
    Characteristics of dataset yahoo train
    rows x columns (473134, 699)
    sparsity: 68.1320434932
    y distribution
    Counter({1.0: 169897, 2.0: 134832, 0.0: 123294, 3.0: 36170, 4.0: 8941})
    num samples in queries: minimum, median, maximum
    (1, '19.0', 139)
    ----------------------------------
    ----------------------------------
    Characteristics of dataset yahoo test
    rows x columns (165660, 699)
    sparsity: 68.0674251017
    y distribution
    Counter({1.0: 59107, 2.0: 48033, 0.0: 42625, 3.0: 12804, 4.0: 3091})
    num samples in queries: minimum, median, maximum
    (1, '19.0', 129)
    ----------------------------------
    """

    train_file = os.path.join(src_path, 'set1.train.txt')
    test_file = os.path.join(src_path, 'set1.test.txt')

    train_out_file = os.path.join(dst_path, 'train.tsv')
    test_out_file = os.path.join(dst_path, 'test.tsv')

    X, y, queries = process_libsvm_file(train_file)
    print_dataset_statistics(X, y, queries, "yahoo train")
    dump_to_file(train_out_file, X, y, queries)

    X, y, queries = process_libsvm_file(test_file)
    print_dataset_statistics(X, y, queries, "yahoo test")
    dump_to_file(test_out_file, X, y, queries)


def yandex(src_path, dst_path):
    """
    0 - qid, 1 - label, 2 - URL, 3 - GroupId, ...features...

    ----------------------------------
    Characteristics of dataset yandex train
    rows x columns (12463, 53)
    sparsity: 54.3395620849
    y distribution
    Counter({0: 6599, 2: 3311, 1: 2303, 3: 196, 4: 56})
    num samples in queries: minimum, median, maximum
    (1, 15.0, 62)
    ----------------------------------

    ----------------------------------
    Characteristics of dataset yandex test
    rows x columns (46594, 53)
    sparsity: 54.0090188955
    y distribution
    Counter({0: 24562, 2: 12487, 1: 8658, 3: 701, 4: 188})
    num samples in queries: minimum, median, maximum
    (1, 14.0, 63)
    ----------------------------------
    """

    train_file = os.path.join(src_path, 'features.txt')
    test_file = os.path.join(src_path, 'featuresTest.txt')

    train = pd.read_csv(train_file, header=None, sep='\t')
    test = pd.read_csv(test_file, header=None, sep='\t')

    # drop url column
    train = train.drop([2], axis=1)
    test = test.drop([2], axis=1)

    # swap query id and label columns
    train[0], train[1] = train[1], train[0]
    test[0], test[1] = test[1], test[0]

    # quantize label column
    bins = [0.07, 0.14, 0.41, 0.61]
    train[0] = np.digitize(train[0], bins)
    test[0] = np.digitize(test[0], bins)

    print_dataset_statistics(train[2:], train[0], train[1], "yandex train")
    print_dataset_statistics(test[2:], test[0], test[1], "yandex test")

    train_out_file = os.path.join(dst_path, 'train.tsv')
    test_out_file = os.path.join(dst_path, 'test.tsv')

    train = train.sort_values(by=[1])
    test = test.sort_values(by=[1])

    train.to_csv(train_out_file, header=None, index=False, sep='\t')
    test.to_csv(test_out_file, header=None, index=False, sep='\t')
