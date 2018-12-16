# This file is modified version of datasets.py.
# datasets.py was released by RAMitchell (Copyright (c) 2018 Rory Mitchell) under MIT License
# and available at https://github.com/RAMitchell/ml_dataset_loader/blob/master/datasets.py
# License text is available at https://github.com/RAMitchell/ml_dataset_loader/blob/master/LICENSE

"""Module for loading preprocessed datasets for machine learning problems"""
import bz2
import os
import re
import gc
import sys
import tarfile

import numpy as np
import pandas as pd

from collections import namedtuple
from sklearn import datasets
from sklearn.model_selection import train_test_split

if sys.version_info[0] >= 3:
    from urllib.request import urlretrieve  # pylint: disable=import-error,no-name-in-module
else:
    from urllib import urlretrieve  # pylint: disable=import-error,no-name-in-module


DEFAULT_TEST_SIZE = 0.2


DATASET_CHARACTERISTIC = {
    "abalone": (4177, 8),
    "airline": (115000000, 13),
    "airline-one-hot": (10100000, 700),
    "bosch": (1184000, 968),
    "cover-type": (581012, 54),
    "epsilon": (500000, 2000),
    "epsilon-sampled": (500000, 28),
    "higgs": (11000000, 28),
    "higgs-sampled": (500000, 28),
    "letters": (20000, 16),
    "msrank": (1200192, 137),
    "msrank-classification": (1200192, 137),
    "synthetic": (10000000, 100),
    "synthetic-5k-features": (100000, 5000),
    "synthetic-classification": (500000, 28),
    "yahoo": (638794, 700),
    "yahoo-classification": (638794, 700),
    "year-msd": (515345, 90)
}


Data = namedtuple("Data", ["name", "X_train", "X_test", "y_train", "y_test"])


def get_from_cache(experiment_name, train_file, test_file):
    print('loading train')
    train = np.fromfile(train_file, sep='\t')

    n_features = DATASET_CHARACTERISTIC[experiment_name][1] + 1
    train = train.reshape((train.shape[0] / n_features, n_features))

    X_train = train[:, 1:]
    y_train = train[:, 0]

    print('loading test')
    test = np.fromfile(test_file, sep='\t')
    test = test.reshape((test.shape[0] / n_features, n_features))

    X_test = test[:, 1:]
    y_test = test[:, 0]

    return Data(experiment_name, X_train, X_test, y_train, y_test)


def save_to_cache(data, train_file, test_file):
    train = np.hstack([data.y_train.reshape(-1, 1), data.X_train])
    train_df = pd.DataFrame(data=train)

    train_df.to_csv(train_file, index=False, header=False, sep='\t')

    test = np.hstack([data.y_test.reshape(-1, 1), data.X_test])
    test_df = pd.DataFrame(data=test)

    test_df.to_csv(test_file, index=False, header=False, sep='\t')


def get_dataset(experiment_name, dataset_dir):
    data_loader = DATA_LOADERS[experiment_name]
    cache_dir = os.path.join(dataset_dir, experiment_name)

    train_file = os.path.join(cache_dir, "train.tsv")
    test_file = os.path.join(cache_dir, "test.tsv")

    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)

    if all([os.path.exists(file_name) for file_name in [train_file, test_file]]):
        print('Loading from cache')
        return get_from_cache(experiment_name, train_file, test_file)

    X, y = data_loader(dataset_dir)

    if experiment_name in ALREADY_SPLIT:
        X_train = X[0]
        y_train = y[0]

        X_test = X[1]
        y_test = y[1]
    else:
        X_train, X_test, y_train, y_test = \
            train_test_split(X, y, test_size=DEFAULT_TEST_SIZE, random_state=0)

    data = Data(experiment_name, X_train, X_test, y_train, y_test)
    save_to_cache(data, train_file, test_file)

    return data


ALREADY_SPLIT = {
    "airline-one-hot",
    "cover-type",
    "epsilon-sampled",
    "msrank",
    "msrank-classification",
    "epsilon"
}


def read_libsvm(file_obj, n_samples, n_features):
    X = np.zeros((n_samples, n_features))
    y = np.zeros((n_samples,))

    counter = 0

    regexp = re.compile(r'[A-Za-z0-9]+:(-?\d*\.?\d+)')

    for line in file_obj:
        line = regexp.sub('\g<1>', line)
        line = line.rstrip(" \n\r").split(' ')

        y[counter] = int(line[0])
        X[counter] = map(float, line[1:])
        if counter < 5:
            print(y)
            print(X[counter])

        counter += 1

    assert counter == n_samples

    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int)


def _make_gen(reader):
    b = reader(1024 * 1024)
    while b:
        yield b
        b = reader(1024 * 1024)


def _count_lines(filename):
    with open(filename, 'rb') as f:
        f_gen = _make_gen(f.read)
        return sum(buf.count(b'\n') for buf in f_gen)


def abalone(dataset_dir):
    """
    https://archive.ics.uci.edu/ml/machine-learning-databases/abalone

    TaskType:regression
    NumberOfFeatures:8
    NumberOfInstances:4177
    """
    url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/abalone/abalone.data'

    filename = os.path.join(dataset_dir, 'abalone.data')
    if not os.path.exists(filename):
        urlretrieve(url, filename)

    abalone = pd.read_csv(filename, header=None)
    abalone[0] = abalone[0].astype('category').cat.codes
    X = abalone.iloc[:, :-1].values
    y = abalone.iloc[:, -1].values
    return X, y


def airline(dataset_dir):
    """
    Airline dataset (http://kt.ijs.si/elena_ikonomovska/data.html)

    TaskType:binclass
    NumberOfFeatures:13
    NumberOfInstances:115M
    """
    url = 'http://kt.ijs.si/elena_ikonomovska/datasets/airline/airline_14col.data.bz2'

    filename = os.path.join(dataset_dir, 'airline_14col.data.bz2')
    if not os.path.exists(filename):
        urlretrieve(url, filename)

    cols = [
        "Year", "Month", "DayofMonth", "DayofWeek", "CRSDepTime",
        "CRSArrTime", "UniqueCarrier", "FlightNum", "ActualElapsedTime",
        "Origin", "Dest", "Distance", "Diverted", "ArrDelay"
    ]

    # load the data as int16
    dtype = np.int16

    dtype_columns = {
        "Year": dtype, "Month": dtype, "DayofMonth": dtype, "DayofWeek": dtype,
        "CRSDepTime": dtype, "CRSArrTime": dtype, "FlightNum": dtype,
        "ActualElapsedTime": dtype, "Distance": dtype,
        "Diverted": dtype, "ArrDelay": dtype,
    }

    df = pd.read_csv(filename, names=cols, dtype=dtype_columns)

    # Encode categoricals as numeric
    for col in df.select_dtypes(['object']).columns:
        df[col] = df[col].astype("category").cat.codes

    # Turn into binary classification problem
    df["ArrDelayBinary"] = 1 * (df["ArrDelay"] > 0)

    X = df[df.columns.difference(["ArrDelay", "ArrDelayBinary"])]
    y = df["ArrDelayBinary"]

    del df
    return X.values, y.values


def airline_one_hot(dataset_dir):
    """
    Dataset from szilard benchmarks: https://github.com/szilard/GBM-perf

    TaskType:binclass
    NumberOfFeatures:700
    NumberOfInstances:10100000
    """
    url = 'https://s3.amazonaws.com/benchm-ml--main/'

    name_train = 'train-10m.csv'
    name_test = 'test.csv'

    sets = []
    labels = []

    categorical_names = ["Month", "DayofMonth", "DayOfWeek", "UniqueCarrier", "Origin", "Dest"]
    categorical_ids = [0, 1, 2, 4, 5, 6]

    numeric_names = ["DepTime", "Distance"]
    numeric_ids = [3, 7]

    for name in [name_train, name_test]:
        filename = os.path.join(dataset_dir, name)
        if not os.path.exists(filename):
            urlretrieve(url + name, filename)

        df = pd.read_csv(filename)
        X = df.drop('dep_delayed_15min', 1)
        y = df["dep_delayed_15min"]

        y_num = np.where(y == "Y", 1, 0)

        sets.append(X)
        labels.append(y_num)

    n_samples_train = sets[0].shape[0]

    X = pd.concat(sets)
    X = pd.get_dummies(X, columns=categorical_names)
    sets = [X[:n_samples_train], X[n_samples_train:]]

    return sets, labels


def bosch(dataset_dir):
    """
    Bosch Production Line Performance data set (
    https://www.kaggle.com/c/bosch-production-line-performance)

    Requires Kaggle API and API token (https://github.com/Kaggle/kaggle-api)

    Contains missing values as NaN.

    TaskType:binclass
    NumberOfFeatures:968
    NumberOfInstances:1.184M
    """

    train_file = "train_numeric.csv.zip"
    if not os.path.exists(train_file):
        os.system("kaggle competitions download -c bosch-production-line-performance -f "
                  "train_numeric.csv.zip -p " + dataset_dir)

    X = pd.read_csv(os.path.join(dataset_dir, train_file), index_col=0, compression='zip', dtype=np.float32)
    y = X.iloc[:, -1]
    X.drop(X.columns[-1], axis=1, inplace=True)
    return X.values, y.values


def cover_type(dataset_dir):
    """
    Cover type dataset from UCI machine learning repository (
    https://archive.ics.uci.edu/ml/datasets/covertype).

    Train/test split was taken from:
    https://www.kaggle.com/c/forest-cover-type-prediction

    y contains 7 unique class labels from 1 to 7 inclusive.

    TaskType:multiclass
    NumberOfFeatures:54
    NumberOfInstances:581012
    """

    train_file = os.path.join(dataset_dir, "train_csv.zip")
    test_file = os.path.join(dataset_dir, "test_csv.zip")

    if not (os.path.exists(train_file) and os.path.exists(test_file)):
        os.system("kaggle competitions download -c forest-cover-type-prediction -p .")

    X_train = pd.read_csv(train_file, index_col=0, compression='zip', dtype=np.float32)
    y_train = X_train.iloc[:, -1]

    X_test = pd.read_csv(test_file, index_col=0, compression='zip', dtype=np.float32)
    y_test = X_test.iloc[:, -1]

    X_train.drop(X_train.columns[-1], axis=1, inplace=True)
    X_test.drop(X_test.columns[-1], axis=1, inplace=True)

    y_train = np.array(y_train.values, dtype=int)
    y_test = np.array(y_test.values, dtype=int)

    return (X_train.values, X_test.values), (y_train, y_test)


def epsilon(dataset_dir):
    """
    TaskType:binclass
    NumberOfFeatures:2000
    NumberOfInstances:500K
    """
    url = 'https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/'

    name_train = 'epsilon_normalized.bz2'
    name_test = 'epsilon_normalized.t.bz2'

    xs = []
    ys = []

    for name in [name_train, name_test]:
        filename = os.path.join(dataset_dir, name)
        if not os.path.exists(filename):
            print('Downloading ' + name)
            urlretrieve(url + name, filename)

        print('Processing')
        if name == name_train:
            n_samples = 400000
        else:
            n_samples = 100000

        with bz2.BZ2File(filename, 'r') as f:
            x, y = read_libsvm(f, n_samples=n_samples, n_features=2000)

            y[y <= 0] = 0
            y[y > 0] = 1
            y.astype(int)

            xs.append(x)
            ys.append(y)

    return xs, ys


def epsilon_sampled(dataset_dir):
    """
    TaskType:binclass
    NumberOfFeatures:28
    NumberOfInstances:500K
    """
    xs, ys = epsilon(dataset_dir)
    feat_ids = np.random.choice(xs[0].shape[1], 28, replace=False)
    xs[0] = xs[0][:, feat_ids]
    xs[1] = xs[1][:, feat_ids]

    return xs, ys


def higgs(dataset_dir):
    """
    Higgs dataset from UCI machine learning repository (
    https://archive.ics.uci.edu/ml/datasets/HIGGS).

    TaskType:binclass
    NumberOfFeatures:28
    NumberOfInstances:11M
    """
    url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00280/HIGGS.csv.gz'

    filename = os.path.join(dataset_dir, 'HIGGS.csv.gz')
    if not os.path.exists(filename):
        urlretrieve(url, filename)
    df = pd.read_csv(filename)
    X = df.iloc[:, 1:].values
    y = df.iloc[:, 0].values

    return X, y


def higgs_sampled(dataset_dir):
    """
    TaskType:binclass
    NumberOfFeatures:28
    NumberOfInstances:500K
    """
    X, y = higgs(dataset_dir)
    ids = np.random.choice(X.shape[0], size=500000, replace=False)
    return X[ids], y[ids]


def letters(dataset_dir):
    """
    http://archive.ics.uci.edu/ml/datasets/Letter+Recognition

    TaskType:multiclass
    NumberOfFeatures:16
    NumberOfInstances:20.000
    """
    url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/letter-recognition/letter-recognition.data'

    filename = os.path.join(dataset_dir, 'letter-recognition.data')
    if not os.path.exists(filename):
        urlretrieve(url, filename)

    letters = pd.read_csv(filename, header=None)
    X = letters.iloc[:, 1:].values
    y = letters.iloc[:, 0]
    y = y.astype('category').cat.codes
    y = y.values

    return X, y


def msrank(dataset_dir):
    """
    Microsoft learning to rank dataset

    TaskType:ranking
    NumberOfFeatures:137 (including query id)
    NumberOfInstances:1200192
    """
    url = "https://storage.mds.yandex.net/get-devtools-opensource/471749/msrank.tar.gz"

    filename = os.path.join(dataset_dir, 'msrank.tar.gz')
    if not os.path.exists(filename):
        urlretrieve(url, filename)

    dirname = os.path.join(dataset_dir, 'MSRank')
    if not os.path.exists(dirname):
        tar = tarfile.open(filename, "r:gz")
        tar.extractall(dataset_dir)
        tar.close()

    sets = []
    labels = []
    n_features = 137

    for set_name in ['train.txt', 'vali.txt', 'test.txt']:
        file_name = os.path.join(dirname, set_name)

        n_samples = _count_lines(file_name)
        with open(file_name, 'r') as file_obj:
            X, y = read_libsvm(file_obj, n_samples, n_features)

        sets.append(X)
        labels.append(y)

    sets[0] = np.vstack((sets[0], sets[1]))
    labels[0] = np.hstack((labels[0], labels[1]))
    del sets[1]
    del labels[1]

    return sets, labels


def synthetic_classification(dataset_dir):
    """
    Synthetic classification generator from sklearn

    TaskType:binclass
    NumberOfFeatures:28
    NumberOfInstances:500K
    """
    return datasets.make_classification(n_samples=500000, n_features=28, n_classes=2, random_state=0)


def synthetic_regression(dataset_dir):
    """
    Synthetic regression generator from sklearn
    http://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_regression.html

    TaskType:regression
    NumberOfFeatures:100
    NumberOfInstances:10M
    """
    return datasets.make_regression(n_samples=10000000, bias=100, noise=1.0, random_state=0)


def synthetic_regression_5k_features(dataset_dir):
    """
    Synthetic regression generator from sklearn
    http://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_regression.html

    TaskType:regression
    NumberOfFeatures:5000
    NumberOfInstances:100K
    """
    return datasets.make_regression(n_samples=100000, n_features=5000, bias=100, noise=1.0, random_state=0)


def yahoo(dataset_dir):
    train_file = os.path.join(dataset_dir, 'train.tsv')
    test_file = os.path.join(dataset_dir, 'test.tsv')

    if not (os.path.exists(train_file) or os.path.exists(test_file)):
        raise Exception('Please download dataset from '
                        'https://webscope.sandbox.yahoo.com/catalog.php?datatype=c'
                        ' and convert it to tsv file, 0 - Label, 1 - QueryId, 2 - ...features...')

    train = pd.read_csv(train_file, sep='\t', header=None)
    test = pd.read_csv(test_file, sep='\t', header=None)


def year(dataset_dir):
    """
    YearPredictionMSD dataset from UCI repository (
    https://archive.ics.uci.edu/ml/datasets/yearpredictionmsd)

    TaskType:regression
    NumberOfFeatures:90
    NumberOfInstances:515345
    """
    url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00203/YearPredictionMSD.txt.zip'

    filename = os.path.join(dataset_dir, 'YearPredictionMSD.txt.zip')
    if not os.path.exists(filename):
        urlretrieve(url, filename)

    df = pd.read_csv(filename, header=None)
    X = df.iloc[:, 1:].values
    y = df.iloc[:, 0].values
    return X, y


DATA_LOADERS = {
    "abalone": abalone,
    "airline": airline,
    "airline-one-hot": airline_one_hot,
    "bosch": bosch,
    "cover-type": cover_type,
    "epsilon": epsilon,
    "epsilon-sampled": epsilon_sampled,
    "higgs": higgs,
    "higgs-sampled": higgs_sampled,
    "letters": letters,
    "msrank": msrank,
    "msrank-classification": msrank,
    "synthetic": synthetic_regression,
    "synthetic-5k-features": synthetic_regression_5k_features,
    "synthetic-classification": synthetic_classification,
    "yahoo": yahoo,
    "yahoo-classification": yahoo,
    "year-msd": year
}
