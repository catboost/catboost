"""Module for loading preprocessed datasets for machine learning problems"""
import bz2
import os
import re
import sys
import tarfile

import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.externals.joblib.memory import Memory
from tqdm import tqdm

if sys.version_info[0] >= 3:
    from urllib.request import urlretrieve  # pylint: disable=import-error,no-name-in-module
else:
    from urllib import urlretrieve  # pylint: disable=import-error,no-name-in-module

mem = Memory("./cached_datasets")

get_airline_url = 'http://kt.ijs.si/elena_ikonomovska/datasets/airline/airline_14col.data.bz2'


@mem.cache
def get_airline(num_rows=None):
    """
    Airline dataset (http://kt.ijs.si/elena_ikonomovska/data.html)

    Has categorical columns converted to ordinal and target variable "Arrival Delay" converted
    to binary target.

    - Dimensions: 115M rows, 13 columns.
    - Task: Binary classification

    :param num_rows:
    :return: X, y
    """

    filename = 'airline_14col.data.bz2'
    if not os.path.isfile(filename):
        urlretrieve(get_airline_url, filename)

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

    df = pd.read_csv(filename,
                     names=cols, dtype=dtype_columns, nrows=num_rows)

    # Encode categoricals as numeric
    for col in df.select_dtypes(['object']).columns:
        df[col] = df[col].astype("category").cat.codes

    # Turn into binary classification problem
    df["ArrDelayBinary"] = 1 * (df["ArrDelay"] > 0)

    X = df[df.columns.difference(["ArrDelay", "ArrDelayBinary"])]
    y = df["ArrDelayBinary"]

    del df
    return X.values, y.values


get_higgs_url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00280/HIGGS.csv.gz'


@mem.cache
def get_higgs(num_rows=None):
    """
    Higgs dataset from UCI machine learning repository (
    https://archive.ics.uci.edu/ml/datasets/HIGGS).

    - Dimensions: 11M rows, 28 columns.
    - Task: Binary classification

    :param num_rows:
    :return: X, y
    """
    filename = 'HIGGS.csv.gz'
    if not os.path.exists(filename):
        urlretrieve(get_higgs_url, filename)
    higgs = pd.read_csv(filename, nrows=num_rows)
    X = higgs.iloc[:, 1:].values
    y = higgs.iloc[:, 0].values

    return X, y


@mem.cache
def get_higgs_sampled():
    X, y = get_higgs()

    ids = np.random.choice(X.shape[0], size=500000, replace=False)

    return X[ids], y[ids]


get_epsilon_url = 'https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/'


def read_libsvm(file_obj, n_samples, n_features):
    X = np.zeros((n_samples, n_features))
    y = np.zeros((n_samples,))

    counter = 0

    regexp = re.compile(r'[A-Za-z0-9]+:(-?\d*\.?\d+)')

    for line in tqdm(file_obj):
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


@mem.cache
def get_epsilon():
    filename_train = 'epsilon_normalized.bz2'
    filename_test = 'epsilon_normalized.t.bz2'

    for filename in [filename_train, filename_test]:
        if not os.path.exists(filename):
            print('Downloading ' + filename)
            urlretrieve(get_epsilon_url + filename, filename)
            print('done')

    print('Processing')

    with bz2.BZ2File(filename_train, 'r') as f_train:
        X_train, y_train = read_libsvm(f_train, n_samples=400000, n_features=2000)

    with bz2.BZ2File(filename_test, 'r') as f_test:
        X_test, y_test = read_libsvm(f_test, n_samples=100000, n_features=2000)

    X_train = np.vstack((X_train, X_test))
    y_train = np.hstack((y_train, y_test))

    y_train[y_train <= 0] = 0
    y_train[y_train > 0] = 1
    y_train.astype(int)

    return X_train, y_train


@mem.cache
def get_epsilon_sampled():
    X, y = get_epsilon()

    feat_ids = np.random.choice(X.shape[1], 28, replace=False)

    return X[:, feat_ids], y


@mem.cache
def get_cover_type():
    """
    Cover type dataset from UCI machine learning repository (
    https://archive.ics.uci.edu/ml/datasets/covertype).

    Train/test split was taken from:
    https://www.kaggle.com/c/forest-cover-type-prediction

    y contains 7 unique class labels from 1 to 7 inclusive.

    - Dimensions: 581012 rows, 54 columns.
    - Task: Multiclass classification

    :return: X, y
    """

    os.system("kaggle competitions download -c forest-cover-type-prediction -p .")
    X_train = pd.read_csv("train.csv.zip", index_col=0, compression='zip', dtype=np.float32)
    y_train = X_train.iloc[:, -1]

    X_test = pd.read_csv("test.csv.zip", index_col=0, compression='zip', dtype=np.float32)
    y_test = X_test.iloc[:, -1]

    X_train.drop(X_train.columns[-1], axis=1, inplace=True)
    X_test.drop(X_test.columns[-1], axis=1, inplace=True)

    y_train = np.array(y_train.values, dtype=int)
    y_test = np.array(y_test.values, dtype=int)

    return (X_train.values, X_test.values), (y_train, y_test)


@mem.cache
def get_synthetic_regression():
    """
    Synthetic regression generator from sklearn (
    http://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_regression.html).

    - Dimensions: 10000000 rows, 100 columns.
    - Task: Regression

    :return: X, y
    """
    return datasets.make_regression(n_samples=10000000, bias=100, noise=1.0, random_state=0)


@mem.cache
def get_synthetic_regression_5k_features():
    """
    Synthetic regression generator from sklearn (
    http://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_regression.html).

    - Dimensions: 100000 rows, 5000 columns.
    - Task: Regression

    :return: X, y
    """
    return datasets.make_regression(n_samples=100000, n_features=5000, bias=100, noise=1.0, random_state=0)


@mem.cache
def get_synthetic_classification(num_rows=None):
    if num_rows is None:
        num_rows = 500000
    return datasets.make_classification(n_samples=num_rows, n_features=28, n_classes=2, random_state=0)


get_year_url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00203/YearPredictionMSD.txt.zip'


@mem.cache
def get_year(num_rows=None):
    """
    YearPredictionMSD dataset from UCI repository (
    https://archive.ics.uci.edu/ml/datasets/yearpredictionmsd)

    - Dimensions: 515345 rows, 90 columns.
    - Task: Regression

    :param num_rows:
    :return: X,y
    """
    filename = 'YearPredictionMSD.txt.zip'
    if not os.path.isfile(filename):
        urlretrieve(get_year_url, filename)

    year = pd.read_csv(filename, header=None, nrows=num_rows)
    X = year.iloc[:, 1:].values
    y = year.iloc[:, 0].values
    return X, y


get_abalone_url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/abalone/abalone.data'


@mem.cache
def get_abalone():
    """
    https://archive.ics.uci.edu/ml/machine-learning-databases/abalone
    :return: X,y
    """

    filename = 'abalone.data'
    if not os.path.exists(filename):
        urlretrieve(get_abalone_url, filename)

    abalone = pd.read_csv(filename, header=None)
    abalone[0] = abalone[0].astype('category').cat.codes
    X = abalone.iloc[:, :-1].values
    y = abalone.iloc[:, -1].values
    return X, y


get_letters_url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/letter-recognition/letter-recognition.data'


@mem.cache
def get_letters():
    """
    http://archive.ics.uci.edu/ml/datasets/Letter+Recognition
    :return: X,y
    """

    filename = 'letter-recognition.data'
    if not os.path.exists(filename):
        urlretrieve(get_letters_url, filename)

    letters = pd.read_csv(filename, header=None)
    X = letters.iloc[:, 1:].values
    y = letters.iloc[:, 0]
    y = y.astype('category').cat.codes
    y = y.values

    return X, y


get_msrank_url = "https://storage.mds.yandex.net/get-devtools-opensource/471749/msrank.tar.gz"


def _make_gen(reader):
    b = reader(1024 * 1024)
    while b:
        yield b
        b = reader(1024*1024)


def count_lines(filename):
    with open(filename, 'rb') as f:
        f_gen = _make_gen(f.read)
        return sum(buf.count(b'\n') for buf in f_gen)


@mem.cache
def get_msrank():
    """
    Microsoft learning to rank dataset

    1200192 total samples
    137 features (including query id)

    :return X,y
    """

    filename = 'msrank.tar.gz'
    if not os.path.exists(filename):
        urlretrieve(get_msrank_url, filename)

    dirname = 'MSRank'
    if not os.path.exists(dirname):
        tar = tarfile.open(filename, "r:gz")
        tar.extractall()
        tar.close()

    sets = []
    labels = []
    n_features = 137

    for set_name in ['train.txt', 'vali.txt', 'test.txt']:
        file_name = os.path.join(dirname, set_name)

        n_samples = count_lines(file_name)
        with open(file_name, 'r') as file_obj:
            X, y = read_libsvm(file_obj, n_samples, n_features)

        sets.append(X)
        labels.append(y)

    return sets, labels


get_url_url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/url/url_svmlight.tar.gz'


@mem.cache
def get_url(num_rows=None):
    """
    URL reputation dataset from UCI repository (
    https://archive.ics.uci.edu/ml/datasets/URL+Reputation)

    Extremely sparse classification dataset. X is returned as a scipy sparse matrix.

    - Dimensions: 2396130 rows, 3231961 columns.
    - Task: Classification

    :param num_rows:
    :return: X,y
    """
    from scipy.sparse import vstack
    filename = 'url_svmlight.tar.gz'
    if not os.path.isfile(filename):
        urlretrieve(get_url_url, filename)
        tar = tarfile.open(filename, "r:gz")
        tar.extractall()
        tar.close()

    num_files = 120
    files = ['url_svmlight/Day{}.svm'.format(day) for day in range(num_files)]
    data = datasets.load_svmlight_files(files)
    X = vstack(data[::2])
    y = np.concatenate(data[1::2])

    y[y < 0.0] = 0.0

    if num_rows is not None:
        X = X[0:num_rows]
        y = y[0:num_rows]

    return X, y


@mem.cache
def get_bosch(num_rows=None):
    """
    Bosch Production Line Performance data set (
    https://www.kaggle.com/c/bosch-production-line-performance)

    Requires Kaggle API and API token (https://github.com/Kaggle/kaggle-api)

    Contains missing values as NaN.

    - Dimensions: 1.184M rows, 968 columns
    - Task: Binary classification

    :param num_rows:
    :return: X,y
    """
    os.system("kaggle competitions download -c bosch-production-line-performance -f "
              "train_numeric.csv.zip -p .")
    X = pd.read_csv("train_numeric.csv.zip", index_col=0, compression='zip', dtype=np.float32,
                    nrows=num_rows)
    y = X.iloc[:, -1]
    X.drop(X.columns[-1], axis=1, inplace=True)
    return X.values, y.values
