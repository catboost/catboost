import hashlib
import logging
import numpy as np
import os
import pandas as pd
import tarfile
import tempfile
import six
import shutil

from .core import PATH_TYPES, fspath


logger = logging.getLogger(__name__)


def _extract(src_file, dst_dir='.'):
    cur_dir = os.getcwd()
    os.chdir(dst_dir)
    try:
        with tarfile.open(src_file, 'r:gz') as f:
            f.extractall()
    finally:
        os.chdir(cur_dir)


def _calc_md5(path):
    hasher = hashlib.md5()
    with open(path, 'rb') as f:
        while True:
            block = f.read(65536)
            if not block:
                break
            hasher.update(block)
    return hasher.hexdigest()


def _ensure_dir_exists(path):
    try:
        os.makedirs(path)
    except OSError:
        if not os.path.isdir(path):
            raise


def _cached_download(url, md5, dst):
    if os.path.isfile(dst) and _calc_md5(dst) == md5:
        return

    def reporthook(blocknum, bs, size):
        logger.debug('downloaded %s bytes', size)

    urls = url if isinstance(url, list) or isinstance(url, tuple) else (url, )

    for u in urls:
        try:
            six.moves.urllib.request.urlretrieve(u, dst, reporthook=reporthook)
            break
        except (six.moves.urllib.error.URLError, IOError):
            logger.debug('failed to download from %s', u)
    else:
        raise RuntimeError('failed to download from %s', urls)

    dst_md5 = _calc_md5(dst)
    if dst_md5 != md5:
        raise RuntimeError('md5 sum mismatch for url {url}; expected {expected}, but got {got}'.format(
            url=u, expected=md5, got=dst_md5))


_cache_path = None


def _get_cache_path():
    global _cache_path
    if _cache_path is None:
         _cache_path = os.path.join(os.getcwd(), 'catboost_cached_datasets')
    return _cache_path


def set_cache_path(path):
    assert isinstance(path, PATH_TYPES), 'expected string or pathlib.Path'
    global _cache_path
    _cache_path = fspath(path)


def _download_dataset(url, md5, dataset_name, train_file, test_file, cache=False):
    # TODO(yazevnul): this is not thread safe (or process safe?), we should take a file lock when
    # enter this function to avoid dataset being overwritten or corrupted or something else that may
    # have happen when OS operated simultaneously on the same file. Same thing should probably be
    # done with `_cached_download`.
    dir_path = os.path.join(_get_cache_path(), dataset_name) if cache else tempfile.mkdtemp()
    train_path = os.path.join(dir_path, train_file)
    test_path = os.path.join(dir_path, test_file)
    if not (os.path.exists(train_path) and os.path.exists(test_path)):
        _ensure_dir_exists(dir_path)
        file_descriptor, file_path = tempfile.mkstemp()
        os.close(file_descriptor)
        try:
            _cached_download(url, md5, file_path)
            _extract(file_path, dir_path)
        finally:
            os.remove(file_path)
    # move files for safe delete of temp dir
    if not cache:
        new_train_path = tempfile.mktemp()
        new_test_path = tempfile.mktemp()
        os.rename(train_path, new_train_path)
        os.rename(test_path, new_test_path)
        shutil.rmtree(dir_path)
        train_path, test_path = new_train_path, new_test_path
    return train_path, test_path


def _load_dataset_pd(url, md5, dataset_name, train_file, test_file, sep=',', header='infer', cache=False):
    train_path, test_path = _download_dataset(url, md5, dataset_name, train_file, test_file, cache)
    train, test = pd.read_csv(train_path, header=header, sep=sep), pd.read_csv(test_path, header=header, sep=sep)
    if not cache:
        os.remove(train_path)
        os.remove(test_path)
    return train, test


def _load_numeric_only_dataset(path, row_count, column_count, sep='\t'):
    # - can't use `pandas.read_csv` because it may result in 5x overhead
    # - can't use `numpy.loadtxt` because it may result in 3x overhead
    # And both mentioned above solutions are very slow compared to the one implemented below.
    dataset = np.zeros((row_count, column_count, ), dtype=np.float32, order='F')
    with open(path, 'rb') as f:
        for line_idx, line in enumerate(f):
            # `str.split()` is too slow, use `numpy.fromstring()`
            row = np.fromstring(line, dtype=np.float32, sep=sep)
            assert row.size == column_count, 'got too many columns at line %d (expected %d columns, got %d)' % (line_idx + 1, column_count, row.size)
            # doing `dataset[line_idx][:]` instead of `dataset[line_idx]` is here on purpose,
            # otherwise we may reallocate memory, while here we just copy
            dataset[line_idx][:] = row

    assert line_idx + 1 == row_count, 'got too many lines (expected %d lines, got %d)' % (row_count, line_idx + 1)

    return pd.DataFrame(dataset)


def titanic():
    url = 'https://storage.mds.yandex.net/get-devtools-opensource/233854/titanic.tar.gz'
    md5 = '9c8bc61d545c6af244a1d37494df3fc3'
    dataset_name, train_file, test_file = 'titanic', 'train.csv', 'test.csv'
    return _load_dataset_pd(url, md5, dataset_name, train_file, test_file)


def amazon():
    url = 'https://storage.mds.yandex.net/get-devtools-opensource/250854/amazon.tar.gz'
    md5 = '8fe3eec12bfd9c4c532b24a181d0aa2c'
    dataset_name, train_file, test_file = 'amazon', 'train.csv', 'test.csv'
    return _load_dataset_pd(url, md5, dataset_name, train_file, test_file)


def msrank():
    url = 'https://storage.mds.yandex.net/get-devtools-opensource/233854/msrank.tar.gz'
    md5 = '34fee225d02419adc106581f4eb36f2e'
    dataset_name, train_file, test_file = 'msrank', 'train.tsv', 'test.tsv'
    return _load_dataset_pd(url, md5, dataset_name, train_file, test_file, header=None, sep='\t', cache=True)


def msrank_10k():
    url = 'https://storage.mds.yandex.net/get-devtools-opensource/250854/msrank_10k.tar.gz'
    md5 = '79c5b67397289c4c8b367c1f34629eae'
    dataset_name, train_file, test_file = 'msrank_10k', 'train.csv', 'test.csv'
    return _load_dataset_pd(url, md5, dataset_name, train_file, test_file, header=None)


def rotten_tomatoes():
    """
    Contains information from kaggle [1], which is made available here under the Open Database License (ODbL) [2].

    Download "rotten_tomatoes" [1] data set.

    Will return two pandas.DataFrame-s, first with train part (rotten_tomatoes.data) and second with test part
    (rotten_tomatoes.test) of the dataset.

    NOTE: This is a preprocessed version of the dataset.

    [1]: https://www.kaggle.com/rpnuser8182/rotten-tomatoes
    [2]: https://opendatacommons.org/licenses/odbl/1-0/index.html
    """
    url = 'https://catboost-opensource.s3.yandex.net/rotten_tomatoes.tar.gz'
    md5 = 'a07fed612805ac9e17ced0d82a96add4'
    dataset_name, train_file, test_file = 'rotten_tomatoes', 'learn.tsv', 'test.tsv'
    return _load_dataset_pd(url, md5, dataset_name, train_file, test_file, sep='\t')


def imdb():
    url = 'https://catboost-opensource.s3.yandex.net/imdb.tar.gz'
    md5 = '0fd62578d631ac3d71a71c3e6ced6f8b'
    dataset_name, train_file, test_file = 'imdb', 'learn.tsv', 'test.tsv'
    return _load_dataset_pd(url, md5, dataset_name, train_file, test_file, sep='\t')


def epsilon():
    """
    Download "epsilon" [1] data set.

    Will return two pandas.DataFrame-s, first with train part (epsilon_normalized) and second with
    test part (epsilon_normalized.t) of the dataset. Object class will be located in the first
    column of dataset.

    NOTE: This is a preprocessed version of the dataset. It was converted from libsvm format into
    tsv (CatBoost doesn't support libsvm format out of the box).

    [1]: https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary.html#epsilon
    """
    urls = (
        'https://proxy.sandbox.yandex-team.ru/785711439',
        'https://storage.mds.yandex.net/get-devtools-opensource/250854/epsilon.tar.gz', )
    md5 = '5bbfac403ac673da7d7ee84bd532e973'
    dataset_name, train_file, test_file = 'epsilon', 'train.tsv', 'test.tsv'
    train_path, test_path = _download_dataset(urls, md5, dataset_name, train_file, test_file, cache=True)
    return (
        _load_numeric_only_dataset(train_path, 400000, 2001, sep='\t'),
        _load_numeric_only_dataset(test_path, 100000, 2001, sep='\t'))


def monotonic1():
    """
    Dataset with monotonic constraints.
    Can be used for poisson regression.
    Has several numerical and several categorical features.
    The first column contains target values. Columns with names Cat* contain categorical features.
    Columns with names Num* contain numerical features.

    Dataset also contains several numerical features, for which monotonic constraints must hold.
    For features in columns named MonotonicNeg*, if feature value decreases, then prediction value must not decrease.
    Thus, if there are two samples x1, x2 with all features being equal except
    for a monotonic negative feature M, such that x1[M] > x2[M], then the following inequality must
    hold for predictions: f(x1) <= f(x2)
    """
    url = 'https://storage.mds.yandex.net/get-devtools-opensource/479623/monotonic1.tar.gz'
    md5 = '1b9d8e15bc3fd6f1498e652e7fc4f4ca'
    dataset_name, train_file, test_file = 'monotonic1', 'train.tsv', 'test.tsv'
    return _load_dataset_pd(url, md5, dataset_name, train_file, test_file, sep='\t', cache=True)


def monotonic2():
    """
    Dataset with monotonic constraints.
    Can be used for regression.
    The first column contains target values.
    Other columns contain contain numerical features, for which monotonic constraints must hold.

    For features in columns named MonotonicNeg*, if feature value decreases, then prediction
    value must not decrease. Thus, if there are two samples x1, x2 with all features being
    equal except for a monotonic negative feature MNeg, such that x1[MNeg] > x2[MNeg], then
    the following inequality must hold for predictions: f(x1) <= f(x2)
    For features in columns named MonotonicPos*, if feature value decreases, then prediction
    value must not increase. Thus, if there are two samples x1, x2 with all features being
    equal except for a monotonic positive feature MPos, such that x1[MPos] > x2[MPos],
    then the following inequality must hold for predictions: f(x1) >= f(x2)
    """
    url = 'https://storage.mds.yandex.net/get-devtools-opensource/250854/monotonic2.tar.gz'
    md5 = 'ce559e212cb72c156269f6f9a641baca'
    dataset_name, train_file, test_file = 'monotonic2', 'train.tsv', 'test.tsv'
    return _load_dataset_pd(url, md5, dataset_name, train_file, test_file, sep='\t')


def adult():
    """
    Download "Adult Data Set" [1] from UCI Machine Learning Repository.

    Will return two pandas.DataFrame-s, first with train part (adult.data) and second with test part
    (adult.test) of the dataset.

    [1]: https://archive.ics.uci.edu/ml/datasets/Adult
    """
    # via https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.names
    names = (
        'age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status', 'occupation',
        'relationship', 'race', 'sex', 'capital-gain', 'capital-loss', 'hours-per-week',
        'native-country', 'income', )
    dtype = {
        'age': float, 'workclass': object, 'fnlwgt': float, 'education': object,
        'education-num': float, 'marital-status': object, 'occupation': object,
        'relationship': object, 'race': object, 'sex': object, 'capital-gain': float,
        'capital-loss': float, 'hours-per-week': float,
        'native-country': object, 'income': object, }

    # proxy.sandbox.yandex-team.ru is Yandex internal storage, we first try to download it from
    # internal storage to avoid putting too much pressure on UCI storage from our internal CI

    train_urls = (
        'https://proxy.sandbox.yandex-team.ru/779118052',
        'https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data', )
    train_md5 = '5d7c39d7b8804f071cdd1f2a7c460872'
    train_path = tempfile.mktemp()
    _cached_download(train_urls, train_md5, train_path)

    test_urls = (
        'https://proxy.sandbox.yandex-team.ru/779120000',
        'https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test', )
    test_md5 = '35238206dfdf7f1fe215bbb874adecdc'
    test_path = tempfile.mktemp()
    _cached_download(test_urls, test_md5, test_path)

    train_df = pd.read_csv(train_path, names=names, header=None, sep=',\s*', na_values=['?'], engine='python')
    os.remove(train_path)

    # lines in test part end with dot, thus we need to fix last column of the dataset
    test_df = pd.read_csv(test_path, names=names, header=None, sep=',\s*', na_values=['?'], skiprows=1, converters={'income': lambda x: x[:-1]}, engine='python')
    os.remove(test_path)

    # pandas 0.19.1 doesn't support `dtype` parameter for `read_csv` when `python` engine is used,
    # so we have to do the casting manually; also we can't use `converters` together with `dtype`
    train_df = train_df.astype(dtype)
    test_df = test_df.astype(dtype)

    return train_df, test_df


def higgs():
    """
    Download "higgs" [1] data set.

    Will return two pandas.DataFrame-s, first with train part and second with
    test part of the dataset. Object class will be located in the first
    column of dataset.

    [1]: https://archive.ics.uci.edu/ml/datasets/HIGGS
    """
    url = 'https://storage.mds.yandex.net/get-devtools-opensource/250854/higgs.tar.gz'
    md5 = 'ad59ba8328a9afa3837d7bf1a0e10e7b'
    dataset_name, train_file, test_file = 'higgs', 'train.tsv', 'test.tsv'
    train_path, test_path = _download_dataset(url, md5, dataset_name, train_file, test_file, cache=True)
    return (
        _load_numeric_only_dataset(train_path, 10500000, 29, sep='\t'),
        _load_numeric_only_dataset(test_path, 500000, 29, sep='\t'))
