import hashlib
import logging
import numpy as np
import os
import pandas as pd
import tarfile
import tempfile
import urllib

try:
    from urllib.request import urlretrieve
except ImportError:
    from urllib import urlretrieve


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
            urlretrieve(u, dst, reporthook=reporthook)
            break
        except (urllib.URLError, IOError):
            logger.debug('failed to download from %s', u)
    else:
        raise RuntimeError('failed to download from %s', urls)

    dst_md5 = _calc_md5(dst)
    if dst_md5 != md5:
        raise RuntimeError('md5 sum mismatch; expected {expected}, but got {got}'.format(
            expected=md5, got=dst_md5))


def _get_cache_path():
    return os.path.join(os.getcwd(), 'catboost_cached_datasets')


def _cached_dataset_load(url, md5, dataset_name, train_file, test_file, sep=',', header='infer'):
    dir_path = os.path.join(_get_cache_path(), dataset_name)
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
    return pd.read_csv(train_path, header=header, sep=sep), pd.read_csv(test_path, header=header, sep=sep)


def titanic():
    url = 'https://storage.mds.yandex.net/get-devtools-opensource/233854/titanic.tar.gz'
    md5 = '9c8bc61d545c6af244a1d37494df3fc3'
    dataset_name, train_file, test_file = 'titanic', 'train.csv', 'test.csv'
    return _cached_dataset_load(url, md5, dataset_name, train_file, test_file)


def amazon():
    url = 'https://storage.mds.yandex.net/get-devtools-opensource/250854/amazon.tar.gz'
    md5 = '8fe3eec12bfd9c4c532b24a181d0aa2c'
    dataset_name, train_file, test_file = 'amazon', 'train.csv', 'test.csv'
    return _cached_dataset_load(url, md5, dataset_name, train_file, test_file)


def msrank():
    url = 'https://storage.mds.yandex.net/get-devtools-opensource/250854/msrank_10k.tar.gz'
    md5 = '79c5b67397289c4c8b367c1f34629eae'
    dataset_name, train_file, test_file = 'msrank', 'train.csv', 'test.csv'
    return _cached_dataset_load(url, md5, dataset_name, train_file, test_file, header=None)


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
    return _cached_dataset_load(urls, md5, dataset_name, train_file, test_file, sep='\t', header=None)


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
        'age': np.float, 'workclass': np.object, 'fnlwgt': np.float, 'education': np.object,
        'education-num': np.float, 'marital-status': np.object, 'occupation': np.object,
        'relationship': np.object, 'race': np.object, 'sex': np.object, 'capital-gain': np.float,
        'capital-loss': np.float, 'hours-per-week': np.float,
        'native-country': np.object, 'income': np.object, }

    dir_path = os.path.join(_get_cache_path(), 'adult')
    _ensure_dir_exists(dir_path)

    # proxy.sandbox.yandex-team.ru is Yandex internal storage, we first try to download it from
    # internal storage to avoid putting too much pressure on UCI storage from our internal CI

    train_urls = (
        'https://proxy.sandbox.yandex-team.ru/779118052',
        'https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data', )
    train_md5 = '5d7c39d7b8804f071cdd1f2a7c460872'
    train_path = os.path.join(dir_path, 'train.csv')
    _cached_download(train_urls, train_md5, train_path)

    test_urls = (
        'https://proxy.sandbox.yandex-team.ru/779120000',
        'https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test', )
    test_md5 = '35238206dfdf7f1fe215bbb874adecdc'
    test_path = os.path.join(dir_path, 'test.csv')
    _cached_download(test_urls, test_md5, test_path)

    train_df = pd.read_csv(train_path, names=names, header=None, sep=',\s*', na_values=['?'], engine='python')

    # lines in test part end with dot, thus we need to fix last column of the dataset
    test_df = pd.read_csv(test_path, names=names, header=None, sep=',\s*', na_values=['?'], skiprows=1, converters={'income': lambda x: x[:-1]}, engine='python')

    # pandas 0.19.1 doesn't support `dtype` parameter for `read_csv` when `python` engine is used,
    # so we have to do the casting manually; also we can't use `converters` together with `dtype`
    train_df = train_df.astype(dtype)
    test_df = test_df.astype(dtype)

    return train_df, test_df
