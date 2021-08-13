"""Test the openml loader.
"""
import gzip
import json
import numpy as np
import os
import re
import scipy.sparse
import sklearn
import pytest

from sklearn import config_context
from sklearn.datasets import fetch_openml
from sklearn.datasets._openml import (_open_openml_url,
                                      _get_data_description_by_id,
                                      _download_data_arff,
                                      _get_local_path,
                                      _retry_with_clean_cache,
                                      _feature_to_dtype)
from sklearn.utils._testing import (assert_warns_message,
                                    assert_raise_message)
from sklearn.utils import is_scalar_nan
from sklearn.utils._testing import assert_allclose, assert_array_equal
from urllib.error import HTTPError
from sklearn.datasets.tests.test_common import check_return_X_y
from functools import partial


currdir = os.path.dirname(os.path.abspath(__file__))
# if True, urlopen will be monkey patched to only use local files
test_offline = True


def _test_features_list(data_id):
    # XXX Test is intended to verify/ensure correct decoding behavior
    # Not usable with sparse data or datasets that have columns marked as
    # {row_identifier, ignore}
    def decode_column(data_bunch, col_idx):
        col_name = data_bunch.feature_names[col_idx]
        if col_name in data_bunch.categories:
            # XXX: This would be faster with np.take, although it does not
            # handle missing values fast (also not with mode='wrap')
            cat = data_bunch.categories[col_name]
            result = [None if is_scalar_nan(idx) else cat[int(idx)]
                      for idx in data_bunch.data[:, col_idx]]
            return np.array(result, dtype='O')
        else:
            # non-nominal attribute
            return data_bunch.data[:, col_idx]

    data_bunch = fetch_openml(data_id=data_id, cache=False, target_column=None)

    # also obtain decoded arff
    data_description = _get_data_description_by_id(data_id, None)
    sparse = data_description['format'].lower() == 'sparse_arff'
    if sparse is True:
        raise ValueError('This test is not intended for sparse data, to keep '
                         'code relatively simple')
    data_arff = _download_data_arff(data_description['file_id'],
                                    sparse, None, False)
    data_downloaded = np.array(list(data_arff['data']), dtype='O')

    for i in range(len(data_bunch.feature_names)):
        # XXX: Test per column, as this makes it easier to avoid problems with
        # missing values

        np.testing.assert_array_equal(data_downloaded[:, i],
                                      decode_column(data_bunch, i))


def _fetch_dataset_from_openml(data_id, data_name, data_version,
                               target_column,
                               expected_observations, expected_features,
                               expected_missing,
                               expected_data_dtype, expected_target_dtype,
                               expect_sparse, compare_default_target):
    # fetches a dataset in three various ways from OpenML, using the
    # fetch_openml function, and does various checks on the validity of the
    # result. Note that this function can be mocked (by invoking
    # _monkey_patch_webbased_functions before invoking this function)
    data_by_name_id = fetch_openml(name=data_name, version=data_version,
                                   cache=False)
    assert int(data_by_name_id.details['id']) == data_id

    # Please note that cache=False is crucial, as the monkey patched files are
    # not consistent with reality
    fetch_openml(name=data_name, cache=False)
    # without specifying the version, there is no guarantee that the data id
    # will be the same

    # fetch with dataset id
    data_by_id = fetch_openml(data_id=data_id, cache=False,
                              target_column=target_column)
    assert data_by_id.details['name'] == data_name
    assert data_by_id.data.shape == (expected_observations, expected_features)
    if isinstance(target_column, str):
        # single target, so target is vector
        assert data_by_id.target.shape == (expected_observations, )
        assert data_by_id.target_names == [target_column]
    elif isinstance(target_column, list):
        # multi target, so target is array
        assert data_by_id.target.shape == (expected_observations,
                                           len(target_column))
        assert data_by_id.target_names == target_column
    assert data_by_id.data.dtype == np.float64
    assert data_by_id.target.dtype == expected_target_dtype
    assert len(data_by_id.feature_names) == expected_features
    for feature in data_by_id.feature_names:
        assert isinstance(feature, str)

    # TODO: pass in a list of expected nominal features
    for feature, categories in data_by_id.categories.items():
        feature_idx = data_by_id.feature_names.index(feature)
        values = np.unique(data_by_id.data[:, feature_idx])
        values = values[np.isfinite(values)]
        assert set(values) <= set(range(len(categories)))

    if compare_default_target:
        # check whether the data by id and data by id target are equal
        data_by_id_default = fetch_openml(data_id=data_id, cache=False)
        if data_by_id.data.dtype == np.float64:
            np.testing.assert_allclose(data_by_id.data,
                                       data_by_id_default.data)
        else:
            assert np.array_equal(data_by_id.data, data_by_id_default.data)
        if data_by_id.target.dtype == np.float64:
            np.testing.assert_allclose(data_by_id.target,
                                       data_by_id_default.target)
        else:
            assert np.array_equal(data_by_id.target, data_by_id_default.target)

    if expect_sparse:
        assert isinstance(data_by_id.data, scipy.sparse.csr_matrix)
    else:
        assert isinstance(data_by_id.data, np.ndarray)
        # np.isnan doesn't work on CSR matrix
        assert (np.count_nonzero(np.isnan(data_by_id.data)) ==
                expected_missing)

    # test return_X_y option
    fetch_func = partial(fetch_openml, data_id=data_id, cache=False,
                         target_column=target_column)
    check_return_X_y(data_by_id, fetch_func)
    return data_by_id


def _monkey_patch_webbased_functions(context,
                                     data_id,
                                     gzip_response):
    # monkey patches the urlopen function. Important note: Do NOT use this
    # in combination with a regular cache directory, as the files that are
    # stored as cache should not be mixed up with real openml datasets
    url_prefix_data_description = "https://openml.org/api/v1/json/data/"
    url_prefix_data_features = "https://openml.org/api/v1/json/data/features/"
    url_prefix_download_data = "https://openml.org/data/v1/"
    url_prefix_data_list = "https://openml.org/api/v1/json/data/list/"

    path_suffix = '.gz'
    read_fn = gzip.open

    class MockHTTPResponse:
        def __init__(self, data, is_gzip):
            self.data = data
            self.is_gzip = is_gzip

        def read(self, amt=-1):
            return self.data.read(amt)

        def tell(self):
            return self.data.tell()

        def seek(self, pos, whence=0):
            return self.data.seek(pos, whence)

        def close(self):
            self.data.close()

        def info(self):
            if self.is_gzip:
                return {'Content-Encoding': 'gzip'}
            return {}

    def _file_name(url, suffix):
        return (re.sub(r'\W', '-', url[len("https://openml.org/"):])
                + suffix + path_suffix)

    def _mock_urlopen_data_description(url, has_gzip_header):
        assert url.startswith(url_prefix_data_description)

        path = os.path.join(currdir, 'data', 'openml', str(data_id),
                            _file_name(url, '.json'))

        if has_gzip_header and gzip_response:
            fp = open(path, 'rb')
            return MockHTTPResponse(fp, True)
        else:
            fp = read_fn(path, 'rb')
            return MockHTTPResponse(fp, False)

    def _mock_urlopen_data_features(url, has_gzip_header):
        assert url.startswith(url_prefix_data_features)
        path = os.path.join(currdir, 'data', 'openml', str(data_id),
                            _file_name(url, '.json'))
        if has_gzip_header and gzip_response:
            fp = open(path, 'rb')
            return MockHTTPResponse(fp, True)
        else:
            fp = read_fn(path, 'rb')
            return MockHTTPResponse(fp, False)

    def _mock_urlopen_download_data(url, has_gzip_header):
        assert (url.startswith(url_prefix_download_data))

        path = os.path.join(currdir, 'data', 'openml', str(data_id),
                            _file_name(url, '.arff'))

        if has_gzip_header and gzip_response:
            fp = open(path, 'rb')
            return MockHTTPResponse(fp, True)
        else:
            fp = read_fn(path, 'rb')
            return MockHTTPResponse(fp, False)

    def _mock_urlopen_data_list(url, has_gzip_header):
        assert url.startswith(url_prefix_data_list)

        json_file_path = os.path.join(currdir, 'data', 'openml',
                                      str(data_id), _file_name(url, '.json'))
        # load the file itself, to simulate a http error
        json_data = json.loads(read_fn(json_file_path, 'rb').
                               read().decode('utf-8'))
        if 'error' in json_data:
            raise HTTPError(url=None, code=412,
                            msg='Simulated mock error',
                            hdrs=None, fp=None)

        if has_gzip_header:
            fp = open(json_file_path, 'rb')
            return MockHTTPResponse(fp, True)
        else:
            fp = read_fn(json_file_path, 'rb')
            return MockHTTPResponse(fp, False)

    def _mock_urlopen(request):
        url = request.get_full_url()
        has_gzip_header = request.get_header('Accept-encoding') == "gzip"
        if url.startswith(url_prefix_data_list):
            return _mock_urlopen_data_list(url, has_gzip_header)
        elif url.startswith(url_prefix_data_features):
            return _mock_urlopen_data_features(url, has_gzip_header)
        elif url.startswith(url_prefix_download_data):
            return _mock_urlopen_download_data(url, has_gzip_header)
        elif url.startswith(url_prefix_data_description):
            return _mock_urlopen_data_description(url, has_gzip_header)
        else:
            raise ValueError('Unknown mocking URL pattern: %s' % url)

    # XXX: Global variable
    if test_offline:
        context.setattr(sklearn.datasets._openml, 'urlopen', _mock_urlopen)


@pytest.mark.parametrize('feature, expected_dtype', [
    ({'data_type': 'string', 'number_of_missing_values': '0'}, object),
    ({'data_type': 'string', 'number_of_missing_values': '1'}, object),
    ({'data_type': 'numeric', 'number_of_missing_values': '0'}, np.float64),
    ({'data_type': 'numeric', 'number_of_missing_values': '1'}, np.float64),
    ({'data_type': 'real', 'number_of_missing_values': '0'}, np.float64),
    ({'data_type': 'real', 'number_of_missing_values': '1'}, np.float64),
    ({'data_type': 'integer', 'number_of_missing_values': '0'}, np.int64),
    ({'data_type': 'integer', 'number_of_missing_values': '1'}, np.float64),
    ({'data_type': 'nominal', 'number_of_missing_values': '0'}, 'category'),
    ({'data_type': 'nominal', 'number_of_missing_values': '1'}, 'category'),
])
def test_feature_to_dtype(feature, expected_dtype):
    assert _feature_to_dtype(feature) == expected_dtype


@pytest.mark.parametrize('feature', [
    {'data_type': 'datatime', 'number_of_missing_values': '0'}
])
def test_feature_to_dtype_error(feature):
    msg = 'Unsupported feature: {}'.format(feature)
    with pytest.raises(ValueError, match=msg):
        _feature_to_dtype(feature)


def test_fetch_openml_iris_pandas(monkeypatch):
    # classification dataset with numeric only columns
    pd = pytest.importorskip('pandas')
    CategoricalDtype = pd.api.types.CategoricalDtype
    data_id = 61
    data_shape = (150, 4)
    target_shape = (150, )
    frame_shape = (150, 5)

    target_dtype = CategoricalDtype(['Iris-setosa', 'Iris-versicolor',
                                     'Iris-virginica'])
    data_dtypes = [np.float64] * 4
    data_names = ['sepallength', 'sepalwidth', 'petallength', 'petalwidth']
    target_name = 'class'

    _monkey_patch_webbased_functions(monkeypatch, data_id, True)

    bunch = fetch_openml(data_id=data_id, as_frame=True, cache=False)
    data = bunch.data
    target = bunch.target
    frame = bunch.frame

    assert isinstance(data, pd.DataFrame)
    assert np.all(data.dtypes == data_dtypes)
    assert data.shape == data_shape
    assert np.all(data.columns == data_names)
    assert np.all(bunch.feature_names == data_names)
    assert bunch.target_names == [target_name]

    assert isinstance(target, pd.Series)
    assert target.dtype == target_dtype
    assert target.shape == target_shape
    assert target.name == target_name
    assert target.index.is_unique

    assert isinstance(frame, pd.DataFrame)
    assert frame.shape == frame_shape
    assert np.all(frame.dtypes == data_dtypes + [target_dtype])
    assert frame.index.is_unique


def test_fetch_openml_iris_pandas_equal_to_no_frame(monkeypatch):
    # as_frame = True returns the same underlying data as as_frame = False
    pytest.importorskip('pandas')
    data_id = 61

    _monkey_patch_webbased_functions(monkeypatch, data_id, True)

    frame_bunch = fetch_openml(data_id=data_id, as_frame=True, cache=False)
    frame_data = frame_bunch.data
    frame_target = frame_bunch.target

    norm_bunch = fetch_openml(data_id=data_id, as_frame=False, cache=False)
    norm_data = norm_bunch.data
    norm_target = norm_bunch.target

    assert_allclose(norm_data, frame_data)
    assert_array_equal(norm_target, frame_target)


def test_fetch_openml_iris_multitarget_pandas(monkeypatch):
    # classification dataset with numeric only columns
    pd = pytest.importorskip('pandas')
    CategoricalDtype = pd.api.types.CategoricalDtype
    data_id = 61
    data_shape = (150, 3)
    target_shape = (150, 2)
    frame_shape = (150, 5)
    target_column = ['petalwidth', 'petallength']

    cat_dtype = CategoricalDtype(['Iris-setosa', 'Iris-versicolor',
                                  'Iris-virginica'])
    data_dtypes = [np.float64, np.float64] + [cat_dtype]
    data_names = ['sepallength', 'sepalwidth', 'class']
    target_dtypes = [np.float64, np.float64]
    target_names = ['petalwidth', 'petallength']

    _monkey_patch_webbased_functions(monkeypatch, data_id, True)

    bunch = fetch_openml(data_id=data_id, as_frame=True, cache=False,
                         target_column=target_column)
    data = bunch.data
    target = bunch.target
    frame = bunch.frame

    assert isinstance(data, pd.DataFrame)
    assert np.all(data.dtypes == data_dtypes)
    assert data.shape == data_shape
    assert np.all(data.columns == data_names)
    assert np.all(bunch.feature_names == data_names)
    assert bunch.target_names == target_names

    assert isinstance(target, pd.DataFrame)
    assert np.all(target.dtypes == target_dtypes)
    assert target.shape == target_shape
    assert np.all(target.columns == target_names)

    assert isinstance(frame, pd.DataFrame)
    assert frame.shape == frame_shape
    assert np.all(frame.dtypes == [np.float64] * 4 + [cat_dtype])


def test_fetch_openml_anneal_pandas(monkeypatch):
    # classification dataset with numeric and categorical columns
    pd = pytest.importorskip('pandas')
    CategoricalDtype = pd.api.types.CategoricalDtype

    data_id = 2
    target_column = 'class'
    data_shape = (11, 38)
    target_shape = (11,)
    frame_shape = (11, 39)
    expected_data_categories = 32
    expected_data_floats = 6

    _monkey_patch_webbased_functions(monkeypatch, data_id, True)

    bunch = fetch_openml(data_id=data_id, as_frame=True,
                         target_column=target_column, cache=False)
    data = bunch.data
    target = bunch.target
    frame = bunch.frame

    assert isinstance(data, pd.DataFrame)
    assert data.shape == data_shape
    n_categories = len([dtype for dtype in data.dtypes
                       if isinstance(dtype, CategoricalDtype)])
    n_floats = len([dtype for dtype in data.dtypes if dtype.kind == 'f'])
    assert expected_data_categories == n_categories
    assert expected_data_floats == n_floats

    assert isinstance(target, pd.Series)
    assert target.shape == target_shape
    assert isinstance(target.dtype, CategoricalDtype)

    assert isinstance(frame, pd.DataFrame)
    assert frame.shape == frame_shape


def test_fetch_openml_cpu_pandas(monkeypatch):
    # regression dataset with numeric and categorical columns
    pd = pytest.importorskip('pandas')
    CategoricalDtype = pd.api.types.CategoricalDtype
    data_id = 561
    data_shape = (209, 7)
    target_shape = (209, )
    frame_shape = (209, 8)

    cat_dtype = CategoricalDtype(['adviser', 'amdahl', 'apollo', 'basf',
                                  'bti', 'burroughs', 'c.r.d', 'cdc',
                                  'cambex', 'dec', 'dg', 'formation',
                                  'four-phase', 'gould', 'hp', 'harris',
                                  'honeywell', 'ibm', 'ipl', 'magnuson',
                                  'microdata', 'nas', 'ncr', 'nixdorf',
                                  'perkin-elmer', 'prime', 'siemens',
                                  'sperry', 'sratus', 'wang'])
    data_dtypes = [cat_dtype] + [np.float64] * 6
    feature_names = ['vendor', 'MYCT', 'MMIN', 'MMAX', 'CACH',
                     'CHMIN', 'CHMAX']
    target_name = 'class'

    _monkey_patch_webbased_functions(monkeypatch, data_id, True)
    bunch = fetch_openml(data_id=data_id, as_frame=True, cache=False)
    data = bunch.data
    target = bunch.target
    frame = bunch.frame

    assert isinstance(data, pd.DataFrame)
    assert data.shape == data_shape
    assert np.all(data.dtypes == data_dtypes)
    assert np.all(data.columns == feature_names)
    assert np.all(bunch.feature_names == feature_names)
    assert bunch.target_names == [target_name]

    assert isinstance(target, pd.Series)
    assert target.shape == target_shape
    assert target.dtype == np.float64
    assert target.name == target_name

    assert isinstance(frame, pd.DataFrame)
    assert frame.shape == frame_shape


def test_fetch_openml_australian_pandas_error_sparse(monkeypatch):
    data_id = 292

    _monkey_patch_webbased_functions(monkeypatch, data_id, True)

    msg = 'Cannot return dataframe with sparse data'
    with pytest.raises(ValueError, match=msg):
        fetch_openml(data_id=data_id, as_frame=True, cache=False)


def test_convert_arff_data_dataframe_warning_low_memory_pandas(monkeypatch):
    pytest.importorskip('pandas')

    data_id = 1119
    _monkey_patch_webbased_functions(monkeypatch, data_id, True)

    msg = 'Could not adhere to working_memory config.'
    with pytest.warns(UserWarning, match=msg):
        with config_context(working_memory=1e-6):
            fetch_openml(data_id=data_id, as_frame=True, cache=False)


def test_fetch_openml_adultcensus_pandas_return_X_y(monkeypatch):
    pd = pytest.importorskip('pandas')
    CategoricalDtype = pd.api.types.CategoricalDtype

    data_id = 1119
    data_shape = (10, 14)
    target_shape = (10, )

    expected_data_categories = 8
    expected_data_floats = 6
    target_column = 'class'

    _monkey_patch_webbased_functions(monkeypatch, data_id, True)
    X, y = fetch_openml(data_id=data_id, as_frame=True, cache=False,
                        return_X_y=True)
    assert isinstance(X, pd.DataFrame)
    assert X.shape == data_shape
    n_categories = len([dtype for dtype in X.dtypes
                       if isinstance(dtype, CategoricalDtype)])
    n_floats = len([dtype for dtype in X.dtypes if dtype.kind == 'f'])
    assert expected_data_categories == n_categories
    assert expected_data_floats == n_floats

    assert isinstance(y, pd.Series)
    assert y.shape == target_shape
    assert y.name == target_column


def test_fetch_openml_adultcensus_pandas(monkeypatch):
    pd = pytest.importorskip('pandas')
    CategoricalDtype = pd.api.types.CategoricalDtype

    # Check because of the numeric row attribute (issue #12329)
    data_id = 1119
    data_shape = (10, 14)
    target_shape = (10, )
    frame_shape = (10, 15)

    expected_data_categories = 8
    expected_data_floats = 6
    target_column = 'class'

    _monkey_patch_webbased_functions(monkeypatch, data_id, True)
    bunch = fetch_openml(data_id=data_id, as_frame=True, cache=False)
    data = bunch.data
    target = bunch.target
    frame = bunch.frame

    assert isinstance(data, pd.DataFrame)
    assert data.shape == data_shape
    n_categories = len([dtype for dtype in data.dtypes
                       if isinstance(dtype, CategoricalDtype)])
    n_floats = len([dtype for dtype in data.dtypes if dtype.kind == 'f'])
    assert expected_data_categories == n_categories
    assert expected_data_floats == n_floats

    assert isinstance(target, pd.Series)
    assert target.shape == target_shape
    assert target.name == target_column

    assert isinstance(frame, pd.DataFrame)
    assert frame.shape == frame_shape


def test_fetch_openml_miceprotein_pandas(monkeypatch):
    # JvR: very important check, as this dataset defined several row ids
    # and ignore attributes. Note that data_features json has 82 attributes,
    # and row id (1), ignore attributes (3) have been removed.
    pd = pytest.importorskip('pandas')
    CategoricalDtype = pd.api.types.CategoricalDtype

    data_id = 40966
    data_shape = (7, 77)
    target_shape = (7, )
    frame_shape = (7, 78)

    target_column = 'class'
    frame_n_categories = 1
    frame_n_floats = 77

    _monkey_patch_webbased_functions(monkeypatch, data_id, True)
    bunch = fetch_openml(data_id=data_id, as_frame=True, cache=False)
    data = bunch.data
    target = bunch.target
    frame = bunch.frame

    assert isinstance(data, pd.DataFrame)
    assert data.shape == data_shape
    assert np.all(data.dtypes == np.float64)

    assert isinstance(target, pd.Series)
    assert isinstance(target.dtype, CategoricalDtype)
    assert target.shape == target_shape
    assert target.name == target_column

    assert isinstance(frame, pd.DataFrame)
    assert frame.shape == frame_shape
    n_categories = len([dtype for dtype in frame.dtypes
                       if isinstance(dtype, CategoricalDtype)])
    n_floats = len([dtype for dtype in frame.dtypes if dtype.kind == 'f'])
    assert frame_n_categories == n_categories
    assert frame_n_floats == n_floats


def test_fetch_openml_emotions_pandas(monkeypatch):
    # classification dataset with multiple targets (natively)
    pd = pytest.importorskip('pandas')
    CategoricalDtype = pd.api.types.CategoricalDtype

    data_id = 40589
    target_column = ['amazed.suprised', 'happy.pleased', 'relaxing.calm',
                     'quiet.still', 'sad.lonely', 'angry.aggresive']
    data_shape = (13, 72)
    target_shape = (13, 6)
    frame_shape = (13, 78)

    expected_frame_categories = 6
    expected_frame_floats = 72

    _monkey_patch_webbased_functions(monkeypatch, data_id, True)
    bunch = fetch_openml(data_id=data_id, as_frame=True, cache=False,
                         target_column=target_column)
    data = bunch.data
    target = bunch.target
    frame = bunch.frame

    assert isinstance(data, pd.DataFrame)
    assert data.shape == data_shape

    assert isinstance(target, pd.DataFrame)
    assert target.shape == target_shape
    assert np.all(target.columns == target_column)

    assert isinstance(frame, pd.DataFrame)
    assert frame.shape == frame_shape
    n_categories = len([dtype for dtype in frame.dtypes
                       if isinstance(dtype, CategoricalDtype)])
    n_floats = len([dtype for dtype in frame.dtypes if dtype.kind == 'f'])
    assert expected_frame_categories == n_categories
    assert expected_frame_floats == n_floats


def test_fetch_openml_titanic_pandas(monkeypatch):
    # dataset with strings
    pd = pytest.importorskip('pandas')
    CategoricalDtype = pd.api.types.CategoricalDtype

    data_id = 40945
    data_shape = (1309, 13)
    target_shape = (1309, )
    frame_shape = (1309, 14)
    name_to_dtype = {
        'pclass': np.float64,
        'name': object,
        'sex': CategoricalDtype(['female', 'male']),
        'age': np.float64,
        'sibsp': np.float64,
        'parch': np.float64,
        'ticket': object,
        'fare': np.float64,
        'cabin': object,
        'embarked': CategoricalDtype(['C', 'Q', 'S']),
        'boat': object,
        'body': np.float64,
        'home.dest': object,
        'survived': CategoricalDtype(['0', '1'])
    }

    frame_columns = ['pclass', 'survived', 'name', 'sex', 'age', 'sibsp',
                     'parch', 'ticket', 'fare', 'cabin', 'embarked',
                     'boat', 'body', 'home.dest']
    frame_dtypes = [name_to_dtype[col] for col in frame_columns]
    feature_names = ['pclass', 'name', 'sex', 'age', 'sibsp',
                     'parch', 'ticket', 'fare', 'cabin', 'embarked',
                     'boat', 'body', 'home.dest']
    target_name = 'survived'

    _monkey_patch_webbased_functions(monkeypatch, data_id, True)
    bunch = fetch_openml(data_id=data_id, as_frame=True, cache=False)
    data = bunch.data
    target = bunch.target
    frame = bunch.frame

    assert isinstance(data, pd.DataFrame)
    assert data.shape == data_shape
    assert np.all(data.columns == feature_names)
    assert bunch.target_names == [target_name]

    assert isinstance(target, pd.Series)
    assert target.shape == target_shape
    assert target.name == target_name
    assert target.dtype == name_to_dtype[target_name]

    assert isinstance(frame, pd.DataFrame)
    assert frame.shape == frame_shape
    assert np.all(frame.dtypes == frame_dtypes)


@pytest.mark.parametrize('gzip_response', [True, False])
def test_fetch_openml_iris(monkeypatch, gzip_response):
    # classification dataset with numeric only columns
    data_id = 61
    data_name = 'iris'
    data_version = 1
    target_column = 'class'
    expected_observations = 150
    expected_features = 4
    expected_missing = 0

    _monkey_patch_webbased_functions(monkeypatch, data_id, gzip_response)
    assert_warns_message(
        UserWarning,
        "Multiple active versions of the dataset matching the name"
        " iris exist. Versions may be fundamentally different, "
        "returning version 1.",
        _fetch_dataset_from_openml,
        **{'data_id': data_id, 'data_name': data_name,
           'data_version': data_version,
           'target_column': target_column,
           'expected_observations': expected_observations,
           'expected_features': expected_features,
           'expected_missing': expected_missing,
           'expect_sparse': False,
           'expected_data_dtype': np.float64,
           'expected_target_dtype': object,
           'compare_default_target': True}
    )


def test_decode_iris(monkeypatch):
    data_id = 61
    _monkey_patch_webbased_functions(monkeypatch, data_id, False)
    _test_features_list(data_id)


@pytest.mark.parametrize('gzip_response', [True, False])
def test_fetch_openml_iris_multitarget(monkeypatch, gzip_response):
    # classification dataset with numeric only columns
    data_id = 61
    data_name = 'iris'
    data_version = 1
    target_column = ['sepallength', 'sepalwidth']
    expected_observations = 150
    expected_features = 3
    expected_missing = 0

    _monkey_patch_webbased_functions(monkeypatch, data_id, gzip_response)
    _fetch_dataset_from_openml(data_id, data_name, data_version, target_column,
                               expected_observations, expected_features,
                               expected_missing,
                               object, np.float64, expect_sparse=False,
                               compare_default_target=False)


@pytest.mark.parametrize('gzip_response', [True, False])
def test_fetch_openml_anneal(monkeypatch, gzip_response):
    # classification dataset with numeric and categorical columns
    data_id = 2
    data_name = 'anneal'
    data_version = 1
    target_column = 'class'
    # Not all original instances included for space reasons
    expected_observations = 11
    expected_features = 38
    expected_missing = 267
    _monkey_patch_webbased_functions(monkeypatch, data_id, gzip_response)
    _fetch_dataset_from_openml(data_id, data_name, data_version, target_column,
                               expected_observations, expected_features,
                               expected_missing,
                               object, object, expect_sparse=False,
                               compare_default_target=True)


def test_decode_anneal(monkeypatch):
    data_id = 2
    _monkey_patch_webbased_functions(monkeypatch, data_id, False)
    _test_features_list(data_id)


@pytest.mark.parametrize('gzip_response', [True, False])
def test_fetch_openml_anneal_multitarget(monkeypatch, gzip_response):
    # classification dataset with numeric and categorical columns
    data_id = 2
    data_name = 'anneal'
    data_version = 1
    target_column = ['class', 'product-type', 'shape']
    # Not all original instances included for space reasons
    expected_observations = 11
    expected_features = 36
    expected_missing = 267
    _monkey_patch_webbased_functions(monkeypatch, data_id, gzip_response)
    _fetch_dataset_from_openml(data_id, data_name, data_version, target_column,
                               expected_observations, expected_features,
                               expected_missing,
                               object, object, expect_sparse=False,
                               compare_default_target=False)


@pytest.mark.parametrize('gzip_response', [True, False])
def test_fetch_openml_cpu(monkeypatch, gzip_response):
    # regression dataset with numeric and categorical columns
    data_id = 561
    data_name = 'cpu'
    data_version = 1
    target_column = 'class'
    expected_observations = 209
    expected_features = 7
    expected_missing = 0
    _monkey_patch_webbased_functions(monkeypatch, data_id, gzip_response)
    _fetch_dataset_from_openml(data_id, data_name, data_version, target_column,
                               expected_observations, expected_features,
                               expected_missing,
                               object, np.float64, expect_sparse=False,
                               compare_default_target=True)


def test_decode_cpu(monkeypatch):
    data_id = 561
    _monkey_patch_webbased_functions(monkeypatch, data_id, False)
    _test_features_list(data_id)


@pytest.mark.parametrize('gzip_response', [True, False])
def test_fetch_openml_australian(monkeypatch, gzip_response):
    # sparse dataset
    # Australian is the only sparse dataset that is reasonably small
    # as it is inactive, we need to catch the warning. Due to mocking
    # framework, it is not deactivated in our tests
    data_id = 292
    data_name = 'Australian'
    data_version = 1
    target_column = 'Y'
    # Not all original instances included for space reasons
    expected_observations = 85
    expected_features = 14
    expected_missing = 0
    _monkey_patch_webbased_functions(monkeypatch, data_id, gzip_response)
    assert_warns_message(
        UserWarning,
        "Version 1 of dataset Australian is inactive,",
        _fetch_dataset_from_openml,
        **{'data_id': data_id, 'data_name': data_name,
           'data_version': data_version,
           'target_column': target_column,
           'expected_observations': expected_observations,
           'expected_features': expected_features,
           'expected_missing': expected_missing,
           'expect_sparse': True,
           'expected_data_dtype': np.float64,
           'expected_target_dtype': object,
           'compare_default_target': False}  # numpy specific check
    )


@pytest.mark.parametrize('gzip_response', [True, False])
def test_fetch_openml_adultcensus(monkeypatch, gzip_response):
    # Check because of the numeric row attribute (issue #12329)
    data_id = 1119
    data_name = 'adult-census'
    data_version = 1
    target_column = 'class'
    # Not all original instances included for space reasons
    expected_observations = 10
    expected_features = 14
    expected_missing = 0
    _monkey_patch_webbased_functions(monkeypatch, data_id, gzip_response)
    _fetch_dataset_from_openml(data_id, data_name, data_version, target_column,
                               expected_observations, expected_features,
                               expected_missing,
                               np.float64, object, expect_sparse=False,
                               compare_default_target=True)


@pytest.mark.parametrize('gzip_response', [True, False])
def test_fetch_openml_miceprotein(monkeypatch, gzip_response):
    # JvR: very important check, as this dataset defined several row ids
    # and ignore attributes. Note that data_features json has 82 attributes,
    # and row id (1), ignore attributes (3) have been removed (and target is
    # stored in data.target)
    data_id = 40966
    data_name = 'MiceProtein'
    data_version = 4
    target_column = 'class'
    # Not all original instances included for space reasons
    expected_observations = 7
    expected_features = 77
    expected_missing = 7
    _monkey_patch_webbased_functions(monkeypatch, data_id, gzip_response)
    _fetch_dataset_from_openml(data_id, data_name, data_version, target_column,
                               expected_observations, expected_features,
                               expected_missing,
                               np.float64, object, expect_sparse=False,
                               compare_default_target=True)


@pytest.mark.parametrize('gzip_response', [True, False])
def test_fetch_openml_emotions(monkeypatch, gzip_response):
    # classification dataset with multiple targets (natively)
    data_id = 40589
    data_name = 'emotions'
    data_version = 3
    target_column = ['amazed.suprised', 'happy.pleased', 'relaxing.calm',
                     'quiet.still', 'sad.lonely', 'angry.aggresive']
    expected_observations = 13
    expected_features = 72
    expected_missing = 0
    _monkey_patch_webbased_functions(monkeypatch, data_id, gzip_response)

    _fetch_dataset_from_openml(data_id, data_name, data_version, target_column,
                               expected_observations, expected_features,
                               expected_missing,
                               np.float64, object, expect_sparse=False,
                               compare_default_target=True)


def test_decode_emotions(monkeypatch):
    data_id = 40589
    _monkey_patch_webbased_functions(monkeypatch, data_id, False)
    _test_features_list(data_id)


@pytest.mark.parametrize('gzip_response', [True, False])
def test_open_openml_url_cache(monkeypatch, gzip_response, tmpdir):
    data_id = 61

    _monkey_patch_webbased_functions(
        monkeypatch, data_id, gzip_response)
    openml_path = sklearn.datasets._openml._DATA_FILE.format(data_id)
    cache_directory = str(tmpdir.mkdir('scikit_learn_data'))
    # first fill the cache
    response1 = _open_openml_url(openml_path, cache_directory)
    # assert file exists
    location = _get_local_path(openml_path, cache_directory)
    assert os.path.isfile(location)
    # redownload, to utilize cache
    response2 = _open_openml_url(openml_path, cache_directory)
    assert response1.read() == response2.read()


@pytest.mark.parametrize('gzip_response', [True, False])
@pytest.mark.parametrize('write_to_disk', [True, False])
def test_open_openml_url_unlinks_local_path(
        monkeypatch, gzip_response, tmpdir, write_to_disk):
    data_id = 61
    openml_path = sklearn.datasets._openml._DATA_FILE.format(data_id)
    cache_directory = str(tmpdir.mkdir('scikit_learn_data'))
    location = _get_local_path(openml_path, cache_directory)

    def _mock_urlopen(request):
        if write_to_disk:
            with open(location, "w") as f:
                f.write("")
        raise ValueError("Invalid request")

    monkeypatch.setattr(sklearn.datasets._openml, 'urlopen', _mock_urlopen)

    with pytest.raises(ValueError, match="Invalid request"):
        _open_openml_url(openml_path, cache_directory)

    assert not os.path.exists(location)


def test_retry_with_clean_cache(tmpdir):
    data_id = 61
    openml_path = sklearn.datasets._openml._DATA_FILE.format(data_id)
    cache_directory = str(tmpdir.mkdir('scikit_learn_data'))
    location = _get_local_path(openml_path, cache_directory)
    os.makedirs(os.path.dirname(location))

    with open(location, 'w') as f:
        f.write("")

    @_retry_with_clean_cache(openml_path, cache_directory)
    def _load_data():
        # The first call will raise an error since location exists
        if os.path.exists(location):
            raise Exception("File exist!")
        return 1

    warn_msg = "Invalid cache, redownloading file"
    with pytest.warns(RuntimeWarning, match=warn_msg):
        result = _load_data()
    assert result == 1


def test_retry_with_clean_cache_http_error(tmpdir):
    data_id = 61
    openml_path = sklearn.datasets._openml._DATA_FILE.format(data_id)
    cache_directory = str(tmpdir.mkdir('scikit_learn_data'))

    @_retry_with_clean_cache(openml_path, cache_directory)
    def _load_data():
        raise HTTPError(url=None, code=412,
                        msg='Simulated mock error',
                        hdrs=None, fp=None)

    error_msg = "Simulated mock error"
    with pytest.raises(HTTPError, match=error_msg):
        _load_data()


@pytest.mark.parametrize('gzip_response', [True, False])
def test_fetch_openml_cache(monkeypatch, gzip_response, tmpdir):
    def _mock_urlopen_raise(request):
        raise ValueError('This mechanism intends to test correct cache'
                         'handling. As such, urlopen should never be '
                         'accessed. URL: %s' % request.get_full_url())
    data_id = 2
    cache_directory = str(tmpdir.mkdir('scikit_learn_data'))
    _monkey_patch_webbased_functions(
        monkeypatch, data_id, gzip_response)
    X_fetched, y_fetched = fetch_openml(data_id=data_id, cache=True,
                                        data_home=cache_directory,
                                        return_X_y=True)

    monkeypatch.setattr(sklearn.datasets._openml, 'urlopen',
                        _mock_urlopen_raise)

    X_cached, y_cached = fetch_openml(data_id=data_id, cache=True,
                                      data_home=cache_directory,
                                      return_X_y=True)
    np.testing.assert_array_equal(X_fetched, X_cached)
    np.testing.assert_array_equal(y_fetched, y_cached)


@pytest.mark.parametrize('gzip_response', [True, False])
def test_fetch_openml_notarget(monkeypatch, gzip_response):
    data_id = 61
    target_column = None
    expected_observations = 150
    expected_features = 5

    _monkey_patch_webbased_functions(monkeypatch, data_id, gzip_response)
    data = fetch_openml(data_id=data_id, target_column=target_column,
                        cache=False)
    assert data.data.shape == (expected_observations, expected_features)
    assert data.target is None


@pytest.mark.parametrize('gzip_response', [True, False])
def test_fetch_openml_inactive(monkeypatch, gzip_response):
    # fetch inactive dataset by id
    data_id = 40675
    _monkey_patch_webbased_functions(monkeypatch, data_id, gzip_response)
    glas2 = assert_warns_message(
        UserWarning, "Version 1 of dataset glass2 is inactive,", fetch_openml,
        data_id=data_id, cache=False)
    # fetch inactive dataset by name and version
    assert glas2.data.shape == (163, 9)
    glas2_by_version = assert_warns_message(
        UserWarning, "Version 1 of dataset glass2 is inactive,", fetch_openml,
        data_id=None, name="glass2", version=1, cache=False)
    assert int(glas2_by_version.details['id']) == data_id


@pytest.mark.parametrize('gzip_response', [True, False])
def test_fetch_nonexiting(monkeypatch, gzip_response):
    # there is no active version of glass2
    data_id = 40675
    _monkey_patch_webbased_functions(monkeypatch, data_id, gzip_response)
    # Note that we only want to search by name (not data id)
    assert_raise_message(ValueError, "No active dataset glass2 found",
                         fetch_openml, name='glass2', cache=False)


@pytest.mark.parametrize('gzip_response', [True, False])
def test_raises_illegal_multitarget(monkeypatch, gzip_response):
    data_id = 61
    targets = ['sepalwidth', 'class']
    _monkey_patch_webbased_functions(monkeypatch, data_id, gzip_response)
    # Note that we only want to search by name (not data id)
    assert_raise_message(ValueError,
                         "Can only handle homogeneous multi-target datasets,",
                         fetch_openml, data_id=data_id,
                         target_column=targets, cache=False)


@pytest.mark.parametrize('gzip_response', [True, False])
def test_warn_ignore_attribute(monkeypatch, gzip_response):
    data_id = 40966
    expected_row_id_msg = "target_column={} has flag is_row_identifier."
    expected_ignore_msg = "target_column={} has flag is_ignore."
    _monkey_patch_webbased_functions(monkeypatch, data_id, gzip_response)
    # single column test
    assert_warns_message(UserWarning, expected_row_id_msg.format('MouseID'),
                         fetch_openml, data_id=data_id,
                         target_column='MouseID',
                         cache=False)
    assert_warns_message(UserWarning, expected_ignore_msg.format('Genotype'),
                         fetch_openml, data_id=data_id,
                         target_column='Genotype',
                         cache=False)
    # multi column test
    assert_warns_message(UserWarning, expected_row_id_msg.format('MouseID'),
                         fetch_openml, data_id=data_id,
                         target_column=['MouseID', 'class'],
                         cache=False)
    assert_warns_message(UserWarning, expected_ignore_msg.format('Genotype'),
                         fetch_openml, data_id=data_id,
                         target_column=['Genotype', 'class'],
                         cache=False)


@pytest.mark.parametrize('gzip_response', [True, False])
def test_string_attribute_without_dataframe(monkeypatch, gzip_response):
    data_id = 40945
    _monkey_patch_webbased_functions(monkeypatch, data_id, gzip_response)
    # single column test
    assert_raise_message(ValueError,
                         ('STRING attributes are not supported for '
                          'array representation. Try as_frame=True'),
                         fetch_openml, data_id=data_id, cache=False)


@pytest.mark.parametrize('gzip_response', [True, False])
def test_dataset_with_openml_error(monkeypatch, gzip_response):
    data_id = 1
    _monkey_patch_webbased_functions(monkeypatch, data_id, gzip_response)
    assert_warns_message(
        UserWarning,
        "OpenML registered a problem with the dataset. It might be unusable. "
        "Error:",
        fetch_openml, data_id=data_id, cache=False
    )


@pytest.mark.parametrize('gzip_response', [True, False])
def test_dataset_with_openml_warning(monkeypatch, gzip_response):
    data_id = 3
    _monkey_patch_webbased_functions(monkeypatch, data_id, gzip_response)
    assert_warns_message(
        UserWarning,
        "OpenML raised a warning on the dataset. It might be unusable. "
        "Warning:",
        fetch_openml, data_id=data_id, cache=False
    )


@pytest.mark.parametrize('gzip_response', [True, False])
def test_illegal_column(monkeypatch, gzip_response):
    data_id = 61
    _monkey_patch_webbased_functions(monkeypatch, data_id, gzip_response)
    assert_raise_message(KeyError, "Could not find target_column=",
                         fetch_openml, data_id=data_id,
                         target_column='undefined', cache=False)

    assert_raise_message(KeyError, "Could not find target_column=",
                         fetch_openml, data_id=data_id,
                         target_column=['undefined', 'class'],
                         cache=False)


@pytest.mark.parametrize('gzip_response', [True, False])
def test_fetch_openml_raises_missing_values_target(monkeypatch, gzip_response):
    data_id = 2
    _monkey_patch_webbased_functions(monkeypatch, data_id, gzip_response)
    assert_raise_message(ValueError, "Target column ",
                         fetch_openml, data_id=data_id, target_column='family')


def test_fetch_openml_raises_illegal_argument():
    assert_raise_message(ValueError, "Dataset data_id=",
                         fetch_openml, data_id=-1, name="name")

    assert_raise_message(ValueError, "Dataset data_id=",
                         fetch_openml, data_id=-1, name=None,
                         version="version")

    assert_raise_message(ValueError, "Dataset data_id=",
                         fetch_openml, data_id=-1, name="name",
                         version="version")

    assert_raise_message(ValueError, "Neither name nor data_id are provided. "
                         "Please provide name or data_id.", fetch_openml)


@pytest.mark.parametrize('gzip_response', [True, False])
def test_fetch_openml_with_ignored_feature(monkeypatch, gzip_response):
    # Regression test for #14340
    # 62 is the ID of the ZOO dataset
    data_id = 62
    _monkey_patch_webbased_functions(monkeypatch, data_id, gzip_response)

    dataset = sklearn.datasets.fetch_openml(data_id=data_id, cache=False)
    assert dataset is not None
    # The dataset has 17 features, including 1 ignored (animal),
    # so we assert that we don't have the ignored feature in the final Bunch
    assert dataset['data'].shape == (101, 16)
    assert 'animal' not in dataset['feature_names']
