# -*- coding: utf-8 -*-

import re

import numpy as np
from scipy import sparse
import pytest

from sklearn.exceptions import NotFittedError
from sklearn.utils._testing import assert_array_equal
from sklearn.utils._testing import assert_allclose

from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import OrdinalEncoder


def toarray(a):
    if hasattr(a, "toarray"):
        a = a.toarray()
    return a


def test_one_hot_encoder_sparse_dense():
    # check that sparse and dense will give the same results

    X = np.array([[3, 2, 1], [0, 1, 1]])
    enc_sparse = OneHotEncoder()
    enc_dense = OneHotEncoder(sparse=False)

    X_trans_sparse = enc_sparse.fit_transform(X)
    X_trans_dense = enc_dense.fit_transform(X)

    assert X_trans_sparse.shape == (2, 5)
    assert X_trans_dense.shape == (2, 5)

    assert sparse.issparse(X_trans_sparse)
    assert not sparse.issparse(X_trans_dense)

    # check outcome
    assert_array_equal(X_trans_sparse.toarray(), [[0., 1., 0., 1., 1.],
                                                  [1., 0., 1., 0., 1.]])
    assert_array_equal(X_trans_sparse.toarray(), X_trans_dense)


def test_one_hot_encoder_diff_n_features():
    X = np.array([[0, 2, 1], [1, 0, 3], [1, 0, 2]])
    X2 = np.array([[1, 0]])
    enc = OneHotEncoder()
    enc.fit(X)
    err_msg = ("The number of features in X is different to the number of "
               "features of the fitted data.")
    with pytest.raises(ValueError, match=err_msg):
        enc.transform(X2)


def test_one_hot_encoder_handle_unknown():
    X = np.array([[0, 2, 1], [1, 0, 3], [1, 0, 2]])
    X2 = np.array([[4, 1, 1]])

    # Test that one hot encoder raises error for unknown features
    # present during transform.
    oh = OneHotEncoder(handle_unknown='error')
    oh.fit(X)
    with pytest.raises(ValueError, match='Found unknown categories'):
        oh.transform(X2)

    # Test the ignore option, ignores unknown features (giving all 0's)
    oh = OneHotEncoder(handle_unknown='ignore')
    oh.fit(X)
    X2_passed = X2.copy()
    assert_array_equal(
        oh.transform(X2_passed).toarray(),
        np.array([[0.,  0.,  0.,  0.,  1.,  0.,  0.]]))
    # ensure transformed data was not modified in place
    assert_allclose(X2, X2_passed)

    # Raise error if handle_unknown is neither ignore or error.
    oh = OneHotEncoder(handle_unknown='42')
    with pytest.raises(ValueError, match='handle_unknown should be either'):
        oh.fit(X)


def test_one_hot_encoder_not_fitted():
    X = np.array([['a'], ['b']])
    enc = OneHotEncoder(categories=['a', 'b'])
    msg = ("This OneHotEncoder instance is not fitted yet. "
           "Call 'fit' with appropriate arguments before using this "
           "estimator.")
    with pytest.raises(NotFittedError, match=msg):
        enc.transform(X)


def test_one_hot_encoder_handle_unknown_strings():
    X = np.array(['11111111', '22', '333', '4444']).reshape((-1, 1))
    X2 = np.array(['55555', '22']).reshape((-1, 1))
    # Non Regression test for the issue #12470
    # Test the ignore option, when categories are numpy string dtype
    # particularly when the known category strings are larger
    # than the unknown category strings
    oh = OneHotEncoder(handle_unknown='ignore')
    oh.fit(X)
    X2_passed = X2.copy()
    assert_array_equal(
        oh.transform(X2_passed).toarray(),
        np.array([[0.,  0.,  0.,  0.], [0.,  1.,  0.,  0.]]))
    # ensure transformed data was not modified in place
    assert_array_equal(X2, X2_passed)


@pytest.mark.parametrize("output_dtype", [np.int32, np.float32, np.float64])
@pytest.mark.parametrize("input_dtype", [np.int32, np.float32, np.float64])
def test_one_hot_encoder_dtype(input_dtype, output_dtype):
    X = np.asarray([[0, 1]], dtype=input_dtype).T
    X_expected = np.asarray([[1, 0], [0, 1]], dtype=output_dtype)

    oh = OneHotEncoder(categories='auto', dtype=output_dtype)
    assert_array_equal(oh.fit_transform(X).toarray(), X_expected)
    assert_array_equal(oh.fit(X).transform(X).toarray(), X_expected)

    oh = OneHotEncoder(categories='auto', dtype=output_dtype, sparse=False)
    assert_array_equal(oh.fit_transform(X), X_expected)
    assert_array_equal(oh.fit(X).transform(X), X_expected)


@pytest.mark.parametrize("output_dtype", [np.int32, np.float32, np.float64])
def test_one_hot_encoder_dtype_pandas(output_dtype):
    pd = pytest.importorskip('pandas')

    X_df = pd.DataFrame({'A': ['a', 'b'], 'B': [1, 2]})
    X_expected = np.array([[1, 0, 1, 0], [0, 1, 0, 1]], dtype=output_dtype)

    oh = OneHotEncoder(dtype=output_dtype)
    assert_array_equal(oh.fit_transform(X_df).toarray(), X_expected)
    assert_array_equal(oh.fit(X_df).transform(X_df).toarray(), X_expected)

    oh = OneHotEncoder(dtype=output_dtype, sparse=False)
    assert_array_equal(oh.fit_transform(X_df), X_expected)
    assert_array_equal(oh.fit(X_df).transform(X_df), X_expected)


def test_one_hot_encoder_feature_names():
    enc = OneHotEncoder()
    X = [['Male', 1, 'girl', 2, 3],
         ['Female', 41, 'girl', 1, 10],
         ['Male', 51, 'boy', 12, 3],
         ['Male', 91, 'girl', 21, 30]]

    enc.fit(X)
    feature_names = enc.get_feature_names()
    assert isinstance(feature_names, np.ndarray)

    assert_array_equal(['x0_Female', 'x0_Male',
                        'x1_1', 'x1_41', 'x1_51', 'x1_91',
                        'x2_boy', 'x2_girl',
                        'x3_1', 'x3_2', 'x3_12', 'x3_21',
                        'x4_3',
                        'x4_10', 'x4_30'], feature_names)

    feature_names2 = enc.get_feature_names(['one', 'two',
                                            'three', 'four', 'five'])

    assert_array_equal(['one_Female', 'one_Male',
                        'two_1', 'two_41', 'two_51', 'two_91',
                        'three_boy', 'three_girl',
                        'four_1', 'four_2', 'four_12', 'four_21',
                        'five_3', 'five_10', 'five_30'], feature_names2)

    with pytest.raises(ValueError, match="input_features should have length"):
        enc.get_feature_names(['one', 'two'])


def test_one_hot_encoder_feature_names_unicode():
    enc = OneHotEncoder()
    X = np.array([['c❤t1', 'dat2']], dtype=object).T
    enc.fit(X)
    feature_names = enc.get_feature_names()
    assert_array_equal(['x0_c❤t1', 'x0_dat2'], feature_names)
    feature_names = enc.get_feature_names(input_features=['n👍me'])
    assert_array_equal(['n👍me_c❤t1', 'n👍me_dat2'], feature_names)


def test_one_hot_encoder_set_params():
    X = np.array([[1, 2]]).T
    oh = OneHotEncoder()
    # set params on not yet fitted object
    oh.set_params(categories=[[0, 1, 2, 3]])
    assert oh.get_params()['categories'] == [[0, 1, 2, 3]]
    assert oh.fit_transform(X).toarray().shape == (2, 4)
    # set params on already fitted object
    oh.set_params(categories=[[0, 1, 2, 3, 4]])
    assert oh.fit_transform(X).toarray().shape == (2, 5)


def check_categorical_onehot(X):
    enc = OneHotEncoder(categories='auto')
    Xtr1 = enc.fit_transform(X)

    enc = OneHotEncoder(categories='auto', sparse=False)
    Xtr2 = enc.fit_transform(X)

    assert_allclose(Xtr1.toarray(), Xtr2)

    assert sparse.isspmatrix_csr(Xtr1)
    return Xtr1.toarray()


@pytest.mark.parametrize("X", [
    [['def', 1, 55], ['abc', 2, 55]],
    np.array([[10, 1, 55], [5, 2, 55]]),
    np.array([['b', 'A', 'cat'], ['a', 'B', 'cat']], dtype=object)
    ], ids=['mixed', 'numeric', 'object'])
def test_one_hot_encoder(X):
    Xtr = check_categorical_onehot(np.array(X)[:, [0]])
    assert_allclose(Xtr, [[0, 1], [1, 0]])

    Xtr = check_categorical_onehot(np.array(X)[:, [0, 1]])
    assert_allclose(Xtr, [[0, 1, 1, 0], [1, 0, 0, 1]])

    Xtr = OneHotEncoder(categories='auto').fit_transform(X)
    assert_allclose(Xtr.toarray(), [[0, 1, 1, 0,  1], [1, 0, 0, 1, 1]])


@pytest.mark.parametrize('sparse_', [False, True])
@pytest.mark.parametrize('drop', [None, 'first'])
def test_one_hot_encoder_inverse(sparse_, drop):
    X = [['abc', 2, 55], ['def', 1, 55], ['abc', 3, 55]]
    enc = OneHotEncoder(sparse=sparse_, drop=drop)
    X_tr = enc.fit_transform(X)
    exp = np.array(X, dtype=object)
    assert_array_equal(enc.inverse_transform(X_tr), exp)

    X = [[2, 55], [1, 55], [3, 55]]
    enc = OneHotEncoder(sparse=sparse_, categories='auto',
                        drop=drop)
    X_tr = enc.fit_transform(X)
    exp = np.array(X)
    assert_array_equal(enc.inverse_transform(X_tr), exp)

    if drop is None:
        # with unknown categories
        # drop is incompatible with handle_unknown=ignore
        X = [['abc', 2, 55], ['def', 1, 55], ['abc', 3, 55]]
        enc = OneHotEncoder(sparse=sparse_, handle_unknown='ignore',
                            categories=[['abc', 'def'], [1, 2],
                                        [54, 55, 56]])
        X_tr = enc.fit_transform(X)
        exp = np.array(X, dtype=object)
        exp[2, 1] = None
        assert_array_equal(enc.inverse_transform(X_tr), exp)

        # with an otherwise numerical output, still object if unknown
        X = [[2, 55], [1, 55], [3, 55]]
        enc = OneHotEncoder(sparse=sparse_, categories=[[1, 2], [54, 56]],
                            handle_unknown='ignore')
        X_tr = enc.fit_transform(X)
        exp = np.array(X, dtype=object)
        exp[2, 0] = None
        exp[:, 1] = None
        assert_array_equal(enc.inverse_transform(X_tr), exp)

    # incorrect shape raises
    X_tr = np.array([[0, 1, 1], [1, 0, 1]])
    msg = re.escape('Shape of the passed X data is not correct')
    with pytest.raises(ValueError, match=msg):
        enc.inverse_transform(X_tr)


@pytest.mark.parametrize("method", ['fit', 'fit_transform'])
@pytest.mark.parametrize("X", [
    [1, 2],
    np.array([3., 4.])
    ])
def test_X_is_not_1D(X, method):
    oh = OneHotEncoder()

    msg = ("Expected 2D array, got 1D array instead")
    with pytest.raises(ValueError, match=msg):
        getattr(oh, method)(X)


@pytest.mark.parametrize("method", ['fit', 'fit_transform'])
def test_X_is_not_1D_pandas(method):
    pd = pytest.importorskip('pandas')
    X = pd.Series([6, 3, 4, 6])
    oh = OneHotEncoder()

    msg = ("Expected 2D array, got 1D array instead")
    with pytest.raises(ValueError, match=msg):
        getattr(oh, method)(X)


@pytest.mark.parametrize("X, cat_exp, cat_dtype", [
    ([['abc', 55], ['def', 55]], [['abc', 'def'], [55]], np.object_),
    (np.array([[1, 2], [3, 2]]), [[1, 3], [2]], np.integer),
    (np.array([['A', 'cat'], ['B', 'cat']], dtype=object),
     [['A', 'B'], ['cat']], np.object_),
    (np.array([['A', 'cat'], ['B', 'cat']]),
     [['A', 'B'], ['cat']], np.str_)
    ], ids=['mixed', 'numeric', 'object', 'string'])
def test_one_hot_encoder_categories(X, cat_exp, cat_dtype):
    # order of categories should not depend on order of samples
    for Xi in [X, X[::-1]]:
        enc = OneHotEncoder(categories='auto')
        enc.fit(Xi)
        # assert enc.categories == 'auto'
        assert isinstance(enc.categories_, list)
        for res, exp in zip(enc.categories_, cat_exp):
            assert res.tolist() == exp
            assert np.issubdtype(res.dtype, cat_dtype)


@pytest.mark.parametrize("X, X2, cats, cat_dtype", [
    (np.array([['a', 'b']], dtype=object).T,
     np.array([['a', 'd']], dtype=object).T,
     [['a', 'b', 'c']], np.object_),
    (np.array([[1, 2]], dtype='int64').T,
     np.array([[1, 4]], dtype='int64').T,
     [[1, 2, 3]], np.int64),
    (np.array([['a', 'b']], dtype=object).T,
     np.array([['a', 'd']], dtype=object).T,
     [np.array(['a', 'b', 'c'])], np.object_),
    ], ids=['object', 'numeric', 'object-string-cat'])
def test_one_hot_encoder_specified_categories(X, X2, cats, cat_dtype):
    enc = OneHotEncoder(categories=cats)
    exp = np.array([[1., 0., 0.],
                    [0., 1., 0.]])
    assert_array_equal(enc.fit_transform(X).toarray(), exp)
    assert list(enc.categories[0]) == list(cats[0])
    assert enc.categories_[0].tolist() == list(cats[0])
    # manually specified categories should have same dtype as
    # the data when coerced from lists
    assert enc.categories_[0].dtype == cat_dtype

    # when specifying categories manually, unknown categories should already
    # raise when fitting
    enc = OneHotEncoder(categories=cats)
    with pytest.raises(ValueError, match="Found unknown categories"):
        enc.fit(X2)
    enc = OneHotEncoder(categories=cats, handle_unknown='ignore')
    exp = np.array([[1., 0., 0.], [0., 0., 0.]])
    assert_array_equal(enc.fit(X2).transform(X2).toarray(), exp)


def test_one_hot_encoder_unsorted_categories():
    X = np.array([['a', 'b']], dtype=object).T

    enc = OneHotEncoder(categories=[['b', 'a', 'c']])
    exp = np.array([[0., 1., 0.],
                    [1., 0., 0.]])
    assert_array_equal(enc.fit(X).transform(X).toarray(), exp)
    assert_array_equal(enc.fit_transform(X).toarray(), exp)
    assert enc.categories_[0].tolist() == ['b', 'a', 'c']
    assert np.issubdtype(enc.categories_[0].dtype, np.object_)

    # unsorted passed categories still raise for numerical values
    X = np.array([[1, 2]]).T
    enc = OneHotEncoder(categories=[[2, 1, 3]])
    msg = 'Unsorted categories are not supported'
    with pytest.raises(ValueError, match=msg):
        enc.fit_transform(X)


def test_one_hot_encoder_specified_categories_mixed_columns():
    # multiple columns
    X = np.array([['a', 'b'], [0, 2]], dtype=object).T
    enc = OneHotEncoder(categories=[['a', 'b', 'c'], [0, 1, 2]])
    exp = np.array([[1., 0., 0., 1., 0., 0.],
                    [0., 1., 0., 0., 0., 1.]])
    assert_array_equal(enc.fit_transform(X).toarray(), exp)
    assert enc.categories_[0].tolist() == ['a', 'b', 'c']
    assert np.issubdtype(enc.categories_[0].dtype, np.object_)
    assert enc.categories_[1].tolist() == [0, 1, 2]
    # integer categories but from object dtype data
    assert np.issubdtype(enc.categories_[1].dtype, np.object_)


def test_one_hot_encoder_pandas():
    pd = pytest.importorskip('pandas')

    X_df = pd.DataFrame({'A': ['a', 'b'], 'B': [1, 2]})

    Xtr = check_categorical_onehot(X_df)
    assert_allclose(Xtr, [[1, 0, 1, 0], [0, 1, 0, 1]])


@pytest.mark.parametrize("drop, expected_names",
                         [('first', ['x0_c', 'x2_b']),
                          (['c', 2, 'b'], ['x0_b', 'x2_a'])],
                         ids=['first', 'manual'])
def test_one_hot_encoder_feature_names_drop(drop, expected_names):
    X = [['c', 2, 'a'],
         ['b', 2, 'b']]

    ohe = OneHotEncoder(drop=drop)
    ohe.fit(X)
    feature_names = ohe.get_feature_names()
    assert isinstance(feature_names, np.ndarray)
    assert_array_equal(expected_names, feature_names)


@pytest.mark.parametrize("X", [np.array([[1, np.nan]]).T,
                               np.array([['a', np.nan]], dtype=object).T],
                         ids=['numeric', 'object'])
@pytest.mark.parametrize("as_data_frame", [False, True],
                         ids=['array', 'dataframe'])
@pytest.mark.parametrize("handle_unknown", ['error', 'ignore'])
def test_one_hot_encoder_raise_missing(X, as_data_frame, handle_unknown):
    if as_data_frame:
        pd = pytest.importorskip('pandas')
        X = pd.DataFrame(X)

    ohe = OneHotEncoder(categories='auto', handle_unknown=handle_unknown)

    with pytest.raises(ValueError, match="Input contains NaN"):
        ohe.fit(X)

    with pytest.raises(ValueError, match="Input contains NaN"):
        ohe.fit_transform(X)

    if as_data_frame:
        X_partial = X.iloc[:1, :]
    else:
        X_partial = X[:1, :]

    ohe.fit(X_partial)

    with pytest.raises(ValueError, match="Input contains NaN"):
        ohe.transform(X)


@pytest.mark.parametrize("X", [
    [['abc', 2, 55], ['def', 1, 55]],
    np.array([[10, 2, 55], [20, 1, 55]]),
    np.array([['a', 'B', 'cat'], ['b', 'A', 'cat']], dtype=object)
    ], ids=['mixed', 'numeric', 'object'])
def test_ordinal_encoder(X):
    enc = OrdinalEncoder()
    exp = np.array([[0, 1, 0],
                    [1, 0, 0]], dtype='int64')
    assert_array_equal(enc.fit_transform(X), exp.astype('float64'))
    enc = OrdinalEncoder(dtype='int64')
    assert_array_equal(enc.fit_transform(X), exp)


@pytest.mark.parametrize("X, X2, cats, cat_dtype", [
    (np.array([['a', 'b']], dtype=object).T,
     np.array([['a', 'd']], dtype=object).T,
     [['a', 'b', 'c']], np.object_),
    (np.array([[1, 2]], dtype='int64').T,
     np.array([[1, 4]], dtype='int64').T,
     [[1, 2, 3]], np.int64),
    (np.array([['a', 'b']], dtype=object).T,
     np.array([['a', 'd']], dtype=object).T,
     [np.array(['a', 'b', 'c'])], np.object_),
    ], ids=['object', 'numeric', 'object-string-cat'])
def test_ordinal_encoder_specified_categories(X, X2, cats, cat_dtype):
    enc = OrdinalEncoder(categories=cats)
    exp = np.array([[0.], [1.]])
    assert_array_equal(enc.fit_transform(X), exp)
    assert list(enc.categories[0]) == list(cats[0])
    assert enc.categories_[0].tolist() == list(cats[0])
    # manually specified categories should have same dtype as
    # the data when coerced from lists
    assert enc.categories_[0].dtype == cat_dtype

    # when specifying categories manually, unknown categories should already
    # raise when fitting
    enc = OrdinalEncoder(categories=cats)
    with pytest.raises(ValueError, match="Found unknown categories"):
        enc.fit(X2)


def test_ordinal_encoder_inverse():
    X = [['abc', 2, 55], ['def', 1, 55]]
    enc = OrdinalEncoder()
    X_tr = enc.fit_transform(X)
    exp = np.array(X, dtype=object)
    assert_array_equal(enc.inverse_transform(X_tr), exp)

    # incorrect shape raises
    X_tr = np.array([[0, 1, 1, 2], [1, 0, 1, 0]])
    msg = re.escape('Shape of the passed X data is not correct')
    with pytest.raises(ValueError, match=msg):
        enc.inverse_transform(X_tr)


@pytest.mark.parametrize("X", [np.array([[1, np.nan]]).T,
                               np.array([['a', np.nan]], dtype=object).T],
                         ids=['numeric', 'object'])
def test_ordinal_encoder_raise_missing(X):
    ohe = OrdinalEncoder()

    with pytest.raises(ValueError, match="Input contains NaN"):
        ohe.fit(X)

    with pytest.raises(ValueError, match="Input contains NaN"):
        ohe.fit_transform(X)

    ohe.fit(X[:1, :])

    with pytest.raises(ValueError, match="Input contains NaN"):
        ohe.transform(X)


def test_ordinal_encoder_raise_categories_shape():

    X = np.array([['Low', 'Medium', 'High', 'Medium', 'Low']], dtype=object).T
    cats = ['Low', 'Medium', 'High']
    enc = OrdinalEncoder(categories=cats)
    msg = ("Shape mismatch: if categories is an array,")

    with pytest.raises(ValueError, match=msg):
        enc.fit(X)


def test_encoder_dtypes():
    # check that dtypes are preserved when determining categories
    enc = OneHotEncoder(categories='auto')
    exp = np.array([[1., 0., 1., 0.], [0., 1., 0., 1.]], dtype='float64')

    for X in [np.array([[1, 2], [3, 4]], dtype='int64'),
              np.array([[1, 2], [3, 4]], dtype='float64'),
              np.array([['a', 'b'], ['c', 'd']]),  # string dtype
              np.array([[1, 'a'], [3, 'b']], dtype='object')]:
        enc.fit(X)
        assert all([enc.categories_[i].dtype == X.dtype for i in range(2)])
        assert_array_equal(enc.transform(X).toarray(), exp)

    X = [[1, 2], [3, 4]]
    enc.fit(X)
    assert all([np.issubdtype(enc.categories_[i].dtype, np.integer)
                for i in range(2)])
    assert_array_equal(enc.transform(X).toarray(), exp)

    X = [[1, 'a'], [3, 'b']]
    enc.fit(X)
    assert all([enc.categories_[i].dtype == 'object' for i in range(2)])
    assert_array_equal(enc.transform(X).toarray(), exp)


def test_encoder_dtypes_pandas():
    # check dtype (similar to test_categorical_encoder_dtypes for dataframes)
    pd = pytest.importorskip('pandas')

    enc = OneHotEncoder(categories='auto')
    exp = np.array([[1., 0., 1., 0., 1., 0.],
                    [0., 1., 0., 1., 0., 1.]], dtype='float64')

    X = pd.DataFrame({'A': [1, 2], 'B': [3, 4], 'C': [5, 6]}, dtype='int64')
    enc.fit(X)
    assert all([enc.categories_[i].dtype == 'int64' for i in range(2)])
    assert_array_equal(enc.transform(X).toarray(), exp)

    X = pd.DataFrame({'A': [1, 2], 'B': ['a', 'b'], 'C': [3., 4.]})
    X_type = [X['A'].dtype, X['B'].dtype, X['C'].dtype]
    enc.fit(X)
    assert all([enc.categories_[i].dtype == X_type[i] for i in range(3)])
    assert_array_equal(enc.transform(X).toarray(), exp)


def test_one_hot_encoder_warning():
    enc = OneHotEncoder()
    X = [['Male', 1], ['Female', 3]]
    np.testing.assert_no_warnings(enc.fit_transform, X)


def test_one_hot_encoder_drop_manual():
    cats_to_drop = ['def', 12, 3, 56]
    enc = OneHotEncoder(drop=cats_to_drop)
    X = [['abc', 12, 2, 55],
         ['def', 12, 1, 55],
         ['def', 12, 3, 56]]
    trans = enc.fit_transform(X).toarray()
    exp = [[1, 0, 1, 1],
           [0, 1, 0, 1],
           [0, 0, 0, 0]]
    assert_array_equal(trans, exp)
    dropped_cats = [cat[feature]
                    for cat, feature in zip(enc.categories_,
                                            enc.drop_idx_)]
    assert_array_equal(dropped_cats, cats_to_drop)
    assert_array_equal(np.array(X, dtype=object),
                       enc.inverse_transform(trans))


@pytest.mark.parametrize(
    "X_fit, params, err_msg",
    [([["Male"], ["Female"]], {'drop': 'second'},
     "Wrong input for parameter `drop`"),
     ([["Male"], ["Female"]], {'drop': 'first', 'handle_unknown': 'ignore'},
     "`handle_unknown` must be 'error'"),
     ([['abc', 2, 55], ['def', 1, 55], ['def', 3, 59]],
      {'drop': np.asarray('b', dtype=object)},
     "Wrong input for parameter `drop`"),
     ([['abc', 2, 55], ['def', 1, 55], ['def', 3, 59]],
      {'drop': ['ghi', 3, 59]},
     "The following categories were supposed")]
)
def test_one_hot_encoder_invalid_params(X_fit, params, err_msg):
    enc = OneHotEncoder(**params)
    with pytest.raises(ValueError, match=err_msg):
        enc.fit(X_fit)


@pytest.mark.parametrize('drop', [['abc', 3], ['abc', 3, 41, 'a']])
def test_invalid_drop_length(drop):
    enc = OneHotEncoder(drop=drop)
    err_msg = "`drop` should have length equal to the number"
    with pytest.raises(ValueError, match=err_msg):
        enc.fit([['abc', 2, 55], ['def', 1, 55], ['def', 3, 59]])


@pytest.mark.parametrize("density", [True, False],
                         ids=['sparse', 'dense'])
@pytest.mark.parametrize("drop", ['first',
                                  ['a', 2, 'b']],
                         ids=['first', 'manual'])
def test_categories(density, drop):
    ohe_base = OneHotEncoder(sparse=density)
    ohe_test = OneHotEncoder(sparse=density, drop=drop)
    X = [['c', 1, 'a'],
         ['a', 2, 'b']]
    ohe_base.fit(X)
    ohe_test.fit(X)
    assert_array_equal(ohe_base.categories_, ohe_test.categories_)
    if drop == 'first':
        assert_array_equal(ohe_test.drop_idx_, 0)
    else:
        for drop_cat, drop_idx, cat_list in zip(drop,
                                                ohe_test.drop_idx_,
                                                ohe_test.categories_):
            assert cat_list[drop_idx] == drop_cat
    assert isinstance(ohe_test.drop_idx_, np.ndarray)
    assert ohe_test.drop_idx_.dtype == np.int_


@pytest.mark.parametrize('Encoder', [OneHotEncoder, OrdinalEncoder])
def test_encoders_has_categorical_tags(Encoder):
    assert 'categorical' in Encoder()._get_tags()['X_types']
