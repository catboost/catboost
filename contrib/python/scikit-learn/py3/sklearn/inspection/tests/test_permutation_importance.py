import pytest
import numpy as np

from numpy.testing import assert_allclose

from sklearn.compose import ColumnTransformer
from sklearn.datasets import load_boston
from sklearn.datasets import load_iris
from sklearn.datasets import make_classification
from sklearn.datasets import make_regression
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import scale
from sklearn.utils import parallel_backend
from sklearn.utils._testing import _convert_container


@pytest.mark.parametrize("n_jobs", [1, 2])
def test_permutation_importance_correlated_feature_regression(n_jobs):
    # Make sure that feature highly correlated to the target have a higher
    # importance
    rng = np.random.RandomState(42)
    n_repeats = 5

    X, y = load_boston(return_X_y=True)
    y_with_little_noise = (
        y + rng.normal(scale=0.001, size=y.shape[0])).reshape(-1, 1)

    X = np.hstack([X, y_with_little_noise])

    clf = RandomForestRegressor(n_estimators=10, random_state=42)
    clf.fit(X, y)

    result = permutation_importance(clf, X, y, n_repeats=n_repeats,
                                    random_state=rng, n_jobs=n_jobs)

    assert result.importances.shape == (X.shape[1], n_repeats)

    # the correlated feature with y was added as the last column and should
    # have the highest importance
    assert np.all(result.importances_mean[-1] >
                  result.importances_mean[:-1])


@pytest.mark.parametrize("n_jobs", [1, 2])
def test_permutation_importance_correlated_feature_regression_pandas(n_jobs):
    pd = pytest.importorskip("pandas")

    # Make sure that feature highly correlated to the target have a higher
    # importance
    rng = np.random.RandomState(42)
    n_repeats = 5

    dataset = load_iris()
    X, y = dataset.data, dataset.target
    y_with_little_noise = (
        y + rng.normal(scale=0.001, size=y.shape[0])).reshape(-1, 1)

    # Adds feature correlated with y as the last column
    X = pd.DataFrame(X, columns=dataset.feature_names)
    X['correlated_feature'] = y_with_little_noise

    clf = RandomForestClassifier(n_estimators=10, random_state=42)
    clf.fit(X, y)

    result = permutation_importance(clf, X, y, n_repeats=n_repeats,
                                    random_state=rng, n_jobs=n_jobs)

    assert result.importances.shape == (X.shape[1], n_repeats)

    # the correlated feature with y was added as the last column and should
    # have the highest importance
    assert np.all(result.importances_mean[-1] > result.importances_mean[:-1])


def test_permutation_importance_mixed_types():
    rng = np.random.RandomState(42)
    n_repeats = 4

    # Last column is correlated with y
    X = np.array([[1.0, 2.0, 3.0, np.nan], [2, 1, 2, 1]]).T
    y = np.array([0, 1, 0, 1])

    clf = make_pipeline(SimpleImputer(), LogisticRegression(solver='lbfgs'))
    clf.fit(X, y)
    result = permutation_importance(clf, X, y, n_repeats=n_repeats,
                                    random_state=rng)

    assert result.importances.shape == (X.shape[1], n_repeats)

    # the correlated feature with y is the last column and should
    # have the highest importance
    assert np.all(result.importances_mean[-1] > result.importances_mean[:-1])

    # use another random state
    rng = np.random.RandomState(0)
    result2 = permutation_importance(clf, X, y, n_repeats=n_repeats,
                                     random_state=rng)
    assert result2.importances.shape == (X.shape[1], n_repeats)

    assert not np.allclose(result.importances, result2.importances)

    # the correlated feature with y is the last column and should
    # have the highest importance
    assert np.all(result2.importances_mean[-1] > result2.importances_mean[:-1])


def test_permutation_importance_mixed_types_pandas():
    pd = pytest.importorskip("pandas")
    rng = np.random.RandomState(42)
    n_repeats = 5

    # Last column is correlated with y
    X = pd.DataFrame({'col1': [1.0, 2.0, 3.0, np.nan],
                      'col2': ['a', 'b', 'a', 'b']})
    y = np.array([0, 1, 0, 1])

    num_preprocess = make_pipeline(SimpleImputer(), StandardScaler())
    preprocess = ColumnTransformer([
        ('num', num_preprocess, ['col1']),
        ('cat', OneHotEncoder(), ['col2'])
    ])
    clf = make_pipeline(preprocess, LogisticRegression(solver='lbfgs'))
    clf.fit(X, y)

    result = permutation_importance(clf, X, y, n_repeats=n_repeats,
                                    random_state=rng)

    assert result.importances.shape == (X.shape[1], n_repeats)
    # the correlated feature with y is the last column and should
    # have the highest importance
    assert np.all(result.importances_mean[-1] > result.importances_mean[:-1])


def test_permutation_importance_linear_regresssion():
    X, y = make_regression(n_samples=500, n_features=10, random_state=0)

    X = scale(X)
    y = scale(y)

    lr = LinearRegression().fit(X, y)

    # this relationship can be computed in closed form
    expected_importances = 2 * lr.coef_**2
    results = permutation_importance(lr, X, y,
                                     n_repeats=50,
                                     scoring='neg_mean_squared_error')
    assert_allclose(expected_importances, results.importances_mean,
                    rtol=1e-1, atol=1e-6)


def test_permutation_importance_equivalence_sequential_parallel():
    # regression test to make sure that sequential and parallel calls will
    # output the same results.
    X, y = make_regression(n_samples=500, n_features=10, random_state=0)
    lr = LinearRegression().fit(X, y)

    importance_sequential = permutation_importance(
        lr, X, y, n_repeats=5, random_state=0, n_jobs=1
    )

    # First check that the problem is structured enough and that the model is
    # complex enough to not yield trivial, constant importances:
    imp_min = importance_sequential['importances'].min()
    imp_max = importance_sequential['importances'].max()
    assert imp_max - imp_min > 0.3

    # The actually check that parallelism does not impact the results
    # either with shared memory (threading) or without isolated memory
    # via process-based parallelism using the default backend
    # ('loky' or 'multiprocessing') depending on the joblib version:

    # process-based parallelism (by default):
    importance_processes = permutation_importance(
        lr, X, y, n_repeats=5, random_state=0, n_jobs=2)
    assert_allclose(
        importance_processes['importances'],
        importance_sequential['importances']
    )

    # thread-based parallelism:
    with parallel_backend("threading"):
        importance_threading = permutation_importance(
            lr, X, y, n_repeats=5, random_state=0, n_jobs=2
        )
    assert_allclose(
        importance_threading['importances'],
        importance_sequential['importances']
    )


@pytest.mark.parametrize("n_jobs", [None, 1, 2])
def test_permutation_importance_equivalence_array_dataframe(n_jobs):
    # This test checks that the column shuffling logic has the same behavior
    # both a dataframe and a simple numpy array.
    pd = pytest.importorskip('pandas')

    # regression test to make sure that sequential and parallel calls will
    # output the same results.
    X, y = make_regression(n_samples=100, n_features=5, random_state=0)
    X_df = pd.DataFrame(X)

    # Add a categorical feature that is statistically linked to y:
    binner = KBinsDiscretizer(n_bins=3, encode="ordinal")
    cat_column = binner.fit_transform(y.reshape(-1, 1))

    # Concatenate the extra column to the numpy array: integers will be
    # cast to float values
    X = np.hstack([X, cat_column])
    assert X.dtype.kind == "f"

    # Insert extra column as a non-numpy-native dtype (while keeping backward
    # compat for old pandas versions):
    if hasattr(pd, "Categorical"):
        cat_column = pd.Categorical(cat_column.ravel())
    else:
        cat_column = cat_column.ravel()
    new_col_idx = len(X_df.columns)
    X_df[new_col_idx] = cat_column
    assert X_df[new_col_idx].dtype == cat_column.dtype

    # Stich an aribtrary index to the dataframe:
    X_df.index = np.arange(len(X_df)).astype(str)

    rf = RandomForestRegressor(n_estimators=5, max_depth=3, random_state=0)
    rf.fit(X, y)

    n_repeats = 3
    importance_array = permutation_importance(
        rf, X, y, n_repeats=n_repeats, random_state=0, n_jobs=n_jobs
    )

    # First check that the problem is structured enough and that the model is
    # complex enough to not yield trivial, constant importances:
    imp_min = importance_array['importances'].min()
    imp_max = importance_array['importances'].max()
    assert imp_max - imp_min > 0.3

    # Now check that importances computed on dataframe matche the values
    # of those computed on the array with the same data.
    importance_dataframe = permutation_importance(
        rf, X_df, y, n_repeats=n_repeats, random_state=0, n_jobs=n_jobs
    )
    assert_allclose(
        importance_array['importances'],
        importance_dataframe['importances']
    )


@pytest.mark.parametrize("input_type", ["array", "dataframe"])
def test_permutation_importance_large_memmaped_data(input_type):
    # Smoke, non-regression test for:
    # https://github.com/scikit-learn/scikit-learn/issues/15810
    n_samples, n_features = int(5e4), 4
    X, y = make_classification(n_samples=n_samples, n_features=n_features,
                               random_state=0)
    assert X.nbytes > 1e6  # trigger joblib memmaping

    X = _convert_container(X, input_type)
    clf = DummyClassifier(strategy='prior').fit(X, y)

    # Actual smoke test: should not raise any error:
    n_repeats = 5
    r = permutation_importance(clf, X, y, n_repeats=n_repeats, n_jobs=2)

    # Auxiliary check: DummyClassifier is feature independent:
    # permutating feature should not change the predictions
    expected_importances = np.zeros((n_features, n_repeats))
    assert_allclose(expected_importances, r.importances)
