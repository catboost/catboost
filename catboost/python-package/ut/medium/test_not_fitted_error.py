import pickle
import subprocess
import sys
import textwrap

import pytest

from catboost import CatBoost, CatBoostClassifier, CatBoostError, CatBoostRanker, CatBoostRegressor


@pytest.mark.parametrize('estimator_class', [CatBoost, CatBoostClassifier, CatBoostRegressor, CatBoostRanker])
@pytest.mark.parametrize('method', ['predict', 'staged_predict'])
def test_prediction_before_fit(estimator_class, method):
    NotFittedError = pytest.importorskip('sklearn.exceptions').NotFittedError
    with pytest.raises(NotFittedError, match='There is no trained model') as error:
        result = getattr(estimator_class(), method)([[0], [1]])
        if method.startswith('staged_'):
            next(result)
    assert isinstance(error.value, CatBoostError)


@pytest.mark.parametrize('method', ['predict_proba', 'predict_log_proba', 'staged_predict_proba', 'staged_predict_log_proba'])
def test_probability_prediction_before_fit(method):
    NotFittedError = pytest.importorskip('sklearn.exceptions').NotFittedError
    with pytest.raises(NotFittedError, match='There is no trained model') as error:
        result = getattr(CatBoostClassifier(), method)([[0], [1]])
        if method.startswith('staged_'):
            next(result)
    assert isinstance(error.value, CatBoostError)


@pytest.mark.parametrize('estimator_class', [CatBoostClassifier, CatBoostRegressor, CatBoostRanker])
def test_score_before_fit(estimator_class):
    NotFittedError = pytest.importorskip('sklearn.exceptions').NotFittedError
    kwargs = {'group_id': [0, 0]} if estimator_class is CatBoostRanker else {}
    with pytest.raises(NotFittedError) as error:
        estimator_class().score([[0], [1]], [0, 1], **kwargs)
    assert isinstance(error.value, CatBoostError)


@pytest.mark.parametrize('estimator_class', [CatBoostClassifier, CatBoostRegressor])
def test_sklearn_unfitted_estimator_check(estimator_class):
    estimator_checks = pytest.importorskip('sklearn.utils.estimator_checks')
    estimator_checks.check_estimators_unfitted(estimator_class.__name__, estimator_class())


def test_not_fitted_error_pickle():
    NotFittedError = pytest.importorskip('sklearn.exceptions').NotFittedError
    with pytest.raises(NotFittedError) as error:
        CatBoostClassifier().predict([[0], [1]])
    restored = pickle.loads(pickle.dumps(error.value))
    assert isinstance(restored, type(error.value))
    assert isinstance(restored, CatBoostError)
    assert isinstance(restored, NotFittedError)
    assert str(restored) == str(error.value)


@pytest.mark.parametrize('estimator_class', [CatBoostClassifier, CatBoostRegressor])
def test_fitted_prediction_errors_are_not_not_fitted_errors(estimator_class):
    NotFittedError = pytest.importorskip('sklearn.exceptions').NotFittedError
    model = estimator_class(iterations=2, depth=2, thread_count=1, verbose=False, allow_writing_files=False)
    model.fit([[0, 0], [1, 1], [2, 0], [3, 1]], [0, 1, 0, 1])
    assert len(model.predict([[0, 0], [1, 1]])) == 2
    with pytest.raises(CatBoostError) as error:
        model.predict([['not a number', 0]])
    assert not isinstance(error.value, NotFittedError)


def test_not_fitted_error_without_sklearn():
    # A fresh interpreter also catches accidental mandatory imports of sklearn.
    code = textwrap.dedent('''
        import importlib.abc
        import sys

        class BlockSklearn(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path=None, target=None):
                if fullname == 'sklearn' or fullname.startswith('sklearn.'):
                    raise ModuleNotFoundError("No module named 'sklearn'", name=fullname)

        sys.meta_path.insert(0, BlockSklearn())
        from catboost import CatBoostClassifier, CatBoostError

        model = CatBoostClassifier(iterations=2, thread_count=1, verbose=False, allow_writing_files=False)
        try:
            model.predict([[0], [1]])
        except CatBoostError as error:
            assert 'There is no trained model' in str(error)
        else:
            raise AssertionError('Unfitted prediction did not raise')

        model.fit([[0], [1], [2], [3]], [0, 1, 0, 1])
        assert len(model.predict([[0], [1]])) == 2
        assert 'sklearn' not in sys.modules
    ''')
    subprocess.run([sys.executable, '-c', code], check=True, capture_output=True, text=True, timeout=60)


def test_fitted_prediction_does_not_import_sklearn():
    code = textwrap.dedent('''
        import importlib.abc
        import sys

        class RejectSklearnImport(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path=None, target=None):
                if fullname == 'sklearn' or fullname.startswith('sklearn.'):
                    raise AssertionError('CatBoost attempted to import sklearn')

        sys.meta_path.insert(0, RejectSklearnImport())
        from catboost import CatBoostClassifier, CatBoostRegressor

        for estimator_class in (CatBoostClassifier, CatBoostRegressor):
            model = estimator_class(iterations=2, thread_count=1, verbose=False, allow_writing_files=False)
            model.fit([[0], [1], [2], [3]], [0, 1, 0, 1])
            assert len(model.predict([[0], [1]])) == 2
            model.score([[0], [1]], [0, 1])
        assert 'sklearn' not in sys.modules
    ''')
    subprocess.run([sys.executable, '-c', code], check=True, capture_output=True, text=True, timeout=60)


def test_unfitted_prediction_before_sklearn_import():
    pytest.importorskip('sklearn.exceptions')
    code = textwrap.dedent('''
        import sys
        from catboost import CatBoostClassifier, CatBoostError

        assert 'sklearn' not in sys.modules
        try:
            CatBoostClassifier().predict([[0], [1]])
        except CatBoostError as error:
            from sklearn.exceptions import NotFittedError
            assert isinstance(error, NotFittedError)
        else:
            raise AssertionError('Unfitted prediction did not raise')
    ''')
    subprocess.run([sys.executable, '-c', code], check=True, capture_output=True, text=True, timeout=60)
