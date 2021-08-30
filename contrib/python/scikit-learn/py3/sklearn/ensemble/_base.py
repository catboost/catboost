"""Base class for ensemble-based estimators."""

# Authors: Gilles Louppe
# License: BSD 3 clause

from abc import ABCMeta, abstractmethod
import numbers
import warnings

import numpy as np

from joblib import effective_n_jobs

from ..base import clone
from ..base import is_classifier, is_regressor
from ..base import BaseEstimator
from ..base import MetaEstimatorMixin
from ..utils import Bunch
from ..utils import check_random_state
from ..utils.metaestimators import _BaseComposition


def _parallel_fit_estimator(estimator, X, y, sample_weight=None):
    """Private function used to fit an estimator within a job."""
    if sample_weight is not None:
        try:
            estimator.fit(X, y, sample_weight=sample_weight)
        except TypeError as exc:
            if "unexpected keyword argument 'sample_weight'" in str(exc):
                raise TypeError(
                    "Underlying estimator {} does not support sample weights."
                    .format(estimator.__class__.__name__)
                ) from exc
            raise
    else:
        estimator.fit(X, y)
    return estimator


def _set_random_states(estimator, random_state=None):
    """Set fixed random_state parameters for an estimator.

    Finds all parameters ending ``random_state`` and sets them to integers
    derived from ``random_state``.

    Parameters
    ----------
    estimator : estimator supporting get/set_params
        Estimator with potential randomness managed by random_state
        parameters.

    random_state : int, RandomState instance or None, optional (default=None)
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.

    Notes
    -----
    This does not necessarily set *all* ``random_state`` attributes that
    control an estimator's randomness, only those accessible through
    ``estimator.get_params()``.  ``random_state``s not controlled include
    those belonging to:

        * cross-validation splitters
        * ``scipy.stats`` rvs
    """
    random_state = check_random_state(random_state)
    to_set = {}
    for key in sorted(estimator.get_params(deep=True)):
        if key == 'random_state' or key.endswith('__random_state'):
            to_set[key] = random_state.randint(np.iinfo(np.int32).max)

    if to_set:
        estimator.set_params(**to_set)


class BaseEnsemble(MetaEstimatorMixin, BaseEstimator, metaclass=ABCMeta):
    """Base class for all ensemble classes.

    Warning: This class should not be used directly. Use derived classes
    instead.

    Parameters
    ----------
    base_estimator : object, optional (default=None)
        The base estimator from which the ensemble is built.

    n_estimators : integer
        The number of estimators in the ensemble.

    estimator_params : list of strings
        The list of attributes to use as parameters when instantiating a
        new base estimator. If none are given, default parameters are used.

    Attributes
    ----------
    base_estimator_ : estimator
        The base estimator from which the ensemble is grown.

    estimators_ : list of estimators
        The collection of fitted base estimators.
    """

    # overwrite _required_parameters from MetaEstimatorMixin
    _required_parameters = []

    @abstractmethod
    def __init__(self, base_estimator, n_estimators=10,
                 estimator_params=tuple()):
        # Set parameters
        self.base_estimator = base_estimator
        self.n_estimators = n_estimators
        self.estimator_params = estimator_params

        # Don't instantiate estimators now! Parameters of base_estimator might
        # still change. Eg., when grid-searching with the nested object syntax.
        # self.estimators_ needs to be filled by the derived classes in fit.

    def _validate_estimator(self, default=None):
        """Check the estimator and the n_estimator attribute.

        Sets the base_estimator_` attributes.
        """
        if not isinstance(self.n_estimators, numbers.Integral):
            raise ValueError("n_estimators must be an integer, "
                             "got {0}.".format(type(self.n_estimators)))

        if self.n_estimators <= 0:
            raise ValueError("n_estimators must be greater than zero, "
                             "got {0}.".format(self.n_estimators))

        if self.base_estimator is not None:
            self.base_estimator_ = self.base_estimator
        else:
            self.base_estimator_ = default

        if self.base_estimator_ is None:
            raise ValueError("base_estimator cannot be None")

    def _make_estimator(self, append=True, random_state=None):
        """Make and configure a copy of the `base_estimator_` attribute.

        Warning: This method should be used to properly instantiate new
        sub-estimators.
        """
        estimator = clone(self.base_estimator_)
        estimator.set_params(**{p: getattr(self, p)
                                for p in self.estimator_params})

        if random_state is not None:
            _set_random_states(estimator, random_state)

        if append:
            self.estimators_.append(estimator)

        return estimator

    def __len__(self):
        """Return the number of estimators in the ensemble."""
        return len(self.estimators_)

    def __getitem__(self, index):
        """Return the index'th estimator in the ensemble."""
        return self.estimators_[index]

    def __iter__(self):
        """Return iterator over estimators in the ensemble."""
        return iter(self.estimators_)


def _partition_estimators(n_estimators, n_jobs):
    """Private function used to partition estimators between jobs."""
    # Compute the number of jobs
    n_jobs = min(effective_n_jobs(n_jobs), n_estimators)

    # Partition estimators between jobs
    n_estimators_per_job = np.full(n_jobs, n_estimators // n_jobs,
                                   dtype=np.int)
    n_estimators_per_job[:n_estimators % n_jobs] += 1
    starts = np.cumsum(n_estimators_per_job)

    return n_jobs, n_estimators_per_job.tolist(), [0] + starts.tolist()


class _BaseHeterogeneousEnsemble(MetaEstimatorMixin, _BaseComposition,
                                 metaclass=ABCMeta):
    """Base class for heterogeneous ensemble of learners.

    Parameters
    ----------
    estimators : list of (str, estimator) tuples
        The ensemble of estimators to use in the ensemble. Each element of the
        list is defined as a tuple of string (i.e. name of the estimator) and
        an estimator instance. An estimator can be set to `'drop'` using
        `set_params`.

    Attributes
    ----------
    estimators_ : list of estimators
        The elements of the estimators parameter, having been fitted on the
        training data. If an estimator has been set to `'drop'`, it will not
        appear in `estimators_`.
    """

    _required_parameters = ['estimators']

    @property
    def named_estimators(self):
        return Bunch(**dict(self.estimators))

    @abstractmethod
    def __init__(self, estimators):
        self.estimators = estimators

    def _validate_estimators(self):
        if self.estimators is None or len(self.estimators) == 0:
            raise ValueError(
                "Invalid 'estimators' attribute, 'estimators' should be a list"
                " of (string, estimator) tuples."
            )
        names, estimators = zip(*self.estimators)
        # defined by MetaEstimatorMixin
        self._validate_names(names)

        # FIXME: deprecate the usage of None to drop an estimator from the
        # ensemble. Remove in 0.24
        if any(est is None for est in estimators):
            warnings.warn(
                "Using 'None' to drop an estimator from the ensemble is "
                "deprecated in 0.22 and support will be dropped in 0.24. "
                "Use the string 'drop' instead.", FutureWarning
            )

        has_estimator = any(est not in (None, 'drop') for est in estimators)
        if not has_estimator:
            raise ValueError(
                "All estimators are dropped. At least one is required "
                "to be an estimator."
            )

        is_estimator_type = (is_classifier if is_classifier(self)
                             else is_regressor)

        for est in estimators:
            if est not in (None, 'drop') and not is_estimator_type(est):
                raise ValueError(
                    "The estimator {} should be a {}.".format(
                        est.__class__.__name__, is_estimator_type.__name__[3:]
                    )
                )

        return names, estimators

    def set_params(self, **params):
        """
        Set the parameters of an estimator from the ensemble.

        Valid parameter keys can be listed with `get_params()`.

        Parameters
        ----------
        **params : keyword arguments
            Specific parameters using e.g.
            `set_params(parameter_name=new_value)`. In addition, to setting the
            parameters of the stacking estimator, the individual estimator of
            the stacking estimators can also be set, or can be removed by
            setting them to 'drop'.
        """
        super()._set_params('estimators', **params)
        return self

    def get_params(self, deep=True):
        """
        Get the parameters of an estimator from the ensemble.

        Parameters
        ----------
        deep : bool
            Setting it to True gets the various classifiers and the parameters
            of the classifiers as well.
        """
        return super()._get_params('estimators', deep=deep)
