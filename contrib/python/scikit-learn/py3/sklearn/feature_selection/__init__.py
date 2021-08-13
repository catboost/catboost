"""
The :mod:`sklearn.feature_selection` module implements feature selection
algorithms. It currently includes univariate filter selection methods and the
recursive feature elimination algorithm.
"""

from ._univariate_selection import chi2
from ._univariate_selection import f_classif
from ._univariate_selection import f_oneway
from ._univariate_selection import f_regression
from ._univariate_selection import SelectPercentile
from ._univariate_selection import SelectKBest
from ._univariate_selection import SelectFpr
from ._univariate_selection import SelectFdr
from ._univariate_selection import SelectFwe
from ._univariate_selection import GenericUnivariateSelect

from ._variance_threshold import VarianceThreshold

from ._rfe import RFE
from ._rfe import RFECV

from ._from_model import SelectFromModel

from ._mutual_info import mutual_info_regression, mutual_info_classif


__all__ = ['GenericUnivariateSelect',
           'RFE',
           'RFECV',
           'SelectFdr',
           'SelectFpr',
           'SelectFwe',
           'SelectKBest',
           'SelectFromModel',
           'SelectPercentile',
           'VarianceThreshold',
           'chi2',
           'f_classif',
           'f_oneway',
           'f_regression',
           'mutual_info_classif',
           'mutual_info_regression']
