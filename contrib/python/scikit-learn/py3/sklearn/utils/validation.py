"""Utilities for input validation"""

# Authors: Olivier Grisel
#          Gael Varoquaux
#          Andreas Mueller
#          Lars Buitinck
#          Alexandre Gramfort
#          Nicolas Tresegnie
#          Sylvain Marie
# License: BSD 3 clause

from functools import wraps
import warnings
import numbers

import numpy as np
import scipy.sparse as sp
from distutils.version import LooseVersion
from inspect import signature, isclass, Parameter

from numpy.core.numeric import ComplexWarning
import joblib

from .fixes import _object_dtype_isnan
from .. import get_config as _get_config
from ..exceptions import NonBLASDotWarning, PositiveSpectrumWarning
from ..exceptions import NotFittedError
from ..exceptions import DataConversionWarning

FLOAT_DTYPES = (np.float64, np.float32, np.float16)

# Silenced by default to reduce verbosity. Turn on at runtime for
# performance profiling.
warnings.simplefilter('ignore', NonBLASDotWarning)


def _assert_all_finite(X, allow_nan=False, msg_dtype=None):
    """Like assert_all_finite, but only for ndarray."""
    # validation is also imported in extmath
    from .extmath import _safe_accumulator_op

    if _get_config()['assume_finite']:
        return
    X = np.asanyarray(X)
    # First try an O(n) time, O(1) space solution for the common case that
    # everything is finite; fall back to O(n) space np.isfinite to prevent
    # false positives from overflow in sum method. The sum is also calculated
    # safely to reduce dtype induced overflows.
    is_float = X.dtype.kind in 'fc'
    if is_float and (np.isfinite(_safe_accumulator_op(np.sum, X))):
        pass
    elif is_float:
        msg_err = "Input contains {} or a value too large for {!r}."
        if (allow_nan and np.isinf(X).any() or
                not allow_nan and not np.isfinite(X).all()):
            type_err = 'infinity' if allow_nan else 'NaN, infinity'
            raise ValueError(
                    msg_err.format
                    (type_err,
                     msg_dtype if msg_dtype is not None else X.dtype)
            )
    # for object dtype data, we only check for NaNs (GH-13254)
    elif X.dtype == np.dtype('object') and not allow_nan:
        if _object_dtype_isnan(X).any():
            raise ValueError("Input contains NaN")


def assert_all_finite(X, allow_nan=False):
    """Throw a ValueError if X contains NaN or infinity.

    Parameters
    ----------
    X : array or sparse matrix

    allow_nan : bool
    """
    _assert_all_finite(X.data if sp.issparse(X) else X, allow_nan)


def as_float_array(X, copy=True, force_all_finite=True):
    """Converts an array-like to an array of floats.

    The new dtype will be np.float32 or np.float64, depending on the original
    type. The function can create a copy or modify the argument depending
    on the argument copy.

    Parameters
    ----------
    X : {array-like, sparse matrix}

    copy : bool, optional
        If True, a copy of X will be created. If False, a copy may still be
        returned if X's dtype is not a floating point type.

    force_all_finite : boolean or 'allow-nan', (default=True)
        Whether to raise an error on np.inf and np.nan in X. The possibilities
        are:

        - True: Force all values of X to be finite.
        - False: accept both np.inf and np.nan in X.
        - 'allow-nan': accept only np.nan values in X. Values cannot be
          infinite.

        .. versionadded:: 0.20
           ``force_all_finite`` accepts the string ``'allow-nan'``.

    Returns
    -------
    XT : {array, sparse matrix}
        An array of type np.float
    """
    if isinstance(X, np.matrix) or (not isinstance(X, np.ndarray)
                                    and not sp.issparse(X)):
        return check_array(X, ['csr', 'csc', 'coo'], dtype=np.float64,
                           copy=copy, force_all_finite=force_all_finite,
                           ensure_2d=False)
    elif sp.issparse(X) and X.dtype in [np.float32, np.float64]:
        return X.copy() if copy else X
    elif X.dtype in [np.float32, np.float64]:  # is numpy array
        return X.copy('F' if X.flags['F_CONTIGUOUS'] else 'C') if copy else X
    else:
        if X.dtype.kind in 'uib' and X.dtype.itemsize <= 4:
            return_dtype = np.float32
        else:
            return_dtype = np.float64
        return X.astype(return_dtype)


def _is_arraylike(x):
    """Returns whether the input is array-like"""
    return (hasattr(x, '__len__') or
            hasattr(x, 'shape') or
            hasattr(x, '__array__'))


def _num_samples(x):
    """Return number of samples in array-like x."""
    message = 'Expected sequence or array-like, got %s' % type(x)
    if hasattr(x, 'fit') and callable(x.fit):
        # Don't get num_samples from an ensembles length!
        raise TypeError(message)

    if not hasattr(x, '__len__') and not hasattr(x, 'shape'):
        if hasattr(x, '__array__'):
            x = np.asarray(x)
        else:
            raise TypeError(message)

    if hasattr(x, 'shape') and x.shape is not None:
        if len(x.shape) == 0:
            raise TypeError("Singleton array %r cannot be considered"
                            " a valid collection." % x)
        # Check that shape is returning an integer or default to len
        # Dask dataframes may not return numeric shape[0] value
        if isinstance(x.shape[0], numbers.Integral):
            return x.shape[0]

    try:
        return len(x)
    except TypeError:
        raise TypeError(message)


def check_memory(memory):
    """Check that ``memory`` is joblib.Memory-like.

    joblib.Memory-like means that ``memory`` can be converted into a
    joblib.Memory instance (typically a str denoting the ``location``)
    or has the same interface (has a ``cache`` method).

    Parameters
    ----------
    memory : None, str or object with the joblib.Memory interface

    Returns
    -------
    memory : object with the joblib.Memory interface

    Raises
    ------
    ValueError
        If ``memory`` is not joblib.Memory-like.
    """

    if memory is None or isinstance(memory, str):
        if LooseVersion(joblib.__version__) < '0.12':
            memory = joblib.Memory(cachedir=memory, verbose=0)
        else:
            memory = joblib.Memory(location=memory, verbose=0)
    elif not hasattr(memory, 'cache'):
        raise ValueError("'memory' should be None, a string or have the same"
                         " interface as joblib.Memory."
                         " Got memory='{}' instead.".format(memory))
    return memory


def check_consistent_length(*arrays):
    """Check that all arrays have consistent first dimensions.

    Checks whether all objects in arrays have the same shape or length.

    Parameters
    ----------
    *arrays : list or tuple of input objects.
        Objects that will be checked for consistent length.
    """

    lengths = [_num_samples(X) for X in arrays if X is not None]
    uniques = np.unique(lengths)
    if len(uniques) > 1:
        raise ValueError("Found input variables with inconsistent numbers of"
                         " samples: %r" % [int(l) for l in lengths])


def _make_indexable(iterable):
    """Ensure iterable supports indexing or convert to an indexable variant.

    Convert sparse matrices to csr and other non-indexable iterable to arrays.
    Let `None` and indexable objects (e.g. pandas dataframes) pass unchanged.

    Parameters
    ----------
    iterable : {list, dataframe, array, sparse} or None
        Object to be converted to an indexable iterable.
    """
    if sp.issparse(iterable):
        return iterable.tocsr()
    elif hasattr(iterable, "__getitem__") or hasattr(iterable, "iloc"):
        return iterable
    elif iterable is None:
        return iterable
    return np.array(iterable)


def indexable(*iterables):
    """Make arrays indexable for cross-validation.

    Checks consistent length, passes through None, and ensures that everything
    can be indexed by converting sparse matrices to csr and converting
    non-interable objects to arrays.

    Parameters
    ----------
    *iterables : lists, dataframes, arrays, sparse matrices
        List of objects to ensure sliceability.
    """
    result = [_make_indexable(X) for X in iterables]
    check_consistent_length(*result)
    return result


def _ensure_sparse_format(spmatrix, accept_sparse, dtype, copy,
                          force_all_finite, accept_large_sparse):
    """Convert a sparse matrix to a given format.

    Checks the sparse format of spmatrix and converts if necessary.

    Parameters
    ----------
    spmatrix : scipy sparse matrix
        Input to validate and convert.

    accept_sparse : string, boolean or list/tuple of strings
        String[s] representing allowed sparse matrix formats ('csc',
        'csr', 'coo', 'dok', 'bsr', 'lil', 'dia'). If the input is sparse but
        not in the allowed format, it will be converted to the first listed
        format. True allows the input to be any format. False means
        that a sparse matrix input will raise an error.

    dtype : string, type or None
        Data type of result. If None, the dtype of the input is preserved.

    copy : boolean
        Whether a forced copy will be triggered. If copy=False, a copy might
        be triggered by a conversion.

    force_all_finite : boolean or 'allow-nan', (default=True)
        Whether to raise an error on np.inf and np.nan in X. The possibilities
        are:

        - True: Force all values of X to be finite.
        - False: accept both np.inf and np.nan in X.
        - 'allow-nan': accept only np.nan values in X. Values cannot be
          infinite.

        .. versionadded:: 0.20
           ``force_all_finite`` accepts the string ``'allow-nan'``.

    Returns
    -------
    spmatrix_converted : scipy sparse matrix.
        Matrix that is ensured to have an allowed type.
    """
    if dtype is None:
        dtype = spmatrix.dtype

    changed_format = False

    if isinstance(accept_sparse, str):
        accept_sparse = [accept_sparse]

    # Indices dtype validation
    _check_large_sparse(spmatrix, accept_large_sparse)

    if accept_sparse is False:
        raise TypeError('A sparse matrix was passed, but dense '
                        'data is required. Use X.toarray() to '
                        'convert to a dense numpy array.')
    elif isinstance(accept_sparse, (list, tuple)):
        if len(accept_sparse) == 0:
            raise ValueError("When providing 'accept_sparse' "
                             "as a tuple or list, it must contain at "
                             "least one string value.")
        # ensure correct sparse format
        if spmatrix.format not in accept_sparse:
            # create new with correct sparse
            spmatrix = spmatrix.asformat(accept_sparse[0])
            changed_format = True
    elif accept_sparse is not True:
        # any other type
        raise ValueError("Parameter 'accept_sparse' should be a string, "
                         "boolean or list of strings. You provided "
                         "'accept_sparse={}'.".format(accept_sparse))

    if dtype != spmatrix.dtype:
        # convert dtype
        spmatrix = spmatrix.astype(dtype)
    elif copy and not changed_format:
        # force copy
        spmatrix = spmatrix.copy()

    if force_all_finite:
        if not hasattr(spmatrix, "data"):
            warnings.warn("Can't check %s sparse matrix for nan or inf."
                          % spmatrix.format, stacklevel=2)
        else:
            _assert_all_finite(spmatrix.data,
                               allow_nan=force_all_finite == 'allow-nan')

    return spmatrix


def _ensure_no_complex_data(array):
    if hasattr(array, 'dtype') and array.dtype is not None \
            and hasattr(array.dtype, 'kind') and array.dtype.kind == "c":
        raise ValueError("Complex data not supported\n"
                         "{}\n".format(array))


def check_array(array, accept_sparse=False, accept_large_sparse=True,
                dtype="numeric", order=None, copy=False, force_all_finite=True,
                ensure_2d=True, allow_nd=False, ensure_min_samples=1,
                ensure_min_features=1, warn_on_dtype=None, estimator=None):

    """Input validation on an array, list, sparse matrix or similar.

    By default, the input is checked to be a non-empty 2D array containing
    only finite values. If the dtype of the array is object, attempt
    converting to float, raising on failure.

    Parameters
    ----------
    array : object
        Input object to check / convert.

    accept_sparse : string, boolean or list/tuple of strings (default=False)
        String[s] representing allowed sparse matrix formats, such as 'csc',
        'csr', etc. If the input is sparse but not in the allowed format,
        it will be converted to the first listed format. True allows the input
        to be any format. False means that a sparse matrix input will
        raise an error.

    accept_large_sparse : bool (default=True)
        If a CSR, CSC, COO or BSR sparse matrix is supplied and accepted by
        accept_sparse, accept_large_sparse=False will cause it to be accepted
        only if its indices are stored with a 32-bit dtype.

        .. versionadded:: 0.20

    dtype : string, type, list of types or None (default="numeric")
        Data type of result. If None, the dtype of the input is preserved.
        If "numeric", dtype is preserved unless array.dtype is object.
        If dtype is a list of types, conversion on the first type is only
        performed if the dtype of the input is not in the list.

    order : 'F', 'C' or None (default=None)
        Whether an array will be forced to be fortran or c-style.
        When order is None (default), then if copy=False, nothing is ensured
        about the memory layout of the output array; otherwise (copy=True)
        the memory layout of the returned array is kept as close as possible
        to the original array.

    copy : boolean (default=False)
        Whether a forced copy will be triggered. If copy=False, a copy might
        be triggered by a conversion.

    force_all_finite : boolean or 'allow-nan', (default=True)
        Whether to raise an error on np.inf and np.nan in array. The
        possibilities are:

        - True: Force all values of array to be finite.
        - False: accept both np.inf and np.nan in array.
        - 'allow-nan': accept only np.nan values in array. Values cannot
          be infinite.

        For object dtyped data, only np.nan is checked and not np.inf.

        .. versionadded:: 0.20
           ``force_all_finite`` accepts the string ``'allow-nan'``.

    ensure_2d : boolean (default=True)
        Whether to raise a value error if array is not 2D.

    allow_nd : boolean (default=False)
        Whether to allow array.ndim > 2.

    ensure_min_samples : int (default=1)
        Make sure that the array has a minimum number of samples in its first
        axis (rows for a 2D array). Setting to 0 disables this check.

    ensure_min_features : int (default=1)
        Make sure that the 2D array has some minimum number of features
        (columns). The default value of 1 rejects empty datasets.
        This check is only enforced when the input data has effectively 2
        dimensions or is originally 1D and ``ensure_2d`` is True. Setting to 0
        disables this check.

    warn_on_dtype : boolean or None, optional (default=None)
        Raise DataConversionWarning if the dtype of the input data structure
        does not match the requested dtype, causing a memory copy.

        .. deprecated:: 0.21
            ``warn_on_dtype`` is deprecated in version 0.21 and will be
            removed in 0.23.

    estimator : str or estimator instance (default=None)
        If passed, include the name of the estimator in warning messages.

    Returns
    -------
    array_converted : object
        The converted and validated array.
    """
    # warn_on_dtype deprecation
    if warn_on_dtype is not None:
        warnings.warn(
            "'warn_on_dtype' is deprecated in version 0.21 and will be "
            "removed in 0.23. Don't set `warn_on_dtype` to remove this "
            "warning.",
            FutureWarning, stacklevel=2)

    # store reference to original array to check if copy is needed when
    # function returns
    array_orig = array

    # store whether originally we wanted numeric dtype
    dtype_numeric = isinstance(dtype, str) and dtype == "numeric"

    dtype_orig = getattr(array, "dtype", None)
    if not hasattr(dtype_orig, 'kind'):
        # not a data type (e.g. a column named dtype in a pandas DataFrame)
        dtype_orig = None

    # check if the object contains several dtypes (typically a pandas
    # DataFrame), and store them. If not, store None.
    dtypes_orig = None
    if hasattr(array, "dtypes") and hasattr(array.dtypes, '__array__'):
        dtypes_orig = list(array.dtypes)
        # pandas boolean dtype __array__ interface coerces bools to objects
        for i, dtype_iter in enumerate(dtypes_orig):
            if dtype_iter.kind == 'b':
                dtypes_orig[i] = np.object

        if all(isinstance(dtype, np.dtype) for dtype in dtypes_orig):
            dtype_orig = np.result_type(*dtypes_orig)

    if dtype_numeric:
        if dtype_orig is not None and dtype_orig.kind == "O":
            # if input is object, convert to float.
            dtype = np.float64
        else:
            dtype = None

    if isinstance(dtype, (list, tuple)):
        if dtype_orig is not None and dtype_orig in dtype:
            # no dtype conversion required
            dtype = None
        else:
            # dtype conversion required. Let's select the first element of the
            # list of accepted types.
            dtype = dtype[0]

    if force_all_finite not in (True, False, 'allow-nan'):
        raise ValueError('force_all_finite should be a bool or "allow-nan"'
                         '. Got {!r} instead'.format(force_all_finite))

    if estimator is not None:
        if isinstance(estimator, str):
            estimator_name = estimator
        else:
            estimator_name = estimator.__class__.__name__
    else:
        estimator_name = "Estimator"
    context = " by %s" % estimator_name if estimator is not None else ""

    if sp.issparse(array):
        _ensure_no_complex_data(array)
        array = _ensure_sparse_format(array, accept_sparse=accept_sparse,
                                      dtype=dtype, copy=copy,
                                      force_all_finite=force_all_finite,
                                      accept_large_sparse=accept_large_sparse)
    else:
        # If np.array(..) gives ComplexWarning, then we convert the warning
        # to an error. This is needed because specifying a non complex
        # dtype to the function converts complex to real dtype,
        # thereby passing the test made in the lines following the scope
        # of warnings context manager.
        with warnings.catch_warnings():
            try:
                warnings.simplefilter('error', ComplexWarning)
                if dtype is not None and np.dtype(dtype).kind in 'iu':
                    # Conversion float -> int should not contain NaN or
                    # inf (numpy#14412). We cannot use casting='safe' because
                    # then conversion float -> int would be disallowed.
                    array = np.asarray(array, order=order)
                    if array.dtype.kind == 'f':
                        _assert_all_finite(array, allow_nan=False,
                                           msg_dtype=dtype)
                    array = array.astype(dtype, casting="unsafe", copy=False)
                else:
                    array = np.asarray(array, order=order, dtype=dtype)
            except ComplexWarning:
                raise ValueError("Complex data not supported\n"
                                 "{}\n".format(array))

        # It is possible that the np.array(..) gave no warning. This happens
        # when no dtype conversion happened, for example dtype = None. The
        # result is that np.array(..) produces an array of complex dtype
        # and we need to catch and raise exception for such cases.
        _ensure_no_complex_data(array)

        if ensure_2d:
            # If input is scalar raise error
            if array.ndim == 0:
                raise ValueError(
                    "Expected 2D array, got scalar array instead:\narray={}.\n"
                    "Reshape your data either using array.reshape(-1, 1) if "
                    "your data has a single feature or array.reshape(1, -1) "
                    "if it contains a single sample.".format(array))
            # If input is 1D raise error
            if array.ndim == 1:
                raise ValueError(
                    "Expected 2D array, got 1D array instead:\narray={}.\n"
                    "Reshape your data either using array.reshape(-1, 1) if "
                    "your data has a single feature or array.reshape(1, -1) "
                    "if it contains a single sample.".format(array))

        # in the future np.flexible dtypes will be handled like object dtypes
        if dtype_numeric and np.issubdtype(array.dtype, np.flexible):
            warnings.warn(
                "Beginning in version 0.22, arrays of bytes/strings will be "
                "converted to decimal numbers if dtype='numeric'. "
                "It is recommended that you convert the array to "
                "a float dtype before using it in scikit-learn, "
                "for example by using "
                "your_array = your_array.astype(np.float64).",
                FutureWarning, stacklevel=2)

        # make sure we actually converted to numeric:
        if dtype_numeric and array.dtype.kind == "O":
            array = array.astype(np.float64)
        if not allow_nd and array.ndim >= 3:
            raise ValueError("Found array with dim %d. %s expected <= 2."
                             % (array.ndim, estimator_name))

        if force_all_finite:
            _assert_all_finite(array,
                               allow_nan=force_all_finite == 'allow-nan')

    if ensure_min_samples > 0:
        n_samples = _num_samples(array)
        if n_samples < ensure_min_samples:
            raise ValueError("Found array with %d sample(s) (shape=%s) while a"
                             " minimum of %d is required%s."
                             % (n_samples, array.shape, ensure_min_samples,
                                context))

    if ensure_min_features > 0 and array.ndim == 2:
        n_features = array.shape[1]
        if n_features < ensure_min_features:
            raise ValueError("Found array with %d feature(s) (shape=%s) while"
                             " a minimum of %d is required%s."
                             % (n_features, array.shape, ensure_min_features,
                                context))

    if warn_on_dtype and dtype_orig is not None and array.dtype != dtype_orig:
        msg = ("Data with input dtype %s was converted to %s%s."
               % (dtype_orig, array.dtype, context))
        warnings.warn(msg, DataConversionWarning, stacklevel=2)

    if copy and np.may_share_memory(array, array_orig):
        array = np.array(array, dtype=dtype, order=order)

    if (warn_on_dtype and dtypes_orig is not None and
            {array.dtype} != set(dtypes_orig)):
        # if there was at the beginning some other types than the final one
        # (for instance in a DataFrame that can contain several dtypes) then
        # some data must have been converted
        msg = ("Data with input dtype %s were all converted to %s%s."
               % (', '.join(map(str, sorted(set(dtypes_orig)))), array.dtype,
                  context))
        warnings.warn(msg, DataConversionWarning, stacklevel=3)

    return array


def _check_large_sparse(X, accept_large_sparse=False):
    """Raise a ValueError if X has 64bit indices and accept_large_sparse=False
    """
    if not accept_large_sparse:
        supported_indices = ["int32"]
        if X.getformat() == "coo":
            index_keys = ['col', 'row']
        elif X.getformat() in ["csr", "csc", "bsr"]:
            index_keys = ['indices', 'indptr']
        else:
            return
        for key in index_keys:
            indices_datatype = getattr(X, key).dtype
            if (indices_datatype not in supported_indices):
                raise ValueError("Only sparse matrices with 32-bit integer"
                                 " indices are accepted. Got %s indices."
                                 % indices_datatype)


def check_X_y(X, y, accept_sparse=False, accept_large_sparse=True,
              dtype="numeric", order=None, copy=False, force_all_finite=True,
              ensure_2d=True, allow_nd=False, multi_output=False,
              ensure_min_samples=1, ensure_min_features=1, y_numeric=False,
              warn_on_dtype=None, estimator=None):
    """Input validation for standard estimators.

    Checks X and y for consistent length, enforces X to be 2D and y 1D. By
    default, X is checked to be non-empty and containing only finite values.
    Standard input checks are also applied to y, such as checking that y
    does not have np.nan or np.inf targets. For multi-label y, set
    multi_output=True to allow 2D and sparse y. If the dtype of X is
    object, attempt converting to float, raising on failure.

    Parameters
    ----------
    X : nd-array, list or sparse matrix
        Input data.

    y : nd-array, list or sparse matrix
        Labels.

    accept_sparse : string, boolean or list of string (default=False)
        String[s] representing allowed sparse matrix formats, such as 'csc',
        'csr', etc. If the input is sparse but not in the allowed format,
        it will be converted to the first listed format. True allows the input
        to be any format. False means that a sparse matrix input will
        raise an error.

    accept_large_sparse : bool (default=True)
        If a CSR, CSC, COO or BSR sparse matrix is supplied and accepted by
        accept_sparse, accept_large_sparse will cause it to be accepted only
        if its indices are stored with a 32-bit dtype.

        .. versionadded:: 0.20

    dtype : string, type, list of types or None (default="numeric")
        Data type of result. If None, the dtype of the input is preserved.
        If "numeric", dtype is preserved unless array.dtype is object.
        If dtype is a list of types, conversion on the first type is only
        performed if the dtype of the input is not in the list.

    order : 'F', 'C' or None (default=None)
        Whether an array will be forced to be fortran or c-style.

    copy : boolean (default=False)
        Whether a forced copy will be triggered. If copy=False, a copy might
        be triggered by a conversion.

    force_all_finite : boolean or 'allow-nan', (default=True)
        Whether to raise an error on np.inf and np.nan in X. This parameter
        does not influence whether y can have np.inf or np.nan values.
        The possibilities are:

        - True: Force all values of X to be finite.
        - False: accept both np.inf and np.nan in X.
        - 'allow-nan': accept only np.nan values in X. Values cannot be
          infinite.

        .. versionadded:: 0.20
           ``force_all_finite`` accepts the string ``'allow-nan'``.

    ensure_2d : boolean (default=True)
        Whether to raise a value error if X is not 2D.

    allow_nd : boolean (default=False)
        Whether to allow X.ndim > 2.

    multi_output : boolean (default=False)
        Whether to allow 2D y (array or sparse matrix). If false, y will be
        validated as a vector. y cannot have np.nan or np.inf values if
        multi_output=True.

    ensure_min_samples : int (default=1)
        Make sure that X has a minimum number of samples in its first
        axis (rows for a 2D array).

    ensure_min_features : int (default=1)
        Make sure that the 2D array has some minimum number of features
        (columns). The default value of 1 rejects empty datasets.
        This check is only enforced when X has effectively 2 dimensions or
        is originally 1D and ``ensure_2d`` is True. Setting to 0 disables
        this check.

    y_numeric : boolean (default=False)
        Whether to ensure that y has a numeric type. If dtype of y is object,
        it is converted to float64. Should only be used for regression
        algorithms.

    warn_on_dtype : boolean or None, optional (default=None)
        Raise DataConversionWarning if the dtype of the input data structure
        does not match the requested dtype, causing a memory copy.

        .. deprecated:: 0.21
            ``warn_on_dtype`` is deprecated in version 0.21 and will be
             removed in 0.23.

    estimator : str or estimator instance (default=None)
        If passed, include the name of the estimator in warning messages.

    Returns
    -------
    X_converted : object
        The converted and validated X.

    y_converted : object
        The converted and validated y.
    """
    if y is None:
        raise ValueError("y cannot be None")

    X = check_array(X, accept_sparse=accept_sparse,
                    accept_large_sparse=accept_large_sparse,
                    dtype=dtype, order=order, copy=copy,
                    force_all_finite=force_all_finite,
                    ensure_2d=ensure_2d, allow_nd=allow_nd,
                    ensure_min_samples=ensure_min_samples,
                    ensure_min_features=ensure_min_features,
                    warn_on_dtype=warn_on_dtype,
                    estimator=estimator)
    if multi_output:
        y = check_array(y, 'csr', force_all_finite=True, ensure_2d=False,
                        dtype=None)
    else:
        y = column_or_1d(y, warn=True)
        _assert_all_finite(y)
    if y_numeric and y.dtype.kind == 'O':
        y = y.astype(np.float64)

    check_consistent_length(X, y)

    return X, y


def column_or_1d(y, warn=False):
    """ Ravel column or 1d numpy array, else raises an error

    Parameters
    ----------
    y : array-like

    warn : boolean, default False
       To control display of warnings.

    Returns
    -------
    y : array

    """
    y = np.asarray(y)
    shape = np.shape(y)
    if len(shape) == 1:
        return np.ravel(y)
    if len(shape) == 2 and shape[1] == 1:
        if warn:
            warnings.warn("A column-vector y was passed when a 1d array was"
                          " expected. Please change the shape of y to "
                          "(n_samples, ), for example using ravel().",
                          DataConversionWarning, stacklevel=2)
        return np.ravel(y)

    raise ValueError("bad input shape {0}".format(shape))


def check_random_state(seed):
    """Turn seed into a np.random.RandomState instance

    Parameters
    ----------
    seed : None | int | instance of RandomState
        If seed is None, return the RandomState singleton used by np.random.
        If seed is an int, return a new RandomState instance seeded with seed.
        If seed is already a RandomState instance, return it.
        Otherwise raise ValueError.
    """
    if seed is None or seed is np.random:
        return np.random.mtrand._rand
    if isinstance(seed, numbers.Integral):
        return np.random.RandomState(seed)
    if isinstance(seed, np.random.RandomState):
        return seed
    raise ValueError('%r cannot be used to seed a numpy.random.RandomState'
                     ' instance' % seed)


def has_fit_parameter(estimator, parameter):
    """Checks whether the estimator's fit method supports the given parameter.

    Parameters
    ----------
    estimator : object
        An estimator to inspect.

    parameter : str
        The searched parameter.

    Returns
    -------
    is_parameter: bool
        Whether the parameter was found to be a named parameter of the
        estimator's fit method.

    Examples
    --------
    >>> from sklearn.svm import SVC
    >>> has_fit_parameter(SVC(), "sample_weight")
    True

    """
    return parameter in signature(estimator.fit).parameters


def check_symmetric(array, tol=1E-10, raise_warning=True,
                    raise_exception=False):
    """Make sure that array is 2D, square and symmetric.

    If the array is not symmetric, then a symmetrized version is returned.
    Optionally, a warning or exception is raised if the matrix is not
    symmetric.

    Parameters
    ----------
    array : nd-array or sparse matrix
        Input object to check / convert. Must be two-dimensional and square,
        otherwise a ValueError will be raised.
    tol : float
        Absolute tolerance for equivalence of arrays. Default = 1E-10.
    raise_warning : boolean (default=True)
        If True then raise a warning if conversion is required.
    raise_exception : boolean (default=False)
        If True then raise an exception if array is not symmetric.

    Returns
    -------
    array_sym : ndarray or sparse matrix
        Symmetrized version of the input array, i.e. the average of array
        and array.transpose(). If sparse, then duplicate entries are first
        summed and zeros are eliminated.
    """
    if (array.ndim != 2) or (array.shape[0] != array.shape[1]):
        raise ValueError("array must be 2-dimensional and square. "
                         "shape = {0}".format(array.shape))

    if sp.issparse(array):
        diff = array - array.T
        # only csr, csc, and coo have `data` attribute
        if diff.format not in ['csr', 'csc', 'coo']:
            diff = diff.tocsr()
        symmetric = np.all(abs(diff.data) < tol)
    else:
        symmetric = np.allclose(array, array.T, atol=tol)

    if not symmetric:
        if raise_exception:
            raise ValueError("Array must be symmetric")
        if raise_warning:
            warnings.warn("Array is not symmetric, and will be converted "
                          "to symmetric by average with its transpose.",
                          stacklevel=2)
        if sp.issparse(array):
            conversion = 'to' + array.format
            array = getattr(0.5 * (array + array.T), conversion)()
        else:
            array = 0.5 * (array + array.T)

    return array


def check_is_fitted(estimator, attributes=None, msg=None, all_or_any=all):
    """Perform is_fitted validation for estimator.

    Checks if the estimator is fitted by verifying the presence of
    fitted attributes (ending with a trailing underscore) and otherwise
    raises a NotFittedError with the given message.

    This utility is meant to be used internally by estimators themselves,
    typically in their own predict / transform methods.

    Parameters
    ----------
    estimator : estimator instance.
        estimator instance for which the check is performed.

    attributes : str, list or tuple of str, default=None
        Attribute name(s) given as string or a list/tuple of strings
        Eg.: ``["coef_", "estimator_", ...], "coef_"``

        If `None`, `estimator` is considered fitted if there exist an
        attribute that ends with a underscore and does not start with double
        underscore.

    msg : string
        The default error message is, "This %(name)s instance is not fitted
        yet. Call 'fit' with appropriate arguments before using this
        estimator."

        For custom messages if "%(name)s" is present in the message string,
        it is substituted for the estimator name.

        Eg. : "Estimator, %(name)s, must be fitted before sparsifying".

    all_or_any : callable, {all, any}, default all
        Specify whether all or any of the given attributes must exist.

    Returns
    -------
    None

    Raises
    ------
    NotFittedError
        If the attributes are not found.
    """
    if isclass(estimator):
        raise TypeError("{} is a class, not an instance.".format(estimator))
    if msg is None:
        msg = ("This %(name)s instance is not fitted yet. Call 'fit' with "
               "appropriate arguments before using this estimator.")

    if not hasattr(estimator, 'fit'):
        raise TypeError("%s is not an estimator instance." % (estimator))

    if attributes is not None:
        if not isinstance(attributes, (list, tuple)):
            attributes = [attributes]
        attrs = all_or_any([hasattr(estimator, attr) for attr in attributes])
    else:
        attrs = [v for v in vars(estimator)
                 if v.endswith("_") and not v.startswith("__")]

    if not attrs:
        raise NotFittedError(msg % {'name': type(estimator).__name__})


def check_non_negative(X, whom):
    """
    Check if there is any negative value in an array.

    Parameters
    ----------
    X : array-like or sparse matrix
        Input data.

    whom : string
        Who passed X to this function.
    """
    # avoid X.min() on sparse matrix since it also sorts the indices
    if sp.issparse(X):
        if X.format in ['lil', 'dok']:
            X = X.tocsr()
        if X.data.size == 0:
            X_min = 0
        else:
            X_min = X.data.min()
    else:
        X_min = X.min()

    if X_min < 0:
        raise ValueError("Negative values in data passed to %s" % whom)


def check_scalar(x, name, target_type, min_val=None, max_val=None):
    """Validate scalar parameters type and value.

    Parameters
    ----------
    x : object
        The scalar parameter to validate.

    name : str
        The name of the parameter to be printed in error messages.

    target_type : type or tuple
        Acceptable data types for the parameter.

    min_val : float or int, optional (default=None)
        The minimum valid value the parameter can take. If None (default) it
        is implied that the parameter does not have a lower bound.

    max_val : float or int, optional (default=None)
        The maximum valid value the parameter can take. If None (default) it
        is implied that the parameter does not have an upper bound.

    Raises
    -------
    TypeError
        If the parameter's type does not match the desired type.

    ValueError
        If the parameter's value violates the given bounds.
    """

    if not isinstance(x, target_type):
        raise TypeError('`{}` must be an instance of {}, not {}.'
                        .format(name, target_type, type(x)))

    if min_val is not None and x < min_val:
        raise ValueError('`{}`= {}, must be >= {}.'.format(name, x, min_val))

    if max_val is not None and x > max_val:
        raise ValueError('`{}`= {}, must be <= {}.'.format(name, x, max_val))


def _check_psd_eigenvalues(lambdas, enable_warnings=False):
    """Check the eigenvalues of a positive semidefinite (PSD) matrix.

    Checks the provided array of PSD matrix eigenvalues for numerical or
    conditioning issues and returns a fixed validated version. This method
    should typically be used if the PSD matrix is user-provided (e.g. a
    Gram matrix) or computed using a user-provided dissimilarity metric
    (e.g. kernel function), or if the decomposition process uses approximation
    methods (randomized SVD, etc.).

    It checks for three things:

    - that there are no significant imaginary parts in eigenvalues (more than
      1e-5 times the maximum real part). If this check fails, it raises a
      ``ValueError``. Otherwise all non-significant imaginary parts that may
      remain are set to zero. This operation is traced with a
      ``PositiveSpectrumWarning`` when ``enable_warnings=True``.

    - that eigenvalues are not all negative. If this check fails, it raises a
      ``ValueError``

    - that there are no significant negative eigenvalues with absolute value
      more than 1e-10 (1e-6) and more than 1e-5 (5e-3) times the largest
      positive eigenvalue in double (simple) precision. If this check fails,
      it raises a ``ValueError``. Otherwise all negative eigenvalues that may
      remain are set to zero. This operation is traced with a
      ``PositiveSpectrumWarning`` when ``enable_warnings=True``.

    Finally, all the positive eigenvalues that are too small (with a value
    smaller than the maximum eigenvalue divided by 1e12) are set to zero.
    This operation is traced with a ``PositiveSpectrumWarning`` when
    ``enable_warnings=True``.

    Parameters
    ----------
    lambdas : array-like of shape (n_eigenvalues,)
        Array of eigenvalues to check / fix.

    enable_warnings : bool, default=False
        When this is set to ``True``, a ``PositiveSpectrumWarning`` will be
        raised when there are imaginary parts, negative eigenvalues, or
        extremely small non-zero eigenvalues. Otherwise no warning will be
        raised. In both cases, imaginary parts, negative eigenvalues, and
        extremely small non-zero eigenvalues will be set to zero.

    Returns
    -------
    lambdas_fixed : ndarray of shape (n_eigenvalues,)
        A fixed validated copy of the array of eigenvalues.

    Examples
    --------
    >>> _check_psd_eigenvalues([1, 2])      # nominal case
    array([1, 2])
    >>> _check_psd_eigenvalues([5, 5j])     # significant imag part
    Traceback (most recent call last):
        ...
    ValueError: There are significant imaginary parts in eigenvalues (1
        of the maximum real part). Either the matrix is not PSD, or there was
        an issue while computing the eigendecomposition of the matrix.
    >>> _check_psd_eigenvalues([5, 5e-5j])  # insignificant imag part
    array([5., 0.])
    >>> _check_psd_eigenvalues([-5, -1])    # all negative
    Traceback (most recent call last):
        ...
    ValueError: All eigenvalues are negative (maximum is -1). Either the
        matrix is not PSD, or there was an issue while computing the
        eigendecomposition of the matrix.
    >>> _check_psd_eigenvalues([5, -1])     # significant negative
    Traceback (most recent call last):
        ...
    ValueError: There are significant negative eigenvalues (0.2 of the
        maximum positive). Either the matrix is not PSD, or there was an issue
        while computing the eigendecomposition of the matrix.
    >>> _check_psd_eigenvalues([5, -5e-5])  # insignificant negative
    array([5., 0.])
    >>> _check_psd_eigenvalues([5, 4e-12])  # bad conditioning (too small)
    array([5., 0.])

    """

    lambdas = np.array(lambdas)
    is_double_precision = lambdas.dtype == np.float64

    # note: the minimum value available is
    #  - single-precision: np.finfo('float32').eps = 1.2e-07
    #  - double-precision: np.finfo('float64').eps = 2.2e-16

    # the various thresholds used for validation
    # we may wish to change the value according to precision.
    significant_imag_ratio = 1e-5
    significant_neg_ratio = 1e-5 if is_double_precision else 5e-3
    significant_neg_value = 1e-10 if is_double_precision else 1e-6
    small_pos_ratio = 1e-12

    # Check that there are no significant imaginary parts
    if not np.isreal(lambdas).all():
        max_imag_abs = np.abs(np.imag(lambdas)).max()
        max_real_abs = np.abs(np.real(lambdas)).max()
        if max_imag_abs > significant_imag_ratio * max_real_abs:
            raise ValueError(
                "There are significant imaginary parts in eigenvalues (%g "
                "of the maximum real part). Either the matrix is not PSD, or "
                "there was an issue while computing the eigendecomposition "
                "of the matrix."
                % (max_imag_abs / max_real_abs))

        # warn about imaginary parts being removed
        if enable_warnings:
            warnings.warn("There are imaginary parts in eigenvalues (%g "
                          "of the maximum real part). Either the matrix is not"
                          " PSD, or there was an issue while computing the "
                          "eigendecomposition of the matrix. Only the real "
                          "parts will be kept."
                          % (max_imag_abs / max_real_abs),
                          PositiveSpectrumWarning)

    # Remove all imaginary parts (even if zero)
    lambdas = np.real(lambdas)

    # Check that there are no significant negative eigenvalues
    max_eig = lambdas.max()
    if max_eig < 0:
        raise ValueError("All eigenvalues are negative (maximum is %g). "
                         "Either the matrix is not PSD, or there was an "
                         "issue while computing the eigendecomposition of "
                         "the matrix." % max_eig)

    else:
        min_eig = lambdas.min()
        if (min_eig < -significant_neg_ratio * max_eig
                and min_eig < -significant_neg_value):
            raise ValueError("There are significant negative eigenvalues (%g"
                             " of the maximum positive). Either the matrix is "
                             "not PSD, or there was an issue while computing "
                             "the eigendecomposition of the matrix."
                             % (-min_eig / max_eig))
        elif min_eig < 0:
            # Remove all negative values and warn about it
            if enable_warnings:
                warnings.warn("There are negative eigenvalues (%g of the "
                              "maximum positive). Either the matrix is not "
                              "PSD, or there was an issue while computing the"
                              " eigendecomposition of the matrix. Negative "
                              "eigenvalues will be replaced with 0."
                              % (-min_eig / max_eig),
                              PositiveSpectrumWarning)
            lambdas[lambdas < 0] = 0

    # Check for conditioning (small positive non-zeros)
    too_small_lambdas = (0 < lambdas) & (lambdas < small_pos_ratio * max_eig)
    if too_small_lambdas.any():
        if enable_warnings:
            warnings.warn("Badly conditioned PSD matrix spectrum: the largest "
                          "eigenvalue is more than %g times the smallest. "
                          "Small eigenvalues will be replaced with 0."
                          "" % (1 / small_pos_ratio),
                          PositiveSpectrumWarning)
        lambdas[too_small_lambdas] = 0

    return lambdas


def _check_sample_weight(sample_weight, X, dtype=None):
    """Validate sample weights.

    Note that passing sample_weight=None will output an array of ones.
    Therefore, in some cases, you may want to protect the call with:
    if sample_weight is not None:
        sample_weight = _check_sample_weight(...)

    Parameters
    ----------
    sample_weight : {ndarray, Number or None}, shape (n_samples,)
       Input sample weights.

    X : nd-array, list or sparse matrix
        Input data.

    dtype: dtype
       dtype of the validated `sample_weight`.
       If None, and the input `sample_weight` is an array, the dtype of the
       input is preserved; otherwise an array with the default numpy dtype
       is be allocated.  If `dtype` is not one of `float32`, `float64`,
       `None`, the output will be of dtype `float64`.

    Returns
    -------
    sample_weight : ndarray, shape (n_samples,)
       Validated sample weight. It is guaranteed to be "C" contiguous.
    """
    n_samples = _num_samples(X)

    if dtype is not None and dtype not in [np.float32, np.float64]:
        dtype = np.float64

    if sample_weight is None or isinstance(sample_weight, numbers.Number):
        if sample_weight is None:
            sample_weight = np.ones(n_samples, dtype=dtype)
        else:
            sample_weight = np.full(n_samples, sample_weight,
                                    dtype=dtype)
    else:
        if dtype is None:
            dtype = [np.float64, np.float32]
        sample_weight = check_array(
            sample_weight, accept_sparse=False, ensure_2d=False, dtype=dtype,
            order="C"
        )
        if sample_weight.ndim != 1:
            raise ValueError("Sample weights must be 1D array or scalar")

        if sample_weight.shape != (n_samples,):
            raise ValueError("sample_weight.shape == {}, expected {}!"
                             .format(sample_weight.shape, (n_samples,)))
    return sample_weight


def _allclose_dense_sparse(x, y, rtol=1e-7, atol=1e-9):
    """Check allclose for sparse and dense data.

    Both x and y need to be either sparse or dense, they
    can't be mixed.

    Parameters
    ----------
    x : array-like or sparse matrix
        First array to compare.

    y : array-like or sparse matrix
        Second array to compare.

    rtol : float, optional
        relative tolerance; see numpy.allclose

    atol : float, optional
        absolute tolerance; see numpy.allclose. Note that the default here is
        more tolerant than the default for numpy.testing.assert_allclose, where
        atol=0.
    """
    if sp.issparse(x) and sp.issparse(y):
        x = x.tocsr()
        y = y.tocsr()
        x.sum_duplicates()
        y.sum_duplicates()
        return (np.array_equal(x.indices, y.indices) and
                np.array_equal(x.indptr, y.indptr) and
                np.allclose(x.data, y.data, rtol=rtol, atol=atol))
    elif not sp.issparse(x) and not sp.issparse(y):
        return np.allclose(x, y, rtol=rtol, atol=atol)
    raise ValueError("Can only compare two sparse matrices, not a sparse "
                     "matrix and an array")


def _deprecate_positional_args(f):
    """Decorator for methods that issues warnings for positional arguments

    Using the keyword-only argument syntax in pep 3102, arguments after the
    * will issue a warning when passed as a positional argument.

    Parameters
    ----------
    f : function
        function to check arguments on
    """
    sig = signature(f)
    kwonly_args = []
    all_args = []

    for name, param in sig.parameters.items():
        if param.kind == Parameter.POSITIONAL_OR_KEYWORD:
            all_args.append(name)
        elif param.kind == Parameter.KEYWORD_ONLY:
            kwonly_args.append(name)

    @wraps(f)
    def inner_f(*args, **kwargs):
        extra_args = len(args) - len(all_args)
        if extra_args > 0:
            # ignore first 'self' argument for instance methods
            args_msg = ['{}={}'.format(name, arg)
                        for name, arg in zip(kwonly_args[:extra_args],
                                             args[-extra_args:])]
            warnings.warn("Pass {} as keyword args. From version 0.24 "
                          "passing these as positional arguments will "
                          "result in an error".format(", ".join(args_msg)),
                          FutureWarning)
        kwargs.update({k: arg for k, arg in zip(all_args, args)})
        return f(**kwargs)
    return inner_f


def _check_fit_params(X, fit_params, indices=None):
    """Check and validate the parameters passed during `fit`.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Data array.

    fit_params : dict
        Dictionary containing the parameters passed at fit.

    indices : array-like of shape (n_samples,), default=None
        Indices to be selected if the parameter has the same size as `X`.

    Returns
    -------
    fit_params_validated : dict
        Validated parameters. We ensure that the values support indexing.
    """
    from . import _safe_indexing
    fit_params_validated = {}
    for param_key, param_value in fit_params.items():
        if (not _is_arraylike(param_value) or
                _num_samples(param_value) != _num_samples(X)):
            # Non-indexable pass-through (for now for backward-compatibility).
            # https://github.com/scikit-learn/scikit-learn/issues/15805
            fit_params_validated[param_key] = param_value
        else:
            # Any other fit_params should support indexing
            # (e.g. for cross-validation).
            fit_params_validated[param_key] = _make_indexable(param_value)
            fit_params_validated[param_key] = _safe_indexing(
                fit_params_validated[param_key], indices
            )

    return fit_params_validated
