import numpy as np

from . import is_scalar_nan
from .fixes import _object_dtype_isnan


def _get_mask(X, value_to_mask):
    """Compute the boolean mask X == missing_values."""
    if is_scalar_nan(value_to_mask):
        if X.dtype.kind == "f":
            return np.isnan(X)
        elif X.dtype.kind in ("i", "u"):
            # can't have NaNs in integer array.
            return np.zeros(X.shape, dtype=bool)
        else:
            # np.isnan does not work on object dtypes.
            return _object_dtype_isnan(X)
    else:
        # X == value_to_mask with object dtypes does not always perform
        # element-wise for old versions of numpy
        return np.equal(X, value_to_mask)
