import numpy as np
from scipy.stats import rankdata as _sp_rankdata
from .fixes import bincount


# To remove when we support scipy 0.13
def _rankdata(a, method="average"):
    """Assign ranks to data, dealing with ties appropriately.

    Ranks begin at 1. The method argument controls how ranks are assigned
    to equal values.

    Parameters
    ----------
    a : array_like
        The array of values to be ranked. The array is first flattened.

    method : str, optional
        The method used to assign ranks to tied elements.
        The options are 'max'.
        'max': The maximum of the ranks that would have been assigned
              to all the tied values is assigned to each value.

    Returns
    -------
    ranks : ndarray
        An array of length equal to the size of a, containing rank scores.

    Notes
    -----
    We only backport the 'max' method

    """
    if method != "max":
        raise NotImplementedError()

    unique_all, inverse = np.unique(a, return_inverse=True)
    count = bincount(inverse, minlength=unique_all.size)
    cum_count = count.cumsum()
    rank = cum_count[inverse]
    return rank

try:
    _sp_rankdata([1.], 'max')
    rankdata = _sp_rankdata

except TypeError as e:
    rankdata = _rankdata


def _weighted_percentile(array, sample_weight, percentile=50):
    """Compute the weighted ``percentile`` of ``array`` with ``sample_weight``. """
    sorted_idx = np.argsort(array)

    # Find index of median prediction for each sample
    weight_cdf = sample_weight[sorted_idx].cumsum()
    percentile_idx = np.searchsorted(
        weight_cdf, (percentile / 100.) * weight_cdf[-1])
    return array[sorted_idx[percentile_idx]]
