"""
Template for each `dtype` helper function using groupby

WARNING: DO NOT edit .pxi FILE directly, .pxi is generated from .pxi.in
"""

cdef extern from "numpy/npy_math.h":
    float64_t NAN "NPY_NAN"
_int64_max = np.iinfo(np.int64).max

# ----------------------------------------------------------------------
# group_add, group_prod, group_var, group_mean, group_ohlc
# ----------------------------------------------------------------------


@cython.wraparound(False)
@cython.boundscheck(False)
def group_add_float64(ndarray[float64_t, ndim=2] out,
                       ndarray[int64_t] counts,
                       ndarray[float64_t, ndim=2] values,
                       ndarray[int64_t] labels,
                       Py_ssize_t min_count=0):
    """
    Only aggregates on axis=0
    """
    cdef:
        Py_ssize_t i, j, N, K, lab, ncounts = len(counts)
        float64_t val, count
        ndarray[float64_t, ndim=2] sumx, nobs

    if not len(values) == len(labels):
        raise AssertionError("len(index) != len(labels)")

    nobs = np.zeros_like(out)
    sumx = np.zeros_like(out)

    N, K = (<object>values).shape

    with nogil:

        for i in range(N):
            lab = labels[i]
            if lab < 0:
                continue

            counts[lab] += 1
            for j in range(K):
                val = values[i, j]

                # not nan
                if val == val:
                    nobs[lab, j] += 1
                    sumx[lab, j] += val

        for i in range(ncounts):
            for j in range(K):
                if nobs[i, j] < min_count:
                    out[i, j] = NAN
                else:
                    out[i, j] = sumx[i, j]


@cython.wraparound(False)
@cython.boundscheck(False)
def group_prod_float64(ndarray[float64_t, ndim=2] out,
                        ndarray[int64_t] counts,
                        ndarray[float64_t, ndim=2] values,
                        ndarray[int64_t] labels,
                        Py_ssize_t min_count=0):
    """
    Only aggregates on axis=0
    """
    cdef:
        Py_ssize_t i, j, N, K, lab, ncounts = len(counts)
        float64_t val, count
        ndarray[float64_t, ndim=2] prodx, nobs

    if not len(values) == len(labels):
        raise AssertionError("len(index) != len(labels)")

    nobs = np.zeros_like(out)
    prodx = np.ones_like(out)

    N, K = (<object>values).shape

    with nogil:
        for i in range(N):
            lab = labels[i]
            if lab < 0:
                continue

            counts[lab] += 1
            for j in range(K):
                val = values[i, j]

                # not nan
                if val == val:
                    nobs[lab, j] += 1
                    prodx[lab, j] *= val

        for i in range(ncounts):
            for j in range(K):
                if nobs[i, j] < min_count:
                    out[i, j] = NAN
                else:
                    out[i, j] = prodx[i, j]


@cython.wraparound(False)
@cython.boundscheck(False)
@cython.cdivision(True)
def group_var_float64(ndarray[float64_t, ndim=2] out,
                       ndarray[int64_t] counts,
                       ndarray[float64_t, ndim=2] values,
                       ndarray[int64_t] labels,
                       Py_ssize_t min_count=-1):
    cdef:
        Py_ssize_t i, j, N, K, lab, ncounts = len(counts)
        float64_t val, ct, oldmean
        ndarray[float64_t, ndim=2] nobs, mean

    assert min_count == -1, "'min_count' only used in add and prod"

    if not len(values) == len(labels):
        raise AssertionError("len(index) != len(labels)")

    nobs = np.zeros_like(out)
    mean = np.zeros_like(out)

    N, K = (<object>values).shape

    out[:, :] = 0.0

    with nogil:
        for i in range(N):
            lab = labels[i]
            if lab < 0:
                continue

            counts[lab] += 1

            for j in range(K):
                val = values[i, j]

                # not nan
                if val == val:
                    nobs[lab, j] += 1
                    oldmean = mean[lab, j]
                    mean[lab, j] += (val - oldmean) / nobs[lab, j]
                    out[lab, j] += (val - mean[lab, j]) * (val - oldmean)

        for i in range(ncounts):
            for j in range(K):
                ct = nobs[i, j]
                if ct < 2:
                    out[i, j] = NAN
                else:
                    out[i, j] /= (ct - 1)
# add passing bin edges, instead of labels


@cython.wraparound(False)
@cython.boundscheck(False)
def group_mean_float64(ndarray[float64_t, ndim=2] out,
                        ndarray[int64_t] counts,
                        ndarray[float64_t, ndim=2] values,
                        ndarray[int64_t] labels,
                        Py_ssize_t min_count=-1):
    cdef:
        Py_ssize_t i, j, N, K, lab, ncounts = len(counts)
        float64_t val, count
        ndarray[float64_t, ndim=2] sumx, nobs

    assert min_count == -1, "'min_count' only used in add and prod"

    if not len(values) == len(labels):
        raise AssertionError("len(index) != len(labels)")

    nobs = np.zeros_like(out)
    sumx = np.zeros_like(out)

    N, K = (<object>values).shape

    with nogil:
        for i in range(N):
            lab = labels[i]
            if lab < 0:
                continue

            counts[lab] += 1
            for j in range(K):
                val = values[i, j]
                # not nan
                if val == val:
                    nobs[lab, j] += 1
                    sumx[lab, j] += val

        for i in range(ncounts):
            for j in range(K):
                count = nobs[i, j]
                if nobs[i, j] == 0:
                    out[i, j] = NAN
                else:
                    out[i, j] = sumx[i, j] / count


@cython.wraparound(False)
@cython.boundscheck(False)
def group_ohlc_float64(ndarray[float64_t, ndim=2] out,
                  ndarray[int64_t] counts,
                  ndarray[float64_t, ndim=2] values,
                  ndarray[int64_t] labels,
                  Py_ssize_t min_count=-1):
    """
    Only aggregates on axis=0
    """
    cdef:
        Py_ssize_t i, j, N, K, lab
        float64_t val, count
        Py_ssize_t ngroups = len(counts)

    assert min_count == -1, "'min_count' only used in add and prod"

    if len(labels) == 0:
        return

    N, K = (<object>values).shape

    if out.shape[1] != 4:
        raise ValueError('Output array must have 4 columns')

    if K > 1:
        raise NotImplementedError("Argument 'values' must have only "
                                  "one dimension")
    out[:] = np.nan

    with nogil:
        for i in range(N):
            lab = labels[i]
            if lab == -1:
                continue

            counts[lab] += 1
            val = values[i, 0]
            if val != val:
                continue

            if out[lab, 0] != out[lab, 0]:
                out[lab, 0] = out[lab, 1] = out[lab, 2] = out[lab, 3] = val
            else:
                out[lab, 1] = max(out[lab, 1], val)
                out[lab, 2] = min(out[lab, 2], val)
                out[lab, 3] = val


@cython.wraparound(False)
@cython.boundscheck(False)
def group_add_float32(ndarray[float32_t, ndim=2] out,
                       ndarray[int64_t] counts,
                       ndarray[float32_t, ndim=2] values,
                       ndarray[int64_t] labels,
                       Py_ssize_t min_count=0):
    """
    Only aggregates on axis=0
    """
    cdef:
        Py_ssize_t i, j, N, K, lab, ncounts = len(counts)
        float32_t val, count
        ndarray[float32_t, ndim=2] sumx, nobs

    if not len(values) == len(labels):
        raise AssertionError("len(index) != len(labels)")

    nobs = np.zeros_like(out)
    sumx = np.zeros_like(out)

    N, K = (<object>values).shape

    with nogil:

        for i in range(N):
            lab = labels[i]
            if lab < 0:
                continue

            counts[lab] += 1
            for j in range(K):
                val = values[i, j]

                # not nan
                if val == val:
                    nobs[lab, j] += 1
                    sumx[lab, j] += val

        for i in range(ncounts):
            for j in range(K):
                if nobs[i, j] < min_count:
                    out[i, j] = NAN
                else:
                    out[i, j] = sumx[i, j]


@cython.wraparound(False)
@cython.boundscheck(False)
def group_prod_float32(ndarray[float32_t, ndim=2] out,
                        ndarray[int64_t] counts,
                        ndarray[float32_t, ndim=2] values,
                        ndarray[int64_t] labels,
                        Py_ssize_t min_count=0):
    """
    Only aggregates on axis=0
    """
    cdef:
        Py_ssize_t i, j, N, K, lab, ncounts = len(counts)
        float32_t val, count
        ndarray[float32_t, ndim=2] prodx, nobs

    if not len(values) == len(labels):
        raise AssertionError("len(index) != len(labels)")

    nobs = np.zeros_like(out)
    prodx = np.ones_like(out)

    N, K = (<object>values).shape

    with nogil:
        for i in range(N):
            lab = labels[i]
            if lab < 0:
                continue

            counts[lab] += 1
            for j in range(K):
                val = values[i, j]

                # not nan
                if val == val:
                    nobs[lab, j] += 1
                    prodx[lab, j] *= val

        for i in range(ncounts):
            for j in range(K):
                if nobs[i, j] < min_count:
                    out[i, j] = NAN
                else:
                    out[i, j] = prodx[i, j]


@cython.wraparound(False)
@cython.boundscheck(False)
@cython.cdivision(True)
def group_var_float32(ndarray[float32_t, ndim=2] out,
                       ndarray[int64_t] counts,
                       ndarray[float32_t, ndim=2] values,
                       ndarray[int64_t] labels,
                       Py_ssize_t min_count=-1):
    cdef:
        Py_ssize_t i, j, N, K, lab, ncounts = len(counts)
        float32_t val, ct, oldmean
        ndarray[float32_t, ndim=2] nobs, mean

    assert min_count == -1, "'min_count' only used in add and prod"

    if not len(values) == len(labels):
        raise AssertionError("len(index) != len(labels)")

    nobs = np.zeros_like(out)
    mean = np.zeros_like(out)

    N, K = (<object>values).shape

    out[:, :] = 0.0

    with nogil:
        for i in range(N):
            lab = labels[i]
            if lab < 0:
                continue

            counts[lab] += 1

            for j in range(K):
                val = values[i, j]

                # not nan
                if val == val:
                    nobs[lab, j] += 1
                    oldmean = mean[lab, j]
                    mean[lab, j] += (val - oldmean) / nobs[lab, j]
                    out[lab, j] += (val - mean[lab, j]) * (val - oldmean)

        for i in range(ncounts):
            for j in range(K):
                ct = nobs[i, j]
                if ct < 2:
                    out[i, j] = NAN
                else:
                    out[i, j] /= (ct - 1)
# add passing bin edges, instead of labels


@cython.wraparound(False)
@cython.boundscheck(False)
def group_mean_float32(ndarray[float32_t, ndim=2] out,
                        ndarray[int64_t] counts,
                        ndarray[float32_t, ndim=2] values,
                        ndarray[int64_t] labels,
                        Py_ssize_t min_count=-1):
    cdef:
        Py_ssize_t i, j, N, K, lab, ncounts = len(counts)
        float32_t val, count
        ndarray[float32_t, ndim=2] sumx, nobs

    assert min_count == -1, "'min_count' only used in add and prod"

    if not len(values) == len(labels):
        raise AssertionError("len(index) != len(labels)")

    nobs = np.zeros_like(out)
    sumx = np.zeros_like(out)

    N, K = (<object>values).shape

    with nogil:
        for i in range(N):
            lab = labels[i]
            if lab < 0:
                continue

            counts[lab] += 1
            for j in range(K):
                val = values[i, j]
                # not nan
                if val == val:
                    nobs[lab, j] += 1
                    sumx[lab, j] += val

        for i in range(ncounts):
            for j in range(K):
                count = nobs[i, j]
                if nobs[i, j] == 0:
                    out[i, j] = NAN
                else:
                    out[i, j] = sumx[i, j] / count


@cython.wraparound(False)
@cython.boundscheck(False)
def group_ohlc_float32(ndarray[float32_t, ndim=2] out,
                  ndarray[int64_t] counts,
                  ndarray[float32_t, ndim=2] values,
                  ndarray[int64_t] labels,
                  Py_ssize_t min_count=-1):
    """
    Only aggregates on axis=0
    """
    cdef:
        Py_ssize_t i, j, N, K, lab
        float32_t val, count
        Py_ssize_t ngroups = len(counts)

    assert min_count == -1, "'min_count' only used in add and prod"

    if len(labels) == 0:
        return

    N, K = (<object>values).shape

    if out.shape[1] != 4:
        raise ValueError('Output array must have 4 columns')

    if K > 1:
        raise NotImplementedError("Argument 'values' must have only "
                                  "one dimension")
    out[:] = np.nan

    with nogil:
        for i in range(N):
            lab = labels[i]
            if lab == -1:
                continue

            counts[lab] += 1
            val = values[i, 0]
            if val != val:
                continue

            if out[lab, 0] != out[lab, 0]:
                out[lab, 0] = out[lab, 1] = out[lab, 2] = out[lab, 3] = val
            else:
                out[lab, 1] = max(out[lab, 1], val)
                out[lab, 2] = min(out[lab, 2], val)
                out[lab, 3] = val

# ----------------------------------------------------------------------
# group_nth, group_last, group_rank
# ----------------------------------------------------------------------


@cython.wraparound(False)
@cython.boundscheck(False)
def group_last_float64(ndarray[float64_t, ndim=2] out,
                        ndarray[int64_t] counts,
                        ndarray[float64_t, ndim=2] values,
                        ndarray[int64_t] labels,
                        Py_ssize_t min_count=-1):
    """
    Only aggregates on axis=0
    """
    cdef:
        Py_ssize_t i, j, N, K, lab, ncounts = len(counts)
        float64_t val
        ndarray[float64_t, ndim=2] resx
        ndarray[int64_t, ndim=2] nobs

    assert min_count == -1, "'min_count' only used in add and prod"

    if not len(values) == len(labels):
        raise AssertionError("len(index) != len(labels)")

    nobs = np.zeros((<object>out).shape, dtype=np.int64)
    resx = np.empty_like(out)

    N, K = (<object>values).shape

    with nogil:
        for i in range(N):
            lab = labels[i]
            if lab < 0:
                continue

            counts[lab] += 1
            for j in range(K):
                val = values[i, j]

                # not nan
                if val == val and val != NAN:
                    nobs[lab, j] += 1
                    resx[lab, j] = val

        for i in range(ncounts):
            for j in range(K):
                if nobs[i, j] == 0:
                    out[i, j] = NAN
                else:
                    out[i, j] = resx[i, j]


@cython.wraparound(False)
@cython.boundscheck(False)
def group_nth_float64(ndarray[float64_t, ndim=2] out,
                       ndarray[int64_t] counts,
                       ndarray[float64_t, ndim=2] values,
                       ndarray[int64_t] labels, int64_t rank,
                       Py_ssize_t min_count=-1):
    """
    Only aggregates on axis=0
    """
    cdef:
        Py_ssize_t i, j, N, K, lab, ncounts = len(counts)
        float64_t val
        ndarray[float64_t, ndim=2] resx
        ndarray[int64_t, ndim=2] nobs

    assert min_count == -1, "'min_count' only used in add and prod"

    if not len(values) == len(labels):
        raise AssertionError("len(index) != len(labels)")

    nobs = np.zeros((<object>out).shape, dtype=np.int64)
    resx = np.empty_like(out)

    N, K = (<object>values).shape

    with nogil:
        for i in range(N):
            lab = labels[i]
            if lab < 0:
                continue

            counts[lab] += 1
            for j in range(K):
                val = values[i, j]

                # not nan
                if val == val and val != NAN:
                    nobs[lab, j] += 1
                    if nobs[lab, j] == rank:
                        resx[lab, j] = val

        for i in range(ncounts):
            for j in range(K):
                if nobs[i, j] == 0:
                    out[i, j] = NAN
                else:
                    out[i, j] = resx[i, j]



@cython.boundscheck(False)
@cython.wraparound(False)
def group_rank_float64(ndarray[float64_t, ndim=2] out,
                        ndarray[float64_t, ndim=2] values,
                        ndarray[int64_t] labels,
                        bint is_datetimelike, object ties_method,
                        bint ascending, bint pct, object na_option):
    """
    Provides the rank of values within each group.

    Parameters
    ----------
    out : array of float64_t values which this method will write its results to
    values : array of float64_t values to be ranked
    labels : array containing unique label for each group, with its ordering
        matching up to the corresponding record in `values`
    is_datetimelike : bool, default False
        unused in this method but provided for call compatibility with other
        Cython transformations
    ties_method : {'average', 'min', 'max', 'first', 'dense'}, default
        'average'
        * average: average rank of group
        * min: lowest rank in group
        * max: highest rank in group
        * first: ranks assigned in order they appear in the array
        * dense: like 'min', but rank always increases by 1 between groups
    ascending : boolean, default True
        False for ranks by high (1) to low (N)
        na_option : {'keep', 'top', 'bottom'}, default 'keep'
    pct : boolean, default False
        Compute percentage rank of data within each group
    na_option : {'keep', 'top', 'bottom'}, default 'keep'
        * keep: leave NA values where they are
        * top: smallest rank if ascending
        * bottom: smallest rank if descending

    Notes
    -----
    This method modifies the `out` parameter rather than returning an object
    """
    cdef:
        TiebreakEnumType tiebreak
        Py_ssize_t i, j, N, K, grp_start=0, dups=0, sum_ranks=0
        Py_ssize_t grp_vals_seen=1, grp_na_count=0, grp_tie_count=0
        ndarray[int64_t] _as
        ndarray[float64_t, ndim=2] grp_sizes
        ndarray[float64_t] masked_vals
        ndarray[uint8_t] mask
        bint keep_na
        float64_t nan_fill_val

    tiebreak = tiebreakers[ties_method]
    keep_na = na_option == 'keep'
    N, K = (<object>values).shape
    grp_sizes = np.ones_like(out)

    # Copy values into new array in order to fill missing data
    # with mask, without obfuscating location of missing data
    # in values array
    masked_vals = np.array(values[:, 0], copy=True)
    mask = np.isnan(masked_vals).astype(np.uint8)

    if ascending ^ (na_option == 'top'):
        nan_fill_val = np.inf
        order = (masked_vals, mask, labels)
    else:
        nan_fill_val = -np.inf
        order = (masked_vals, ~mask, labels)
    np.putmask(masked_vals, mask, nan_fill_val)

    # lexsort using labels, then mask, then actual values
    # each label corresponds to a different group value,
    # the mask helps you differentiate missing values before
    # performing sort on the actual values
    _as = np.lexsort(order).astype(np.int64, copy=False)

    if not ascending:
        _as = _as[::-1]

    with nogil:
        # Loop over the length of the value array
        # each incremental i value can be looked up in the _as array
        # that we sorted previously, which gives us the location of
        # that sorted value for retrieval back from the original
        # values / masked_vals arrays
        for i in range(N):
            # dups and sum_ranks will be incremented each loop where
            # the value / group remains the same, and should be reset
            # when either of those change
            # Used to calculate tiebreakers
            dups += 1
            sum_ranks += i - grp_start + 1

            # Update out only when there is a transition of values or labels.
            # When a new value or group is encountered, go back #dups steps(
            # the number of occurrence of current value) and assign the ranks
            # based on the the starting index of the current group (grp_start)
            # and the current index
            if (i == N - 1 or
                    (masked_vals[_as[i]] != masked_vals[_as[i+1]]) or
                    (mask[_as[i]] ^ mask[_as[i+1]]) or
                    (labels[_as[i]] != labels[_as[i+1]])):
                # if keep_na, check for missing values and assign back
                # to the result where appropriate
                if keep_na and mask[_as[i]]:
                    for j in range(i - dups + 1, i + 1):
                        out[_as[j], 0] = NaN
                        grp_na_count = dups
                elif tiebreak == TIEBREAK_AVERAGE:
                    for j in range(i - dups + 1, i + 1):
                        out[_as[j], 0] = sum_ranks / <float64_t>dups
                elif tiebreak == TIEBREAK_MIN:
                    for j in range(i - dups + 1, i + 1):
                        out[_as[j], 0] = i - grp_start - dups + 2
                elif tiebreak == TIEBREAK_MAX:
                    for j in range(i - dups + 1, i + 1):
                        out[_as[j], 0] = i - grp_start + 1
                elif tiebreak == TIEBREAK_FIRST:
                    for j in range(i - dups + 1, i + 1):
                        if ascending:
                            out[_as[j], 0] = j + 1 - grp_start
                        else:
                            out[_as[j], 0] = 2 * i - j - dups + 2 - grp_start
                elif tiebreak == TIEBREAK_DENSE:
                    for j in range(i - dups + 1, i + 1):
                        out[_as[j], 0] = grp_vals_seen

                # look forward to the next value (using the sorting in _as)
                # if the value does not equal the current value then we need to
                # reset the dups and sum_ranks, knowing that a new value is
                # coming up. the conditional also needs to handle nan equality
                # and the end of iteration
                if (i == N - 1 or
                        (masked_vals[_as[i]] != masked_vals[_as[i+1]]) or
                        (mask[_as[i]] ^ mask[_as[i+1]])):
                    dups = sum_ranks = 0
                    grp_vals_seen += 1
                    grp_tie_count += 1

                # Similar to the previous conditional, check now if we are
                # moving to a new group. If so, keep track of the index where
                # the new group occurs, so the tiebreaker calculations can
                # decrement that from their position. fill in the size of each
                # group encountered (used by pct calculations later). also be
                # sure to reset any of the items helping to calculate dups
                if i == N - 1 or labels[_as[i]] != labels[_as[i+1]]:
                    if tiebreak != TIEBREAK_DENSE:
                        for j in range(grp_start, i + 1):
                            grp_sizes[_as[j], 0] = (i - grp_start + 1 -
                                                    grp_na_count)
                    else:
                        for j in range(grp_start, i + 1):
                            grp_sizes[_as[j], 0] = (grp_tie_count -
                                                    (grp_na_count > 0))
                    dups = sum_ranks = 0
                    grp_na_count = 0
                    grp_tie_count = 0
                    grp_start = i + 1
                    grp_vals_seen = 1

        if pct:
            for i in range(N):
                # We don't include NaN values in percentage
                # rankings, so we assign them percentages of NaN.
                if out[i, 0] != out[i, 0] or out[i, 0] == NAN:
                    out[i, 0] = NAN
                elif grp_sizes[i, 0] != 0:
                    out[i, 0] = out[i, 0] / grp_sizes[i, 0]


@cython.wraparound(False)
@cython.boundscheck(False)
def group_last_float32(ndarray[float32_t, ndim=2] out,
                        ndarray[int64_t] counts,
                        ndarray[float32_t, ndim=2] values,
                        ndarray[int64_t] labels,
                        Py_ssize_t min_count=-1):
    """
    Only aggregates on axis=0
    """
    cdef:
        Py_ssize_t i, j, N, K, lab, ncounts = len(counts)
        float32_t val
        ndarray[float32_t, ndim=2] resx
        ndarray[int64_t, ndim=2] nobs

    assert min_count == -1, "'min_count' only used in add and prod"

    if not len(values) == len(labels):
        raise AssertionError("len(index) != len(labels)")

    nobs = np.zeros((<object>out).shape, dtype=np.int64)
    resx = np.empty_like(out)

    N, K = (<object>values).shape

    with nogil:
        for i in range(N):
            lab = labels[i]
            if lab < 0:
                continue

            counts[lab] += 1
            for j in range(K):
                val = values[i, j]

                # not nan
                if val == val and val != NAN:
                    nobs[lab, j] += 1
                    resx[lab, j] = val

        for i in range(ncounts):
            for j in range(K):
                if nobs[i, j] == 0:
                    out[i, j] = NAN
                else:
                    out[i, j] = resx[i, j]


@cython.wraparound(False)
@cython.boundscheck(False)
def group_nth_float32(ndarray[float32_t, ndim=2] out,
                       ndarray[int64_t] counts,
                       ndarray[float32_t, ndim=2] values,
                       ndarray[int64_t] labels, int64_t rank,
                       Py_ssize_t min_count=-1):
    """
    Only aggregates on axis=0
    """
    cdef:
        Py_ssize_t i, j, N, K, lab, ncounts = len(counts)
        float32_t val
        ndarray[float32_t, ndim=2] resx
        ndarray[int64_t, ndim=2] nobs

    assert min_count == -1, "'min_count' only used in add and prod"

    if not len(values) == len(labels):
        raise AssertionError("len(index) != len(labels)")

    nobs = np.zeros((<object>out).shape, dtype=np.int64)
    resx = np.empty_like(out)

    N, K = (<object>values).shape

    with nogil:
        for i in range(N):
            lab = labels[i]
            if lab < 0:
                continue

            counts[lab] += 1
            for j in range(K):
                val = values[i, j]

                # not nan
                if val == val and val != NAN:
                    nobs[lab, j] += 1
                    if nobs[lab, j] == rank:
                        resx[lab, j] = val

        for i in range(ncounts):
            for j in range(K):
                if nobs[i, j] == 0:
                    out[i, j] = NAN
                else:
                    out[i, j] = resx[i, j]



@cython.boundscheck(False)
@cython.wraparound(False)
def group_rank_float32(ndarray[float64_t, ndim=2] out,
                        ndarray[float32_t, ndim=2] values,
                        ndarray[int64_t] labels,
                        bint is_datetimelike, object ties_method,
                        bint ascending, bint pct, object na_option):
    """
    Provides the rank of values within each group.

    Parameters
    ----------
    out : array of float64_t values which this method will write its results to
    values : array of float32_t values to be ranked
    labels : array containing unique label for each group, with its ordering
        matching up to the corresponding record in `values`
    is_datetimelike : bool, default False
        unused in this method but provided for call compatibility with other
        Cython transformations
    ties_method : {'average', 'min', 'max', 'first', 'dense'}, default
        'average'
        * average: average rank of group
        * min: lowest rank in group
        * max: highest rank in group
        * first: ranks assigned in order they appear in the array
        * dense: like 'min', but rank always increases by 1 between groups
    ascending : boolean, default True
        False for ranks by high (1) to low (N)
        na_option : {'keep', 'top', 'bottom'}, default 'keep'
    pct : boolean, default False
        Compute percentage rank of data within each group
    na_option : {'keep', 'top', 'bottom'}, default 'keep'
        * keep: leave NA values where they are
        * top: smallest rank if ascending
        * bottom: smallest rank if descending

    Notes
    -----
    This method modifies the `out` parameter rather than returning an object
    """
    cdef:
        TiebreakEnumType tiebreak
        Py_ssize_t i, j, N, K, grp_start=0, dups=0, sum_ranks=0
        Py_ssize_t grp_vals_seen=1, grp_na_count=0, grp_tie_count=0
        ndarray[int64_t] _as
        ndarray[float64_t, ndim=2] grp_sizes
        ndarray[float32_t] masked_vals
        ndarray[uint8_t] mask
        bint keep_na
        float32_t nan_fill_val

    tiebreak = tiebreakers[ties_method]
    keep_na = na_option == 'keep'
    N, K = (<object>values).shape
    grp_sizes = np.ones_like(out)

    # Copy values into new array in order to fill missing data
    # with mask, without obfuscating location of missing data
    # in values array
    masked_vals = np.array(values[:, 0], copy=True)
    mask = np.isnan(masked_vals).astype(np.uint8)

    if ascending ^ (na_option == 'top'):
        nan_fill_val = np.inf
        order = (masked_vals, mask, labels)
    else:
        nan_fill_val = -np.inf
        order = (masked_vals, ~mask, labels)
    np.putmask(masked_vals, mask, nan_fill_val)

    # lexsort using labels, then mask, then actual values
    # each label corresponds to a different group value,
    # the mask helps you differentiate missing values before
    # performing sort on the actual values
    _as = np.lexsort(order).astype(np.int64, copy=False)

    if not ascending:
        _as = _as[::-1]

    with nogil:
        # Loop over the length of the value array
        # each incremental i value can be looked up in the _as array
        # that we sorted previously, which gives us the location of
        # that sorted value for retrieval back from the original
        # values / masked_vals arrays
        for i in range(N):
            # dups and sum_ranks will be incremented each loop where
            # the value / group remains the same, and should be reset
            # when either of those change
            # Used to calculate tiebreakers
            dups += 1
            sum_ranks += i - grp_start + 1

            # Update out only when there is a transition of values or labels.
            # When a new value or group is encountered, go back #dups steps(
            # the number of occurrence of current value) and assign the ranks
            # based on the the starting index of the current group (grp_start)
            # and the current index
            if (i == N - 1 or
                    (masked_vals[_as[i]] != masked_vals[_as[i+1]]) or
                    (mask[_as[i]] ^ mask[_as[i+1]]) or
                    (labels[_as[i]] != labels[_as[i+1]])):
                # if keep_na, check for missing values and assign back
                # to the result where appropriate
                if keep_na and mask[_as[i]]:
                    for j in range(i - dups + 1, i + 1):
                        out[_as[j], 0] = NaN
                        grp_na_count = dups
                elif tiebreak == TIEBREAK_AVERAGE:
                    for j in range(i - dups + 1, i + 1):
                        out[_as[j], 0] = sum_ranks / <float64_t>dups
                elif tiebreak == TIEBREAK_MIN:
                    for j in range(i - dups + 1, i + 1):
                        out[_as[j], 0] = i - grp_start - dups + 2
                elif tiebreak == TIEBREAK_MAX:
                    for j in range(i - dups + 1, i + 1):
                        out[_as[j], 0] = i - grp_start + 1
                elif tiebreak == TIEBREAK_FIRST:
                    for j in range(i - dups + 1, i + 1):
                        if ascending:
                            out[_as[j], 0] = j + 1 - grp_start
                        else:
                            out[_as[j], 0] = 2 * i - j - dups + 2 - grp_start
                elif tiebreak == TIEBREAK_DENSE:
                    for j in range(i - dups + 1, i + 1):
                        out[_as[j], 0] = grp_vals_seen

                # look forward to the next value (using the sorting in _as)
                # if the value does not equal the current value then we need to
                # reset the dups and sum_ranks, knowing that a new value is
                # coming up. the conditional also needs to handle nan equality
                # and the end of iteration
                if (i == N - 1 or
                        (masked_vals[_as[i]] != masked_vals[_as[i+1]]) or
                        (mask[_as[i]] ^ mask[_as[i+1]])):
                    dups = sum_ranks = 0
                    grp_vals_seen += 1
                    grp_tie_count += 1

                # Similar to the previous conditional, check now if we are
                # moving to a new group. If so, keep track of the index where
                # the new group occurs, so the tiebreaker calculations can
                # decrement that from their position. fill in the size of each
                # group encountered (used by pct calculations later). also be
                # sure to reset any of the items helping to calculate dups
                if i == N - 1 or labels[_as[i]] != labels[_as[i+1]]:
                    if tiebreak != TIEBREAK_DENSE:
                        for j in range(grp_start, i + 1):
                            grp_sizes[_as[j], 0] = (i - grp_start + 1 -
                                                    grp_na_count)
                    else:
                        for j in range(grp_start, i + 1):
                            grp_sizes[_as[j], 0] = (grp_tie_count -
                                                    (grp_na_count > 0))
                    dups = sum_ranks = 0
                    grp_na_count = 0
                    grp_tie_count = 0
                    grp_start = i + 1
                    grp_vals_seen = 1

        if pct:
            for i in range(N):
                # We don't include NaN values in percentage
                # rankings, so we assign them percentages of NaN.
                if out[i, 0] != out[i, 0] or out[i, 0] == NAN:
                    out[i, 0] = NAN
                elif grp_sizes[i, 0] != 0:
                    out[i, 0] = out[i, 0] / grp_sizes[i, 0]


@cython.wraparound(False)
@cython.boundscheck(False)
def group_last_int64(ndarray[int64_t, ndim=2] out,
                        ndarray[int64_t] counts,
                        ndarray[int64_t, ndim=2] values,
                        ndarray[int64_t] labels,
                        Py_ssize_t min_count=-1):
    """
    Only aggregates on axis=0
    """
    cdef:
        Py_ssize_t i, j, N, K, lab, ncounts = len(counts)
        int64_t val
        ndarray[int64_t, ndim=2] resx
        ndarray[int64_t, ndim=2] nobs

    assert min_count == -1, "'min_count' only used in add and prod"

    if not len(values) == len(labels):
        raise AssertionError("len(index) != len(labels)")

    nobs = np.zeros((<object>out).shape, dtype=np.int64)
    resx = np.empty_like(out)

    N, K = (<object>values).shape

    with nogil:
        for i in range(N):
            lab = labels[i]
            if lab < 0:
                continue

            counts[lab] += 1
            for j in range(K):
                val = values[i, j]

                # not nan
                if val == val and val != NPY_NAT:
                    nobs[lab, j] += 1
                    resx[lab, j] = val

        for i in range(ncounts):
            for j in range(K):
                if nobs[i, j] == 0:
                    out[i, j] = NPY_NAT
                else:
                    out[i, j] = resx[i, j]


@cython.wraparound(False)
@cython.boundscheck(False)
def group_nth_int64(ndarray[int64_t, ndim=2] out,
                       ndarray[int64_t] counts,
                       ndarray[int64_t, ndim=2] values,
                       ndarray[int64_t] labels, int64_t rank,
                       Py_ssize_t min_count=-1):
    """
    Only aggregates on axis=0
    """
    cdef:
        Py_ssize_t i, j, N, K, lab, ncounts = len(counts)
        int64_t val
        ndarray[int64_t, ndim=2] resx
        ndarray[int64_t, ndim=2] nobs

    assert min_count == -1, "'min_count' only used in add and prod"

    if not len(values) == len(labels):
        raise AssertionError("len(index) != len(labels)")

    nobs = np.zeros((<object>out).shape, dtype=np.int64)
    resx = np.empty_like(out)

    N, K = (<object>values).shape

    with nogil:
        for i in range(N):
            lab = labels[i]
            if lab < 0:
                continue

            counts[lab] += 1
            for j in range(K):
                val = values[i, j]

                # not nan
                if val == val and val != NPY_NAT:
                    nobs[lab, j] += 1
                    if nobs[lab, j] == rank:
                        resx[lab, j] = val

        for i in range(ncounts):
            for j in range(K):
                if nobs[i, j] == 0:
                    out[i, j] = NPY_NAT
                else:
                    out[i, j] = resx[i, j]



@cython.boundscheck(False)
@cython.wraparound(False)
def group_rank_int64(ndarray[float64_t, ndim=2] out,
                        ndarray[int64_t, ndim=2] values,
                        ndarray[int64_t] labels,
                        bint is_datetimelike, object ties_method,
                        bint ascending, bint pct, object na_option):
    """
    Provides the rank of values within each group.

    Parameters
    ----------
    out : array of float64_t values which this method will write its results to
    values : array of int64_t values to be ranked
    labels : array containing unique label for each group, with its ordering
        matching up to the corresponding record in `values`
    is_datetimelike : bool, default False
        unused in this method but provided for call compatibility with other
        Cython transformations
    ties_method : {'average', 'min', 'max', 'first', 'dense'}, default
        'average'
        * average: average rank of group
        * min: lowest rank in group
        * max: highest rank in group
        * first: ranks assigned in order they appear in the array
        * dense: like 'min', but rank always increases by 1 between groups
    ascending : boolean, default True
        False for ranks by high (1) to low (N)
        na_option : {'keep', 'top', 'bottom'}, default 'keep'
    pct : boolean, default False
        Compute percentage rank of data within each group
    na_option : {'keep', 'top', 'bottom'}, default 'keep'
        * keep: leave NA values where they are
        * top: smallest rank if ascending
        * bottom: smallest rank if descending

    Notes
    -----
    This method modifies the `out` parameter rather than returning an object
    """
    cdef:
        TiebreakEnumType tiebreak
        Py_ssize_t i, j, N, K, grp_start=0, dups=0, sum_ranks=0
        Py_ssize_t grp_vals_seen=1, grp_na_count=0, grp_tie_count=0
        ndarray[int64_t] _as
        ndarray[float64_t, ndim=2] grp_sizes
        ndarray[int64_t] masked_vals
        ndarray[uint8_t] mask
        bint keep_na
        int64_t nan_fill_val

    tiebreak = tiebreakers[ties_method]
    keep_na = na_option == 'keep'
    N, K = (<object>values).shape
    grp_sizes = np.ones_like(out)

    # Copy values into new array in order to fill missing data
    # with mask, without obfuscating location of missing data
    # in values array
    masked_vals = np.array(values[:, 0], copy=True)
    mask = (masked_vals == NPY_NAT).astype(np.uint8)

    if ascending ^ (na_option == 'top'):
        nan_fill_val = np.iinfo(np.int64).max
        order = (masked_vals, mask, labels)
    else:
        nan_fill_val = np.iinfo(np.int64).min
        order = (masked_vals, ~mask, labels)
    np.putmask(masked_vals, mask, nan_fill_val)

    # lexsort using labels, then mask, then actual values
    # each label corresponds to a different group value,
    # the mask helps you differentiate missing values before
    # performing sort on the actual values
    _as = np.lexsort(order).astype(np.int64, copy=False)

    if not ascending:
        _as = _as[::-1]

    with nogil:
        # Loop over the length of the value array
        # each incremental i value can be looked up in the _as array
        # that we sorted previously, which gives us the location of
        # that sorted value for retrieval back from the original
        # values / masked_vals arrays
        for i in range(N):
            # dups and sum_ranks will be incremented each loop where
            # the value / group remains the same, and should be reset
            # when either of those change
            # Used to calculate tiebreakers
            dups += 1
            sum_ranks += i - grp_start + 1

            # Update out only when there is a transition of values or labels.
            # When a new value or group is encountered, go back #dups steps(
            # the number of occurrence of current value) and assign the ranks
            # based on the the starting index of the current group (grp_start)
            # and the current index
            if (i == N - 1 or
                    (masked_vals[_as[i]] != masked_vals[_as[i+1]]) or
                    (mask[_as[i]] ^ mask[_as[i+1]]) or
                    (labels[_as[i]] != labels[_as[i+1]])):
                # if keep_na, check for missing values and assign back
                # to the result where appropriate
                if keep_na and mask[_as[i]]:
                    for j in range(i - dups + 1, i + 1):
                        out[_as[j], 0] = NaN
                        grp_na_count = dups
                elif tiebreak == TIEBREAK_AVERAGE:
                    for j in range(i - dups + 1, i + 1):
                        out[_as[j], 0] = sum_ranks / <float64_t>dups
                elif tiebreak == TIEBREAK_MIN:
                    for j in range(i - dups + 1, i + 1):
                        out[_as[j], 0] = i - grp_start - dups + 2
                elif tiebreak == TIEBREAK_MAX:
                    for j in range(i - dups + 1, i + 1):
                        out[_as[j], 0] = i - grp_start + 1
                elif tiebreak == TIEBREAK_FIRST:
                    for j in range(i - dups + 1, i + 1):
                        if ascending:
                            out[_as[j], 0] = j + 1 - grp_start
                        else:
                            out[_as[j], 0] = 2 * i - j - dups + 2 - grp_start
                elif tiebreak == TIEBREAK_DENSE:
                    for j in range(i - dups + 1, i + 1):
                        out[_as[j], 0] = grp_vals_seen

                # look forward to the next value (using the sorting in _as)
                # if the value does not equal the current value then we need to
                # reset the dups and sum_ranks, knowing that a new value is
                # coming up. the conditional also needs to handle nan equality
                # and the end of iteration
                if (i == N - 1 or
                        (masked_vals[_as[i]] != masked_vals[_as[i+1]]) or
                        (mask[_as[i]] ^ mask[_as[i+1]])):
                    dups = sum_ranks = 0
                    grp_vals_seen += 1
                    grp_tie_count += 1

                # Similar to the previous conditional, check now if we are
                # moving to a new group. If so, keep track of the index where
                # the new group occurs, so the tiebreaker calculations can
                # decrement that from their position. fill in the size of each
                # group encountered (used by pct calculations later). also be
                # sure to reset any of the items helping to calculate dups
                if i == N - 1 or labels[_as[i]] != labels[_as[i+1]]:
                    if tiebreak != TIEBREAK_DENSE:
                        for j in range(grp_start, i + 1):
                            grp_sizes[_as[j], 0] = (i - grp_start + 1 -
                                                    grp_na_count)
                    else:
                        for j in range(grp_start, i + 1):
                            grp_sizes[_as[j], 0] = (grp_tie_count -
                                                    (grp_na_count > 0))
                    dups = sum_ranks = 0
                    grp_na_count = 0
                    grp_tie_count = 0
                    grp_start = i + 1
                    grp_vals_seen = 1

        if pct:
            for i in range(N):
                # We don't include NaN values in percentage
                # rankings, so we assign them percentages of NaN.
                if out[i, 0] != out[i, 0] or out[i, 0] == NAN:
                    out[i, 0] = NAN
                elif grp_sizes[i, 0] != 0:
                    out[i, 0] = out[i, 0] / grp_sizes[i, 0]


@cython.wraparound(False)
@cython.boundscheck(False)
def group_last_object(ndarray[object, ndim=2] out,
                        ndarray[int64_t] counts,
                        ndarray[object, ndim=2] values,
                        ndarray[int64_t] labels,
                        Py_ssize_t min_count=-1):
    """
    Only aggregates on axis=0
    """
    cdef:
        Py_ssize_t i, j, N, K, lab, ncounts = len(counts)
        object val
        ndarray[object, ndim=2] resx
        ndarray[int64_t, ndim=2] nobs

    assert min_count == -1, "'min_count' only used in add and prod"

    if not len(values) == len(labels):
        raise AssertionError("len(index) != len(labels)")

    nobs = np.zeros((<object>out).shape, dtype=np.int64)
    resx = np.empty((<object>out).shape, dtype=object)

    N, K = (<object>values).shape

    if True:  # make templating happy
        for i in range(N):
            lab = labels[i]
            if lab < 0:
                continue

            counts[lab] += 1
            for j in range(K):
                val = values[i, j]

                # not nan
                if val == val and val != NAN:
                    nobs[lab, j] += 1
                    resx[lab, j] = val

        for i in range(ncounts):
            for j in range(K):
                if nobs[i, j] == 0:
                    out[i, j] = NAN
                else:
                    out[i, j] = resx[i, j]


@cython.wraparound(False)
@cython.boundscheck(False)
def group_nth_object(ndarray[object, ndim=2] out,
                       ndarray[int64_t] counts,
                       ndarray[object, ndim=2] values,
                       ndarray[int64_t] labels, int64_t rank,
                       Py_ssize_t min_count=-1):
    """
    Only aggregates on axis=0
    """
    cdef:
        Py_ssize_t i, j, N, K, lab, ncounts = len(counts)
        object val
        ndarray[object, ndim=2] resx
        ndarray[int64_t, ndim=2] nobs

    assert min_count == -1, "'min_count' only used in add and prod"

    if not len(values) == len(labels):
        raise AssertionError("len(index) != len(labels)")

    nobs = np.zeros((<object>out).shape, dtype=np.int64)
    resx = np.empty((<object>out).shape, dtype=object)

    N, K = (<object>values).shape

    if True:  # make templating happy
        for i in range(N):
            lab = labels[i]
            if lab < 0:
                continue

            counts[lab] += 1
            for j in range(K):
                val = values[i, j]

                # not nan
                if val == val and val != NAN:
                    nobs[lab, j] += 1
                    if nobs[lab, j] == rank:
                        resx[lab, j] = val

        for i in range(ncounts):
            for j in range(K):
                if nobs[i, j] == 0:
                    out[i, j] = NAN
                else:
                    out[i, j] = resx[i, j]



# ----------------------------------------------------------------------
# group_min, group_max
# ----------------------------------------------------------------------

# TODO: consider implementing for more dtypes
ctypedef fused groupby_t:
    float64_t
    float32_t
    int64_t


@cython.wraparound(False)
@cython.boundscheck(False)
def group_max(ndarray[groupby_t, ndim=2] out,
              ndarray[int64_t] counts,
              ndarray[groupby_t, ndim=2] values,
              ndarray[int64_t] labels,
              Py_ssize_t min_count=-1):
    """
    Only aggregates on axis=0
    """
    cdef:
        Py_ssize_t i, j, N, K, lab, ncounts = len(counts)
        groupby_t val, count, nan_val
        ndarray[groupby_t, ndim=2] maxx, nobs

    assert min_count == -1, "'min_count' only used in add and prod"

    if not len(values) == len(labels):
        raise AssertionError("len(index) != len(labels)")

    nobs = np.zeros_like(out)

    maxx = np.empty_like(out)
    if groupby_t is int64_t:
        # Note: evaluated at compile-time
        maxx[:] = -_int64_max
        nan_val = NPY_NAT
    else:
        maxx[:] = -np.inf
        nan_val = NAN

    N, K = (<object>values).shape

    with nogil:
        for i in range(N):
            lab = labels[i]
            if lab < 0:
                continue

            counts[lab] += 1
            for j in range(K):
                val = values[i, j]

                # not nan
                if groupby_t is int64_t:
                    if val != nan_val:
                        nobs[lab, j] += 1
                        if val > maxx[lab, j]:
                            maxx[lab, j] = val
                else:
                    if val == val and val != nan_val:
                        nobs[lab, j] += 1
                        if val > maxx[lab, j]:
                            maxx[lab, j] = val

        for i in range(ncounts):
            for j in range(K):
                if nobs[i, j] == 0:
                    out[i, j] = nan_val
                else:
                    out[i, j] = maxx[i, j]


@cython.wraparound(False)
@cython.boundscheck(False)
def group_min(ndarray[groupby_t, ndim=2] out,
              ndarray[int64_t] counts,
              ndarray[groupby_t, ndim=2] values,
              ndarray[int64_t] labels,
              Py_ssize_t min_count=-1):
    """
    Only aggregates on axis=0
    """
    cdef:
        Py_ssize_t i, j, N, K, lab, ncounts = len(counts)
        groupby_t val, count, nan_val
        ndarray[groupby_t, ndim=2] minx, nobs

    assert min_count == -1, "'min_count' only used in add and prod"

    if not len(values) == len(labels):
        raise AssertionError("len(index) != len(labels)")

    nobs = np.zeros_like(out)

    minx = np.empty_like(out)
    if groupby_t is int64_t:
        minx[:] = _int64_max
        nan_val = NPY_NAT
    else:
        minx[:] = np.inf
        nan_val = NAN

    N, K = (<object>values).shape

    with nogil:
        for i in range(N):
            lab = labels[i]
            if lab < 0:
                continue

            counts[lab] += 1
            for j in range(K):
                val = values[i, j]

                # not nan
                if groupby_t is int64_t:
                    if val != nan_val:
                        nobs[lab, j] += 1
                        if val < minx[lab, j]:
                            minx[lab, j] = val
                else:
                    if val == val and val != nan_val:
                        nobs[lab, j] += 1
                        if val < minx[lab, j]:
                            minx[lab, j] = val

        for i in range(ncounts):
            for j in range(K):
                if nobs[i, j] == 0:
                    out[i, j] = nan_val
                else:
                    out[i, j] = minx[i, j]


@cython.boundscheck(False)
@cython.wraparound(False)
def group_cummin(ndarray[groupby_t, ndim=2] out,
                 ndarray[groupby_t, ndim=2] values,
                 ndarray[int64_t] labels,
                 bint is_datetimelike):
    """
    Only transforms on axis=0
    """
    cdef:
        Py_ssize_t i, j, N, K, size
        groupby_t val, mval
        ndarray[groupby_t, ndim=2] accum
        int64_t lab

    N, K = (<object>values).shape
    accum = np.empty_like(values)
    if groupby_t is int64_t:
        accum[:] = _int64_max
    else:
        accum[:] = np.inf

    with nogil:
        for i in range(N):
            lab = labels[i]

            if lab < 0:
                continue
            for j in range(K):
                val = values[i, j]

                # val = nan
                if groupby_t is int64_t:
                    if is_datetimelike and val == NPY_NAT:
                        out[i, j] = NPY_NAT
                    else:
                        mval = accum[lab, j]
                        if val < mval:
                            accum[lab, j] = mval = val
                        out[i, j] = mval
                else:
                    if val == val:
                        mval = accum[lab, j]
                        if val < mval:
                            accum[lab, j] = mval = val
                        out[i, j] = mval


@cython.boundscheck(False)
@cython.wraparound(False)
def group_cummax(ndarray[groupby_t, ndim=2] out,
                 ndarray[groupby_t, ndim=2] values,
                 ndarray[int64_t] labels,
                 bint is_datetimelike):
    """
    Only transforms on axis=0
    """
    cdef:
        Py_ssize_t i, j, N, K, size
        groupby_t val, mval
        ndarray[groupby_t, ndim=2] accum
        int64_t lab

    N, K = (<object>values).shape
    accum = np.empty_like(values)
    if groupby_t is int64_t:
        accum[:] = -_int64_max
    else:
        accum[:] = -np.inf

    with nogil:
        for i in range(N):
            lab = labels[i]

            if lab < 0:
                continue
            for j in range(K):
                val = values[i, j]

                if groupby_t is int64_t:
                    if is_datetimelike and val == NPY_NAT:
                        out[i, j] = NPY_NAT
                    else:
                        mval = accum[lab, j]
                        if val > mval:
                            accum[lab, j] = mval = val
                        out[i, j] = mval
                else:
                    if val == val:
                        mval = accum[lab, j]
                        if val > mval:
                            accum[lab, j] = mval = val
                        out[i, j] = mval
