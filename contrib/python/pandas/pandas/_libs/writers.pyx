# -*- coding: utf-8 -*-

import cython
from cython import Py_ssize_t

from cpython cimport PyBytes_GET_SIZE, PyUnicode_GET_SIZE

try:
    from cpython cimport PyString_GET_SIZE
except ImportError:
    from cpython cimport PyUnicode_GET_SIZE as PyString_GET_SIZE

import numpy as np
from numpy cimport ndarray, uint8_t


ctypedef fused pandas_string:
    str
    unicode
    bytes


@cython.boundscheck(False)
@cython.wraparound(False)
def write_csv_rows(list data, ndarray data_index,
                   Py_ssize_t nlevels, ndarray cols, object writer):
    """
    Write the given data to the writer object, pre-allocating where possible
    for performance improvements.

    Parameters
    ----------
    data : list
    data_index : ndarray
    nlevels : int
    cols : ndarray
    writer : object
    """
    # In crude testing, N>100 yields little marginal improvement
    cdef:
        Py_ssize_t i, j, k = len(data_index), N = 100, ncols = len(cols)
        list rows

    # pre-allocate rows
    rows = [[None] * (nlevels + ncols) for _ in range(N)]

    if nlevels == 1:
        for j in range(k):
            row = rows[j % N]
            row[0] = data_index[j]
            for i in range(ncols):
                row[1 + i] = data[i][j]

            if j >= N - 1 and j % N == N - 1:
                writer.writerows(rows)
    elif nlevels > 1:
        for j in range(k):
            row = rows[j % N]
            row[:nlevels] = list(data_index[j])
            for i in range(ncols):
                row[nlevels + i] = data[i][j]

            if j >= N - 1 and j % N == N - 1:
                writer.writerows(rows)
    else:
        for j in range(k):
            row = rows[j % N]
            for i in range(ncols):
                row[i] = data[i][j]

            if j >= N - 1 and j % N == N - 1:
                writer.writerows(rows)

    if j >= 0 and (j < N - 1 or (j % N) != N - 1):
        writer.writerows(rows[:((j + 1) % N)])


@cython.boundscheck(False)
@cython.wraparound(False)
def convert_json_to_lines(object arr):
    """
    replace comma separated json with line feeds, paying special attention
    to quotes & brackets
    """
    cdef:
        Py_ssize_t i = 0, num_open_brackets_seen = 0, length
        bint in_quotes = 0, is_escaping = 0
        ndarray[uint8_t, ndim=1] narr
        unsigned char val, newline, comma, left_bracket, right_bracket, quote
        unsigned char backslash

    newline = ord('\n')
    comma = ord(',')
    left_bracket = ord('{')
    right_bracket = ord('}')
    quote = ord('"')
    backslash = ord('\\')

    narr = np.frombuffer(arr.encode('utf-8'), dtype='u1').copy()
    length = narr.shape[0]
    for i in range(length):
        val = narr[i]
        if val == quote and i > 0 and not is_escaping:
            in_quotes = ~in_quotes
        if val == backslash or is_escaping:
            is_escaping = ~is_escaping
        if val == comma:  # commas that should be \n
            if num_open_brackets_seen == 0 and not in_quotes:
                narr[i] = newline
        elif val == left_bracket:
            if not in_quotes:
                num_open_brackets_seen += 1
        elif val == right_bracket:
            if not in_quotes:
                num_open_brackets_seen -= 1

    return narr.tostring().decode('utf-8')


# stata, pytables
@cython.boundscheck(False)
@cython.wraparound(False)
def max_len_string_array(pandas_string[:] arr) -> Py_ssize_t:
    """ return the maximum size of elements in a 1-dim string array """
    cdef:
        Py_ssize_t i, m = 0, l = 0, length = arr.shape[0]
        pandas_string val

    for i in range(length):
        val = arr[i]
        if isinstance(val, str):
            l = PyString_GET_SIZE(val)
        elif isinstance(val, bytes):
            l = PyBytes_GET_SIZE(val)
        elif isinstance(val, unicode):
            l = PyUnicode_GET_SIZE(val)

        if l > m:
            m = l

    return m


# ------------------------------------------------------------------
# PyTables Helpers


@cython.boundscheck(False)
@cython.wraparound(False)
def string_array_replace_from_nan_rep(
        ndarray[object, ndim=1] arr, object nan_rep,
        object replace=None):
    """
    Replace the values in the array with 'replacement' if
    they are 'nan_rep'. Return the same array.
    """
    cdef:
        Py_ssize_t length = len(arr), i = 0

    if replace is None:
        replace = np.nan

    for i in range(length):
        if arr[i] == nan_rep:
            arr[i] = replace

    return arr
