from libc.math cimport log, exp

cimport numpy as cnp

cnp.import_array()
ctypedef cnp.float64_t DTYPE_t


cdef inline DTYPE_t _inner_log_logistic_sigmoid(const DTYPE_t x):
    """Log of the logistic sigmoid function log(1 / (1 + e ** -x))"""
    if x > 0:
        return -log(1. + exp(-x))
    else:
        return x - log(1. + exp(x))


def _log_logistic_sigmoid(unsigned int n_samples,
                          unsigned int n_features,
                          DTYPE_t[:, :] X,
                          DTYPE_t[:, :] out):
    cdef:
        unsigned int i
        unsigned int j

    for i in range(n_samples):
        for j in range(n_features):
            out[i, j] = _inner_log_logistic_sigmoid(X[i, j])
    return out
