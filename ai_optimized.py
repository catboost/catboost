from __future__ import absolute_import, print_function, division
import numpy as np
import warnings

import theano
from theano import Op, Apply
import theano.tensor as T
from theano.scalar import as_scalar
import copy

class MultinomialFromUniform(Op):
    """
    Converts samples from a uniform distribution into samples from a multinomial distribution.
    """

    __props__ = ("odtype",)

    def __init__(self, odtype):
        self.odtype = odtype

    def __str__(self):
        return '%s{%s}' % (self.__class__.__name__, self.odtype)

    def __setstate__(self, dct):
        self.__dict__.update(dct)
        if 'odtype' not in dct:
            self.odtype = 'auto'

    def make_node(self, pvals, unis, n=1):
        pvals = T.as_tensor_variable(pvals)
        unis = T.as_tensor_variable(unis)
        if pvals.ndim != 2:
            raise NotImplementedError('pvals ndim should be 2', pvals.ndim)
        if unis.ndim != 1:
            raise NotImplementedError('unis ndim should be 1', unis.ndim)
        if self.odtype == 'auto':
            odtype = pvals.dtype
        else:
            odtype = self.odtype
        out = T.tensor(dtype=odtype, broadcastable=pvals.type.broadcastable)
        return Apply(self, [pvals, unis, as_scalar(n)], [out])

    def grad(self, ins, outgrads):
        pvals, unis, n = ins
        (gz,) = outgrads
        return [T.zeros_like(x, dtype=theano.config.floatX) if x.dtype in T.discrete_dtypes else T.zeros_like(x) for x in ins]

    def c_code_cache_version(self):
        return (9,)

    def c_code(self, node, name, ins, outs, sub):
        (pvals, unis, n) = ins
        (z,) = outs
        if self.odtype == 'auto':
            t = "PyArray_TYPE(%(pvals)s)" % locals()
        else:
            t = theano.scalar.Scalar(self.odtype).dtype_specs()[1]
            if t.startswith('theano_complex'):
                t = t.replace('theano_complex', 'NPY_COMPLEX')
            else:
                t = t.upper()
        fail = sub['fail']
        return """
        if (PyArray_NDIM(%(pvals)s) != 2)
        {
            PyErr_Format(PyExc_TypeError, "pvals ndim should be 2");
            %(fail)s;
        }
        if (PyArray_NDIM(%(unis)s) != 1)
        {
            PyErr_Format(PyExc_TypeError, "unis ndim should be 1");
            %(fail)s;
        }

        if (PyArray_DIMS(%(unis)s)[0] != (PyArray_DIMS(%(pvals)s)[0] * %(n)s))
        {
            PyErr_Format(PyExc_ValueError, "unis.shape[0] != pvals.shape[0] * n");
            %(fail)s;
        }

        if ((NULL == %(z)s)
            || ((PyArray_DIMS(%(z)s))[0] != (PyArray_DIMS(%(pvals)s))[0])
            || ((PyArray_DIMS(%(z)s))[1] != (PyArray_DIMS(%(pvals)s))[1])
        )
        {
            Py_XDECREF(%(z)s);
            %(z)s = (PyArrayObject*) PyArray_EMPTY(2,
                PyArray_DIMS(%(pvals)s),
                %(t)s,
                0);
            if (!%(z)s)
            {
                PyErr_SetString(PyExc_MemoryError, "failed to alloc z output");
                %(fail)s;
            }
        }

        { // NESTED SCOPE

        const int nb_multi = PyArray_DIMS(%(pvals)s)[0];
        const int nb_outcomes = PyArray_DIMS(%(pvals)s)[1];
        const int n_samples = %(n)s;

        for (int c = 0; c < n_samples; ++c){
            for (int n = 0; n < nb_multi; ++n)
            {
                int waiting = 1;
                double cummul = 0.;
                const dtype_%(unis)s* unis_n = (dtype_%(unis)s*)PyArray_GETPTR1(%(unis)s, c*nb_multi + n);
                for (int m = 0; m < nb_outcomes; ++m)
                {
                    dtype_%(z)s* z_nm = (dtype_%(z)s*)PyArray_GETPTR2(%(z)s, n,m);
                    const dtype_%(pvals)s* pvals_nm = (dtype_%(pvals)s*)PyArray_GETPTR2(%(pvals)s, n,m);
                    cummul += *pvals_nm;
                    if (c == 0)
                    {
                        if (waiting && (cummul > *unis_n))
                        {
                            *z_nm = 1.;
                            waiting = 0;
                        }
                        else
                        {
                            *z_nm = 0.;
                        }
                    }
                    else {
                        if (cummul > *unis_n)
                        {
                            *z_nm = *z_nm + 1.;
                            break;
                        }
                    }
                }
            }
        }
        } // END NESTED SCOPE
        """ % locals()

    def perform(self, node, ins, outs):
        if len(ins) == 2:
            (pvals, unis) = ins
            n_samples = 1
        else:
            (pvals, unis, n_samples) = ins
        (z,) = outs

        if unis.shape[0] != pvals.shape[0] * n_samples:
            raise ValueError("unis.shape[0] != pvals.shape[0] * n_samples",
                             unis.shape[0], pvals.shape[0], n_samples)
        if z[0] is None or z[0].shape != pvals.shape:
            z[0] = np.zeros(pvals.shape, dtype=node.outputs[0].dtype)
        else:
            z[0].fill(0)

        nb_multi = pvals.shape[0]
        for c in range(n_samples):
            for n in range(nb_multi):
                unis_n = unis[c * nb_multi + n]
                cumsum = pvals[n].cumsum(dtype='float64')
                z[0][n, np.searchsorted(cumsum, unis_n)] += 1

class ChoiceFromUniform(MultinomialFromUniform):
    """
    Converts samples from a uniform distribution into samples (without replacement) from a multinomial distribution.
    """

    __props__ = ("odtype", "replace",)

    def __init__(self, odtype, replace=False, *args, **kwargs):
        self.replace = replace
        super(ChoiceFromUniform, self).__init__(odtype=odtype, *args, **kwargs)

    def __setstate__(self, state):
        self.__dict__.update(state)
        if "replace" not in state:
            self.replace = False

    def make_node(self, pvals, unis, n=1):
        pvals = T.as_tensor_variable(pvals)
        unis = T.as_tensor_variable(unis)
        if pvals.ndim != 2:
            raise NotImplementedError('pvals ndim should be 2', pvals.ndim)
        if unis.ndim != 1:
            raise NotImplementedError('unis ndim should be 1', unis.ndim)
        if self.odtype == 'auto':
            odtype = 'int64'
        else:
            odtype = self.odtype
        out = T.tensor(dtype=odtype, broadcastable=pvals.type.broadcastable)
        return Apply(self, [pvals, unis, as_scalar(n)], [out])

    def c_code_cache_version(self):
        return (2,)

    def c_code(self, node, name, ins, outs, sub):
        (pvals, unis, n) = ins
        (z,) = outs
        replace = int(self.replace)
        if self.odtype == 'auto':
            t = "NPY_INT64"
        else:
            t = theano.scalar.Scalar(self.odtype).dtype_specs()[1]
            if t.startswith('theano_complex'):
                t = t.replace('theano_complex', 'NPY_COMPLEX')
            else:
                t = t.upper()
        fail = sub['fail']
        return """
        PyArrayObject* pvals_copy = NULL;

        if (PyArray_NDIM(%(pvals)s) != 2)
        {
            PyErr_Format(PyExc_TypeError, "pvals ndim should be 2");
            %(fail)s;
        }
        if (PyArray_NDIM(%(unis)s) != 1)
        {
            PyErr_Format(PyExc_TypeError, "unis ndim should be 1");
            %(fail)s;
        }

        if ( %(n)s > (PyArray_DIMS(%(pvals)s)[1]) )
        {
            PyErr_Format(PyExc_ValueError, "Cannot sample without replacement n samples bigger than the size of the distribution.");
            %(fail)s;
        }

        if (PyArray_DIMS(%(unis)s)[0] != (PyArray_DIMS(%(pvals)s)[0] * %(n)s))
        {
            PyErr_Format(PyExc_ValueError, "unis.shape[0] != pvals.shape[0] * n");
            %(fail)s;
        }

        if ((NULL == %(z)s)
            || ((PyArray_DIMS(%(z)s))[0] != (PyArray_DIMS(%(pvals)s))[0])
            || ((PyArray_DIMS(%(z)s))[1] != (PyArray_DIMS(%(pvals)s))[1])
        )
        {
            Py_XDECREF(%(z)s);
            %(z)s = (PyArrayObject*) PyArray_EMPTY(2,
                PyArray_DIMS(%(pvals)s),
                %(t)s,
                0);
            if (!%(z)s)
            {
                PyErr_SetString(PyExc_MemoryError, "failed to alloc z output");
                %(fail)s;
            }
        }

        if (!%(replace)s) {
            PyObject* pvals_copy = PyArray_Copy(%(pvals)s);
            if (!pvals_copy)
            {
                PyErr_SetString(PyExc_MemoryError, "failed to copy pvals");
                %(fail)s;
            }

            npy_intp nb_multi = PyArray_DIMS(%(pvals)s)[0];
            npy_intp nb_outcomes = PyArray_DIMS(%(pvals)s)[1];
            npy_intp n_samples = %(n)s;

            for (int c = 0; c < n_samples; ++c){
                for (int n = 0; n < nb_multi; ++n)
                {
                    npy_intp outcome = -1;
                    double cummul = 0.;
                    const dtype_%(unis)s* unis_n = (dtype_%(unis)s*)PyArray_GETPTR1(%(unis)s, c*nb_multi + n);
                    for (int m = 0; m < nb_outcomes; ++m)
                    {
                        dtype_%(pvals)s* pvals_nm = (dtype_%(pvals)s*)PyArray_GETPTR2(pvals_copy, n,m);
                        cummul += *pvals_nm;
                        if (cummul > *unis_n)
                        {
                            outcome = m;
                            break;
                        }
                    }
                    if (outcome != -1)
                    {
                        dtype_%(z)s* z_nm = (dtype_%(z)s*)PyArray_GETPTR2(%(z)s, n, outcome);
                        *z_nm += 1.;
                    }
                }
            }
            Py_XDECREF(pvals_copy);
        } else {
            npy_intp nb_multi = PyArray_DIMS(%(pvals)s)[0];
            npy_intp nb_outcomes = PyArray_DIMS(%(pvals)s)[1];
            npy_intp n_samples = %(n)s;

            for (int c = 0; c < n_samples; ++c){
                for (int n = 0; n < nb_multi; ++n)
                {
                    double cummul = 0.;
                    const dtype_%(unis)s* unis_n = (dtype_%(unis)s*)PyArray_GETPTR1(%(unis)s, c*nb_multi + n);
                    int outcome = -1;
                    for (int m = 0; m < nb_outcomes; ++m)
                    {
                        dtype_%(pvals)s* pvals_nm = (dtype_%(pvals)s*)PyArray_GETPTR2(%(pvals)s, n,m);
                        cummul += *pvals_nm;
                        if (cummul > *unis_n)
                        {
                            outcome = m;
                            break;
                        }
                    }
                    if (outcome != -1)
                    {
                        dtype_%(z)s* z_nm = (dtype_%(z)s*)PyArray_GETPTR2(%(z)s, n, outcome);
                        *z_nm += 1.;
                    }
                }
            }
        }
        """ % locals()

    def perform(self, node, ins, outs):
        if len(ins) == 2:
            (pvals, unis) = ins
            n_samples = 1
        else:
            (pvals, unis, n_samples) = ins
        (z,) = outs

        if unis.shape[0] != pvals.shape[0] * n_samples:
            raise ValueError("unis.shape[0] != pvals.shape[0] * n_samples",
                             unis.shape[0], pvals.shape[0], n_samples)
        if z[0] is None or z[0].shape != pvals.shape:
            z[0] = np.zeros(pvals.shape, dtype=node.outputs[0].dtype)
        else:
            z[0].fill(0)

        nb_multi = pvals.shape[0]
        for c in range(n_samples):
            for n in range(nb_multi):
                unis_n = unis[c * nb_multi + n]
                cumsum = pvals[n].cumsum(dtype='float64')
                outcome = np.searchsorted(cumsum, unis_n)
                if self.replace:
                    z[0][n, outcome] += 1
                else:
                    if z[0][n].sum() == 0:
                        z[0][n, outcome] += 1

class MultinomialSampler:
    def __init__(self, num_samples, num_outcomes):
        self.num_samples = num_samples
        self.num_outcomes = num_outcomes
        self.data = None

    def generate(self):
        self.data = np.random.multinomial(self.num_samples, np.ones(self.num_outcomes) / self.num_outcomes, size=1)

    def get_sample(self):
        if self.data is None:
            raise ValueError("No samples generated yet.")
        return self.data

# Add the new functions for advanced AI features
def sample_probability_distribution(num_samples, pvals):
    """
    Sample from a given probability distribution using a multinomial distribution.
    
    Args:
    num_samples (int): Number of samples to generate.
    pvals (array-like): Probability values of the multinomial distribution.
    
    Returns:
    np.ndarray: Sampled values from the multinomial distribution.
    """
    num_outcomes = len(pvals)
    return np.random.multinomial(num_samples, pvals)

def evaluate_multinomial_samples(samples, pvals):
    """
    Evaluate the given samples against the expected probability distribution.
    
    Args:
    samples (np.ndarray): Sampled values from the multinomial distribution.
    pvals (array-like): Probability values of the multinomial distribution.
    
    Returns:
    float: Log-likelihood of the samples given the probability distribution.
    """
    pvals = np.array(pvals)
    total_samples = samples.sum()
    probabilities = pvals ** samples
    likelihood = probabilities.prod()
    return np.log(likelihood)

if __name__ == "__main__":
    # Example usage of MultinomialSampler
    sampler = MultinomialSampler(num_samples=10, num_outcomes=5)
    sampler.generate()
    print("Generated sample:", sampler.get_sample())

    # Example usage of the new functions
    pvals = [0.1, 0.2, 0.3, 0.4]
    num_samples = 1000
    sampled_values = sample_probability_distribution(num_samples, pvals)
    print("Sampled values:", sampled_values)

    log_likelihood = evaluate_multinomial_samples(sampled_values, pvals)
    print("Log-likelihood of the samples:", log_likelihood)
