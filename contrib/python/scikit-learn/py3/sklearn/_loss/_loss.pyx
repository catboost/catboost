"""
WARNING: Do not edit `sklearn/_loss/_loss.pyx` file directly, as it is generated from
`sklearn/_loss/_loss.pyx.tp`. Changes must be made there.
"""
#------------------------------------------------------------------------------

# Design:
# See https://github.com/scikit-learn/scikit-learn/issues/15123 for reasons.
# a) Merge link functions into loss functions for speed and numerical
#    stability, i.e. use raw_prediction instead of y_pred in signature.
# b) Pure C functions (nogil) calculate single points (single sample)
# c) Wrap C functions in a loop to get Python functions operating on ndarrays.
#   - Write loops manually---use Tempita for this.
#     Reason: There is still some performance overhead when using a wrapper
#     function "wrap" that carries out the loop and gets as argument a function
#     pointer to one of the C functions from b), e.g.
#     wrap(closs_half_poisson, y_true, ...)
#   - Pass n_threads as argument to prange and propagate option to all callers.
# d) Provide classes (Cython extension types) per loss (names start with Cy) in
#    order to have semantical structured objects.
#    - Member functions for single points just call the C function from b).
#      These are used e.g. in SGD `_plain_sgd`.
#    - Member functions operating on ndarrays, see c), looping over calls to C
#      functions from b).
# e) Provide convenience Python classes that compose from these extension types
#    elsewhere (see loss.py)
#    - Example: loss.gradient calls CyLoss.gradient but does some input
#      checking like None -> np.empty().
#
# Note: We require 1-dim ndarrays to be contiguous.
# TODO: Use const memoryviews with fused types with Cython 3.0 where
#       appropriate (arguments marked by "# IN").

from cython.parallel import parallel, prange
import numpy as np

from libc.math cimport exp, fabs, log, log1p, pow
from libc.stdlib cimport malloc, free


# -------------------------------------
# Helper functions
# -------------------------------------
# Numerically stable version of log(1 + exp(x)) for double precision, see Eq. (10) of
# https://cran.r-project.org/web/packages/Rmpfr/vignettes/log1mexp-note.pdf
# Note: The only important cutoff is at x = 18. All others are to save computation
# time. Compared to the reference, we add the additional case distinction x <= -2 in
# order to use log instead of log1p for improved performance. As with the other
# cutoffs, this is accurate within machine precision of double.
cdef inline double log1pexp(double x) nogil:
    if x <= -37:
        return exp(x)
    elif x <= -2:
        return log1p(exp(x))
    elif x <= 18:
        return log(1. + exp(x))
    elif x <= 33.3:
        return x + exp(-x)
    else:
        return x


cdef inline void sum_exp_minus_max(
    const int i,
    Y_DTYPE_C[:, :] raw_prediction,  # IN
    Y_DTYPE_C *p                     # OUT
) nogil:
    # Thread local buffers are used to stores results of this function via p.
    # The results are stored as follows:
    #     p[k] = exp(raw_prediction_i_k - max_value) for k = 0 to n_classes-1
    #     p[-2] = max(raw_prediction_i_k, k = 0 to n_classes-1)
    #     p[-1] = sum(p[k], k = 0 to n_classes-1) = sum of exponentials
    # len(p) must be n_classes + 2
    # Notes:
    # - Using "by reference" arguments doesn't work well, therefore we use a
    #   longer p, see https://github.com/cython/cython/issues/1863
    # - i needs to be passed (and stays constant) because otherwise Cython does
    #   not generate optimal code, see
    #   https://github.com/scikit-learn/scikit-learn/issues/17299
    # - We do not normalize p by calculating p[k] = p[k] / sum_exps.
    #   This helps to save one loop over k.
    cdef:
        int k
        int n_classes = raw_prediction.shape[1]
        double max_value = raw_prediction[i, 0]
        double sum_exps = 0
    for k in range(1, n_classes):
        # Compute max value of array for numerical stability
        if max_value < raw_prediction[i, k]:
            max_value = raw_prediction[i, k]

    for k in range(n_classes):
        p[k] = exp(raw_prediction[i, k] - max_value)
        sum_exps += p[k]

    p[n_classes] = max_value     # same as p[-2]
    p[n_classes + 1] = sum_exps  # same as p[-1]


# -------------------------------------
# Single point inline C functions
# -------------------------------------
# Half Squared Error
cdef inline double closs_half_squared_error(
    double y_true,
    double raw_prediction
) nogil:
    return 0.5 * (raw_prediction - y_true) * (raw_prediction - y_true)


cdef inline double cgradient_half_squared_error(
    double y_true,
    double raw_prediction
) nogil:
    return raw_prediction - y_true


cdef inline double_pair cgrad_hess_half_squared_error(
    double y_true,
    double raw_prediction
) nogil:
    cdef double_pair gh
    gh.val1 = raw_prediction - y_true  # gradient
    gh.val2 = 1.                       # hessian
    return gh


# Absolute Error
cdef inline double closs_absolute_error(
    double y_true,
    double raw_prediction
) nogil:
    return fabs(raw_prediction - y_true)


cdef inline double cgradient_absolute_error(
    double y_true,
    double raw_prediction
) nogil:
    return 1. if raw_prediction > y_true else -1.


cdef inline double_pair cgrad_hess_absolute_error(
    double y_true,
    double raw_prediction
) nogil:
    cdef double_pair gh
    # Note that exact hessian = 0 almost everywhere. Optimization routines like
    # in HGBT, however, need a hessian > 0. Therefore, we assign 1.
    gh.val1 = 1. if raw_prediction > y_true else -1.  # gradient
    gh.val2 = 1.                                      # hessian
    return gh


# Quantile Loss / Pinball Loss
cdef inline double closs_pinball_loss(
    double y_true,
    double raw_prediction,
    double quantile
) nogil:
    return (quantile * (y_true - raw_prediction) if y_true >= raw_prediction
            else (1. - quantile) * (raw_prediction - y_true))


cdef inline double cgradient_pinball_loss(
    double y_true,
    double raw_prediction,
    double quantile
) nogil:
    return -quantile if y_true >=raw_prediction else 1. - quantile


cdef inline double_pair cgrad_hess_pinball_loss(
    double y_true,
    double raw_prediction,
    double quantile
) nogil:
    cdef double_pair gh
    # Note that exact hessian = 0 almost everywhere. Optimization routines like
    # in HGBT, however, need a hessian > 0. Therefore, we assign 1.
    gh.val1 = -quantile if y_true >=raw_prediction else 1. - quantile  # gradient
    gh.val2 = 1.                                                       # hessian
    return gh


# Half Poisson Deviance with Log-Link, dropping constant terms
cdef inline double closs_half_poisson(
    double y_true,
    double raw_prediction
) nogil:
    return exp(raw_prediction) - y_true * raw_prediction


cdef inline double cgradient_half_poisson(
    double y_true,
    double raw_prediction
) nogil:
    # y_pred - y_true
    return exp(raw_prediction) - y_true


cdef inline double_pair closs_grad_half_poisson(
    double y_true,
    double raw_prediction
) nogil:
    cdef double_pair lg
    lg.val2 = exp(raw_prediction)                # used as temporary
    lg.val1 = lg.val2 - y_true * raw_prediction  # loss
    lg.val2 -= y_true                            # gradient
    return lg


cdef inline double_pair cgrad_hess_half_poisson(
    double y_true,
    double raw_prediction
) nogil:
    cdef double_pair gh
    gh.val2 = exp(raw_prediction)  # hessian
    gh.val1 = gh.val2 - y_true     # gradient
    return gh


# Half Gamma Deviance with Log-Link, dropping constant terms
cdef inline double closs_half_gamma(
    double y_true,
    double raw_prediction
) nogil:
    return raw_prediction + y_true * exp(-raw_prediction)


cdef inline double cgradient_half_gamma(
    double y_true,
    double raw_prediction
) nogil:
    return 1. - y_true * exp(-raw_prediction)


cdef inline double_pair closs_grad_half_gamma(
    double y_true,
    double raw_prediction
) nogil:
    cdef double_pair lg
    lg.val2 = exp(-raw_prediction)               # used as temporary
    lg.val1 = raw_prediction + y_true * lg.val2  # loss
    lg.val2 = 1. - y_true * lg.val2              # gradient
    return lg


cdef inline double_pair cgrad_hess_half_gamma(
    double y_true,
    double raw_prediction
) nogil:
    cdef double_pair gh
    gh.val2 = exp(-raw_prediction)   # used as temporary
    gh.val1 = 1. - y_true * gh.val2  # gradient
    gh.val2 *= y_true                # hessian
    return gh


# Half Tweedie Deviance with Log-Link, dropping constant terms
# Note that by dropping constants this is no longer continuous in parameter power.
cdef inline double closs_half_tweedie(
    double y_true,
    double raw_prediction,
    double power
) nogil:
    if power == 0.:
        return closs_half_squared_error(y_true, exp(raw_prediction))
    elif power == 1.:
        return closs_half_poisson(y_true, raw_prediction)
    elif power == 2.:
        return closs_half_gamma(y_true, raw_prediction)
    else:
        return (exp((2. - power) * raw_prediction) / (2. - power)
                - y_true * exp((1. - power) * raw_prediction) / (1. - power))


cdef inline double cgradient_half_tweedie(
    double y_true,
    double raw_prediction,
    double power
) nogil:
    cdef double exp1
    if power == 0.:
        exp1 = exp(raw_prediction)
        return exp1 * (exp1 - y_true)
    elif power == 1.:
        return cgradient_half_poisson(y_true, raw_prediction)
    elif power == 2.:
        return cgradient_half_gamma(y_true, raw_prediction)
    else:
        return (exp((2. - power) * raw_prediction)
                - y_true * exp((1. - power) * raw_prediction))


cdef inline double_pair closs_grad_half_tweedie(
    double y_true,
    double raw_prediction,
    double power
) nogil:
    cdef double_pair lg
    cdef double exp1, exp2
    if power == 0.:
        exp1 = exp(raw_prediction)
        lg.val1 = closs_half_squared_error(y_true, exp1)  # loss
        lg.val2 = exp1 * (exp1 - y_true)                  # gradient
    elif power == 1.:
        return closs_grad_half_poisson(y_true, raw_prediction)
    elif power == 2.:
        return closs_grad_half_gamma(y_true, raw_prediction)
    else:
        exp1 = exp((1. - power) * raw_prediction)
        exp2 = exp((2. - power) * raw_prediction)
        lg.val1 = exp2 / (2. - power) - y_true * exp1 / (1. - power)  # loss
        lg.val2 = exp2 - y_true * exp1                                # gradient
    return lg


cdef inline double_pair cgrad_hess_half_tweedie(
    double y_true,
    double raw_prediction,
    double power
) nogil:
    cdef double_pair gh
    cdef double exp1, exp2
    if power == 0.:
        exp1 = exp(raw_prediction)
        gh.val1 = exp1 * (exp1 - y_true)      # gradient
        gh.val2 = exp1 * (2 * exp1 - y_true)  # hessian
    elif power == 1.:
        return cgrad_hess_half_poisson(y_true, raw_prediction)
    elif power == 2.:
        return cgrad_hess_half_gamma(y_true, raw_prediction)
    else:
        exp1 = exp((1. - power) * raw_prediction)
        exp2 = exp((2. - power) * raw_prediction)
        gh.val1 = exp2 - y_true * exp1                                # gradient
        gh.val2 = (2. - power) * exp2 - (1. - power) * y_true * exp1  # hessian
    return gh


# Half Tweedie Deviance with identity link, without dropping constant terms!
# Therefore, best loss value is zero.
cdef inline double closs_half_tweedie_identity(
    double y_true,
    double raw_prediction,
    double power
) nogil:
    cdef double tmp
    if power == 0.:
        return closs_half_squared_error(y_true, raw_prediction)
    elif power == 1.:
        if y_true == 0:
            return raw_prediction
        else:
            return y_true * log(y_true/raw_prediction) + raw_prediction - y_true
    elif power == 2.:
        return log(raw_prediction/y_true) + y_true/raw_prediction - 1.
    else:
        tmp = pow(raw_prediction, 1. - power)
        tmp = raw_prediction * tmp / (2. - power) - y_true * tmp / (1. - power)
        if y_true > 0:
            tmp += pow(y_true, 2. - power) / ((1. - power) * (2. - power))
        return tmp


cdef inline double cgradient_half_tweedie_identity(
    double y_true,
    double raw_prediction,
    double power
) nogil:
    if power == 0.:
        return raw_prediction - y_true
    elif power == 1.:
        return 1. - y_true / raw_prediction
    elif power == 2.:
        return (raw_prediction - y_true) / (raw_prediction * raw_prediction)
    else:
        return pow(raw_prediction, -power) * (raw_prediction - y_true)


cdef inline double_pair closs_grad_half_tweedie_identity(
    double y_true,
    double raw_prediction,
    double power
) nogil:
    cdef double_pair lg
    cdef double tmp
    if power == 0.:
        lg.val2 = raw_prediction - y_true  # gradient
        lg.val1 = 0.5 * lg.val2 * lg.val2  # loss
    elif power == 1.:
        if y_true == 0:
            lg.val1 = raw_prediction
        else:
            lg.val1 = (y_true * log(y_true/raw_prediction)  # loss
                       + raw_prediction - y_true)
        lg.val2 = 1. - y_true / raw_prediction              # gradient
    elif power == 2.:
        lg.val1 = log(raw_prediction/y_true) + y_true/raw_prediction - 1.  # loss
        tmp = raw_prediction * raw_prediction
        lg.val2 = (raw_prediction - y_true) / tmp                          # gradient
    else:
        tmp = pow(raw_prediction, 1. - power)
        lg.val1 = (raw_prediction * tmp / (2. - power)  # loss
                   - y_true * tmp / (1. - power))
        if y_true > 0:
            lg.val1 += (pow(y_true, 2. - power)
                        / ((1. - power) * (2. - power)))
        lg.val2 = tmp * (1. - y_true / raw_prediction)    # gradient
    return lg


cdef inline double_pair cgrad_hess_half_tweedie_identity(
    double y_true,
    double raw_prediction,
    double power
) nogil:
    cdef double_pair gh
    cdef double tmp
    if power == 0.:
        gh.val1 = raw_prediction - y_true  # gradient
        gh.val2 = 1.                       # hessian
    elif power == 1.:
        gh.val1 = 1. - y_true / raw_prediction                # gradient
        gh.val2 = y_true / (raw_prediction * raw_prediction)  # hessian
    elif power == 2.:
        tmp = raw_prediction * raw_prediction
        gh.val1 = (raw_prediction - y_true) / tmp             # gradient
        gh.val2 = (-1. + 2. * y_true / raw_prediction) / tmp  # hessian
    else:
        tmp = pow(raw_prediction, -power)
        gh.val1 = tmp * (raw_prediction - y_true)                         # gradient
        gh.val2 = tmp * ((1. - power) + power * y_true / raw_prediction)  # hessian
    return gh


# Half Binomial deviance with logit-link, aka log-loss or binary cross entropy
cdef inline double closs_half_binomial(
    double y_true,
    double raw_prediction
) nogil:
    # log1p(exp(raw_prediction)) - y_true * raw_prediction
    return log1pexp(raw_prediction) - y_true * raw_prediction


cdef inline double cgradient_half_binomial(
    double y_true,
    double raw_prediction
) nogil:
    # y_pred - y_true = expit(raw_prediction) - y_true
    # Numerically more stable, see
    # http://fa.bianp.net/blog/2019/evaluate_logistic/
    #     if raw_prediction < 0:
    #         exp_tmp = exp(raw_prediction)
    #         return ((1 - y_true) * exp_tmp - y_true) / (1 + exp_tmp)
    #     else:
    #         exp_tmp = exp(-raw_prediction)
    #         return ((1 - y_true) - y_true * exp_tmp) / (1 + exp_tmp)
    # Note that optimal speed would be achieved, at the cost of precision, by
    #     return expit(raw_prediction) - y_true
    # i.e. no "if else" and an own inline implementation of expit instead of
    #     from scipy.special.cython_special cimport expit
    # The case distinction raw_prediction < 0 in the stable implementation
    # does not provide significant better precision. Therefore we go without
    # it.
    cdef double exp_tmp
    exp_tmp = exp(-raw_prediction)
    return ((1 - y_true) - y_true * exp_tmp) / (1 + exp_tmp)


cdef inline double_pair closs_grad_half_binomial(
    double y_true,
    double raw_prediction
) nogil:
    cdef double_pair lg
    if raw_prediction <= 0:
        lg.val2 = exp(raw_prediction)  # used as temporary
        if raw_prediction <= -37:
            lg.val1 = lg.val2 - y_true * raw_prediction              # loss
        else:
            lg.val1 = log1p(lg.val2) - y_true * raw_prediction       # loss
        lg.val2 = ((1 - y_true) * lg.val2 - y_true) / (1 + lg.val2)  # gradient
    else:
        lg.val2 = exp(-raw_prediction)  # used as temporary
        if raw_prediction <= 18:
            # log1p(exp(x)) = log(1 + exp(x)) = x + log1p(exp(-x))
            lg.val1 = log1p(lg.val2) + (1 - y_true) * raw_prediction  # loss
        else:
            lg.val1 = lg.val2 + (1 - y_true) * raw_prediction         # loss
        lg.val2 = ((1 - y_true) - y_true * lg.val2) / (1 + lg.val2)   # gradient
    return lg


cdef inline double_pair cgrad_hess_half_binomial(
    double y_true,
    double raw_prediction
) nogil:
    # with y_pred = expit(raw)
    # hessian = y_pred * (1 - y_pred) = exp(raw) / (1 + exp(raw))**2
    #                                 = exp(-raw) / (1 + exp(-raw))**2
    cdef double_pair gh
    gh.val2 = exp(-raw_prediction)  # used as temporary
    gh.val1 = ((1 - y_true) - y_true * gh.val2) / (1 + gh.val2)  # gradient
    gh.val2 = gh.val2 / (1 + gh.val2)**2                         # hessian
    return gh


# ---------------------------------------------------
# Extension Types for Loss Functions of 1-dim targets
# ---------------------------------------------------
cdef class CyLossFunction:
    """Base class for convex loss functions."""

    cdef double cy_loss(self, double y_true, double raw_prediction) nogil:
        """Compute the loss for a single sample.

        Parameters
        ----------
        y_true : double
            Observed, true target value.
        raw_prediction : double
            Raw prediction value (in link space).

        Returns
        -------
        double
            The loss evaluated at `y_true` and `raw_prediction`.
        """
        pass

    cdef double cy_gradient(self, double y_true, double raw_prediction) nogil:
        """Compute gradient of loss w.r.t. raw_prediction for a single sample.

        Parameters
        ----------
        y_true : double
            Observed, true target value.
        raw_prediction : double
            Raw prediction value (in link space).

        Returns
        -------
        double
            The derivative of the loss function w.r.t. `raw_prediction`.
        """
        pass

    cdef double_pair cy_grad_hess(self, double y_true, double raw_prediction) nogil:
        """Compute gradient and hessian.

        Gradient and hessian of loss w.r.t. raw_prediction for a single sample.

        This is usually diagonal in raw_prediction_i and raw_prediction_j.
        Therefore, we return the diagonal element i=j.

        For a loss with a non-canonical link, this might implement the diagonal
        of the Fisher matrix (=expected hessian) instead of the hessian.

        Parameters
        ----------
        y_true : double
            Observed, true target value.
        raw_prediction : double
            Raw prediction value (in link space).

        Returns
        -------
        double_pair
            Gradient and hessian of the loss function w.r.t. `raw_prediction`.
        """
        pass

    # Note: With Cython 3.0, fused types can be used together with const:
    #       const Y_DTYPE_C double[::1] y_true
    # See release notes 3.0.0 alpha1
    # https://cython.readthedocs.io/en/latest/src/changes.html#alpha-1-2020-04-12
    def loss(
        self,
        Y_DTYPE_C[::1] y_true,          # IN
        Y_DTYPE_C[::1] raw_prediction,  # IN
        Y_DTYPE_C[::1] sample_weight,   # IN
        G_DTYPE_C[::1] loss_out,        # OUT
        int n_threads=1
    ):
        """Compute the pointwise loss value for each input.

        Parameters
        ----------
        y_true : array of shape (n_samples,)
            Observed, true target values.
        raw_prediction : array of shape (n_samples,)
            Raw prediction values (in link space).
        sample_weight : array of shape (n_samples,) or None
            Sample weights.
        loss_out : array of shape (n_samples,)
            A location into which the result is stored.
        n_threads : int
            Number of threads used by OpenMP (if any).

        Returns
        -------
        loss : array of shape (n_samples,)
            Element-wise loss function.
        """
        pass

    def gradient(
        self,
        Y_DTYPE_C[::1] y_true,          # IN
        Y_DTYPE_C[::1] raw_prediction,  # IN
        Y_DTYPE_C[::1] sample_weight,   # IN
        G_DTYPE_C[::1] gradient_out,    # OUT
        int n_threads=1
    ):
        """Compute gradient of loss w.r.t raw_prediction for each input.

        Parameters
        ----------
        y_true : array of shape (n_samples,)
            Observed, true target values.
        raw_prediction : array of shape (n_samples,)
            Raw prediction values (in link space).
        sample_weight : array of shape (n_samples,) or None
            Sample weights.
        gradient_out : array of shape (n_samples,)
            A location into which the result is stored.
        n_threads : int
            Number of threads used by OpenMP (if any).

        Returns
        -------
        gradient : array of shape (n_samples,)
            Element-wise gradients.
        """
        pass

    def loss_gradient(
        self,
        Y_DTYPE_C[::1] y_true,          # IN
        Y_DTYPE_C[::1] raw_prediction,  # IN
        Y_DTYPE_C[::1] sample_weight,   # IN
        G_DTYPE_C[::1] loss_out,        # OUT
        G_DTYPE_C[::1] gradient_out,    # OUT
        int n_threads=1
    ):
        """Compute loss and gradient of loss w.r.t raw_prediction.

        Parameters
        ----------
        y_true : array of shape (n_samples,)
            Observed, true target values.
        raw_prediction : array of shape (n_samples,)
            Raw prediction values (in link space).
        sample_weight : array of shape (n_samples,) or None
            Sample weights.
        loss_out : array of shape (n_samples,) or None
            A location into which the element-wise loss is stored.
        gradient_out : array of shape (n_samples,)
            A location into which the gradient is stored.
        n_threads : int
            Number of threads used by OpenMP (if any).

        Returns
        -------
        loss : array of shape (n_samples,)
            Element-wise loss function.

        gradient : array of shape (n_samples,)
            Element-wise gradients.
        """
        self.loss(y_true, raw_prediction, sample_weight, loss_out, n_threads)
        self.gradient(y_true, raw_prediction, sample_weight, gradient_out, n_threads)
        return np.asarray(loss_out), np.asarray(gradient_out)

    def gradient_hessian(
        self,
        Y_DTYPE_C[::1] y_true,          # IN
        Y_DTYPE_C[::1] raw_prediction,  # IN
        Y_DTYPE_C[::1] sample_weight,   # IN
        G_DTYPE_C[::1] gradient_out,    # OUT
        G_DTYPE_C[::1] hessian_out,     # OUT
        int n_threads=1
    ):
        """Compute gradient and hessian of loss w.r.t raw_prediction.

        Parameters
        ----------
        y_true : array of shape (n_samples,)
            Observed, true target values.
        raw_prediction : array of shape (n_samples,)
            Raw prediction values (in link space).
        sample_weight : array of shape (n_samples,) or None
            Sample weights.
        gradient_out : array of shape (n_samples,)
            A location into which the gradient is stored.
        hessian_out : array of shape (n_samples,)
            A location into which the hessian is stored.
        n_threads : int
            Number of threads used by OpenMP (if any).

        Returns
        -------
        gradient : array of shape (n_samples,)
            Element-wise gradients.

        hessian : array of shape (n_samples,)
            Element-wise hessians.
        """
        pass


cdef class CyHalfSquaredError(CyLossFunction):
    """Half Squared Error with identity link.

    Domain:
    y_true and y_pred all real numbers

    Link:
    y_pred = raw_prediction
    """


    cdef inline double cy_loss(self, double y_true, double raw_prediction) nogil:
        return closs_half_squared_error(y_true, raw_prediction)

    cdef inline double cy_gradient(self, double y_true, double raw_prediction) nogil:
        return cgradient_half_squared_error(y_true, raw_prediction)

    cdef inline double_pair cy_grad_hess(self, double y_true, double raw_prediction) nogil:
        return cgrad_hess_half_squared_error(y_true, raw_prediction)

    def loss(
        self,
        Y_DTYPE_C[::1] y_true,          # IN
        Y_DTYPE_C[::1] raw_prediction,  # IN
        Y_DTYPE_C[::1] sample_weight,   # IN
        G_DTYPE_C[::1] loss_out,        # OUT
        int n_threads=1
    ):
        cdef:
            int i
            int n_samples = y_true.shape[0]

        if sample_weight is None:
            for i in prange(
                n_samples, schedule='static', nogil=True, num_threads=n_threads
            ):
                loss_out[i] = closs_half_squared_error(y_true[i], raw_prediction[i])
        else:
            for i in prange(
                n_samples, schedule='static', nogil=True, num_threads=n_threads
            ):
                loss_out[i] = sample_weight[i] * closs_half_squared_error(y_true[i], raw_prediction[i])

        return np.asarray(loss_out)


    def gradient(
        self,
        Y_DTYPE_C[::1] y_true,          # IN
        Y_DTYPE_C[::1] raw_prediction,  # IN
        Y_DTYPE_C[::1] sample_weight,   # IN
        G_DTYPE_C[::1] gradient_out,    # OUT
        int n_threads=1
    ):
        cdef:
            int i
            int n_samples = y_true.shape[0]

        if sample_weight is None:
            for i in prange(
                n_samples, schedule='static', nogil=True, num_threads=n_threads
            ):
                gradient_out[i] = cgradient_half_squared_error(y_true[i], raw_prediction[i])
        else:
            for i in prange(
                n_samples, schedule='static', nogil=True, num_threads=n_threads
            ):
                gradient_out[i] = sample_weight[i] * cgradient_half_squared_error(y_true[i], raw_prediction[i])

        return np.asarray(gradient_out)

    def gradient_hessian(
        self,
        Y_DTYPE_C[::1] y_true,          # IN
        Y_DTYPE_C[::1] raw_prediction,  # IN
        Y_DTYPE_C[::1] sample_weight,   # IN
        G_DTYPE_C[::1] gradient_out,    # OUT
        G_DTYPE_C[::1] hessian_out,     # OUT
        int n_threads=1
    ):
        cdef:
            int i
            int n_samples = y_true.shape[0]
            double_pair dbl2

        if sample_weight is None:
            for i in prange(
                n_samples, schedule='static', nogil=True, num_threads=n_threads
            ):
                dbl2 = cgrad_hess_half_squared_error(y_true[i], raw_prediction[i])
                gradient_out[i] = dbl2.val1
                hessian_out[i] = dbl2.val2
        else:
            for i in prange(
                n_samples, schedule='static', nogil=True, num_threads=n_threads
            ):
                dbl2 = cgrad_hess_half_squared_error(y_true[i], raw_prediction[i])
                gradient_out[i] = sample_weight[i] * dbl2.val1
                hessian_out[i] = sample_weight[i] * dbl2.val2

        return np.asarray(gradient_out), np.asarray(hessian_out)

cdef class CyAbsoluteError(CyLossFunction):
    """Absolute Error with identity link.

    Domain:
    y_true and y_pred all real numbers

    Link:
    y_pred = raw_prediction
    """


    cdef inline double cy_loss(self, double y_true, double raw_prediction) nogil:
        return closs_absolute_error(y_true, raw_prediction)

    cdef inline double cy_gradient(self, double y_true, double raw_prediction) nogil:
        return cgradient_absolute_error(y_true, raw_prediction)

    cdef inline double_pair cy_grad_hess(self, double y_true, double raw_prediction) nogil:
        return cgrad_hess_absolute_error(y_true, raw_prediction)

    def loss(
        self,
        Y_DTYPE_C[::1] y_true,          # IN
        Y_DTYPE_C[::1] raw_prediction,  # IN
        Y_DTYPE_C[::1] sample_weight,   # IN
        G_DTYPE_C[::1] loss_out,        # OUT
        int n_threads=1
    ):
        cdef:
            int i
            int n_samples = y_true.shape[0]

        if sample_weight is None:
            for i in prange(
                n_samples, schedule='static', nogil=True, num_threads=n_threads
            ):
                loss_out[i] = closs_absolute_error(y_true[i], raw_prediction[i])
        else:
            for i in prange(
                n_samples, schedule='static', nogil=True, num_threads=n_threads
            ):
                loss_out[i] = sample_weight[i] * closs_absolute_error(y_true[i], raw_prediction[i])

        return np.asarray(loss_out)


    def gradient(
        self,
        Y_DTYPE_C[::1] y_true,          # IN
        Y_DTYPE_C[::1] raw_prediction,  # IN
        Y_DTYPE_C[::1] sample_weight,   # IN
        G_DTYPE_C[::1] gradient_out,    # OUT
        int n_threads=1
    ):
        cdef:
            int i
            int n_samples = y_true.shape[0]

        if sample_weight is None:
            for i in prange(
                n_samples, schedule='static', nogil=True, num_threads=n_threads
            ):
                gradient_out[i] = cgradient_absolute_error(y_true[i], raw_prediction[i])
        else:
            for i in prange(
                n_samples, schedule='static', nogil=True, num_threads=n_threads
            ):
                gradient_out[i] = sample_weight[i] * cgradient_absolute_error(y_true[i], raw_prediction[i])

        return np.asarray(gradient_out)

    def gradient_hessian(
        self,
        Y_DTYPE_C[::1] y_true,          # IN
        Y_DTYPE_C[::1] raw_prediction,  # IN
        Y_DTYPE_C[::1] sample_weight,   # IN
        G_DTYPE_C[::1] gradient_out,    # OUT
        G_DTYPE_C[::1] hessian_out,     # OUT
        int n_threads=1
    ):
        cdef:
            int i
            int n_samples = y_true.shape[0]
            double_pair dbl2

        if sample_weight is None:
            for i in prange(
                n_samples, schedule='static', nogil=True, num_threads=n_threads
            ):
                dbl2 = cgrad_hess_absolute_error(y_true[i], raw_prediction[i])
                gradient_out[i] = dbl2.val1
                hessian_out[i] = dbl2.val2
        else:
            for i in prange(
                n_samples, schedule='static', nogil=True, num_threads=n_threads
            ):
                dbl2 = cgrad_hess_absolute_error(y_true[i], raw_prediction[i])
                gradient_out[i] = sample_weight[i] * dbl2.val1
                hessian_out[i] = sample_weight[i] * dbl2.val2

        return np.asarray(gradient_out), np.asarray(hessian_out)

cdef class CyPinballLoss(CyLossFunction):
    """Quantile Loss aka Pinball Loss with identity link.

    Domain:
    y_true and y_pred all real numbers
    quantile in (0, 1)

    Link:
    y_pred = raw_prediction

    Note: 2 * cPinballLoss(quantile=0.5) equals cAbsoluteError()
    """

    def __init__(self, quantile):
        self.quantile = quantile

    cdef inline double cy_loss(self, double y_true, double raw_prediction) nogil:
        return closs_pinball_loss(y_true, raw_prediction, self.quantile)

    cdef inline double cy_gradient(self, double y_true, double raw_prediction) nogil:
        return cgradient_pinball_loss(y_true, raw_prediction, self.quantile)

    cdef inline double_pair cy_grad_hess(self, double y_true, double raw_prediction) nogil:
        return cgrad_hess_pinball_loss(y_true, raw_prediction, self.quantile)

    def loss(
        self,
        Y_DTYPE_C[::1] y_true,          # IN
        Y_DTYPE_C[::1] raw_prediction,  # IN
        Y_DTYPE_C[::1] sample_weight,   # IN
        G_DTYPE_C[::1] loss_out,        # OUT
        int n_threads=1
    ):
        cdef:
            int i
            int n_samples = y_true.shape[0]

        if sample_weight is None:
            for i in prange(
                n_samples, schedule='static', nogil=True, num_threads=n_threads
            ):
                loss_out[i] = closs_pinball_loss(y_true[i], raw_prediction[i], self.quantile)
        else:
            for i in prange(
                n_samples, schedule='static', nogil=True, num_threads=n_threads
            ):
                loss_out[i] = sample_weight[i] * closs_pinball_loss(y_true[i], raw_prediction[i], self.quantile)

        return np.asarray(loss_out)


    def gradient(
        self,
        Y_DTYPE_C[::1] y_true,          # IN
        Y_DTYPE_C[::1] raw_prediction,  # IN
        Y_DTYPE_C[::1] sample_weight,   # IN
        G_DTYPE_C[::1] gradient_out,    # OUT
        int n_threads=1
    ):
        cdef:
            int i
            int n_samples = y_true.shape[0]

        if sample_weight is None:
            for i in prange(
                n_samples, schedule='static', nogil=True, num_threads=n_threads
            ):
                gradient_out[i] = cgradient_pinball_loss(y_true[i], raw_prediction[i], self.quantile)
        else:
            for i in prange(
                n_samples, schedule='static', nogil=True, num_threads=n_threads
            ):
                gradient_out[i] = sample_weight[i] * cgradient_pinball_loss(y_true[i], raw_prediction[i], self.quantile)

        return np.asarray(gradient_out)

    def gradient_hessian(
        self,
        Y_DTYPE_C[::1] y_true,          # IN
        Y_DTYPE_C[::1] raw_prediction,  # IN
        Y_DTYPE_C[::1] sample_weight,   # IN
        G_DTYPE_C[::1] gradient_out,    # OUT
        G_DTYPE_C[::1] hessian_out,     # OUT
        int n_threads=1
    ):
        cdef:
            int i
            int n_samples = y_true.shape[0]
            double_pair dbl2

        if sample_weight is None:
            for i in prange(
                n_samples, schedule='static', nogil=True, num_threads=n_threads
            ):
                dbl2 = cgrad_hess_pinball_loss(y_true[i], raw_prediction[i], self.quantile)
                gradient_out[i] = dbl2.val1
                hessian_out[i] = dbl2.val2
        else:
            for i in prange(
                n_samples, schedule='static', nogil=True, num_threads=n_threads
            ):
                dbl2 = cgrad_hess_pinball_loss(y_true[i], raw_prediction[i], self.quantile)
                gradient_out[i] = sample_weight[i] * dbl2.val1
                hessian_out[i] = sample_weight[i] * dbl2.val2

        return np.asarray(gradient_out), np.asarray(hessian_out)

cdef class CyHalfPoissonLoss(CyLossFunction):
    """Half Poisson deviance loss with log-link.

    Domain:
    y_true in non-negative real numbers
    y_pred in positive real numbers

    Link:
    y_pred = exp(raw_prediction)

    Half Poisson deviance with log-link is
        y_true * log(y_true/y_pred) + y_pred - y_true
        = y_true * log(y_true) - y_true * raw_prediction
          + exp(raw_prediction) - y_true

    Dropping constant terms, this gives:
        exp(raw_prediction) - y_true * raw_prediction
    """


    cdef inline double cy_loss(self, double y_true, double raw_prediction) nogil:
        return closs_half_poisson(y_true, raw_prediction)

    cdef inline double cy_gradient(self, double y_true, double raw_prediction) nogil:
        return cgradient_half_poisson(y_true, raw_prediction)

    cdef inline double_pair cy_grad_hess(self, double y_true, double raw_prediction) nogil:
        return cgrad_hess_half_poisson(y_true, raw_prediction)

    def loss(
        self,
        Y_DTYPE_C[::1] y_true,          # IN
        Y_DTYPE_C[::1] raw_prediction,  # IN
        Y_DTYPE_C[::1] sample_weight,   # IN
        G_DTYPE_C[::1] loss_out,        # OUT
        int n_threads=1
    ):
        cdef:
            int i
            int n_samples = y_true.shape[0]

        if sample_weight is None:
            for i in prange(
                n_samples, schedule='static', nogil=True, num_threads=n_threads
            ):
                loss_out[i] = closs_half_poisson(y_true[i], raw_prediction[i])
        else:
            for i in prange(
                n_samples, schedule='static', nogil=True, num_threads=n_threads
            ):
                loss_out[i] = sample_weight[i] * closs_half_poisson(y_true[i], raw_prediction[i])

        return np.asarray(loss_out)

    def loss_gradient(
        self,
        Y_DTYPE_C[::1] y_true,          # IN
        Y_DTYPE_C[::1] raw_prediction,  # IN
        Y_DTYPE_C[::1] sample_weight,   # IN
        G_DTYPE_C[::1] loss_out,        # OUT
        G_DTYPE_C[::1] gradient_out,    # OUT
        int n_threads=1
    ):
        cdef:
            int i
            int n_samples = y_true.shape[0]
            double_pair dbl2

        if sample_weight is None:
            for i in prange(
                n_samples, schedule='static', nogil=True, num_threads=n_threads
            ):
                dbl2 = closs_grad_half_poisson(y_true[i], raw_prediction[i])
                loss_out[i] = dbl2.val1
                gradient_out[i] = dbl2.val2
        else:
            for i in prange(
                n_samples, schedule='static', nogil=True, num_threads=n_threads
            ):
                dbl2 = closs_grad_half_poisson(y_true[i], raw_prediction[i])
                loss_out[i] = sample_weight[i] * dbl2.val1
                gradient_out[i] = sample_weight[i] * dbl2.val2

        return np.asarray(loss_out), np.asarray(gradient_out)

    def gradient(
        self,
        Y_DTYPE_C[::1] y_true,          # IN
        Y_DTYPE_C[::1] raw_prediction,  # IN
        Y_DTYPE_C[::1] sample_weight,   # IN
        G_DTYPE_C[::1] gradient_out,    # OUT
        int n_threads=1
    ):
        cdef:
            int i
            int n_samples = y_true.shape[0]

        if sample_weight is None:
            for i in prange(
                n_samples, schedule='static', nogil=True, num_threads=n_threads
            ):
                gradient_out[i] = cgradient_half_poisson(y_true[i], raw_prediction[i])
        else:
            for i in prange(
                n_samples, schedule='static', nogil=True, num_threads=n_threads
            ):
                gradient_out[i] = sample_weight[i] * cgradient_half_poisson(y_true[i], raw_prediction[i])

        return np.asarray(gradient_out)

    def gradient_hessian(
        self,
        Y_DTYPE_C[::1] y_true,          # IN
        Y_DTYPE_C[::1] raw_prediction,  # IN
        Y_DTYPE_C[::1] sample_weight,   # IN
        G_DTYPE_C[::1] gradient_out,    # OUT
        G_DTYPE_C[::1] hessian_out,     # OUT
        int n_threads=1
    ):
        cdef:
            int i
            int n_samples = y_true.shape[0]
            double_pair dbl2

        if sample_weight is None:
            for i in prange(
                n_samples, schedule='static', nogil=True, num_threads=n_threads
            ):
                dbl2 = cgrad_hess_half_poisson(y_true[i], raw_prediction[i])
                gradient_out[i] = dbl2.val1
                hessian_out[i] = dbl2.val2
        else:
            for i in prange(
                n_samples, schedule='static', nogil=True, num_threads=n_threads
            ):
                dbl2 = cgrad_hess_half_poisson(y_true[i], raw_prediction[i])
                gradient_out[i] = sample_weight[i] * dbl2.val1
                hessian_out[i] = sample_weight[i] * dbl2.val2

        return np.asarray(gradient_out), np.asarray(hessian_out)

cdef class CyHalfGammaLoss(CyLossFunction):
    """Half Gamma deviance loss with log-link.

    Domain:
    y_true and y_pred in positive real numbers

    Link:
    y_pred = exp(raw_prediction)

    Half Gamma deviance with log-link is
        log(y_pred/y_true) + y_true/y_pred - 1
        = raw_prediction - log(y_true) + y_true * exp(-raw_prediction) - 1

    Dropping constant terms, this gives:
        raw_prediction + y_true * exp(-raw_prediction)
    """


    cdef inline double cy_loss(self, double y_true, double raw_prediction) nogil:
        return closs_half_gamma(y_true, raw_prediction)

    cdef inline double cy_gradient(self, double y_true, double raw_prediction) nogil:
        return cgradient_half_gamma(y_true, raw_prediction)

    cdef inline double_pair cy_grad_hess(self, double y_true, double raw_prediction) nogil:
        return cgrad_hess_half_gamma(y_true, raw_prediction)

    def loss(
        self,
        Y_DTYPE_C[::1] y_true,          # IN
        Y_DTYPE_C[::1] raw_prediction,  # IN
        Y_DTYPE_C[::1] sample_weight,   # IN
        G_DTYPE_C[::1] loss_out,        # OUT
        int n_threads=1
    ):
        cdef:
            int i
            int n_samples = y_true.shape[0]

        if sample_weight is None:
            for i in prange(
                n_samples, schedule='static', nogil=True, num_threads=n_threads
            ):
                loss_out[i] = closs_half_gamma(y_true[i], raw_prediction[i])
        else:
            for i in prange(
                n_samples, schedule='static', nogil=True, num_threads=n_threads
            ):
                loss_out[i] = sample_weight[i] * closs_half_gamma(y_true[i], raw_prediction[i])

        return np.asarray(loss_out)

    def loss_gradient(
        self,
        Y_DTYPE_C[::1] y_true,          # IN
        Y_DTYPE_C[::1] raw_prediction,  # IN
        Y_DTYPE_C[::1] sample_weight,   # IN
        G_DTYPE_C[::1] loss_out,        # OUT
        G_DTYPE_C[::1] gradient_out,    # OUT
        int n_threads=1
    ):
        cdef:
            int i
            int n_samples = y_true.shape[0]
            double_pair dbl2

        if sample_weight is None:
            for i in prange(
                n_samples, schedule='static', nogil=True, num_threads=n_threads
            ):
                dbl2 = closs_grad_half_gamma(y_true[i], raw_prediction[i])
                loss_out[i] = dbl2.val1
                gradient_out[i] = dbl2.val2
        else:
            for i in prange(
                n_samples, schedule='static', nogil=True, num_threads=n_threads
            ):
                dbl2 = closs_grad_half_gamma(y_true[i], raw_prediction[i])
                loss_out[i] = sample_weight[i] * dbl2.val1
                gradient_out[i] = sample_weight[i] * dbl2.val2

        return np.asarray(loss_out), np.asarray(gradient_out)

    def gradient(
        self,
        Y_DTYPE_C[::1] y_true,          # IN
        Y_DTYPE_C[::1] raw_prediction,  # IN
        Y_DTYPE_C[::1] sample_weight,   # IN
        G_DTYPE_C[::1] gradient_out,    # OUT
        int n_threads=1
    ):
        cdef:
            int i
            int n_samples = y_true.shape[0]

        if sample_weight is None:
            for i in prange(
                n_samples, schedule='static', nogil=True, num_threads=n_threads
            ):
                gradient_out[i] = cgradient_half_gamma(y_true[i], raw_prediction[i])
        else:
            for i in prange(
                n_samples, schedule='static', nogil=True, num_threads=n_threads
            ):
                gradient_out[i] = sample_weight[i] * cgradient_half_gamma(y_true[i], raw_prediction[i])

        return np.asarray(gradient_out)

    def gradient_hessian(
        self,
        Y_DTYPE_C[::1] y_true,          # IN
        Y_DTYPE_C[::1] raw_prediction,  # IN
        Y_DTYPE_C[::1] sample_weight,   # IN
        G_DTYPE_C[::1] gradient_out,    # OUT
        G_DTYPE_C[::1] hessian_out,     # OUT
        int n_threads=1
    ):
        cdef:
            int i
            int n_samples = y_true.shape[0]
            double_pair dbl2

        if sample_weight is None:
            for i in prange(
                n_samples, schedule='static', nogil=True, num_threads=n_threads
            ):
                dbl2 = cgrad_hess_half_gamma(y_true[i], raw_prediction[i])
                gradient_out[i] = dbl2.val1
                hessian_out[i] = dbl2.val2
        else:
            for i in prange(
                n_samples, schedule='static', nogil=True, num_threads=n_threads
            ):
                dbl2 = cgrad_hess_half_gamma(y_true[i], raw_prediction[i])
                gradient_out[i] = sample_weight[i] * dbl2.val1
                hessian_out[i] = sample_weight[i] * dbl2.val2

        return np.asarray(gradient_out), np.asarray(hessian_out)

cdef class CyHalfTweedieLoss(CyLossFunction):
    """Half Tweedie deviance loss with log-link.

    Domain:
    y_true in real numbers if p <= 0
    y_true in non-negative real numbers if 0 < p < 2
    y_true in positive real numbers if p >= 2
    y_pred and power in positive real numbers

    Link:
    y_pred = exp(raw_prediction)

    Half Tweedie deviance with log-link and p=power is
        max(y_true, 0)**(2-p) / (1-p) / (2-p)
        - y_true * y_pred**(1-p) / (1-p)
        + y_pred**(2-p) / (2-p)
        = max(y_true, 0)**(2-p) / (1-p) / (2-p)
        - y_true * exp((1-p) * raw_prediction) / (1-p)
        + exp((2-p) * raw_prediction) / (2-p)

    Dropping constant terms, this gives:
        exp((2-p) * raw_prediction) / (2-p)
        - y_true * exp((1-p) * raw_prediction) / (1-p)

    Notes:
    - Poisson with p=1 and and Gamma with p=2 have different terms dropped such
      that cHalfTweedieLoss is not continuous in p=power at p=1 and p=2.
    - While the Tweedie distribution only exists for p<=0 or p>=1, the range
      0<p<1 still gives a strictly consistent scoring function for the
      expectation.
    """

    def __init__(self, power):
        self.power = power

    cdef inline double cy_loss(self, double y_true, double raw_prediction) nogil:
        return closs_half_tweedie(y_true, raw_prediction, self.power)

    cdef inline double cy_gradient(self, double y_true, double raw_prediction) nogil:
        return cgradient_half_tweedie(y_true, raw_prediction, self.power)

    cdef inline double_pair cy_grad_hess(self, double y_true, double raw_prediction) nogil:
        return cgrad_hess_half_tweedie(y_true, raw_prediction, self.power)

    def loss(
        self,
        Y_DTYPE_C[::1] y_true,          # IN
        Y_DTYPE_C[::1] raw_prediction,  # IN
        Y_DTYPE_C[::1] sample_weight,   # IN
        G_DTYPE_C[::1] loss_out,        # OUT
        int n_threads=1
    ):
        cdef:
            int i
            int n_samples = y_true.shape[0]

        if sample_weight is None:
            for i in prange(
                n_samples, schedule='static', nogil=True, num_threads=n_threads
            ):
                loss_out[i] = closs_half_tweedie(y_true[i], raw_prediction[i], self.power)
        else:
            for i in prange(
                n_samples, schedule='static', nogil=True, num_threads=n_threads
            ):
                loss_out[i] = sample_weight[i] * closs_half_tweedie(y_true[i], raw_prediction[i], self.power)

        return np.asarray(loss_out)

    def loss_gradient(
        self,
        Y_DTYPE_C[::1] y_true,          # IN
        Y_DTYPE_C[::1] raw_prediction,  # IN
        Y_DTYPE_C[::1] sample_weight,   # IN
        G_DTYPE_C[::1] loss_out,        # OUT
        G_DTYPE_C[::1] gradient_out,    # OUT
        int n_threads=1
    ):
        cdef:
            int i
            int n_samples = y_true.shape[0]
            double_pair dbl2

        if sample_weight is None:
            for i in prange(
                n_samples, schedule='static', nogil=True, num_threads=n_threads
            ):
                dbl2 = closs_grad_half_tweedie(y_true[i], raw_prediction[i], self.power)
                loss_out[i] = dbl2.val1
                gradient_out[i] = dbl2.val2
        else:
            for i in prange(
                n_samples, schedule='static', nogil=True, num_threads=n_threads
            ):
                dbl2 = closs_grad_half_tweedie(y_true[i], raw_prediction[i], self.power)
                loss_out[i] = sample_weight[i] * dbl2.val1
                gradient_out[i] = sample_weight[i] * dbl2.val2

        return np.asarray(loss_out), np.asarray(gradient_out)

    def gradient(
        self,
        Y_DTYPE_C[::1] y_true,          # IN
        Y_DTYPE_C[::1] raw_prediction,  # IN
        Y_DTYPE_C[::1] sample_weight,   # IN
        G_DTYPE_C[::1] gradient_out,    # OUT
        int n_threads=1
    ):
        cdef:
            int i
            int n_samples = y_true.shape[0]

        if sample_weight is None:
            for i in prange(
                n_samples, schedule='static', nogil=True, num_threads=n_threads
            ):
                gradient_out[i] = cgradient_half_tweedie(y_true[i], raw_prediction[i], self.power)
        else:
            for i in prange(
                n_samples, schedule='static', nogil=True, num_threads=n_threads
            ):
                gradient_out[i] = sample_weight[i] * cgradient_half_tweedie(y_true[i], raw_prediction[i], self.power)

        return np.asarray(gradient_out)

    def gradient_hessian(
        self,
        Y_DTYPE_C[::1] y_true,          # IN
        Y_DTYPE_C[::1] raw_prediction,  # IN
        Y_DTYPE_C[::1] sample_weight,   # IN
        G_DTYPE_C[::1] gradient_out,    # OUT
        G_DTYPE_C[::1] hessian_out,     # OUT
        int n_threads=1
    ):
        cdef:
            int i
            int n_samples = y_true.shape[0]
            double_pair dbl2

        if sample_weight is None:
            for i in prange(
                n_samples, schedule='static', nogil=True, num_threads=n_threads
            ):
                dbl2 = cgrad_hess_half_tweedie(y_true[i], raw_prediction[i], self.power)
                gradient_out[i] = dbl2.val1
                hessian_out[i] = dbl2.val2
        else:
            for i in prange(
                n_samples, schedule='static', nogil=True, num_threads=n_threads
            ):
                dbl2 = cgrad_hess_half_tweedie(y_true[i], raw_prediction[i], self.power)
                gradient_out[i] = sample_weight[i] * dbl2.val1
                hessian_out[i] = sample_weight[i] * dbl2.val2

        return np.asarray(gradient_out), np.asarray(hessian_out)

cdef class CyHalfTweedieLossIdentity(CyLossFunction):
    """Half Tweedie deviance loss with identity link.

    Domain:
    y_true in real numbers if p <= 0
    y_true in non-negative real numbers if 0 < p < 2
    y_true in positive real numbers if p >= 2
    y_pred and power in positive real numbers, y_pred may be negative for p=0.

    Link:
    y_pred = raw_prediction

    Half Tweedie deviance with identity link and p=power is
        max(y_true, 0)**(2-p) / (1-p) / (2-p)
        - y_true * y_pred**(1-p) / (1-p)
        + y_pred**(2-p) / (2-p)

    Notes:
    - Here, we do not drop constant terms in contrast to the version with log-link.
    """

    def __init__(self, power):
        self.power = power

    cdef inline double cy_loss(self, double y_true, double raw_prediction) nogil:
        return closs_half_tweedie_identity(y_true, raw_prediction, self.power)

    cdef inline double cy_gradient(self, double y_true, double raw_prediction) nogil:
        return cgradient_half_tweedie_identity(y_true, raw_prediction, self.power)

    cdef inline double_pair cy_grad_hess(self, double y_true, double raw_prediction) nogil:
        return cgrad_hess_half_tweedie_identity(y_true, raw_prediction, self.power)

    def loss(
        self,
        Y_DTYPE_C[::1] y_true,          # IN
        Y_DTYPE_C[::1] raw_prediction,  # IN
        Y_DTYPE_C[::1] sample_weight,   # IN
        G_DTYPE_C[::1] loss_out,        # OUT
        int n_threads=1
    ):
        cdef:
            int i
            int n_samples = y_true.shape[0]

        if sample_weight is None:
            for i in prange(
                n_samples, schedule='static', nogil=True, num_threads=n_threads
            ):
                loss_out[i] = closs_half_tweedie_identity(y_true[i], raw_prediction[i], self.power)
        else:
            for i in prange(
                n_samples, schedule='static', nogil=True, num_threads=n_threads
            ):
                loss_out[i] = sample_weight[i] * closs_half_tweedie_identity(y_true[i], raw_prediction[i], self.power)

        return np.asarray(loss_out)

    def loss_gradient(
        self,
        Y_DTYPE_C[::1] y_true,          # IN
        Y_DTYPE_C[::1] raw_prediction,  # IN
        Y_DTYPE_C[::1] sample_weight,   # IN
        G_DTYPE_C[::1] loss_out,        # OUT
        G_DTYPE_C[::1] gradient_out,    # OUT
        int n_threads=1
    ):
        cdef:
            int i
            int n_samples = y_true.shape[0]
            double_pair dbl2

        if sample_weight is None:
            for i in prange(
                n_samples, schedule='static', nogil=True, num_threads=n_threads
            ):
                dbl2 = closs_grad_half_tweedie_identity(y_true[i], raw_prediction[i], self.power)
                loss_out[i] = dbl2.val1
                gradient_out[i] = dbl2.val2
        else:
            for i in prange(
                n_samples, schedule='static', nogil=True, num_threads=n_threads
            ):
                dbl2 = closs_grad_half_tweedie_identity(y_true[i], raw_prediction[i], self.power)
                loss_out[i] = sample_weight[i] * dbl2.val1
                gradient_out[i] = sample_weight[i] * dbl2.val2

        return np.asarray(loss_out), np.asarray(gradient_out)

    def gradient(
        self,
        Y_DTYPE_C[::1] y_true,          # IN
        Y_DTYPE_C[::1] raw_prediction,  # IN
        Y_DTYPE_C[::1] sample_weight,   # IN
        G_DTYPE_C[::1] gradient_out,    # OUT
        int n_threads=1
    ):
        cdef:
            int i
            int n_samples = y_true.shape[0]

        if sample_weight is None:
            for i in prange(
                n_samples, schedule='static', nogil=True, num_threads=n_threads
            ):
                gradient_out[i] = cgradient_half_tweedie_identity(y_true[i], raw_prediction[i], self.power)
        else:
            for i in prange(
                n_samples, schedule='static', nogil=True, num_threads=n_threads
            ):
                gradient_out[i] = sample_weight[i] * cgradient_half_tweedie_identity(y_true[i], raw_prediction[i], self.power)

        return np.asarray(gradient_out)

    def gradient_hessian(
        self,
        Y_DTYPE_C[::1] y_true,          # IN
        Y_DTYPE_C[::1] raw_prediction,  # IN
        Y_DTYPE_C[::1] sample_weight,   # IN
        G_DTYPE_C[::1] gradient_out,    # OUT
        G_DTYPE_C[::1] hessian_out,     # OUT
        int n_threads=1
    ):
        cdef:
            int i
            int n_samples = y_true.shape[0]
            double_pair dbl2

        if sample_weight is None:
            for i in prange(
                n_samples, schedule='static', nogil=True, num_threads=n_threads
            ):
                dbl2 = cgrad_hess_half_tweedie_identity(y_true[i], raw_prediction[i], self.power)
                gradient_out[i] = dbl2.val1
                hessian_out[i] = dbl2.val2
        else:
            for i in prange(
                n_samples, schedule='static', nogil=True, num_threads=n_threads
            ):
                dbl2 = cgrad_hess_half_tweedie_identity(y_true[i], raw_prediction[i], self.power)
                gradient_out[i] = sample_weight[i] * dbl2.val1
                hessian_out[i] = sample_weight[i] * dbl2.val2

        return np.asarray(gradient_out), np.asarray(hessian_out)

cdef class CyHalfBinomialLoss(CyLossFunction):
    """Half Binomial deviance loss with logit link.

    Domain:
    y_true in [0, 1]
    y_pred in (0, 1), i.e. boundaries excluded

    Link:
    y_pred = expit(raw_prediction)
    """


    cdef inline double cy_loss(self, double y_true, double raw_prediction) nogil:
        return closs_half_binomial(y_true, raw_prediction)

    cdef inline double cy_gradient(self, double y_true, double raw_prediction) nogil:
        return cgradient_half_binomial(y_true, raw_prediction)

    cdef inline double_pair cy_grad_hess(self, double y_true, double raw_prediction) nogil:
        return cgrad_hess_half_binomial(y_true, raw_prediction)

    def loss(
        self,
        Y_DTYPE_C[::1] y_true,          # IN
        Y_DTYPE_C[::1] raw_prediction,  # IN
        Y_DTYPE_C[::1] sample_weight,   # IN
        G_DTYPE_C[::1] loss_out,        # OUT
        int n_threads=1
    ):
        cdef:
            int i
            int n_samples = y_true.shape[0]

        if sample_weight is None:
            for i in prange(
                n_samples, schedule='static', nogil=True, num_threads=n_threads
            ):
                loss_out[i] = closs_half_binomial(y_true[i], raw_prediction[i])
        else:
            for i in prange(
                n_samples, schedule='static', nogil=True, num_threads=n_threads
            ):
                loss_out[i] = sample_weight[i] * closs_half_binomial(y_true[i], raw_prediction[i])

        return np.asarray(loss_out)

    def loss_gradient(
        self,
        Y_DTYPE_C[::1] y_true,          # IN
        Y_DTYPE_C[::1] raw_prediction,  # IN
        Y_DTYPE_C[::1] sample_weight,   # IN
        G_DTYPE_C[::1] loss_out,        # OUT
        G_DTYPE_C[::1] gradient_out,    # OUT
        int n_threads=1
    ):
        cdef:
            int i
            int n_samples = y_true.shape[0]
            double_pair dbl2

        if sample_weight is None:
            for i in prange(
                n_samples, schedule='static', nogil=True, num_threads=n_threads
            ):
                dbl2 = closs_grad_half_binomial(y_true[i], raw_prediction[i])
                loss_out[i] = dbl2.val1
                gradient_out[i] = dbl2.val2
        else:
            for i in prange(
                n_samples, schedule='static', nogil=True, num_threads=n_threads
            ):
                dbl2 = closs_grad_half_binomial(y_true[i], raw_prediction[i])
                loss_out[i] = sample_weight[i] * dbl2.val1
                gradient_out[i] = sample_weight[i] * dbl2.val2

        return np.asarray(loss_out), np.asarray(gradient_out)

    def gradient(
        self,
        Y_DTYPE_C[::1] y_true,          # IN
        Y_DTYPE_C[::1] raw_prediction,  # IN
        Y_DTYPE_C[::1] sample_weight,   # IN
        G_DTYPE_C[::1] gradient_out,    # OUT
        int n_threads=1
    ):
        cdef:
            int i
            int n_samples = y_true.shape[0]

        if sample_weight is None:
            for i in prange(
                n_samples, schedule='static', nogil=True, num_threads=n_threads
            ):
                gradient_out[i] = cgradient_half_binomial(y_true[i], raw_prediction[i])
        else:
            for i in prange(
                n_samples, schedule='static', nogil=True, num_threads=n_threads
            ):
                gradient_out[i] = sample_weight[i] * cgradient_half_binomial(y_true[i], raw_prediction[i])

        return np.asarray(gradient_out)

    def gradient_hessian(
        self,
        Y_DTYPE_C[::1] y_true,          # IN
        Y_DTYPE_C[::1] raw_prediction,  # IN
        Y_DTYPE_C[::1] sample_weight,   # IN
        G_DTYPE_C[::1] gradient_out,    # OUT
        G_DTYPE_C[::1] hessian_out,     # OUT
        int n_threads=1
    ):
        cdef:
            int i
            int n_samples = y_true.shape[0]
            double_pair dbl2

        if sample_weight is None:
            for i in prange(
                n_samples, schedule='static', nogil=True, num_threads=n_threads
            ):
                dbl2 = cgrad_hess_half_binomial(y_true[i], raw_prediction[i])
                gradient_out[i] = dbl2.val1
                hessian_out[i] = dbl2.val2
        else:
            for i in prange(
                n_samples, schedule='static', nogil=True, num_threads=n_threads
            ):
                dbl2 = cgrad_hess_half_binomial(y_true[i], raw_prediction[i])
                gradient_out[i] = sample_weight[i] * dbl2.val1
                hessian_out[i] = sample_weight[i] * dbl2.val2

        return np.asarray(gradient_out), np.asarray(hessian_out)


# The multinomial deviance loss is also known as categorical cross-entropy or
# multinomial log-likelihood
cdef class CyHalfMultinomialLoss(CyLossFunction):
    """Half Multinomial deviance loss with multinomial logit link.

    Domain:
    y_true in {0, 1, 2, 3, .., n_classes - 1}
    y_pred in (0, 1)**n_classes, i.e. interval with boundaries excluded

    Link:
    y_pred = softmax(raw_prediction)

    Note: Label encoding is built-in, i.e. {0, 1, 2, 3, .., n_classes - 1} is
    mapped to (y_true == k) for k = 0 .. n_classes - 1 which is either 0 or 1.
    """

    # Note that we do not assume memory alignment/contiguity of 2d arrays.
    # There seems to be little benefit in doing so. Benchmarks proofing the
    # opposite are welcome.
    def loss(
        self,
        Y_DTYPE_C[::1] y_true,           # IN
        Y_DTYPE_C[:, :] raw_prediction,  # IN
        Y_DTYPE_C[::1] sample_weight,    # IN
        G_DTYPE_C[::1] loss_out,         # OUT
        int n_threads=1
    ):
        cdef:
            int i, k
            int n_samples = y_true.shape[0]
            int n_classes = raw_prediction.shape[1]
            Y_DTYPE_C max_value, sum_exps
            Y_DTYPE_C*  p  # temporary buffer

        # We assume n_samples > n_classes. In this case having the inner loop
        # over n_classes is a good default.
        # TODO: If every memoryview is contiguous and raw_prediction is
        #       f-contiguous, can we write a better algo (loops) to improve
        #       performance?
        if sample_weight is None:
            # inner loop over n_classes
            with nogil, parallel(num_threads=n_threads):
                # Define private buffer variables as each thread might use its
                # own.
                p = <Y_DTYPE_C *> malloc(sizeof(Y_DTYPE_C) * (n_classes + 2))

                for i in prange(n_samples, schedule='static'):
                    sum_exp_minus_max(i, raw_prediction, p)
                    max_value = p[n_classes]     # p[-2]
                    sum_exps = p[n_classes + 1]  # p[-1]
                    loss_out[i] = log(sum_exps) + max_value

                    for k in range(n_classes):
                        # label decode y_true
                        if y_true[i] == k:
                            loss_out[i] -= raw_prediction[i, k]

                free(p)
        else:
            with nogil, parallel(num_threads=n_threads):
                p = <Y_DTYPE_C *> malloc(sizeof(Y_DTYPE_C) * (n_classes + 2))

                for i in prange(n_samples, schedule='static'):
                    sum_exp_minus_max(i, raw_prediction, p)
                    max_value = p[n_classes]     # p[-2]
                    sum_exps = p[n_classes + 1]  # p[-1]
                    loss_out[i] = log(sum_exps) + max_value

                    for k in range(n_classes):
                        # label decode y_true
                        if y_true[i] == k:
                            loss_out[i] -= raw_prediction[i, k]

                    loss_out[i] *= sample_weight[i]

                free(p)

        return np.asarray(loss_out)

    def loss_gradient(
        self,
        Y_DTYPE_C[::1] y_true,           # IN
        Y_DTYPE_C[:, :] raw_prediction,  # IN
        Y_DTYPE_C[::1] sample_weight,    # IN
        G_DTYPE_C[::1] loss_out,         # OUT
        G_DTYPE_C[:, :] gradient_out,    # OUT
        int n_threads=1
    ):
        cdef:
            int i, k
            int n_samples = y_true.shape[0]
            int n_classes = raw_prediction.shape[1]
            Y_DTYPE_C max_value, sum_exps
            Y_DTYPE_C*  p  # temporary buffer

        if sample_weight is None:
            # inner loop over n_classes
            with nogil, parallel(num_threads=n_threads):
                # Define private buffer variables as each thread might use its
                # own.
                p = <Y_DTYPE_C *> malloc(sizeof(Y_DTYPE_C) * (n_classes + 2))

                for i in prange(n_samples, schedule='static'):
                    sum_exp_minus_max(i, raw_prediction, p)
                    max_value = p[n_classes]  # p[-2]
                    sum_exps = p[n_classes + 1]  # p[-1]
                    loss_out[i] = log(sum_exps) + max_value

                    for k in range(n_classes):
                        # label decode y_true
                        if y_true [i] == k:
                            loss_out[i] -= raw_prediction[i, k]
                        p[k] /= sum_exps  # p_k = y_pred_k = prob of class k
                        # gradient_k = p_k - (y_true == k)
                        gradient_out[i, k] = p[k] - (y_true[i] == k)

                free(p)
        else:
            with nogil, parallel(num_threads=n_threads):
                p = <Y_DTYPE_C *> malloc(sizeof(Y_DTYPE_C) * (n_classes + 2))

                for i in prange(n_samples, schedule='static'):
                    sum_exp_minus_max(i, raw_prediction, p)
                    max_value = p[n_classes]  # p[-2]
                    sum_exps = p[n_classes + 1]  # p[-1]
                    loss_out[i] = log(sum_exps) + max_value

                    for k in range(n_classes):
                        # label decode y_true
                        if y_true [i] == k:
                            loss_out[i] -= raw_prediction[i, k]
                        p[k] /= sum_exps  # p_k = y_pred_k = prob of class k
                        # gradient_k = (p_k - (y_true == k)) * sw
                        gradient_out[i, k] = (p[k] - (y_true[i] == k)) * sample_weight[i]

                    loss_out[i] *= sample_weight[i]

                free(p)

        return np.asarray(loss_out), np.asarray(gradient_out)

    def gradient(
        self,
        Y_DTYPE_C[::1] y_true,           # IN
        Y_DTYPE_C[:, :] raw_prediction,  # IN
        Y_DTYPE_C[::1] sample_weight,    # IN
        G_DTYPE_C[:, :] gradient_out,    # OUT
        int n_threads=1
    ):
        cdef:
            int i, k
            int n_samples = y_true.shape[0]
            int n_classes = raw_prediction.shape[1]
            Y_DTYPE_C sum_exps
            Y_DTYPE_C*  p  # temporary buffer

        if sample_weight is None:
            # inner loop over n_classes
            with nogil, parallel(num_threads=n_threads):
                # Define private buffer variables as each thread might use its
                # own.
                p = <Y_DTYPE_C *> malloc(sizeof(Y_DTYPE_C) * (n_classes + 2))

                for i in prange(n_samples, schedule='static'):
                    sum_exp_minus_max(i, raw_prediction, p)
                    sum_exps = p[n_classes + 1]  # p[-1]

                    for k in range(n_classes):
                        p[k] /= sum_exps  # p_k = y_pred_k = prob of class k
                        # gradient_k = y_pred_k - (y_true == k)
                        gradient_out[i, k] = p[k] - (y_true[i] == k)

                free(p)
        else:
            with nogil, parallel(num_threads=n_threads):
                p = <Y_DTYPE_C *> malloc(sizeof(Y_DTYPE_C) * (n_classes + 2))

                for i in prange(n_samples, schedule='static'):
                    sum_exp_minus_max(i, raw_prediction, p)
                    sum_exps = p[n_classes + 1]  # p[-1]

                    for k in range(n_classes):
                        p[k] /= sum_exps  # p_k = y_pred_k = prob of class k
                        # gradient_k = (p_k - (y_true == k)) * sw
                        gradient_out[i, k] = (p[k] - (y_true[i] == k)) * sample_weight[i]

                free(p)

        return np.asarray(gradient_out)

    def gradient_hessian(
        self,
        Y_DTYPE_C[::1] y_true,           # IN
        Y_DTYPE_C[:, :] raw_prediction,  # IN
        Y_DTYPE_C[::1] sample_weight,    # IN
        G_DTYPE_C[:, :] gradient_out,    # OUT
        G_DTYPE_C[:, :] hessian_out,     # OUT
        int n_threads=1
    ):
        cdef:
            int i, k
            int n_samples = y_true.shape[0]
            int n_classes = raw_prediction.shape[1]
            Y_DTYPE_C sum_exps
            Y_DTYPE_C* p  # temporary buffer

        if sample_weight is None:
            # inner loop over n_classes
            with nogil, parallel(num_threads=n_threads):
                # Define private buffer variables as each thread might use its
                # own.
                p = <Y_DTYPE_C *> malloc(sizeof(Y_DTYPE_C) * (n_classes + 2))

                for i in prange(n_samples, schedule='static'):
                    sum_exp_minus_max(i, raw_prediction, p)
                    sum_exps = p[n_classes + 1]  # p[-1]

                    for k in range(n_classes):
                        p[k] /= sum_exps  # p_k = y_pred_k = prob of class k
                        # hessian_k = p_k * (1 - p_k)
                        # gradient_k = p_k - (y_true == k)
                        gradient_out[i, k] = p[k] - (y_true[i] == k)
                        hessian_out[i, k] = p[k] * (1. - p[k])

                free(p)
        else:
            with nogil, parallel(num_threads=n_threads):
                p = <Y_DTYPE_C *> malloc(sizeof(Y_DTYPE_C) * (n_classes + 2))

                for i in prange(n_samples, schedule='static'):
                    sum_exp_minus_max(i, raw_prediction, p)
                    sum_exps = p[n_classes + 1]  # p[-1]

                    for k in range(n_classes):
                        p[k] /= sum_exps  # p_k = y_pred_k = prob of class k
                        # gradient_k = (p_k - (y_true == k)) * sw
                        # hessian_k = p_k * (1 - p_k) * sw
                        gradient_out[i, k] = (p[k] - (y_true[i] == k)) * sample_weight[i]
                        hessian_out[i, k] = (p[k] * (1. - p[k])) * sample_weight[i]

                free(p)

        return np.asarray(gradient_out), np.asarray(hessian_out)


    # This method simplifies the implementation of hessp in linear models,
    # i.e. the matrix-vector product of the full hessian, not only of the
    # diagonal (in the classes) approximation as implemented above.
    def gradient_proba(
        self,
        Y_DTYPE_C[::1] y_true,           # IN
        Y_DTYPE_C[:, :] raw_prediction,  # IN
        Y_DTYPE_C[::1] sample_weight,    # IN
        G_DTYPE_C[:, :] gradient_out,    # OUT
        G_DTYPE_C[:, :] proba_out,       # OUT
        int n_threads=1
    ):
        cdef:
            int i, k
            int n_samples = y_true.shape[0]
            int n_classes = raw_prediction.shape[1]
            Y_DTYPE_C sum_exps
            Y_DTYPE_C*  p  # temporary buffer

        if sample_weight is None:
            # inner loop over n_classes
            with nogil, parallel(num_threads=n_threads):
                # Define private buffer variables as each thread might use its
                # own.
                p = <Y_DTYPE_C *> malloc(sizeof(Y_DTYPE_C) * (n_classes + 2))

                for i in prange(n_samples, schedule='static'):
                    sum_exp_minus_max(i, raw_prediction, p)
                    sum_exps = p[n_classes + 1]  # p[-1]

                    for k in range(n_classes):
                        proba_out[i, k] = p[k] / sum_exps  # y_pred_k = prob of class k
                        # gradient_k = y_pred_k - (y_true == k)
                        gradient_out[i, k] = proba_out[i, k] - (y_true[i] == k)

                free(p)
        else:
            with nogil, parallel(num_threads=n_threads):
                p = <Y_DTYPE_C *> malloc(sizeof(Y_DTYPE_C) * (n_classes + 2))

                for i in prange(n_samples, schedule='static'):
                    sum_exp_minus_max(i, raw_prediction, p)
                    sum_exps = p[n_classes + 1]  # p[-1]

                    for k in range(n_classes):
                        proba_out[i, k] = p[k] / sum_exps  # y_pred_k = prob of class k
                        # gradient_k = (p_k - (y_true == k)) * sw
                        gradient_out[i, k] = (proba_out[i, k] - (y_true[i] == k)) * sample_weight[i]

                free(p)

        return np.asarray(gradient_out), np.asarray(proba_out)
