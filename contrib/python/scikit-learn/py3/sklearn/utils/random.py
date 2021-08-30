# Author: Hamzeh Alsalhi <ha258@cornell.edu>
#
# License: BSD 3 clause
import numpy as np
import scipy.sparse as sp
import array

from . import check_random_state
from ._random import sample_without_replacement
from . import deprecated

__all__ = ['sample_without_replacement']


@deprecated("random_choice_csc is deprecated in version "
            "0.22 and will be removed in version 0.24.")
def random_choice_csc(n_samples, classes, class_probability=None,
                      random_state=None):
    return _random_choice_csc(n_samples, classes, class_probability,
                              random_state)


def _random_choice_csc(n_samples, classes, class_probability=None,
                       random_state=None):
    """Generate a sparse random matrix given column class distributions

    Parameters
    ----------
    n_samples : int,
        Number of samples to draw in each column.

    classes : list of size n_outputs of arrays of size (n_classes,)
        List of classes for each column.

    class_probability : list of size n_outputs of arrays of size (n_classes,)
        Optional (default=None). Class distribution of each column. If None the
        uniform distribution is assumed.

    random_state : int, RandomState instance or None, optional (default=None)
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.

    Returns
    -------
    random_matrix : sparse csc matrix of size (n_samples, n_outputs)

    """
    data = array.array('i')
    indices = array.array('i')
    indptr = array.array('i', [0])

    for j in range(len(classes)):
        classes[j] = np.asarray(classes[j])
        if classes[j].dtype.kind != 'i':
            raise ValueError("class dtype %s is not supported" %
                             classes[j].dtype)
        classes[j] = classes[j].astype(np.int64, copy=False)

        # use uniform distribution if no class_probability is given
        if class_probability is None:
            class_prob_j = np.empty(shape=classes[j].shape[0])
            class_prob_j.fill(1 / classes[j].shape[0])
        else:
            class_prob_j = np.asarray(class_probability[j])

        if not np.isclose(np.sum(class_prob_j), 1.0):
            raise ValueError("Probability array at index {0} does not sum to "
                             "one".format(j))

        if class_prob_j.shape[0] != classes[j].shape[0]:
            raise ValueError("classes[{0}] (length {1}) and "
                             "class_probability[{0}] (length {2}) have "
                             "different length.".format(j,
                                                        classes[j].shape[0],
                                                        class_prob_j.shape[0]))

        # If 0 is not present in the classes insert it with a probability 0.0
        if 0 not in classes[j]:
            classes[j] = np.insert(classes[j], 0, 0)
            class_prob_j = np.insert(class_prob_j, 0, 0.0)

        # If there are nonzero classes choose randomly using class_probability
        rng = check_random_state(random_state)
        if classes[j].shape[0] > 1:
            p_nonzero = 1 - class_prob_j[classes[j] == 0]
            nnz = int(n_samples * p_nonzero)
            ind_sample = sample_without_replacement(n_population=n_samples,
                                                    n_samples=nnz,
                                                    random_state=random_state)
            indices.extend(ind_sample)

            # Normalize probabilities for the nonzero elements
            classes_j_nonzero = classes[j] != 0
            class_probability_nz = class_prob_j[classes_j_nonzero]
            class_probability_nz_norm = (class_probability_nz /
                                         np.sum(class_probability_nz))
            classes_ind = np.searchsorted(class_probability_nz_norm.cumsum(),
                                          rng.rand(nnz))
            data.extend(classes[j][classes_j_nonzero][classes_ind])
        indptr.append(len(indices))

    return sp.csc_matrix((data, indices, indptr),
                         (n_samples, len(classes)),
                         dtype=int)
