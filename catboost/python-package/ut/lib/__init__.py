import math

import numpy as np

from catboost import MultiTargetCustomObjective


class LoglossObjective(object):
    def calc_ders_range(self, approxes, targets, weights):
        assert len(approxes) == len(targets)
        if weights is not None:
            assert len(weights) == len(approxes)

        exponents = []
        for index in range(len(approxes)):
            exponents.append(math.exp(approxes[index]))

        result = []
        for index in range(len(targets)):
            p = exponents[index] / (1 + exponents[index])
            der1 = (1 - p) if targets[index] > 0.0 else -p
            der2 = -p * (1 - p)

            if weights is not None:
                der1 *= weights[index]
                der2 *= weights[index]

            result.append((der1, der2))

        return result


class LoglossObjectiveNumpy(object):
    def __init__(self):
        self._objective = LoglossObjective()

    def calc_ders_range(self, approxes, targets, weights):
        return np.array(self._objective.calc_ders_range(approxes, targets, weights))


class LoglossObjectiveNumpy32(object):
    def __init__(self):
        self._objective = LoglossObjective()

    def calc_ders_range(self, approxes, targets, weights):
        return np.array(self._objective.calc_ders_range(approxes, targets, weights), dtype=np.float32)


class MSEObjective(object):
    def calc_ders_range(self, approxes, targets, weights):
        assert len(approxes) == len(targets)
        der1 = 2.0 * (np.array(approxes) - np.array(targets))
        der2 = np.full(len(approxes), -2.0)
        if weights is not None:
            assert len(weights) == len(targets)
            der1 *= np.array(weights)
            der2 *= np.array(weights)
        return list(zip(der1, der2))


class MultiRMSEObjective(MultiTargetCustomObjective):
    def calc_ders_multi(self, approxes, targets, weight):
        assert len(approxes) == len(targets)

        grad = []
        hess = [[0 for j in range(len(targets))] for i in range(len(targets))]

        for index in range(len(targets)):
            der1 = (targets[index] - approxes[index]) * weight
            der2 = -weight

            grad.append(der1)
            hess[index][index] = der2

        return (grad, hess)
