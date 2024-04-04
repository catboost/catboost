/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
/*                                                                       */
/*    This file is part of the HiGHS linear optimization suite           */
/*                                                                       */
/*    Written and engineered 2008-2022 at the University of Edinburgh    */
/*                                                                       */
/*    Available as open-source under the MIT License                     */
/*                                                                       */
/*    Authors: Julian Hall, Ivet Galabova, Leona Gottwald and Michael    */
/*    Feldmeier                                                          */
/*                                                                       */
/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
#include "util/HighsLinearSumBounds.h"

#include <algorithm>

void HighsLinearSumBounds::add(HighsInt sum, HighsInt var, double coefficient) {
  double vLower = implVarLowerSource[var] == sum
                      ? varLower[var]
                      : std::max(implVarLower[var], varLower[var]);
  double vUpper = implVarUpperSource[var] == sum
                      ? varUpper[var]
                      : std::min(implVarUpper[var], varUpper[var]);

  if (coefficient > 0) {
    // coefficient is positive, therefore variable lower contributes to sum
    // lower bound
    if (vLower == -kHighsInf)
      numInfSumLower[sum] += 1;
    else
      sumLower[sum] += vLower * coefficient;

    if (vUpper == kHighsInf)
      numInfSumUpper[sum] += 1;
    else
      sumUpper[sum] += vUpper * coefficient;

    if (varLower[var] == -kHighsInf)
      numInfSumLowerOrig[sum] += 1;
    else
      sumLowerOrig[sum] += varLower[var] * coefficient;

    if (varUpper[var] == kHighsInf)
      numInfSumUpperOrig[sum] += 1;
    else
      sumUpperOrig[sum] += varUpper[var] * coefficient;
  } else {
    // coefficient is negative, therefore variable upper contributes to sum
    // lower bound
    if (vUpper == kHighsInf)
      numInfSumLower[sum] += 1;
    else
      sumLower[sum] += vUpper * coefficient;

    if (vLower == -kHighsInf)
      numInfSumUpper[sum] += 1;
    else
      sumUpper[sum] += vLower * coefficient;

    if (varUpper[var] == kHighsInf)
      numInfSumLowerOrig[sum] += 1;
    else
      sumLowerOrig[sum] += varUpper[var] * coefficient;

    if (varLower[var] == -kHighsInf)
      numInfSumUpperOrig[sum] += 1;
    else
      sumUpperOrig[sum] += varLower[var] * coefficient;
  }
}

void HighsLinearSumBounds::remove(HighsInt sum, HighsInt var,
                                  double coefficient) {
  double vLower = implVarLowerSource[var] == sum
                      ? varLower[var]
                      : std::max(implVarLower[var], varLower[var]);
  double vUpper = implVarUpperSource[var] == sum
                      ? varUpper[var]
                      : std::min(implVarUpper[var], varUpper[var]);

  if (coefficient > 0) {
    // coefficient is positive, therefore variable lower contributes to sum
    // lower bound
    if (vLower == -kHighsInf)
      numInfSumLower[sum] -= 1;
    else
      sumLower[sum] -= vLower * coefficient;

    if (vUpper == kHighsInf)
      numInfSumUpper[sum] -= 1;
    else
      sumUpper[sum] -= vUpper * coefficient;

    if (varLower[var] == -kHighsInf)
      numInfSumLowerOrig[sum] -= 1;
    else
      sumLowerOrig[sum] -= varLower[var] * coefficient;

    if (varUpper[var] == kHighsInf)
      numInfSumUpperOrig[sum] -= 1;
    else
      sumUpperOrig[sum] -= varUpper[var] * coefficient;
  } else {
    // coefficient is negative, therefore variable upper contributes to sum
    // lower bound
    if (vUpper == kHighsInf)
      numInfSumLower[sum] -= 1;
    else
      sumLower[sum] -= vUpper * coefficient;

    if (vLower == -kHighsInf)
      numInfSumUpper[sum] -= 1;
    else
      sumUpper[sum] -= vLower * coefficient;

    if (varUpper[var] == kHighsInf)
      numInfSumLowerOrig[sum] -= 1;
    else
      sumLowerOrig[sum] -= varUpper[var] * coefficient;

    if (varLower[var] == -kHighsInf)
      numInfSumUpperOrig[sum] -= 1;
    else
      sumUpperOrig[sum] -= varLower[var] * coefficient;
  }
}

void HighsLinearSumBounds::updatedVarUpper(HighsInt sum, HighsInt var,
                                           double coefficient,
                                           double oldVarUpper) {
  double oldVUpper = implVarUpperSource[var] == sum
                         ? oldVarUpper
                         : std::min(implVarUpper[var], oldVarUpper);

  double vUpper = implVarUpperSource[var] == sum
                      ? varUpper[var]
                      : std::min(implVarUpper[var], varUpper[var]);

  if (coefficient > 0) {
    if (vUpper != oldVUpper) {
      if (oldVUpper == kHighsInf)
        numInfSumUpper[sum] -= 1;
      else
        sumUpper[sum] -= oldVUpper * coefficient;

      if (vUpper == kHighsInf)
        numInfSumUpper[sum] += 1;
      else
        sumUpper[sum] += vUpper * coefficient;
    }
    if (oldVarUpper == kHighsInf)
      numInfSumUpperOrig[sum] -= 1;
    else
      sumUpperOrig[sum] -= oldVarUpper * coefficient;

    if (varUpper[var] == kHighsInf)
      numInfSumUpperOrig[sum] += 1;
    else
      sumUpperOrig[sum] += varUpper[var] * coefficient;
  } else {
    if (vUpper != oldVUpper) {
      if (oldVUpper == kHighsInf)
        numInfSumLower[sum] -= 1;
      else
        sumLower[sum] -= oldVUpper * coefficient;

      if (vUpper == kHighsInf)
        numInfSumLower[sum] += 1;
      else
        sumLower[sum] += vUpper * coefficient;
    }
    if (oldVarUpper == kHighsInf)
      numInfSumLowerOrig[sum] -= 1;
    else
      sumLowerOrig[sum] -= oldVarUpper * coefficient;

    if (varUpper[var] == kHighsInf)
      numInfSumLowerOrig[sum] += 1;
    else
      sumLowerOrig[sum] += varUpper[var] * coefficient;
  }
}

void HighsLinearSumBounds::updatedVarLower(HighsInt sum, HighsInt var,
                                           double coefficient,
                                           double oldVarLower) {
  double oldVLower = implVarLowerSource[var] == sum
                         ? oldVarLower
                         : std::max(implVarLower[var], oldVarLower);

  double vLower = implVarLowerSource[var] == sum
                      ? varLower[var]
                      : std::max(implVarLower[var], varLower[var]);

  if (coefficient > 0) {
    if (vLower != oldVLower) {
      if (oldVLower == -kHighsInf)
        numInfSumLower[sum] -= 1;
      else
        sumLower[sum] -= oldVLower * coefficient;

      if (vLower == -kHighsInf)
        numInfSumLower[sum] += 1;
      else
        sumLower[sum] += vLower * coefficient;
    }

    if (oldVarLower == -kHighsInf)
      numInfSumLowerOrig[sum] -= 1;
    else
      sumLowerOrig[sum] -= oldVarLower * coefficient;

    if (varLower[var] == -kHighsInf)
      numInfSumLowerOrig[sum] += 1;
    else
      sumLowerOrig[sum] += varLower[var] * coefficient;

  } else {
    if (vLower != oldVLower) {
      if (oldVLower == -kHighsInf)
        numInfSumUpper[sum] -= 1;
      else
        sumUpper[sum] -= oldVLower * coefficient;

      if (vLower == -kHighsInf)
        numInfSumUpper[sum] += 1;
      else
        sumUpper[sum] += vLower * coefficient;
    }
    if (oldVarLower == -kHighsInf)
      numInfSumUpperOrig[sum] -= 1;
    else
      sumUpperOrig[sum] -= oldVarLower * coefficient;

    if (varLower[var] == -kHighsInf)
      numInfSumUpperOrig[sum] += 1;
    else
      sumUpperOrig[sum] += varLower[var] * coefficient;
  }
}

void HighsLinearSumBounds::updatedImplVarUpper(HighsInt sum, HighsInt var,
                                               double coefficient,
                                               double oldImplVarUpper,
                                               HighsInt oldImplVarUpperSource) {
  double oldVUpper = oldImplVarUpperSource == sum
                         ? varUpper[var]
                         : std::min(oldImplVarUpper, varUpper[var]);

  double vUpper = implVarUpperSource[var] == sum
                      ? varUpper[var]
                      : std::min(implVarUpper[var], varUpper[var]);

  if (vUpper == oldVUpper) return;

  if (coefficient > 0) {
    if (oldVUpper == kHighsInf)
      numInfSumUpper[sum] -= 1;
    else
      sumUpper[sum] -= oldVUpper * coefficient;

    if (vUpper == kHighsInf)
      numInfSumUpper[sum] += 1;
    else
      sumUpper[sum] += vUpper * coefficient;
  } else {
    if (oldVUpper == kHighsInf)
      numInfSumLower[sum] -= 1;
    else
      sumLower[sum] -= oldVUpper * coefficient;

    if (vUpper == kHighsInf)
      numInfSumLower[sum] += 1;
    else
      sumLower[sum] += vUpper * coefficient;
  }
}

void HighsLinearSumBounds::updatedImplVarLower(HighsInt sum, HighsInt var,
                                               double coefficient,
                                               double oldImplVarLower,
                                               HighsInt oldImplVarLowerSource) {
  double oldVLower = oldImplVarLowerSource == sum
                         ? varLower[var]
                         : std::max(oldImplVarLower, varLower[var]);

  double vLower = implVarLowerSource[var] == sum
                      ? varLower[var]
                      : std::max(implVarLower[var], varLower[var]);

  if (vLower == oldVLower) return;

  if (coefficient > 0) {
    if (oldVLower == -kHighsInf)
      numInfSumLower[sum] -= 1;
    else
      sumLower[sum] -= oldVLower * coefficient;

    if (vLower == -kHighsInf)
      numInfSumLower[sum] += 1;
    else
      sumLower[sum] += vLower * coefficient;

  } else {
    if (oldVLower == -kHighsInf)
      numInfSumUpper[sum] -= 1;
    else
      sumUpper[sum] -= oldVLower * coefficient;

    if (vLower == -kHighsInf)
      numInfSumUpper[sum] += 1;
    else
      sumUpper[sum] += vLower * coefficient;
  }
}

double HighsLinearSumBounds::getResidualSumLower(HighsInt sum, HighsInt var,
                                                 double coefficient) const {
  switch (numInfSumLower[sum]) {
    case 0:
      if (coefficient > 0) {
        double vLower = implVarLowerSource[var] == sum
                            ? varLower[var]
                            : std::max(implVarLower[var], varLower[var]);
        return double(sumLower[sum] - vLower * coefficient);
      } else {
        double vUpper = implVarUpperSource[var] == sum
                            ? varUpper[var]
                            : std::min(implVarUpper[var], varUpper[var]);
        return double(sumLower[sum] - vUpper * coefficient);
      }
      break;
    case 1:
      if (coefficient > 0) {
        double vLower = implVarLowerSource[var] == sum
                            ? varLower[var]
                            : std::max(implVarLower[var], varLower[var]);
        return vLower == -kHighsInf ? double(sumLower[sum]) : -kHighsInf;
      } else {
        double vUpper = implVarUpperSource[var] == sum
                            ? varUpper[var]
                            : std::min(implVarUpper[var], varUpper[var]);
        return vUpper == kHighsInf ? double(sumLower[sum]) : -kHighsInf;
      }
      break;
    default:
      return -kHighsInf;
  }
}

double HighsLinearSumBounds::getResidualSumUpper(HighsInt sum, HighsInt var,
                                                 double coefficient) const {
  switch (numInfSumUpper[sum]) {
    case 0:
      if (coefficient > 0) {
        double vUpper = implVarUpperSource[var] == sum
                            ? varUpper[var]
                            : std::min(implVarUpper[var], varUpper[var]);
        return double(sumUpper[sum] - vUpper * coefficient);
      } else {
        double vLower = implVarLowerSource[var] == sum
                            ? varLower[var]
                            : std::max(implVarLower[var], varLower[var]);
        return double(sumUpper[sum] - vLower * coefficient);
      }
      break;
    case 1:
      if (coefficient > 0) {
        double vUpper = implVarUpperSource[var] == sum
                            ? varUpper[var]
                            : std::min(implVarUpper[var], varUpper[var]);
        return vUpper == kHighsInf ? double(sumUpper[sum]) : kHighsInf;
      } else {
        double vLower = implVarLowerSource[var] == sum
                            ? varLower[var]
                            : std::max(implVarLower[var], varLower[var]);
        return vLower == -kHighsInf ? double(sumUpper[sum]) : kHighsInf;
      }
      break;
    default:
      return kHighsInf;
  }
}

double HighsLinearSumBounds::getResidualSumLowerOrig(HighsInt sum, HighsInt var,
                                                     double coefficient) const {
  switch (numInfSumLowerOrig[sum]) {
    case 0:
      if (coefficient > 0)
        return double(sumLowerOrig[sum] - varLower[var] * coefficient);
      else
        return double(sumLowerOrig[sum] - varUpper[var] * coefficient);
      break;
    case 1:
      if (coefficient > 0)
        return varLower[var] == -kHighsInf ? double(sumLowerOrig[sum])
                                           : -kHighsInf;
      else
        return varUpper[var] == kHighsInf ? double(sumLowerOrig[sum])
                                          : -kHighsInf;
      break;
    default:
      return -kHighsInf;
  }
}

double HighsLinearSumBounds::getResidualSumUpperOrig(HighsInt sum, HighsInt var,
                                                     double coefficient) const {
  switch (numInfSumUpperOrig[sum]) {
    case 0:
      if (coefficient > 0)
        return double(sumUpperOrig[sum] - varUpper[var] * coefficient);
      else
        return double(sumUpperOrig[sum] - varLower[var] * coefficient);
      break;
    case 1:
      if (coefficient > 0)
        return varUpper[var] == kHighsInf ? double(sumUpperOrig[sum])
                                          : kHighsInf;
      else
        return varLower[var] == -kHighsInf ? double(sumUpperOrig[sum])
                                           : kHighsInf;
      break;
    default:
      return kHighsInf;
  }
}

void HighsLinearSumBounds::shrink(const std::vector<HighsInt>& newIndices,
                                  HighsInt newSize) {
  HighsInt oldNumInds = newIndices.size();
  for (HighsInt i = 0; i != oldNumInds; ++i) {
    if (newIndices[i] != -1) {
      sumLower[newIndices[i]] = sumLower[i];
      sumUpper[newIndices[i]] = sumUpper[i];
      numInfSumLower[newIndices[i]] = numInfSumLower[i];
      numInfSumUpper[newIndices[i]] = numInfSumUpper[i];
      sumLowerOrig[newIndices[i]] = sumLowerOrig[i];
      sumUpperOrig[newIndices[i]] = sumUpperOrig[i];
      numInfSumLowerOrig[newIndices[i]] = numInfSumLowerOrig[i];
      numInfSumUpperOrig[newIndices[i]] = numInfSumUpperOrig[i];
    }
  }

  sumLower.resize(newSize);
  sumUpper.resize(newSize);
  numInfSumLower.resize(newSize);
  numInfSumUpper.resize(newSize);
  sumLowerOrig.resize(newSize);
  sumUpperOrig.resize(newSize);
  numInfSumLowerOrig.resize(newSize);
  numInfSumUpperOrig.resize(newSize);
}
