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
/**@file lp_data/HSimplexDebug.h
 * @brief
 */
#ifndef SIMPLEX_HSIMPLEXDEBUG_H_
#define SIMPLEX_HSIMPLEXDEBUG_H_

#include <set>

#include "lp_data/HighsLpSolverObject.h"

// Methods for Ekk

void debugDualChuzcFailNorms(
    const HighsInt workCount,
    const std::vector<std::pair<HighsInt, double>>& workData,
    double& workDataNorm, const HighsInt numVar, const double* workDual,
    double& workDualNorm);

HighsDebugStatus debugDualChuzcFailQuad0(
    const HighsOptions& options, const HighsInt workCount,
    const std::vector<std::pair<HighsInt, double>>& workData,
    const HighsInt numVar, const double* workDual, const double selectTheta,
    const double remainTheta, const bool force = false);

HighsDebugStatus debugDualChuzcFailQuad1(
    const HighsOptions& options, const HighsInt workCount,
    const std::vector<std::pair<HighsInt, double>>& workData,
    const HighsInt numVar, const double* workDual, const double selectTheta,
    const bool force = false);

HighsDebugStatus debugDualChuzcFailHeap(
    const HighsOptions& options, const HighsInt workCount,
    const std::vector<std::pair<HighsInt, double>>& workData,
    const HighsInt numVar, const double* workDual, const double selectTheta,
    const bool force = false);

HighsDebugStatus debugNonbasicFlagConsistent(const HighsOptions& options,
                                             const HighsLp& lp,
                                             const SimplexBasis& basis);

#endif  // SIMPLEX_HSIMPLEXDEBUG_H_
