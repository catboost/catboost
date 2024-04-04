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
/**@file lp_data/HighsSolve.h
 * @brief Class-independent utilities for HiGHS
 */
#ifndef LP_DATA_HIGHSSOLVE_H_
#define LP_DATA_HIGHSSOLVE_H_

#include "lp_data/HighsModelUtils.h"
HighsStatus solveLp(HighsLpSolverObject& solver_object, const string message);
HighsStatus solveUnconstrainedLp(HighsLpSolverObject& solver_object);
HighsStatus solveUnconstrainedLp(const HighsOptions& options, const HighsLp& lp,
                                 HighsModelStatus& model_status,
                                 HighsInfo& highs_info, HighsSolution& solution,
                                 HighsBasis& basis);
#endif  // LP_DATA_HIGHSSOLVE_H_
