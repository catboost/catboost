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
/**@file lp_data/HighsSolve.cpp
 * @brief Class-independent utilities for HiGHS
 */

#include "ipm/IpxWrapper.h"
#include "lp_data/HighsSolutionDebug.h"
#include "simplex/HApp.h"

// The method below runs simplex or ipx solver on the lp.
HighsStatus solveLp(HighsLpSolverObject& solver_object, const string message) {
  HighsStatus return_status = HighsStatus::kOk;
  HighsStatus call_status;
  HighsOptions& options = solver_object.options_;
  // Reset unscaled model status and solution params - except for
  // iteration counts
  resetModelStatusAndHighsInfo(solver_object);
  highsLogUser(options.log_options, HighsLogType::kInfo,
               (message + "\n").c_str());
  if (options.highs_debug_level > kHighsDebugLevelMin) {
    // Shouldn't have to check validity of the LP since this is done when it is
    // loaded or modified
    call_status = assessLp(solver_object.lp_, options);
    // If any errors have been found or normalisation carried out,
    // call_status will be ERROR or WARNING, so only valid return is OK.
    assert(call_status == HighsStatus::kOk);
    return_status = interpretCallStatus(options.log_options, call_status,
                                        return_status, "assessLp");
    if (return_status == HighsStatus::kError) return return_status;
  }
  if (!solver_object.lp_.num_row_) {
    // Unconstrained LP so solve directly
    call_status = solveUnconstrainedLp(solver_object);
    return_status = interpretCallStatus(options.log_options, call_status,
                                        return_status, "solveUnconstrainedLp");
    if (return_status == HighsStatus::kError) return return_status;
  } else if (options.solver == kIpmString) {
    // Use IPM
    bool imprecise_solution;
    // Use IPX to solve the LP
    try {
      call_status = solveLpIpx(solver_object);
    } catch (const std::exception& exception) {
      highsLogDev(options.log_options, HighsLogType::kError,
                  "Exception %s in solveLpIpx\n", exception.what());
      call_status = HighsStatus::kError;
    }
    // Non-error return requires a primal solution
    assert(solver_object.solution_.value_valid);
    // Set the return_status, model status and, for completeness, scaled
    // model status
    return_status = interpretCallStatus(options.log_options, call_status,
                                        return_status, "solveLpIpx");
    if (return_status == HighsStatus::kError) return return_status;
    // Get the objective and any KKT failures
    solver_object.highs_info_.objective_function_value =
        solver_object.lp_.objectiveValue(solver_object.solution_.col_value);
    getLpKktFailures(options, solver_object.lp_, solver_object.solution_,
                     solver_object.basis_, solver_object.highs_info_);
    // Seting the IPM-specific values of (highs_)info_ has been done in
    // solveLpIpx
    if ((solver_object.model_status_ == HighsModelStatus::kUnknown ||
         (solver_object.model_status_ ==
              HighsModelStatus::kUnboundedOrInfeasible &&
          !options.allow_unbounded_or_infeasible)) &&
        options.run_crossover) {
      // IPX has returned a model status that HiGHS would rather
      // avoid, so perform simplex clean-up if crossover was allowed.
      //
      // This is an unusual situation, and the cost will usually be
      // acceptable. Worst case is if crossover wasn't run, in which
      // case there's no basis to start simplex
      //
      // ToDo: Check whether simplex can exploit the primal solution returned by
      // IPX
      highsLogUser(options.log_options, HighsLogType::kWarning,
                   "Imprecise solution returned from IPX, so use simplex to "
                   "clean up\n");
      // Reset the return status since it will now be determined by
      // the outcome of the simplex solve
      return_status = HighsStatus::kOk;
      call_status = solveLpSimplex(solver_object);
      return_status = interpretCallStatus(options.log_options, call_status,
                                          return_status, "solveLpSimplex");
      if (return_status == HighsStatus::kError) return return_status;
      if (!isSolutionRightSize(solver_object.lp_, solver_object.solution_)) {
        highsLogUser(options.log_options, HighsLogType::kError,
                     "Inconsistent solution returned from solver\n");
        return HighsStatus::kError;
      }
    }
  } else {
    // Use Simplex
    call_status = solveLpSimplex(solver_object);
    return_status = interpretCallStatus(options.log_options, call_status,
                                        return_status, "solveLpSimplex");
    if (return_status == HighsStatus::kError) return return_status;
    if (!isSolutionRightSize(solver_object.lp_, solver_object.solution_)) {
      highsLogUser(options.log_options, HighsLogType::kError,
                   "Inconsistent solution returned from solver\n");
      return HighsStatus::kError;
    }
  }
  // Analyse the HiGHS (basic) solution
  if (debugHighsLpSolution(message, solver_object) ==
      HighsDebugStatus::kLogicalError)
    return_status = HighsStatus::kError;
  return return_status;
}

// Solves an unconstrained LP without scaling, setting HighsBasis, HighsSolution
// and HighsInfo
HighsStatus solveUnconstrainedLp(HighsLpSolverObject& solver_object) {
  return (solveUnconstrainedLp(solver_object.options_, solver_object.lp_,
                               solver_object.model_status_,
                               solver_object.highs_info_,
                               solver_object.solution_, solver_object.basis_));
}

// Solves an unconstrained LP without scaling, setting HighsBasis, HighsSolution
// and HighsInfo
HighsStatus solveUnconstrainedLp(const HighsOptions& options, const HighsLp& lp,
                                 HighsModelStatus& model_status,
                                 HighsInfo& highs_info, HighsSolution& solution,
                                 HighsBasis& basis) {
  // Aliase to model status and solution parameters
  resetModelStatusAndHighsInfo(model_status, highs_info);

  // Check that the LP really is unconstrained!
  assert(lp.num_row_ == 0);
  if (lp.num_row_ != 0) return HighsStatus::kError;

  highsLogUser(options.log_options, HighsLogType::kInfo,
               "Solving an unconstrained LP with %" HIGHSINT_FORMAT
               " columns\n",
               lp.num_col_);

  solution.col_value.assign(lp.num_col_, 0);
  solution.col_dual.assign(lp.num_col_, 0);
  basis.col_status.assign(lp.num_col_, HighsBasisStatus::kNonbasic);
  // No rows for primal solution, dual solution or basis
  solution.row_value.clear();
  solution.row_dual.clear();
  basis.row_status.clear();

  double primal_feasibility_tolerance = options.primal_feasibility_tolerance;
  double dual_feasibility_tolerance = options.dual_feasibility_tolerance;

  // Initialise the objective value calculation. Done using
  // HighsSolution so offset is vanilla
  double objective = lp.offset_;
  bool infeasible = false;
  bool unbounded = false;

  highs_info.num_primal_infeasibilities = 0;
  highs_info.max_primal_infeasibility = 0;
  highs_info.sum_primal_infeasibilities = 0;
  highs_info.num_dual_infeasibilities = 0;
  highs_info.max_dual_infeasibility = 0;
  highs_info.sum_dual_infeasibilities = 0;

  for (HighsInt iCol = 0; iCol < lp.num_col_; iCol++) {
    double cost = lp.col_cost_[iCol];
    double dual = (HighsInt)lp.sense_ * cost;
    double lower = lp.col_lower_[iCol];
    double upper = lp.col_upper_[iCol];
    double value;
    double primal_infeasibility = 0;
    double dual_infeasibility = -1;
    HighsBasisStatus status = HighsBasisStatus::kNonbasic;
    if (lower > upper) {
      // Inconsistent bounds, so set the variable to lower bound,
      // unless it's infinite. Otherwise set the variable to upper
      // bound, unless it's infinite. Otherwise set the variable to
      // zero.
      if (highs_isInfinity(lower)) {
        // Lower bound of +inf
        if (highs_isInfinity(-upper)) {
          // Upper bound of -inf
          value = 0;
          status = HighsBasisStatus::kZero;
          primal_infeasibility = kHighsInf;
          dual_infeasibility = std::fabs(dual);
        } else {
          // Finite upper bound - since lower exceeds it
          value = upper;
          status = HighsBasisStatus::kUpper;
          primal_infeasibility = lower - value;
          dual_infeasibility = std::max(dual, 0.);
        }
      } else {
        // Finite lower bound
        value = lower;
        status = HighsBasisStatus::kLower;
        primal_infeasibility = value - upper;
        dual_infeasibility = std::max(-dual, 0.);
      }
    } else if (highs_isInfinity(-lower) && highs_isInfinity(upper)) {
      // Free column: set to zero and record dual infeasiblility
      value = 0;
      status = HighsBasisStatus::kZero;
      dual_infeasibility = std::fabs(dual);
    } else if (dual >= dual_feasibility_tolerance) {
      // Column with sufficiently positive dual
      if (!highs_isInfinity(-lower)) {
        // Set to this finite lower bound
        value = lower;
        status = HighsBasisStatus::kLower;
        dual_infeasibility = 0;
      } else {
        // Infinite lower bound so set to upper bound and record dual
        // infeasiblility
        value = upper;
        status = HighsBasisStatus::kUpper;
        dual_infeasibility = dual;
      }
    } else if (dual <= -dual_feasibility_tolerance) {
      // Column with sufficiently negative dual
      if (!highs_isInfinity(upper)) {
        // Set to this finite upper bound
        value = upper;
        status = HighsBasisStatus::kUpper;
        dual_infeasibility = 0;
      } else {
        // Infinite upper bound so set to lower bound and record dual
        // infeasiblility
        value = lower;
        status = HighsBasisStatus::kLower;
        dual_infeasibility = -dual;
      }
    } else {
      // Column with sufficiently small dual: set to lower bound (if
      // finite) otherwise upper bound
      if (highs_isInfinity(-lower)) {
        value = upper;
        status = HighsBasisStatus::kUpper;
      } else {
        value = lower;
        status = HighsBasisStatus::kLower;
      }
      dual_infeasibility = std::fabs(dual);
    }
    assert(status != HighsBasisStatus::kNonbasic);
    assert(dual_infeasibility >= 0);
    solution.col_value[iCol] = value;
    solution.col_dual[iCol] = (HighsInt)lp.sense_ * dual;
    basis.col_status[iCol] = status;
    objective += value * cost;
    if (primal_infeasibility > primal_feasibility_tolerance)
      highs_info.num_primal_infeasibilities++;
    highs_info.sum_primal_infeasibilities += primal_infeasibility;
    highs_info.max_primal_infeasibility =
        std::max(primal_infeasibility, highs_info.max_primal_infeasibility);
    if (dual_infeasibility > dual_feasibility_tolerance)
      highs_info.num_dual_infeasibilities++;
    highs_info.sum_dual_infeasibilities += dual_infeasibility;
    highs_info.max_dual_infeasibility =
        std::max(dual_infeasibility, highs_info.max_dual_infeasibility);
  }
  highs_info.objective_function_value = objective;
  solution.value_valid = true;
  solution.dual_valid = true;
  basis.valid = true;
  highs_info.basis_validity = kBasisValidityValid;
  setSolutionStatus(highs_info);
  if (highs_info.num_primal_infeasibilities) {
    // Primal infeasible
    model_status = HighsModelStatus::kInfeasible;
  } else if (highs_info.num_dual_infeasibilities) {
    // Dual infeasible => primal unbounded for unconstrained LP
    model_status = HighsModelStatus::kUnbounded;
  } else {
    model_status = HighsModelStatus::kOptimal;
  }

  return HighsStatus::kOk;
}
