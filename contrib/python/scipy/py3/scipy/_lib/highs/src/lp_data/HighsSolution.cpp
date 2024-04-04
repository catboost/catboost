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
/**@file lp_data/HighsSolution.cpp
 * @brief Class-independent utilities for HiGHS
 */
#include "lp_data/HighsSolution.h"

#include <string>
#include <vector>

#include "io/HighsIO.h"
#include "ipm/IpxSolution.h"
#include "ipm/ipx/include/ipx_status.h"
#include "ipm/ipx/src/lp_solver.h"
#include "lp_data/HighsLpUtils.h"
#include "lp_data/HighsModelUtils.h"
#include "lp_data/HighsSolutionDebug.h"

void getKktFailures(const HighsOptions& options, const HighsModel& model,
                    const HighsSolution& solution, const HighsBasis& basis,
                    HighsInfo& highs_info) {
  HighsPrimalDualErrors primal_dual_errors;
  getKktFailures(options, model, solution, basis, highs_info,
                 primal_dual_errors);
}

void getKktFailures(const HighsOptions& options, const HighsModel& model,
                    const HighsSolution& solution, const HighsBasis& basis,
                    HighsInfo& highs_info,
                    HighsPrimalDualErrors& primal_dual_errors,
                    const bool get_residuals) {
  vector<double> gradient;
  model.objectiveGradient(solution.col_value, gradient);
  getKktFailures(options, model.lp_, gradient, solution, basis, highs_info,
                 primal_dual_errors, get_residuals);
}

void getLpKktFailures(const HighsOptions& options, const HighsLp& lp,
                      const HighsSolution& solution, const HighsBasis& basis,
                      HighsInfo& highs_info) {
  HighsPrimalDualErrors primal_dual_errors;
  getLpKktFailures(options, lp, solution, basis, highs_info,
                   primal_dual_errors);
}

void getLpKktFailures(const HighsOptions& options, const HighsLp& lp,
                      const HighsSolution& solution, const HighsBasis& basis,
                      HighsInfo& highs_info,
                      HighsPrimalDualErrors& primal_dual_errors,
                      const bool get_residuals) {
  getKktFailures(options, lp, lp.col_cost_, solution, basis, highs_info,
                 primal_dual_errors, get_residuals);
}

void getKktFailures(const HighsOptions& options, const HighsLp& lp,
                    const std::vector<double>& gradient,
                    const HighsSolution& solution, const HighsBasis& basis,
                    HighsInfo& highs_info,
                    HighsPrimalDualErrors& primal_dual_errors,
                    const bool get_residuals) {
  double primal_feasibility_tolerance = options.primal_feasibility_tolerance;
  double dual_feasibility_tolerance = options.dual_feasibility_tolerance;
  // highs_info are the values computed in this method.

  HighsInt& num_primal_infeasibility = highs_info.num_primal_infeasibilities;

  HighsInt& max_absolute_primal_infeasibility_index =
      primal_dual_errors.max_primal_infeasibility.absolute_index;
  double& max_absolute_primal_infeasibility_value =
      highs_info.max_primal_infeasibility;
  HighsInt& max_relative_primal_infeasibility_index =
      primal_dual_errors.max_primal_infeasibility.relative_index;
  double& max_relative_primal_infeasibility_value =
      primal_dual_errors.max_primal_infeasibility.relative_value;

  double& sum_primal_infeasibility = highs_info.sum_primal_infeasibilities;

  HighsInt& num_dual_infeasibility = highs_info.num_dual_infeasibilities;

  HighsInt& max_dual_infeasibility_index =
      primal_dual_errors.max_dual_infeasibility.absolute_index;
  double& max_dual_infeasibility_value = highs_info.max_dual_infeasibility;

  double& sum_dual_infeasibility = highs_info.sum_dual_infeasibilities;

  num_primal_infeasibility = kHighsIllegalInfeasibilityCount;
  max_absolute_primal_infeasibility_value = kHighsIllegalInfeasibilityMeasure;
  sum_primal_infeasibility = kHighsIllegalInfeasibilityMeasure;
  primal_dual_errors.max_primal_infeasibility.invalidate();
  highs_info.primal_solution_status = kSolutionStatusNone;

  num_dual_infeasibility = kHighsIllegalInfeasibilityCount;
  max_dual_infeasibility_value = kHighsIllegalInfeasibilityMeasure;
  sum_dual_infeasibility = kHighsIllegalInfeasibilityMeasure;
  primal_dual_errors.max_dual_infeasibility.invalidate();
  highs_info.dual_solution_status = kSolutionStatusNone;

  const bool& have_primal_solution = solution.value_valid;
  const bool& have_dual_solution = solution.dual_valid;
  const bool& have_basis = basis.valid;
  const bool have_integrality = lp.integrality_.size() > 0;
  // Check that there is no dual solution if there's no primal solution
  assert(have_primal_solution || !have_dual_solution);
  // Check that there is no basis if there's no dual solution
  assert(have_dual_solution || !have_basis);

  if (have_primal_solution) {
    // There's a primal solution, so check its size and initialise the
    // infeasiblilty counts
    assert((int)solution.col_value.size() >= lp.num_col_);
    assert((int)solution.row_value.size() >= lp.num_row_);
    num_primal_infeasibility = 0;
    max_absolute_primal_infeasibility_value = 0;
    sum_primal_infeasibility = 0;
    primal_dual_errors.max_primal_infeasibility.reset();
    if (have_dual_solution) {
      // There's a dual solution, so check its size and initialise the
      // infeasiblilty counts
      assert((int)solution.col_dual.size() >= lp.num_col_);
      assert((int)solution.row_dual.size() >= lp.num_row_);
      num_dual_infeasibility = 0;
      max_dual_infeasibility_value = 0;
      sum_dual_infeasibility = 0;
      primal_dual_errors.max_dual_infeasibility.reset();
    }
  }

  HighsInt& num_primal_residual = primal_dual_errors.num_primal_residual;

  HighsInt& max_absolute_primal_residual_index =
      primal_dual_errors.max_primal_residual.absolute_index;
  double& max_absolute_primal_residual_value =
      primal_dual_errors.max_primal_residual.absolute_value;
  HighsInt& max_relative_primal_residual_index =
      primal_dual_errors.max_primal_residual.relative_index;
  double& max_relative_primal_residual_value =
      primal_dual_errors.max_primal_residual.relative_value;

  double& sum_primal_residual = primal_dual_errors.sum_primal_residual;

  HighsInt& num_dual_residual = primal_dual_errors.num_dual_residual;

  HighsInt& max_absolute_dual_residual_index =
      primal_dual_errors.max_dual_residual.absolute_index;
  double& max_absolute_dual_residual_value =
      primal_dual_errors.max_dual_residual.absolute_value;
  HighsInt& max_relative_dual_residual_index =
      primal_dual_errors.max_dual_residual.relative_index;
  double& max_relative_dual_residual_value =
      primal_dual_errors.max_dual_residual.relative_value;

  double& sum_dual_residual = primal_dual_errors.sum_dual_residual;

  HighsInt& num_nonzero_basic_duals =
      primal_dual_errors.num_nonzero_basic_duals;
  HighsInt& num_large_nonzero_basic_duals =
      primal_dual_errors.num_large_nonzero_basic_duals;
  double& max_nonzero_basic_dual = primal_dual_errors.max_nonzero_basic_dual;
  double& sum_nonzero_basic_duals = primal_dual_errors.sum_nonzero_basic_duals;

  HighsInt& num_off_bound_nonbasic = primal_dual_errors.num_off_bound_nonbasic;
  double& max_off_bound_nonbasic = primal_dual_errors.max_off_bound_nonbasic;
  double& sum_off_bound_nonbasic = primal_dual_errors.sum_off_bound_nonbasic;

  // Initialise HighsPrimalDualErrors

  if (have_primal_solution && get_residuals) {
    num_primal_residual = 0;
    max_absolute_primal_residual_value = 0;
    sum_primal_residual = 0;
    primal_dual_errors.max_primal_residual.reset();
  } else {
    num_primal_residual = kHighsIllegalInfeasibilityCount;
    max_absolute_primal_residual_value = kHighsIllegalInfeasibilityMeasure;
    sum_primal_residual = kHighsIllegalInfeasibilityMeasure;
    primal_dual_errors.max_primal_residual.invalidate();
  }
  if (have_dual_solution && get_residuals) {
    num_dual_residual = 0;
    max_absolute_dual_residual_value = 0;
    sum_dual_residual = 0;
    primal_dual_errors.max_dual_residual.reset();
  } else {
    num_dual_residual = kHighsIllegalInfeasibilityCount;
    max_absolute_dual_residual_value = kHighsIllegalInfeasibilityMeasure;
    sum_dual_residual = kHighsIllegalInfeasibilityMeasure;
    primal_dual_errors.max_dual_residual.invalidate();
  }
  if (have_basis) {
    num_nonzero_basic_duals = 0;
    num_large_nonzero_basic_duals = 0;
    max_nonzero_basic_dual = 0;
    sum_nonzero_basic_duals = 0;

    num_off_bound_nonbasic = 0;
    max_off_bound_nonbasic = 0;
    sum_off_bound_nonbasic = 0;
  } else {
    num_nonzero_basic_duals = kHighsIllegalInfeasibilityCount;
    num_large_nonzero_basic_duals = kHighsIllegalInfeasibilityCount;
    max_nonzero_basic_dual = kHighsIllegalInfeasibilityMeasure;
    sum_nonzero_basic_duals = kHighsIllegalInfeasibilityMeasure;

    num_off_bound_nonbasic = kHighsIllegalInfeasibilityCount;
    max_off_bound_nonbasic = kHighsIllegalInfeasibilityMeasure;
    sum_off_bound_nonbasic = kHighsIllegalInfeasibilityMeasure;
  }

  // Without a primal solution, nothing can be done!
  if (!have_primal_solution) return;
  std::vector<double> primal_positive_sum;
  std::vector<double> primal_negative_sum;
  std::vector<double> dual_positive_sum;
  std::vector<double> dual_negative_sum;
  if (get_residuals) {
    primal_positive_sum.assign(lp.num_row_, 0);
    primal_negative_sum.assign(lp.num_row_, 0);
    if (have_dual_solution) {
      dual_positive_sum.resize(lp.num_col_);
      dual_negative_sum.resize(lp.num_col_);
    }
  }
  HighsInt num_basic_var = 0;
  HighsInt num_non_basic_var = 0;

  // Set status to a value so the compiler doesn't think it might be unassigned.
  HighsBasisStatus status = HighsBasisStatus::kNonbasic;
  // Set status_pointer to be NULL unless we have a basis, in which
  // case it's the pointer to status. Can't make it a pointer to the
  // entry of basis since this is const.
  const HighsBasisStatus* status_pointer = have_basis ? &status : NULL;

  double absolute_primal_infeasibility;
  double relative_primal_infeasibility;
  double dual_infeasibility;
  double value_residual;
  double lower;
  double upper;
  double value;
  double dual = 0;
  HighsVarType integrality = HighsVarType::kContinuous;
  for (HighsInt iVar = 0; iVar < lp.num_col_ + lp.num_row_; iVar++) {
    if (iVar < lp.num_col_) {
      HighsInt iCol = iVar;
      lower = lp.col_lower_[iCol];
      upper = lp.col_upper_[iCol];
      value = solution.col_value[iCol];
      if (have_dual_solution) dual = solution.col_dual[iCol];
      if (have_basis) status = basis.col_status[iCol];
      if (have_integrality) integrality = lp.integrality_[iCol];
    } else {
      HighsInt iRow = iVar - lp.num_col_;
      lower = lp.row_lower_[iRow];
      upper = lp.row_upper_[iRow];
      value = solution.row_value[iRow];
      // @FlipRowDual -solution.row_dual[iRow]; became solution.row_dual[iRow];
      if (have_dual_solution) dual = solution.row_dual[iRow];
      if (have_basis) status = basis.row_status[iRow];
      integrality = HighsVarType::kContinuous;
    }
    // Flip dual according to lp.sense_
    dual *= (HighsInt)lp.sense_;
    getVariableKktFailures(
        primal_feasibility_tolerance, dual_feasibility_tolerance, lower, upper,
        value, dual, status_pointer, integrality, absolute_primal_infeasibility,
        relative_primal_infeasibility, dual_infeasibility, value_residual);
    // Accumulate primal infeasiblilties
    if (absolute_primal_infeasibility > primal_feasibility_tolerance)
      num_primal_infeasibility++;
    if (max_absolute_primal_infeasibility_value <
        absolute_primal_infeasibility) {
      max_absolute_primal_infeasibility_index = iVar;
      max_absolute_primal_infeasibility_value = absolute_primal_infeasibility;
    }
    if (max_relative_primal_infeasibility_value <
        relative_primal_infeasibility) {
      max_relative_primal_infeasibility_index = iVar;
      max_relative_primal_infeasibility_value = relative_primal_infeasibility;
    }
    sum_primal_infeasibility += absolute_primal_infeasibility;

    if (have_dual_solution) {
      // Accumulate dual infeasiblilties
      if (dual_infeasibility > dual_feasibility_tolerance)
        num_dual_infeasibility++;
      if (max_dual_infeasibility_value < dual_infeasibility) {
        max_dual_infeasibility_value = dual_infeasibility;
        max_dual_infeasibility_index = iVar;
      }
      sum_dual_infeasibility += dual_infeasibility;
    }
    if (have_basis) {
      if (status == HighsBasisStatus::kBasic) {
        num_basic_var++;
        double abs_basic_dual = dual_infeasibility;
        if (abs_basic_dual > 0) {
          num_nonzero_basic_duals++;
          if (abs_basic_dual > dual_feasibility_tolerance)
            num_large_nonzero_basic_duals++;
          max_nonzero_basic_dual =
              std::max(abs_basic_dual, max_nonzero_basic_dual);
          sum_nonzero_basic_duals += abs_basic_dual;
        }
      } else {
        num_non_basic_var++;
        double off_bound_nonbasic = value_residual;
        if (off_bound_nonbasic > 0) num_off_bound_nonbasic++;
        max_off_bound_nonbasic =
            std::max(off_bound_nonbasic, max_off_bound_nonbasic);
        sum_off_bound_nonbasic += off_bound_nonbasic;
      }
    }
    if (iVar < lp.num_col_ && get_residuals) {
      HighsInt iCol = iVar;
      if (have_dual_solution) {
        if (gradient[iCol] > 0) {
          dual_positive_sum[iCol] = gradient[iCol];
        } else {
          dual_negative_sum[iCol] = -gradient[iCol];
        }
      }
      for (HighsInt el = lp.a_matrix_.start_[iCol];
           el < lp.a_matrix_.start_[iCol + 1]; el++) {
        HighsInt iRow = lp.a_matrix_.index_[el];
        double Avalue = lp.a_matrix_.value_[el];
        double term = value * Avalue;
        if (term > 0) {
          primal_positive_sum[iRow] += term;
        } else {
          primal_negative_sum[iRow] -= term;
        }
        // @FlipRowDual += became -=
        if (have_dual_solution) {
          double term = -solution.row_dual[iRow] * Avalue;
          if (term > 0) {
            dual_positive_sum[iCol] += term;
          } else {
            dual_negative_sum[iCol] -= term;
          }
        }
      }
    }
  }
  if (get_residuals) {
    const double large_residual_error = 1e-12;
    for (HighsInt iRow = 0; iRow < lp.num_row_; iRow++) {
      double term = -solution.row_value[iRow];
      if (term > 0) {
        primal_positive_sum[iRow] += term;
      } else {
        primal_negative_sum[iRow] -= term;
      }
      assert(primal_positive_sum[iRow] >= 0);
      assert(primal_negative_sum[iRow] >= 0);
      double absolute_primal_residual =
          std::fabs(primal_positive_sum[iRow] - primal_negative_sum[iRow]);
      double relative_primal_residual =
          absolute_primal_residual /
          (1 + primal_positive_sum[iRow] + primal_negative_sum[iRow]);

      if (absolute_primal_residual > large_residual_error)
        num_primal_residual++;
      if (max_absolute_primal_residual_value < absolute_primal_residual) {
        max_absolute_primal_residual_value = absolute_primal_residual;
        max_absolute_primal_residual_index = iRow;
      }
      if (max_relative_primal_residual_value < relative_primal_residual) {
        max_relative_primal_residual_value = relative_primal_residual;
        max_relative_primal_residual_index = iRow;
      }
      sum_primal_residual += absolute_primal_residual;
    }
    if (have_dual_solution) {
      for (HighsInt iCol = 0; iCol < lp.num_col_; iCol++) {
        double term = -solution.col_dual[iCol];
        if (term > 0) {
          dual_positive_sum[iCol] += term;
        } else {
          dual_negative_sum[iCol] -= term;
        }
        assert(dual_positive_sum[iCol] >= 0);
        assert(dual_negative_sum[iCol] >= 0);
        double absolute_dual_residual =
            std::fabs(dual_positive_sum[iCol] - dual_negative_sum[iCol]);
        double relative_dual_residual =
            absolute_dual_residual /
            (1 + dual_positive_sum[iCol] + dual_negative_sum[iCol]);
        if (absolute_dual_residual > large_residual_error) num_dual_residual++;
        if (max_absolute_dual_residual_value < absolute_dual_residual) {
          max_absolute_dual_residual_value = absolute_dual_residual;
          max_absolute_dual_residual_index = iCol;
        }
        if (max_relative_dual_residual_value < relative_dual_residual) {
          max_relative_dual_residual_value = relative_dual_residual;
          max_relative_dual_residual_index = iCol;
        }
        sum_dual_residual += absolute_dual_residual;
      }
    }
  }
  // Assign primal solution status
  if (num_primal_infeasibility) {
    highs_info.primal_solution_status = kSolutionStatusInfeasible;
  } else {
    highs_info.primal_solution_status = kSolutionStatusFeasible;
  }
  if (have_dual_solution) {
    // Assign dual solution status
    if (num_dual_infeasibility) {
      highs_info.dual_solution_status = kSolutionStatusInfeasible;
    } else {
      highs_info.dual_solution_status = kSolutionStatusFeasible;
    }
  }
  // Assign the two entries in primal_dual_errors that are accumulated
  // in highs_info
  primal_dual_errors.max_primal_infeasibility.absolute_value =
      highs_info.max_primal_infeasibility;
  ;
  primal_dual_errors.max_dual_infeasibility.absolute_value =
      highs_info.max_dual_infeasibility;
  // Relative dual_infeasibility data is the same as absolute
  primal_dual_errors.max_dual_infeasibility.relative_value =
      primal_dual_errors.max_dual_infeasibility.absolute_value;
  primal_dual_errors.max_dual_infeasibility.relative_index =
      primal_dual_errors.max_dual_infeasibility.absolute_index;
}

// Gets the KKT failures for a variable. The lack of a basis status is
// indicated by status_pointer being null.
//
// Value and dual are used compute the primal and dual infeasibility -
// according to the basis status (if valid) or primal value. It's up
// to the calling method to ignore these if the value or dual are not
// valid.
//
// If the basis status is valid, then the numbers of basic and
// nonbasic variables are updated, and the extent to which a nonbasic
// variable is off its bound is returned.
void getVariableKktFailures(const double primal_feasibility_tolerance,
                            const double dual_feasibility_tolerance,
                            const double lower, const double upper,
                            const double value, const double dual,
                            const HighsBasisStatus* status_pointer,
                            const HighsVarType integrality,
                            double& absolute_primal_infeasibility,
                            double& relative_primal_infeasibility,
                            double& dual_infeasibility,
                            double& value_residual) {
  const double middle = (lower + upper) * 0.5;
  // @primal_infeasibility calculation
  absolute_primal_infeasibility = 0;
  relative_primal_infeasibility = 0;
  if (value < lower - primal_feasibility_tolerance) {
    // Below lower
    absolute_primal_infeasibility = lower - value;
    relative_primal_infeasibility =
        absolute_primal_infeasibility / (1 + std::fabs(lower));
  } else if (value > upper + primal_feasibility_tolerance) {
    // Above upper
    absolute_primal_infeasibility = value - upper;
    relative_primal_infeasibility =
        absolute_primal_infeasibility / (1 + std::fabs(upper));
  }
  // Account for semi-variables
  if (absolute_primal_infeasibility > 0 &&
      (integrality == HighsVarType::kSemiContinuous ||
       integrality == HighsVarType::kSemiInteger) &&
      std::fabs(value) < primal_feasibility_tolerance) {
    absolute_primal_infeasibility = 0;
    relative_primal_infeasibility = 0;
  }
  value_residual = std::min(std::fabs(lower - value), std::fabs(value - upper));
  // Determine whether the variable is at a bound using the basis
  // status (if valid) or value residual.
  bool at_a_bound = value_residual <= primal_feasibility_tolerance;
  if (status_pointer != NULL) {
    // If the variable is basic, then consider it not to be at a bound
    // so that any dual value yields an infeasibility value
    if (*status_pointer == HighsBasisStatus::kBasic) at_a_bound = false;
  }
  if (at_a_bound) {
    // At a bound
    double middle = (lower + upper) * 0.5;
    if (lower < upper) {
      // Non-fixed variable
      if (value < middle) {
        // At lower
        dual_infeasibility = std::max(-dual, 0.);
      } else {
        // At upper
        dual_infeasibility = std::max(dual, 0.);
      }
    } else {
      // Fixed variable
      dual_infeasibility = 0;
    }
  } else {
    // Off bounds (or free)
    dual_infeasibility = fabs(dual);
  }
}

void HighsError::print(std::string message) {
  printf(
      "\n%s\nAbsolute value = %11.4g; index = %9d\nRelative value = %11.4g; "
      "index = %9d\n",
      message.c_str(), this->absolute_value, (int)this->absolute_index,
      this->relative_value, (int)this->relative_index);
}

void HighsError::reset() {
  this->absolute_value = 0;
  this->absolute_index = 0;
  this->relative_value = 0;
  this->relative_index = 0;
}

void HighsError::invalidate() {
  this->absolute_value = kHighsIllegalErrorValue;
  this->absolute_index = kHighsIllegalErrorIndex;
  this->relative_value = kHighsIllegalErrorValue;
  this->relative_index = kHighsIllegalErrorIndex;
}

double computeObjectiveValue(const HighsLp& lp, const HighsSolution& solution) {
  double objective_value = 0;
  for (HighsInt iCol = 0; iCol < lp.num_col_; iCol++)
    objective_value += lp.col_cost_[iCol] * solution.col_value[iCol];
  objective_value += lp.offset_;
  return objective_value;
}

// Refine any HighsBasisStatus::kNonbasic settings according to the LP
// and any solution values
void refineBasis(const HighsLp& lp, const HighsSolution& solution,
                 HighsBasis& basis) {
  assert(basis.valid);
  assert(isBasisRightSize(lp, basis));
  const bool have_highs_solution = solution.value_valid;

  const HighsInt num_col = lp.num_col_;
  const HighsInt num_row = lp.num_row_;
  for (HighsInt iCol = 0; iCol < num_col; iCol++) {
    if (basis.col_status[iCol] != HighsBasisStatus::kNonbasic) continue;
    const double lower = lp.col_lower_[iCol];
    const double upper = lp.col_upper_[iCol];
    HighsBasisStatus status = HighsBasisStatus::kNonbasic;
    if (lower == upper) {
      status = HighsBasisStatus::kLower;
    } else if (!highs_isInfinity(-lower)) {
      if (!highs_isInfinity(upper)) {
        if (have_highs_solution) {
          if (solution.col_value[iCol] < 0.5 * (lower + upper)) {
            status = HighsBasisStatus::kLower;
          } else {
            status = HighsBasisStatus::kUpper;
          }
        } else {
          if (fabs(lower) < fabs(upper)) {
            status = HighsBasisStatus::kLower;
          } else {
            status = HighsBasisStatus::kUpper;
          }
        }
      } else {
        status = HighsBasisStatus::kLower;
      }
    } else if (!highs_isInfinity(upper)) {
      status = HighsBasisStatus::kUpper;
    } else {
      status = HighsBasisStatus::kZero;
    }
    assert(status != HighsBasisStatus::kNonbasic);
    basis.col_status[iCol] = status;
  }

  for (HighsInt iRow = 0; iRow < num_row; iRow++) {
    if (basis.row_status[iRow] != HighsBasisStatus::kNonbasic) continue;
    const double lower = lp.row_lower_[iRow];
    const double upper = lp.row_upper_[iRow];
    HighsBasisStatus status = HighsBasisStatus::kNonbasic;
    if (lower == upper) {
      status = HighsBasisStatus::kLower;
    } else if (!highs_isInfinity(-lower)) {
      if (!highs_isInfinity(upper)) {
        if (have_highs_solution) {
          if (solution.row_value[iRow] < 0.5 * (lower + upper)) {
            status = HighsBasisStatus::kLower;
          } else {
            status = HighsBasisStatus::kUpper;
          }
        } else {
          if (fabs(lower) < fabs(upper)) {
            status = HighsBasisStatus::kLower;
          } else {
            status = HighsBasisStatus::kUpper;
          }
        }
      } else {
        status = HighsBasisStatus::kLower;
      }
    } else if (!highs_isInfinity(upper)) {
      status = HighsBasisStatus::kUpper;
    } else {
      status = HighsBasisStatus::kZero;
    }
    assert(status != HighsBasisStatus::kNonbasic);
    basis.row_status[iRow] = status;
  }
}

HighsStatus ipxSolutionToHighsSolution(
    const HighsOptions& options, const HighsLp& lp,
    const std::vector<double>& rhs, const std::vector<char>& constraint_type,
    const HighsInt ipx_num_col, const HighsInt ipx_num_row,
    const std::vector<double>& ipx_x, const std::vector<double>& ipx_slack_vars,
    const std::vector<double>& ipx_y, const std::vector<double>& ipx_zl,
    const std::vector<double>& ipx_zu, const HighsModelStatus model_status,
    HighsSolution& highs_solution) {
  // Resize the HighsSolution
  highs_solution.col_value.resize(lp.num_col_);
  highs_solution.row_value.resize(lp.num_row_);
  highs_solution.col_dual.resize(lp.num_col_);
  highs_solution.row_dual.resize(lp.num_row_);

  const double primal_feasibility_tolerance =
      options.primal_feasibility_tolerance;
  const double dual_feasibility_tolerance = options.dual_feasibility_tolerance;
  const std::vector<double>& ipx_col_value = ipx_x;
  const std::vector<double>& ipx_row_value = ipx_slack_vars;

  // Row activities are needed to set activity values of free rows -
  // which are ignored by IPX
  vector<double> row_activity;
  const bool get_row_activities = true;  // ipx_num_row < lp.num_row_;
  if (get_row_activities) row_activity.assign(lp.num_row_, 0);
  HighsInt ipx_slack = lp.num_col_;
  assert(ipx_num_row == lp.num_row_);
  double dual_residual_norm = 0;
  for (HighsInt col = 0; col < lp.num_col_; col++) {
    double lower = lp.col_lower_[col];
    double upper = lp.col_upper_[col];
    double value = ipx_col_value[col];
    if (get_row_activities) {
      // Accumulate row activities to assign value to free rows
      double check_dual = lp.col_cost_[col];
      for (HighsInt el = lp.a_matrix_.start_[col];
           el < lp.a_matrix_.start_[col + 1]; el++) {
        HighsInt row = lp.a_matrix_.index_[el];
        row_activity[row] += value * lp.a_matrix_.value_[el];
        check_dual -= ipx_y[row] * lp.a_matrix_.value_[el];
      }
      double dual_residual =
          std::fabs(check_dual - (ipx_zl[col] - ipx_zu[col]));
      dual_residual_norm = std::max(dual_residual, dual_residual_norm);
    }
    double dual = ipx_zl[col] - ipx_zu[col];
    highs_solution.col_value[col] = value;
    highs_solution.col_dual[col] = dual;
  }
  HighsInt ipx_row = 0;
  ipx_slack = lp.num_col_;
  double delta_norm = 0;
  for (HighsInt row = 0; row < lp.num_row_; row++) {
    double lower = lp.row_lower_[row];
    double upper = lp.row_upper_[row];
    if (lower <= -kHighsInf && upper >= kHighsInf) {
      // Free row - removed by IPX so set it to its row activity
      highs_solution.row_value[row] = row_activity[row];
      highs_solution.row_dual[row] = 0;
      continue;
    }
    // Non-free row, so IPX will have it
    double value = 0;
    double dual = 0;
    if ((lower > -kHighsInf && upper < kHighsInf) && (lower < upper)) {
      assert(constraint_type[ipx_row] == '=');
      // Boxed row - look at its slack
      value = ipx_col_value[ipx_slack];
      dual = ipx_zl[ipx_slack] - ipx_zu[ipx_slack];
      // Update the slack to be used for boxed rows
      ipx_slack++;
    } else {
      value = rhs[ipx_row] - ipx_row_value[ipx_row];
      dual = ipx_y[ipx_row];
    }
    delta_norm = std::max(std::fabs(value - row_activity[row]), delta_norm);
    highs_solution.row_value[row] = value;
    highs_solution.row_dual[row] = dual;
    // Update the IPX row index
    ipx_row++;
  }
  //  if (delta_norm >= dual_feasibility_tolerance)
  highsLogDev(
      options.log_options, HighsLogType::kInfo,
      "ipxSolutionToHighsSolution: Norm of dual residual values is %10.4g\n",
      dual_residual_norm);
  highsLogDev(
      options.log_options, HighsLogType::kInfo,
      "ipxSolutionToHighsSolution: Norm of delta     row values is %10.4g\n",
      delta_norm);
  const bool force_dual_feasibility = false;  // true;
  const bool minimal_truncation = true;
  if (model_status == HighsModelStatus::kOptimal &&
      (force_dual_feasibility || minimal_truncation)) {
    double primal_truncation_norm = 0;
    double dual_truncation_norm = 0;
    double col_primal_truncation_norm = 0;
    double col_dual_truncation_norm = 0;
    HighsInt num_primal_truncations = 0;
    HighsInt num_dual_truncations = 0;
    HighsInt num_col_primal_truncations = 0;
    HighsInt num_col_dual_truncations = 0;
    HighsInt col = 0, row = 0;
    double lower, upper, value, dual, residual;
    const HighsInt check_col = -127;
    const HighsInt check_row = -37;
    // Truncating to tolerances can lead to infeasibilities by margin
    // of machine precision
    const double primal_margin = 0;  // primal_feasibility_tolerance;
    const double dual_margin = 0;    // dual_feasibility_tolerance;
    for (HighsInt var = 0; var < lp.num_col_ + lp.num_row_; var++) {
      if (var == lp.num_col_) {
        col_primal_truncation_norm = primal_truncation_norm;
        col_dual_truncation_norm = dual_truncation_norm;
        num_col_primal_truncations = num_primal_truncations;
        num_col_dual_truncations = num_dual_truncations;
        primal_truncation_norm = 0;
        dual_truncation_norm = 0;
        num_primal_truncations = 0;
        num_dual_truncations = 0;
      }
      const bool is_col = var < lp.num_col_;
      if (is_col) {
        col = var;
        lower = lp.col_lower_[col];
        upper = lp.col_upper_[col];
        value = highs_solution.col_value[col];
        dual = highs_solution.col_dual[col];
        if (col == check_col) {
          printf("Column %d\n", (int)check_col);
        }
      } else {
        row = var - lp.num_col_;
        lower = lp.row_lower_[row];
        upper = lp.row_upper_[row];
        value = highs_solution.row_value[row];
        dual = highs_solution.row_dual[row];
        if (row == check_row) {
          printf("Row %d\n", (int)check_row);
        }
      }
      // Continue if equality: cannot have dual infeasibility
      if (lower >= upper) continue;
      double dual_truncation = 0;
      double primal_truncation = 0;
      double dual_infeasibility = 0;
      double residual = std::fabs(std::max(lower - value, value - upper));
      double new_value = value;
      double new_dual = dual;
      const bool at_lower = value <= lower + primal_feasibility_tolerance;
      const bool at_upper = value >= upper - primal_feasibility_tolerance;
      // Continue if at distinct bounds: cannot have dual infeasibility
      if (at_lower && at_upper) continue;
      if (at_lower) {
        dual_infeasibility = -dual;
      } else if (at_upper) {
        dual_infeasibility = dual;
      } else {
        dual_infeasibility = std::fabs(dual);
      }
      // Continue if no dual infeasibility
      if (dual_infeasibility <= dual_feasibility_tolerance) continue;
      if (residual < dual_infeasibility && !force_dual_feasibility) {
        // Residual is less than dual infeasibility, or not forcing
        // dual feasibility, so truncate value
        if (at_lower) {
          assert(10 == 50);
        } else if (at_upper) {
          assert(11 == 50);
        } else {
          // Off bound
          if (lower <= -kHighsInf) {
            if (upper >= kHighsInf) {
              // Free shouldn't be possible, as residual would be inf
              assert(12 == 50);
            } else {
              // Upper bounded, so assume dual is negative
              if (dual > 0) assert(13 == 50);
            }
          } else if (upper >= kHighsInf) {
            // Lower bounded, so assume dual is positive
            if (dual < 0) assert(14 == 50);
          }
        }
        num_primal_truncations++;
        if (dual > 0) {
          // Put closest to lower
          if (value < lower) {
            new_value = lower - primal_margin;
          } else {
            new_value = lower + primal_margin;
          }
        } else {
          // Put closest to upper
          if (value < upper) {
            new_value = upper - primal_margin;
          } else {
            new_value = upper + primal_margin;
          }
        }
      } else {
        // Residual is greater than dual infeasibility, or forcing
        // dual feasibility, so truncate dual
        num_dual_truncations++;
        if (at_lower) {
          // At lower bound, so possibly truncate to -dual_margin
          new_dual = -dual_margin;
        } else if (at_upper) {
          // At upper bound, so possibly truncate to dual_margin
          new_dual = dual_margin;
        } else {
          // Between bounds so possibly set dual to signed dual_margin
          if (dual > 0) {
            new_dual = dual_margin;
          } else {
            new_dual = -dual_margin;
          }
        }
      }
      primal_truncation = std::fabs(value - new_value);
      dual_truncation = std::fabs(dual - new_dual);
      primal_truncation_norm =
          std::max(primal_truncation, primal_truncation_norm);
      dual_truncation_norm = std::max(dual_truncation, dual_truncation_norm);
      /*if (dual_truncation > 1e-2 || primal_truncation > 1e-2)
        printf(
            "%s %4d: [%11.4g, %11.4g, %11.4g] residual = %11.4g; new = %11.4g; "
            "truncation = %11.4g | "
            "dual = %11.4g; new = %11.4g; truncation = %11.4g\n",
            is_col ? "Col" : "Row", (int)(is_col ? col : row), lower, value,
            upper, residual, new_value, primal_truncation, dual, new_dual,
            dual_truncation);
      */
      if (is_col) {
        highs_solution.col_value[col] = new_value;
        highs_solution.col_dual[col] = new_dual;
      } else {
        highs_solution.row_value[row] = new_value;
        highs_solution.row_dual[row] = new_dual;
      }
    }
    // Assess the truncations
    //  if (dual_truncation_norm >= dual_margin)
    highsLogDev(options.log_options, HighsLogType::kInfo,
                "ipxSolutionToHighsSolution: Norm of %6d col  primal "
                "truncations is %10.4g\n",
                num_col_primal_truncations, col_primal_truncation_norm);
    highsLogDev(options.log_options, HighsLogType::kInfo,
                "ipxSolutionToHighsSolution: Norm of %6d row  primal "
                "truncations is %10.4g\n",
                num_primal_truncations, primal_truncation_norm);
    highsLogDev(options.log_options, HighsLogType::kInfo,
                "ipxSolutionToHighsSolution: Norm of %6d col    dual "
                "truncations is %10.4g\n",
                num_col_dual_truncations, col_dual_truncation_norm);
    highsLogDev(options.log_options, HighsLogType::kInfo,
                "ipxSolutionToHighsSolution: Norm of %6d row    dual "
                "truncations is %10.4g\n",
                num_dual_truncations, dual_truncation_norm);
    // Determine the new residuals

    vector<double> final_row_activity;
    final_row_activity.assign(lp.num_row_, 0);
    dual_residual_norm = 0;
    for (HighsInt col = 0; col < lp.num_col_; col++) {
      double check_dual = lp.col_cost_[col];
      for (HighsInt el = lp.a_matrix_.start_[col];
           el < lp.a_matrix_.start_[col + 1]; el++) {
        HighsInt row = lp.a_matrix_.index_[el];
        final_row_activity[row] +=
            highs_solution.col_value[col] * lp.a_matrix_.value_[el];
        check_dual -= highs_solution.row_dual[row] * lp.a_matrix_.value_[el];
      }
      double dual_residual =
          std::fabs(check_dual - highs_solution.col_dual[col]);
      dual_residual_norm = std::max(dual_residual, dual_residual_norm);
    }
    double primal_residual_norm = 0;
    for (HighsInt row = 0; row < lp.num_row_; row++) {
      double primal_residual =
          std::fabs(final_row_activity[row] - highs_solution.row_value[row]);
      primal_residual_norm = std::max(primal_residual, primal_residual_norm);
    }
    highsLogDev(options.log_options, HighsLogType::kInfo,
                "ipxSolutionToHighsSolution: Final norm of primal residual "
                "values is %10.4g\n",
                primal_residual_norm);
    highsLogDev(options.log_options, HighsLogType::kInfo,
                "ipxSolutionToHighsSolution: Final norm of dual   residual "
                "values is %10.4g\n",
                dual_residual_norm);
  }

  assert(ipx_row == ipx_num_row);
  assert(ipx_slack == ipx_num_col);
  // Indicate that the primal and dual solution are known
  highs_solution.value_valid = true;
  highs_solution.dual_valid = true;
  return HighsStatus::kOk;
}

HighsStatus ipxBasicSolutionToHighsBasicSolution(
    const HighsLogOptions& log_options, const HighsLp& lp,
    const std::vector<double>& rhs, const std::vector<char>& constraint_type,
    const IpxSolution& ipx_solution, HighsBasis& highs_basis,
    HighsSolution& highs_solution) {
  // Resize the HighsSolution and HighsBasis
  highs_solution.col_value.resize(lp.num_col_);
  highs_solution.row_value.resize(lp.num_row_);
  highs_solution.col_dual.resize(lp.num_col_);
  highs_solution.row_dual.resize(lp.num_row_);
  highs_basis.col_status.resize(lp.num_col_);
  highs_basis.row_status.resize(lp.num_row_);

  const std::vector<double>& ipx_col_value = ipx_solution.ipx_col_value;
  const std::vector<double>& ipx_row_value = ipx_solution.ipx_row_value;
  const std::vector<double>& ipx_col_dual = ipx_solution.ipx_col_dual;
  const std::vector<double>& ipx_row_dual = ipx_solution.ipx_row_dual;
  const std::vector<ipx::Int>& ipx_col_status = ipx_solution.ipx_col_status;
  const std::vector<ipx::Int>& ipx_row_status = ipx_solution.ipx_row_status;

  // Set up meaningful names for values of ipx_col_status and ipx_row_status to
  // be used later in comparisons
  const ipx::Int ipx_basic = 0;
  const ipx::Int ipx_nonbasic_at_lb = -1;
  const ipx::Int ipx_nonbasic_at_ub = -2;
  const ipx::Int ipx_superbasic = -3;
  // Row activities are needed to set activity values of free rows -
  // which are ignored by IPX
  vector<double> row_activity;
  bool get_row_activities = ipx_solution.num_row < lp.num_row_;
  if (get_row_activities) row_activity.assign(lp.num_row_, 0);
  HighsInt num_basic_variables = 0;
  for (HighsInt col = 0; col < lp.num_col_; col++) {
    bool unrecognised = false;
    const double lower = lp.col_lower_[col];
    const double upper = lp.col_upper_[col];
    if (ipx_col_status[col] == ipx_basic) {
      // Column is basic
      highs_basis.col_status[col] = HighsBasisStatus::kBasic;
      highs_solution.col_value[col] = ipx_col_value[col];
      highs_solution.col_dual[col] = 0;
    } else {
      // Column is nonbasic. Setting of ipx_col_status is consistent
      // with dual value for fixed columns
      if (ipx_col_status[col] == ipx_nonbasic_at_lb) {
        // Column is at lower bound
        highs_basis.col_status[col] = HighsBasisStatus::kLower;
        highs_solution.col_value[col] = ipx_col_value[col];
        highs_solution.col_dual[col] = ipx_col_dual[col];
      } else if (ipx_col_status[col] == ipx_nonbasic_at_ub) {
        // Column is at upper bound
        highs_basis.col_status[col] = HighsBasisStatus::kUpper;
        highs_solution.col_value[col] = ipx_col_value[col];
        highs_solution.col_dual[col] = ipx_col_dual[col];
      } else if (ipx_col_status[col] == ipx_superbasic) {
        // Column is superbasic
        highs_basis.col_status[col] = HighsBasisStatus::kZero;
        highs_solution.col_value[col] = ipx_col_value[col];
        highs_solution.col_dual[col] = ipx_col_dual[col];
      } else {
        unrecognised = true;
        highsLogDev(log_options, HighsLogType::kError,
                    "\nError in IPX conversion: Unrecognised value "
                    "ipx_col_status[%2" HIGHSINT_FORMAT
                    "] = "
                    "%" HIGHSINT_FORMAT "\n",
                    col, (HighsInt)ipx_col_status[col]);
      }
    }
    if (unrecognised) {
      highsLogDev(log_options, HighsLogType::kError,
                  "Bounds [%11.4g, %11.4g]\n", lp.col_lower_[col],
                  lp.col_upper_[col]);
      highsLogDev(log_options, HighsLogType::kError,
                  "Col %2" HIGHSINT_FORMAT " ipx_col_status[%2" HIGHSINT_FORMAT
                  "] = %2" HIGHSINT_FORMAT "; x[%2" HIGHSINT_FORMAT
                  "] = %11.4g; z[%2" HIGHSINT_FORMAT
                  "] = "
                  "%11.4g\n",
                  col, col, (HighsInt)ipx_col_status[col], col,
                  ipx_col_value[col], col, ipx_col_dual[col]);
      assert(!unrecognised);
      highsLogUser(log_options, HighsLogType::kError,
                   "Unrecognised ipx_col_status value from IPX\n");
      return HighsStatus::kError;
    }
    if (get_row_activities) {
      // Accumulate row activities to assign value to free rows
      for (HighsInt el = lp.a_matrix_.start_[col];
           el < lp.a_matrix_.start_[col + 1]; el++) {
        HighsInt row = lp.a_matrix_.index_[el];
        row_activity[row] +=
            highs_solution.col_value[col] * lp.a_matrix_.value_[el];
      }
    }
    if (highs_basis.col_status[col] == HighsBasisStatus::kBasic)
      num_basic_variables++;
  }
  HighsInt ipx_row = 0;
  HighsInt ipx_slack = lp.num_col_;
  HighsInt num_boxed_rows = 0;
  HighsInt num_boxed_rows_basic = 0;
  HighsInt num_boxed_row_slacks_basic = 0;
  for (HighsInt row = 0; row < lp.num_row_; row++) {
    bool unrecognised = false;
    double lower = lp.row_lower_[row];
    double upper = lp.row_upper_[row];
    HighsInt this_ipx_row = ipx_row;
    if (lower <= -kHighsInf && upper >= kHighsInf) {
      // Free row - removed by IPX so make it basic at its row activity
      highs_basis.row_status[row] = HighsBasisStatus::kBasic;
      highs_solution.row_value[row] = row_activity[row];
      highs_solution.row_dual[row] = 0;
    } else {
      // Non-free row, so IPX will have it
      if ((lower > -kHighsInf && upper < kHighsInf) && (lower < upper)) {
        // Boxed row - look at its slack
        num_boxed_rows++;
        double slack_value = ipx_col_value[ipx_slack];
        double slack_dual = ipx_col_dual[ipx_slack];
        double value = slack_value;
        // @FlipRowDual -slack_dual became slack_dual
        double dual = slack_dual;
        if (ipx_row_status[ipx_row] == ipx_basic) {
          // Row is basic
          num_boxed_rows_basic++;
          highs_basis.row_status[row] = HighsBasisStatus::kBasic;
          highs_solution.row_value[row] = value;
          highs_solution.row_dual[row] = 0;
        } else if (ipx_col_status[ipx_slack] == ipx_basic) {
          // Slack is basic
          num_boxed_row_slacks_basic++;
          highs_basis.row_status[row] = HighsBasisStatus::kBasic;
          highs_solution.row_value[row] = value;
          highs_solution.row_dual[row] = 0;
        } else if (ipx_col_status[ipx_slack] == ipx_nonbasic_at_lb) {
          // Slack at lower bound
          highs_basis.row_status[row] = HighsBasisStatus::kLower;
          highs_solution.row_value[row] = value;
          highs_solution.row_dual[row] = dual;
        } else if (ipx_col_status[ipx_slack] == ipx_nonbasic_at_ub) {
          // Slack is at its upper bound
          assert(ipx_col_status[ipx_slack] == ipx_nonbasic_at_ub);
          highs_basis.row_status[row] = HighsBasisStatus::kUpper;
          highs_solution.row_value[row] = value;
          highs_solution.row_dual[row] = dual;
        } else {
          unrecognised = true;
          highsLogDev(log_options, HighsLogType::kError,
                      "Error in IPX conversion: Row %2" HIGHSINT_FORMAT
                      " (IPX row %2" HIGHSINT_FORMAT
                      ") has "
                      "unrecognised value ipx_col_status[%2" HIGHSINT_FORMAT
                      "] = %" HIGHSINT_FORMAT "\n",
                      row, ipx_row, ipx_slack,
                      (HighsInt)ipx_col_status[ipx_slack]);
        }
        // Update the slack to be used for boxed rows
        ipx_slack++;
      } else if (ipx_row_status[ipx_row] == ipx_basic) {
        // Row is basic
        highs_basis.row_status[row] = HighsBasisStatus::kBasic;
        highs_solution.row_value[row] = rhs[ipx_row] - ipx_row_value[ipx_row];
        highs_solution.row_dual[row] = 0;
      } else {
        // Nonbasic row at fixed value, lower bound or upper bound
        assert(ipx_row_status[ipx_row] ==
               -1);  // const ipx::Int ipx_nonbasic_row = -1;
        double value = rhs[ipx_row] - ipx_row_value[ipx_row];
        // @FlipRowDual -ipx_row_dual[ipx_row]; became ipx_row_dual[ipx_row];
        double dual = ipx_row_dual[ipx_row];
        if (constraint_type[ipx_row] == '>') {
          // Row is at its lower bound
          highs_basis.row_status[row] = HighsBasisStatus::kLower;
          highs_solution.row_value[row] = value;
          highs_solution.row_dual[row] = dual;
        } else if (constraint_type[ipx_row] == '<') {
          // Row is at its upper bound
          highs_basis.row_status[row] = HighsBasisStatus::kUpper;
          highs_solution.row_value[row] = value;
          highs_solution.row_dual[row] = dual;
        } else if (constraint_type[ipx_row] == '=') {
          // Row is at its fixed value: set HighsBasisStatus according
          // to sign of dual.
          //
          // Don't worry about maximization problems. IPX solves them
          // as minimizations with negated costs, so a negative dual
          // yields HighsBasisStatus::kUpper here, and dual signs are
          // then flipped below, so HighsBasisStatus::kUpper will have
          // corresponding positive dual.
          highs_basis.row_status[row] =
              dual >= 0 ? HighsBasisStatus::kLower : HighsBasisStatus::kUpper;
          highs_solution.row_value[row] = value;
          highs_solution.row_dual[row] = dual;
        } else {
          unrecognised = true;
          highsLogDev(log_options, HighsLogType::kError,
                      "Error in IPX conversion: Row %2" HIGHSINT_FORMAT
                      ": cannot handle "
                      "constraint_type[%2" HIGHSINT_FORMAT
                      "] = %" HIGHSINT_FORMAT "\n",
                      row, ipx_row, constraint_type[ipx_row]);
        }
      }
      // Update the IPX row index
      ipx_row++;
    }
    if (unrecognised) {
      highsLogDev(log_options, HighsLogType::kError,
                  "Bounds [%11.4g, %11.4g]\n", lp.row_lower_[row],
                  lp.row_upper_[row]);
      highsLogDev(log_options, HighsLogType::kError,
                  "Row %2" HIGHSINT_FORMAT " ipx_row_status[%2" HIGHSINT_FORMAT
                  "] = %2" HIGHSINT_FORMAT "; s[%2" HIGHSINT_FORMAT
                  "] = %11.4g; y[%2" HIGHSINT_FORMAT
                  "] = "
                  "%11.4g\n",
                  row, this_ipx_row, (HighsInt)ipx_row_status[this_ipx_row],
                  this_ipx_row, ipx_row_value[this_ipx_row], this_ipx_row,
                  ipx_row_dual[this_ipx_row]);
      assert(!unrecognised);
      highsLogUser(log_options, HighsLogType::kError,
                   "Unrecognised ipx_row_status value from IPX\n");
      return HighsStatus::kError;
    }
    if (highs_basis.row_status[row] == HighsBasisStatus::kBasic)
      num_basic_variables++;
  }
  assert(num_basic_variables == lp.num_row_);
  assert(ipx_row == ipx_solution.num_row);
  assert(ipx_slack == ipx_solution.num_col);

  // Flip dual according to lp.sense_
  for (HighsInt iCol = 0; iCol < lp.num_col_; iCol++) {
    highs_solution.col_dual[iCol] *= (HighsInt)lp.sense_;
  }
  for (HighsInt iRow = 0; iRow < lp.num_row_; iRow++) {
    highs_solution.row_dual[iRow] *= (HighsInt)lp.sense_;
  }

  if (num_boxed_rows)
    highsLogDev(log_options, HighsLogType::kInfo,
                "Of %" HIGHSINT_FORMAT " boxed rows: %" HIGHSINT_FORMAT
                " are basic and %" HIGHSINT_FORMAT " have basic slacks\n",
                num_boxed_rows, num_boxed_rows_basic,
                num_boxed_row_slacks_basic);
  // Indicate that the primal solution, dual solution and basis are valid
  highs_solution.value_valid = true;
  highs_solution.dual_valid = true;
  highs_basis.valid = true;
  return HighsStatus::kOk;
}

HighsStatus formSimplexLpBasisAndFactor(HighsLpSolverObject& solver_object,
                                        const bool only_from_known_basis) {
  // Ideally, forms a SimplexBasis from the HighsBasis in the
  // HighsLpSolverObject
  //
  // If only_from_known_basis is true and
  // initialiseSimplexLpBasisAndFactor finds that there is no simplex
  // basis, then its error return is passed down
  //
  // If only_from_known_basis is false, then the basis is completed
  // with logicals if it is rank deficient (from singularity or being
  // incomplete)
  //
  HighsStatus return_status = HighsStatus::kOk;
  HighsStatus call_status;
  HighsLp& lp = solver_object.lp_;
  HighsBasis& basis = solver_object.basis_;
  HighsOptions& options = solver_object.options_;
  HEkk& ekk_instance = solver_object.ekk_instance_;
  HighsLp& ekk_lp = ekk_instance.lp_;
  HighsSimplexStatus& ekk_status = ekk_instance.status_;
  lp.ensureColwise();
  // Consider scaling the LP
  const bool new_scaling = considerScaling(options, lp);
  // If new scaling is performed, the hot start information is
  // no longer valid
  if (new_scaling) ekk_instance.clearHotStart();
  if (basis.alien) {
    // An alien basis needs to be checked for rank deficiency, and
    // possibly completed if it is rectangular
    assert(!only_from_known_basis);
    accommodateAlienBasis(solver_object);
    basis.alien = false;
    lp.unapplyScale();
    return HighsStatus::kOk;
  }
  // Move the HighsLpSolverObject's LP to EKK
  ekk_instance.moveLp(solver_object);
  if (!ekk_status.has_basis) {
    // The Ekk instance has no simplex basis, so pass the HiGHS basis
    HighsStatus call_status = ekk_instance.setBasis(basis);
    return_status = interpretCallStatus(options.log_options, call_status,
                                        return_status, "setBasis");
    if (return_status == HighsStatus::kError) return return_status;
  }
  // Now form the invert
  assert(ekk_status.has_basis);
  call_status =
      ekk_instance.initialiseSimplexLpBasisAndFactor(only_from_known_basis);
  if (call_status != HighsStatus::kOk) return HighsStatus::kError;
  // Once the invert is formed, move back the LP and remove any scaling.
  lp.moveBackLpAndUnapplyScaling(ekk_lp);
  // If the current basis cannot be inverted, return an error
  return HighsStatus::kOk;
}

void accommodateAlienBasis(HighsLpSolverObject& solver_object) {
  HighsLp& lp = solver_object.lp_;
  HighsBasis& basis = solver_object.basis_;
  HighsOptions& options = solver_object.options_;
  assert(basis.alien);
  HighsInt num_row = lp.num_row_;
  HighsInt num_col = lp.num_col_;
  assert((int)basis.col_status.size() >= num_col);
  assert((int)basis.row_status.size() >= num_row);
  std::vector<HighsInt> basic_index;
  for (HighsInt iCol = 0; iCol < num_col; iCol++) {
    if (basis.col_status[iCol] == HighsBasisStatus::kBasic)
      basic_index.push_back(iCol);
  }
  for (HighsInt iRow = 0; iRow < num_row; iRow++) {
    if (basis.row_status[iRow] == HighsBasisStatus::kBasic)
      basic_index.push_back(num_col + iRow);
  }
  HighsInt num_basic_variables = basic_index.size();
  HFactor factor;
  factor.setupGeneral(&lp.a_matrix_, num_basic_variables, &basic_index[0],
                      kDefaultPivotThreshold, kDefaultPivotTolerance,
                      kHighsDebugLevelMin, &options.log_options);
  HighsInt rank_deficiency = factor.build();
  // Deduce the basis from basic_index
  //
  // Set all basic variables to nonbasic
  for (HighsInt iCol = 0; iCol < num_col; iCol++) {
    if (basis.col_status[iCol] == HighsBasisStatus::kBasic)
      basis.col_status[iCol] = HighsBasisStatus::kNonbasic;
  }
  for (HighsInt iRow = 0; iRow < num_row; iRow++) {
    if (basis.row_status[iRow] == HighsBasisStatus::kBasic)
      basis.row_status[iRow] = HighsBasisStatus::kNonbasic;
  }
  // Set at most the first num_row variables in basic_index to basic
  const HighsInt use_basic_variables = std::min(num_row, num_basic_variables);
  // num_basic_variables is no longer needed, so can be used as a check
  num_basic_variables = 0;
  for (HighsInt iRow = 0; iRow < use_basic_variables; iRow++) {
    HighsInt iVar = basic_index[iRow];
    if (iVar < num_col) {
      basis.col_status[iVar] = HighsBasisStatus::kBasic;
    } else {
      basis.row_status[iVar - num_col] = HighsBasisStatus::kBasic;
    }
    num_basic_variables++;
  }
  // Complete the assignment of basic variables using the logicals of
  // non-pivotal rows
  const HighsInt num_missing = num_row - num_basic_variables;
  for (HighsInt k = 0; k < num_missing; k++) {
    HighsInt iRow = factor.row_with_no_pivot[rank_deficiency + k];
    basis.row_status[iRow] = HighsBasisStatus::kBasic;
    num_basic_variables++;
  }
  assert(num_basic_variables == num_row);
}

void resetModelStatusAndHighsInfo(HighsLpSolverObject& solver_object) {
  solver_object.model_status_ = HighsModelStatus::kNotset;
  solver_object.highs_info_.objective_function_value = 0;
  solver_object.highs_info_.primal_solution_status = kSolutionStatusNone;
  solver_object.highs_info_.dual_solution_status = kSolutionStatusNone;
  solver_object.highs_info_.num_primal_infeasibilities =
      kHighsIllegalInfeasibilityCount;
  solver_object.highs_info_.max_primal_infeasibility =
      kHighsIllegalInfeasibilityMeasure;
  solver_object.highs_info_.sum_primal_infeasibilities =
      kHighsIllegalInfeasibilityMeasure;
  solver_object.highs_info_.num_dual_infeasibilities =
      kHighsIllegalInfeasibilityCount;
  solver_object.highs_info_.max_dual_infeasibility =
      kHighsIllegalInfeasibilityMeasure;
  solver_object.highs_info_.sum_dual_infeasibilities =
      kHighsIllegalInfeasibilityMeasure;
}

void resetModelStatusAndHighsInfo(HighsModelStatus& model_status,
                                  HighsInfo& highs_info) {
  model_status = HighsModelStatus::kNotset;
  highs_info.objective_function_value = 0;
  highs_info.primal_solution_status = kSolutionStatusNone;
  highs_info.dual_solution_status = kSolutionStatusNone;
  highs_info.num_primal_infeasibilities = kHighsIllegalInfeasibilityCount;
  highs_info.max_primal_infeasibility = kHighsIllegalInfeasibilityMeasure;
  highs_info.sum_primal_infeasibilities = kHighsIllegalInfeasibilityMeasure;
  highs_info.num_dual_infeasibilities = kHighsIllegalInfeasibilityCount;
  highs_info.max_dual_infeasibility = kHighsIllegalInfeasibilityMeasure;
  highs_info.sum_dual_infeasibilities = kHighsIllegalInfeasibilityMeasure;
}

bool isBasisConsistent(const HighsLp& lp, const HighsBasis& basis) {
  bool consistent = true;
  consistent = isBasisRightSize(lp, basis) && consistent;
  if (consistent) {
    HighsInt num_basic_variables = 0;
    for (HighsInt iCol = 0; iCol < lp.num_col_; iCol++) {
      if (basis.col_status[iCol] == HighsBasisStatus::kBasic)
        num_basic_variables++;
    }
    for (HighsInt iRow = 0; iRow < lp.num_row_; iRow++) {
      if (basis.row_status[iRow] == HighsBasisStatus::kBasic)
        num_basic_variables++;
    }
    bool right_num_basic_variables = num_basic_variables == lp.num_row_;
    consistent = right_num_basic_variables && consistent;
  }
  return consistent;
}

bool isPrimalSolutionRightSize(const HighsLp& lp,
                               const HighsSolution& solution) {
  return (HighsInt)solution.col_value.size() == lp.num_col_ &&
         (HighsInt)solution.row_value.size() == lp.num_row_;
}

bool isDualSolutionRightSize(const HighsLp& lp, const HighsSolution& solution) {
  return (HighsInt)solution.col_dual.size() == lp.num_col_ &&
         (HighsInt)solution.row_dual.size() == lp.num_row_;
}

bool isSolutionRightSize(const HighsLp& lp, const HighsSolution& solution) {
  return isPrimalSolutionRightSize(lp, solution) &&
         isDualSolutionRightSize(lp, solution);
}

bool isBasisRightSize(const HighsLp& lp, const HighsBasis& basis) {
  return (HighsInt)basis.col_status.size() == lp.num_col_ &&
         (HighsInt)basis.row_status.size() == lp.num_row_;
}

void HighsSolution::invalidate() {
  this->value_valid = false;
  this->dual_valid = false;
}

void HighsSolution::clear() {
  this->invalidate();
  this->col_value.clear();
  this->row_value.clear();
  this->col_dual.clear();
  this->row_dual.clear();
}

void HighsBasis::invalidate() {
  this->valid = false;
  this->alien = true;
  this->was_alien = true;
  this->debug_id = -1;
  this->debug_update_count = -1;
  this->debug_origin_name = "None";
}

void HighsBasis::clear() {
  this->invalidate();
  this->row_status.clear();
  this->col_status.clear();
}
