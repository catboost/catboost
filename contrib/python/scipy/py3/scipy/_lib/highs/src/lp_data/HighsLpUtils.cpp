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
/**@file lp_data/HighsUtils.cpp
 * @brief Class-independent utilities for HiGHS
 */
#include "lp_data/HighsLpUtils.h"

#include <algorithm>
#include <cassert>

#include "HConfig.h"
#include "io/Filereader.h"
#include "io/HMPSIO.h"
#include "io/HighsIO.h"
#include "lp_data/HighsModelUtils.h"
#include "lp_data/HighsSolution.h"
#include "lp_data/HighsStatus.h"
#include "util/HighsCDouble.h"
#include "util/HighsMatrixUtils.h"
#include "util/HighsSort.h"
#include "util/HighsTimer.h"

using std::fabs;
using std::max;
using std::min;

HighsStatus assessLp(HighsLp& lp, const HighsOptions& options) {
  HighsStatus return_status = HighsStatus::kOk;
  HighsStatus call_status = lpDimensionsOk("assessLp", lp, options.log_options)
                                ? HighsStatus::kOk
                                : HighsStatus::kError;
  return_status = interpretCallStatus(options.log_options, call_status,
                                      return_status, "assessLpDimensions");
  if (return_status == HighsStatus::kError) return return_status;

  // If the LP has no columns there is nothing left to test
  if (lp.num_col_ == 0) return HighsStatus::kOk;
  assert(lp.a_matrix_.isColwise());

  // From here, any LP has lp.num_col_ > 0 and lp.a_matrix_.start_[lp.num_col_]
  // exists (as the number of nonzeros)
  assert(lp.num_col_ > 0);

  // Assess the LP column costs
  HighsIndexCollection index_collection;
  index_collection.dimension_ = lp.num_col_;
  index_collection.is_interval_ = true;
  index_collection.from_ = 0;
  index_collection.to_ = lp.num_col_ - 1;
  call_status = assessCosts(options, 0, index_collection, lp.col_cost_,
                            options.infinite_cost);
  return_status = interpretCallStatus(options.log_options, call_status,
                                      return_status, "assessCosts");
  if (return_status == HighsStatus::kError) return return_status;
  // Assess the LP column bounds
  call_status = assessBounds(options, "Col", 0, index_collection, lp.col_lower_,
                             lp.col_upper_, options.infinite_bound);
  return_status = interpretCallStatus(options.log_options, call_status,
                                      return_status, "assessBounds");
  if (return_status == HighsStatus::kError) return return_status;
  if (lp.num_row_) {
    // Assess the LP row bounds
    index_collection.dimension_ = lp.num_row_;
    index_collection.is_interval_ = true;
    index_collection.from_ = 0;
    index_collection.to_ = lp.num_row_ - 1;
    call_status =
        assessBounds(options, "Row", 0, index_collection, lp.row_lower_,
                     lp.row_upper_, options.infinite_bound);
    return_status = interpretCallStatus(options.log_options, call_status,
                                        return_status, "assessBounds");
    if (return_status == HighsStatus::kError) return return_status;
  }
  // Assess the LP matrix - even if there are no rows!
  call_status =
      lp.a_matrix_.assess(options.log_options, "LP", options.small_matrix_value,
                          options.large_matrix_value);
  return_status = interpretCallStatus(options.log_options, call_status,
                                      return_status, "assessMatrix");
  if (return_status == HighsStatus::kError) return return_status;
  HighsInt lp_num_nz = lp.a_matrix_.start_[lp.num_col_];
  // If entries have been removed from the matrix, resize the index
  // and value vectors to prevent bug in presolve
  if ((HighsInt)lp.a_matrix_.index_.size() > lp_num_nz)
    lp.a_matrix_.index_.resize(lp_num_nz);
  if ((HighsInt)lp.a_matrix_.value_.size() > lp_num_nz)
    lp.a_matrix_.value_.resize(lp_num_nz);
  if ((HighsInt)lp.a_matrix_.index_.size() > lp_num_nz)
    lp.a_matrix_.index_.resize(lp_num_nz);
  if ((HighsInt)lp.a_matrix_.value_.size() > lp_num_nz)
    lp.a_matrix_.value_.resize(lp_num_nz);

  //  if (return_status == HighsStatus::kError)
  //    return_status = HighsStatus::kError;
  //  else
  //    return_status = HighsStatus::kOk;
  if (return_status != HighsStatus::kOk)
    highsLogDev(options.log_options, HighsLogType::kInfo,
                "assessLp returns HighsStatus = %s\n",
                highsStatusToString(return_status).c_str());
  return return_status;
}

bool lpDimensionsOk(std::string message, const HighsLp& lp,
                    const HighsLogOptions& log_options) {
  bool ok = true;
  const HighsInt num_col = lp.num_col_;
  const HighsInt num_row = lp.num_row_;
  if (!(num_col >= 0))
    highsLogUser(log_options, HighsLogType::kError,
                 "LP dimension validation (%s) fails on num_col = %d >= 0\n",
                 message.c_str(), (int)num_col);
  ok = num_col >= 0 && ok;
  if (!(num_row >= 0))
    highsLogUser(log_options, HighsLogType::kError,
                 "LP dimension validation (%s) fails on num_row = %d >= 0\n",
                 message.c_str(), (int)num_row);
  ok = num_row >= 0 && ok;
  if (!ok) return ok;

  HighsInt col_cost_size = lp.col_cost_.size();
  HighsInt col_lower_size = lp.col_lower_.size();
  HighsInt col_upper_size = lp.col_upper_.size();
  HighsInt matrix_start_size = lp.a_matrix_.start_.size();
  bool legal_col_cost_size = col_cost_size >= num_col;
  bool legal_col_lower_size = col_lower_size >= num_col;
  bool legal_col_upper_size = col_lower_size >= num_col;
  if (!legal_col_cost_size)
    highsLogUser(log_options, HighsLogType::kError,
                 "LP dimension validation (%s) fails on col_cost.size() = %d < "
                 "%d = num_col\n",
                 message.c_str(), (int)col_cost_size, (int)num_col);
  ok = legal_col_cost_size && ok;
  if (!legal_col_lower_size)
    highsLogUser(log_options, HighsLogType::kError,
                 "LP dimension validation (%s) fails on col_lower.size() = %d "
                 "< %d = num_col\n",
                 message.c_str(), (int)col_lower_size, (int)num_col);
  ok = legal_col_lower_size && ok;
  if (!legal_col_upper_size)
    highsLogUser(log_options, HighsLogType::kError,
                 "LP dimension validation (%s) fails on col_upper.size() = %d "
                 "< %d = num_col\n",
                 message.c_str(), (int)col_upper_size, (int)num_col);
  ok = legal_col_upper_size && ok;

  bool legal_format = lp.a_matrix_.format_ == MatrixFormat::kColwise ||
                      lp.a_matrix_.format_ == MatrixFormat::kRowwise;
  if (!legal_format)
    highsLogUser(log_options, HighsLogType::kError,
                 "LP dimension validation (%s) fails on a_matrix_.format\n",
                 message.c_str());
  ok = legal_format && ok;
  HighsInt num_vec;
  if (lp.a_matrix_.isColwise()) {
    num_vec = num_col;
  } else {
    num_vec = num_row;
  }
  const bool partitioned = false;
  vector<HighsInt> a_matrix_p_end;
  bool legal_matrix_dimensions =
      assessMatrixDimensions(log_options, num_vec, partitioned,
                             lp.a_matrix_.start_, a_matrix_p_end,
                             lp.a_matrix_.index_,
                             lp.a_matrix_.value_) == HighsStatus::kOk;
  if (!legal_matrix_dimensions)
    highsLogUser(log_options, HighsLogType::kError,
                 "LP dimension validation (%s) fails on a_matrix dimensions\n",
                 message.c_str());
  ok = legal_matrix_dimensions && ok;

  HighsInt row_lower_size = lp.row_lower_.size();
  HighsInt row_upper_size = lp.row_upper_.size();
  bool legal_row_lower_size = row_lower_size >= num_row;
  bool legal_row_upper_size = row_upper_size >= num_row;
  if (!legal_row_lower_size)
    highsLogUser(log_options, HighsLogType::kError,
                 "LP dimension validation (%s) fails on row_lower.size() = %d "
                 "< %d = num_row\n",
                 message.c_str(), (int)row_lower_size, (int)num_row);
  ok = legal_row_lower_size && ok;
  if (!legal_row_upper_size)
    highsLogUser(log_options, HighsLogType::kError,
                 "LP dimension validation (%s) fails on row_upper.size() = %d "
                 "< %d = num_row\n",
                 message.c_str(), (int)row_upper_size, (int)num_row);
  ok = legal_row_upper_size && ok;

  bool legal_a_matrix_num_col = lp.a_matrix_.num_col_ == num_col;
  bool legal_a_matrix_num_row = lp.a_matrix_.num_row_ == num_row;
  if (!legal_a_matrix_num_col)
    highsLogUser(log_options, HighsLogType::kError,
                 "LP dimension validation (%s) fails on a_matrix.num_col_ = %d "
                 "!= %d = num_col\n",
                 message.c_str(), (int)lp.a_matrix_.num_col_, (int)num_col);
  ok = legal_a_matrix_num_col && ok;
  if (!legal_a_matrix_num_row)
    highsLogUser(log_options, HighsLogType::kError,
                 "LP dimension validation (%s) fails on a_matrix.num_row_ = %d "
                 "!= %d = num_row\n",
                 message.c_str(), (int)lp.a_matrix_.num_row_, (int)num_row);
  ok = legal_a_matrix_num_row && ok;

  HighsInt scale_strategy = (HighsInt)lp.scale_.strategy;
  bool legal_scale_strategy = scale_strategy >= 0;
  if (!legal_scale_strategy)
    highsLogUser(
        log_options, HighsLogType::kError,
        "LP dimension validation (%s) fails on scale_.scale_strategy\n",
        message.c_str());
  ok = legal_scale_strategy && ok;
  HighsInt scale_row_size = (HighsInt)lp.scale_.row.size();
  HighsInt scale_col_size = (HighsInt)lp.scale_.col.size();
  bool legal_scale_num_col = false;
  bool legal_scale_num_row = false;
  bool legal_scale_row_size = false;
  bool legal_scale_col_size = false;
  if (lp.scale_.has_scaling) {
    legal_scale_num_col = lp.scale_.num_col == num_col;
    legal_scale_num_row = lp.scale_.num_row == num_row;
    legal_scale_row_size = scale_row_size >= num_row;
    legal_scale_col_size = scale_col_size >= num_col;
  } else {
    legal_scale_num_col = lp.scale_.num_col == 0;
    legal_scale_num_row = lp.scale_.num_row == 0;
    legal_scale_row_size = scale_row_size == 0;
    legal_scale_col_size = scale_col_size == 0;
  }
  if (!legal_scale_num_col)
    highsLogUser(
        log_options, HighsLogType::kError,
        "LP dimension validation (%s) fails on scale_.num_col = %d != %d\n",
        message.c_str(), (int)lp.scale_.num_col,
        (int)(lp.scale_.has_scaling ? num_col : 0));
  ok = legal_scale_num_col && ok;
  if (!legal_scale_num_row)
    highsLogUser(
        log_options, HighsLogType::kError,
        "LP dimension validation (%s) fails on scale_.num_row = %d != %d\n",
        message.c_str(), (int)lp.scale_.num_row,
        (int)(lp.scale_.has_scaling ? num_row : 0));
  ok = legal_scale_num_row && ok;
  if (!legal_scale_col_size)
    highsLogUser(
        log_options, HighsLogType::kError,
        "LP dimension validation (%s) fails on scale_.col.size() = %d %s %d\n",
        message.c_str(), (int)scale_col_size,
        lp.scale_.has_scaling ? ">=" : "==",
        (int)(lp.scale_.has_scaling ? num_col : 0));
  ok = legal_scale_col_size && ok;
  if (!legal_scale_row_size)
    highsLogUser(
        log_options, HighsLogType::kError,
        "LP dimension validation (%s) fails on scale_.row.size() = %d %s %d\n",
        message.c_str(), (int)scale_row_size,
        lp.scale_.has_scaling ? ">=" : "==",
        (int)(lp.scale_.has_scaling ? num_row : 0));
  ok = legal_scale_row_size && ok;
  if (!ok) {
    highsLogUser(log_options, HighsLogType::kError,
                 "LP dimension validation (%s) fails\n", message.c_str());
  }

  return ok;
}

HighsStatus assessCosts(const HighsOptions& options, const HighsInt ml_col_os,
                        const HighsIndexCollection& index_collection,
                        vector<double>& cost, const double infinite_cost) {
  HighsStatus return_status = HighsStatus::kOk;
  assert(ok(index_collection));
  HighsInt from_k;
  HighsInt to_k;
  limits(index_collection, from_k, to_k);
  if (from_k > to_k) return return_status;

  return_status = HighsStatus::kOk;
  bool error_found = false;
  // Work through the data to be assessed.
  //
  // Loop is k \in [from_k...to_k) covering the entries in the
  // interval, set or mask to be considered.
  //
  // For an interval or mask, these values of k are the columns to be
  // considered in a local sense, as well as the entries in the
  // cost data to be assessed
  //
  // For a set, these values of k are the indices in the set, from
  // which the columns to be considered in a local sense are
  // drawn. The entries in the cost data to be assessed correspond
  // to the values of k
  //
  // Adding the value of ml_col_os to local_col yields the value of
  // ml_col, being the column in a global (whole-model) sense. This is
  // necessary when assessing the costs of columns being added to a
  // model, since they are specified using an interval
  // [0...num_new_col) which must be offset by the current number of
  // columns in the model.
  //
  HighsInt local_col;
  HighsInt ml_col;
  HighsInt usr_col = -1;
  for (HighsInt k = from_k; k < to_k + 1; k++) {
    if (index_collection.is_interval_ || index_collection.is_mask_) {
      local_col = k;
    } else {
      local_col = index_collection.set_[k];
    }
    if (index_collection.is_interval_) {
      usr_col++;
    } else {
      usr_col = k;
    }
    ml_col = ml_col_os + local_col;
    if (index_collection.is_mask_ && !index_collection.mask_[local_col])
      continue;
    double abs_cost = fabs(cost[usr_col]);
    bool legal_cost = abs_cost < infinite_cost;
    if (!legal_cost) {
      error_found = !kHighsAllowInfiniteCosts;
      HighsLogType log_type = HighsLogType::kWarning;
      if (error_found) log_type = HighsLogType::kError;
      highsLogUser(options.log_options, log_type,
                   "Col  %12" HIGHSINT_FORMAT " has |cost| of %12g >= %12g\n",
                   ml_col, abs_cost, infinite_cost);
    }
  }
  if (error_found)
    return_status = HighsStatus::kError;
  else
    return_status = HighsStatus::kOk;

  return return_status;
}

HighsStatus assessBounds(const HighsOptions& options, const char* type,
                         const HighsInt ml_ix_os,
                         const HighsIndexCollection& index_collection,
                         vector<double>& lower, vector<double>& upper,
                         const double infinite_bound) {
  HighsStatus return_status = HighsStatus::kOk;
  assert(ok(index_collection));
  HighsInt from_k;
  HighsInt to_k;
  limits(index_collection, from_k, to_k);
  if (from_k > to_k) return HighsStatus::kOk;

  return_status = HighsStatus::kOk;
  bool error_found = false;
  bool warning_found = false;
  // Work through the data to be assessed.
  //
  // Loop is k \in [from_k...to_k) covering the entries in the
  // interval, set or mask to be considered.
  //
  // For an interval or mask, these values of k are the row/column
  // indices to be considered in a local sense, as well as the entries
  // in the lower and upper bound data to be assessed
  //
  // For a set, these values of k are the indices in the set, from
  // which the indices to be considered in a local sense are
  // drawn. The entries in the lower and
  // upper bound data to be assessed correspond to the values of
  // k.
  //
  // Adding the value of ml_ix_os to local_ix yields the value of
  // ml_ix, being the index in a global (whole-model) sense. This is
  // necessary when assessing the bounds of rows/columns being added
  // to a model, since they are specified using an interval
  // [0...num_new_row/col) which must be offset by the current number
  // of rows/columns (generically indices) in the model.
  //
  HighsInt num_infinite_lower_bound = 0;
  HighsInt num_infinite_upper_bound = 0;
  HighsInt local_ix;
  HighsInt ml_ix;
  HighsInt usr_ix = -1;
  for (HighsInt k = from_k; k < to_k + 1; k++) {
    if (index_collection.is_interval_ || index_collection.is_mask_) {
      local_ix = k;
    } else {
      local_ix = index_collection.set_[k];
    }
    if (index_collection.is_interval_) {
      usr_ix++;
    } else {
      usr_ix = k;
    }
    ml_ix = ml_ix_os + local_ix;
    if (index_collection.is_mask_ && !index_collection.mask_[local_ix])
      continue;

    if (!highs_isInfinity(-lower[usr_ix])) {
      // Check whether a finite lower bound will be treated as -Infinity
      bool infinite_lower_bound = lower[usr_ix] <= -infinite_bound;
      if (infinite_lower_bound) {
        lower[usr_ix] = -kHighsInf;
        num_infinite_lower_bound++;
      }
    }
    if (!highs_isInfinity(upper[usr_ix])) {
      // Check whether a finite upper bound will be treated as Infinity
      bool infinite_upper_bound = upper[usr_ix] >= infinite_bound;
      if (infinite_upper_bound) {
        upper[usr_ix] = kHighsInf;
        num_infinite_upper_bound++;
      }
    }
    // Check that the lower bound does not exceed the upper bound
    bool legalLowerUpperBound = lower[usr_ix] <= upper[usr_ix];
    if (!legalLowerUpperBound) {
      // Leave inconsistent bounds to be used to deduce infeasibility
      highsLogUser(options.log_options, HighsLogType::kWarning,
                   "%3s  %12" HIGHSINT_FORMAT
                   " has inconsistent bounds [%12g, %12g]\n",
                   type, ml_ix, lower[usr_ix], upper[usr_ix]);
      warning_found = true;
    }
    // Check that the lower bound is not as much as +Infinity
    bool legalLowerBound = lower[usr_ix] < infinite_bound;
    if (!legalLowerBound) {
      highsLogUser(options.log_options, HighsLogType::kError,
                   "%3s  %12" HIGHSINT_FORMAT
                   " has lower bound of %12g >= %12g\n",
                   type, ml_ix, lower[usr_ix], infinite_bound);
      error_found = true;
    }
    // Check that the upper bound is not as little as -Infinity
    bool legalUpperBound = upper[usr_ix] > -infinite_bound;
    if (!legalUpperBound) {
      highsLogUser(options.log_options, HighsLogType::kError,
                   "%3s  %12" HIGHSINT_FORMAT
                   " has upper bound of %12g <= %12g\n",
                   type, ml_ix, upper[usr_ix], -infinite_bound);
      error_found = true;
    }
  }
  if (num_infinite_lower_bound) {
    highsLogUser(options.log_options, HighsLogType::kInfo,
                 "%3ss:%12" HIGHSINT_FORMAT
                 " lower bounds exceeding %12g are treated as -Infinity\n",
                 type, num_infinite_lower_bound, -infinite_bound);
  }
  if (num_infinite_upper_bound) {
    highsLogUser(options.log_options, HighsLogType::kInfo,
                 "%3ss:%12" HIGHSINT_FORMAT
                 " upper bounds exceeding %12g are treated as +Infinity\n",
                 type, num_infinite_upper_bound, infinite_bound);
  }

  if (error_found)
    return_status = HighsStatus::kError;
  else if (warning_found)
    return_status = HighsStatus::kWarning;
  else
    return_status = HighsStatus::kOk;

  return return_status;
}

HighsStatus assessIntegrality(HighsLp& lp, const HighsOptions& options) {
  HighsStatus return_status = HighsStatus::kOk;
  if (!lp.integrality_.size()) return return_status;
  assert(lp.integrality_.size() == lp.num_col_);
  HighsInt num_illegal_lower = 0;
  HighsInt num_illegal_upper = 0;
  HighsInt num_modified_upper = 0;
  HighsInt num_non_semi = 0;
  HighsInt num_non_continuous_variables = 0;
  const double kLowerBoundMu = 10.0;
  std::vector<HighsInt>& upper_bound_index =
      lp.mods_.save_semi_variable_upper_bound_index;
  std::vector<double>& upper_bound_value =
      lp.mods_.save_semi_variable_upper_bound_value;

  for (HighsInt iCol = 0; iCol < lp.num_col_; iCol++) {
    if (lp.integrality_[iCol] == HighsVarType::kSemiContinuous ||
        lp.integrality_[iCol] == HighsVarType::kSemiInteger) {
      // Semi-variables with zero lower bound aren't semi
      if (lp.col_lower_[iCol] == 0) {
        num_non_semi++;
        if (lp.integrality_[iCol] == HighsVarType::kSemiContinuous) {
          // Semi-continuous become continuous
          lp.integrality_[iCol] = HighsVarType::kContinuous;
        } else {
          // Semi-integer become integer
          lp.integrality_[iCol] = HighsVarType::kInteger;
          num_non_continuous_variables++;
        }
        continue;
      }
      if (lp.col_lower_[iCol] < 0) {
        // Semi-variables must have a positive lower bound
        num_illegal_lower++;
      } else if (lp.col_upper_[iCol] > kMaxSemiVariableUpper) {
        // Semi-variables must have upper bound that's not too large,
        // so see whether the limiting value is sufficiently larger than the
        // lower bound
        if (kLowerBoundMu * lp.col_lower_[iCol] > kMaxSemiVariableUpper) {
          num_illegal_upper++;
        } else {
          // Record the upper bound change
          upper_bound_index.push_back(iCol);
          upper_bound_value.push_back(kMaxSemiVariableUpper);
          num_modified_upper++;
        }
      }
      num_non_continuous_variables++;
    } else if (lp.integrality_[iCol] == HighsVarType::kInteger) {
      num_non_continuous_variables++;
    }
  }
  if (num_non_semi) {
    highsLogUser(options.log_options, HighsLogType::kWarning,
                 "%" HIGHSINT_FORMAT
                 " semi-continuous/integer variable(s) have zero lower bound "
                 "so are continuous/integer\n",
                 num_non_semi);
    return_status = HighsStatus::kWarning;
  }
  if (!num_non_continuous_variables) {
    highsLogUser(options.log_options, HighsLogType::kWarning,
                 "No semi-integer/integer variables in model with non-empty "
                 "integrality\n");
    return_status = HighsStatus::kWarning;
  }
  const bool has_illegal_bounds = num_illegal_lower || num_illegal_upper;
  if (num_modified_upper) {
    highsLogUser(options.log_options, HighsLogType::kWarning,
                 "%" HIGHSINT_FORMAT
                 " semi-continuous/integer variable(s) have upper bounds "
                 "exceeding %g that can be modified to %g > %g*lower)\n",
                 num_modified_upper, kMaxSemiVariableUpper,
                 kMaxSemiVariableUpper, kLowerBoundMu);
    return_status = HighsStatus::kWarning;
    if (has_illegal_bounds) {
      // Don't apply upper bound modifications if there are illegal bounds
      assert(num_illegal_lower || num_illegal_upper);
      upper_bound_index.clear();
      upper_bound_value.clear();
    } else {
      // Apply the upper bound modifications, saving the over-written
      // values
      for (HighsInt k = 0; k < num_modified_upper; k++) {
        const double use_upper_bound = upper_bound_value[k];
        const HighsInt iCol = upper_bound_index[k];
        upper_bound_value[k] = lp.col_upper_[iCol];
        lp.col_upper_[iCol] = use_upper_bound;
      }
    }
  }
  if (num_illegal_lower) {
    highsLogUser(
        options.log_options, HighsLogType::kError,
        "%" HIGHSINT_FORMAT
        " semi-continuous/integer variable(s) have negative lower bounds\n",
        num_illegal_lower);
    return_status = HighsStatus::kError;
  }
  if (num_illegal_upper) {
    highsLogUser(
        options.log_options, HighsLogType::kError,
        "%" HIGHSINT_FORMAT
        " semi-continuous/integer variables have upper bounds "
        "exceeding %g that cannot be modified due to large lower bounds\n",
        num_illegal_upper, kMaxSemiVariableUpper);
    return_status = HighsStatus::kError;
  }
  return return_status;
}

bool activeModifiedUpperBounds(const HighsOptions& options, const HighsLp& lp,
                               const std::vector<double> col_value) {
  const std::vector<HighsInt>& upper_bound_index =
      lp.mods_.save_semi_variable_upper_bound_index;
  const HighsInt num_modified_upper = upper_bound_index.size();
  HighsInt num_active_modified_upper = 0;
  double min_semi_variable_margin = kHighsInf;
  for (HighsInt k = 0; k < num_modified_upper; k++) {
    const double value = col_value[upper_bound_index[k]];
    const double upper = lp.col_upper_[upper_bound_index[k]];
    double semi_variable_margin = upper - value;
    if (value > upper - options.primal_feasibility_tolerance) {
      min_semi_variable_margin = 0;
      num_active_modified_upper++;
    } else {
      min_semi_variable_margin =
          std::min(semi_variable_margin, min_semi_variable_margin);
    }
  }
  if (num_active_modified_upper) {
    highsLogUser(options.log_options, HighsLogType::kError,
                 "%" HIGHSINT_FORMAT
                 " semi-variables are active at modified upper bounds\n",
                 num_active_modified_upper);
  } else if (num_modified_upper) {
    highsLogUser(options.log_options, HighsLogType::kWarning,
                 "No semi-variables are active at modified upper bounds:"
                 " a large minimum margin (%g) suggests optimality,"
                 " but there is no guarantee\n",
                 min_semi_variable_margin);
  }
  return num_active_modified_upper;
}

HighsStatus cleanBounds(const HighsOptions& options, HighsLp& lp) {
  double max_residual = 0;
  HighsInt num_change = 0;
  for (HighsInt iCol = 0; iCol < lp.num_col_; iCol++) {
    double residual = lp.col_lower_[iCol] - lp.col_upper_[iCol];
    if (residual > options.primal_feasibility_tolerance) {
      highsLogUser(options.log_options, HighsLogType::kError,
                   "Column %" HIGHSINT_FORMAT
                   " has inconsistent bounds [%g, %g] (residual = "
                   "%g) after presolve\n",
                   iCol, lp.col_lower_[iCol], lp.col_upper_[iCol], residual);
      return HighsStatus::kError;
    } else if (residual > 0) {
      num_change++;
      max_residual = std::max(residual, max_residual);
      double mid = 0.5 * (lp.col_lower_[iCol] + lp.col_upper_[iCol]);
      lp.col_lower_[iCol] = mid;
      lp.col_upper_[iCol] = mid;
    }
  }
  for (HighsInt iRow = 0; iRow < lp.num_row_; iRow++) {
    double residual = lp.row_lower_[iRow] - lp.row_upper_[iRow];
    if (residual > options.primal_feasibility_tolerance) {
      highsLogUser(options.log_options, HighsLogType::kError,
                   "Row %" HIGHSINT_FORMAT
                   " has inconsistent bounds [%g, %g] (residual = %g) "
                   "after presolve\n",
                   iRow, lp.row_lower_[iRow], lp.row_upper_[iRow], residual);
      return HighsStatus::kError;
    } else if (residual > 0) {
      num_change++;
      max_residual = std::max(residual, max_residual);
      double mid = 0.5 * (lp.row_lower_[iRow] + lp.row_upper_[iRow]);
      lp.row_lower_[iRow] = mid;
      lp.row_upper_[iRow] = mid;
    }
  }
  if (num_change) {
    highsLogUser(options.log_options, HighsLogType::kWarning,
                 "Resolved %" HIGHSINT_FORMAT
                 " inconsistent bounds (maximum residual = "
                 "%9.4g) after presolve\n",
                 num_change, max_residual);
    return HighsStatus::kWarning;
  }
  return HighsStatus::kOk;
}

bool considerScaling(const HighsOptions& options, HighsLp& lp) {
  // Indicate whether new scaling has been determined in the return value.
  bool new_scaling = false;
  // Consider scaling the LP - either by finding new factors or by
  // applying any existing factors
  const bool allow_scaling =
      lp.num_col_ > 0 &&
      options.simplex_scale_strategy != kSimplexScaleStrategyOff;
  if (lp.scale_.has_scaling && !allow_scaling) {
    // LP had scaling before, but now it is not permitted, clear any
    // scaling. Return true as scaling position has changed
    lp.clearScale();
    return true;
  }
  const bool scaling_not_tried = lp.scale_.strategy == kSimplexScaleStrategyOff;
  const bool new_scaling_strategy =
      options.simplex_scale_strategy != lp.scale_.strategy &&
      options.simplex_scale_strategy != kSimplexScaleStrategyChoose;
  const bool try_scaling =
      allow_scaling && (scaling_not_tried || new_scaling_strategy);
  if (try_scaling) {
    // Scaling will be tried, so ensure that any previous scaling is not applied
    lp.unapplyScale();
    const bool analyse_lp_data =
        kHighsAnalysisLevelModelData & options.highs_analysis_level;
    if (analyse_lp_data) analyseLp(options.log_options, lp);
    scaleLp(options, lp);
    // If the LP is now scaled, then the scaling is new
    new_scaling = lp.is_scaled_;
    if (analyse_lp_data && lp.is_scaled_) analyseLp(options.log_options, lp);
  } else if (lp.scale_.has_scaling) {
    // Scaling factors are known, so ensure that they are applied
    lp.applyScale();
  }
  // Ensure that either the LP has scale factors and is scaled, or
  // it doesn't have scale factors and isn't scaled
  assert(lp.scale_.has_scaling == lp.is_scaled_);
  return new_scaling;
}

void scaleLp(const HighsOptions& options, HighsLp& lp) {
  lp.clearScaling();
  HighsInt numCol = lp.num_col_;
  HighsInt numRow = lp.num_row_;
  // Scaling not well defined for models with no columns
  assert(numCol > 0);
  vector<double>& colCost = lp.col_cost_;
  vector<double>& colLower = lp.col_lower_;
  vector<double>& colUpper = lp.col_upper_;
  vector<double>& rowLower = lp.row_lower_;
  vector<double>& rowUpper = lp.row_upper_;

  // Save the simplex_scale_strategy so that the option can be
  // modified for the course of this method
  HighsInt simplex_scale_strategy = options.simplex_scale_strategy;
  // Determine the actual strategy to use
  HighsInt use_scale_strategy = simplex_scale_strategy;
  if (use_scale_strategy == kSimplexScaleStrategyChoose) {
    // HiGHS is left to choose: currently use forced equilibration, but maybe do
    // something more intelligent
    use_scale_strategy = kSimplexScaleStrategyForcedEquilibration;
  }
  bool allow_cost_scaling = options.allowed_cost_scale_factor > 0;
  // Find out range of matrix values and skip matrix scaling if all
  // |values| are in [0.2, 5]
  const double no_scaling_original_matrix_min_value = 0.2;
  const double no_scaling_original_matrix_max_value = 5.0;
  double original_matrix_min_value = kHighsInf;
  double original_matrix_max_value = 0;
  lp.a_matrix_.range(original_matrix_min_value, original_matrix_max_value);
  bool no_scaling =
      (original_matrix_min_value >= no_scaling_original_matrix_min_value) &&
      (original_matrix_max_value <= no_scaling_original_matrix_max_value);
  const bool force_scaling = false;
  if (force_scaling) {
    no_scaling = false;
    printf("!!!! FORCE SCALING !!!!\n");
  }
  bool scaled_matrix = false;
  if (no_scaling) {
    // No matrix scaling, but possible cost scaling
    if (options.highs_analysis_level)
      highsLogDev(options.log_options, HighsLogType::kInfo,
                  "Scaling: Matrix has [min, max] values of [%g, %g] within "
                  "[%g, %g] so no scaling performed\n",
                  original_matrix_min_value, original_matrix_max_value,
                  no_scaling_original_matrix_min_value,
                  no_scaling_original_matrix_max_value);
  } else {
    // Try scaling, so assign unit factors - partly because initial
    // factors may be assumed by the scaling method, but also because
    // scaling factors may not be computed for empty rows/columns
    HighsScale& scale = lp.scale_;
    scale.col.assign(numCol, 1);
    scale.row.assign(numRow, 1);
    const bool equilibration_scaling =
        use_scale_strategy == kSimplexScaleStrategyEquilibration ||
        use_scale_strategy == kSimplexScaleStrategyForcedEquilibration;
    // Try scaling. Value of scaled_matrix indicates whether scaling
    // was considered valuable (and performed). If it's not valuable
    // then the matrix remains unscaled
    if (equilibration_scaling) {
      scaled_matrix = equilibrationScaleMatrix(options, lp, use_scale_strategy);
    } else {
      scaled_matrix = maxValueScaleMatrix(options, lp, use_scale_strategy);
    }
    if (scaled_matrix) {
      // Matrix is scaled, so scale the bounds and costs
      for (HighsInt iCol = 0; iCol < numCol; iCol++) {
        colLower[iCol] /= scale.col[iCol];
        colUpper[iCol] /= scale.col[iCol];
        colCost[iCol] *= scale.col[iCol];
      }
      for (HighsInt iRow = 0; iRow < numRow; iRow++) {
        rowLower[iRow] *= scale.row[iRow];
        rowUpper[iRow] *= scale.row[iRow];
      }
      scale.has_scaling = true;
      scale.num_col = numCol;
      scale.num_row = numRow;
      scale.cost = 1.0;
      lp.is_scaled_ = true;
    } else {
      // Matrix is not scaled, so clear the scaling
      lp.clearScaling();
    }
  }
  // Record the scaling strategy used
  lp.scale_.strategy = use_scale_strategy;
  // Possibly scale the costs
  //  if (allow_cost_scaling) scaleSimplexCost(options, lp, scale.cost);

  // If matrix is unscaled, then LP is only scaled if there is a cost scaling
  // factor
  //  if (!scaled_matrix) lp.is_scaled_ = scale.cost != 1;
}

bool equilibrationScaleMatrix(const HighsOptions& options, HighsLp& lp,
                              const HighsInt use_scale_strategy) {
  HighsInt numCol = lp.num_col_;
  HighsInt numRow = lp.num_row_;
  HighsScale& scale = lp.scale_;
  vector<double>& colScale = scale.col;
  vector<double>& rowScale = scale.row;
  vector<HighsInt>& Astart = lp.a_matrix_.start_;
  vector<HighsInt>& Aindex = lp.a_matrix_.index_;
  vector<double>& Avalue = lp.a_matrix_.value_;
  vector<double>& colCost = lp.col_cost_;

  HighsInt simplex_scale_strategy = use_scale_strategy;

  double original_matrix_min_value = kHighsInf;
  double original_matrix_max_value = 0;
  for (HighsInt k = 0, AnX = Astart[numCol]; k < AnX; k++) {
    double value = fabs(Avalue[k]);
    original_matrix_min_value = min(original_matrix_min_value, value);
    original_matrix_max_value = max(original_matrix_max_value, value);
  }

  // Include cost in scaling if minimum nonzero cost is less than 0.1
  double min_nonzero_cost = kHighsInf;
  for (HighsInt i = 0; i < numCol; i++) {
    if (colCost[i]) min_nonzero_cost = min(fabs(colCost[i]), min_nonzero_cost);
  }
  bool include_cost_in_scaling = false;
  include_cost_in_scaling = min_nonzero_cost < 0.1;

  // Limits on scaling factors
  double max_allow_scale;
  double min_allow_scale;
  // Now that kHighsInf =
  // std::numeric_limits<double>::infinity(), this Qi-trick doesn't
  // work so, in recognition, use the old value of kHighsInf
  const double finite_infinity = 1e200;
  max_allow_scale = pow(2.0, options.allowed_matrix_scale_factor);
  min_allow_scale = 1 / max_allow_scale;

  double min_allow_col_scale = min_allow_scale;
  double max_allow_col_scale = max_allow_scale;
  double min_allow_row_scale = min_allow_scale;
  double max_allow_row_scale = max_allow_scale;

  // Search up to 6 times
  vector<double> row_min_value(numRow, finite_infinity);
  vector<double> row_max_value(numRow, 1 / finite_infinity);
  for (HighsInt search_count = 0; search_count < 6; search_count++) {
    // Find column scale, prepare row data
    for (HighsInt iCol = 0; iCol < numCol; iCol++) {
      // For column scale (find)
      double col_min_value = finite_infinity;
      double col_max_value = 1 / finite_infinity;
      double abs_col_cost = fabs(colCost[iCol]);
      if (include_cost_in_scaling && abs_col_cost != 0) {
        col_min_value = min(col_min_value, abs_col_cost);
        col_max_value = max(col_max_value, abs_col_cost);
      }
      for (HighsInt k = Astart[iCol]; k < Astart[iCol + 1]; k++) {
        double value = fabs(Avalue[k]) * rowScale[Aindex[k]];
        col_min_value = min(col_min_value, value);
        col_max_value = max(col_max_value, value);
      }
      double col_equilibration = 1 / sqrt(col_min_value * col_max_value);
      // Ensure that column scale factor is not excessively large or small
      colScale[iCol] =
          min(max(min_allow_col_scale, col_equilibration), max_allow_col_scale);
      // For row scale (only collect)
      for (HighsInt k = Astart[iCol]; k < Astart[iCol + 1]; k++) {
        HighsInt iRow = Aindex[k];
        double value = fabs(Avalue[k]) * colScale[iCol];
        row_min_value[iRow] = min(row_min_value[iRow], value);
        row_max_value[iRow] = max(row_max_value[iRow], value);
      }
    }
    // For row scale (find)
    for (HighsInt iRow = 0; iRow < numRow; iRow++) {
      double row_equilibration =
          1 / sqrt(row_min_value[iRow] * row_max_value[iRow]);
      // Ensure that row scale factor is not excessively large or small
      rowScale[iRow] =
          min(max(min_allow_row_scale, row_equilibration), max_allow_row_scale);
    }
    row_min_value.assign(numRow, finite_infinity);
    row_max_value.assign(numRow, 1 / finite_infinity);
  }
  // Make it numerically better
  // Also determine the max and min row and column scaling factors
  double min_col_scale = finite_infinity;
  double max_col_scale = 1 / finite_infinity;
  double min_row_scale = finite_infinity;
  double max_row_scale = 1 / finite_infinity;
  const double log2 = log(2.0);
  for (HighsInt iCol = 0; iCol < numCol; iCol++) {
    colScale[iCol] = pow(2.0, floor(log(colScale[iCol]) / log2 + 0.5));
    min_col_scale = min(colScale[iCol], min_col_scale);
    max_col_scale = max(colScale[iCol], max_col_scale);
  }
  for (HighsInt iRow = 0; iRow < numRow; iRow++) {
    rowScale[iRow] = pow(2.0, floor(log(rowScale[iRow]) / log2 + 0.5));
    min_row_scale = min(rowScale[iRow], min_row_scale);
    max_row_scale = max(rowScale[iRow], max_row_scale);
  }
  // Apply scaling to matrix and bounds
  double matrix_min_value = finite_infinity;
  double matrix_max_value = 0;
  double min_original_col_equilibration = finite_infinity;
  double sum_original_log_col_equilibration = 0;
  double max_original_col_equilibration = 0;
  double min_original_row_equilibration = finite_infinity;
  double sum_original_log_row_equilibration = 0;
  double max_original_row_equilibration = 0;
  double min_col_equilibration = finite_infinity;
  double sum_log_col_equilibration = 0;
  double max_col_equilibration = 0;
  double min_row_equilibration = finite_infinity;
  double sum_log_row_equilibration = 0;
  double max_row_equilibration = 0;
  vector<double> original_row_min_value(numRow, finite_infinity);
  vector<double> original_row_max_value(numRow, 1 / finite_infinity);
  row_min_value.assign(numRow, finite_infinity);
  row_max_value.assign(numRow, 1 / finite_infinity);
  for (HighsInt iCol = 0; iCol < numCol; iCol++) {
    double original_col_min_value = finite_infinity;
    double original_col_max_value = 1 / finite_infinity;
    double col_min_value = finite_infinity;
    double col_max_value = 1 / finite_infinity;
    for (HighsInt k = Astart[iCol]; k < Astart[iCol + 1]; k++) {
      HighsInt iRow = Aindex[k];
      const double original_value = fabs(Avalue[k]);
      original_col_min_value = min(original_value, original_col_min_value);
      original_col_max_value = max(original_value, original_col_max_value);
      original_row_min_value[iRow] =
          min(original_row_min_value[iRow], original_value);
      original_row_max_value[iRow] =
          max(original_row_max_value[iRow], original_value);
      Avalue[k] *= (colScale[iCol] * rowScale[iRow]);
      const double value = fabs(Avalue[k]);
      col_min_value = min(value, col_min_value);
      col_max_value = max(value, col_max_value);
      row_min_value[iRow] = min(row_min_value[iRow], value);
      row_max_value[iRow] = max(row_max_value[iRow], value);
    }
    matrix_min_value = min(matrix_min_value, col_min_value);
    matrix_max_value = max(matrix_max_value, col_max_value);

    const double original_col_equilibration =
        1 / sqrt(original_col_min_value * original_col_max_value);
    min_original_col_equilibration =
        min(original_col_equilibration, min_original_col_equilibration);
    sum_original_log_col_equilibration += log(original_col_equilibration);
    max_original_col_equilibration =
        max(original_col_equilibration, max_original_col_equilibration);
    const double col_equilibration = 1 / sqrt(col_min_value * col_max_value);
    min_col_equilibration = min(col_equilibration, min_col_equilibration);
    sum_log_col_equilibration += log(col_equilibration);
    max_col_equilibration = max(col_equilibration, max_col_equilibration);
  }

  for (HighsInt iRow = 0; iRow < numRow; iRow++) {
    const double original_row_equilibration =
        1 / sqrt(original_row_min_value[iRow] * original_row_max_value[iRow]);
    min_original_row_equilibration =
        min(original_row_equilibration, min_original_row_equilibration);
    sum_original_log_row_equilibration += log(original_row_equilibration);
    max_original_row_equilibration =
        max(original_row_equilibration, max_original_row_equilibration);
    const double row_equilibration =
        1 / sqrt(row_min_value[iRow] * row_max_value[iRow]);
    min_row_equilibration = min(row_equilibration, min_row_equilibration);
    sum_log_row_equilibration += log(row_equilibration);
    max_row_equilibration = max(row_equilibration, max_row_equilibration);
  }
  const double geomean_original_col_equilibration =
      exp(sum_original_log_col_equilibration / numCol);
  const double geomean_original_row_equilibration =
      exp(sum_original_log_row_equilibration / numRow);
  const double geomean_col_equilibration =
      exp(sum_log_col_equilibration / numCol);
  const double geomean_row_equilibration =
      exp(sum_log_row_equilibration / numRow);
  if (options.highs_analysis_level) {
    highsLogDev(
        options.log_options, HighsLogType::kInfo,
        "Scaling: Original equilibration: min/mean/max %11.4g/%11.4g/%11.4g "
        "(cols); min/mean/max %11.4g/%11.4g/%11.4g (rows)\n",
        min_original_col_equilibration, geomean_original_col_equilibration,
        max_original_col_equilibration, min_original_row_equilibration,
        geomean_original_row_equilibration, max_original_row_equilibration);
    highsLogDev(
        options.log_options, HighsLogType::kInfo,
        "Scaling: Final    equilibration: min/mean/max %11.4g/%11.4g/%11.4g "
        "(cols); min/mean/max %11.4g/%11.4g/%11.4g (rows)\n",
        min_col_equilibration, geomean_col_equilibration, max_col_equilibration,
        min_row_equilibration, geomean_row_equilibration,
        max_row_equilibration);
  }

  // Compute the mean equilibration improvement
  const double geomean_original_col =
      max(geomean_original_col_equilibration,
          1 / geomean_original_col_equilibration);
  const double geomean_original_row =
      max(geomean_original_row_equilibration,
          1 / geomean_original_row_equilibration);
  const double geomean_col =
      max(geomean_col_equilibration, 1 / geomean_col_equilibration);
  const double geomean_row =
      max(geomean_row_equilibration, 1 / geomean_row_equilibration);
  const double mean_equilibration_improvement =
      sqrt((geomean_original_col * geomean_original_row) /
           (geomean_col * geomean_row));
  // Compute the extreme equilibration improvement
  const double original_col_ratio =
      max_original_col_equilibration / min_original_col_equilibration;
  const double original_row_ratio =
      max_original_row_equilibration / min_original_row_equilibration;
  const double col_ratio = max_col_equilibration / min_col_equilibration;
  const double row_ratio = max_row_equilibration / min_row_equilibration;
  const double extreme_equilibration_improvement =
      (original_col_ratio + original_row_ratio) / (col_ratio + row_ratio);
  // Compute the max/min matrix value improvement
  const double matrix_value_ratio = matrix_max_value / matrix_min_value;
  const double original_matrix_value_ratio =
      original_matrix_max_value / original_matrix_min_value;
  const double matrix_value_ratio_improvement =
      original_matrix_value_ratio / matrix_value_ratio;
  if (options.highs_analysis_level) {
    highsLogDev(
        options.log_options, HighsLogType::kInfo,
        "Scaling: Extreme equilibration improvement =      ( %11.4g + "
        "%11.4g) / ( %11.4g + %11.4g)  =      %11.4g / %11.4g  = %11.4g\n",
        original_col_ratio, original_row_ratio, col_ratio, row_ratio,
        (original_col_ratio + original_row_ratio), (col_ratio + row_ratio),
        extreme_equilibration_improvement);
    highsLogDev(
        options.log_options, HighsLogType::kInfo,
        "Scaling: Mean    equilibration improvement = sqrt(( %11.4g * "
        "%11.4g) / ( %11.4g * %11.4g)) = sqrt(%11.4g / %11.4g) = %11.4g\n",
        geomean_original_col, geomean_original_row, geomean_col, geomean_row,
        (geomean_original_col * geomean_original_row),
        (geomean_col * geomean_row), mean_equilibration_improvement);
    highsLogDev(
        options.log_options, HighsLogType::kInfo,
        "Scaling: Yields [min, max, ratio] matrix values of [%0.4g, %0.4g, "
        "%0.4g]; Originally [%0.4g, %0.4g, %0.4g]: Improvement of %0.4g\n",
        matrix_min_value, matrix_max_value, matrix_value_ratio,
        original_matrix_min_value, original_matrix_max_value,
        original_matrix_value_ratio, matrix_value_ratio_improvement);
    highsLogDev(options.log_options, HighsLogType::kInfo,
                "Scaling: Improves    mean equilibration by a factor %0.4g\n",
                mean_equilibration_improvement);
    highsLogDev(options.log_options, HighsLogType::kInfo,
                "Scaling: Improves extreme equilibration by a factor %0.4g\n",
                extreme_equilibration_improvement);
    highsLogDev(options.log_options, HighsLogType::kInfo,
                "Scaling: Improves max/min matrix values by a factor %0.4g\n",
                matrix_value_ratio_improvement);
  }
  const bool possibly_abandon_scaling =
      simplex_scale_strategy != kSimplexScaleStrategyForcedEquilibration;
  const double improvement_factor = extreme_equilibration_improvement *
                                    mean_equilibration_improvement *
                                    matrix_value_ratio_improvement;

  const double improvement_factor_required = 1.0;
  const bool poor_improvement =
      improvement_factor < improvement_factor_required;

  // Possibly abandon scaling if it's not improved equlibration significantly
  if (possibly_abandon_scaling && poor_improvement) {
    // Unscale the matrix
    for (HighsInt iCol = 0; iCol < numCol; iCol++) {
      for (HighsInt k = Astart[iCol]; k < Astart[iCol + 1]; k++) {
        HighsInt iRow = Aindex[k];
        Avalue[k] /= (colScale[iCol] * rowScale[iRow]);
      }
    }
    if (options.highs_analysis_level)
      highsLogDev(options.log_options, HighsLogType::kInfo,
                  "Scaling: Improvement factor %0.4g < %0.4g required, so no "
                  "scaling applied\n",
                  improvement_factor, improvement_factor_required);
    return false;
  } else {
    if (options.highs_analysis_level) {
      highsLogDev(options.log_options, HighsLogType::kInfo,
                  "Scaling: Improvement factor is %0.4g >= %0.4g so scale LP\n",
                  improvement_factor, improvement_factor_required);
      if (extreme_equilibration_improvement < 1.0) {
        highsLogDev(
            options.log_options, HighsLogType::kWarning,
            "Scaling: Applying scaling with extreme improvement of %0.4g\n",
            extreme_equilibration_improvement);
      }
      if (mean_equilibration_improvement < 1.0) {
        highsLogDev(
            options.log_options, HighsLogType::kWarning,
            "Scaling: Applying scaling with mean improvement of %0.4g\n",
            mean_equilibration_improvement);
      }
      if (matrix_value_ratio_improvement < 1.0) {
        highsLogDev(options.log_options, HighsLogType::kWarning,
                    "Scaling: Applying scaling with matrix value ratio "
                    "improvement of %0.4g\n",
                    matrix_value_ratio_improvement);
      }
      if (improvement_factor < 10 * improvement_factor_required) {
        highsLogDev(options.log_options, HighsLogType::kWarning,
                    "Scaling: Applying scaling with improvement factor %0.4g "
                    "< 10*(%0.4g) improvement\n",
                    improvement_factor, improvement_factor_required);
      }
    }
  }
  return true;
}

bool maxValueScaleMatrix(const HighsOptions& options, HighsLp& lp,
                         const HighsInt use_scale_strategy) {
  HighsInt numCol = lp.num_col_;
  HighsInt numRow = lp.num_row_;
  HighsScale& scale = lp.scale_;
  vector<double>& colScale = scale.col;
  vector<double>& rowScale = scale.row;
  vector<HighsInt>& Astart = lp.a_matrix_.start_;
  vector<HighsInt>& Aindex = lp.a_matrix_.index_;
  vector<double>& Avalue = lp.a_matrix_.value_;

  HighsInt simplex_scale_strategy = use_scale_strategy;

  assert(options.simplex_scale_strategy == kSimplexScaleStrategyMaxValue015 ||
         options.simplex_scale_strategy == kSimplexScaleStrategyMaxValue0157);
  const double log2 = log(2.0);
  const double max_allow_scale = pow(2.0, options.allowed_matrix_scale_factor);
  const double min_allow_scale = 1 / max_allow_scale;

  const double min_allow_col_scale = min_allow_scale;
  const double max_allow_col_scale = max_allow_scale;
  const double min_allow_row_scale = min_allow_scale;
  const double max_allow_row_scale = max_allow_scale;

  double min_row_scale = kHighsInf;
  double max_row_scale = 0;
  double original_matrix_min_value = kHighsInf;
  double original_matrix_max_value = 0;
  // Determine the row scaling. Also determine the max/min row scaling
  // factors, and max/min original matrix values
  vector<double> row_max_value(numRow, 0);
  for (HighsInt iCol = 0; iCol < numCol; iCol++) {
    for (HighsInt k = Astart[iCol]; k < Astart[iCol + 1]; k++) {
      const HighsInt iRow = Aindex[k];
      const double value = fabs(Avalue[k]);
      row_max_value[iRow] = max(row_max_value[iRow], value);
      original_matrix_min_value = min(original_matrix_min_value, value);
      original_matrix_max_value = max(original_matrix_max_value, value);
    }
  }
  for (HighsInt iRow = 0; iRow < numRow; iRow++) {
    if (row_max_value[iRow]) {
      double row_scale_value = 1 / row_max_value[iRow];
      // Convert the row scale factor to the nearest power of two, and
      // ensure that it is not excessively large or small
      row_scale_value = pow(2.0, floor(log(row_scale_value) / log2 + 0.5));
      row_scale_value =
          min(max(min_allow_row_scale, row_scale_value), max_allow_row_scale);
      min_row_scale = min(row_scale_value, min_row_scale);
      max_row_scale = max(row_scale_value, max_row_scale);
      rowScale[iRow] = row_scale_value;
    }
  }
  // Determine the column scaling, whilst applying the row scaling
  // Also determine the max/min column scaling factors, and max/min
  // matrix values
  double min_col_scale = kHighsInf;
  double max_col_scale = 0;
  double matrix_min_value = kHighsInf;
  double matrix_max_value = 0;
  for (HighsInt iCol = 0; iCol < numCol; iCol++) {
    double col_max_value = 0;
    for (HighsInt k = Astart[iCol]; k < Astart[iCol + 1]; k++) {
      const HighsInt iRow = Aindex[k];
      Avalue[k] *= rowScale[iRow];
      const double value = fabs(Avalue[k]);
      col_max_value = max(col_max_value, value);
    }
    if (col_max_value) {
      double col_scale_value = 1 / col_max_value;
      // Convert the col scale factor to the nearest power of two, and
      // ensure that it is not excessively large or small
      col_scale_value = pow(2.0, floor(log(col_scale_value) / log2 + 0.5));
      col_scale_value =
          min(max(min_allow_col_scale, col_scale_value), max_allow_col_scale);
      min_col_scale = min(col_scale_value, min_col_scale);
      max_col_scale = max(col_scale_value, max_col_scale);
      colScale[iCol] = col_scale_value;
      for (HighsInt k = Astart[iCol]; k < Astart[iCol + 1]; k++) {
        Avalue[k] *= colScale[iCol];
        const double value = fabs(Avalue[k]);
        matrix_min_value = min(matrix_min_value, value);
        matrix_max_value = max(matrix_max_value, value);
      }
    }
  }
  const double matrix_value_ratio = matrix_max_value / matrix_min_value;
  const double original_matrix_value_ratio =
      original_matrix_max_value / original_matrix_min_value;
  const double matrix_value_ratio_improvement =
      original_matrix_value_ratio / matrix_value_ratio;

  const double improvement_factor = matrix_value_ratio_improvement;

  const double improvement_factor_required = 1.0;
  const bool poor_improvement =
      improvement_factor < improvement_factor_required;

  if (poor_improvement) {
    // Unscale the matrix
    for (HighsInt iCol = 0; iCol < numCol; iCol++) {
      for (HighsInt k = Astart[iCol]; k < Astart[iCol + 1]; k++) {
        HighsInt iRow = Aindex[k];
        Avalue[k] /= (colScale[iCol] * rowScale[iRow]);
      }
    }
    if (options.highs_analysis_level)
      highsLogDev(options.log_options, HighsLogType::kInfo,
                  "Scaling: Improvement factor %0.4g < %0.4g required, so no "
                  "scaling applied\n",
                  improvement_factor, improvement_factor_required);
    return false;
  } else {
    if (options.highs_analysis_level) {
      highsLogDev(options.log_options, HighsLogType::kInfo,
                  "Scaling: Factors are in [%0.4g, %0.4g] for columns and in "
                  "[%0.4g, %0.4g] for rows\n",
                  min_col_scale, max_col_scale, min_row_scale, max_row_scale);
      highsLogDev(
          options.log_options, HighsLogType::kInfo,
          "Scaling: Yields [min, max, ratio] matrix values of [%0.4g, %0.4g, "
          "%0.4g]; Originally [%0.4g, %0.4g, %0.4g]: Improvement of %0.4g\n",
          matrix_min_value, matrix_max_value, matrix_value_ratio,
          original_matrix_min_value, original_matrix_max_value,
          original_matrix_value_ratio, matrix_value_ratio_improvement);
    }
    return true;
  }
}

HighsStatus applyScalingToLpCol(HighsLp& lp, const HighsInt col,
                                const double colScale) {
  if (col < 0) return HighsStatus::kError;
  if (col >= lp.num_col_) return HighsStatus::kError;
  if (!colScale) return HighsStatus::kError;

  for (HighsInt el = lp.a_matrix_.start_[col];
       el < lp.a_matrix_.start_[col + 1]; el++)
    lp.a_matrix_.value_[el] *= colScale;
  lp.a_matrix_.scaleCol(col, colScale);
  lp.col_cost_[col] *= colScale;
  if (colScale > 0) {
    lp.col_lower_[col] /= colScale;
    lp.col_upper_[col] /= colScale;
  } else {
    const double new_upper = lp.col_lower_[col] / colScale;
    lp.col_lower_[col] = lp.col_upper_[col] / colScale;
    lp.col_upper_[col] = new_upper;
  }
  return HighsStatus::kOk;
}

HighsStatus applyScalingToLpRow(HighsLp& lp, const HighsInt row,
                                const double rowScale) {
  if (row < 0) return HighsStatus::kError;
  if (row >= lp.num_row_) return HighsStatus::kError;
  if (!rowScale) return HighsStatus::kError;

  for (HighsInt col = 0; col < lp.num_col_; col++) {
    for (HighsInt el = lp.a_matrix_.start_[col];
         el < lp.a_matrix_.start_[col + 1]; el++) {
      if (lp.a_matrix_.index_[el] == row) lp.a_matrix_.value_[el] *= rowScale;
    }
  }
  lp.a_matrix_.scaleRow(row, rowScale);
  if (rowScale > 0) {
    lp.row_lower_[row] /= rowScale;
    lp.row_upper_[row] /= rowScale;
  } else {
    const double new_upper = lp.row_lower_[row] / rowScale;
    lp.row_lower_[row] = lp.row_upper_[row] / rowScale;
    lp.row_upper_[row] = new_upper;
  }
  return HighsStatus::kOk;
}

void appendColsToLpVectors(HighsLp& lp, const HighsInt num_new_col,
                           const vector<double>& colCost,
                           const vector<double>& colLower,
                           const vector<double>& colUpper) {
  assert(num_new_col >= 0);
  if (num_new_col == 0) return;
  HighsInt new_num_col = lp.num_col_ + num_new_col;
  lp.col_cost_.resize(new_num_col);
  lp.col_lower_.resize(new_num_col);
  lp.col_upper_.resize(new_num_col);
  bool have_names = lp.col_names_.size();
  if (have_names) lp.col_names_.resize(new_num_col);
  for (HighsInt new_col = 0; new_col < num_new_col; new_col++) {
    HighsInt iCol = lp.num_col_ + new_col;
    lp.col_cost_[iCol] = colCost[new_col];
    lp.col_lower_[iCol] = colLower[new_col];
    lp.col_upper_[iCol] = colUpper[new_col];
    // Cannot guarantee to create unique names, so name is blank
    if (have_names) lp.col_names_[iCol] = "";
  }
}

void appendRowsToLpVectors(HighsLp& lp, const HighsInt num_new_row,
                           const vector<double>& rowLower,
                           const vector<double>& rowUpper) {
  assert(num_new_row >= 0);
  if (num_new_row == 0) return;
  HighsInt new_num_row = lp.num_row_ + num_new_row;
  lp.row_lower_.resize(new_num_row);
  lp.row_upper_.resize(new_num_row);
  bool have_names = lp.row_names_.size();
  if (have_names) lp.row_names_.resize(new_num_row);

  for (HighsInt new_row = 0; new_row < num_new_row; new_row++) {
    HighsInt iRow = lp.num_row_ + new_row;
    lp.row_lower_[iRow] = rowLower[new_row];
    lp.row_upper_[iRow] = rowUpper[new_row];
    // Cannot guarantee to create unique names, so name is blank
    if (have_names) lp.row_names_[iRow] = "";
  }
}

void deleteLpCols(HighsLp& lp, const HighsIndexCollection& index_collection) {
  HighsInt new_num_col;
  HighsStatus call_status;
  deleteColsFromLpVectors(lp, new_num_col, index_collection);
  lp.a_matrix_.deleteCols(index_collection);
  lp.num_col_ = new_num_col;
}

void deleteColsFromLpVectors(HighsLp& lp, HighsInt& new_num_col,
                             const HighsIndexCollection& index_collection) {
  assert(ok(index_collection));
  HighsInt from_k;
  HighsInt to_k;
  limits(index_collection, from_k, to_k);
  ;
  // Initialise new_num_col in case none is removed due to from_k > to_k
  new_num_col = lp.num_col_;
  if (from_k > to_k) return;

  HighsInt delete_from_col;
  HighsInt delete_to_col;
  HighsInt keep_from_col;
  HighsInt keep_to_col = -1;
  HighsInt current_set_entry = 0;

  HighsInt col_dim = lp.num_col_;
  new_num_col = 0;
  bool have_names = lp.col_names_.size();
  for (HighsInt k = from_k; k <= to_k; k++) {
    updateOutInIndex(index_collection, delete_from_col, delete_to_col,
                     keep_from_col, keep_to_col, current_set_entry);
    // Account for the initial columns being kept
    if (k == from_k) new_num_col = delete_from_col;
    if (delete_to_col >= col_dim - 1) break;
    assert(delete_to_col < col_dim);
    for (HighsInt col = keep_from_col; col <= keep_to_col; col++) {
      lp.col_cost_[new_num_col] = lp.col_cost_[col];
      lp.col_lower_[new_num_col] = lp.col_lower_[col];
      lp.col_upper_[new_num_col] = lp.col_upper_[col];
      if (have_names) lp.col_names_[new_num_col] = lp.col_names_[col];
      new_num_col++;
    }
    if (keep_to_col >= col_dim - 1) break;
  }
  lp.col_cost_.resize(new_num_col);
  lp.col_lower_.resize(new_num_col);
  lp.col_upper_.resize(new_num_col);
  if (have_names) lp.col_names_.resize(new_num_col);
}

void deleteLpRows(HighsLp& lp, const HighsIndexCollection& index_collection) {
  HighsStatus call_status;
  HighsInt new_num_row;
  deleteRowsFromLpVectors(lp, new_num_row, index_collection);
  lp.a_matrix_.deleteRows(index_collection);
  lp.num_row_ = new_num_row;
}

void deleteRowsFromLpVectors(HighsLp& lp, HighsInt& new_num_row,
                             const HighsIndexCollection& index_collection) {
  assert(ok(index_collection));
  HighsInt from_k;
  HighsInt to_k;
  limits(index_collection, from_k, to_k);
  // Initialise new_num_row in case none is removed due to from_k > to_k
  new_num_row = lp.num_row_;
  if (from_k > to_k) return;

  HighsInt delete_from_row;
  HighsInt delete_to_row;
  HighsInt keep_from_row;
  HighsInt keep_to_row = -1;
  HighsInt current_set_entry = 0;

  HighsInt row_dim = lp.num_row_;
  new_num_row = 0;
  bool have_names = (HighsInt)lp.row_names_.size() > 0;
  for (HighsInt k = from_k; k <= to_k; k++) {
    updateOutInIndex(index_collection, delete_from_row, delete_to_row,
                     keep_from_row, keep_to_row, current_set_entry);
    if (k == from_k) {
      // Account for the initial rows being kept
      new_num_row = delete_from_row;
    }
    if (delete_to_row >= row_dim - 1) break;
    assert(delete_to_row < row_dim);
    for (HighsInt row = keep_from_row; row <= keep_to_row; row++) {
      lp.row_lower_[new_num_row] = lp.row_lower_[row];
      lp.row_upper_[new_num_row] = lp.row_upper_[row];
      if (have_names) lp.row_names_[new_num_row] = lp.row_names_[row];
      new_num_row++;
    }
    if (keep_to_row >= row_dim - 1) break;
  }
  lp.row_lower_.resize(new_num_row);
  lp.row_upper_.resize(new_num_row);
  if (have_names) lp.row_names_.resize(new_num_row);
}

void deleteScale(vector<double>& scale,
                 const HighsIndexCollection& index_collection) {
  HighsStatus return_status = HighsStatus::kOk;
  assert(ok(index_collection));
  HighsInt from_k;
  HighsInt to_k;
  limits(index_collection, from_k, to_k);
  if (from_k > to_k) return;

  HighsInt delete_from_col;
  HighsInt delete_to_col;
  HighsInt keep_from_col;
  HighsInt keep_to_col = -1;
  HighsInt current_set_entry = 0;

  HighsInt col_dim = index_collection.dimension_;
  HighsInt new_num_col = 0;
  for (HighsInt k = from_k; k <= to_k; k++) {
    updateOutInIndex(index_collection, delete_from_col, delete_to_col,
                     keep_from_col, keep_to_col, current_set_entry);
    // Account for the initial columns being kept
    if (k == from_k) new_num_col = delete_from_col;
    if (delete_to_col >= col_dim - 1) break;
    assert(delete_to_col < col_dim);
    for (HighsInt col = keep_from_col; col <= keep_to_col; col++) {
      assert((HighsInt)scale.size() > new_num_col);
      scale[new_num_col] = scale[col];
      new_num_col++;
    }
    if (keep_to_col >= col_dim - 1) break;
  }
}

void changeLpMatrixCoefficient(HighsLp& lp, const HighsInt row,
                               const HighsInt col, const double new_value,
                               const bool zero_new_value) {
  assert(0 <= row && row < lp.num_row_);
  assert(0 <= col && col < lp.num_col_);

  // Determine whether the coefficient corresponds to an existing
  // nonzero
  HighsInt change_el = -1;
  for (HighsInt el = lp.a_matrix_.start_[col];
       el < lp.a_matrix_.start_[col + 1]; el++) {
    if (lp.a_matrix_.index_[el] == row) {
      change_el = el;
      break;
    }
  }
  if (change_el < 0) {
    // Coefficient doesn't correspond to an existing nonzero
    //
    // If coefficient is small, then just ignore it
    if (zero_new_value) return;
    // New nonzero goes at the end of column "col", so have to shift
    // all index and value entries forward by 1 to accommodate it
    change_el = lp.a_matrix_.start_[col + 1];
    HighsInt new_num_nz = lp.a_matrix_.start_[lp.num_col_] + 1;
    lp.a_matrix_.index_.resize(new_num_nz);
    lp.a_matrix_.value_.resize(new_num_nz);
    for (HighsInt i = col + 1; i <= lp.num_col_; i++) lp.a_matrix_.start_[i]++;
    for (HighsInt el = new_num_nz - 1; el > change_el; el--) {
      lp.a_matrix_.index_[el] = lp.a_matrix_.index_[el - 1];
      lp.a_matrix_.value_[el] = lp.a_matrix_.value_[el - 1];
    }
  } else if (zero_new_value) {
    // Coefficient zeroes an existing nonzero, so shift all index and
    // value entries backward by 1 to eliminate it
    HighsInt new_num_nz = lp.a_matrix_.start_[lp.num_col_] - 1;
    for (HighsInt i = col + 1; i <= lp.num_col_; i++) lp.a_matrix_.start_[i]--;
    for (HighsInt el = change_el; el < new_num_nz; el++) {
      lp.a_matrix_.index_[el] = lp.a_matrix_.index_[el + 1];
      lp.a_matrix_.value_[el] = lp.a_matrix_.value_[el + 1];
    }
    return;
  }
  lp.a_matrix_.index_[change_el] = row;
  lp.a_matrix_.value_[change_el] = new_value;
}

void changeLpIntegrality(HighsLp& lp,
                         const HighsIndexCollection& index_collection,
                         const vector<HighsVarType>& new_integrality) {
  assert(ok(index_collection));
  HighsInt from_k;
  HighsInt to_k;
  limits(index_collection, from_k, to_k);
  if (from_k > to_k) return;

  const bool& interval = index_collection.is_interval_;
  const bool& mask = index_collection.is_mask_;
  const vector<HighsInt>& col_set = index_collection.set_;
  const vector<HighsInt>& col_mask = index_collection.mask_;

  // Change the integrality to the user-supplied integrality, according to the
  // technique
  HighsInt lp_col;
  HighsInt usr_col = -1;
  // May be adding integrality to a pure LP for which lp.integrality_
  // is of size 0.
  lp.integrality_.resize(lp.num_col_);
  for (HighsInt k = from_k; k < to_k + 1; k++) {
    if (interval || mask) {
      lp_col = k;
    } else {
      lp_col = col_set[k];
    }
    HighsInt col = lp_col;
    if (interval) {
      usr_col++;
    } else {
      usr_col = k;
    }
    if (mask && !col_mask[col]) continue;
    lp.integrality_[col] = new_integrality[usr_col];
  }
}

void changeLpCosts(HighsLp& lp, const HighsIndexCollection& index_collection,
                   const vector<double>& new_col_cost) {
  assert(ok(index_collection));
  HighsInt from_k;
  HighsInt to_k;
  limits(index_collection, from_k, to_k);
  if (from_k > to_k) return;

  const bool& interval = index_collection.is_interval_;
  const bool& mask = index_collection.is_mask_;
  const vector<HighsInt>& col_set = index_collection.set_;
  const vector<HighsInt>& col_mask = index_collection.mask_;

  // Change the costs to the user-supplied costs, according to the technique
  HighsInt lp_col;
  HighsInt usr_col = -1;
  for (HighsInt k = from_k; k < to_k + 1; k++) {
    if (interval || mask) {
      lp_col = k;
    } else {
      lp_col = col_set[k];
    }
    HighsInt col = lp_col;
    if (interval) {
      usr_col++;
    } else {
      usr_col = k;
    }
    if (mask && !col_mask[col]) continue;
    lp.col_cost_[col] = new_col_cost[usr_col];
  }
}

void changeLpColBounds(HighsLp& lp,
                       const HighsIndexCollection& index_collection,
                       const vector<double>& new_col_lower,
                       const vector<double>& new_col_upper) {
  changeBounds(lp.col_lower_, lp.col_upper_, index_collection, new_col_lower,
               new_col_upper);
}

void changeLpRowBounds(HighsLp& lp,
                       const HighsIndexCollection& index_collection,
                       const vector<double>& new_row_lower,
                       const vector<double>& new_row_upper) {
  changeBounds(lp.row_lower_, lp.row_upper_, index_collection, new_row_lower,
               new_row_upper);
}

void changeBounds(vector<double>& lower, vector<double>& upper,
                  const HighsIndexCollection& index_collection,
                  const vector<double>& new_lower,
                  const vector<double>& new_upper) {
  assert(ok(index_collection));
  HighsInt from_k;
  HighsInt to_k;
  limits(index_collection, from_k, to_k);
  if (from_k > to_k) return;

  const bool& interval = index_collection.is_interval_;
  const bool& mask = index_collection.is_mask_;
  const vector<HighsInt>& ix_set = index_collection.set_;
  const vector<HighsInt>& ix_mask = index_collection.mask_;

  // Change the bounds to the user-supplied bounds, according to the technique
  HighsInt lp_ix;
  HighsInt usr_ix = -1;
  for (HighsInt k = from_k; k < to_k + 1; k++) {
    if (interval || mask) {
      lp_ix = k;
    } else {
      lp_ix = ix_set[k];
    }
    HighsInt ix = lp_ix;
    if (interval) {
      usr_ix++;
    } else {
      usr_ix = k;
    }
    if (mask && !ix_mask[ix]) continue;
    lower[ix] = new_lower[usr_ix];
    upper[ix] = new_upper[usr_ix];
  }
}

HighsInt getNumInt(const HighsLp& lp) {
  HighsInt num_int = 0;
  if (lp.integrality_.size()) {
    for (HighsInt iCol = 0; iCol < lp.num_col_; iCol++)
      if (lp.integrality_[iCol] == HighsVarType::kInteger) num_int++;
  }
  return num_int;
}

void getLpCosts(const HighsLp& lp, const HighsInt from_col,
                const HighsInt to_col, double* XcolCost) {
  assert(0 <= from_col && to_col < lp.num_col_);
  if (from_col > to_col) return;
  for (HighsInt col = from_col; col < to_col + 1; col++)
    XcolCost[col - from_col] = lp.col_cost_[col];
}

void getLpColBounds(const HighsLp& lp, const HighsInt from_col,
                    const HighsInt to_col, double* XcolLower,
                    double* XcolUpper) {
  assert(0 <= from_col && to_col < lp.num_col_);
  if (from_col > to_col) return;
  for (HighsInt col = from_col; col < to_col + 1; col++) {
    if (XcolLower != NULL) XcolLower[col - from_col] = lp.col_lower_[col];
    if (XcolUpper != NULL) XcolUpper[col - from_col] = lp.col_upper_[col];
  }
}

void getLpRowBounds(const HighsLp& lp, const HighsInt from_row,
                    const HighsInt to_row, double* XrowLower,
                    double* XrowUpper) {
  assert(0 <= to_row && from_row < lp.num_row_);
  if (from_row > to_row) return;
  for (HighsInt row = from_row; row < to_row + 1; row++) {
    if (XrowLower != NULL) XrowLower[row - from_row] = lp.row_lower_[row];
    if (XrowUpper != NULL) XrowUpper[row - from_row] = lp.row_upper_[row];
  }
}

// Get a single coefficient from the matrix
void getLpMatrixCoefficient(const HighsLp& lp, const HighsInt Xrow,
                            const HighsInt Xcol, double* val) {
  assert(0 <= Xrow && Xrow < lp.num_row_);
  assert(0 <= Xcol && Xcol < lp.num_col_);

  HighsInt get_el = -1;
  for (HighsInt el = lp.a_matrix_.start_[Xcol];
       el < lp.a_matrix_.start_[Xcol + 1]; el++) {
    if (lp.a_matrix_.index_[el] == Xrow) {
      get_el = el;
      break;
    }
  }
  if (get_el < 0) {
    *val = 0;
  } else {
    *val = lp.a_matrix_.value_[get_el];
  }
}

// Methods for reporting an LP, including its row and column data and matrix
//
// Report the whole LP
void reportLp(const HighsLogOptions& log_options, const HighsLp& lp,
              const HighsLogType report_level) {
  reportLpBrief(log_options, lp);
  if ((HighsInt)report_level >= (HighsInt)HighsLogType::kDetailed) {
    reportLpColVectors(log_options, lp);
    reportLpRowVectors(log_options, lp);
    if ((HighsInt)report_level >= (HighsInt)HighsLogType::kVerbose)
      reportLpColMatrix(log_options, lp);
  }
}

// Report the LP briefly
void reportLpBrief(const HighsLogOptions& log_options, const HighsLp& lp) {
  reportLpDimensions(log_options, lp);
  reportLpObjSense(log_options, lp);
}

// Report the LP dimensions
void reportLpDimensions(const HighsLogOptions& log_options, const HighsLp& lp) {
  HighsInt lp_num_nz;
  if (lp.num_col_ == 0)
    lp_num_nz = 0;
  else
    lp_num_nz = lp.a_matrix_.start_[lp.num_col_];
  highsLogUser(log_options, HighsLogType::kInfo,
               "LP has %" HIGHSINT_FORMAT " columns, %" HIGHSINT_FORMAT " rows",
               lp.num_col_, lp.num_row_);
  HighsInt num_int = getNumInt(lp);
  if (num_int) {
    highsLogUser(log_options, HighsLogType::kInfo,
                 ", %" HIGHSINT_FORMAT " nonzeros and %" HIGHSINT_FORMAT
                 " integer columns\n",
                 lp_num_nz, num_int);
  } else {
    highsLogUser(log_options, HighsLogType::kInfo,
                 " and %" HIGHSINT_FORMAT " nonzeros\n", lp_num_nz, num_int);
  }
}

// Report the LP objective sense
void reportLpObjSense(const HighsLogOptions& log_options, const HighsLp& lp) {
  if (lp.sense_ == ObjSense::kMinimize)
    highsLogUser(log_options, HighsLogType::kInfo,
                 "Objective sense is minimize\n");
  else if (lp.sense_ == ObjSense::kMaximize)
    highsLogUser(log_options, HighsLogType::kInfo,
                 "Objective sense is maximize\n");
  else
    highsLogUser(log_options, HighsLogType::kInfo,
                 "Objective sense is ill-defined as %" HIGHSINT_FORMAT "\n",
                 lp.sense_);
}

static std::string getBoundType(const double lower, const double upper) {
  std::string type;
  if (highs_isInfinity(-lower)) {
    if (highs_isInfinity(upper)) {
      type = "FR";
    } else {
      type = "UB";
    }
  } else {
    if (highs_isInfinity(upper)) {
      type = "LB";
    } else {
      if (lower < upper) {
        type = "BX";
      } else {
        type = "FX";
      }
    }
  }
  return type;
}

// Report the vectors of LP column data
void reportLpColVectors(const HighsLogOptions& log_options, const HighsLp& lp) {
  if (lp.num_col_ <= 0) return;
  std::string type;
  HighsInt count;
  bool have_integer_columns = getNumInt(lp);
  bool have_col_names = lp.col_names_.size();

  highsLogUser(log_options, HighsLogType::kInfo,
               "  Column        Lower        Upper         Cost       "
               "Type        Count");
  if (have_integer_columns)
    highsLogUser(log_options, HighsLogType::kInfo, "  Discrete");
  if (have_col_names) highsLogUser(log_options, HighsLogType::kInfo, "  Name");
  highsLogUser(log_options, HighsLogType::kInfo, "\n");

  for (HighsInt iCol = 0; iCol < lp.num_col_; iCol++) {
    type = getBoundType(lp.col_lower_[iCol], lp.col_upper_[iCol]);
    count = lp.a_matrix_.start_[iCol + 1] - lp.a_matrix_.start_[iCol];
    highsLogUser(log_options, HighsLogType::kInfo,
                 "%8" HIGHSINT_FORMAT
                 " %12g %12g %12g         %2s %12" HIGHSINT_FORMAT "",
                 iCol, lp.col_lower_[iCol], lp.col_upper_[iCol],
                 lp.col_cost_[iCol], type.c_str(), count);
    if (have_integer_columns) {
      std::string integer_column = "";
      if (lp.integrality_[iCol] == HighsVarType::kInteger) {
        if (lp.col_lower_[iCol] == 0 && lp.col_upper_[iCol] == 1) {
          integer_column = "Binary";
        } else {
          integer_column = "Integer";
        }
      }
      highsLogUser(log_options, HighsLogType::kInfo, "  %-8s",
                   integer_column.c_str());
    }
    if (have_col_names)
      highsLogUser(log_options, HighsLogType::kInfo, "  %-s",
                   lp.col_names_[iCol].c_str());
    highsLogUser(log_options, HighsLogType::kInfo, "\n");
  }
}

// Report the vectors of LP row data
void reportLpRowVectors(const HighsLogOptions& log_options, const HighsLp& lp) {
  if (lp.num_row_ <= 0) return;
  std::string type;
  vector<HighsInt> count;
  bool have_row_names = lp.row_names_.size();

  count.resize(lp.num_row_, 0);
  if (lp.num_col_ > 0) {
    for (HighsInt el = 0; el < lp.a_matrix_.start_[lp.num_col_]; el++)
      count[lp.a_matrix_.index_[el]]++;
  }

  highsLogUser(log_options, HighsLogType::kInfo,
               "     Row        Lower        Upper       Type        Count");
  if (have_row_names) highsLogUser(log_options, HighsLogType::kInfo, "  Name");
  highsLogUser(log_options, HighsLogType::kInfo, "\n");

  for (HighsInt iRow = 0; iRow < lp.num_row_; iRow++) {
    type = getBoundType(lp.row_lower_[iRow], lp.row_upper_[iRow]);
    std::string name = "";
    highsLogUser(log_options, HighsLogType::kInfo,
                 "%8" HIGHSINT_FORMAT
                 " %12g %12g         %2s %12" HIGHSINT_FORMAT "",
                 iRow, lp.row_lower_[iRow], lp.row_upper_[iRow], type.c_str(),
                 count[iRow]);
    if (have_row_names)
      highsLogUser(log_options, HighsLogType::kInfo, "  %-s",
                   lp.row_names_[iRow].c_str());
    highsLogUser(log_options, HighsLogType::kInfo, "\n");
  }
}

// Report the LP column-wise matrix
void reportLpColMatrix(const HighsLogOptions& log_options, const HighsLp& lp) {
  if (lp.num_col_ <= 0) return;
  if (lp.num_row_) {
    // With postitive number of rows, can assume that there are index and value
    // vectors to pass
    reportMatrix(log_options, "Column", lp.num_col_,
                 lp.a_matrix_.start_[lp.num_col_], &lp.a_matrix_.start_[0],
                 &lp.a_matrix_.index_[0], &lp.a_matrix_.value_[0]);
  } else {
    // With no rows, can's assume that there are index and value vectors to pass
    reportMatrix(log_options, "Column", lp.num_col_,
                 lp.a_matrix_.start_[lp.num_col_], &lp.a_matrix_.start_[0],
                 NULL, NULL);
  }
}

void reportMatrix(const HighsLogOptions& log_options, const std::string message,
                  const HighsInt num_col, const HighsInt num_nz,
                  const HighsInt* start, const HighsInt* index,
                  const double* value) {
  if (num_col <= 0) return;
  highsLogUser(log_options, HighsLogType::kInfo,
               "%-7s Index              Value\n", message.c_str());
  for (HighsInt col = 0; col < num_col; col++) {
    highsLogUser(log_options, HighsLogType::kInfo,
                 "    %8" HIGHSINT_FORMAT " Start   %10" HIGHSINT_FORMAT "\n",
                 col, start[col]);
    HighsInt to_el = (col < num_col - 1 ? start[col + 1] : num_nz);
    for (HighsInt el = start[col]; el < to_el; el++)
      highsLogUser(log_options, HighsLogType::kInfo,
                   "          %8" HIGHSINT_FORMAT " %12g\n", index[el],
                   value[el]);
  }
  highsLogUser(log_options, HighsLogType::kInfo,
               "             Start   %10" HIGHSINT_FORMAT "\n", num_nz);
}

void analyseLp(const HighsLogOptions& log_options, const HighsLp& lp) {
  vector<double> min_colBound;
  vector<double> min_rowBound;
  vector<double> colRange;
  vector<double> rowRange;
  min_colBound.resize(lp.num_col_);
  min_rowBound.resize(lp.num_row_);
  colRange.resize(lp.num_col_);
  rowRange.resize(lp.num_row_);
  for (HighsInt col = 0; col < lp.num_col_; col++)
    min_colBound[col] = min(fabs(lp.col_lower_[col]), fabs(lp.col_upper_[col]));
  for (HighsInt row = 0; row < lp.num_row_; row++)
    min_rowBound[row] = min(fabs(lp.row_lower_[row]), fabs(lp.row_upper_[row]));
  for (HighsInt col = 0; col < lp.num_col_; col++)
    colRange[col] = lp.col_upper_[col] - lp.col_lower_[col];
  for (HighsInt row = 0; row < lp.num_row_; row++)
    rowRange[row] = lp.row_upper_[row] - lp.row_lower_[row];

  std::string message;
  if (lp.is_scaled_) {
    message = "Scaled";
  } else {
    message = "Unscaled";
  }
  highsLogDev(log_options, HighsLogType::kInfo, "\n%s model data: Analysis\n",
              message.c_str());
  if (lp.is_scaled_) {
    const HighsScale& scale = lp.scale_;
    analyseVectorValues(&log_options, "Column scaling factors", lp.num_col_,
                        scale.col, true, lp.model_name_);
    analyseVectorValues(&log_options, "Row    scaling factors", lp.num_row_,
                        scale.row, true, lp.model_name_);
  }
  analyseVectorValues(&log_options, "Column costs", lp.num_col_, lp.col_cost_,
                      true, lp.model_name_);
  analyseVectorValues(&log_options, "Column lower bounds", lp.num_col_,
                      lp.col_lower_, true, lp.model_name_);
  analyseVectorValues(&log_options, "Column upper bounds", lp.num_col_,
                      lp.col_upper_, true, lp.model_name_);
  analyseVectorValues(&log_options, "Column min abs bound", lp.num_col_,
                      min_colBound, true, lp.model_name_);
  analyseVectorValues(&log_options, "Column range", lp.num_col_, colRange, true,
                      lp.model_name_);
  analyseVectorValues(&log_options, "Row lower bounds", lp.num_row_,
                      lp.row_lower_, true, lp.model_name_);
  analyseVectorValues(&log_options, "Row upper bounds", lp.num_row_,
                      lp.row_upper_, true, lp.model_name_);
  analyseVectorValues(&log_options, "Row min abs bound", lp.num_row_,
                      min_rowBound, true, lp.model_name_);
  analyseVectorValues(&log_options, "Row range", lp.num_row_, rowRange, true,
                      lp.model_name_);
  analyseVectorValues(&log_options, "Matrix sparsity",
                      lp.a_matrix_.start_[lp.num_col_], lp.a_matrix_.value_,
                      true, lp.model_name_);
  analyseMatrixSparsity(log_options, "Constraint matrix", lp.num_col_,
                        lp.num_row_, lp.a_matrix_.start_, lp.a_matrix_.index_);
  analyseModelBounds(log_options, "Column", lp.num_col_, lp.col_lower_,
                     lp.col_upper_);
  analyseModelBounds(log_options, "Row", lp.num_row_, lp.row_lower_,
                     lp.row_upper_);
}

HighsStatus readSolutionFile(const std::string filename,
                             const HighsOptions& options, const HighsLp& lp,
                             HighsBasis& basis, HighsSolution& solution,
                             const HighsInt style) {
  const HighsLogOptions& log_options = options.log_options;
  if (style != kSolutionStyleRaw) {
    highsLogUser(log_options, HighsLogType::kError,
                 "readSolutionFile: Cannot read file of style %d\n",
                 (int)style);
    return HighsStatus::kError;
  }
  const HighsInt kMaxLineLength = 80;
  std::ifstream in_file(filename);
  if (in_file.fail()) {
    highsLogUser(log_options, HighsLogType::kError,
                 "readSolutionFile: Cannot open readable file \"%s\"\n",
                 filename.c_str());
    return HighsStatus::kError;
  }
  std::string keyword;
  std::string name;
  HighsInt num_col;
  HighsInt num_row;
  const HighsInt lp_num_col = lp.num_col_;
  const HighsInt lp_num_row = lp.num_row_;
  // Define idetifiers for reading in
  HighsSolution read_solution = solution;
  HighsBasis read_basis = basis;
  std::string section_name;
  HighsInt status;
  in_file.ignore(kMaxLineLength, '\n');  // Model status
  in_file.ignore(kMaxLineLength, '\n');  // Optimal
  in_file.ignore(kMaxLineLength, '\n');  //
  in_file.ignore(kMaxLineLength, '\n');  // # Primal solution values
  in_file >> keyword;
  if (keyword != "None") {
    in_file.ignore(kMaxLineLength, '\n');  // Status
    in_file.ignore(kMaxLineLength, '\n');  // Objective
    // Read in the column values
    in_file >> keyword >> keyword >> num_col;
    assert(keyword == "Columns");
    if (num_col != lp_num_col) {
      highsLogUser(log_options, HighsLogType::kError,
                   "readSolutionFile: Solution file is for %" HIGHSINT_FORMAT
                   " columns, not %" HIGHSINT_FORMAT "\n",
                   num_col, lp_num_col);
      return HighsStatus::kError;
    }
    for (HighsInt iCol = 0; iCol < num_col; iCol++)
      in_file >> name >> read_solution.col_value[iCol];
    // Read in the row values
    in_file >> keyword >> keyword >> num_row;
    assert(keyword == "Rows");
    if (num_row != lp_num_row) {
      highsLogUser(log_options, HighsLogType::kError,
                   "readSolutionFile: Solution file is for %" HIGHSINT_FORMAT
                   " rows, not %" HIGHSINT_FORMAT "\n",
                   num_row, lp_num_row);
      return HighsStatus::kError;
    }
    for (HighsInt iRow = 0; iRow < num_row; iRow++)
      in_file >> name >> read_solution.row_value[iRow];
  }
  in_file.ignore(kMaxLineLength, '\n');
  in_file.ignore(kMaxLineLength, '\n');  //
  in_file.ignore(kMaxLineLength, '\n');  // # Dual solution values
  in_file >> keyword;
  if (keyword != "None") {
    in_file.ignore(kMaxLineLength, '\n');  // Status
    in_file >> keyword >> keyword >> num_col;
    assert(keyword == "Columns");
    for (HighsInt iCol = 0; iCol < num_col; iCol++)
      in_file >> name >> read_solution.col_dual[iCol];
    in_file >> keyword >> keyword >> num_row;
    assert(keyword == "Rows");
    for (HighsInt iRow = 0; iRow < num_row; iRow++)
      in_file >> name >> read_solution.row_dual[iRow];
  }
  in_file.ignore(kMaxLineLength, '\n');  //
  in_file.ignore(kMaxLineLength, '\n');  //
  in_file.ignore(kMaxLineLength, '\n');  // # Basis
  if (readBasisStream(log_options, read_basis, in_file) == HighsStatus::kError)
    return HighsStatus::kError;
  solution = read_solution;
  basis = read_basis;
  return HighsStatus::kOk;
}

void checkLpSolutionFeasibility(const HighsOptions& options, const HighsLp& lp,
                                const HighsSolution& solution) {
  HighsInt num_col_infeasibilities = 0;
  double max_col_infeasibility = 0;
  double sum_col_infeasibilities = 0;
  HighsInt num_integer_infeasibilities = 0;
  double max_integer_infeasibility = 0;
  double sum_integer_infeasibilities = 0;
  HighsInt num_row_infeasibilities = 0;
  double max_row_infeasibility = 0;
  double sum_row_infeasibilities = 0;
  HighsInt num_row_residuals = 0;
  double max_row_residual = 0;
  double sum_row_residuals = 0;
  const double kRowResidualTolerance = 1e-12;
  const vector<HighsInt>& start = lp.a_matrix_.start_;
  const vector<HighsInt>& index = lp.a_matrix_.index_;
  const vector<double>& value = lp.a_matrix_.value_;
  vector<double> row_activity;
  row_activity.assign(lp.num_row_, 0);
  const bool have_integrality = lp.integrality_.size();
  for (HighsInt iCol = 0; iCol < lp.num_col_; iCol++) {
    const double primal = solution.col_value[iCol];
    const double lower = lp.col_lower_[iCol];
    const double upper = lp.col_upper_[iCol];
    const HighsVarType type =
        have_integrality ? lp.integrality_[iCol] : HighsVarType::kContinuous;
    // @primal_infeasibility calculation
    double col_infeasibility = 0;
    if (primal < lower - options.primal_feasibility_tolerance) {
      col_infeasibility = lower - primal;
    } else if (primal > upper + options.primal_feasibility_tolerance) {
      col_infeasibility = primal - upper;
    }
    double integer_infeasibility = 0;
    if (type == HighsVarType::kInteger || type == HighsVarType::kSemiInteger) {
      double nearest_integer = std::round(primal);
      double integer_infeasibility = std::fabs(primal - nearest_integer);
    }
    if (col_infeasibility > 0 && (type == HighsVarType::kSemiContinuous ||
                                  type == HighsVarType::kSemiInteger)) {
      // Semi-variables at zero will have positive col
      // infeasibility, so possibly zero this
      if (std::fabs(primal) <= options.mip_feasibility_tolerance)
        col_infeasibility = 0;
    }
    if (col_infeasibility > 0) {
      if (col_infeasibility > options.primal_feasibility_tolerance) {
        if (col_infeasibility > 2 * max_col_infeasibility)
          highsLogUser(options.log_options, HighsLogType::kWarning,
                       "Col %6d has         infeasiblilty of %11.4g from "
                       "[lower, value, upper] = [%15.8g; %15.8g; %15.8g]\n",
                       (int)iCol, col_infeasibility, lower, primal, upper);
        num_col_infeasibilities++;
      }
      max_col_infeasibility =
          std::max(col_infeasibility, max_col_infeasibility);
      sum_col_infeasibilities += col_infeasibility;
    }
    if (integer_infeasibility > 0) {
      if (integer_infeasibility > options.mip_feasibility_tolerance) {
        if (integer_infeasibility > 2 * max_integer_infeasibility)
          highsLogUser(options.log_options, HighsLogType::kWarning,
                       "Col %6d has integer infeasiblilty of %11.4g\n",
                       (int)iCol, integer_infeasibility);
        num_integer_infeasibilities++;
      }
      max_integer_infeasibility =
          std::max(integer_infeasibility, max_integer_infeasibility);
      sum_integer_infeasibilities += integer_infeasibility;
    }
    for (HighsInt iEl = start[iCol]; iEl < start[iCol + 1]; iEl++)
      row_activity[index[iEl]] += primal * value[iEl];
  }
  for (HighsInt iRow = 0; iRow < lp.num_row_; iRow++) {
    const double primal = solution.row_value[iRow];
    const double lower = lp.row_lower_[iRow];
    const double upper = lp.row_upper_[iRow];
    // @primal_infeasibility calculation
    double row_infeasibility = 0;
    if (primal < lower - options.primal_feasibility_tolerance) {
      row_infeasibility = lower - primal;
    } else if (primal > upper + options.primal_feasibility_tolerance) {
      row_infeasibility = primal - upper;
    }
    if (row_infeasibility > 0) {
      if (row_infeasibility > options.primal_feasibility_tolerance) {
        if (row_infeasibility > 2 * max_row_infeasibility)
          highsLogUser(options.log_options, HighsLogType::kWarning,
                       "Row %6d has         infeasiblilty of %11.4g from "
                       "[lower, value, upper] = [%15.8g; %15.8g; %15.8g]\n",
                       (int)iRow, row_infeasibility, lower, primal, upper);
        num_row_infeasibilities++;
      }
      max_row_infeasibility =
          std::max(row_infeasibility, max_row_infeasibility);
      sum_row_infeasibilities += row_infeasibility;
    }
    double row_residual = fabs(primal - row_activity[iRow]);
    if (row_residual > kRowResidualTolerance) {
      if (row_residual > 2 * max_row_residual) {
        highsLogUser(options.log_options, HighsLogType::kWarning,
                     "Row %6d has         residual      of %11.4g\n", (int)iRow,
                     row_residual);
      }
      num_row_residuals++;
    }
    max_row_residual = std::max(row_residual, max_row_residual);
    sum_row_residuals += row_residual;
  }
  highsLogUser(options.log_options, HighsLogType::kInfo,
               "Solution has               num          max          sum\n");
  highsLogUser(options.log_options, HighsLogType::kInfo,
               "Col     infeasibilities %6d  %11.4g  %11.4g\n",
               (int)num_col_infeasibilities, max_col_infeasibility,
               sum_col_infeasibilities);
  if (lp.isMip())
    highsLogUser(options.log_options, HighsLogType::kInfo,
                 "Integer infeasibilities %6d  %11.4g  %11.4g\n",
                 (int)num_integer_infeasibilities, max_integer_infeasibility,
                 sum_integer_infeasibilities);
  highsLogUser(options.log_options, HighsLogType::kInfo,
               "Row     infeasibilities %6d  %11.4g  %11.4g\n",
               (int)num_row_infeasibilities, max_row_infeasibility,
               sum_row_infeasibilities);
  highsLogUser(options.log_options, HighsLogType::kInfo,
               "Row     residuals       %6d  %11.4g  %11.4g\n",
               (int)num_row_residuals, max_row_residual, sum_row_residuals);
}

void writeBasisFile(FILE*& file, const HighsBasis& basis) {
  fprintf(file, "HiGHS v%d\n", (int)HIGHS_VERSION_MAJOR);
  if (basis.valid == false) {
    fprintf(file, "None\n");
    return;
  }
  fprintf(file, "Valid\n");
  fprintf(file, "# Columns %d\n", (int)basis.col_status.size());
  for (const auto& status : basis.col_status) fprintf(file, "%d ", (int)status);
  fprintf(file, "\n");
  fprintf(file, "# Rows %d\n", (int)basis.row_status.size());
  for (const auto& status : basis.row_status) fprintf(file, "%d ", (int)status);
  fprintf(file, "\n");
}

HighsStatus readBasisFile(const HighsLogOptions& log_options, HighsBasis& basis,
                          const std::string filename) {
  // Opens a basis file as an ifstream
  HighsStatus return_status = HighsStatus::kOk;
  std::ifstream in_file;
  in_file.open(filename.c_str(), std::ios::in);
  if (in_file.is_open()) {
    return_status = readBasisStream(log_options, basis, in_file);
    in_file.close();
  } else {
    highsLogUser(log_options, HighsLogType::kError,
                 "readBasisFile: Cannot open readable file \"%s\"\n",
                 filename.c_str());
    return_status = HighsStatus::kError;
  }
  return return_status;
}

HighsStatus readBasisStream(const HighsLogOptions& log_options,
                            HighsBasis& basis, std::ifstream& in_file) {
  // Reads a basis as an ifstream, returning an error if what's read is
  // inconsistent with the sizes of the HighsBasis passed in
  HighsStatus return_status = HighsStatus::kOk;
  std::string string_highs, string_version;
  in_file >> string_highs >> string_version;
  if (string_version == "v1") {
    std::string keyword;
    in_file >> keyword;
    if (keyword == "None") {
      basis.valid = false;
      return HighsStatus::kOk;
    }
    const HighsInt basis_num_col = (HighsInt)basis.col_status.size();
    const HighsInt basis_num_row = (HighsInt)basis.row_status.size();
    HighsInt int_status;
    assert(keyword == "Valid");
    HighsInt num_col, num_row;
    // Read in the columns section
    in_file >> keyword >> keyword;
    assert(keyword == "Columns");
    in_file >> num_col;
    if (num_col != basis_num_col) {
      highsLogUser(log_options, HighsLogType::kError,
                   "readBasisFile: Basis file is for %" HIGHSINT_FORMAT
                   " columns, not %" HIGHSINT_FORMAT "\n",
                   num_col, basis_num_col);
      return HighsStatus::kError;
    }
    for (HighsInt iCol = 0; iCol < num_col; iCol++) {
      in_file >> int_status;
      basis.col_status[iCol] = (HighsBasisStatus)int_status;
    }
    // Read in the rows section
    in_file >> keyword >> keyword;
    assert(keyword == "Rows");
    in_file >> num_row;
    if (num_row != basis_num_row) {
      highsLogUser(log_options, HighsLogType::kError,
                   "readBasisFile: Basis file is for %" HIGHSINT_FORMAT
                   " rows, not %" HIGHSINT_FORMAT "\n",
                   num_row, basis_num_row);
      return HighsStatus::kError;
    }
    for (HighsInt iRow = 0; iRow < num_row; iRow++) {
      in_file >> int_status;
      basis.row_status[iRow] = (HighsBasisStatus)int_status;
    }
  } else {
    highsLogUser(log_options, HighsLogType::kError,
                 "readBasisFile: Cannot read basis file for HiGHS %s\n",
                 string_version.c_str());
    return_status = HighsStatus::kError;
  }
  return return_status;
}

HighsStatus calculateColDuals(const HighsLp& lp, HighsSolution& solution) {
  //  assert(solution.row_dual.size() > 0);
  if (int(solution.row_dual.size()) < lp.num_row_) return HighsStatus::kError;

  solution.col_dual.assign(lp.num_col_, 0);

  for (HighsInt col = 0; col < lp.num_col_; col++) {
    for (HighsInt i = lp.a_matrix_.start_[col];
         i < lp.a_matrix_.start_[col + 1]; i++) {
      const HighsInt row = lp.a_matrix_.index_[i];
      assert(row >= 0);
      assert(row < lp.num_row_);
      // @FlipRowDual -= became +=
      solution.col_dual[col] += solution.row_dual[row] * lp.a_matrix_.value_[i];
    }
    solution.col_dual[col] += lp.col_cost_[col];
  }

  return HighsStatus::kOk;
}

HighsStatus calculateRowValues(const HighsLp& lp, HighsSolution& solution) {
  // assert(solution.col_value.size() > 0);
  if (int(solution.col_value.size()) < lp.num_col_) return HighsStatus::kError;

  solution.row_value.clear();
  solution.row_value.assign(lp.num_row_, 0);

  for (HighsInt col = 0; col < lp.num_col_; col++) {
    for (HighsInt i = lp.a_matrix_.start_[col];
         i < lp.a_matrix_.start_[col + 1]; i++) {
      const HighsInt row = lp.a_matrix_.index_[i];
      assert(row >= 0);
      assert(row < lp.num_row_);

      solution.row_value[row] +=
          solution.col_value[col] * lp.a_matrix_.value_[i];
    }
  }

  return HighsStatus::kOk;
}

HighsStatus calculateRowValuesQuad(const HighsLp& lp, HighsSolution& solution) {
  // assert(solution.col_value.size() > 0);
  if (int(solution.col_value.size()) != lp.num_col_) return HighsStatus::kError;

  std::vector<HighsCDouble> row_value;
  row_value.assign(lp.num_row_, HighsCDouble{0.0});

  solution.row_value.assign(lp.num_row_, 0);

  for (HighsInt col = 0; col < lp.num_col_; col++) {
    for (HighsInt i = lp.a_matrix_.start_[col];
         i < lp.a_matrix_.start_[col + 1]; i++) {
      const HighsInt row = lp.a_matrix_.index_[i];
      assert(row >= 0);
      assert(row < lp.num_row_);

      row_value[row] += solution.col_value[col] * lp.a_matrix_.value_[i];
    }
  }

  // assign quad values to double vector
  solution.row_value.resize(lp.num_row_);
  std::transform(row_value.begin(), row_value.end(), solution.row_value.begin(),
                 [](HighsCDouble x) { return double(x); });

  return HighsStatus::kOk;
}

bool isBoundInfeasible(const HighsLogOptions& log_options, const HighsLp& lp) {
  HighsInt num_bound_infeasible = 0;
  for (HighsInt iCol = 0; iCol < lp.num_col_; iCol++)
    if (lp.col_upper_[iCol] < lp.col_lower_[iCol]) num_bound_infeasible++;
  for (HighsInt iRow = 0; iRow < lp.num_row_; iRow++)
    if (lp.row_upper_[iRow] < lp.row_lower_[iRow]) num_bound_infeasible++;
  if (num_bound_infeasible > 0)
    highsLogUser(log_options, HighsLogType::kInfo,
                 "Model infeasible due to %" HIGHSINT_FORMAT
                 " inconsistent bound(s)\n",
                 num_bound_infeasible);
  return num_bound_infeasible > 0;
}

bool isColDataNull(const HighsLogOptions& log_options,
                   const double* usr_col_cost, const double* usr_col_lower,
                   const double* usr_col_upper) {
  bool null_data = false;
  null_data =
      doubleUserDataNotNull(log_options, usr_col_cost, "column costs") ||
      null_data;
  null_data = doubleUserDataNotNull(log_options, usr_col_lower,
                                    "column lower bounds") ||
              null_data;
  null_data = doubleUserDataNotNull(log_options, usr_col_upper,
                                    "column upper bounds") ||
              null_data;
  return null_data;
}

bool isRowDataNull(const HighsLogOptions& log_options,
                   const double* usr_row_lower, const double* usr_row_upper) {
  bool null_data = false;
  null_data =
      doubleUserDataNotNull(log_options, usr_row_lower, "row lower bounds") ||
      null_data;
  null_data =
      doubleUserDataNotNull(log_options, usr_row_upper, "row upper bounds") ||
      null_data;
  return null_data;
}

bool isMatrixDataNull(const HighsLogOptions& log_options,
                      const HighsInt* usr_matrix_start,
                      const HighsInt* usr_matrix_index,
                      const double* usr_matrix_value) {
  bool null_data = false;
  null_data =
      intUserDataNotNull(log_options, usr_matrix_start, "matrix starts") ||
      null_data;
  null_data =
      intUserDataNotNull(log_options, usr_matrix_index, "matrix indices") ||
      null_data;
  null_data =
      doubleUserDataNotNull(log_options, usr_matrix_value, "matrix values") ||
      null_data;
  return null_data;
}

void reportPresolveReductions(const HighsLogOptions& log_options,
                              const HighsLp& lp, const HighsLp& presolve_lp) {
  HighsInt num_col_from = lp.num_col_;
  HighsInt num_row_from = lp.num_row_;
  HighsInt num_els_from = lp.a_matrix_.start_[num_col_from];
  HighsInt num_col_to = presolve_lp.num_col_;
  HighsInt num_row_to = presolve_lp.num_row_;
  HighsInt num_els_to;
  if (num_col_to) {
    num_els_to = presolve_lp.a_matrix_.start_[num_col_to];
  } else {
    num_els_to = 0;
  }
  char elemsignchar = '-';
  HighsInt elemdelta = num_els_from - num_els_to;
  if (num_els_from < num_els_to) {
    elemdelta = -elemdelta;
    elemsignchar = '+';
  }
  highsLogUser(
      log_options, HighsLogType::kInfo,
      "Presolve : Reductions: rows %" HIGHSINT_FORMAT "(-%" HIGHSINT_FORMAT
      "); columns %" HIGHSINT_FORMAT "(-%" HIGHSINT_FORMAT
      "); "
      "elements %" HIGHSINT_FORMAT "(%c%" HIGHSINT_FORMAT ")\n",
      num_row_to, (num_row_from - num_row_to), num_col_to,
      (num_col_from - num_col_to), num_els_to, elemsignchar, elemdelta);
}

void reportPresolveReductions(const HighsLogOptions& log_options,
                              const HighsLp& lp, const bool presolve_to_empty) {
  HighsInt num_col_from = lp.num_col_;
  HighsInt num_row_from = lp.num_row_;
  HighsInt num_els_from = lp.a_matrix_.start_[num_col_from];
  HighsInt num_col_to;
  HighsInt num_row_to;
  HighsInt num_els_to;
  std::string message;
  if (presolve_to_empty) {
    num_col_to = 0;
    num_row_to = 0;
    num_els_to = 0;
    message = "- Reduced to empty";
  } else {
    num_col_to = num_col_from;
    num_row_to = num_row_from;
    num_els_to = num_els_from;
    message = "- Not reduced";
  }
  highsLogUser(log_options, HighsLogType::kInfo,
               "Presolve : Reductions: rows %" HIGHSINT_FORMAT
               "(-%" HIGHSINT_FORMAT "); columns %" HIGHSINT_FORMAT
               "(-%" HIGHSINT_FORMAT
               "); "
               "elements %" HIGHSINT_FORMAT "(-%" HIGHSINT_FORMAT ") %s\n",
               num_row_to, (num_row_from - num_row_to), num_col_to,
               (num_col_from - num_col_to), num_els_to,
               (num_els_from - num_els_to), message.c_str());
}

bool isLessInfeasibleDSECandidate(const HighsLogOptions& log_options,
                                  const HighsLp& lp) {
  HighsInt max_col_num_en = -1;
  const HighsInt max_allowed_col_num_en = 24;
  const HighsInt max_assess_col_num_en =
      std::max(HighsInt{9}, max_allowed_col_num_en);
  const HighsInt max_average_col_num_en = 6;
  vector<HighsInt> col_length_k;
  col_length_k.resize(1 + max_assess_col_num_en, 0);
  bool LiDSE_candidate = true;
  for (HighsInt col = 0; col < lp.num_col_; col++) {
    // Check limit on number of entries in the column has not been breached
    HighsInt col_num_en =
        lp.a_matrix_.start_[col + 1] - lp.a_matrix_.start_[col];
    max_col_num_en = std::max(col_num_en, max_col_num_en);
    if (col_num_en > max_assess_col_num_en) return false;
    col_length_k[col_num_en]++;
    for (HighsInt en = lp.a_matrix_.start_[col];
         en < lp.a_matrix_.start_[col + 1]; en++) {
      double value = lp.a_matrix_.value_[en];
      // All nonzeros must be +1 or -1
      if (fabs(value) != 1) return false;
    }
  }
  double average_col_num_en = lp.a_matrix_.start_[lp.num_col_];
  average_col_num_en = average_col_num_en / lp.num_col_;
  LiDSE_candidate =
      LiDSE_candidate && average_col_num_en <= max_average_col_num_en;
  highsLogDev(log_options, HighsLogType::kInfo,
              "LP %s has all |entries|=1; max column count = %" HIGHSINT_FORMAT
              " (limit %" HIGHSINT_FORMAT
              "); average "
              "column count = %0.2g (limit %" HIGHSINT_FORMAT
              "): LP is %s a candidate for LiDSE\n",
              lp.model_name_.c_str(), max_col_num_en, max_allowed_col_num_en,
              average_col_num_en, max_average_col_num_en,
              LiDSE_candidate ? "is" : "is not");
  return LiDSE_candidate;
}

HighsLp withoutSemiVariables(const HighsLp& lp_, HighsSolution& solution,
                             const double primal_feasibility_tolerance) {
  HighsLp lp = lp_;
  HighsInt num_col = lp.num_col_;
  HighsInt num_row = lp.num_row_;
  HighsInt num_semi_variables = 0;
  for (HighsInt iCol = 0; iCol < num_col; iCol++) {
    if (lp.integrality_[iCol] == HighsVarType::kSemiContinuous ||
        lp.integrality_[iCol] == HighsVarType::kSemiInteger)
      num_semi_variables++;
  }
  assert(num_semi_variables);
  // Insert spaces for index/value of new coefficients for
  // semi-variables
  vector<HighsInt>& start = lp.a_matrix_.start_;
  vector<HighsInt>& index = lp.a_matrix_.index_;
  vector<double>& value = lp.a_matrix_.value_;
  HighsInt num_nz = start[num_col];
  HighsInt new_num_nz = num_nz + 2 * num_semi_variables;
  HighsInt new_el = new_num_nz;
  index.resize(new_num_nz);
  value.resize(new_num_nz);
  for (HighsInt iCol = num_col - 1; iCol >= 0; iCol--) {
    HighsInt from_el = start[iCol + 1] - 1;
    start[iCol + 1] = new_el;
    if (lp.integrality_[iCol] == HighsVarType::kSemiContinuous ||
        lp.integrality_[iCol] == HighsVarType::kSemiInteger)
      new_el -= 2;
    for (HighsInt iEl = from_el; iEl >= start[iCol]; iEl--) {
      new_el--;
      index[new_el] = index[iEl];
      value[new_el] = value[iEl];
    }
  }
  assert(new_el == 0);
  // Insert the new coefficients for semi-variables
  HighsInt row_num = num_row;
  for (HighsInt iCol = 0; iCol < num_col; iCol++) {
    if (lp.integrality_[iCol] == HighsVarType::kSemiContinuous ||
        lp.integrality_[iCol] == HighsVarType::kSemiInteger) {
      HighsInt iEl = start[iCol + 1] - 2;
      index[iEl] = row_num++;
      value[iEl] = 1;
      iEl++;
      index[iEl] = row_num++;
      value[iEl] = 1;
    }
  }
  num_nz = start[num_col];
  new_num_nz = num_nz + 2 * num_semi_variables;
  row_num = num_row;
  HighsInt semi_col_num = 0;
  HighsInt semi_row_num = 0;
  // Insert the new variables and their coefficients
  std::stringstream ss;
  const bool has_col_names = lp.col_names_.size();
  const bool has_row_names = lp.row_names_.size();
  const bool has_solution = solution.value_valid;
  if (has_solution) assert(solution.col_value.size() == lp_.num_col_);
  for (HighsInt iCol = 0; iCol < num_col; iCol++) {
    if (lp.integrality_[iCol] == HighsVarType::kSemiContinuous ||
        lp.integrality_[iCol] == HighsVarType::kSemiInteger) {
      // Add a binary variable with zero cost
      lp.col_cost_.push_back(0);
      lp.col_lower_.push_back(0);
      lp.col_upper_.push_back(1);
      // Complete x - l*y >= 0
      lp.row_lower_.push_back(0);
      lp.row_upper_.push_back(kHighsInf);
      if (has_col_names) {
        // Create a column name
        ss.str(std::string());
        ss << "semi_binary_" << semi_col_num++;
        lp.col_names_.push_back(ss.str());
      }
      if (has_row_names) {
        // Create a row name
        ss.str(std::string());
        ss << "semi_lb_" << semi_row_num;
        lp.row_names_.push_back(ss.str());
      }
      index.push_back(row_num++);
      value.push_back(-lp.col_lower_[iCol]);
      // Accommodate any primal solution
      if (has_solution) {
        if (solution.col_value[iCol] <= primal_feasibility_tolerance) {
          // Currently at or below zero, so binary is 0
          solution.col_value[iCol] = 0;
          solution.col_value.push_back(0);
        } else {
          // Otherwise, solution is at least lower bound, and binary
          // is 1
          solution.col_value[iCol] =
              std::max(lp.col_lower_[iCol], solution.col_value[iCol]);
          solution.col_value.push_back(1);
        }
      }
      // Complete x - u*y <= 0
      lp.row_lower_.push_back(-kHighsInf);
      lp.row_upper_.push_back(0);
      if (has_row_names) {
        // Create a row name
        ss.str(std::string());
        ss << "semi_ub_" << semi_row_num++;
        lp.row_names_.push_back(ss.str());
      }
      index.push_back(row_num++);
      value.push_back(-lp.col_upper_[iCol]);
      // Add the next start
      start.push_back(index.size());
      lp.integrality_.push_back(HighsVarType::kInteger);
      if (lp.integrality_[iCol] == HighsVarType::kSemiContinuous) {
        lp.integrality_[iCol] = HighsVarType::kContinuous;
      } else if (lp.integrality_[iCol] == HighsVarType::kSemiInteger) {
        lp.integrality_[iCol] = HighsVarType::kInteger;
      }
      // Change the lower bound to on the semi-variable to
      // zero. Cannot do this earlier, as its original value is used
      // in constraint 0 <= x-l*y
      lp.col_lower_[iCol] = 0;
    }
  }
  num_col += num_semi_variables;
  lp.num_col_ += num_semi_variables;
  lp.num_row_ += 2 * num_semi_variables;
  assert(index.size() == new_num_nz);
  // Clear any modifications inherited from lp_
  lp.mods_.clear();
  return lp;
}

void removeRowsOfCountOne(const HighsLogOptions& log_options, HighsLp& lp) {
  HighsLp row_wise_lp = lp;
  vector<HighsInt>& a_start = lp.a_matrix_.start_;
  vector<HighsInt>& a_index = lp.a_matrix_.index_;
  vector<double>& a_value = lp.a_matrix_.value_;
  vector<HighsInt> a_count;
  vector<HighsInt> ar_count;
  vector<HighsInt> ar_start;
  vector<HighsInt> ar_index;
  vector<double> ar_value;
  const bool has_name = lp.row_names_.size() > 0;
  HighsInt num_nz = a_start[lp.num_col_];
  const HighsInt original_num_nz = num_nz;
  const HighsInt original_num_row = lp.num_row_;
  HighsInt num_row_count_1 = 0;
  ar_count.assign(lp.num_row_, 0);
  for (HighsInt iCol = 0; iCol < lp.num_col_; iCol++) {
    for (HighsInt iEl = a_start[iCol]; iEl < a_start[iCol + 1]; iEl++)
      ar_count[a_index[iEl]]++;
  }
  ar_start.push_back(0);
  for (HighsInt iRow = 0; iRow < lp.num_row_; iRow++) {
    ar_start.push_back(ar_start[iRow] + ar_count[iRow]);
    ar_count[iRow] = ar_start[iRow];
  }
  ar_index.resize(num_nz);
  ar_value.resize(num_nz);
  for (HighsInt iCol = 0; iCol < lp.num_col_; iCol++) {
    for (HighsInt iEl = a_start[iCol]; iEl < a_start[iCol + 1]; iEl++) {
      HighsInt iRow = a_index[iEl];
      ar_index[ar_count[iRow]] = iCol;
      ar_value[ar_count[iRow]] = a_value[iEl];
      ar_count[iRow]++;
    }
  }
  HighsInt newRow = 0;
  HighsInt newEl = 0;
  for (HighsInt iRow = 0; iRow < lp.num_row_; iRow++) {
    const HighsInt row_count = ar_start[iRow + 1] - ar_start[iRow];
    if (row_count == 1) {
      HighsInt iCol = ar_index[ar_start[iRow]];
      double value = ar_value[ar_start[iRow]];
      assert(value);
      if (value > 0) {
        if (lp.row_lower_[iRow] > -kHighsInf)
          lp.col_lower_[iCol] =
              std::max(lp.row_lower_[iRow] / value, lp.col_lower_[iCol]);
        if (lp.row_upper_[iRow] < kHighsInf)
          lp.col_upper_[iCol] =
              std::min(lp.row_upper_[iRow] / value, lp.col_upper_[iCol]);
      } else {
        if (lp.row_lower_[iRow] > -kHighsInf)
          lp.col_upper_[iCol] =
              std::min(lp.row_lower_[iRow] / value, lp.col_upper_[iCol]);
        if (lp.row_upper_[iRow] < kHighsInf)
          lp.col_lower_[iCol] =
              std::max(lp.row_upper_[iRow] / value, lp.col_lower_[iCol]);
      }
      num_row_count_1++;
      continue;
    }
    lp.row_lower_[newRow] = lp.row_lower_[iRow];
    lp.row_upper_[newRow] = lp.row_upper_[iRow];
    if (has_name) lp.row_names_[newRow] = lp.row_names_[iRow];
    ar_start[newRow] = newEl;
    for (HighsInt iEl = ar_start[iRow]; iEl < ar_start[iRow + 1]; iEl++) {
      ar_index[newEl] = ar_index[iEl];
      ar_value[newEl] = ar_value[iEl];
      newEl++;
    }
    newRow++;
  }
  ar_start[newRow] = newEl;
  lp.num_row_ = newRow;
  lp.row_lower_.resize(newRow);
  lp.row_upper_.resize(newRow);
  if (has_name) lp.row_names_.resize(newRow);

  num_nz = ar_start[lp.num_row_];
  a_count.assign(lp.num_col_, 0);
  for (HighsInt iRow = 0; iRow < lp.num_row_; iRow++) {
    for (HighsInt iEl = ar_start[iRow]; iEl < ar_start[iRow + 1]; iEl++)
      a_count[ar_index[iEl]]++;
  }
  a_start[0] = 0;
  for (HighsInt iCol = 0; iCol < lp.num_col_; iCol++) {
    a_start[iCol + 1] = a_start[iCol] + a_count[iCol];
    a_count[iCol] = a_start[iCol];
  }
  a_index.resize(num_nz);
  a_value.resize(num_nz);
  for (HighsInt iRow = 0; iRow < lp.num_row_; iRow++) {
    for (HighsInt iEl = ar_start[iRow]; iEl < ar_start[iRow + 1]; iEl++) {
      HighsInt iCol = ar_index[iEl];
      a_index[a_count[iCol]] = iRow;
      a_value[a_count[iCol]] = ar_value[iEl];
      a_count[iCol]++;
    }
  }
  assert(original_num_row - lp.num_row_ == num_row_count_1);
  assert(original_num_nz - num_nz == num_row_count_1);
  highsLogUser(log_options, HighsLogType::kWarning,
               "Removed %d rows of count 1\n", (int)num_row_count_1);
}
