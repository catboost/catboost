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
/**@file util/HighsMatrixUtils.cpp
 * @brief Class-independent utilities for HiGHS
 */

#include "util/HighsMatrixUtils.h"

#include <algorithm>
#include <cmath>
#include <cstdio>

HighsStatus assessMatrix(const HighsLogOptions& log_options,
                         const std::string matrix_name, const HighsInt vec_dim,
                         const HighsInt num_vec, vector<HighsInt>& matrix_start,
                         vector<HighsInt>& matrix_index,
                         vector<double>& matrix_value,
                         const double small_matrix_value,
                         const double large_matrix_value) {
  vector<HighsInt> matrix_p_end;
  const bool partitioned = false;
  return assessMatrix(log_options, matrix_name, vec_dim, num_vec, partitioned,
                      matrix_start, matrix_p_end, matrix_index, matrix_value,
                      small_matrix_value, large_matrix_value);
}

HighsStatus assessMatrix(const HighsLogOptions& log_options,
                         const std::string matrix_name, const HighsInt vec_dim,
                         const HighsInt num_vec, vector<HighsInt>& matrix_start,
                         vector<HighsInt>& matrix_p_end,
                         vector<HighsInt>& matrix_index,
                         vector<double>& matrix_value,
                         const double small_matrix_value,
                         const double large_matrix_value) {
  const bool partitioned = false;
  return assessMatrix(log_options, matrix_name, vec_dim, num_vec, partitioned,
                      matrix_start, matrix_p_end, matrix_index, matrix_value,
                      small_matrix_value, large_matrix_value);
}

HighsStatus assessMatrix(
    const HighsLogOptions& log_options, const std::string matrix_name,
    const HighsInt vec_dim, const HighsInt num_vec, const bool partitioned,
    vector<HighsInt>& matrix_start, vector<HighsInt>& matrix_p_end,
    vector<HighsInt>& matrix_index, vector<double>& matrix_value,
    const double small_matrix_value, const double large_matrix_value) {
  if (assessMatrixDimensions(log_options, num_vec, partitioned, matrix_start,
                             matrix_p_end, matrix_index,
                             matrix_value) == HighsStatus::kError) {
    return HighsStatus::kError;
  }

  bool error_found = false;
  bool warning_found = false;

  const HighsInt num_nz = matrix_start[num_vec];
  // Assess the starts
  //
  // Check whether the first start is zero
  if (matrix_start[0]) {
    highsLogUser(log_options, HighsLogType::kWarning,
                 "%s matrix start vector begins with %" HIGHSINT_FORMAT
                 " rather than 0\n",
                 matrix_name.c_str(), matrix_start[0]);
    return HighsStatus::kError;
  }
  // Set up previous_start for a fictitious previous empty packed vector
  HighsInt previous_start = matrix_start[0];
  // Set up this_start to be the first start in case num_vec = 0
  HighsInt this_start = matrix_start[0];
  HighsInt this_p_end = 0;
  if (partitioned) this_p_end = matrix_p_end[0];
  for (HighsInt ix = 0; ix < num_vec; ix++) {
    this_start = matrix_start[ix];
    HighsInt next_start = matrix_start[ix + 1];
    bool this_start_too_small = this_start < previous_start;
    if (this_start_too_small) {
      highsLogUser(log_options, HighsLogType::kError,
                   "%s matrix packed vector %" HIGHSINT_FORMAT
                   " has illegal start of %" HIGHSINT_FORMAT
                   " < %" HIGHSINT_FORMAT
                   " = "
                   "previous start\n",
                   matrix_name.c_str(), ix, this_start, previous_start);
      return HighsStatus::kError;
    }
    if (partitioned) {
      this_p_end = matrix_p_end[ix];
      bool this_p_end_too_small = this_p_end < this_start;
      if (this_p_end_too_small) {
        highsLogUser(log_options, HighsLogType::kError,
                     "%s matrix packed vector %" HIGHSINT_FORMAT
                     " has illegal partition end of %" HIGHSINT_FORMAT
                     " < %" HIGHSINT_FORMAT
                     " = "
                     " start\n",
                     matrix_name.c_str(), ix, this_p_end, this_start);
        return HighsStatus::kError;
      }
    }
    previous_start = this_start;
  }
  bool this_start_too_big = this_start > num_nz;
  if (this_start_too_big) {
    highsLogUser(log_options, HighsLogType::kError,
                 "%s matrix packed vector %" HIGHSINT_FORMAT
                 " has illegal start of %" HIGHSINT_FORMAT
                 " > %" HIGHSINT_FORMAT
                 " = "
                 "number of nonzeros\n",
                 matrix_name.c_str(), num_vec, this_start, num_nz);
    return HighsStatus::kError;
  }
  if (partitioned) {
    bool this_p_end_too_big = this_p_end > num_nz;
    if (this_p_end_too_big) {
      highsLogUser(log_options, HighsLogType::kError,
                   "%s matrix packed vector %" HIGHSINT_FORMAT
                   " has illegal partition end of %" HIGHSINT_FORMAT
                   " > %" HIGHSINT_FORMAT
                   " = "
                   "number of nonzeros\n",
                   matrix_name.c_str(), num_vec, this_p_end, num_nz);
      return HighsStatus::kError;
    }
  }
  // Assess the indices and values
  // Count the number of acceptable indices/values
  HighsInt num_new_nz = 0;
  HighsInt num_small_values = 0;
  double max_small_value = 0;
  double min_small_value = kHighsInf;
  HighsInt num_large_values = 0;
  double max_large_value = 0;
  double min_large_value = kHighsInf;
  // Set up a zeroed vector to detect duplicate indices
  vector<HighsInt> check_vector;
  if (vec_dim > 0) check_vector.assign(vec_dim, 0);
  for (HighsInt ix = 0; ix < num_vec; ix++) {
    HighsInt from_el = matrix_start[ix];
    HighsInt to_el = matrix_start[ix + 1];
    // Account for any index-value pairs removed so far
    matrix_start[ix] = num_new_nz;
    for (HighsInt el = from_el; el < to_el; el++) {
      // Check the index
      HighsInt component = matrix_index[el];
      // Check that the index is non-negative
      bool legal_component = component >= 0;
      if (!legal_component) {
        highsLogUser(log_options, HighsLogType::kError,
                     "%s matrix packed vector %" HIGHSINT_FORMAT
                     ", entry %" HIGHSINT_FORMAT
                     ", is illegal index %" HIGHSINT_FORMAT "\n",
                     matrix_name.c_str(), ix, el, component);
        return HighsStatus::kError;
      }
      // Check that the index does not exceed the vector dimension
      legal_component = component < vec_dim;
      if (!legal_component) {
        highsLogUser(log_options, HighsLogType::kError,
                     "%s matrix packed vector %" HIGHSINT_FORMAT
                     ", entry %" HIGHSINT_FORMAT
                     ", is illegal index "
                     "%12" HIGHSINT_FORMAT " >= %" HIGHSINT_FORMAT
                     " = vector dimension\n",
                     matrix_name.c_str(), ix, el, component, vec_dim);
        return HighsStatus::kError;
      }
      // Check that the index has not already ocurred
      legal_component = check_vector[component] == 0;
      if (!legal_component) {
        highsLogUser(log_options, HighsLogType::kError,
                     "%s matrix packed vector %" HIGHSINT_FORMAT
                     ", entry %" HIGHSINT_FORMAT
                     ", is duplicate index %" HIGHSINT_FORMAT "\n",
                     matrix_name.c_str(), ix, el, component);
        return HighsStatus::kError;
      }
      // Indicate that the index has occurred
      check_vector[component] = 1;
      // Check the value
      double abs_value = fabs(matrix_value[el]);
      // Check that the value is not too large
      bool large_value = abs_value > large_matrix_value;
      if (large_value) {
        if (max_large_value < abs_value) max_large_value = abs_value;
        if (min_large_value > abs_value) min_large_value = abs_value;
        num_large_values++;
      }
      bool ok_value = abs_value > small_matrix_value;
      if (!ok_value) {
        if (max_small_value < abs_value) max_small_value = abs_value;
        if (min_small_value > abs_value) min_small_value = abs_value;
        num_small_values++;
      }
      if (ok_value) {
        // Shift the index and value of the OK entry to the new
        // position in the index and value vectors, and increment
        // the new number of nonzeros
        matrix_index[num_new_nz] = matrix_index[el];
        matrix_value[num_new_nz] = matrix_value[el];
        num_new_nz++;
      } else {
        // Zero the check_vector entry since the small value
        // _hasn't_ occurred
        check_vector[component] = 0;
      }
    }
    // Zero check_vector
    for (HighsInt el = matrix_start[ix]; el < num_new_nz; el++)
      check_vector[matrix_index[el]] = 0;
    // NB This is very expensive so shouldn't be true
    const bool check_check_vector = false;
    if (check_check_vector) {
      // Check zeroing of check vector
      for (HighsInt component = 0; component < vec_dim; component++) {
        if (check_vector[component]) error_found = true;
      }
      if (error_found)
        highsLogUser(log_options, HighsLogType::kError,
                     "assessMatrix: check_vector not zeroed\n");
    }
  }
  if (num_large_values) {
    highsLogUser(log_options, HighsLogType::kError,
                 "%s matrix packed vector contains %" HIGHSINT_FORMAT
                 " |values| in [%g, %g] greater than %g\n",
                 matrix_name.c_str(), num_large_values, min_large_value,
                 max_large_value, large_matrix_value);
    error_found = true;
  }
  if (num_small_values) {
    if (partitioned) {
      // Shouldn't happen with a partitioned row-wise matrix since its
      // values should be OK and the code above doesn't handle p_end
      highsLogUser(
          log_options, HighsLogType::kError,
          "%s matrix packed partitioned vector contains %" HIGHSINT_FORMAT
          " |values| in [%g, %g] less than or equal to %g: ignored\n",
          matrix_name.c_str(), num_small_values, min_small_value,
          max_small_value, small_matrix_value);
      error_found = true;
      assert(num_small_values == 0);
    }
    highsLogUser(log_options, HighsLogType::kWarning,
                 "%s matrix packed vector contains %" HIGHSINT_FORMAT
                 " |values| in [%g, %g] "
                 "less than or equal to %g: ignored\n",
                 matrix_name.c_str(), num_small_values, min_small_value,
                 max_small_value, small_matrix_value);
    warning_found = true;
  }
  matrix_start[num_vec] = num_new_nz;
  HighsStatus return_status = HighsStatus::kOk;
  if (error_found)
    return_status = HighsStatus::kError;
  else if (warning_found)
    return_status = HighsStatus::kWarning;
  return return_status;
}

HighsStatus assessMatrixDimensions(const HighsLogOptions& log_options,
                                   const HighsInt num_vec,
                                   const bool partitioned,
                                   const vector<HighsInt>& matrix_start,
                                   const vector<HighsInt>& matrix_p_end,
                                   const vector<HighsInt>& matrix_index,
                                   const vector<double>& matrix_value) {
  bool ok = true;
  // Assess main dimensions
  const bool legal_num_vec = num_vec >= 0;
  if (!legal_num_vec)
    highsLogUser(
        log_options, HighsLogType::kError,
        "Matrix dimension validation fails on number of vectors = %d < 0\n",
        (int)num_vec);
  ok = legal_num_vec && ok;
  const bool legal_matrix_start_size = matrix_start.size() >= num_vec + 1;
  if (!legal_matrix_start_size)
    highsLogUser(log_options, HighsLogType::kError,
                 "Matrix dimension validation fails on start size = %d < %d = "
                 "num vectors + 1\n",
                 (int)matrix_start.size(), (int)(num_vec + 1));
  ok = legal_matrix_start_size && ok;
  if (partitioned) {
    const bool legal_matrix_p_end_size = matrix_p_end.size() >= num_vec + 1;
    if (!legal_matrix_p_end_size)
      highsLogUser(log_options, HighsLogType::kError,
                   "Matrix dimension validation fails on p_end size = %d < %d "
                   "= num vectors + 1\n",
                   (int)matrix_p_end.size(), (int)(num_vec + 1));
    ok = matrix_p_end.size() >= num_vec + 1 && ok;
  }
  // Possibly check the sizes of the index and value vectors. Can only
  // do this with the number of nonzeros, and this is only known if
  // the start vector has a legal size. Setting num_nz = 0 otherwise
  // means that all tests pass, as they just check that the sizes of
  // the index and value vectors are non-negative.
  const HighsInt num_nz = legal_matrix_start_size ? matrix_start[num_vec] : 0;
  if (num_nz >= 0) {
    const bool legal_matrix_index_size = matrix_index.size() >= num_nz;
    if (!legal_matrix_index_size)
      highsLogUser(log_options, HighsLogType::kError,
                   "Matrix dimension validation fails on index size = %d < %d "
                   "= number of nonzeros\n",
                   (int)matrix_index.size(), (int)num_nz);
    ok = legal_matrix_index_size && ok;
    const bool legal_matrix_value_size = matrix_value.size() >= num_nz;
    if (!legal_matrix_value_size)
      highsLogUser(log_options, HighsLogType::kError,
                   "Matrix dimension validation fails on value size = %d < %d "
                   "= number of nonzeros\n",
                   (int)matrix_value.size(), (int)num_nz);
    ok = legal_matrix_value_size && ok;
  } else {
    highsLogUser(
        log_options, HighsLogType::kError,
        "Matrix dimension validation fails on number of nonzeros = %d < 0\n",
        (int)num_nz);
    ok = false;
  }
  if (ok) return HighsStatus::kOk;
  return HighsStatus::kError;
}
