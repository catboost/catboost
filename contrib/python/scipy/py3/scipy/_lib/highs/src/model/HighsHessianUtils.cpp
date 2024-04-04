/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
/*                                                                       */
/*    This file is part of the HiGHS linear optimization suite           */
/*                                                                       */
/*    Written and engineered 2008-2021 at the University of Edinburgh    */
/*                                                                       */
/*    Available as open-source under the MIT License                     */
/*                                                                       */
/*    Authors: Julian Hall, Ivet Galabova, Qi Huangfu, Leona Gottwald    */
/*    and Michael Feldmeier                                              */
/*                                                                       */
/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
/**@file lp_data/HighsHessianUtils.cpp
 * @brief
 */
#include "model/HighsHessianUtils.h"

#include <algorithm>
#include <cmath>

#include "lp_data/HighsModelUtils.h"
#include "util/HighsMatrixUtils.h"
#include "util/HighsSort.h"

using std::fabs;

HighsStatus assessHessian(HighsHessian& hessian, const HighsOptions& options) {
  HighsStatus return_status = HighsStatus::kOk;
  HighsStatus call_status;

  // Assess the Hessian dimensions and vector sizes, returning on error
  vector<HighsInt> hessian_p_end;
  const bool partitioned = false;
  call_status = assessMatrixDimensions(
      options.log_options, hessian.dim_, partitioned, hessian.start_,
      hessian_p_end, hessian.index_, hessian.value_);
  return_status = interpretCallStatus(options.log_options, call_status,
                                      return_status, "assessMatrixDimensions");
  if (return_status == HighsStatus::kError) return return_status;

  // If the Hessian has no columns there is nothing left to test
  if (hessian.dim_ == 0) return HighsStatus::kOk;

  // Assess the Hessian matrix
  //
  // The start of column 0 must be zero.
  if (hessian.start_[0]) {
    highsLogUser(options.log_options, HighsLogType::kError,
                 "Hessian has nonzero value (%" HIGHSINT_FORMAT
                 ") for the start of column 0\n",
                 hessian.start_[0]);
    return HighsStatus::kError;
  }
  // Assess G, deferring the assessment of values (other than those
  // which are identically zero)
  call_status = assessMatrix(options.log_options, "Hessian", hessian.dim_,
                             hessian.dim_, hessian.start_, hessian.index_,
                             hessian.value_, 0, kHighsInf);
  return_status = interpretCallStatus(options.log_options, call_status,
                                      return_status, "assessMatrix");
  if (return_status == HighsStatus::kError) return return_status;

  if (hessian.format_ == HessianFormat::kSquare) {
    // Form Q = (G+G^T)/2
    call_status = normaliseHessian(options, hessian);
    return_status = interpretCallStatus(options.log_options, call_status,
                                        return_status, "normaliseHessian");
    if (return_status == HighsStatus::kError) return return_status;
  }
  // Extract the triangular part of Q: lower triangle column-wise
  // or, equivalently, upper triangle row-wise, ensuring that the
  // diagonal entry comes first, unless it's zero
  call_status = extractTriangularHessian(options, hessian);
  return_status =
      interpretCallStatus(options.log_options, call_status, return_status,
                          "extractTriangularHessian");
  if (return_status == HighsStatus::kError) return return_status;

  // Assess Q
  call_status =
      assessMatrix(options.log_options, "Hessian", hessian.dim_, hessian.dim_,
                   hessian.start_, hessian.index_, hessian.value_,
                   options.small_matrix_value, options.large_matrix_value);
  return_status = interpretCallStatus(options.log_options, call_status,
                                      return_status, "assessMatrix");
  if (return_status == HighsStatus::kError) return return_status;

  HighsInt hessian_num_nz = hessian.numNz();
  // If the Hessian has nonzeros, complete its diagonal with explicit
  // zeros if necessary
  if (hessian_num_nz) {
    completeHessianDiagonal(options, hessian);
    hessian_num_nz = hessian.numNz();
  }
  // If entries have been removed from the matrix, resize the index
  // and value vectors
  if ((HighsInt)hessian.index_.size() > hessian_num_nz)
    hessian.index_.resize(hessian_num_nz);
  if ((HighsInt)hessian.value_.size() > hessian_num_nz)
    hessian.value_.resize(hessian_num_nz);

  if (return_status != HighsStatus::kError) return_status = HighsStatus::kOk;
  if (return_status != HighsStatus::kOk)
    highsLogDev(options.log_options, HighsLogType::kInfo,
                "assessHessian returns HighsStatus = %s\n",
                highsStatusToString(return_status).c_str());
  return return_status;
}

void completeHessianDiagonal(const HighsOptions& options,
                             HighsHessian& hessian) {
  // Count the number of missing diagonal entries
  HighsInt num_missing_diagonal_entries = 0;
  const HighsInt dim = hessian.dim_;
  const HighsInt num_nz = hessian.numNz();
  for (HighsInt iCol = 0; iCol < dim; iCol++) {
    HighsInt iEl = hessian.start_[iCol];
    if (iEl < num_nz) {
      if (hessian.index_[iEl] != iCol) num_missing_diagonal_entries++;
    } else {
      num_missing_diagonal_entries++;
    }
  }
  highsLogDev(options.log_options, HighsLogType::kInfo,
              "Hessian has dimension %d and %d nonzeros: inserting %d zeros "
              "onto the diagonal\n",
              (int)dim, (int)num_nz, (int)num_missing_diagonal_entries);
  assert(num_missing_diagonal_entries >= dim - num_nz);
  if (!num_missing_diagonal_entries) return;
  // There are missing diagonal entries to be inserted as explicit zeros
  const HighsInt new_num_nz = hessian.numNz() + num_missing_diagonal_entries;
  HighsInt to_iEl = new_num_nz;
  hessian.index_.resize(new_num_nz);
  hessian.value_.resize(new_num_nz);
  HighsInt next_start = hessian.numNz();
  hessian.start_[dim] = to_iEl;
  HighsInt num_missing_diagonal_entries_added = 0;
  for (HighsInt iCol = dim - 1; iCol >= 0; iCol--) {
    // Shift the entries that are sure to be off-diagonal
    for (HighsInt iEl = next_start - 1; iEl > hessian.start_[iCol]; iEl--) {
      assert(hessian.index_[iEl] != iCol);
      to_iEl--;
      hessian.index_[to_iEl] = hessian.index_[iEl];
      hessian.value_[to_iEl] = hessian.value_[iEl];
    }
    // Now consider any first entry. If there is none, or if it's not
    // the diagonal, then there is no diagonal entry for this column
    bool no_diagonal_entry;
    if (hessian.start_[iCol] < next_start) {
      const HighsInt iEl = hessian.start_[iCol];
      // Copy the first entry
      to_iEl--;
      hessian.index_[to_iEl] = hessian.index_[iEl];
      hessian.value_[to_iEl] = hessian.value_[iEl];
      // If the first entry isn't the diagonal, then there is no
      // diagonal entry for this column
      no_diagonal_entry = hessian.index_[iEl] != iCol;
    } else {
      no_diagonal_entry = true;
    }
    if (no_diagonal_entry) {
      // There is no diagonal entry, so have insert an explicit zero
      to_iEl--;
      hessian.index_[to_iEl] = iCol;
      hessian.value_[to_iEl] = 0;
      num_missing_diagonal_entries_added++;
      assert(num_missing_diagonal_entries_added <=
             num_missing_diagonal_entries);
    }
    next_start = hessian.start_[iCol];
    hessian.start_[iCol] = to_iEl;
  }
  assert(to_iEl == 0);
}

bool okHessianDiagonal(const HighsOptions& options, HighsHessian& hessian,
                       const ObjSense sense) {
  double min_diagonal_value = kHighsInf;
  double max_diagonal_value = -kHighsInf;
  const HighsInt dim = hessian.dim_;
  const HighsInt sense_sign = (HighsInt)sense;
  HighsInt num_illegal_diagonal_value = 0;
  for (HighsInt iCol = 0; iCol < dim; iCol++) {
    double diagonal_value = 0;
    // Assumes that the diagonal entry is always first, possibly with explicit
    // zero value
    HighsInt iEl = hessian.start_[iCol];
    assert(hessian.index_[iEl] == iCol);
    diagonal_value = sense_sign * hessian.value_[iEl];
    min_diagonal_value = std::min(diagonal_value, min_diagonal_value);
    max_diagonal_value = std::max(diagonal_value, max_diagonal_value);
    // Diagonal entries signed by sense must be non-negative
    if (diagonal_value < 0) num_illegal_diagonal_value++;
  }

  const bool certainly_not_positive_semidefinite =
      num_illegal_diagonal_value > 0;
  if (certainly_not_positive_semidefinite) {
    if (sense == ObjSense::kMinimize) {
      highsLogUser(options.log_options, HighsLogType::kError,
                   "Hessian has %" HIGHSINT_FORMAT
                   " diagonal entries in [%g, 0) so is not positive "
                   "semidefinite for minimization\n",
                   num_illegal_diagonal_value, min_diagonal_value);
    } else {
      highsLogUser(options.log_options, HighsLogType::kError,
                   "Hessian has %" HIGHSINT_FORMAT
                   " diagonal entries in (0, %g] so is not negative "
                   "semidefinite for maximization\n",
                   num_illegal_diagonal_value, -min_diagonal_value);
    }
  }
  return !certainly_not_positive_semidefinite;
}

HighsStatus extractTriangularHessian(const HighsOptions& options,
                                     HighsHessian& hessian) {
  // Viewing the Hessian column-wise, remove any entries in the strict
  // upper triangle
  HighsStatus return_status = HighsStatus::kOk;
  const HighsInt dim = hessian.dim_;
  HighsInt nnz = 0;
  for (HighsInt iCol = 0; iCol < dim; iCol++) {
    double diagonal_value = 0;
    const HighsInt nnz0 = nnz;
    for (HighsInt iEl = hessian.start_[iCol]; iEl < hessian.start_[iCol + 1];
         iEl++) {
      HighsInt iRow = hessian.index_[iEl];
      if (iRow < iCol) continue;
      hessian.index_[nnz] = iRow;
      hessian.value_[nnz] = hessian.value_[iEl];
      if (iRow == iCol && nnz > nnz0) {
        // Diagonal entry is not first in column so swap it in
        hessian.index_[nnz] = hessian.index_[nnz0];
        hessian.value_[nnz] = hessian.value_[nnz0];
        hessian.index_[nnz0] = iRow;
        hessian.value_[nnz0] = hessian.value_[iEl];
      }
      nnz++;
    }
    hessian.start_[iCol] = nnz0;
  }
  const HighsInt num_ignored_nz = hessian.start_[dim] - nnz;
  assert(num_ignored_nz >= 0);
  if (num_ignored_nz) {
    if (hessian.format_ == HessianFormat::kTriangular) {
      highsLogUser(options.log_options, HighsLogType::kWarning,
                   "Ignored %" HIGHSINT_FORMAT
                   " entries of Hessian in opposite triangle\n",
                   num_ignored_nz);
      return_status = HighsStatus::kWarning;
    }
    hessian.start_[dim] = nnz;
  }
  assert(hessian.start_[dim] == nnz);
  hessian.format_ = HessianFormat::kTriangular;
  return return_status;
}

void triangularToSquareHessian(const HighsHessian& hessian,
                               vector<HighsInt>& start, vector<HighsInt>& index,
                               vector<double>& value) {
  const HighsInt dim = hessian.dim_;
  if (dim <= 0) {
    start.assign(1, 0);
    return;
  }
  assert(hessian.format_ == HessianFormat::kTriangular);
  const HighsInt nnz = hessian.start_[dim];
  const HighsInt square_nnz = nnz + (nnz - dim);
  start.resize(dim + 1);
  index.resize(square_nnz);
  value.resize(square_nnz);
  vector<HighsInt> length;
  length.assign(dim, 0);
  for (HighsInt iCol = 0; iCol < dim; iCol++) {
    HighsInt iRow = hessian.index_[hessian.start_[iCol]];
    assert(iRow == iCol);
    length[iCol]++;
    for (HighsInt iEl = hessian.start_[iCol] + 1;
         iEl < hessian.start_[iCol + 1]; iEl++) {
      HighsInt iRow = hessian.index_[iEl];
      assert(iRow > iCol);
      length[iRow]++;
      length[iCol]++;
    }
  }
  start[0] = 0;
  for (HighsInt iCol = 0; iCol < dim; iCol++)
    start[iCol + 1] = start[iCol] + length[iCol];
  assert(square_nnz == start[dim]);
  for (HighsInt iCol = 0; iCol < dim; iCol++) {
    HighsInt iEl = hessian.start_[iCol];
    HighsInt iRow = hessian.index_[iEl];
    HighsInt toEl = start[iCol];
    index[toEl] = iRow;
    value[toEl] = hessian.value_[iEl];
    start[iCol]++;
    for (HighsInt iEl = hessian.start_[iCol] + 1;
         iEl < hessian.start_[iCol + 1]; iEl++) {
      HighsInt iRow = hessian.index_[iEl];
      HighsInt toEl = start[iRow];
      index[toEl] = iCol;
      value[toEl] = hessian.value_[iEl];
      start[iRow]++;
      toEl = start[iCol];
      index[toEl] = iRow;
      value[toEl] = hessian.value_[iEl];
      start[iCol]++;
    }
  }
  start[0] = 0;
  for (HighsInt iCol = 0; iCol < dim; iCol++)
    start[iCol + 1] = start[iCol] + length[iCol];
}

HighsStatus normaliseHessian(const HighsOptions& options,
                             HighsHessian& hessian) {
  // Only relevant for a Hessian with format HessianFormat::kSquare
  assert(hessian.format_ == HessianFormat::kSquare);
  // Normalise the Hessian to be (Q + Q^T)/2, where Q is the matrix
  // supplied. This guarantees that what's used internally is
  // symmetric.
  //
  // So someone preferring to supply only the upper triangle would
  // have to double its values..
  HighsStatus return_status = HighsStatus::kOk;
  const HighsInt dim = hessian.dim_;
  const HighsInt hessian_num_nz = hessian.start_[dim];
  if (hessian_num_nz <= 0) return HighsStatus::kOk;
  bool warning_found = false;

  HighsHessian transpose;
  transpose.dim_ = dim;
  transpose.start_.resize(dim + 1);
  transpose.index_.resize(hessian_num_nz);
  transpose.value_.resize(hessian_num_nz);
  // Form transpose of Hessian
  vector<HighsInt> qr_length;
  qr_length.assign(dim, 0);
  for (HighsInt iEl = 0; iEl < hessian_num_nz; iEl++)
    qr_length[hessian.index_[iEl]]++;

  transpose.start_[0] = 0;
  for (HighsInt iRow = 0; iRow < dim; iRow++)
    transpose.start_[iRow + 1] = transpose.start_[iRow] + qr_length[iRow];
  for (HighsInt iCol = 0; iCol < dim; iCol++) {
    for (HighsInt iEl = hessian.start_[iCol]; iEl < hessian.start_[iCol + 1];
         iEl++) {
      HighsInt iRow = hessian.index_[iEl];
      HighsInt iRowEl = transpose.start_[iRow];
      transpose.index_[iRowEl] = iCol;
      transpose.value_[iRowEl] = hessian.value_[iEl];
      transpose.start_[iRow]++;
    }
  }

  transpose.start_[0] = 0;
  for (HighsInt iRow = 0; iRow < dim; iRow++)
    transpose.start_[iRow + 1] = transpose.start_[iRow] + qr_length[iRow];
  // Instantiate a square format Hessian in which to accumulate (Q + Q^T)/2
  HighsHessian normalised;
  normalised.format_ = HessianFormat::kSquare;
  HighsInt normalised_num_nz = 0;
  HighsInt normalised_size = hessian_num_nz;
  normalised.dim_ = dim;
  normalised.start_.resize(dim + 1);
  normalised.index_.resize(normalised_size);
  normalised.value_.resize(normalised_size);
  vector<double> column_value;
  vector<HighsInt> column_index;
  column_index.resize(dim);
  column_value.assign(dim, 0.0);
  const bool check_column_value_zero = false;
  const double small_matrix_value = 0;
  HighsInt num_small_values = 0;
  double max_small_value = 0;
  double min_small_value = kHighsInf;
  normalised.start_[0] = 0;
  for (HighsInt iCol = 0; iCol < dim; iCol++) {
    HighsInt column_num_nz = 0;
    for (HighsInt iEl = hessian.start_[iCol]; iEl < hessian.start_[iCol + 1];
         iEl++) {
      HighsInt iRow = hessian.index_[iEl];
      column_value[iRow] = hessian.value_[iEl];
      column_index[column_num_nz] = iRow;
      column_num_nz++;
    }
    for (HighsInt iEl = transpose.start_[iCol];
         iEl < transpose.start_[iCol + 1]; iEl++) {
      HighsInt iRow = transpose.index_[iEl];
      if (column_value[iRow]) {
        column_value[iRow] += transpose.value_[iEl];
      } else {
        column_value[iRow] = transpose.value_[iEl];
        column_index[column_num_nz] = iRow;
        column_num_nz++;
      }
    }
    if (normalised_num_nz + column_num_nz > normalised_size) {
      normalised_size =
          std::max(normalised_num_nz + column_num_nz, 2 * normalised_size);
      normalised.index_.resize(normalised_size);
      normalised.value_.resize(normalised_size);
    }
    // Halve the values, zeroing and accounting for any small ones
    for (HighsInt ix = 0; ix < column_num_nz; ix++) {
      HighsInt iRow = column_index[ix];
      double value = 0.5 * column_value[iRow];
      double abs_value = std::fabs(value);
      bool ok_value = abs_value > small_matrix_value;
      if (!ok_value) {
        value = 0;
        if (max_small_value < abs_value) max_small_value = abs_value;
        if (min_small_value > abs_value) min_small_value = abs_value;
        num_small_values++;
      }
      column_value[iRow] = value;
    }
    // Decide whether to exploit sparsity in extracting the indices
    // and values of nonzeros
    const HighsInt kDimTolerance = 10;
    const double kDensityTolerance = 0.1;
    const double density = (1.0 * column_num_nz) / (1.0 * dim);
    HighsInt to_ix = dim;
    const bool exploit_sparsity =
        dim > kDimTolerance && density < kDensityTolerance;
    if (exploit_sparsity) {
      // Exploit sparsity
      to_ix = column_num_nz;
      sortSetData(column_num_nz, column_index, NULL, NULL);
    } else {
      to_ix = dim;
    }
    for (HighsInt ix = 0; ix < to_ix; ix++) {
      HighsInt iRow;
      if (exploit_sparsity) {
        iRow = column_index[ix];
      } else {
        iRow = ix;
      }
      double value = column_value[iRow];
      if (value) {
        normalised.index_[normalised_num_nz] = iRow;
        normalised.value_[normalised_num_nz] = value;
        normalised_num_nz++;
        column_value[iRow] = 0;
      }
    }
    if (check_column_value_zero) {
      for (HighsInt iRow = 0; iRow < dim; iRow++)
        assert(column_value[iRow] == 0);
    }
    normalised.start_[iCol + 1] = normalised_num_nz;
  }
  if (num_small_values) {
    highsLogUser(options.log_options, HighsLogType::kWarning,
                 "Normalised Hessian contains %" HIGHSINT_FORMAT
                 " |values| in [%g, %g] "
                 "less than %g: ignored\n",
                 num_small_values, min_small_value, max_small_value,
                 small_matrix_value);
    warning_found = true;
  }
  // Replace the Hessian by the normalised form
  hessian = normalised;
  assert(hessian.format_ == HessianFormat::kSquare);
  if (warning_found)
    return_status = HighsStatus::kWarning;
  else
    return_status = HighsStatus::kOk;

  return return_status;
}

void reportHessian(const HighsLogOptions& log_options, const HighsInt dim,
                   const HighsInt num_nz, const HighsInt* start,
                   const HighsInt* index, const double* value) {
  if (dim <= 0) return;
  highsLogUser(log_options, HighsLogType::kInfo,
               "Hessian Index              Value\n");
  for (HighsInt col = 0; col < dim; col++) {
    highsLogUser(log_options, HighsLogType::kInfo,
                 "    %8" HIGHSINT_FORMAT " Start   %10" HIGHSINT_FORMAT "\n",
                 col, start[col]);
    HighsInt to_el = (col < dim - 1 ? start[col + 1] : num_nz);
    for (HighsInt el = start[col]; el < to_el; el++)
      highsLogUser(log_options, HighsLogType::kInfo,
                   "          %8" HIGHSINT_FORMAT " %12g\n", index[el],
                   value[el]);
  }
  highsLogUser(log_options, HighsLogType::kInfo,
               "             Start   %10" HIGHSINT_FORMAT "\n", num_nz);
}
