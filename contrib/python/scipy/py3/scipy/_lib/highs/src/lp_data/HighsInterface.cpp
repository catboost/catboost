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
/**@file lp_data/HighsInterface.cpp
 * @brief
 */
#include "Highs.h"
#include "lp_data/HighsLpUtils.h"
#include "lp_data/HighsModelUtils.h"
#include "simplex/HSimplex.h"
#include "util/HighsMatrixUtils.h"
#include "util/HighsSort.h"

HighsStatus Highs::basisForSolution() {
  HighsLp& lp = model_.lp_;
  assert(!lp.isMip());
  assert(solution_.value_valid);
  invalidateBasis();
  HighsInt num_basic = 0;
  HighsBasis basis;
  for (HighsInt iCol = 0; iCol < lp.num_col_; iCol++) {
    if (std::fabs(lp.col_lower_[iCol] - solution_.col_value[iCol]) <=
        options_.primal_feasibility_tolerance) {
      basis.col_status.push_back(HighsBasisStatus::kLower);
    } else if (std::fabs(lp.col_upper_[iCol] - solution_.col_value[iCol]) <=
               options_.primal_feasibility_tolerance) {
      basis.col_status.push_back(HighsBasisStatus::kUpper);
    } else {
      num_basic++;
      basis.col_status.push_back(HighsBasisStatus::kBasic);
    }
  }
  const HighsInt num_basic_col = num_basic;
  for (HighsInt iRow = 0; iRow < lp.num_row_; iRow++) {
    if (std::fabs(lp.row_lower_[iRow] - solution_.row_value[iRow]) <=
        options_.primal_feasibility_tolerance) {
      basis.row_status.push_back(HighsBasisStatus::kLower);
    } else if (std::fabs(lp.row_upper_[iRow] - solution_.row_value[iRow]) <=
               options_.primal_feasibility_tolerance) {
      basis.row_status.push_back(HighsBasisStatus::kUpper);
    } else {
      num_basic++;
      basis.row_status.push_back(HighsBasisStatus::kBasic);
    }
  }
  const HighsInt num_basic_row = num_basic - num_basic_col;
  assert((int)basis.col_status.size() == lp.num_col_);
  assert((int)basis.row_status.size() == lp.num_row_);
  highsLogUser(options_.log_options, HighsLogType::kInfo,
               "LP has %d rows and %d basic variables (%d / %d; %d / %d)\n",
               (int)lp.num_row_, (int)num_basic, (int)num_basic_col,
               (int)lp.num_col_, (int)num_basic_row, (int)lp.num_row_);
  return this->setBasis(basis);
}

HighsStatus Highs::addColsInterface(
    HighsInt ext_num_new_col, const double* ext_col_cost,
    const double* ext_col_lower, const double* ext_col_upper,
    HighsInt ext_num_new_nz, const HighsInt* ext_a_start,
    const HighsInt* ext_a_index, const double* ext_a_value) {
  HighsStatus return_status = HighsStatus::kOk;
  HighsOptions& options = options_;
  if (ext_num_new_col < 0) return HighsStatus::kError;
  if (ext_num_new_nz < 0) return HighsStatus::kError;
  if (ext_num_new_col == 0) return HighsStatus::kOk;
  if (ext_num_new_col > 0)
    if (isColDataNull(options.log_options, ext_col_cost, ext_col_lower,
                      ext_col_upper))
      return HighsStatus::kError;
  if (ext_num_new_nz > 0)
    if (isMatrixDataNull(options.log_options, ext_a_start, ext_a_index,
                         ext_a_value))
      return HighsStatus::kError;

  HighsLp& lp = model_.lp_;
  HighsBasis& basis = basis_;
  HighsScale& scale = lp.scale_;
  bool& valid_basis = basis.valid;
  bool& lp_has_scaling = lp.scale_.has_scaling;

  // Check that if nonzeros are to be added then the model has a positive number
  // of rows
  if (lp.num_row_ <= 0 && ext_num_new_nz > 0) return HighsStatus::kError;

  // Record the new number of columns
  HighsInt newNumCol = lp.num_col_ + ext_num_new_col;

  HighsIndexCollection index_collection;
  index_collection.dimension_ = ext_num_new_col;
  index_collection.is_interval_ = true;
  index_collection.from_ = 0;
  index_collection.to_ = ext_num_new_col - 1;

  // Take a copy of the cost and bounds that can be normalised
  std::vector<double> local_colCost{ext_col_cost,
                                    ext_col_cost + ext_num_new_col};
  std::vector<double> local_colLower{ext_col_lower,
                                     ext_col_lower + ext_num_new_col};
  std::vector<double> local_colUpper{ext_col_upper,
                                     ext_col_upper + ext_num_new_col};

  return_status =
      interpretCallStatus(options_.log_options,
                          assessCosts(options, lp.num_col_, index_collection,
                                      local_colCost, options.infinite_cost),
                          return_status, "assessCosts");
  if (return_status == HighsStatus::kError) return return_status;
  // Assess the column bounds
  return_status = interpretCallStatus(
      options_.log_options,
      assessBounds(options, "Col", lp.num_col_, index_collection,
                   local_colLower, local_colUpper, options.infinite_bound),
      return_status, "assessBounds");
  if (return_status == HighsStatus::kError) return return_status;
  // Append the columns to the LP vectors and matrix
  appendColsToLpVectors(lp, ext_num_new_col, local_colCost, local_colLower,
                        local_colUpper);

  // Form a column-wise HighsSparseMatrix of the new matrix columns so
  // that is is easy to handle and, if there are nonzeros, it can be
  // normalised
  HighsSparseMatrix local_a_matrix;
  local_a_matrix.num_col_ = ext_num_new_col;
  local_a_matrix.num_row_ = lp.num_row_;
  local_a_matrix.format_ = MatrixFormat::kColwise;
  if (ext_num_new_nz) {
    local_a_matrix.start_ = {ext_a_start, ext_a_start + ext_num_new_col};
    local_a_matrix.start_.resize(ext_num_new_col + 1);
    local_a_matrix.start_[ext_num_new_col] = ext_num_new_nz;
    local_a_matrix.index_ = {ext_a_index, ext_a_index + ext_num_new_nz};
    local_a_matrix.value_ = {ext_a_value, ext_a_value + ext_num_new_nz};
    // Assess the matrix rows
    return_status =
        interpretCallStatus(options_.log_options,
                            local_a_matrix.assess(options.log_options, "LP",
                                                  options.small_matrix_value,
                                                  options.large_matrix_value),
                            return_status, "assessMatrix");
    if (return_status == HighsStatus::kError) return return_status;
  } else {
    // No nonzeros so, whether the constraint matrix is column-wise or
    // row-wise, adding the empty matrix is trivial. Complete the
    // setup of an empty column-wise HighsSparseMatrix of the new
    // matrix columns
    local_a_matrix.start_.assign(ext_num_new_col + 1, 0);
  }
  // Append the columns to LP matrix
  lp.a_matrix_.addCols(local_a_matrix);
  if (lp_has_scaling) {
    // Extend the column scaling factors
    scale.col.resize(newNumCol);
    for (HighsInt iCol = 0; iCol < ext_num_new_col; iCol++)
      scale.col[lp.num_col_ + iCol] = 1.0;
    scale.num_col = newNumCol;
    // Apply the existing row scaling to the new columns
    local_a_matrix.applyRowScale(scale);
    // Consider applying column scaling to the new columns.
    local_a_matrix.considerColScaling(options.allowed_matrix_scale_factor,
                                      &scale.col[lp.num_col_]);
  }
  // Update the basis correponding to new nonbasic columns
  if (valid_basis) appendNonbasicColsToBasisInterface(ext_num_new_col);
  // Increase the number of columns in the LP
  lp.num_col_ += ext_num_new_col;
  assert(lpDimensionsOk("addCols", lp, options.log_options));

  // Deduce the consequences of adding new columns
  invalidateModelStatusSolutionAndInfo();

  // Determine any implications for simplex data
  ekk_instance_.addCols(lp, local_a_matrix);
  return return_status;
}

HighsStatus Highs::addRowsInterface(HighsInt ext_num_new_row,
                                    const double* ext_row_lower,
                                    const double* ext_row_upper,
                                    HighsInt ext_num_new_nz,
                                    const HighsInt* ext_ar_start,
                                    const HighsInt* ext_ar_index,
                                    const double* ext_ar_value) {
  // addRows is fundamentally different from addCols, since the new
  // matrix data are held row-wise, so we have to insert data into the
  // column-wise matrix of the LP.
  if (kExtendInvertWhenAddingRows) {
    if (ekk_instance_.status_.has_nla)
      ekk_instance_.debugNlaCheckInvert("Start of Highs::addRowsInterface",
                                        kHighsDebugLevelExpensive + 1);
  }
  HighsStatus return_status = HighsStatus::kOk;
  HighsOptions& options = options_;
  if (ext_num_new_row < 0) return HighsStatus::kError;
  if (ext_num_new_nz < 0) return HighsStatus::kError;
  if (ext_num_new_row == 0) return HighsStatus::kOk;
  if (ext_num_new_row > 0)
    if (isRowDataNull(options.log_options, ext_row_lower, ext_row_upper))
      return HighsStatus::kError;
  if (ext_num_new_nz > 0)
    if (isMatrixDataNull(options.log_options, ext_ar_start, ext_ar_index,
                         ext_ar_value))
      return HighsStatus::kError;

  HighsLp& lp = model_.lp_;
  HighsBasis& basis = basis_;
  HighsScale& scale = lp.scale_;
  bool& valid_basis = basis.valid;
  bool& lp_has_scaling = lp.scale_.has_scaling;

  // Check that if nonzeros are to be added then the model has a positive number
  // of columns
  if (lp.num_col_ <= 0 && ext_num_new_nz > 0) return HighsStatus::kError;

  // Record the new number of rows
  HighsInt newNumRow = lp.num_row_ + ext_num_new_row;

  HighsIndexCollection index_collection;
  index_collection.dimension_ = ext_num_new_row;
  index_collection.is_interval_ = true;
  index_collection.from_ = 0;
  index_collection.to_ = ext_num_new_row - 1;
  // Take a copy of the bounds that can be normalised
  std::vector<double> local_rowLower{ext_row_lower,
                                     ext_row_lower + ext_num_new_row};
  std::vector<double> local_rowUpper{ext_row_upper,
                                     ext_row_upper + ext_num_new_row};

  return_status = interpretCallStatus(
      options_.log_options,
      assessBounds(options, "Row", lp.num_row_, index_collection,
                   local_rowLower, local_rowUpper, options.infinite_bound),
      return_status, "assessBounds");
  if (return_status == HighsStatus::kError) return return_status;

  // Append the rows to the LP vectors
  appendRowsToLpVectors(lp, ext_num_new_row, local_rowLower, local_rowUpper);

  // Form a row-wise HighsSparseMatrix of the new matrix rows so that
  // is is easy to handle and, if there are nonzeros, it can be
  // normalised
  HighsSparseMatrix local_ar_matrix;
  local_ar_matrix.num_col_ = lp.num_col_;
  local_ar_matrix.num_row_ = ext_num_new_row;
  local_ar_matrix.format_ = MatrixFormat::kRowwise;
  if (ext_num_new_nz) {
    local_ar_matrix.start_ = {ext_ar_start, ext_ar_start + ext_num_new_row};
    local_ar_matrix.start_.resize(ext_num_new_row + 1);
    local_ar_matrix.start_[ext_num_new_row] = ext_num_new_nz;
    local_ar_matrix.index_ = {ext_ar_index, ext_ar_index + ext_num_new_nz};
    local_ar_matrix.value_ = {ext_ar_value, ext_ar_value + ext_num_new_nz};
    // Assess the matrix columns
    return_status =
        interpretCallStatus(options_.log_options,
                            local_ar_matrix.assess(options.log_options, "LP",
                                                   options.small_matrix_value,
                                                   options.large_matrix_value),
                            return_status, "assessMatrix");
    if (return_status == HighsStatus::kError) return return_status;
  } else {
    // No nonzeros so, whether the constraint matrix is row-wise or
    // column-wise, adding the empty matrix is trivial. Complete the
    // setup of an empty row-wise HighsSparseMatrix of the new matrix
    // rows
    local_ar_matrix.start_.assign(ext_num_new_row + 1, 0);
  }
  // Append the rows to LP matrix
  lp.a_matrix_.addRows(local_ar_matrix);
  if (lp_has_scaling) {
    // Extend the row scaling factors
    scale.row.resize(newNumRow);
    for (HighsInt iRow = 0; iRow < ext_num_new_row; iRow++)
      scale.row[lp.num_row_ + iRow] = 1.0;
    scale.num_row = newNumRow;
    // Apply the existing column scaling to the new rows
    local_ar_matrix.applyColScale(scale);
    // Consider applying row scaling to the new rows.
    local_ar_matrix.considerRowScaling(options.allowed_matrix_scale_factor,
                                       &scale.row[lp.num_row_]);
  }
  // Update the basis correponding to new basic rows
  if (valid_basis) appendBasicRowsToBasisInterface(ext_num_new_row);

  // Increase the number of rows in the LP
  lp.num_row_ += ext_num_new_row;
  assert(lpDimensionsOk("addRows", lp, options.log_options));

  // Deduce the consequences of adding new rows
  invalidateModelStatusSolutionAndInfo();
  // Determine any implications for simplex data
  ekk_instance_.addRows(lp, local_ar_matrix);
  return return_status;
}

void Highs::deleteColsInterface(HighsIndexCollection& index_collection) {
  HighsLp& lp = model_.lp_;
  HighsBasis& basis = basis_;
  lp.ensureColwise();

  // Keep a copy of the original number of columns to check whether
  // any columns have been removed, and if there is mask to be updated
  HighsInt original_num_col = lp.num_col_;

  deleteLpCols(lp, index_collection);
  assert(lp.num_col_ <= original_num_col);
  if (lp.num_col_ < original_num_col) {
    // Nontrivial deletion so reset the model_status and invalidate
    // the Highs basis
    model_status_ = HighsModelStatus::kNotset;
    basis.valid = false;
  }
  if (lp.scale_.has_scaling) {
    deleteScale(lp.scale_.col, index_collection);
    lp.scale_.col.resize(lp.num_col_);
    lp.scale_.num_col = lp.num_col_;
  }
  // Deduce the consequences of deleting columns
  invalidateModelStatusSolutionAndInfo();

  // Determine any implications for simplex data
  ekk_instance_.deleteCols(index_collection);

  if (index_collection.is_mask_) {
    // Set the mask values to indicate the new index value of the
    // remaining columns
    HighsInt new_col = 0;
    for (HighsInt col = 0; col < original_num_col; col++) {
      if (!index_collection.mask_[col]) {
        index_collection.mask_[col] = new_col;
        new_col++;
      } else {
        index_collection.mask_[col] = -1;
      }
    }
    assert(new_col == lp.num_col_);
  }
  assert(lpDimensionsOk("deleteCols", lp, options_.log_options));
}

void Highs::deleteRowsInterface(HighsIndexCollection& index_collection) {
  HighsLp& lp = model_.lp_;
  HighsBasis& basis = basis_;
  lp.ensureColwise();
  // Keep a copy of the original number of rows to check whether
  // any rows have been removed, and if there is mask to be updated
  HighsInt original_num_row = lp.num_row_;

  deleteLpRows(lp, index_collection);
  assert(lp.num_row_ <= original_num_row);
  if (lp.num_row_ < original_num_row) {
    // Nontrivial deletion so reset the model_status and invalidate
    // the Highs basis
    model_status_ = HighsModelStatus::kNotset;
    basis.valid = false;
  }
  if (lp.scale_.has_scaling) {
    deleteScale(lp.scale_.row, index_collection);
    lp.scale_.row.resize(lp.num_row_);
    lp.scale_.num_row = lp.num_row_;
  }
  // Deduce the consequences of deleting rows
  invalidateModelStatusSolutionAndInfo();

  // Determine any implications for simplex data
  ekk_instance_.deleteRows(index_collection);
  if (index_collection.is_mask_) {
    HighsInt new_row = 0;
    for (HighsInt row = 0; row < original_num_row; row++) {
      if (!index_collection.mask_[row]) {
        index_collection.mask_[row] = new_row;
        new_row++;
      } else {
        index_collection.mask_[row] = -1;
      }
    }
    assert(new_row == lp.num_row_);
  }
  assert(lpDimensionsOk("deleteRows", lp, options_.log_options));
}

void Highs::getColsInterface(const HighsIndexCollection& index_collection,
                             HighsInt& get_num_col, double* col_cost,
                             double* col_lower, double* col_upper,
                             HighsInt& get_num_nz, HighsInt* col_matrix_start,
                             HighsInt* col_matrix_index,
                             double* col_matrix_value) {
  HighsLp& lp = model_.lp_;
  // Ensure that the LP is column-wise
  lp.ensureColwise();
  assert(ok(index_collection));
  HighsInt from_k;
  HighsInt to_k;
  limits(index_collection, from_k, to_k);
  // Surely this is checked elsewhere
  assert(0 <= from_k && to_k < lp.num_col_);
  assert(from_k <= to_k);
  HighsInt out_from_col;
  HighsInt out_to_col;
  HighsInt in_from_col;
  HighsInt in_to_col = -1;
  HighsInt current_set_entry = 0;
  HighsInt col_dim = lp.num_col_;
  get_num_col = 0;
  get_num_nz = 0;
  for (HighsInt k = from_k; k <= to_k; k++) {
    updateOutInIndex(index_collection, out_from_col, out_to_col, in_from_col,
                     in_to_col, current_set_entry);
    assert(out_to_col < col_dim);
    assert(in_to_col < col_dim);
    for (HighsInt iCol = out_from_col; iCol <= out_to_col; iCol++) {
      if (col_cost != NULL) col_cost[get_num_col] = lp.col_cost_[iCol];
      if (col_lower != NULL) col_lower[get_num_col] = lp.col_lower_[iCol];
      if (col_upper != NULL) col_upper[get_num_col] = lp.col_upper_[iCol];
      if (col_matrix_start != NULL)
        col_matrix_start[get_num_col] = get_num_nz + lp.a_matrix_.start_[iCol] -
                                        lp.a_matrix_.start_[out_from_col];
      get_num_col++;
    }
    for (HighsInt iEl = lp.a_matrix_.start_[out_from_col];
         iEl < lp.a_matrix_.start_[out_to_col + 1]; iEl++) {
      if (col_matrix_index != NULL)
        col_matrix_index[get_num_nz] = lp.a_matrix_.index_[iEl];
      if (col_matrix_value != NULL)
        col_matrix_value[get_num_nz] = lp.a_matrix_.value_[iEl];
      get_num_nz++;
    }
    if (out_to_col == col_dim - 1 || in_to_col == col_dim - 1) break;
  }
}

void Highs::getRowsInterface(const HighsIndexCollection& index_collection,
                             HighsInt& get_num_row, double* row_lower,
                             double* row_upper, HighsInt& get_num_nz,
                             HighsInt* row_matrix_start,
                             HighsInt* row_matrix_index,
                             double* row_matrix_value) {
  HighsLp& lp = model_.lp_;
  // Ensure that the LP is column-wise
  lp.ensureColwise();
  assert(ok(index_collection));
  HighsInt from_k;
  HighsInt to_k;
  limits(index_collection, from_k, to_k);
  // Surely this is checked elsewhere
  assert(0 <= from_k && to_k < lp.num_row_);
  assert(from_k <= to_k);
  // "Out" means not in the set to be extrated
  // "In" means in the set to be extrated
  HighsInt out_from_row;
  HighsInt out_to_row;
  HighsInt in_from_row;
  HighsInt in_to_row = -1;
  HighsInt current_set_entry = 0;
  HighsInt row_dim = lp.num_row_;
  // Ensure that the LP is column-wise
  lp.ensureColwise();
  // Set up a row mask so that entries to be got from the column-wise
  // matrix can be identified and have their correct row index.
  vector<HighsInt> new_index;
  new_index.resize(lp.num_row_);

  get_num_row = 0;
  get_num_nz = 0;
  if (!index_collection.is_mask_) {
    out_to_row = -1;
    current_set_entry = 0;
    for (HighsInt k = from_k; k <= to_k; k++) {
      updateOutInIndex(index_collection, in_from_row, in_to_row, out_from_row,
                       out_to_row, current_set_entry);
      if (k == from_k) {
        // Account for any initial rows not being extracted
        for (HighsInt iRow = 0; iRow < in_from_row; iRow++) {
          new_index[iRow] = -1;
        }
      }
      for (HighsInt iRow = in_from_row; iRow <= in_to_row; iRow++) {
        new_index[iRow] = get_num_row;
        get_num_row++;
      }
      for (HighsInt iRow = out_from_row; iRow <= out_to_row; iRow++) {
        new_index[iRow] = -1;
      }
      if (out_to_row >= row_dim - 1) break;
    }
  } else {
    for (HighsInt iRow = 0; iRow < lp.num_row_; iRow++) {
      if (index_collection.mask_[iRow]) {
        new_index[iRow] = get_num_row;
        get_num_row++;
      } else {
        new_index[iRow] = -1;
      }
    }
  }

  // Bail out if no rows are to be extracted
  if (get_num_row == 0) return;

  for (HighsInt iRow = 0; iRow < lp.num_row_; iRow++) {
    HighsInt new_iRow = new_index[iRow];
    if (new_iRow >= 0) {
      assert(new_iRow < get_num_row);
      if (row_lower != NULL) row_lower[new_iRow] = lp.row_lower_[iRow];
      if (row_upper != NULL) row_upper[new_iRow] = lp.row_upper_[iRow];
    }
  }
  // bail out if no matrix is to be extracted
  const bool extract_start = row_matrix_start != NULL;
  const bool extract_index = row_matrix_index != NULL;
  const bool extract_value = row_matrix_value != NULL;
  const bool extract_matrix = extract_index || extract_value;
  // Someone might just want the values, but to get them makes use of
  // the starts so tough!
  if (!extract_start) return;
  // Allocate an array of lengths for the row-wise matrix to be extracted
  vector<HighsInt> row_matrix_length;
  row_matrix_length.assign(get_num_row, 0);
  // Identify the lengths of the rows in the row-wise matrix to be extracted
  for (HighsInt col = 0; col < lp.num_col_; col++) {
    for (HighsInt iEl = lp.a_matrix_.start_[col];
         iEl < lp.a_matrix_.start_[col + 1]; iEl++) {
      HighsInt iRow = lp.a_matrix_.index_[iEl];
      HighsInt new_iRow = new_index[iRow];
      if (new_iRow >= 0) row_matrix_length[new_iRow]++;
    }
  }
  row_matrix_start[0] = 0;
  for (HighsInt iRow = 0; iRow < get_num_row - 1; iRow++) {
    row_matrix_start[iRow + 1] =
        row_matrix_start[iRow] + row_matrix_length[iRow];
    row_matrix_length[iRow] = row_matrix_start[iRow];
  }
  HighsInt iRow = get_num_row - 1;
  get_num_nz = row_matrix_start[iRow] + row_matrix_length[iRow];
  // Bail out if matrix indices and values are not required
  if (!extract_matrix) return;
  row_matrix_length[iRow] = row_matrix_start[iRow];
  // Fill the row-wise matrix with indices and values
  for (HighsInt col = 0; col < lp.num_col_; col++) {
    for (HighsInt iEl = lp.a_matrix_.start_[col];
         iEl < lp.a_matrix_.start_[col + 1]; iEl++) {
      HighsInt iRow = lp.a_matrix_.index_[iEl];
      HighsInt new_iRow = new_index[iRow];
      if (new_iRow >= 0) {
        HighsInt row_iEl = row_matrix_length[new_iRow];
        if (extract_index) row_matrix_index[row_iEl] = col;
        if (extract_value) row_matrix_value[row_iEl] = lp.a_matrix_.value_[iEl];
        row_matrix_length[new_iRow]++;
      }
    }
  }
}

void Highs::getCoefficientInterface(const HighsInt ext_row,
                                    const HighsInt ext_col, double& value) {
  HighsLp& lp = model_.lp_;
  assert(0 <= ext_row && ext_row < lp.num_row_);
  assert(0 <= ext_col && ext_col < lp.num_col_);
  value = 0;
  // Ensure that the LP is column-wise
  lp.ensureColwise();
  for (HighsInt el = lp.a_matrix_.start_[ext_col];
       el < lp.a_matrix_.start_[ext_col + 1]; el++) {
    if (lp.a_matrix_.index_[el] == ext_row) {
      value = lp.a_matrix_.value_[el];
      break;
    }
  }
}

HighsStatus Highs::changeIntegralityInterface(
    HighsIndexCollection& index_collection, const HighsVarType* integrality) {
  HighsInt num_integrality = dataSize(index_collection);
  // If a non-positive number of integrality (may) need changing nothing needs
  // to be done
  if (num_integrality <= 0) return HighsStatus::kOk;
  if (highsVarTypeUserDataNotNull(options_.log_options, integrality,
                                  "column integrality"))
    return HighsStatus::kError;
  // Take a copy of the integrality that can be normalised
  std::vector<HighsVarType> local_integrality{integrality,
                                              integrality + num_integrality};
  // If changing the integrality for a set of columns, verify that the
  // set entries are in ascending order
  if (index_collection.is_set_)
    assert(increasingSetOk(index_collection.set_, 0,
                           index_collection.dimension_, true));
  HighsStatus return_status = HighsStatus::kOk;
  HighsStatus call_status;
  changeLpIntegrality(model_.lp_, index_collection, local_integrality);
  // Deduce the consequences of new integrality
  invalidateModelStatus();
  return HighsStatus::kOk;
}

HighsStatus Highs::changeCostsInterface(HighsIndexCollection& index_collection,
                                        const double* cost) {
  HighsInt num_cost = dataSize(index_collection);
  // If a non-positive number of costs (may) need changing nothing needs to be
  // done
  if (num_cost <= 0) return HighsStatus::kOk;
  if (doubleUserDataNotNull(options_.log_options, cost, "column costs"))
    return HighsStatus::kError;
  // Take a copy of the cost that can be normalised
  std::vector<double> local_colCost{cost, cost + num_cost};
  HighsStatus return_status = HighsStatus::kOk;
  return_status =
      interpretCallStatus(options_.log_options,
                          assessCosts(options_, 0, index_collection,
                                      local_colCost, options_.infinite_cost),
                          return_status, "assessCosts");
  if (return_status == HighsStatus::kError) return return_status;
  changeLpCosts(model_.lp_, index_collection, local_colCost);
  // Deduce the consequences of new costs
  invalidateModelStatusSolutionAndInfo();
  // Determine any implications for simplex data
  ekk_instance_.updateStatus(LpAction::kNewCosts);
  return HighsStatus::kOk;
}

HighsStatus Highs::changeColBoundsInterface(
    HighsIndexCollection& index_collection, const double* col_lower,
    const double* col_upper) {
  HighsInt num_col_bounds = dataSize(index_collection);
  // If a non-positive number of costs (may) need changing nothing needs to be
  // done
  if (num_col_bounds <= 0) return HighsStatus::kOk;
  bool null_data = false;
  null_data = doubleUserDataNotNull(options_.log_options, col_lower,
                                    "column lower bounds") ||
              null_data;
  null_data = doubleUserDataNotNull(options_.log_options, col_upper,
                                    "column upper bounds") ||
              null_data;
  if (null_data) return HighsStatus::kError;
  // Take a copy of the cost that can be normalised
  std::vector<double> local_colLower{col_lower, col_lower + num_col_bounds};
  std::vector<double> local_colUpper{col_upper, col_upper + num_col_bounds};
  // If changing the bounds for a set of columns, ensure that the
  // set and data are in ascending order
  if (index_collection.is_set_)
    sortSetData(index_collection.set_num_entries_, index_collection.set_,
                col_lower, col_upper, NULL, &local_colLower[0],
                &local_colUpper[0], NULL);
  HighsStatus return_status = HighsStatus::kOk;
  return_status = interpretCallStatus(
      options_.log_options,
      assessBounds(options_, "col", 0, index_collection, local_colLower,
                   local_colUpper, options_.infinite_bound),
      return_status, "assessBounds");
  if (return_status == HighsStatus::kError) return return_status;

  HighsStatus call_status;
  changeLpColBounds(model_.lp_, index_collection, local_colLower,
                    local_colUpper);
  // Update HiGHS basis status and (any) simplex move status of
  // nonbasic variables whose bounds have changed
  setNonbasicStatusInterface(index_collection, true);
  // Deduce the consequences of new col bounds
  invalidateModelStatusSolutionAndInfo();
  // Determine any implications for simplex data
  ekk_instance_.updateStatus(LpAction::kNewBounds);
  return HighsStatus::kOk;
}

HighsStatus Highs::changeRowBoundsInterface(
    HighsIndexCollection& index_collection, const double* lower,
    const double* upper) {
  HighsInt num_row_bounds = dataSize(index_collection);
  // If a non-positive number of costs (may) need changing nothing needs to be
  // done
  if (num_row_bounds <= 0) return HighsStatus::kOk;
  bool null_data = false;
  null_data =
      doubleUserDataNotNull(options_.log_options, lower, "row lower bounds") ||
      null_data;
  null_data =
      doubleUserDataNotNull(options_.log_options, upper, "row upper bounds") ||
      null_data;
  if (null_data) return HighsStatus::kError;
  // Take a copy of the cost that can be normalised
  std::vector<double> local_rowLower{lower, lower + num_row_bounds};
  std::vector<double> local_rowUpper{upper, upper + num_row_bounds};
  // If changing the bounds for a set of rows, ensure that the
  // set and data are in ascending order
  if (index_collection.is_set_)
    sortSetData(index_collection.set_num_entries_, index_collection.set_, lower,
                upper, NULL, &local_rowLower[0], &local_rowUpper[0], NULL);
  HighsStatus return_status = HighsStatus::kOk;
  return_status = interpretCallStatus(
      options_.log_options,
      assessBounds(options_, "row", 0, index_collection, local_rowLower,
                   local_rowUpper, options_.infinite_bound),
      return_status, "assessBounds");
  if (return_status == HighsStatus::kError) return return_status;

  HighsStatus call_status;
  changeLpRowBounds(model_.lp_, index_collection, local_rowLower,
                    local_rowUpper);
  // Update HiGHS basis status and (any) simplex move status of
  // nonbasic variables whose bounds have changed
  setNonbasicStatusInterface(index_collection, false);
  // Deduce the consequences of new row bounds
  invalidateModelStatusSolutionAndInfo();
  // Determine any implications for simplex data
  ekk_instance_.updateStatus(LpAction::kNewBounds);
  return HighsStatus::kOk;
}

// Change a single coefficient in the matrix
void Highs::changeCoefficientInterface(const HighsInt ext_row,
                                       const HighsInt ext_col,
                                       const double ext_new_value) {
  HighsLp& lp = model_.lp_;
  // Ensure that the LP is column-wise
  lp.ensureColwise();
  assert(0 <= ext_row && ext_row < lp.num_row_);
  assert(0 <= ext_col && ext_col < lp.num_col_);
  const bool zero_new_value =
      std::fabs(ext_new_value) <= options_.small_matrix_value;
  changeLpMatrixCoefficient(lp, ext_row, ext_col, ext_new_value,
                            zero_new_value);
  // Deduce the consequences of a changed element
  //
  // ToDo: Can do something more intelligent if element is in nonbasic column.
  // Otherwise, treat it as if it's a new row
  invalidateModelStatusSolutionAndInfo();

  // Determine any implications for simplex data
  ekk_instance_.updateStatus(LpAction::kNewRows);
}

HighsStatus Highs::scaleColInterface(const HighsInt col,
                                     const double scale_value) {
  HighsStatus return_status = HighsStatus::kOk;
  HighsLp& lp = model_.lp_;
  HighsBasis& basis = basis_;
  HighsSimplexStatus& simplex_status = ekk_instance_.status_;

  // Ensure that the LP is column-wise
  lp.ensureColwise();
  if (col < 0) return HighsStatus::kError;
  if (col >= lp.num_col_) return HighsStatus::kError;
  if (!scale_value) return HighsStatus::kError;

  return_status = interpretCallStatus(options_.log_options,
                                      applyScalingToLpCol(lp, col, scale_value),
                                      return_status, "applyScalingToLpCol");
  if (return_status == HighsStatus::kError) return return_status;

  if (scale_value < 0 && basis.valid) {
    // Negative, so flip any nonbasic status
    if (basis.col_status[col] == HighsBasisStatus::kLower) {
      basis.col_status[col] = HighsBasisStatus::kUpper;
    } else if (basis.col_status[col] == HighsBasisStatus::kUpper) {
      basis.col_status[col] = HighsBasisStatus::kLower;
    }
  }
  if (simplex_status.initialised_for_solve) {
    SimplexBasis& simplex_basis = ekk_instance_.basis_;
    if (scale_value < 0 && simplex_status.has_basis) {
      // Negative, so flip any nonbasic status
      if (simplex_basis.nonbasicMove_[col] == kNonbasicMoveUp) {
        simplex_basis.nonbasicMove_[col] = kNonbasicMoveDn;
      } else if (simplex_basis.nonbasicMove_[col] == kNonbasicMoveDn) {
        simplex_basis.nonbasicMove_[col] = kNonbasicMoveUp;
      }
    }
  }
  // Deduce the consequences of a scaled column
  invalidateModelStatusSolutionAndInfo();

  // Determine any implications for simplex data
  ekk_instance_.updateStatus(LpAction::kScaledCol);
  return HighsStatus::kOk;
}

HighsStatus Highs::scaleRowInterface(const HighsInt row,
                                     const double scale_value) {
  HighsStatus return_status = HighsStatus::kOk;
  HighsLp& lp = model_.lp_;
  HighsBasis& basis = basis_;
  HighsSimplexStatus& simplex_status = ekk_instance_.status_;

  // Ensure that the LP is column-wise
  lp.ensureColwise();

  if (row < 0) return HighsStatus::kError;
  if (row >= lp.num_row_) return HighsStatus::kError;
  if (!scale_value) return HighsStatus::kError;

  return_status = interpretCallStatus(options_.log_options,
                                      applyScalingToLpRow(lp, row, scale_value),
                                      return_status, "applyScalingToLpRow");
  if (return_status == HighsStatus::kError) return return_status;

  if (scale_value < 0 && basis.valid) {
    // Negative, so flip any nonbasic status
    if (basis.row_status[row] == HighsBasisStatus::kLower) {
      basis.row_status[row] = HighsBasisStatus::kUpper;
    } else if (basis.row_status[row] == HighsBasisStatus::kUpper) {
      basis.row_status[row] = HighsBasisStatus::kLower;
    }
  }
  if (simplex_status.initialised_for_solve) {
    SimplexBasis& simplex_basis = ekk_instance_.basis_;
    if (scale_value < 0 && simplex_status.has_basis) {
      // Negative, so flip any nonbasic status
      const HighsInt var = lp.num_col_ + row;
      if (simplex_basis.nonbasicMove_[var] == kNonbasicMoveUp) {
        simplex_basis.nonbasicMove_[var] = kNonbasicMoveDn;
      } else if (simplex_basis.nonbasicMove_[var] == kNonbasicMoveDn) {
        simplex_basis.nonbasicMove_[var] = kNonbasicMoveUp;
      }
    }
  }
  // Deduce the consequences of a scaled row
  invalidateModelStatusSolutionAndInfo();

  // Determine any implications for simplex data
  ekk_instance_.updateStatus(LpAction::kScaledRow);
  return HighsStatus::kOk;
}

void Highs::setNonbasicStatusInterface(
    const HighsIndexCollection& index_collection, const bool columns) {
  HighsBasis& highs_basis = basis_;
  if (!highs_basis.valid) return;
  const bool has_simplex_basis = ekk_instance_.status_.has_basis;
  SimplexBasis& simplex_basis = ekk_instance_.basis_;
  HighsLp& lp = model_.lp_;

  assert(ok(index_collection));
  HighsInt from_k;
  HighsInt to_k;
  limits(index_collection, from_k, to_k);
  HighsInt ix_dim;
  if (columns) {
    ix_dim = lp.num_col_;
  } else {
    ix_dim = lp.num_row_;
  }
  // Surely this is checked elsewhere
  assert(0 <= from_k && to_k < ix_dim);
  assert(from_k <= to_k);
  HighsInt set_from_ix;
  HighsInt set_to_ix;
  HighsInt ignore_from_ix;
  HighsInt ignore_to_ix = -1;
  HighsInt current_set_entry = 0;
  // Given a basic-nonbasic partition, all status settings are defined
  // by the bounds unless boxed, in which case any definitive (ie not
  // just kNonbasic) existing status is retained. Otherwise, set to
  // bound nearer to zero
  for (HighsInt k = from_k; k <= to_k; k++) {
    updateOutInIndex(index_collection, set_from_ix, set_to_ix, ignore_from_ix,
                     ignore_to_ix, current_set_entry);
    assert(set_to_ix < ix_dim);
    assert(ignore_to_ix < ix_dim);
    if (columns) {
      for (HighsInt iCol = set_from_ix; iCol <= set_to_ix; iCol++) {
        if (highs_basis.col_status[iCol] == HighsBasisStatus::kBasic) continue;
        // Nonbasic column
        double lower = lp.col_lower_[iCol];
        double upper = lp.col_upper_[iCol];
        HighsBasisStatus status = highs_basis.col_status[iCol];
        HighsInt move = kIllegalMoveValue;
        if (lower == upper) {
          if (status == HighsBasisStatus::kNonbasic)
            status = HighsBasisStatus::kLower;
          move = kNonbasicMoveZe;
        } else if (!highs_isInfinity(-lower)) {
          // Finite lower bound so boxed or lower
          if (!highs_isInfinity(upper)) {
            // Finite upper bound so boxed
            if (status == HighsBasisStatus::kNonbasic) {
              // No definitive status, so set to bound nearer to zero
              if (fabs(lower) < fabs(upper)) {
                status = HighsBasisStatus::kLower;
                move = kNonbasicMoveUp;
              } else {
                status = HighsBasisStatus::kUpper;
                move = kNonbasicMoveDn;
              }
            } else if (status == HighsBasisStatus::kLower) {
              move = kNonbasicMoveUp;
            } else {
              move = kNonbasicMoveDn;
            }
          } else {
            // Lower (since upper bound is infinite)
            status = HighsBasisStatus::kLower;
            move = kNonbasicMoveUp;
          }
        } else if (!highs_isInfinity(upper)) {
          // Upper
          status = HighsBasisStatus::kUpper;
          move = kNonbasicMoveDn;
        } else {
          // FREE
          status = HighsBasisStatus::kZero;
          move = kNonbasicMoveZe;
        }
        highs_basis.col_status[iCol] = status;
        if (has_simplex_basis) {
          assert(move != kIllegalMoveValue);
          simplex_basis.nonbasicFlag_[iCol] = kNonbasicFlagTrue;
          simplex_basis.nonbasicMove_[iCol] = move;
        }
      }
    } else {
      for (HighsInt iRow = set_from_ix; iRow <= set_to_ix; iRow++) {
        if (highs_basis.row_status[iRow] == HighsBasisStatus::kBasic) continue;
        // Nonbasic column
        double lower = lp.row_lower_[iRow];
        double upper = lp.row_upper_[iRow];
        HighsBasisStatus status = highs_basis.row_status[iRow];
        HighsInt move = kIllegalMoveValue;
        if (lower == upper) {
          if (status == HighsBasisStatus::kNonbasic)
            status = HighsBasisStatus::kLower;
          move = kNonbasicMoveZe;
        } else if (!highs_isInfinity(-lower)) {
          // Finite lower bound so boxed or lower
          if (!highs_isInfinity(upper)) {
            // Finite upper bound so boxed
            if (status == HighsBasisStatus::kNonbasic) {
              // No definitive status, so set to bound nearer to zero
              if (fabs(lower) < fabs(upper)) {
                status = HighsBasisStatus::kLower;
                move = kNonbasicMoveDn;
              } else {
                status = HighsBasisStatus::kUpper;
                move = kNonbasicMoveUp;
              }
            } else if (status == HighsBasisStatus::kLower) {
              move = kNonbasicMoveDn;
            } else {
              move = kNonbasicMoveUp;
            }
          } else {
            // Lower (since upper bound is infinite)
            status = HighsBasisStatus::kLower;
            move = kNonbasicMoveDn;
          }
        } else if (!highs_isInfinity(upper)) {
          // Upper
          status = HighsBasisStatus::kUpper;
          move = kNonbasicMoveUp;
        } else {
          // FREE
          status = HighsBasisStatus::kZero;
          move = kNonbasicMoveZe;
        }
        highs_basis.row_status[iRow] = status;
        if (has_simplex_basis) {
          assert(move != kIllegalMoveValue);
          simplex_basis.nonbasicFlag_[lp.num_col_ + iRow] = kNonbasicFlagTrue;
          simplex_basis.nonbasicMove_[lp.num_col_ + iRow] = move;
        }
      }
    }
    if (ignore_to_ix >= ix_dim - 1) break;
  }
}

void Highs::appendNonbasicColsToBasisInterface(const HighsInt ext_num_new_col) {
  HighsBasis& highs_basis = basis_;
  if (!highs_basis.valid) return;
  const bool has_simplex_basis = ekk_instance_.status_.has_basis;
  SimplexBasis& simplex_basis = ekk_instance_.basis_;
  HighsLp& lp = model_.lp_;

  // Add nonbasic structurals
  if (ext_num_new_col == 0) return;
  HighsInt newNumCol = lp.num_col_ + ext_num_new_col;
  HighsInt newNumTot = newNumCol + lp.num_row_;
  highs_basis.col_status.resize(newNumCol);
  if (has_simplex_basis) {
    simplex_basis.nonbasicFlag_.resize(newNumTot);
    simplex_basis.nonbasicMove_.resize(newNumTot);
    // Shift the row data in basicIndex, nonbasicFlag and nonbasicMove if
    // necessary
    for (HighsInt iRow = lp.num_row_ - 1; iRow >= 0; iRow--) {
      HighsInt iCol = simplex_basis.basicIndex_[iRow];
      if (iCol >= lp.num_col_) {
        // This basic variable is a row, so shift its index
        simplex_basis.basicIndex_[iRow] += ext_num_new_col;
      }
      simplex_basis.nonbasicFlag_[newNumCol + iRow] =
          simplex_basis.nonbasicFlag_[lp.num_col_ + iRow];
      simplex_basis.nonbasicMove_[newNumCol + iRow] =
          simplex_basis.nonbasicMove_[lp.num_col_ + iRow];
    }
  }
  // Make any new columns nonbasic
  for (HighsInt iCol = lp.num_col_; iCol < newNumCol; iCol++) {
    double lower = lp.col_lower_[iCol];
    double upper = lp.col_upper_[iCol];
    HighsBasisStatus status = HighsBasisStatus::kNonbasic;
    HighsInt move = kIllegalMoveValue;
    if (lower == upper) {
      // Fixed
      status = HighsBasisStatus::kLower;
      move = kNonbasicMoveZe;
    } else if (!highs_isInfinity(-lower)) {
      // Finite lower bound so boxed or lower
      if (!highs_isInfinity(upper)) {
        // Finite upper bound so boxed
        if (fabs(lower) < fabs(upper)) {
          status = HighsBasisStatus::kLower;
          move = kNonbasicMoveUp;
        } else {
          status = HighsBasisStatus::kUpper;
          move = kNonbasicMoveDn;
        }
      } else {
        // Lower (since upper bound is infinite)
        status = HighsBasisStatus::kLower;
        move = kNonbasicMoveUp;
      }
    } else if (!highs_isInfinity(upper)) {
      // Upper
      status = HighsBasisStatus::kUpper;
      move = kNonbasicMoveDn;
    } else {
      // FREE
      status = HighsBasisStatus::kZero;
      move = kNonbasicMoveZe;
    }
    assert(status != HighsBasisStatus::kNonbasic);
    highs_basis.col_status[iCol] = status;
    if (has_simplex_basis) {
      assert(move != kIllegalMoveValue);
      simplex_basis.nonbasicFlag_[iCol] = kNonbasicFlagTrue;
      simplex_basis.nonbasicMove_[iCol] = move;
    }
  }
}

void Highs::appendBasicRowsToBasisInterface(const HighsInt ext_num_new_row) {
  HighsBasis& highs_basis = basis_;
  if (!highs_basis.valid) return;
  const bool has_simplex_basis = ekk_instance_.status_.has_basis;
  SimplexBasis& simplex_basis = ekk_instance_.basis_;
  HighsLp& lp = model_.lp_;
  // Add basic logicals
  if (ext_num_new_row == 0) return;
  // Add the new rows to the Highs basis
  HighsInt newNumRow = lp.num_row_ + ext_num_new_row;
  highs_basis.row_status.resize(newNumRow);
  for (HighsInt iRow = lp.num_row_; iRow < newNumRow; iRow++)
    highs_basis.row_status[iRow] = HighsBasisStatus::kBasic;
  if (has_simplex_basis) {
    // Add the new rows to the simplex basis
    HighsInt newNumTot = lp.num_col_ + newNumRow;
    simplex_basis.nonbasicFlag_.resize(newNumTot);
    simplex_basis.nonbasicMove_.resize(newNumTot);
    simplex_basis.basicIndex_.resize(newNumRow);
    for (HighsInt iRow = lp.num_row_; iRow < newNumRow; iRow++) {
      simplex_basis.nonbasicFlag_[lp.num_col_ + iRow] = kNonbasicFlagFalse;
      simplex_basis.nonbasicMove_[lp.num_col_ + iRow] = 0;
      simplex_basis.basicIndex_[iRow] = lp.num_col_ + iRow;
    }
  }
}

// Get the basic variables, performing INVERT if necessary
HighsStatus Highs::getBasicVariablesInterface(HighsInt* basic_variables) {
  HighsStatus return_status = HighsStatus::kOk;
  HighsLp& lp = model_.lp_;
  HighsLp& ekk_lp = ekk_instance_.lp_;
  HighsInt num_row = lp.num_row_;
  HighsInt num_col = lp.num_col_;
  HighsSimplexStatus& ekk_status = ekk_instance_.status_;
  // For an LP with no rows the solution is vacuous
  if (num_row == 0) return return_status;
  if (!basis_.valid) {
    highsLogUser(options_.log_options, HighsLogType::kError,
                 "getBasicVariables called without a HiGHS basis\n");
    return HighsStatus::kError;
  }
  if (!ekk_status.has_invert) {
    // The LP has no invert to use, so have to set one up, but only
    // for the current basis, so return_value is the rank deficiency.
    //
    // Create a HighsLpSolverObject
    HighsLpSolverObject solver_object(lp, basis_, solution_, info_,
                                      ekk_instance_, options_, timer_);
    const bool only_from_known_basis = true;
    return_status = interpretCallStatus(
        options_.log_options,
        formSimplexLpBasisAndFactor(solver_object, only_from_known_basis),
        return_status, "formSimplexLpBasisAndFactor");
    if (return_status != HighsStatus::kOk) return return_status;
  }
  assert(ekk_status.has_invert);

  for (HighsInt row = 0; row < num_row; row++) {
    HighsInt var = ekk_instance_.basis_.basicIndex_[row];
    if (var < num_col) {
      basic_variables[row] = var;
    } else {
      basic_variables[row] = -(1 + var - num_col);
    }
  }
  return return_status;
}

// Solve (transposed) system involving the basis matrix

HighsStatus Highs::basisSolveInterface(const vector<double>& rhs,
                                       double* solution_vector,
                                       HighsInt* solution_num_nz,
                                       HighsInt* solution_indices,
                                       bool transpose) {
  HighsStatus return_status = HighsStatus::kOk;
  HighsLp& lp = model_.lp_;
  HighsInt num_row = lp.num_row_;
  HighsInt num_col = lp.num_col_;
  // For an LP with no rows the solution is vacuous
  if (num_row == 0) return return_status;
  // EKK must have an INVERT, but simplex NLA may need the pointer to
  // its LP to be refreshed so that it can use its scale factors
  assert(ekk_instance_.status_.has_invert);
  // Reset the simplex NLA LP and scale pointers for the unscaled LP
  ekk_instance_.setNlaPointersForLpAndScale(lp);
  assert(!lp.is_moved_);
  // Set up solve vector with suitably scaled RHS
  HVector solve_vector;
  solve_vector.setup(num_row);
  solve_vector.clear();
  HighsScale& scale = lp.scale_;
  HighsInt rhs_num_nz = 0;
  for (HighsInt iRow = 0; iRow < num_row; iRow++) {
    if (rhs[iRow]) {
      solve_vector.index[rhs_num_nz++] = iRow;
      solve_vector.array[iRow] = rhs[iRow];
    }
  }
  solve_vector.count = rhs_num_nz;
  //
  // Note that solve_vector.count is just used to determine whether
  // hyper-sparse solves should be used. The indices of the nonzeros
  // in the solution are always accumulated. There's no switch (such
  // as setting solve_vector.count = num_row+1) to not do this.
  //
  // Get expected_density from analysis during simplex solve.
  const double expected_density = 1;
  if (transpose) {
    ekk_instance_.btran(solve_vector, expected_density);
  } else {
    ekk_instance_.ftran(solve_vector, expected_density);
  }
  // Extract the solution
  if (solution_indices == NULL) {
    // Nonzeros in the solution not required
    if (solve_vector.count > num_row) {
      // Solution nonzeros not known
      for (HighsInt iRow = 0; iRow < num_row; iRow++) {
        solution_vector[iRow] = solve_vector.array[iRow];
      }
    } else {
      // Solution nonzeros are known
      for (HighsInt iRow = 0; iRow < num_row; iRow++) solution_vector[iRow] = 0;
      for (HighsInt iX = 0; iX < solve_vector.count; iX++) {
        HighsInt iRow = solve_vector.index[iX];
        solution_vector[iRow] = solve_vector.array[iRow];
      }
    }
  } else {
    // Nonzeros in the solution are required
    if (solve_vector.count > num_row) {
      // Solution nonzeros not known
      solution_num_nz = 0;
      for (HighsInt iRow = 0; iRow < num_row; iRow++) {
        solution_vector[iRow] = 0;
        if (solve_vector.array[iRow]) {
          solution_vector[iRow] = solve_vector.array[iRow];
          solution_indices[*solution_num_nz++] = iRow;
        }
      }
    } else {
      // Solution nonzeros are known
      for (HighsInt iRow = 0; iRow < num_row; iRow++) solution_vector[iRow] = 0;
      for (HighsInt iX = 0; iX < solve_vector.count; iX++) {
        HighsInt iRow = solve_vector.index[iX];
        solution_vector[iRow] = solve_vector.array[iRow];
        solution_indices[iX] = iRow;
      }
      *solution_num_nz = solve_vector.count;
    }
  }
  return HighsStatus::kOk;
}

HighsStatus Highs::setHotStartInterface(const HotStart& hot_start) {
  assert(hot_start.valid);
  HighsLp& lp = model_.lp_;
  HighsInt num_col = lp.num_col_;
  HighsInt num_row = lp.num_row_;
  HighsInt num_tot = num_col + num_row;
  bool hot_start_ok = true;
  HighsInt hot_start_num_row;
  HighsInt hot_start_num_tot;
  hot_start_num_row = (int)hot_start.refactor_info.pivot_row.size();
  if (hot_start_num_row != num_row) {
    hot_start_ok = false;
    highsLogDev(options_.log_options, HighsLogType::kError,
                "setHotStart: refactor_info.pivot_row.size of %d and LP with "
                "%d rows are incompatible\n",
                (int)hot_start_num_row, (int)num_row);
  }
  hot_start_num_row = (int)hot_start.refactor_info.pivot_var.size();
  if (hot_start_num_row != num_row) {
    hot_start_ok = false;
    highsLogDev(options_.log_options, HighsLogType::kError,
                "setHotStart: refactor_info.pivot_var.size of %d and LP with "
                "%d rows are incompatible\n",
                (int)hot_start_num_row, (int)num_row);
  }
  hot_start_num_row = (int)hot_start.refactor_info.pivot_type.size();
  if (hot_start_num_row != num_row) {
    hot_start_ok = false;
    highsLogDev(options_.log_options, HighsLogType::kError,
                "setHotStart: refactor_info.pivot_type.size of %d and LP with "
                "%d rows are incompatible\n",
                (int)hot_start_num_row, (int)num_row);
  }
  hot_start_num_tot = (int)hot_start.nonbasicMove.size();
  if (hot_start_num_tot != num_tot) {
    hot_start_ok = false;
    highsLogDev(options_.log_options, HighsLogType::kError,
                "setHotStart: nonbasicMove.size of %d and LP with %d "
                "columns+rows are incompatible\n",
                (int)hot_start_num_tot, (int)num_tot);
  }
  if (!hot_start_ok) {
    highsLogUser(options_.log_options, HighsLogType::kError,
                 "setHotStart called with incompatible data\n");
    return HighsStatus::kError;
  }
  // Set up the HiGHS and Ekk basis
  vector<int8_t>& nonbasicFlag = ekk_instance_.basis_.nonbasicFlag_;
  vector<int8_t>& nonbasicMove = ekk_instance_.basis_.nonbasicMove_;
  vector<HighsInt>& basicIndex = ekk_instance_.basis_.basicIndex_;
  basis_.col_status.assign(num_col, HighsBasisStatus::kBasic);
  basis_.row_status.resize(num_row, HighsBasisStatus::kBasic);
  basicIndex = hot_start.refactor_info.pivot_var;
  nonbasicFlag.assign(num_tot, kNonbasicFlagTrue);
  ekk_instance_.basis_.nonbasicMove_ = hot_start.nonbasicMove;
  ekk_instance_.hot_start_.refactor_info = hot_start.refactor_info;
  // Complete nonbasicFlag by setting the entries for basic variables
  for (HighsInt iRow = 0; iRow < num_row; iRow++)
    nonbasicFlag[basicIndex[iRow]] = kNonbasicFlagFalse;
  // Complete the HiGHS basis column status and adjust nonbasicMove
  // for nonbasic variables
  for (HighsInt iCol = 0; iCol < num_col; iCol++) {
    if (nonbasicFlag[iCol] == kNonbasicFlagFalse) continue;
    const double lower = lp.col_lower_[iCol];
    const double upper = lp.col_upper_[iCol];
    HighsBasisStatus status = HighsBasisStatus::kNonbasic;
    HighsInt move = kIllegalMoveValue;
    if (lower == upper) {
      // Fixed
      status = HighsBasisStatus::kLower;
      move = kNonbasicMoveZe;
    } else if (!highs_isInfinity(-lower)) {
      // Finite lower bound so boxed or lower
      if (!highs_isInfinity(upper)) {
        // Finite upper bound so boxed: use nonbasicMove to choose
        if (nonbasicMove[iCol] == kNonbasicMoveUp) {
          status = HighsBasisStatus::kLower;
          move = kNonbasicMoveUp;
        } else {
          status = HighsBasisStatus::kUpper;
          move = kNonbasicMoveDn;
        }
      } else {
        // Lower (since upper bound is infinite)
        status = HighsBasisStatus::kLower;
        move = kNonbasicMoveUp;
      }
    } else if (!highs_isInfinity(upper)) {
      // Upper
      status = HighsBasisStatus::kUpper;
      move = kNonbasicMoveDn;
    } else {
      // FREE
      status = HighsBasisStatus::kZero;
      move = kNonbasicMoveZe;
    }
    assert(status != HighsBasisStatus::kNonbasic);
    basis_.col_status[iCol] = status;
    assert(move != kIllegalMoveValue);
    nonbasicMove[iCol] = move;
  }
  // Complete the HiGHS basis row status and adjust nonbasicMove
  // for nonbasic variables
  for (HighsInt iRow = 0; iRow < num_row; iRow++) {
    if (nonbasicFlag[num_col + iRow] == kNonbasicFlagFalse) continue;
    const double lower = lp.row_lower_[iRow];
    const double upper = lp.row_upper_[iRow];
    HighsBasisStatus status = HighsBasisStatus::kNonbasic;
    HighsInt move = kIllegalMoveValue;
    if (lower == upper) {
      // Fixed
      status = HighsBasisStatus::kLower;
      move = kNonbasicMoveZe;
    } else if (!highs_isInfinity(-lower)) {
      // Finite lower bound so boxed or lower
      if (!highs_isInfinity(upper)) {
        // Finite upper bound so boxed: use nonbasicMove to choose
        if (nonbasicMove[num_col + iRow] == kNonbasicMoveDn) {
          status = HighsBasisStatus::kLower;
          move = kNonbasicMoveDn;
        } else {
          status = HighsBasisStatus::kUpper;
          move = kNonbasicMoveUp;
        }
      } else {
        // Lower (since upper bound is infinite)
        status = HighsBasisStatus::kLower;
        move = kNonbasicMoveDn;
      }
    } else if (!highs_isInfinity(upper)) {
      // Upper
      status = HighsBasisStatus::kUpper;
      move = kNonbasicMoveUp;
    } else {
      // FREE
      status = HighsBasisStatus::kZero;
      move = kNonbasicMoveZe;
    }
    assert(status != HighsBasisStatus::kNonbasic);
    basis_.row_status[iRow] = status;
    assert(move != kIllegalMoveValue);
    nonbasicMove[num_col + iRow] = move;
  }
  basis_.valid = true;
  ekk_instance_.status_.has_basis = true;
  ekk_instance_.setNlaRefactorInfo();
  ekk_instance_.updateStatus(LpAction::kHotStart);
  return HighsStatus::kOk;
}

void Highs::zeroIterationCounts() {
  info_.simplex_iteration_count = 0;
  info_.ipm_iteration_count = 0;
  info_.crossover_iteration_count = 0;
  info_.qp_iteration_count = 0;
}

HighsStatus Highs::getDualRayInterface(bool& has_dual_ray,
                                       double* dual_ray_value) {
  HighsStatus return_status = HighsStatus::kOk;
  HighsLp& lp = model_.lp_;
  HighsInt num_row = lp.num_row_;
  // For an LP with no rows the dual ray is vacuous
  if (num_row == 0) return return_status;
  assert(ekk_instance_.status_.has_invert);
  assert(!lp.is_moved_);
  has_dual_ray = ekk_instance_.status_.has_dual_ray;
  if (has_dual_ray && dual_ray_value != NULL) {
    vector<double> rhs;
    HighsInt iRow = ekk_instance_.info_.dual_ray_row_;
    rhs.assign(num_row, 0);
    rhs[iRow] = ekk_instance_.info_.dual_ray_sign_;
    HighsInt* dual_ray_num_nz = 0;
    basisSolveInterface(rhs, dual_ray_value, dual_ray_num_nz, NULL, true);
  }
  return return_status;
}

HighsStatus Highs::getPrimalRayInterface(bool& has_primal_ray,
                                         double* primal_ray_value) {
  HighsStatus return_status = HighsStatus::kOk;
  HighsLp& lp = model_.lp_;
  HighsInt num_row = lp.num_row_;
  HighsInt num_col = lp.num_col_;
  // For an LP with no rows the primal ray is vacuous
  if (num_row == 0) return return_status;
  assert(ekk_instance_.status_.has_invert);
  assert(!lp.is_moved_);
  has_primal_ray = ekk_instance_.status_.has_primal_ray;
  if (has_primal_ray && primal_ray_value != NULL) {
    HighsInt col = ekk_instance_.info_.primal_ray_col_;
    assert(ekk_instance_.basis_.nonbasicFlag_[col] == kNonbasicFlagTrue);
    // Get this pivotal column
    vector<double> rhs;
    vector<double> column;
    column.assign(num_row, 0);
    rhs.assign(num_row, 0);
    lp.ensureColwise();
    HighsInt primal_ray_sign = ekk_instance_.info_.primal_ray_sign_;
    if (col < num_col) {
      for (HighsInt iEl = lp.a_matrix_.start_[col];
           iEl < lp.a_matrix_.start_[col + 1]; iEl++)
        rhs[lp.a_matrix_.index_[iEl]] =
            primal_ray_sign * lp.a_matrix_.value_[iEl];
    } else {
      rhs[col - num_col] = primal_ray_sign;
    }
    HighsInt* column_num_nz = 0;
    basisSolveInterface(rhs, &column[0], column_num_nz, NULL, false);
    // Now zero primal_ray_value and scatter the column according to
    // the basic variables.
    for (HighsInt iCol = 0; iCol < num_col; iCol++) primal_ray_value[iCol] = 0;
    for (HighsInt iRow = 0; iRow < num_row; iRow++) {
      HighsInt iCol = ekk_instance_.basis_.basicIndex_[iRow];
      if (iCol < num_col) primal_ray_value[iCol] = column[iRow];
    }
    if (col < num_col) primal_ray_value[col] = -primal_ray_sign;
  }
  return return_status;
}

bool Highs::aFormatOk(const HighsInt num_nz, const HighsInt format) {
  if (!num_nz) return true;
  const bool ok_format = format == (HighsInt)MatrixFormat::kColwise ||
                         format == (HighsInt)MatrixFormat::kRowwise;
  assert(ok_format);
  if (!ok_format)
    highsLogUser(
        options_.log_options, HighsLogType::kError,
        "Non-empty Constraint matrix has illegal format = %" HIGHSINT_FORMAT
        "\n",
        format);
  return ok_format;
}

bool Highs::qFormatOk(const HighsInt num_nz, const HighsInt format) {
  if (!num_nz) return true;
  const bool ok_format = format == (HighsInt)HessianFormat::kTriangular;
  assert(ok_format);
  if (!ok_format)
    highsLogUser(
        options_.log_options, HighsLogType::kError,
        "Non-empty Hessian matrix has illegal format = %" HIGHSINT_FORMAT "\n",
        format);
  return ok_format;
}

void Highs::clearZeroHessian() {
  HighsHessian& hessian = model_.hessian_;
  if (hessian.dim_) {
    // Clear any zero Hessian
    if (hessian.numNz() == 0) {
      highsLogUser(options_.log_options, HighsLogType::kInfo,
                   "Hessian has dimension %" HIGHSINT_FORMAT
                   " but no nonzeros, so is ignored\n",
                   hessian.dim_);
      hessian.clear();
    }
  }
}

HighsStatus Highs::checkOptimality(const std::string& solver_type,
                                   HighsStatus return_status) {
  // Check for infeasibility measures incompatible with optimality
  assert(return_status != HighsStatus::kError);
  // Cannot expect to have no dual_infeasibilities since the QP solver
  // (and, of course, the MIP solver) give no dual information
  if (info_.num_primal_infeasibilities == 0 &&
      info_.num_dual_infeasibilities <= 0)
    return HighsStatus::kOk;
  HighsLogType log_type = HighsLogType::kWarning;
  return_status = HighsStatus::kWarning;
  if (info_.max_primal_infeasibility >
          sqrt(options_.primal_feasibility_tolerance) ||
      (info_.dual_solution_status != kSolutionStatusNone &&
       info_.max_dual_infeasibility >
           sqrt(options_.dual_feasibility_tolerance))) {
    // Check for gross errors
    log_type = HighsLogType::kError;
    return_status = HighsStatus::kError;
  }
  std::stringstream ss;
  ss.str(std::string());
  ss << highsFormatToString(
      "%s solver claims optimality, but with num/sum/max "
      "primal(%" HIGHSINT_FORMAT "/%g/%g)",
      solver_type.c_str(), info_.num_primal_infeasibilities,
      info_.sum_primal_infeasibilities, info_.max_primal_infeasibility);
  if (info_.num_dual_infeasibilities > 0)
    ss << highsFormatToString(
        "and dual(%" HIGHSINT_FORMAT "/%g/%g)", info_.num_dual_infeasibilities,
        info_.sum_dual_infeasibilities, info_.max_dual_infeasibility);
  ss << " infeasibilities\n";
  const std::string report_string = ss.str();
  highsLogUser(options_.log_options, log_type, "%s", report_string.c_str());
  return return_status;
}

HighsStatus Highs::invertRequirementError(std::string method_name) {
  assert(!ekk_instance_.status_.has_invert);
  highsLogUser(options_.log_options, HighsLogType::kError,
               "No invertible representation for %s\n", method_name.c_str());
  return HighsStatus::kError;
}
