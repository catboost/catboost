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
/**@file lp_data/HighsLp.cpp
 * @brief
 */
#include "lp_data/HighsLp.h"

#include <cassert>

#include "util/HighsMatrixUtils.h"

bool HighsLp::isMip() const {
  HighsInt integrality_size = this->integrality_.size();
  if (integrality_size) {
    assert(integrality_size == this->num_col_);
    for (HighsInt iCol = 0; iCol < this->num_col_; iCol++)
      if (this->integrality_[iCol] != HighsVarType::kContinuous) return true;
  }
  return false;
}

bool HighsLp::hasSemiVariables() const {
  HighsInt integrality_size = this->integrality_.size();
  if (integrality_size) {
    assert(integrality_size == this->num_col_);
    for (HighsInt iCol = 0; iCol < this->num_col_; iCol++)
      if (this->integrality_[iCol] == HighsVarType::kSemiContinuous ||
          this->integrality_[iCol] == HighsVarType::kSemiInteger)
        return true;
  }
  return false;
}

bool HighsLp::operator==(const HighsLp& lp) {
  bool equal = equalButForNames(lp);
  equal = this->objective_name_ == lp.objective_name_ && equal;
  equal = this->row_names_ == lp.row_names_ && equal;
  equal = this->col_names_ == lp.col_names_ && equal;
  return equal;
}

bool HighsLp::equalButForNames(const HighsLp& lp) const {
  bool equal = true;
  equal = this->num_col_ == lp.num_col_ && equal;
  equal = this->num_row_ == lp.num_row_ && equal;
  equal = this->sense_ == lp.sense_ && equal;
  equal = this->offset_ == lp.offset_ && equal;
  equal = this->model_name_ == lp.model_name_ && equal;
  equal = this->col_cost_ == lp.col_cost_ && equal;
  equal = this->col_upper_ == lp.col_upper_ && equal;
  equal = this->col_lower_ == lp.col_lower_ && equal;
  equal = this->row_upper_ == lp.row_upper_ && equal;
  equal = this->row_lower_ == lp.row_lower_ && equal;

  equal = this->a_matrix_ == lp.a_matrix_;

  equal = this->scale_.strategy == lp.scale_.strategy && equal;
  equal = this->scale_.has_scaling == lp.scale_.has_scaling && equal;
  equal = this->scale_.num_col == lp.scale_.num_col && equal;
  equal = this->scale_.num_row == lp.scale_.num_row && equal;
  equal = this->scale_.cost == lp.scale_.cost && equal;
  equal = this->scale_.col == lp.scale_.col && equal;
  equal = this->scale_.row == lp.scale_.row && equal;
  return equal;
}

double HighsLp::objectiveValue(const std::vector<double>& solution) const {
  assert((int)solution.size() >= this->num_col_);
  double objective_function_value = this->offset_;
  for (HighsInt iCol = 0; iCol < this->num_col_; iCol++)
    objective_function_value += this->col_cost_[iCol] * solution[iCol];
  return objective_function_value;
}

void HighsLp::setMatrixDimensions() {
  this->a_matrix_.num_col_ = this->num_col_;
  this->a_matrix_.num_row_ = this->num_row_;
}

void HighsLp::resetScale() {
  // Should allow user-supplied scale to be retained
  //  const bool dimensions_ok =
  //    this->scale_.num_col_ == this->num_col_ &&
  //    this->scale_.num_row_ == this->num_row_;
  this->clearScale();
}

void HighsLp::setFormat(const MatrixFormat format) {
  this->a_matrix_.setFormat(format);
}

void HighsLp::exactResize() {
  this->col_cost_.resize(this->num_col_);
  this->col_lower_.resize(this->num_col_);
  this->col_upper_.resize(this->num_col_);
  this->row_lower_.resize(this->num_row_);
  this->row_upper_.resize(this->num_row_);
  this->a_matrix_.exactResize();

  if ((int)this->col_names_.size()) this->col_names_.resize(this->num_col_);
  if ((int)this->row_names_.size()) this->row_names_.resize(this->num_row_);
  if ((int)this->integrality_.size()) this->integrality_.resize(this->num_col_);
}

void HighsLp::clear() {
  this->num_col_ = 0;
  this->num_row_ = 0;

  this->col_cost_.clear();
  this->col_lower_.clear();
  this->col_upper_.clear();
  this->row_lower_.clear();
  this->row_upper_.clear();

  this->a_matrix_.clear();

  this->sense_ = ObjSense::kMinimize;
  this->offset_ = 0;

  this->model_name_ = "";
  this->objective_name_ = "";

  this->col_names_.clear();
  this->row_names_.clear();

  this->integrality_.clear();

  this->clearScale();
  this->is_scaled_ = false;
  this->is_moved_ = false;
  this->cost_row_location_ = -1;
  this->mods_.clear();
}

void HighsLp::clearScale() {
  this->scale_.strategy = kSimplexScaleStrategyOff;
  this->scale_.has_scaling = false;
  this->scale_.num_col = 0;
  this->scale_.num_row = 0;
  this->scale_.cost = 0;
  this->scale_.col.clear();
  this->scale_.row.clear();
}

void HighsLp::clearScaling() {
  this->unapplyScale();
  this->clearScale();
}

void HighsLp::applyScale() {
  // Ensure that any scaling is applied
  const HighsScale& scale = this->scale_;
  if (this->is_scaled_) {
    // Already scaled - so check that there is scaling and return
    assert(scale.has_scaling);
    return;
  }
  // No scaling currently applied
  this->is_scaled_ = false;
  if (scale.has_scaling) {
    // Apply the scaling to the bounds, costs and matrix, and record
    // that it has been applied
    for (HighsInt iCol = 0; iCol < this->num_col_; iCol++) {
      this->col_lower_[iCol] /= scale.col[iCol];
      this->col_upper_[iCol] /= scale.col[iCol];
      this->col_cost_[iCol] *= scale.col[iCol];
    }
    for (HighsInt iRow = 0; iRow < this->num_row_; iRow++) {
      this->row_lower_[iRow] *= scale.row[iRow];
      this->row_upper_[iRow] *= scale.row[iRow];
    }
    this->a_matrix_.applyScale(scale);
    this->is_scaled_ = true;
  }
}

void HighsLp::unapplyScale() {
  // Ensure that any scaling is not applied
  const HighsScale& scale = this->scale_;
  if (!this->is_scaled_) {
    // Not scaled so return
    return;
  }
  // Already scaled - so check that there is scaling
  assert(scale.has_scaling);
  // Unapply the scaling to the bounds, costs and matrix, and record
  // that it has been unapplied
  for (HighsInt iCol = 0; iCol < this->num_col_; iCol++) {
    this->col_lower_[iCol] *= scale.col[iCol];
    this->col_upper_[iCol] *= scale.col[iCol];
    this->col_cost_[iCol] /= scale.col[iCol];
  }
  for (HighsInt iRow = 0; iRow < this->num_row_; iRow++) {
    this->row_lower_[iRow] /= scale.row[iRow];
    this->row_upper_[iRow] /= scale.row[iRow];
  }
  this->a_matrix_.unapplyScale(scale);
  this->is_scaled_ = false;
}

void HighsLp::moveBackLpAndUnapplyScaling(HighsLp& lp) {
  assert(this->is_moved_ == true);
  *this = std::move(lp);
  this->unapplyScale();
  assert(this->is_moved_ == false);
}

void HighsLp::unapplyMods() {
  std::vector<HighsInt>& upper_bound_index =
      this->mods_.save_semi_variable_upper_bound_index;
  std::vector<double>& upper_bound_value =
      this->mods_.save_semi_variable_upper_bound_value;
  const HighsInt num_upper_bound = upper_bound_index.size();
  if (!num_upper_bound) {
    assert(!upper_bound_value.size());
    return;
  }
  for (HighsInt k = 0; k < num_upper_bound; k++) {
    HighsInt iCol = upper_bound_index[k];
    this->col_upper_[iCol] = upper_bound_value[k];
  }
  this->mods_.clear();
}

void HighsLpMods::clear() {
  this->save_semi_variable_upper_bound_index.clear();
  this->save_semi_variable_upper_bound_value.clear();
}

bool HighsLpMods::isClear() {
  if (this->save_semi_variable_upper_bound_index.size()) return false;
  if (this->save_semi_variable_upper_bound_value.size()) return false;
  return true;
}
