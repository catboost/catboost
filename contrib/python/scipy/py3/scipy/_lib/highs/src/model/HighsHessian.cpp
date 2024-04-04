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
/**@file lp_data/HighsHessian.cpp
 * @brief
 */
#include "model/HighsHessian.h"

#include <cassert>
#include <cstdio>

void HighsHessian::clear() {
  this->dim_ = 0;
  this->start_.clear();
  this->index_.clear();
  this->value_.clear();
  this->format_ = HessianFormat::kTriangular;
  this->start_.assign(1, 0);
}

void HighsHessian::exactResize() {
  if (this->dim_) {
    this->start_.resize(this->dim_ + 1);
    HighsInt num_nz = this->start_[this->dim_];
    this->index_.resize(num_nz);
    this->value_.resize(num_nz);
  } else {
    this->start_.clear();
    this->index_.clear();
    this->value_.clear();
  }
}

HighsInt HighsHessian::numNz() const {
  assert(this->formatOk());
  assert((HighsInt)this->start_.size() >= this->dim_ + 1);
  return this->start_[this->dim_];
}

void HighsHessian::print() const {
  HighsInt num_nz = this->numNz();
  printf("Hessian of dimension %" HIGHSINT_FORMAT " and %" HIGHSINT_FORMAT
         " entries\n",
         dim_, num_nz);
  printf("Start; Index; Value of sizes %d; %d; %d\n", (int)this->start_.size(),
         (int)this->index_.size(), (int)this->value_.size());
  if (dim_ <= 0) return;
  printf(" Row|");
  for (int iCol = 0; iCol < dim_; iCol++) printf(" %4d", iCol);
  printf("\n");
  printf("-----");
  for (int iCol = 0; iCol < dim_; iCol++) printf("-----");
  printf("\n");
  std::vector<double> col;
  col.assign(dim_, 0);
  for (HighsInt iCol = 0; iCol < dim_; iCol++) {
    for (HighsInt iEl = this->start_[iCol]; iEl < this->start_[iCol + 1]; iEl++)
      col[this->index_[iEl]] = this->value_[iEl];
    printf("%4d|", (int)iCol);
    for (int iRow = 0; iRow < dim_; iRow++) printf(" %4g", col[iRow]);
    printf("\n");
    for (HighsInt iEl = this->start_[iCol]; iEl < this->start_[iCol + 1]; iEl++)
      col[this->index_[iEl]] = 0;
  }
}
bool HighsHessian::operator==(const HighsHessian& hessian) {
  bool equal = true;
  equal = this->dim_ == hessian.dim_ && equal;
  equal = this->start_ == hessian.start_ && equal;
  equal = this->index_ == hessian.index_ && equal;
  equal = this->value_ == hessian.value_ && equal;
  return equal;
}

void HighsHessian::product(const std::vector<double>& solution,
                           std::vector<double>& product) const {
  if (this->dim_ <= 0) return;
  product.assign(this->dim_, 0);
  for (HighsInt iCol = 0; iCol < this->dim_; iCol++) {
    for (HighsInt iEl = this->start_[iCol]; iEl < this->start_[iCol + 1];
         iEl++) {
      const HighsInt iRow = this->index_[iEl];
      product[iRow] += this->value_[iEl] * solution[iCol];
    }
  }
}

double HighsHessian::objectiveValue(const std::vector<double>& solution) const {
  double objective_function_value = 0;
  for (HighsInt iCol = 0; iCol < this->dim_; iCol++) {
    HighsInt iEl = this->start_[iCol];
    assert(this->index_[iEl] == iCol);
    objective_function_value +=
        0.5 * solution[iCol] * this->value_[iEl] * solution[iCol];
    for (HighsInt iEl = this->start_[iCol] + 1; iEl < this->start_[iCol + 1];
         iEl++)
      objective_function_value +=
          solution[iCol] * this->value_[iEl] * solution[this->index_[iEl]];
  }
  return objective_function_value;
}
