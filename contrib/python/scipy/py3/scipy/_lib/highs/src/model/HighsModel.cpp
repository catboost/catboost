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
/**@file lp_data/HighsModel.cpp
 * @brief
 */
#include "model/HighsModel.h"

#include <cassert>

void HighsModel::clear() {
  this->lp_.clear();
  this->hessian_.clear();
}

double HighsModel::objectiveValue(const std::vector<double>& solution) const {
  return this->hessian_.objectiveValue(solution) +
         this->lp_.objectiveValue(solution);
}

void HighsModel::objectiveGradient(const std::vector<double>& solution,
                                   std::vector<double>& gradient) const {
  if (this->hessian_.dim_ > 0) {
    this->hessian_.product(solution, gradient);
  } else {
    gradient.assign(this->lp_.num_col_, 0);
  }
  for (HighsInt iCol = 0; iCol < this->lp_.num_col_; iCol++)
    gradient[iCol] += this->lp_.col_cost_[iCol];
}
