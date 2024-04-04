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
/**@file model/HighsModel.h
 * @brief
 */
#ifndef MODEL_HIGHS_MODEL_H_
#define MODEL_HIGHS_MODEL_H_

#include <vector>

#include "lp_data/HighsLp.h"
#include "model/HighsHessian.h"

class HighsModel;

class HighsModel {
 public:
  HighsLp lp_;
  HighsHessian hessian_;
  bool isQp() const { return this->hessian_.dim_; }
  bool isMip() const { return this->lp_.isMip(); }
  bool isEmpty() const {
    return (this->lp_.num_col_ == 0 && this->lp_.num_row_ == 0);
  }
  void clear();
  double objectiveValue(const std::vector<double>& solution) const;
  void objectiveGradient(const std::vector<double>& solution,
                         std::vector<double>& gradient) const;
};

#endif
