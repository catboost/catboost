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
/**@file model/HighsHessian.h
 * @brief
 */
#ifndef MODEL_HIGHS_HESSIAN_H_
#define MODEL_HIGHS_HESSIAN_H_

#include <vector>

#include "lp_data/HConst.h"

// class HighsHessian;

class HighsHessian {
 public:
  HighsHessian() { clear(); }
  HighsInt dim_;
  HessianFormat format_;
  std::vector<HighsInt> start_;
  std::vector<HighsInt> index_;
  std::vector<double> value_;
  bool operator==(const HighsHessian& hessian);
  void product(const std::vector<double>& solution,
               std::vector<double>& product) const;
  double objectiveValue(const std::vector<double>& solution) const;
  void exactResize();
  void clear();
  bool formatOk() const {
    return (this->format_ == HessianFormat::kTriangular ||
            this->format_ == HessianFormat::kSquare);
  };
  HighsInt numNz() const;
  void print() const;
};

#endif
