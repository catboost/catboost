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
/**@file lp_data/HighsLp.h
 * @brief
 */
#ifndef LP_DATA_HIGHS_LP_H_
#define LP_DATA_HIGHS_LP_H_

#include <string>
#include <vector>

#include "lp_data/HStruct.h"
#include "util/HighsSparseMatrix.h"

class HighsLp {
 public:
  HighsLp() { clear(); }
  // Model data
  HighsInt num_col_;
  HighsInt num_row_;

  std::vector<double> col_cost_;
  std::vector<double> col_lower_;
  std::vector<double> col_upper_;
  std::vector<double> row_lower_;
  std::vector<double> row_upper_;

  HighsSparseMatrix a_matrix_;

  ObjSense sense_;
  double offset_;

  std::string model_name_;
  std::string objective_name_;

  std::vector<std::string> col_names_;
  std::vector<std::string> row_names_;

  std::vector<HighsVarType> integrality_;

  HighsScale scale_;
  bool is_scaled_;
  bool is_moved_;
  HighsInt cost_row_location_;
  HighsLpMods mods_;

  bool operator==(const HighsLp& lp);
  bool equalButForNames(const HighsLp& lp) const;
  bool isMip() const;
  bool hasSemiVariables() const;
  double objectiveValue(const std::vector<double>& solution) const;
  void setMatrixDimensions();
  void setFormat(const MatrixFormat format);
  void ensureColwise() { this->a_matrix_.ensureColwise(); };
  void ensureRowwise() { this->a_matrix_.ensureRowwise(); };
  void clearScaling();
  void resetScale();
  void clearScale();
  void applyScale();
  void unapplyScale();
  void moveBackLpAndUnapplyScaling(HighsLp& lp);
  void exactResize();
  void unapplyMods();
  void clear();
};

#endif
