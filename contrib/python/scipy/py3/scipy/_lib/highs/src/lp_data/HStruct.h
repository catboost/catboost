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
/**@file lp_data/HStruct.h
 * @brief Structs for HiGHS
 */
#ifndef LP_DATA_HSTRUCT_H_
#define LP_DATA_HSTRUCT_H_

#include <vector>

#include "lp_data/HConst.h"

struct HighsIterationCounts {
  HighsInt simplex = 0;
  HighsInt ipm = 0;
  HighsInt crossover = 0;
  HighsInt qp = 0;
};

struct HighsSolution {
  bool value_valid = false;
  bool dual_valid = false;
  std::vector<double> col_value;
  std::vector<double> col_dual;
  std::vector<double> row_value;
  std::vector<double> row_dual;
  void invalidate();
  void clear();
};

struct RefactorInfo {
  bool use = false;
  std::vector<HighsInt> pivot_row;
  std::vector<HighsInt> pivot_var;
  std::vector<int8_t> pivot_type;
  double build_synthetic_tick;
  void clear();
};

struct HotStart {
  bool valid = false;
  RefactorInfo refactor_info;
  std::vector<int8_t> nonbasicMove;
  void clear();
};

struct HighsBasis {
  bool valid = false;
  bool alien = true;
  bool was_alien = true;
  HighsInt debug_id = -1;
  HighsInt debug_update_count = -1;
  std::string debug_origin_name = "None";
  std::vector<HighsBasisStatus> col_status;
  std::vector<HighsBasisStatus> row_status;
  void invalidate();
  void clear();
};

struct HighsScale {
  HighsInt strategy;
  bool has_scaling;
  HighsInt num_col;
  HighsInt num_row;
  double cost;
  std::vector<double> col;
  std::vector<double> row;
};

struct HighsLpMods {
  std::vector<HighsInt> save_semi_variable_upper_bound_index;
  std::vector<double> save_semi_variable_upper_bound_value;
  void clear();
  bool isClear();
};

#endif /* LP_DATA_HSTRUCT_H_ */
