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
#include "highs_c_api.h"

#include "Highs.h"

HighsInt Highs_lpCall(const HighsInt num_col, const HighsInt num_row,
                      const HighsInt num_nz, const HighsInt a_format,
                      const HighsInt sense, const double offset,
                      const double* col_cost, const double* col_lower,
                      const double* col_upper, const double* row_lower,
                      const double* row_upper, const HighsInt* a_start,
                      const HighsInt* a_index, const double* a_value,
                      double* col_value, double* col_dual, double* row_value,
                      double* row_dual, HighsInt* col_basis_status,
                      HighsInt* row_basis_status, HighsInt* model_status) {
  Highs highs;
  highs.setOptionValue("output_flag", false);
  HighsStatus status = highs.passModel(
      num_col, num_row, num_nz, a_format, sense, offset, col_cost, col_lower,
      col_upper, row_lower, row_upper, a_start, a_index, a_value);
  if (status != HighsStatus::kOk) return (HighsInt)status;

  status = highs.run();

  if (status == HighsStatus::kOk) {
    HighsSolution solution;
    HighsBasis basis;
    solution = highs.getSolution();
    basis = highs.getBasis();
    *model_status = (HighsInt)highs.getModelStatus();
    const HighsInfo& info = highs.getInfo();

    const bool copy_col_value =
        col_value != nullptr &&
        info.primal_solution_status != SolutionStatus::kSolutionStatusNone;
    const bool copy_col_dual =
        col_dual != nullptr &&
        info.dual_solution_status != SolutionStatus::kSolutionStatusNone;
    const bool copy_col_basis = col_basis_status != nullptr && basis.valid;
    for (HighsInt i = 0; i < num_col; i++) {
      if (copy_col_value) col_value[i] = solution.col_value[i];
      if (copy_col_dual) col_dual[i] = solution.col_dual[i];
      if (copy_col_basis) col_basis_status[i] = (HighsInt)basis.col_status[i];
    }

    const bool copy_row_value =
        row_value != nullptr &&
        info.primal_solution_status != SolutionStatus::kSolutionStatusNone;
    const bool copy_row_dual =
        row_dual != nullptr &&
        info.dual_solution_status != SolutionStatus::kSolutionStatusNone;
    const bool copy_row_basis = row_basis_status != nullptr && basis.valid;
    for (HighsInt i = 0; i < num_row; i++) {
      if (copy_row_value) row_value[i] = solution.row_value[i];
      if (copy_row_dual) row_dual[i] = solution.row_dual[i];
      if (copy_row_basis) row_basis_status[i] = (HighsInt)basis.row_status[i];
    }
  }
  return (HighsInt)status;
}

HighsInt Highs_mipCall(const HighsInt num_col, const HighsInt num_row,
                       const HighsInt num_nz, const HighsInt a_format,
                       const HighsInt sense, const double offset,
                       const double* col_cost, const double* col_lower,
                       const double* col_upper, const double* row_lower,
                       const double* row_upper, const HighsInt* a_start,
                       const HighsInt* a_index, const double* a_value,
                       const HighsInt* integrality, double* col_value,
                       double* row_value, HighsInt* model_status) {
  Highs highs;
  highs.setOptionValue("output_flag", false);
  HighsStatus status = highs.passModel(
      num_col, num_row, num_nz, a_format, sense, offset, col_cost, col_lower,
      col_upper, row_lower, row_upper, a_start, a_index, a_value, integrality);
  if (status != HighsStatus::kOk) return (HighsInt)status;

  status = highs.run();

  if (status == HighsStatus::kOk) {
    HighsSolution solution;
    solution = highs.getSolution();
    *model_status = (HighsInt)highs.getModelStatus();
    const HighsInfo& info = highs.getInfo();
    const bool copy_col_value =
        col_value != nullptr &&
        info.primal_solution_status != SolutionStatus::kSolutionStatusNone;

    if (copy_col_value) {
      for (HighsInt i = 0; i < num_col; i++)
        col_value[i] = solution.col_value[i];
    }
    const bool copy_row_value =
        row_value != nullptr &&
        info.primal_solution_status != SolutionStatus::kSolutionStatusNone;
    if (copy_row_value) {
      for (HighsInt i = 0; i < num_row; i++)
        row_value[i] = solution.row_value[i];
    }
  }

  return (HighsInt)status;
}

HighsInt Highs_qpCall(
    const HighsInt num_col, const HighsInt num_row, const HighsInt num_nz,
    const HighsInt q_num_nz, const HighsInt a_format, const HighsInt q_format,
    const HighsInt sense, const double offset, const double* col_cost,
    const double* col_lower, const double* col_upper, const double* row_lower,
    const double* row_upper, const HighsInt* a_start, const HighsInt* a_index,
    const double* a_value, const HighsInt* q_start, const HighsInt* q_index,
    const double* q_value, double* col_value, double* col_dual,
    double* row_value, double* row_dual, HighsInt* col_basis_status,
    HighsInt* row_basis_status, HighsInt* model_status) {
  Highs highs;
  highs.setOptionValue("output_flag", false);
  HighsStatus status = highs.passModel(
      num_col, num_row, num_nz, q_num_nz, a_format, q_format, sense, offset,
      col_cost, col_lower, col_upper, row_lower, row_upper, a_start, a_index,
      a_value, q_start, q_index, q_value);
  if (status != HighsStatus::kOk) return (HighsInt)status;

  status = highs.run();

  if (status == HighsStatus::kOk) {
    HighsSolution solution;
    HighsBasis basis;
    solution = highs.getSolution();
    basis = highs.getBasis();
    *model_status = (HighsInt)highs.getModelStatus();
    const HighsInfo& info = highs.getInfo();

    const bool copy_col_value =
        col_value != nullptr &&
        info.primal_solution_status != SolutionStatus::kSolutionStatusNone;
    const bool copy_col_dual =
        col_dual != nullptr &&
        info.dual_solution_status != SolutionStatus::kSolutionStatusNone;
    const bool copy_col_basis = col_basis_status != nullptr && basis.valid;
    for (HighsInt i = 0; i < num_col; i++) {
      if (copy_col_value) col_value[i] = solution.col_value[i];
      if (copy_col_dual) col_dual[i] = solution.col_dual[i];
      if (copy_col_basis) col_basis_status[i] = (HighsInt)basis.col_status[i];
    }

    const bool copy_row_value =
        row_value != nullptr &&
        info.primal_solution_status != SolutionStatus::kSolutionStatusNone;
    const bool copy_row_dual =
        row_dual != nullptr &&
        info.dual_solution_status != SolutionStatus::kSolutionStatusNone;
    const bool copy_row_basis = row_basis_status != nullptr && basis.valid;
    for (HighsInt i = 0; i < num_row; i++) {
      if (copy_row_value) row_value[i] = solution.row_value[i];
      if (copy_row_dual) row_dual[i] = solution.row_dual[i];
      if (copy_row_basis) row_basis_status[i] = (HighsInt)basis.row_status[i];
    }
  }
  return (HighsInt)status;
}

void* Highs_create() { return new Highs(); }

void Highs_destroy(void* highs) { delete (Highs*)highs; }

HighsInt Highs_run(void* highs) { return (HighsInt)((Highs*)highs)->run(); }

HighsInt Highs_readModel(void* highs, const char* filename) {
  return (HighsInt)((Highs*)highs)->readModel(std::string(filename));
}

HighsInt Highs_writeModel(void* highs, const char* filename) {
  return (HighsInt)((Highs*)highs)->writeModel(std::string(filename));
}

HighsInt Highs_writeSolution(const void* highs, const char* filename) {
  return (HighsInt)((Highs*)highs)
      ->writeSolution(std::string(filename), kSolutionStyleRaw);
}

HighsInt Highs_writeSolutionPretty(const void* highs, const char* filename) {
  return (HighsInt)((Highs*)highs)
      ->writeSolution(std::string(filename), kSolutionStylePretty);
}

HighsInt Highs_passLp(void* highs, const HighsInt num_col,
                      const HighsInt num_row, const HighsInt num_nz,
                      const HighsInt a_format, const HighsInt sense,
                      const double offset, const double* col_cost,
                      const double* col_lower, const double* col_upper,
                      const double* row_lower, const double* row_upper,
                      const HighsInt* a_start, const HighsInt* a_index,
                      const double* a_value) {
  return (HighsInt)((Highs*)highs)
      ->passModel(num_col, num_row, num_nz, a_format, sense, offset, col_cost,
                  col_lower, col_upper, row_lower, row_upper, a_start, a_index,
                  a_value);
}

HighsInt Highs_passMip(void* highs, const HighsInt num_col,
                       const HighsInt num_row, const HighsInt num_nz,
                       const HighsInt a_format, const HighsInt sense,
                       const double offset, const double* col_cost,
                       const double* col_lower, const double* col_upper,
                       const double* row_lower, const double* row_upper,
                       const HighsInt* a_start, const HighsInt* a_index,
                       const double* a_value, const HighsInt* integrality) {
  return (HighsInt)((Highs*)highs)
      ->passModel(num_col, num_row, num_nz, a_format, sense, offset, col_cost,
                  col_lower, col_upper, row_lower, row_upper, a_start, a_index,
                  a_value, integrality);
}

HighsInt Highs_passModel(void* highs, const HighsInt num_col,
                         const HighsInt num_row, const HighsInt num_nz,
                         const HighsInt q_num_nz, const HighsInt a_format,
                         const HighsInt q_format, const HighsInt sense,
                         const double offset, const double* col_cost,
                         const double* col_lower, const double* col_upper,
                         const double* row_lower, const double* row_upper,
                         const HighsInt* a_start, const HighsInt* a_index,
                         const double* a_value, const HighsInt* q_start,
                         const HighsInt* q_index, const double* q_value,
                         const HighsInt* integrality) {
  return (HighsInt)((Highs*)highs)
      ->passModel(num_col, num_row, num_nz, q_num_nz, a_format, q_format, sense,
                  offset, col_cost, col_lower, col_upper, row_lower, row_upper,
                  a_start, a_index, a_value, q_start, q_index, q_value,
                  integrality);
}

HighsInt Highs_passHessian(void* highs, const HighsInt dim,
                           const HighsInt num_nz, const HighsInt format,
                           const HighsInt* start, const HighsInt* index,
                           const double* value) {
  return (HighsInt)((Highs*)highs)
      ->passHessian(dim, num_nz, format, start, index, value);
}

HighsInt Highs_clear(void* highs) { return (HighsInt)((Highs*)highs)->clear(); }

HighsInt Highs_clearModel(void* highs) {
  return (HighsInt)((Highs*)highs)->clearModel();
}

HighsInt Highs_clearSolver(void* highs) {
  return (HighsInt)((Highs*)highs)->clearSolver();
}

HighsInt Highs_setBoolOptionValue(void* highs, const char* option,
                                  const HighsInt value) {
  return (HighsInt)((Highs*)highs)
      ->setOptionValue(std::string(option), (bool)value);
}

HighsInt Highs_setIntOptionValue(void* highs, const char* option,
                                 const HighsInt value) {
  return (HighsInt)((Highs*)highs)->setOptionValue(std::string(option), value);
}

HighsInt Highs_setDoubleOptionValue(void* highs, const char* option,
                                    const double value) {
  return (HighsInt)((Highs*)highs)->setOptionValue(std::string(option), value);
}

HighsInt Highs_setStringOptionValue(void* highs, const char* option,
                                    const char* value) {
  return (HighsInt)((Highs*)highs)
      ->setOptionValue(std::string(option), std::string(value));
}

HighsInt Highs_getBoolOptionValue(const void* highs, const char* option,
                                  HighsInt* value) {
  bool v;
  HighsInt retcode =
      (HighsInt)((Highs*)highs)->getOptionValue(std::string(option), v);
  *value = (HighsInt)v;
  return retcode;
}

HighsInt Highs_getIntOptionValue(const void* highs, const char* option,
                                 HighsInt* value) {
  return (HighsInt)((Highs*)highs)->getOptionValue(std::string(option), *value);
}

HighsInt Highs_getDoubleOptionValue(const void* highs, const char* option,
                                    double* value) {
  return (HighsInt)((Highs*)highs)->getOptionValue(std::string(option), *value);
}

HighsInt Highs_getStringOptionValue(const void* highs, const char* option,
                                    char* value) {
  std::string v;
  memset(value, 0, 7);
  HighsInt retcode =
      (HighsInt)((Highs*)highs)->getOptionValue(std::string(option), v);
  strcpy(value, v.c_str());
  return retcode;
}

HighsInt Highs_getOptionType(const void* highs, const char* option,
                             HighsInt* type) {
  HighsOptionType t;
  HighsInt retcode =
      (HighsInt)((Highs*)highs)->getOptionType(std::string(option), t);
  *type = (HighsInt)t;
  return retcode;
}

HighsInt Highs_resetOptions(void* highs) {
  return (HighsInt)((Highs*)highs)->resetOptions();
}

HighsInt Highs_writeOptions(const void* highs, const char* filename) {
  return (HighsInt)((Highs*)highs)->writeOptions(filename);
}

HighsInt Highs_writeOptionsDeviations(const void* highs, const char* filename) {
  return (HighsInt)((Highs*)highs)->writeOptions(filename, true);
}

HighsInt Highs_getIntInfoValue(const void* highs, const char* info,
                               HighsInt* value) {
  return (HighsInt)((Highs*)highs)->getInfoValue(info, *value);
}

HighsInt Highs_getDoubleInfoValue(const void* highs, const char* info,
                                  double* value) {
  return (HighsInt)((Highs*)highs)->getInfoValue(info, *value);
}

HighsInt Highs_getInt64InfoValue(const void* highs, const char* info,
                                 int64_t* value) {
  return (HighsInt)((Highs*)highs)->getInfoValue(info, *value);
}

HighsInt Highs_getSolution(const void* highs, double* col_value,
                           double* col_dual, double* row_value,
                           double* row_dual) {
  HighsSolution solution = ((Highs*)highs)->getSolution();

  if (col_value != nullptr) {
    for (HighsInt i = 0; i < (HighsInt)solution.col_value.size(); i++) {
      col_value[i] = solution.col_value[i];
    }
  }

  if (col_dual != nullptr) {
    for (HighsInt i = 0; i < (HighsInt)solution.col_dual.size(); i++) {
      col_dual[i] = solution.col_dual[i];
    }
  }

  if (row_value != nullptr) {
    for (HighsInt i = 0; i < (HighsInt)solution.row_value.size(); i++) {
      row_value[i] = solution.row_value[i];
    }
  }

  if (row_dual != nullptr) {
    for (HighsInt i = 0; i < (HighsInt)solution.row_dual.size(); i++) {
      row_dual[i] = solution.row_dual[i];
    }
  }
  return kHighsStatusOk;
}

HighsInt Highs_getBasis(const void* highs, HighsInt* col_status,
                        HighsInt* row_status) {
  HighsBasis basis = ((Highs*)highs)->getBasis();
  for (HighsInt i = 0; i < (HighsInt)basis.col_status.size(); i++) {
    col_status[i] = (HighsInt)basis.col_status[i];
  }

  for (HighsInt i = 0; i < (HighsInt)basis.row_status.size(); i++) {
    row_status[i] = (HighsInt)basis.row_status[i];
  }
  return kHighsStatusOk;
}

HighsInt Highs_getModelStatus(const void* highs) {
  return (HighsInt)((Highs*)highs)->getModelStatus();
}

HighsInt Highs_getDualRay(const void* highs, HighsInt* has_dual_ray,
                          double* dual_ray_value) {
  bool v;
  HighsInt retcode = (HighsInt)((Highs*)highs)->getDualRay(v, dual_ray_value);
  *has_dual_ray = (HighsInt)v;
  return retcode;
}

HighsInt Highs_getPrimalRay(const void* highs, HighsInt* has_primal_ray,
                            double* primal_ray_value) {
  bool v;
  HighsInt retcode =
      (HighsInt)((Highs*)highs)->getPrimalRay(v, primal_ray_value);
  *has_primal_ray = (HighsInt)v;
  return retcode;
}

double Highs_getObjectiveValue(const void* highs) {
  return ((Highs*)highs)->getObjectiveValue();
}

HighsInt Highs_getBasicVariables(const void* highs, HighsInt* basic_variables) {
  return (HighsInt)((Highs*)highs)->getBasicVariables(basic_variables);
}

HighsInt Highs_getBasisInverseRow(const void* highs, const HighsInt row,
                                  double* row_vector, HighsInt* row_num_nz,
                                  HighsInt* row_index) {
  return (HighsInt)((Highs*)highs)
      ->getBasisInverseRow(row, row_vector, row_num_nz, row_index);
}

HighsInt Highs_getBasisInverseCol(const void* highs, const HighsInt col,
                                  double* col_vector, HighsInt* col_num_nz,
                                  HighsInt* col_index) {
  return (HighsInt)((Highs*)highs)
      ->getBasisInverseCol(col, col_vector, col_num_nz, col_index);
}

HighsInt Highs_getBasisSolve(const void* highs, const double* rhs,
                             double* solution_vector, HighsInt* solution_num_nz,
                             HighsInt* solution_index) {
  return (HighsInt)((Highs*)highs)
      ->getBasisSolve(rhs, solution_vector, solution_num_nz, solution_index);
}

HighsInt Highs_getBasisTransposeSolve(const void* highs, const double* rhs,
                                      double* solution_vector,
                                      HighsInt* solution_nz,
                                      HighsInt* solution_index) {
  return (HighsInt)((Highs*)highs)
      ->getBasisTransposeSolve(rhs, solution_vector, solution_nz,
                               solution_index);
}

HighsInt Highs_getReducedRow(const void* highs, const HighsInt row,
                             double* row_vector, HighsInt* row_num_nz,
                             HighsInt* row_index) {
  return (HighsInt)((Highs*)highs)
      ->getReducedRow(row, row_vector, row_num_nz, row_index);
}

HighsInt Highs_getReducedColumn(const void* highs, const HighsInt col,
                                double* col_vector, HighsInt* col_num_nz,
                                HighsInt* col_index) {
  return (HighsInt)((Highs*)highs)
      ->getReducedColumn(col, col_vector, col_num_nz, col_index);
}

HighsInt Highs_setBasis(void* highs, const HighsInt* col_status,
                        const HighsInt* row_status) {
  HighsBasis basis;
  const HighsInt num__col = Highs_getNumCol(highs);
  if (num__col > 0) {
    basis.col_status.resize(num__col);
    for (HighsInt i = 0; i < num__col; i++) {
      if (col_status[i] == (HighsInt)HighsBasisStatus::kLower) {
        basis.col_status[i] = HighsBasisStatus::kLower;
      } else if (col_status[i] == (HighsInt)HighsBasisStatus::kBasic) {
        basis.col_status[i] = HighsBasisStatus::kBasic;
      } else if (col_status[i] == (HighsInt)HighsBasisStatus::kUpper) {
        basis.col_status[i] = HighsBasisStatus::kUpper;
      } else if (col_status[i] == (HighsInt)HighsBasisStatus::kZero) {
        basis.col_status[i] = HighsBasisStatus::kZero;
      } else if (col_status[i] == (HighsInt)HighsBasisStatus::kNonbasic) {
        basis.col_status[i] = HighsBasisStatus::kNonbasic;
      } else {
        return (HighsInt)HighsStatus::kError;
      }
    }
  }
  const HighsInt num__row = Highs_getNumRow(highs);
  if (num__row > 0) {
    basis.row_status.resize(num__row);
    for (HighsInt i = 0; i < num__row; i++) {
      if (row_status[i] == (HighsInt)HighsBasisStatus::kLower) {
        basis.row_status[i] = HighsBasisStatus::kLower;
      } else if (row_status[i] == (HighsInt)HighsBasisStatus::kBasic) {
        basis.row_status[i] = HighsBasisStatus::kBasic;
      } else if (row_status[i] == (HighsInt)HighsBasisStatus::kUpper) {
        basis.row_status[i] = HighsBasisStatus::kUpper;
      } else if (row_status[i] == (HighsInt)HighsBasisStatus::kZero) {
        basis.row_status[i] = HighsBasisStatus::kZero;
      } else if (row_status[i] == (HighsInt)HighsBasisStatus::kNonbasic) {
        basis.row_status[i] = HighsBasisStatus::kNonbasic;
      } else {
        return (HighsInt)HighsStatus::kError;
      }
    }
  }
  return (HighsInt)((Highs*)highs)->setBasis(basis);
}

HighsInt Highs_setLogicalBasis(void* highs) {
  return (HighsInt)((Highs*)highs)->setBasis();
}

HighsInt Highs_setSolution(void* highs, const double* col_value,
                           const double* row_value, const double* col_dual,
                           const double* row_dual) {
  HighsSolution solution;
  const HighsInt num__col = Highs_getNumCol(highs);
  if (num__col > 0) {
    if (col_value) {
      solution.col_value.resize(num__col);
      for (HighsInt i = 0; i < num__col; i++)
        solution.col_value[i] = col_value[i];
    }
    if (col_dual) {
      solution.col_dual.resize(num__col);
      for (HighsInt i = 0; i < num__col; i++)
        solution.col_dual[i] = col_dual[i];
    }
  }
  const HighsInt num__row = Highs_getNumRow(highs);
  if (num__row > 0) {
    if (row_value) {
      solution.row_value.resize(num__row);
      for (HighsInt i = 0; i < num__row; i++)
        solution.row_value[i] = row_value[i];
    }
    if (row_dual) {
      solution.row_dual.resize(num__row);
      for (HighsInt i = 0; i < num__row; i++)
        solution.row_dual[i] = row_dual[i];
    }
  }

  return (HighsInt)((Highs*)highs)->setSolution(solution);
}

double Highs_getRunTime(const void* highs) {
  return (double)((Highs*)highs)->getRunTime();
}

HighsInt Highs_addCol(void* highs, const double cost, const double lower,
                      const double upper, const HighsInt num_new_nz,
                      const HighsInt* index, const double* value) {
  return (HighsInt)((Highs*)highs)
      ->addCol(cost, lower, upper, num_new_nz, index, value);
}

HighsInt Highs_addCols(void* highs, const HighsInt num_new_col,
                       const double* costs, const double* lower,
                       const double* upper, const HighsInt num_new_nz,
                       const HighsInt* starts, const HighsInt* index,
                       const double* value) {
  return (HighsInt)((Highs*)highs)
      ->addCols(num_new_col, costs, lower, upper, num_new_nz, starts, index,
                value);
}

HighsInt Highs_addVar(void* highs, const double lower, const double upper) {
  return (HighsInt)((Highs*)highs)->addVar(lower, upper);
}

HighsInt Highs_addVars(void* highs, const HighsInt num_new_var,
                       const double* lower, const double* upper) {
  return (HighsInt)((Highs*)highs)->addVars(num_new_var, lower, upper);
}

HighsInt Highs_addRow(void* highs, const double lower, const double upper,
                      const HighsInt num_new_nz, const HighsInt* index,
                      const double* value) {
  return (HighsInt)((Highs*)highs)
      ->addRow(lower, upper, num_new_nz, index, value);
}

HighsInt Highs_addRows(void* highs, const HighsInt num_new_row,
                       const double* lower, const double* upper,
                       const HighsInt num_new_nz, const HighsInt* starts,
                       const HighsInt* index, const double* value) {
  return (HighsInt)((Highs*)highs)
      ->addRows(num_new_row, lower, upper, num_new_nz, starts, index, value);
}

HighsInt Highs_changeObjectiveSense(void* highs, const HighsInt sense) {
  ObjSense pass_sense = ObjSense::kMinimize;
  if (sense == (HighsInt)ObjSense::kMaximize) pass_sense = ObjSense::kMaximize;
  return (HighsInt)((Highs*)highs)->changeObjectiveSense(pass_sense);
}

HighsInt Highs_changeObjectiveOffset(void* highs, const double offset) {
  return (HighsInt)((Highs*)highs)->changeObjectiveOffset(offset);
}

HighsInt Highs_changeColIntegrality(void* highs, const HighsInt col,
                                    const HighsInt integrality) {
  return (HighsInt)((Highs*)highs)
      ->changeColIntegrality(col, (HighsVarType)integrality);
}

HighsInt Highs_changeColsIntegralityByRange(void* highs,
                                            const HighsInt from_col,
                                            const HighsInt to_col,
                                            const HighsInt* integrality) {
  vector<HighsVarType> pass_integrality;
  HighsInt num_ix = to_col - from_col + 1;
  if (num_ix > 0) {
    pass_integrality.resize(num_ix);
    for (HighsInt ix = 0; ix < num_ix; ix++) {
      pass_integrality[ix] = (HighsVarType)integrality[ix];
    }
  }
  return (HighsInt)((Highs*)highs)
      ->changeColsIntegrality(from_col, to_col, &pass_integrality[0]);
}

HighsInt Highs_changeColsIntegralityBySet(void* highs,
                                          const HighsInt num_set_entries,
                                          const HighsInt* set,
                                          const HighsInt* integrality) {
  vector<HighsVarType> pass_integrality;
  if (num_set_entries > 0) {
    pass_integrality.resize(num_set_entries);
    for (HighsInt ix = 0; ix < num_set_entries; ix++) {
      pass_integrality[ix] = (HighsVarType)integrality[ix];
    }
  }
  return (HighsInt)((Highs*)highs)
      ->changeColsIntegrality(num_set_entries, set, &pass_integrality[0]);
}

HighsInt Highs_changeColsIntegralityByMask(void* highs, const HighsInt* mask,
                                           const HighsInt* integrality) {
  const HighsInt num__col = Highs_getNumCol(highs);
  vector<HighsVarType> pass_integrality;
  if (num__col > 0) {
    pass_integrality.resize(num__col);
    for (HighsInt iCol = 0; iCol < num__col; iCol++) {
      pass_integrality[iCol] = (HighsVarType)integrality[iCol];
    }
  }
  return (HighsInt)((Highs*)highs)
      ->changeColsIntegrality(mask, &pass_integrality[0]);
}

HighsInt Highs_changeColCost(void* highs, const HighsInt col,
                             const double cost) {
  return (HighsInt)((Highs*)highs)->changeColCost(col, cost);
}

HighsInt Highs_changeColsCostByRange(void* highs, const HighsInt from_col,
                                     const HighsInt to_col,
                                     const double* cost) {
  return (HighsInt)((Highs*)highs)->changeColsCost(from_col, to_col, cost);
}

HighsInt Highs_changeColsCostBySet(void* highs, const HighsInt num_set_entries,
                                   const HighsInt* set, const double* cost) {
  return (HighsInt)((Highs*)highs)->changeColsCost(num_set_entries, set, cost);
}

HighsInt Highs_changeColsCostByMask(void* highs, const HighsInt* mask,
                                    const double* cost) {
  return (HighsInt)((Highs*)highs)->changeColsCost(mask, cost);
}

HighsInt Highs_changeColBounds(void* highs, const HighsInt col,
                               const double lower, const double upper) {
  return (HighsInt)((Highs*)highs)->changeColBounds(col, lower, upper);
}

HighsInt Highs_changeColsBoundsByRange(void* highs, const HighsInt from_col,
                                       const HighsInt to_col,
                                       const double* lower,
                                       const double* upper) {
  return (HighsInt)((Highs*)highs)
      ->changeColsBounds(from_col, to_col, lower, upper);
}

HighsInt Highs_changeColsBoundsBySet(void* highs,
                                     const HighsInt num_set_entries,
                                     const HighsInt* set, const double* lower,
                                     const double* upper) {
  return (HighsInt)((Highs*)highs)
      ->changeColsBounds(num_set_entries, set, lower, upper);
}

HighsInt Highs_changeColsBoundsByMask(void* highs, const HighsInt* mask,
                                      const double* lower,
                                      const double* upper) {
  return (HighsInt)((Highs*)highs)->changeColsBounds(mask, lower, upper);
}

HighsInt Highs_changeRowBounds(void* highs, const HighsInt row,
                               const double lower, const double upper) {
  return (HighsInt)((Highs*)highs)->changeRowBounds(row, lower, upper);
}

HighsInt Highs_changeRowsBoundsBySet(void* highs,
                                     const HighsInt num_set_entries,
                                     const HighsInt* set, const double* lower,
                                     const double* upper) {
  return (HighsInt)((Highs*)highs)
      ->changeRowsBounds(num_set_entries, set, lower, upper);
}

HighsInt Highs_changeRowsBoundsByMask(void* highs, const HighsInt* mask,
                                      const double* lower,
                                      const double* upper) {
  return (HighsInt)((Highs*)highs)->changeRowsBounds(mask, lower, upper);
}

HighsInt Highs_changeCoeff(void* highs, const HighsInt row, const HighsInt col,
                           const double value) {
  return (HighsInt)((Highs*)highs)->changeCoeff(row, col, value);
}

HighsInt Highs_getObjectiveSense(const void* highs, HighsInt* sense) {
  ObjSense get_sense;
  HighsStatus status = ((Highs*)highs)->getObjectiveSense(get_sense);
  *sense = (HighsInt)get_sense;
  return (HighsInt)status;
}

HighsInt Highs_getObjectiveOffset(const void* highs, double* offset) {
  return (HighsInt)((Highs*)highs)->getObjectiveOffset(*offset);
}

HighsInt Highs_getColsByRange(const void* highs, const HighsInt from_col,
                              const HighsInt to_col, HighsInt* num_col,
                              double* costs, double* lower, double* upper,
                              HighsInt* num_nz, HighsInt* matrix_start,
                              HighsInt* matrix_index, double* matrix_value) {
  HighsInt local_num_col, local_num_nz;
  HighsStatus status =
      ((Highs*)highs)
          ->getCols(from_col, to_col, local_num_col, costs, lower, upper,
                    local_num_nz, matrix_start, matrix_index, matrix_value);
  *num_col = local_num_col;
  *num_nz = local_num_nz;
  return (HighsInt)status;
}

HighsInt Highs_getColsBySet(const void* highs, const HighsInt num_set_entries,
                            const HighsInt* set, HighsInt* num_col,
                            double* costs, double* lower, double* upper,
                            HighsInt* num_nz, HighsInt* matrix_start,
                            HighsInt* matrix_index, double* matrix_value) {
  HighsInt local_num_col, local_num_nz;
  HighsStatus status =
      ((Highs*)highs)
          ->getCols(num_set_entries, set, local_num_col, costs, lower, upper,
                    local_num_nz, matrix_start, matrix_index, matrix_value);
  *num_col = local_num_col;
  *num_nz = local_num_nz;
  return (HighsInt)status;
}

HighsInt Highs_getColsByMask(const void* highs, const HighsInt* mask,
                             HighsInt* num_col, double* costs, double* lower,
                             double* upper, HighsInt* num_nz,
                             HighsInt* matrix_start, HighsInt* matrix_index,
                             double* matrix_value) {
  HighsInt local_num_col, local_num_nz;
  HighsStatus status =
      ((Highs*)highs)
          ->getCols(mask, local_num_col, costs, lower, upper, local_num_nz,
                    matrix_start, matrix_index, matrix_value);
  *num_col = local_num_col;
  *num_nz = local_num_nz;
  return (HighsInt)status;
}

HighsInt Highs_getRowsByRange(const void* highs, const HighsInt from_row,
                              const HighsInt to_row, HighsInt* num_row,
                              double* lower, double* upper, HighsInt* num_nz,
                              HighsInt* matrix_start, HighsInt* matrix_index,
                              double* matrix_value) {
  HighsInt local_num_row, local_num_nz;
  HighsStatus status =
      ((Highs*)highs)
          ->getRows(from_row, to_row, local_num_row, lower, upper, local_num_nz,
                    matrix_start, matrix_index, matrix_value);
  *num_row = local_num_row;
  *num_nz = local_num_nz;
  return (HighsInt)status;
}

HighsInt Highs_getRowsBySet(const void* highs, const HighsInt num_set_entries,
                            const HighsInt* set, HighsInt* num_row,
                            double* lower, double* upper, HighsInt* num_nz,
                            HighsInt* matrix_start, HighsInt* matrix_index,
                            double* matrix_value) {
  HighsInt local_num_row, local_num_nz;
  HighsStatus status =
      ((Highs*)highs)
          ->getRows(num_set_entries, set, local_num_row, lower, upper,
                    local_num_nz, matrix_start, matrix_index, matrix_value);
  *num_row = local_num_row;
  *num_nz = local_num_nz;
  return (HighsInt)status;
}

HighsInt Highs_getRowsByMask(const void* highs, const HighsInt* mask,
                             HighsInt* num_row, double* lower, double* upper,
                             HighsInt* num_nz, HighsInt* matrix_start,
                             HighsInt* matrix_index, double* matrix_value) {
  HighsInt local_num_row, local_num_nz;
  HighsStatus status =
      ((Highs*)highs)
          ->getRows(mask, local_num_row, lower, upper, local_num_nz,
                    matrix_start, matrix_index, matrix_value);
  *num_row = local_num_row;
  *num_nz = local_num_nz;
  return (HighsInt)status;
}

HighsInt Highs_deleteColsByRange(void* highs, const HighsInt from_col,
                                 const HighsInt to_col) {
  return (HighsInt)((Highs*)highs)->deleteCols(from_col, to_col);
}

HighsInt Highs_deleteColsBySet(void* highs, const HighsInt num_set_entries,
                               const HighsInt* set) {
  return (HighsInt)((Highs*)highs)->deleteCols(num_set_entries, set);
}

HighsInt Highs_deleteColsByMask(void* highs, HighsInt* mask) {
  return (HighsInt)((Highs*)highs)->deleteCols(mask);
}

HighsInt Highs_deleteRowsByRange(void* highs, const HighsInt from_row,
                                 const HighsInt to_row) {
  return (HighsInt)((Highs*)highs)->deleteRows(from_row, to_row);
}

HighsInt Highs_deleteRowsBySet(void* highs, const HighsInt num_set_entries,
                               const HighsInt* set) {
  return (HighsInt)((Highs*)highs)->deleteRows(num_set_entries, set);
}

HighsInt Highs_deleteRowsByMask(void* highs, HighsInt* mask) {
  return (HighsInt)((Highs*)highs)->deleteRows(mask);
}

HighsInt Highs_scaleCol(void* highs, const HighsInt col,
                        const double scaleval) {
  return (HighsInt)((Highs*)highs)->scaleCol(col, scaleval);
}

HighsInt Highs_scaleRow(void* highs, const HighsInt row,
                        const double scaleval) {
  return (HighsInt)((Highs*)highs)->scaleRow(row, scaleval);
}

double Highs_getInfinity(const void* highs) {
  return ((Highs*)highs)->getInfinity();
}

HighsInt Highs_getNumCol(const void* highs) {
  return ((Highs*)highs)->getNumCol();
}

HighsInt Highs_getNumRow(const void* highs) {
  return ((Highs*)highs)->getNumRow();
}

HighsInt Highs_getNumNz(const void* highs) {
  return ((Highs*)highs)->getNumNz();
}

HighsInt Highs_getHessianNumNz(const void* highs) {
  return ((Highs*)highs)->getHessianNumNz();
}

HighsInt Highs_getModel(const void* highs, const HighsInt a_format,
                        const HighsInt q_format, HighsInt* num_col,
                        HighsInt* num_row, HighsInt* num_nz, HighsInt* q_num_nz,
                        HighsInt* sense, double* offset, double* col_cost,
                        double* col_lower, double* col_upper, double* row_lower,
                        double* row_upper, HighsInt* a_start, HighsInt* a_index,
                        double* a_value, HighsInt* q_start, HighsInt* q_index,
                        double* q_value, HighsInt* integrality) {
  const HighsModel& model = ((Highs*)highs)->getModel();
  const HighsLp& lp = model.lp_;
  const HighsHessian& hessian = model.hessian_;
  ObjSense obj_sense = ObjSense::kMinimize;
  *sense = (HighsInt)obj_sense;
  *offset = lp.offset_;
  *num_col = lp.num_col_;
  *num_row = lp.num_row_;
  if (*num_col > 0) {
    memcpy(col_cost, &lp.col_cost_[0], *num_col * sizeof(double));
    memcpy(col_lower, &lp.col_lower_[0], *num_col * sizeof(double));
    memcpy(col_upper, &lp.col_upper_[0], *num_col * sizeof(double));
  }
  if (*num_row > 0) {
    memcpy(row_lower, &lp.row_lower_[0], *num_row * sizeof(double));
    memcpy(row_upper, &lp.row_upper_[0], *num_row * sizeof(double));
  }

  // Save the original orientation so that it is recovered
  MatrixFormat original_a_format = lp.a_matrix_.format_;
  // Determine the desired orientation and number of start entries to
  // be copied
  MatrixFormat desired_a_format = MatrixFormat::kColwise;
  HighsInt num_start_entries = *num_col;
  if (a_format == (HighsInt)MatrixFormat::kRowwise) {
    desired_a_format = MatrixFormat::kRowwise;
    num_start_entries = *num_row;
  }
  // Ensure the desired orientation
  HighsInt return_status;
  return_status = (HighsInt)((Highs*)highs)->setMatrixFormat(desired_a_format);
  if (return_status != kHighsStatusOk) return return_status;

  if (*num_col > 0 && *num_row > 0) {
    *num_nz = lp.a_matrix_.numNz();
    memcpy(a_start, &lp.a_matrix_.start_[0],
           num_start_entries * sizeof(HighsInt));
    memcpy(a_index, &lp.a_matrix_.index_[0], *num_nz * sizeof(HighsInt));
    memcpy(a_value, &lp.a_matrix_.value_[0], *num_nz * sizeof(double));
  }
  if (hessian.dim_ > 0) {
    *q_num_nz = hessian.start_[*num_col];
    memcpy(q_start, &hessian.start_[0], *num_col * sizeof(HighsInt));
    memcpy(q_index, &hessian.index_[0], *q_num_nz * sizeof(HighsInt));
    memcpy(q_value, &hessian.value_[0], *q_num_nz * sizeof(double));
  }
  if ((HighsInt)lp.integrality_.size()) {
    for (int iCol = 0; iCol < *num_col; iCol++)
      integrality[iCol] = (HighsInt)lp.integrality_[iCol];
  }
  // Restore the original orientation
  return_status = (HighsInt)((Highs*)highs)->setMatrixFormat(original_a_format);
  if (return_status != kHighsStatusOk) return return_status;
  return kHighsStatusOk;
}

HighsInt Highs_crossover(void* highs, const int num_col, const int num_row,
                         const double* col_value, const double* col_dual,
                         const double* row_dual) {
  HighsSolution solution;
  if (col_value) {
    solution.value_valid = true;
    solution.col_value.resize(num_col);
    for (int col = 0; col < num_col; col++)
      solution.col_value[col] = col_value[col];
  }

  if (col_dual && row_dual) {
    solution.dual_valid = true;
    solution.col_dual.resize(num_col);
    solution.row_dual.resize(num_row);
    for (int col = 0; col < num_col; col++)
      solution.col_dual[col] = col_dual[col];
    for (int row = 0; row < num_row; row++)
      solution.row_dual[row] = row_dual[row];
  }

  return (HighsInt)((Highs*)highs)->crossover(solution);
}

void Highs_resetGlobalScheduler(HighsInt blocking) {
  Highs::resetGlobalScheduler(blocking);
}

// *********************
// * Deprecated methods*
// *********************

HighsInt Highs_call(const HighsInt num_col, const HighsInt num_row,
                    const HighsInt num_nz, const double* col_cost,
                    const double* col_lower, const double* col_upper,
                    const double* row_lower, const double* row_upper,
                    const HighsInt* a_start, const HighsInt* a_index,
                    const double* a_value, double* col_value, double* col_dual,
                    double* row_value, double* row_dual,
                    HighsInt* col_basis_status, HighsInt* row_basis_status,
                    HighsInt* model_status) {
  printf(
      "Method Highs_call is deprecated: alternative method is Highs_lpCall\n");
  const HighsInt aformat_columnwise = 1;
  const HighsInt sense = 1;
  const double offset = 0;
  return Highs_lpCall(num_col, num_row, num_nz, aformat_columnwise, sense,
                      offset, col_cost, col_lower, col_upper, row_lower,
                      row_upper, a_start, a_index, a_value, col_value, col_dual,
                      row_value, row_dual, col_basis_status, row_basis_status,
                      model_status);
}

HighsInt Highs_setOptionValue(void* highs, const char* option,
                              const char* value) {
  ((Highs*)highs)
      ->deprecationMessage("Highs_setOptionValue",
                           "Highs_setStringOptionValue");
  return (HighsInt)((Highs*)highs)
      ->setOptionValue(std::string(option), std::string(value));
}

HighsInt Highs_runQuiet(void* highs) {
  ((Highs*)highs)->deprecationMessage("Highs_runQuiet", "None");
  return (HighsInt)((Highs*)highs)->setOptionValue("output_flag", false);
}

HighsInt Highs_setHighsLogfile(void* highs, const void* logfile) {
  ((Highs*)highs)->deprecationMessage("Highs_setHighsLogfile", "None");
  return (HighsInt)((Highs*)highs)->setOptionValue("output_flag", false);
}

HighsInt Highs_setHighsOutput(void* highs, const void* outputfile) {
  ((Highs*)highs)->deprecationMessage("Highs_setHighsOutput", "None");
  return (HighsInt)((Highs*)highs)->setOptionValue("output_flag", false);
}

HighsInt Highs_getIterationCount(const void* highs) {
  ((Highs*)highs)
      ->deprecationMessage("Highs_getIterationCount", "Highs_getIntInfoValue");
  return (HighsInt)((Highs*)highs)->getInfo().simplex_iteration_count;
}

HighsInt Highs_getSimplexIterationCount(const void* highs) {
  ((Highs*)highs)
      ->deprecationMessage("Highs_getSimplexIterationCount",
                           "Highs_getIntInfoValue");
  return (HighsInt)((Highs*)highs)->getInfo().simplex_iteration_count;
}

HighsInt Highs_setHighsBoolOptionValue(void* highs, const char* option,
                                       const HighsInt value) {
  ((Highs*)highs)
      ->deprecationMessage("Highs_setHighsBoolOptionValue",
                           "Highs_setBoolOptionValue");
  return Highs_setBoolOptionValue(highs, option, value);
}

HighsInt Highs_setHighsIntOptionValue(void* highs, const char* option,
                                      const HighsInt value) {
  ((Highs*)highs)
      ->deprecationMessage("Highs_setHighsIntOptionValue",
                           "Highs_setIntOptionValue");
  return Highs_setIntOptionValue(highs, option, value);
}

HighsInt Highs_setHighsDoubleOptionValue(void* highs, const char* option,
                                         const double value) {
  ((Highs*)highs)
      ->deprecationMessage("Highs_setHighsDoubleOptionValue",
                           "Highs_setDoubleOptionValue");
  return Highs_setDoubleOptionValue(highs, option, value);
}

HighsInt Highs_setHighsStringOptionValue(void* highs, const char* option,
                                         const char* value) {
  ((Highs*)highs)
      ->deprecationMessage("Highs_setHighsStringOptionValue",
                           "Highs_setStringOptionValue");
  return Highs_setStringOptionValue(highs, option, value);
}

HighsInt Highs_setHighsOptionValue(void* highs, const char* option,
                                   const char* value) {
  ((Highs*)highs)
      ->deprecationMessage("Highs_setHighsOptionValue", "Highs_setOptionValue");
  return Highs_setOptionValue(highs, option, value);
}

HighsInt Highs_getHighsBoolOptionValue(const void* highs, const char* option,
                                       HighsInt* value) {
  ((Highs*)highs)
      ->deprecationMessage("Highs_getHighsBoolOptionValue",
                           "Highs_getBoolOptionValue");
  return Highs_getBoolOptionValue(highs, option, value);
}

HighsInt Highs_getHighsIntOptionValue(const void* highs, const char* option,
                                      HighsInt* value) {
  ((Highs*)highs)
      ->deprecationMessage("Highs_getHighsIntOptionValue",
                           "Highs_getIntOptionValue");
  return Highs_getIntOptionValue(highs, option, value);
}

HighsInt Highs_getHighsDoubleOptionValue(const void* highs, const char* option,
                                         double* value) {
  ((Highs*)highs)
      ->deprecationMessage("Highs_getHighsDoubleOptionValue",
                           "Highs_getDoubleOptionValue");
  return Highs_getDoubleOptionValue(highs, option, value);
}

HighsInt Highs_getHighsStringOptionValue(const void* highs, const char* option,
                                         char* value) {
  ((Highs*)highs)
      ->deprecationMessage("Highs_getHighsStringOptionValue",
                           "Highs_getStringOptionValue");
  return Highs_getStringOptionValue(highs, option, value);
}

HighsInt Highs_getHighsOptionType(const void* highs, const char* option,
                                  HighsInt* type) {
  ((Highs*)highs)
      ->deprecationMessage("Highs_getHighsOptionType", "Highs_getOptionType");
  return Highs_getOptionType(highs, option, type);
}

HighsInt Highs_resetHighsOptions(void* highs) {
  ((Highs*)highs)
      ->deprecationMessage("Highs_resetHighsOptions", "Highs_resetOptions");
  return Highs_resetOptions(highs);
}

HighsInt Highs_getHighsIntInfoValue(const void* highs, const char* info,
                                    HighsInt* value) {
  ((Highs*)highs)
      ->deprecationMessage("Highs_getHighsIntInfoValue",
                           "Highs_getIntInfoValue");
  return Highs_getIntInfoValue(highs, info, value);
}

HighsInt Highs_getHighsDoubleInfoValue(const void* highs, const char* info,
                                       double* value) {
  ((Highs*)highs)
      ->deprecationMessage("Highs_getHighsDoubleInfoValue",
                           "Highs_getDoubleInfoValue");
  return Highs_getDoubleInfoValue(highs, info, value);
}

HighsInt Highs_getNumCols(const void* highs) { return Highs_getNumCol(highs); }
HighsInt Highs_getNumRows(const void* highs) { return Highs_getNumRow(highs); }

double Highs_getHighsRunTime(const void* highs) {
  ((Highs*)highs)
      ->deprecationMessage("Highs_getHighsRunTime", "Highs_getRunTime");
  return Highs_getRunTime(highs);
}

double Highs_getHighsInfinity(const void* highs) {
  ((Highs*)highs)
      ->deprecationMessage("Highs_getHighsInfinity", "Highs_getInfinity");
  return Highs_getInfinity(highs);
}

HighsInt Highs_getScaledModelStatus(const void* highs) {
  return (HighsInt)((Highs*)highs)->getModelStatus(true);
}
