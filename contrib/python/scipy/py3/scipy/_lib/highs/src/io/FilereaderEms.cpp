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
/**@file io/FilereaderEms.cpp
 * @brief
 */

#include "io/FilereaderEms.h"

#include <fstream>
#include <iomanip>

#include "lp_data/HConst.h"
#include "lp_data/HighsLpUtils.h"
#include "util/stringutil.h"

FilereaderRetcode FilereaderEms::readModelFromFile(const HighsOptions& options,
                                                   const std::string filename,
                                                   HighsModel& model) {
  std::ifstream f;
  HighsInt i;

  HighsLp& lp = model.lp_;
  f.open(filename, std::ios::in);
  if (f.is_open()) {
    std::string line;
    HighsInt numCol, numRow, AcountX, num_int;
    bool indices_from_one = false;

    // counts
    std::getline(f, line);
    if (trim(line) != "n_rows") {
      while (trim(line) != "n_rows" && f) std::getline(f, line);
      indices_from_one = true;
    }
    if (!f) {
      highsLogUser(options.log_options, HighsLogType::kError,
                   "n_rows not found in EMS file\n");
      return FilereaderRetcode::kParserError;
    }
    f >> numRow;

    std::getline(f, line);
    while (trim(line) == "") std::getline(f, line);
    if (trim(line) != "n_columns") {
      highsLogUser(options.log_options, HighsLogType::kError,
                   "n_columns not found in EMS file\n");
      return FilereaderRetcode::kParserError;
    }
    f >> numCol;

    std::getline(f, line);
    while (trim(line) == "") std::getline(f, line);
    if (trim(line) != "n_matrix_elements") {
      highsLogUser(options.log_options, HighsLogType::kError,
                   "n_matrix_elements not found in EMS file\n");
      return FilereaderRetcode::kParserError;
    }
    f >> AcountX;

    lp.num_col_ = numCol;
    lp.num_row_ = numRow;

    // matrix
    std::getline(f, line);
    while (trim(line) == "") std::getline(f, line);
    if (trim(line) != "matrix") {
      highsLogUser(options.log_options, HighsLogType::kError,
                   "matrix not found in EMS file\n");
      return FilereaderRetcode::kParserError;
    }
    lp.a_matrix_.format_ = MatrixFormat::kColwise;
    lp.a_matrix_.start_.resize(numCol + 1);
    lp.a_matrix_.index_.resize(AcountX);
    lp.a_matrix_.value_.resize(AcountX);

    for (i = 0; i < numCol + 1; i++) {
      f >> lp.a_matrix_.start_[i];
      if (indices_from_one) lp.a_matrix_.start_[i]--;
    }

    for (i = 0; i < AcountX; i++) {
      f >> lp.a_matrix_.index_[i];
      if (indices_from_one) lp.a_matrix_.index_[i]--;
    }

    for (i = 0; i < AcountX; i++) f >> lp.a_matrix_.value_[i];

    // cost and bounds
    std::getline(f, line);
    while (trim(line) == "") std::getline(f, line);
    if (trim(line) != "column_bounds") {
      highsLogUser(options.log_options, HighsLogType::kError,
                   "column_bounds not found in EMS file\n");
      return FilereaderRetcode::kParserError;
    }
    lp.col_lower_.reserve(numCol);
    lp.col_upper_.reserve(numCol);

    lp.col_lower_.assign(numCol, -kHighsInf);
    lp.col_upper_.assign(numCol, kHighsInf);

    for (i = 0; i < numCol; i++) {
      f >> lp.col_lower_[i];
    }

    for (i = 0; i < numCol; i++) {
      f >> lp.col_upper_[i];
    }

    std::getline(f, line);
    while (trim(line) == "") std::getline(f, line);
    if (trim(line) != "row_bounds") {
      highsLogUser(options.log_options, HighsLogType::kError,
                   "row_bounds not found in EMS file\n");
      return FilereaderRetcode::kParserError;
    }
    lp.row_lower_.reserve(numRow);
    lp.row_upper_.reserve(numRow);
    lp.row_lower_.assign(numRow, -kHighsInf);
    lp.row_upper_.assign(numRow, kHighsInf);

    for (i = 0; i < numRow; i++) {
      f >> lp.row_lower_[i];
    }

    for (i = 0; i < numRow; i++) {
      f >> lp.row_upper_[i];
    }

    std::getline(f, line);
    while (trim(line) == "") std::getline(f, line);
    if (trim(line) != "column_costs") {
      highsLogUser(options.log_options, HighsLogType::kError,
                   "column_costs not found in EMS file\n");
      return FilereaderRetcode::kParserError;
    }
    lp.col_cost_.reserve(numCol);
    lp.col_cost_.assign(numCol, 0);
    for (i = 0; i < numCol; i++) {
      f >> lp.col_cost_[i];
    }

    // Get the next keyword
    std::getline(f, line);
    while (trim(line) == "" && f) std::getline(f, line);

    if (trim(line) == "integer_columns") {
      f >> num_int;
      if (num_int) {
        lp.integrality_.resize(lp.num_col_, HighsVarType::kContinuous);
        HighsInt iCol;
        for (i = 0; i < num_int; i++) {
          f >> iCol;
          if (indices_from_one) iCol--;
          lp.integrality_[iCol] = HighsVarType::kInteger;
        }
      }
      // Get the next keyword. If there's no integer_columns section
      // then it will already have been read
      std::getline(f, line);
      while (trim(line) == "" && f) std::getline(f, line);
    }

    // Act if the next keyword is end_linear
    if (trim(line) == "end_linear") {
      // File read completed OK
      f.close();
      lp.ensureColwise();
    }

    // Act if the next keyword is names
    if (trim(line) == "names") {
      // Ignore length since we support any length.
      std::getline(f, line);
      if (trim(line) != "columns") std::getline(f, line);
      if (trim(line) != "columns") return FilereaderRetcode::kParserError;

      lp.row_names_.resize(numRow);
      lp.col_names_.resize(numCol);

      for (i = 0; i < numCol; i++) {
        std::getline(f, line);
        lp.col_names_[i] = trim(line);
      }

      std::getline(f, line);
      if (trim(line) != "rows") return FilereaderRetcode::kParserError;

      for (i = 0; i < numRow; i++) {
        std::getline(f, line);
        lp.row_names_[i] = trim(line);
      }
    } else {
      // OK if file just ends after the integer_columns section without
      // end_linear
      if (!f) lp.ensureColwise();
      highsLogUser(options.log_options, HighsLogType::kError,
                   "names not found in EMS file\n");
      return FilereaderRetcode::kParserError;
    }
    f.close();
  } else {
    highsLogUser(options.log_options, HighsLogType::kError,
                 "EMS file not found\n");
    return FilereaderRetcode::kFileNotFound;
  }
  lp.ensureColwise();
  return FilereaderRetcode::kOk;
}

HighsStatus FilereaderEms::writeModelToFile(const HighsOptions& options,
                                            const std::string filename,
                                            const HighsModel& model) {
  std::ofstream f;
  f.open(filename, std::ios::out);
  const HighsLp& lp = model.lp_;
  HighsInt num_nz = lp.a_matrix_.start_[lp.num_col_];

  // counts
  f << "n_rows" << std::endl;
  f << lp.num_row_ << std::endl;
  f << "n_columns" << std::endl;
  f << lp.num_col_ << std::endl;
  f << "n_matrix_elements" << std::endl;
  f << num_nz << std::endl;

  // matrix
  f << "matrix" << std::endl;
  for (HighsInt i = 0; i < lp.num_col_ + 1; i++)
    f << lp.a_matrix_.start_[i] << " ";
  f << std::endl;

  for (HighsInt i = 0; i < num_nz; i++) f << lp.a_matrix_.index_[i] << " ";
  f << std::endl;

  f << std::setprecision(9);
  for (HighsInt i = 0; i < num_nz; i++) f << lp.a_matrix_.value_[i] << " ";
  f << std::endl;

  // cost and bounds
  f << std::setprecision(9);

  f << "column_bounds" << std::endl;
  for (HighsInt i = 0; i < lp.num_col_; i++) f << lp.col_lower_[i] << " ";
  f << std::endl;

  for (HighsInt i = 0; i < lp.num_col_; i++) f << lp.col_upper_[i] << " ";
  f << std::endl;

  f << "row_bounds" << std::endl;
  f << std::setprecision(9);
  for (HighsInt i = 0; i < lp.num_row_; i++) f << lp.row_lower_[i] << " ";
  f << std::endl;

  for (HighsInt i = 0; i < lp.num_row_; i++) f << lp.row_upper_[i] << " ";
  f << std::endl;

  f << "column_costs" << std::endl;
  for (HighsInt i = 0; i < lp.num_col_; i++)
    f << (HighsInt)lp.sense_ * lp.col_cost_[i] << " ";
  f << std::endl;

  if (lp.row_names_.size() > 0 && lp.col_names_.size() > 0) {
    f << "names" << std::endl;

    f << "columns" << std::endl;
    for (HighsInt i = 0; i < (HighsInt)lp.col_names_.size(); i++)
      f << lp.col_names_[i] << std::endl;

    f << "rows" << std::endl;
    for (HighsInt i = 0; i < (HighsInt)lp.row_names_.size(); i++)
      f << lp.row_names_[i] << std::endl;
  }

  // todo: integer variables.

  if (lp.offset_ != 0)
    f << "shift" << std::endl << (HighsInt)lp.sense_ * lp.offset_ << std::endl;

  f << std::endl;
  f.close();
  return HighsStatus::kOk;
}
