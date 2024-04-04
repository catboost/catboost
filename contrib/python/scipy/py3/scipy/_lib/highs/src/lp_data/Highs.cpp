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
/**@file lp_data/Highs.cpp
 * @brief
 */
#include "Highs.h"

#include <algorithm>
#include <cassert>
#include <iostream>
#include <memory>
#include <sstream>

#include "io/Filereader.h"
#include "io/LoadOptions.h"
#include "lp_data/HighsInfoDebug.h"
#include "lp_data/HighsLpSolverObject.h"
#include "lp_data/HighsSolve.h"
#include "mip/HighsMipSolver.h"
#include "model/HighsHessianUtils.h"
#include "parallel/HighsParallel.h"
#include "presolve/ICrashX.h"
#include "qpsolver/quass.hpp"
#include "simplex/HSimplex.h"
#include "simplex/HSimplexDebug.h"
#include "util/HighsMatrixPic.h"
#include "util/HighsSort.h"

Highs::Highs() {}

HighsStatus Highs::clear() {
  resetOptions();
  return clearModel();
}

HighsStatus Highs::clearModel() {
  model_.clear();
  return clearSolver();
}

HighsStatus Highs::clearSolver() {
  HighsStatus return_status = HighsStatus::kOk;
  clearPresolve();
  invalidateUserSolverData();
  return returnFromHighs(return_status);
}

HighsStatus Highs::setOptionValue(const std::string& option, const bool value) {
  if (setLocalOptionValue(options_.log_options, option, options_.records,
                          value) == OptionStatus::kOk)
    return HighsStatus::kOk;
  return HighsStatus::kError;
}

HighsStatus Highs::setOptionValue(const std::string& option,
                                  const HighsInt value) {
  if (setLocalOptionValue(options_.log_options, option, options_.records,
                          value) == OptionStatus::kOk)
    return HighsStatus::kOk;
  return HighsStatus::kError;
}

HighsStatus Highs::setOptionValue(const std::string& option,
                                  const double value) {
  if (setLocalOptionValue(options_.log_options, option, options_.records,
                          value) == OptionStatus::kOk)
    return HighsStatus::kOk;
  return HighsStatus::kError;
}

HighsStatus Highs::setOptionValue(const std::string& option,
                                  const std::string& value) {
  HighsLogOptions report_log_options = options_.log_options;
  if (setLocalOptionValue(report_log_options, option, options_.log_options,
                          options_.records, value) == OptionStatus::kOk)
    return HighsStatus::kOk;
  return HighsStatus::kError;
}

HighsStatus Highs::setOptionValue(const std::string& option,
                                  const char* value) {
  HighsLogOptions report_log_options = options_.log_options;
  if (setLocalOptionValue(report_log_options, option, options_.log_options,
                          options_.records, value) == OptionStatus::kOk)
    return HighsStatus::kOk;
  return HighsStatus::kError;
}

HighsStatus Highs::readOptions(const std::string& filename) {
  if (filename.size() <= 0) {
    highsLogUser(options_.log_options, HighsLogType::kWarning,
                 "Empty file name so not reading options\n");
    return HighsStatus::kWarning;
  }
  HighsLogOptions report_log_options = options_.log_options;
  if (!loadOptionsFromFile(report_log_options, options_, filename))
    return HighsStatus::kError;
  return HighsStatus::kOk;
}

HighsStatus Highs::passOptions(const HighsOptions& options) {
  if (passLocalOptions(options_.log_options, options, options_) ==
      OptionStatus::kOk)
    return HighsStatus::kOk;
  return HighsStatus::kError;
}

HighsStatus Highs::getOptionValue(const std::string& option,
                                  bool& value) const {
  if (getLocalOptionValue(options_.log_options, option, options_.records,
                          value) == OptionStatus::kOk)
    return HighsStatus::kOk;
  return HighsStatus::kError;
}

HighsStatus Highs::getOptionValue(const std::string& option,
                                  HighsInt& value) const {
  if (getLocalOptionValue(options_.log_options, option, options_.records,
                          value) == OptionStatus::kOk)
    return HighsStatus::kOk;
  return HighsStatus::kError;
}

HighsStatus Highs::getOptionValue(const std::string& option,
                                  double& value) const {
  if (getLocalOptionValue(options_.log_options, option, options_.records,
                          value) == OptionStatus::kOk)
    return HighsStatus::kOk;
  return HighsStatus::kError;
}

HighsStatus Highs::getOptionValue(const std::string& option,
                                  std::string& value) const {
  if (getLocalOptionValue(options_.log_options, option, options_.records,
                          value) == OptionStatus::kOk)
    return HighsStatus::kOk;
  return HighsStatus::kError;
}

HighsStatus Highs::getOptionType(const std::string& option,
                                 HighsOptionType& type) const {
  if (getLocalOptionType(options_.log_options, option, options_.records,
                         type) == OptionStatus::kOk)
    return HighsStatus::kOk;
  return HighsStatus::kError;
}

HighsStatus Highs::resetOptions() {
  resetLocalOptions(options_.records);
  return HighsStatus::kOk;
}

HighsStatus Highs::writeOptions(const std::string& filename,
                                const bool report_only_deviations) const {
  HighsStatus return_status = HighsStatus::kOk;
  FILE* file;
  bool html;
  return_status = interpretCallStatus(
      options_.log_options, openWriteFile(filename, "writeOptions", file, html),
      return_status, "openWriteFile");
  if (return_status == HighsStatus::kError) return return_status;

  return_status = interpretCallStatus(
      options_.log_options,
      writeOptionsToFile(file, options_.records, report_only_deviations, html),
      return_status, "writeOptionsToFile");
  if (file != stdout) fclose(file);
  return return_status;
}

HighsStatus Highs::getInfoValue(const std::string& info,
                                HighsInt& value) const {
  InfoStatus status =
      getLocalInfoValue(options_, info, info_.valid, info_.records, value);
  if (status == InfoStatus::kOk) {
    return HighsStatus::kOk;
  } else if (status == InfoStatus::kUnavailable) {
    return HighsStatus::kWarning;
  } else {
    return HighsStatus::kError;
  }
}

#ifndef HIGHSINT64
HighsStatus Highs::getInfoValue(const std::string& info, int64_t& value) const {
  InfoStatus status =
      getLocalInfoValue(options_, info, info_.valid, info_.records, value);
  if (status == InfoStatus::kOk) {
    return HighsStatus::kOk;
  } else if (status == InfoStatus::kUnavailable) {
    return HighsStatus::kWarning;
  } else {
    return HighsStatus::kError;
  }
}
#endif

HighsStatus Highs::getInfoValue(const std::string& info, double& value) const {
  InfoStatus status =
      getLocalInfoValue(options_, info, info_.valid, info_.records, value);
  if (status == InfoStatus::kOk) {
    return HighsStatus::kOk;
  } else if (status == InfoStatus::kUnavailable) {
    return HighsStatus::kWarning;
  } else {
    return HighsStatus::kError;
  }
}

HighsStatus Highs::writeInfo(const std::string& filename) const {
  HighsStatus return_status = HighsStatus::kOk;
  FILE* file;
  bool html;
  return_status = interpretCallStatus(
      options_.log_options, openWriteFile(filename, "writeInfo", file, html),
      return_status, "openWriteFile");
  if (return_status == HighsStatus::kError) return return_status;

  return_status = interpretCallStatus(
      options_.log_options,
      writeInfoToFile(file, info_.valid, info_.records, html), return_status,
      "writeInfoToFile");
  if (file != stdout) fclose(file);
  return return_status;
}

// Methods below change the incumbent model or solver infomation
// associated with it. Hence returnFromHighs is called at the end of
// each
HighsStatus Highs::passModel(HighsModel model) {
  // This is the "master" Highs::passModel, in that all the others
  // eventually call it
  this->logHeader();
  HighsStatus return_status = HighsStatus::kOk;
  // Clear the incumbent model and any associated data
  clearModel();
  HighsLp& lp = model_.lp_;
  HighsHessian& hessian = model_.hessian_;
  // Move the model's LP and Hessian to the internal LP and Hessian
  lp = std::move(model.lp_);
  hessian = std::move(model.hessian_);
  assert(lp.a_matrix_.formatOk());
  if (lp.num_col_ == 0 || lp.num_row_ == 0) {
    // Model constraint matrix has either no columns or no
    // rows. Clearly the matrix is empty, so may have no orientation
    // or starts assigned. HiGHS assumes that such a model will have
    // null starts, so make it column-wise
    lp.a_matrix_.format_ = MatrixFormat::kColwise;
    lp.a_matrix_.start_.assign(lp.num_col_ + 1, 0);
    lp.a_matrix_.index_.clear();
    lp.a_matrix_.value_.clear();
  } else {
    // Matrix has rows and columns, so a_matrix format must be valid
    if (!lp.a_matrix_.formatOk()) return HighsStatus::kError;
  }
  // Dimensions in a_matrix_ may not be set, so take them from lp.
  lp.setMatrixDimensions();
  // Residual scale factors may be present. ToDo Allow user-defined
  // scale factors to be passed
  assert(!lp.is_scaled_);
  assert(!lp.is_moved_);
  lp.resetScale();
  // Check that the LP array dimensions are valid
  if (!lpDimensionsOk("passModel", lp, options_.log_options))
    return HighsStatus::kError;
  // Check that the Hessian format is valid
  if (!hessian.formatOk()) return HighsStatus::kError;
  // Ensure that the LP is column-wise
  lp.ensureColwise();
  // Check validity of the LP, normalising its values
  return_status = interpretCallStatus(
      options_.log_options, assessLp(lp, options_), return_status, "assessLp");
  if (return_status == HighsStatus::kError) return return_status;
  // Check validity of any Hessian, normalising its entries
  return_status = interpretCallStatus(options_.log_options,
                                      assessHessian(hessian, options_),
                                      return_status, "assessHessian");
  if (return_status == HighsStatus::kError) return return_status;
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
  // Clear solver status, solution, basis and info associated with any
  // previous model; clear any HiGHS model object; create a HiGHS
  // model object for this LP
  return_status = interpretCallStatus(options_.log_options, clearSolver(),
                                      return_status, "clearSolver");
  return returnFromHighs(return_status);
}

HighsStatus Highs::passModel(HighsLp lp) {
  HighsModel model;
  model.lp_ = std::move(lp);
  return passModel(std::move(model));
}

HighsStatus Highs::passModel(
    const HighsInt num_col, const HighsInt num_row, const HighsInt a_num_nz,
    const HighsInt q_num_nz, const HighsInt a_format, const HighsInt q_format,
    const HighsInt sense, const double offset, const double* costs,
    const double* col_lower, const double* col_upper, const double* row_lower,
    const double* row_upper, const HighsInt* a_start, const HighsInt* a_index,
    const double* a_value, const HighsInt* q_start, const HighsInt* q_index,
    const double* q_value, const HighsInt* integrality) {
  this->logHeader();
  HighsModel model;
  HighsLp& lp = model.lp_;
  // Check that the formats of the constraint matrix and Hessian are valid
  if (!aFormatOk(a_num_nz, a_format)) {
    highsLogUser(options_.log_options, HighsLogType::kError,
                 "Model has illegal constraint matrix format\n");
    return HighsStatus::kError;
  }
  if (!qFormatOk(q_num_nz, q_format)) {
    highsLogUser(options_.log_options, HighsLogType::kError,
                 "Model has illegal Hessian matrix format\n");
    return HighsStatus::kError;
  }
  const bool a_rowwise =
      a_num_nz > 0 ? a_format == (HighsInt)MatrixFormat::kRowwise : false;
  //  if (num_nz) a_rowwise = a_format == (HighsInt)MatrixFormat::kRowwise;

  lp.num_col_ = num_col;
  lp.num_row_ = num_row;
  if (num_col > 0) {
    assert(costs != NULL);
    assert(col_lower != NULL);
    assert(col_upper != NULL);
    lp.col_cost_.assign(costs, costs + num_col);
    lp.col_lower_.assign(col_lower, col_lower + num_col);
    lp.col_upper_.assign(col_upper, col_upper + num_col);
  }
  if (num_row > 0) {
    assert(row_lower != NULL);
    assert(row_upper != NULL);
    lp.row_lower_.assign(row_lower, row_lower + num_row);
    lp.row_upper_.assign(row_upper, row_upper + num_row);
  }
  if (a_num_nz > 0) {
    assert(num_col > 0);
    assert(num_row > 0);
    assert(a_start != NULL);
    assert(a_index != NULL);
    assert(a_value != NULL);
    if (a_rowwise) {
      lp.a_matrix_.start_.assign(a_start, a_start + num_row);
    } else {
      lp.a_matrix_.start_.assign(a_start, a_start + num_col);
    }
    lp.a_matrix_.index_.assign(a_index, a_index + a_num_nz);
    lp.a_matrix_.value_.assign(a_value, a_value + a_num_nz);
  }
  if (a_rowwise) {
    lp.a_matrix_.start_.resize(num_row + 1);
    lp.a_matrix_.start_[num_row] = a_num_nz;
    lp.a_matrix_.format_ = MatrixFormat::kRowwise;
  } else {
    lp.a_matrix_.start_.resize(num_col + 1);
    lp.a_matrix_.start_[num_col] = a_num_nz;
    lp.a_matrix_.format_ = MatrixFormat::kColwise;
  }
  if (sense == (HighsInt)ObjSense::kMaximize) {
    lp.sense_ = ObjSense::kMaximize;
  } else {
    lp.sense_ = ObjSense::kMinimize;
  }
  lp.offset_ = offset;
  if (num_col > 0 && integrality != NULL) {
    lp.integrality_.resize(num_col);
    for (HighsInt iCol = 0; iCol < num_col; iCol++) {
      HighsInt integrality_status = integrality[iCol];
      const bool legal_integrality_status =
          integrality_status == (HighsInt)HighsVarType::kContinuous ||
          integrality_status == (HighsInt)HighsVarType::kInteger ||
          integrality_status == (HighsInt)HighsVarType::kSemiContinuous ||
          integrality_status == (HighsInt)HighsVarType::kSemiInteger;
      if (!legal_integrality_status) {
        highsLogDev(
            options_.log_options, HighsLogType::kError,
            "Model has illegal integer value of %d for integrality[%d]\n",
            (int)integrality_status, iCol);
        return HighsStatus::kError;
      }
      lp.integrality_[iCol] = (HighsVarType)integrality_status;
    }
  }
  if (q_num_nz > 0) {
    assert(num_col > 0);
    assert(q_start != NULL);
    assert(q_index != NULL);
    assert(q_value != NULL);
    HighsHessian& hessian = model.hessian_;
    hessian.dim_ = num_col;
    hessian.format_ = HessianFormat::kTriangular;
    hessian.start_.assign(q_start, q_start + num_col);
    hessian.start_.resize(num_col + 1);
    hessian.start_[num_col] = q_num_nz;
    hessian.index_.assign(q_index, q_index + q_num_nz);
    hessian.value_.assign(q_value, q_value + q_num_nz);
  }
  return passModel(std::move(model));
}

HighsStatus Highs::passModel(const HighsInt num_col, const HighsInt num_row,
                             const HighsInt num_nz, const HighsInt a_format,
                             const HighsInt sense, const double offset,
                             const double* costs, const double* col_lower,
                             const double* col_upper, const double* row_lower,
                             const double* row_upper, const HighsInt* a_start,
                             const HighsInt* a_index, const double* a_value,
                             const HighsInt* integrality) {
  return passModel(num_col, num_row, num_nz, 0, a_format, 0, sense, offset,
                   costs, col_lower, col_upper, row_lower, row_upper, a_start,
                   a_index, a_value, NULL, NULL, NULL, integrality);
}

HighsStatus Highs::passHessian(HighsHessian hessian_) {
  this->logHeader();
  HighsStatus return_status = HighsStatus::kOk;
  HighsHessian& hessian = model_.hessian_;
  hessian = std::move(hessian_);
  // Check validity of any Hessian, normalising its entries
  return_status = interpretCallStatus(options_.log_options,
                                      assessHessian(hessian, options_),
                                      return_status, "assessHessian");
  if (return_status == HighsStatus::kError) return return_status;
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
  return_status = interpretCallStatus(options_.log_options, clearSolver(),
                                      return_status, "clearSolver");
  return returnFromHighs(return_status);
}

HighsStatus Highs::passHessian(const HighsInt dim, const HighsInt num_nz,
                               const HighsInt format, const HighsInt* start,
                               const HighsInt* index, const double* value) {
  this->logHeader();
  HighsHessian hessian;
  if (!qFormatOk(num_nz, format)) {
    highsLogUser(options_.log_options, HighsLogType::kError,
                 "Model has illegal Hessian matrix format\n");
    return HighsStatus::kError;
  }
  HighsInt num_col = model_.lp_.num_col_;
  if (dim != num_col) return HighsStatus::kError;
  hessian.dim_ = num_col;
  hessian.format_ = HessianFormat::kTriangular;
  if (dim > 0) {
    assert(start != NULL);
    hessian.start_.assign(start, start + num_col);
    hessian.start_.resize(num_col + 1);
    hessian.start_[num_col] = num_nz;
  }
  if (num_nz > 0) {
    assert(index != NULL);
    assert(value != NULL);
    hessian.index_.assign(index, index + num_nz);
    hessian.value_.assign(value, value + num_nz);
  }
  return passHessian(hessian);
}

HighsStatus Highs::readModel(const std::string& filename) {
  this->logHeader();
  HighsStatus return_status = HighsStatus::kOk;
  Filereader* reader =
      Filereader::getFilereader(options_.log_options, filename);
  if (reader == NULL) {
    highsLogUser(options_.log_options, HighsLogType::kError,
                 "Model file %s not supported\n", filename.c_str());
    return HighsStatus::kError;
  }

  HighsModel model;
  FilereaderRetcode call_code =
      reader->readModelFromFile(options_, filename, model);
  delete reader;
  if (call_code != FilereaderRetcode::kOk) {
    interpretFilereaderRetcode(options_.log_options, filename.c_str(),
                               call_code);
    return_status =
        interpretCallStatus(options_.log_options, HighsStatus::kError,
                            return_status, "readModelFromFile");
    if (return_status == HighsStatus::kError) return return_status;
  }
  model.lp_.model_name_ = extractModelName(filename);
  const bool remove_rows_of_count_1 = false;
  if (remove_rows_of_count_1) {
    // .lp files from PWSC (notably st-test23.lp) have bounds for
    // semi-continuous variables in the constraints section. By default,
    // these are interpreted as constraints, so the semi-continuous
    // variables are not set up correctly. Fix is to remove all rows of
    // count 1, interpreting their bounds as bounds on the corresponding
    // variable.
    removeRowsOfCountOne(options_.log_options, model.lp_);
  }
  return_status =
      interpretCallStatus(options_.log_options, passModel(std::move(model)),
                          return_status, "passModel");
  return returnFromHighs(return_status);
}

HighsStatus Highs::readBasis(const std::string& filename) {
  this->logHeader();
  HighsStatus return_status = HighsStatus::kOk;
  // Try to read basis file into read_basis
  HighsBasis read_basis = basis_;
  return_status = interpretCallStatus(
      options_.log_options,
      readBasisFile(options_.log_options, read_basis, filename), return_status,
      "readBasis");
  if (return_status != HighsStatus::kOk) return return_status;
  // Basis read OK: check whether it's consistent with the LP
  if (!isBasisConsistent(model_.lp_, read_basis)) {
    highsLogUser(options_.log_options, HighsLogType::kError,
                 "readBasis: invalid basis\n");
    return HighsStatus::kError;
  }
  // Update the HiGHS basis and invalidate any simplex basis for the model
  basis_ = read_basis;
  basis_.valid = true;
  // Follow implications of a new HiGHS basis
  newHighsBasis();
  // Can't use returnFromHighs since...
  return HighsStatus::kOk;
}

HighsStatus Highs::writeModel(const std::string& filename) {
  HighsStatus return_status = HighsStatus::kOk;

  // Ensure that the LP is column-wise
  model_.lp_.ensureColwise();
  if (filename == "") {
    // Empty file name: report model on logging stream
    reportModel();
    return_status = HighsStatus::kOk;
  } else {
    Filereader* writer =
        Filereader::getFilereader(options_.log_options, filename);
    if (writer == NULL) {
      highsLogUser(options_.log_options, HighsLogType::kError,
                   "Model file %s not supported\n", filename.c_str());
      return HighsStatus::kError;
    }
    return_status = interpretCallStatus(
        options_.log_options,
        writer->writeModelToFile(options_, filename, model_), return_status,
        "writeModelToFile");
    delete writer;
  }
  return returnFromHighs(return_status);
}

HighsStatus Highs::writeBasis(const std::string& filename) {
  HighsStatus return_status = HighsStatus::kOk;
  HighsStatus call_status;
  FILE* file;
  bool html;
  call_status = openWriteFile(filename, "writebasis", file, html);
  return_status = interpretCallStatus(options_.log_options, call_status,
                                      return_status, "openWriteFile");
  if (return_status == HighsStatus::kError) return return_status;
  writeBasisFile(file, basis_);
  if (file != stdout) fclose(file);
  return returnFromHighs(return_status);
}

HighsStatus Highs::presolve() {
  HighsStatus return_status = HighsStatus::kOk;

  clearPresolve();
  if (model_.isEmpty()) {
    model_presolve_status_ = HighsPresolveStatus::kNotReduced;
  } else {
    const bool force_presolve = true;
    // make sure global scheduler is initialized before calling presolve, since
    // MIP presolve may use parallelism
    highs::parallel::initialize_scheduler(options_.threads);
    max_threads = highs::parallel::num_threads();
    if (options_.threads != 0 && max_threads != options_.threads) {
      highsLogUser(
          options_.log_options, HighsLogType::kError,
          "Option 'threads' is set to %d but global scheduler has already been "
          "initialized to use %d threads. The previous scheduler instance can "
          "be destroyed by calling Highs::resetGlobalScheduler().\n",
          (int)options_.threads, max_threads);
      return HighsStatus::kError;
    }
    model_presolve_status_ = runPresolve(force_presolve);
  }

  bool using_reduced_lp = false;
  switch (model_presolve_status_) {
    case HighsPresolveStatus::kNotPresolved: {
      // Shouldn't happen
      assert(model_presolve_status_ != HighsPresolveStatus::kNotPresolved);
      return_status = HighsStatus::kError;
      break;
    }
    case HighsPresolveStatus::kNotReduced:
    case HighsPresolveStatus::kInfeasible:
    case HighsPresolveStatus::kReduced:
    case HighsPresolveStatus::kReducedToEmpty:
    case HighsPresolveStatus::kUnboundedOrInfeasible: {
      // All OK
      if (model_presolve_status_ == HighsPresolveStatus::kInfeasible) {
        // Infeasible model, so indicate that the incumbent model is
        // known as such
        setHighsModelStatusAndClearSolutionAndBasis(
            HighsModelStatus::kInfeasible);
      } else if (model_presolve_status_ == HighsPresolveStatus::kNotReduced) {
        // No reduction, so fill Highs presolved model with the
        // incumbent model
        presolved_model_ = model_;
      } else if (model_presolve_status_ == HighsPresolveStatus::kReduced) {
        // Nontrivial reduction, so fill Highs presolved model with the
        // presolved model
        using_reduced_lp = true;
      }
      return_status = HighsStatus::kOk;
      break;
    }
    case HighsPresolveStatus::kTimeout: {
      // Timeout, so assume that it's OK to fill the Highs presolved model with
      // the presolved model, but return warning.
      using_reduced_lp = true;
      return_status = HighsStatus::kWarning;
      break;
    }
    default: {
      // case HighsPresolveStatus::kError
      setHighsModelStatusAndClearSolutionAndBasis(
          HighsModelStatus::kPresolveError);
      return_status = HighsStatus::kError;
    }
  }
  if (using_reduced_lp) {
    presolved_model_.lp_ = presolve_.getReducedProblem();
    presolved_model_.lp_.setMatrixDimensions();
  }

  highsLogUser(
      options_.log_options, HighsLogType::kInfo, "Presolve status: %s\n",
      presolve_.presolveStatusToString(model_presolve_status_).c_str());
  return returnFromHighs(return_status);
}

// Checks the options calls presolve and postsolve if needed. Solvers are called
// with callSolveLp(..)
HighsStatus Highs::run() {
  HighsInt min_highs_debug_level = kHighsDebugLevelMin;
  // kHighsDebugLevelCostly;
  // kHighsDebugLevelMax;
  //
  //  if (model_.lp_.num_row_>0 && model_.lp_.num_col_>0)
  //  writeLpMatrixPicToFile(options_, "LpMatrix", model_.lp_);
  if (options_.highs_debug_level < min_highs_debug_level)
    options_.highs_debug_level = min_highs_debug_level;

  const bool possibly_use_log_dev_level_2 = false;
  const HighsInt log_dev_level = options_.log_dev_level;
  const bool output_flag = options_.output_flag;
  HighsInt use_log_dev_level = log_dev_level;
  bool use_output_flag = output_flag;
  const HighsInt check_debug_run_call_num = -103757;
  const HighsInt check_num_col = -317;
  const HighsInt check_num_row = -714;
  if (possibly_use_log_dev_level_2) {
    if (this->debug_run_call_num_ == check_debug_run_call_num &&
        model_.lp_.num_col_ == check_num_col &&
        model_.lp_.num_row_ == check_num_row) {
      std::string message =
          "Entering Highs::run(): run/col/row matching check ";
      highsLogDev(options_.log_options, HighsLogType::kInfo,
                  "%s: run %d: LP(%6d, %6d)\n", message.c_str(),
                  (int)this->debug_run_call_num_, (int)model_.lp_.num_col_,
                  (int)model_.lp_.num_row_);
      // highsPause(true, message);
      use_log_dev_level = 2;
      use_output_flag = true;
    }
  }
  if (ekk_instance_.status_.has_nla)
    assert(ekk_instance_.lpFactorRowCompatible(model_.lp_.num_row_));

  highs::parallel::initialize_scheduler(options_.threads);

  max_threads = highs::parallel::num_threads();
  if (options_.threads != 0 && max_threads != options_.threads) {
    highsLogUser(
        options_.log_options, HighsLogType::kError,
        "Option 'threads' is set to %d but global scheduler has already been "
        "initialized to use %d threads. The previous scheduler instance can "
        "be destroyed by calling Highs::resetGlobalScheduler().\n",
        (int)options_.threads, max_threads);
    return HighsStatus::kError;
  }
  assert(max_threads > 0);
  if (max_threads <= 0)
    highsLogDev(options_.log_options, HighsLogType::kWarning,
                "WARNING: max_threads() returns %" HIGHSINT_FORMAT "\n",
                max_threads);
  highsLogDev(options_.log_options, HighsLogType::kDetailed,
              "Running with %" HIGHSINT_FORMAT " thread(s)\n", max_threads);
  assert(called_return_from_run);
  if (!called_return_from_run) {
    highsLogDev(options_.log_options, HighsLogType::kError,
                "Highs::run() called with called_return_from_run false\n");
    return HighsStatus::kError;
  }
  // Ensure that all vectors in the model have exactly the right size
  exactResizeModel();
  // Set this so that calls to returnFromRun() can be checked
  called_return_from_run = false;
  // From here all return statements execute returnFromRun()
  HighsStatus return_status = HighsStatus::kOk;
  HighsStatus call_status;
  // Initialise the HiGHS model status
  model_status_ = HighsModelStatus::kNotset;
  // Clear the run info
  invalidateInfo();
  // Zero the iteration counts
  zeroIterationCounts();
  // Start the HiGHS run clock
  timer_.startRunHighsClock();
  // Return immediately if the model has no columns
  if (!model_.lp_.num_col_) {
    setHighsModelStatusAndClearSolutionAndBasis(HighsModelStatus::kModelEmpty);
    return returnFromRun(HighsStatus::kOk);
  }
  // Return immediately if the model is infeasible due to inconsistent bounds
  if (isBoundInfeasible(options_.log_options, model_.lp_)) {
    setHighsModelStatusAndClearSolutionAndBasis(HighsModelStatus::kInfeasible);
    return returnFromRun(return_status);
  }
  // Ensure that the LP (and any simplex LP) has the matrix column-wise
  model_.lp_.ensureColwise();
  // Ensure that the matrix has no large values
  if (model_.lp_.a_matrix_.hasLargeValue(options_.large_matrix_value)) {
    highsLogUser(options_.log_options, HighsLogType::kError,
                 "Cannot solve a model with a |value| exceeding %g in "
                 "constraint matrix\n",
                 options_.large_matrix_value);
    return returnFromRun(HighsStatus::kError);
  }
  if (options_.highs_debug_level > min_highs_debug_level) {
    // Shouldn't have to check validity of the LP since this is done when it is
    // loaded or modified
    call_status = assessLp(model_.lp_, options_);
    // If any errors have been found or normalisation carried out,
    // call_status will be kError or kWarning, so only valid return is OK.
    assert(call_status == HighsStatus::kOk);
    return_status = interpretCallStatus(options_.log_options, call_status,
                                        return_status, "assessLp");
    if (return_status == HighsStatus::kError)
      return returnFromRun(return_status);
    // Shouldn't have to check that the options settings are legal,
    // since they are checked when modified
    if (checkOptions(options_.log_options, options_.records) !=
        OptionStatus::kOk) {
      return_status = HighsStatus::kError;
      return returnFromRun(return_status);
    }
  }

  if (model_.lp_.model_name_.compare(""))
    highsLogDev(options_.log_options, HighsLogType::kVerbose,
                "Solving model: %s\n", model_.lp_.model_name_.c_str());

  // Check validity of any integrality, keeping a record of any upper
  // bound modifications for semi-variables
  call_status = assessIntegrality(model_.lp_, options_);
  if (call_status == HighsStatus::kError) {
    setHighsModelStatusAndClearSolutionAndBasis(HighsModelStatus::kSolveError);
    return returnFromRun(HighsStatus::kError);
  }

  if (!options_.solver.compare(kHighsChooseString) && model_.isQp()) {
    // Choosing method according to model class, and model is a QP
    //
    // Ensure that it's not MIQP!
    if (model_.isMip()) {
      highsLogUser(options_.log_options, HighsLogType::kError,
                   "Cannot solve MIQP problems with HiGHS\n");
      return returnFromRun(HighsStatus::kError);
    }
    // Ensure that its diagonal entries are OK in the context of the
    // objective sense. It's OK to be semi-definite
    if (!okHessianDiagonal(options_, model_.hessian_, model_.lp_.sense_)) {
      highsLogUser(options_.log_options, HighsLogType::kError,
                   "Cannot solve non-convex QP problems with HiGHS\n");
      return returnFromRun(HighsStatus::kError);
    }
    call_status = callSolveQp();
    return_status = interpretCallStatus(options_.log_options, call_status,
                                        return_status, "callSolveQp");
    return returnFromRun(return_status);
  }

  if (!options_.solver.compare(kHighsChooseString) && model_.isMip()) {
    // Choosing method according to model class, and model is a MIP
    call_status = callSolveMip();
    return_status = interpretCallStatus(options_.log_options, call_status,
                                        return_status, "callSolveMip");
    return returnFromRun(return_status);
  }
  // Solve the model as an LP
  HighsLp& incumbent_lp = model_.lp_;
  HighsLogOptions& log_options = options_.log_options;
  bool no_incumbent_lp_solution_or_basis = false;
  //
  // Record the initial time and set the component times and postsolve
  // iteration count to -1 to identify whether they are not required
  double initial_time = timer_.readRunHighsClock();
  double this_presolve_time = -1;
  double this_solve_presolved_lp_time = -1;
  double this_postsolve_time = -1;
  double this_solve_original_lp_time = -1;
  HighsInt postsolve_iteration_count = -1;
  const bool ipx_no_crossover =
      options_.solver == kIpmString && !options_.run_crossover;

  if (options_.icrash) {
    ICrashStrategy strategy = ICrashStrategy::kICA;
    bool strategy_ok = parseICrashStrategy(options_.icrash_strategy, strategy);
    if (!strategy_ok) {
      // std::cout << "ICrash error: unknown strategy." << std::endl;
      highsLogUser(options_.log_options, HighsLogType::kError,
                   "ICrash error: unknown strategy.\n");
      return HighsStatus::kError;
    }
    ICrashOptions icrash_options{
        options_.icrash_dualize,         strategy,
        options_.icrash_starting_weight, options_.icrash_iterations,
        options_.icrash_approx_iter,     options_.icrash_exact,
        options_.icrash_breakpoints,     options_.log_options};

    HighsStatus icrash_status =
        callICrash(model_.lp_, icrash_options, icrash_info_);

    if (icrash_status != HighsStatus::kOk) return returnFromRun(icrash_status);

    // for now set the solution_.col_value
    solution_.col_value = icrash_info_.x_values;
    // Better not to use Highs::crossover
    const bool use_highs_crossover = false;
    if (use_highs_crossover) {
      crossover(solution_);
      // loops:
      called_return_from_run = true;

      options_.icrash = false;  // to avoid loop
    } else {
      HighsStatus crossover_status = callCrossover(
          options_, model_.lp_, basis_, solution_, model_status_, info_);
      // callCrossover can return HighsStatus::kWarning due to
      // imprecise dual values. Ignore this since primal simplex will
      // be called to clean up duals
      highsLogUser(log_options, HighsLogType::kInfo,
                   "Crossover following iCrash has return status of %s, and "
                   "problem status is %s\n",
                   highsStatusToString(crossover_status).c_str(),
                   modelStatusToString(model_status_).c_str());
      if (crossover_status == HighsStatus::kError)
        return returnFromRun(crossover_status);
      assert(options_.simplex_strategy == kSimplexStrategyPrimal);
    }
    // timer_.stopRunHighsClock();
    // run();

    // todo: add "dual" values
    // return HighsStatus::kOk;
  }

  if (!basis_.valid && solution_.value_valid) {
    // There is no valid basis, but there is a valid solution, so use
    // it to construct a basis
    return_status =
        interpretCallStatus(options_.log_options, basisForSolution(),
                            return_status, "basisForSolution");
    if (return_status == HighsStatus::kError)
      return returnFromRun(return_status);
    assert(basis_.valid);
  }

  if (basis_.valid || options_.presolve == kHighsOffString) {
    // There is a valid basis for the problem or presolve is off
    ekk_instance_.lp_name_ = "LP without presolve or with basis";
    // If there is a valid HiGHS basis, refine any status values that
    // are simply HighsBasisStatus::kNonbasic
    if (basis_.valid) refineBasis(incumbent_lp, solution_, basis_);
    this_solve_original_lp_time = -timer_.read(timer_.solve_clock);
    if (possibly_use_log_dev_level_2) {
      options_.log_dev_level = use_log_dev_level;
      options_.output_flag = use_output_flag;
    }
    timer_.start(timer_.solve_clock);
    call_status =
        callSolveLp(incumbent_lp, "Solving LP without presolve or with basis");
    timer_.stop(timer_.solve_clock);
    if (possibly_use_log_dev_level_2) {
      options_.log_dev_level = log_dev_level;
      options_.output_flag = output_flag;
    }
    this_solve_original_lp_time += timer_.read(timer_.solve_clock);
    return_status = interpretCallStatus(options_.log_options, call_status,
                                        return_status, "callSolveLp");
    if (return_status == HighsStatus::kError)
      return returnFromRun(return_status);
  } else {
    // No HiGHS basis so consider presolve
    //
    // If using IPX to solve the reduced LP, but not crossover, set
    // lp_presolve_requires_basis_postsolve so that presolve can use
    // rules for which postsolve does not generate a basis.
    const bool lp_presolve_requires_basis_postsolve =
        options_.lp_presolve_requires_basis_postsolve;
    if (ipx_no_crossover) options_.lp_presolve_requires_basis_postsolve = false;
    // Possibly presolve - according to option_.presolve
    const double from_presolve_time = timer_.read(timer_.presolve_clock);
    this_presolve_time = -from_presolve_time;
    timer_.start(timer_.presolve_clock);
    model_presolve_status_ = runPresolve();
    timer_.stop(timer_.presolve_clock);
    const double to_presolve_time = timer_.read(timer_.presolve_clock);
    this_presolve_time += to_presolve_time;
    presolve_.info_.presolve_time = this_presolve_time;
    // Recover any modified options
    options_.lp_presolve_requires_basis_postsolve =
        lp_presolve_requires_basis_postsolve;

    // Set an illegal local pivot threshold value that's updated after
    // solving the presolved LP - if simplex is used
    double factor_pivot_threshold = -1;

    // Run solver.
    bool have_optimal_solution = false;
    // ToDo Put solution of presolved problem in a separate method
    switch (model_presolve_status_) {
      case HighsPresolveStatus::kNotPresolved: {
        ekk_instance_.lp_name_ = "Original LP";
        this_solve_original_lp_time = -timer_.read(timer_.solve_clock);
        if (possibly_use_log_dev_level_2) {
          options_.log_dev_level = use_log_dev_level;
          options_.output_flag = use_output_flag;
        }
        timer_.start(timer_.solve_clock);
        call_status =
            callSolveLp(incumbent_lp, "Not presolved: solving the LP");
        timer_.stop(timer_.solve_clock);
        if (possibly_use_log_dev_level_2) {
          options_.log_dev_level = log_dev_level;
          options_.output_flag = output_flag;
        }
        this_solve_original_lp_time += timer_.read(timer_.solve_clock);
        return_status = interpretCallStatus(options_.log_options, call_status,
                                            return_status, "callSolveLp");
        if (return_status == HighsStatus::kError)
          return returnFromRun(return_status);
        break;
      }
      case HighsPresolveStatus::kNotReduced: {
        ekk_instance_.lp_name_ = "Unreduced LP";
        // Log the presolve reductions
        reportPresolveReductions(log_options, incumbent_lp, false);
        this_solve_original_lp_time = -timer_.read(timer_.solve_clock);
        if (possibly_use_log_dev_level_2) {
          options_.log_dev_level = use_log_dev_level;
          options_.output_flag = use_output_flag;
        }
        timer_.start(timer_.solve_clock);
        call_status = callSolveLp(
            incumbent_lp, "Problem not reduced by presolve: solving the LP");
        timer_.stop(timer_.solve_clock);
        if (possibly_use_log_dev_level_2) {
          options_.log_dev_level = log_dev_level;
          options_.output_flag = output_flag;
        }
        this_solve_original_lp_time += timer_.read(timer_.solve_clock);
        return_status = interpretCallStatus(options_.log_options, call_status,
                                            return_status, "callSolveLp");
        if (return_status == HighsStatus::kError)
          return returnFromRun(return_status);
        break;
      }
      case HighsPresolveStatus::kReduced: {
        HighsLp& reduced_lp = presolve_.getReducedProblem();
        reduced_lp.setMatrixDimensions();
        // Validate the reduced LP
        assert(assessLp(reduced_lp, options_) == HighsStatus::kOk);
        call_status = cleanBounds(options_, reduced_lp);
        // Ignore any warning from clean bounds since the original LP
        // is still solved after presolve
        if (interpretCallStatus(options_.log_options, call_status,
                                return_status,
                                "cleanBounds") == HighsStatus::kError)
          return HighsStatus::kError;
        // Log the presolve reductions
        reportPresolveReductions(log_options, incumbent_lp, reduced_lp);
        // Solving the presolved LP with strictly reduced dimensions
        // so ensure that the Ekk instance is cleared
        ekk_instance_.clear();
        ekk_instance_.lp_name_ = "Presolved LP";
        // Don't try dual cut-off when solving the presolved LP, as the
        // objective values aren't correct
        const double save_objective_bound = options_.objective_bound;
        options_.objective_bound = kHighsInf;
        this_solve_presolved_lp_time = -timer_.read(timer_.solve_clock);
        if (possibly_use_log_dev_level_2) {
          options_.log_dev_level = use_log_dev_level;
          options_.output_flag = use_output_flag;
        }
        timer_.start(timer_.solve_clock);
        call_status = callSolveLp(reduced_lp, "Solving the presolved LP");
        timer_.stop(timer_.solve_clock);
        if (possibly_use_log_dev_level_2) {
          options_.log_dev_level = log_dev_level;
          options_.output_flag = output_flag;
        }
        this_solve_presolved_lp_time += timer_.read(timer_.solve_clock);
        if (ekk_instance_.status_.initialised_for_solve) {
          // Record the pivot threshold resulting from solving the presolved LP
          // with simplex
          factor_pivot_threshold = ekk_instance_.info_.factor_pivot_threshold;
        }
        // Restore the dual objective cut-off
        options_.objective_bound = save_objective_bound;
        return_status = interpretCallStatus(options_.log_options, call_status,
                                            return_status, "callSolveLp");
        if (return_status == HighsStatus::kError)
          return returnFromRun(return_status);
        have_optimal_solution = model_status_ == HighsModelStatus::kOptimal;
        no_incumbent_lp_solution_or_basis =
            model_status_ == HighsModelStatus::kInfeasible ||
            model_status_ == HighsModelStatus::kUnbounded ||
            model_status_ == HighsModelStatus::kUnboundedOrInfeasible ||
            model_status_ == HighsModelStatus::kTimeLimit ||
            model_status_ == HighsModelStatus::kIterationLimit;
        break;
      }
      case HighsPresolveStatus::kReducedToEmpty: {
        reportPresolveReductions(log_options, incumbent_lp, true);
        // Create a trivial optimal solution for postsolve to use
        solution_.clear();
        basis_.clear();
        basis_.debug_origin_name = "Presolve to empty";
        basis_.valid = true;
        basis_.alien = false;
        basis_.was_alien = false;
        solution_.value_valid = true;
        solution_.dual_valid = true;
        have_optimal_solution = true;
        break;
      }
      case HighsPresolveStatus::kInfeasible: {
        setHighsModelStatusAndClearSolutionAndBasis(
            HighsModelStatus::kInfeasible);
        highsLogUser(log_options, HighsLogType::kInfo,
                     "Problem status detected on presolve: %s\n",
                     modelStatusToString(model_status_).c_str());
        return returnFromRun(return_status);
      }
      case HighsPresolveStatus::kUnboundedOrInfeasible: {
        if (options_.allow_unbounded_or_infeasible) {
          setHighsModelStatusAndClearSolutionAndBasis(
              HighsModelStatus::kUnboundedOrInfeasible);
          highsLogUser(log_options, HighsLogType::kInfo,
                       "Problem status detected on presolve: %s\n",
                       modelStatusToString(model_status_).c_str());
          return returnFromRun(return_status);
        }
        // Presolve has returned kUnboundedOrInfeasible, but HiGHS
        // can't reurn this. Use primal simplex solver on the original
        // LP
        HighsOptions save_options = options_;
        options_.solver = "simplex";
        options_.simplex_strategy = kSimplexStrategyPrimal;
        this_solve_original_lp_time = -timer_.read(timer_.solve_clock);
        if (possibly_use_log_dev_level_2) {
          options_.log_dev_level = use_log_dev_level;
          options_.output_flag = use_output_flag;
        }
        timer_.start(timer_.solve_clock);
        call_status = callSolveLp(incumbent_lp,
                                  "Solving the original LP with primal simplex "
                                  "to determine infeasible or unbounded");
        timer_.stop(timer_.solve_clock);
        if (possibly_use_log_dev_level_2) {
          options_.log_dev_level = log_dev_level;
          options_.output_flag = output_flag;
        }
        this_solve_original_lp_time += timer_.read(timer_.solve_clock);
        // Recover the options
        options_ = save_options;
        if (return_status == HighsStatus::kError)
          return returnFromRun(return_status);
        // ToDo Eliminate setBasisValidity once ctest passes. Asserts
        // verify that it does nothing - other than setting
        // info_.valid = true;
        setBasisValidity();
        assert(model_status_ == HighsModelStatus::kInfeasible ||
               model_status_ == HighsModelStatus::kUnbounded);
        return returnFromRun(return_status);
      }
      case HighsPresolveStatus::kTimeout: {
        setHighsModelStatusAndClearSolutionAndBasis(
            HighsModelStatus::kTimeLimit);
        highsLogDev(log_options, HighsLogType::kError,
                    "Presolve reached timeout\n");
        return returnFromRun(HighsStatus::kWarning);
      }
      case HighsPresolveStatus::kOptionsError: {
        setHighsModelStatusAndClearSolutionAndBasis(
            HighsModelStatus::kPresolveError);
        highsLogDev(log_options, HighsLogType::kError,
                    "Presolve options error\n");
        return returnFromRun(HighsStatus::kError);
      }
      default: {
        assert(model_presolve_status_ == HighsPresolveStatus::kNullError);
        setHighsModelStatusAndClearSolutionAndBasis(
            HighsModelStatus::kPresolveError);
        highsLogDev(log_options, HighsLogType::kError,
                    "Presolve returned status %d\n",
                    (int)model_presolve_status_);
        return returnFromRun(HighsStatus::kError);
      }
    }
    // End of presolve
    assert(model_presolve_status_ == HighsPresolveStatus::kNotPresolved ||
           model_presolve_status_ == HighsPresolveStatus::kNotReduced ||
           model_presolve_status_ == HighsPresolveStatus::kReduced ||
           model_presolve_status_ == HighsPresolveStatus::kReducedToEmpty);

    // Postsolve. Does nothing if there were no reductions during presolve.

    if (have_optimal_solution) {
      // ToDo Put this in a separate method
      assert(model_status_ == HighsModelStatus::kOptimal ||
             model_presolve_status_ == HighsPresolveStatus::kReducedToEmpty);
      if (model_presolve_status_ == HighsPresolveStatus::kReduced ||
          model_presolve_status_ == HighsPresolveStatus::kReducedToEmpty) {
        // If presolve is nontrivial, extract the optimal solution
        // and basis for the presolved problem in order to generate
        // the solution and basis for postsolve to use to generate a
        // solution(?) and basis that is, hopefully, optimal. This is
        // confirmed or corrected by hot-starting the simplex solver
        presolve_.data_.recovered_solution_ = solution_;
        presolve_.data_.recovered_basis_ = basis_;

        this_postsolve_time = -timer_.read(timer_.postsolve_clock);
        timer_.start(timer_.postsolve_clock);
        HighsPostsolveStatus postsolve_status = runPostsolve();
        timer_.stop(timer_.postsolve_clock);
        this_postsolve_time += -timer_.read(timer_.postsolve_clock);
        presolve_.info_.postsolve_time = this_postsolve_time;

        if (postsolve_status == HighsPostsolveStatus::kSolutionRecovered) {
          highsLogDev(log_options, HighsLogType::kVerbose,
                      "Postsolve finished\n");
          // Set solution and its status
          solution_.clear();
          solution_ = presolve_.data_.recovered_solution_;
          solution_.value_valid = true;
          if (ipx_no_crossover) {
            // IPX was used without crossover, so have a dual solution, but no
            // basis
            solution_.dual_valid = true;
            basis_.invalidate();
          } else {
            //
            // Hot-start the simplex solver for the incumbent LP
            //
            solution_.dual_valid = true;
            // Set basis and its status
            basis_.valid = true;
            basis_.col_status = presolve_.data_.recovered_basis_.col_status;
            basis_.row_status = presolve_.data_.recovered_basis_.row_status;
            basis_.debug_origin_name += ": after postsolve";
            // Basic primal activities are wrong after postsolve, so
            // possibly skip KKT check
            const bool perform_kkt_check = true;
            if (perform_kkt_check) {
              // Possibly force debug to perform KKT check on what's
              // returned from postsolve
              const bool force_debug = false;
              HighsInt save_highs_debug_level = options_.highs_debug_level;
              if (force_debug)
                options_.highs_debug_level = kHighsDebugLevelCostly;
              if (debugHighsSolution("After returning from postsolve", options_,
                                     model_, solution_,
                                     basis_) == HighsDebugStatus::kLogicalError)
                return returnFromRun(HighsStatus::kError);
              options_.highs_debug_level = save_highs_debug_level;
            }
            // Save the options to allow the best simplex strategy to
            // be used
            HighsOptions save_options = options_;
            const bool full_logging = false;
            if (full_logging) options_.log_dev_level = kHighsLogDevLevelVerbose;
            // Force the use of simplex to clean up if IPM has been used
            // to solve the presolved problem
            if (options_.solver == kIpmString) options_.solver = kSimplexString;
            options_.simplex_strategy = kSimplexStrategyChoose;
            // Ensure that the parallel solver isn't used
            options_.simplex_min_concurrency = 1;
            options_.simplex_max_concurrency = 1;
            // Use any pivot threshold resulting from solving the presolved LP
            if (factor_pivot_threshold > 0)
              options_.factor_pivot_threshold = factor_pivot_threshold;
            // The basis returned from postsolve is just basic/nonbasic
            // and EKK expects a refined basis, so set it up now
            refineBasis(incumbent_lp, solution_, basis_);
            // Scrap the EKK data from solving the presolved LP
            ekk_instance_.invalidate();
            ekk_instance_.lp_name_ = "Postsolve LP";
            // Set up the iteration count and timing records so that
            // adding the corresponding values after callSolveLp gives
            // difference
            postsolve_iteration_count = -info_.simplex_iteration_count;
            this_solve_original_lp_time = -timer_.read(timer_.solve_clock);
            if (possibly_use_log_dev_level_2) {
              options_.log_dev_level = use_log_dev_level;
              options_.output_flag = use_output_flag;
            }
            timer_.start(timer_.solve_clock);
            call_status = callSolveLp(
                incumbent_lp,
                "Solving the original LP from the solution after postsolve");
            timer_.stop(timer_.solve_clock);
            if (possibly_use_log_dev_level_2) {
              options_.log_dev_level = log_dev_level;
              options_.output_flag = output_flag;
            }
            // Determine the iteration count and timing records
            postsolve_iteration_count += info_.simplex_iteration_count;
            this_solve_original_lp_time += timer_.read(timer_.solve_clock);
            return_status =
                interpretCallStatus(options_.log_options, call_status,
                                    return_status, "callSolveLp");
            // Recover the options
            options_ = save_options;
            if (return_status == HighsStatus::kError)
              return returnFromRun(return_status);
          }
        } else {
          highsLogUser(log_options, HighsLogType::kError,
                       "Postsolve return status is %d\n",
                       (int)postsolve_status);
          setHighsModelStatusAndClearSolutionAndBasis(
              HighsModelStatus::kPostsolveError);
          return returnFromRun(HighsStatus::kError);
        }
      } else {
        // LP was not reduced by presolve, so have simply solved the original LP
        assert(model_presolve_status_ == HighsPresolveStatus::kNotReduced);
      }
    }
  }
  // Cycling can yield model_status_ == HighsModelStatus::kNotset,
  //  assert(model_status_ != HighsModelStatus::kNotset);
  if (no_incumbent_lp_solution_or_basis) {
    // In solving the (strictly reduced) presolved LP, it is found to
    // be infeasible or unbounded, the time/iteration limit has been
    // reached, or the status is unknown (cycling)
    assert(model_status_ == HighsModelStatus::kInfeasible ||
           model_status_ == HighsModelStatus::kUnbounded ||
           model_status_ == HighsModelStatus::kUnboundedOrInfeasible ||
           model_status_ == HighsModelStatus::kTimeLimit ||
           model_status_ == HighsModelStatus::kIterationLimit ||
           model_status_ == HighsModelStatus::kUnknown);
    // The HEkk data correspond to the (strictly reduced) presolved LP
    // so must be cleared
    ekk_instance_.clear();
    setHighsModelStatusAndClearSolutionAndBasis(model_status_);
  } else {
    // ToDo Eliminate setBasisValidity once ctest passes. Asserts
    // verify that it does nothing - other than setting info_.valid =
    // true;
    setBasisValidity();
  }
  double lp_solve_final_time = timer_.readRunHighsClock();
  double this_solve_time = lp_solve_final_time - initial_time;
  if (postsolve_iteration_count < 0) {
    highsLogDev(log_options, HighsLogType::kInfo, "Postsolve  : \n");
  } else {
    highsLogDev(log_options, HighsLogType::kInfo,
                "Postsolve  : %" HIGHSINT_FORMAT "\n",
                postsolve_iteration_count);
  }
  highsLogDev(log_options, HighsLogType::kInfo, "Time       : %8.2f\n",
              this_solve_time);
  highsLogDev(log_options, HighsLogType::kInfo, "Time Pre   : %8.2f\n",
              this_presolve_time);
  highsLogDev(log_options, HighsLogType::kInfo, "Time PreLP : %8.2f\n",
              this_solve_presolved_lp_time);
  highsLogDev(log_options, HighsLogType::kInfo, "Time PostLP: %8.2f\n",
              this_solve_original_lp_time);
  if (this_solve_time > 0) {
    highsLogDev(log_options, HighsLogType::kInfo, "For LP %16s",
                incumbent_lp.model_name_.c_str());
    double sum_time = 0;
    if (this_presolve_time > 0) {
      sum_time += this_presolve_time;
      HighsInt pct = (100 * this_presolve_time) / this_solve_time;
      highsLogDev(log_options, HighsLogType::kInfo,
                  ": Presolve %8.2f (%3" HIGHSINT_FORMAT "%%)",
                  this_presolve_time, pct);
    }
    if (this_solve_presolved_lp_time > 0) {
      sum_time += this_solve_presolved_lp_time;
      HighsInt pct = (100 * this_solve_presolved_lp_time) / this_solve_time;
      highsLogDev(log_options, HighsLogType::kInfo,
                  ": Solve presolved LP %8.2f (%3" HIGHSINT_FORMAT "%%)",
                  this_solve_presolved_lp_time, pct);
    }
    if (this_postsolve_time > 0) {
      sum_time += this_postsolve_time;
      HighsInt pct = (100 * this_postsolve_time) / this_solve_time;
      highsLogDev(log_options, HighsLogType::kInfo,
                  ": Postsolve %8.2f (%3" HIGHSINT_FORMAT "%%)",
                  this_postsolve_time, pct);
    }
    if (this_solve_original_lp_time > 0) {
      sum_time += this_solve_original_lp_time;
      HighsInt pct = (100 * this_solve_original_lp_time) / this_solve_time;
      highsLogDev(log_options, HighsLogType::kInfo,
                  ": Solve original LP %8.2f (%3" HIGHSINT_FORMAT "%%)",
                  this_solve_original_lp_time, pct);
    }
    highsLogDev(log_options, HighsLogType::kInfo, "\n");
    double rlv_time_difference =
        fabs(sum_time - this_solve_time) / this_solve_time;
    if (rlv_time_difference > 0.1)
      highsLogDev(options_.log_options, HighsLogType::kInfo,
                  "Strange: Solve time = %g; Sum times = %g: relative "
                  "difference = %g\n",
                  this_solve_time, sum_time, rlv_time_difference);
  }
  // Assess success according to the scaled model status, unless
  // something worse has happened earlier
  call_status = highsStatusFromHighsModelStatus(model_status_);
  return_status =
      interpretCallStatus(options_.log_options, call_status, return_status);
  return returnFromRun(return_status);
}

HighsStatus Highs::getDualRay(bool& has_dual_ray, double* dual_ray_value) {
  if (!ekk_instance_.status_.has_invert)
    return invertRequirementError("getDualRay");
  return getDualRayInterface(has_dual_ray, dual_ray_value);
}

HighsStatus Highs::getDualRaySparse(bool& has_dual_ray,
                                    HVector& row_ep_buffer) {
  has_dual_ray = ekk_instance_.status_.has_dual_ray;
  if (has_dual_ray) {
    ekk_instance_.setNlaPointersForLpAndScale(model_.lp_);
    row_ep_buffer.clear();
    row_ep_buffer.count = 1;
    row_ep_buffer.packFlag = true;
    HighsInt iRow = ekk_instance_.info_.dual_ray_row_;
    row_ep_buffer.index[0] = iRow;
    row_ep_buffer.array[iRow] = ekk_instance_.info_.dual_ray_sign_;

    ekk_instance_.btran(row_ep_buffer, ekk_instance_.info_.row_ep_density);
  }

  return HighsStatus::kOk;
}

HighsStatus Highs::getPrimalRay(bool& has_primal_ray,
                                double* primal_ray_value) {
  if (!ekk_instance_.status_.has_invert)
    return invertRequirementError("getPrimalRay");
  return getPrimalRayInterface(has_primal_ray, primal_ray_value);
}

HighsStatus Highs::getRanging() {
  // Create a HighsLpSolverObject of references to data in the Highs
  // class, and the scaled/unscaled model status
  HighsLpSolverObject solver_object(model_.lp_, basis_, solution_, info_,
                                    ekk_instance_, options_, timer_);
  solver_object.model_status_ = model_status_;
  return getRangingData(this->ranging_, solver_object);
}

HighsStatus Highs::getRanging(HighsRanging& ranging) {
  HighsStatus return_status = getRanging();
  ranging = this->ranging_;
  return return_status;
}

bool Highs::hasInvert() const { return ekk_instance_.status_.has_invert; }

const HighsInt* Highs::getBasicVariablesArray() const {
  assert(ekk_instance_.status_.has_invert);
  return ekk_instance_.basis_.basicIndex_.data();
}

HighsStatus Highs::getBasicVariables(HighsInt* basic_variables) {
  if (basic_variables == NULL) {
    highsLogUser(options_.log_options, HighsLogType::kError,
                 "getBasicVariables: basic_variables is NULL\n");
    return HighsStatus::kError;
  }
  return getBasicVariablesInterface(basic_variables);
}

HighsStatus Highs::getBasisInverseRowSparse(const HighsInt row,
                                            HVector& row_ep_buffer) {
  ekk_instance_.setNlaPointersForLpAndScale(model_.lp_);
  row_ep_buffer.clear();
  row_ep_buffer.count = 1;
  row_ep_buffer.index[0] = row;
  row_ep_buffer.array[row] = 1;
  row_ep_buffer.packFlag = true;

  ekk_instance_.btran(row_ep_buffer, ekk_instance_.info_.row_ep_density);

  return HighsStatus::kOk;
}

HighsStatus Highs::getBasisInverseRow(const HighsInt row, double* row_vector,
                                      HighsInt* row_num_nz,
                                      HighsInt* row_indices) {
  if (row_vector == NULL) {
    highsLogUser(options_.log_options, HighsLogType::kError,
                 "getBasisInverseRow: row_vector is NULL\n");
    return HighsStatus::kError;
  }
  // row_indices can be NULL - it's the trigger that determines
  // whether they are identified or not
  HighsInt num_row = model_.lp_.num_row_;
  if (row < 0 || row >= num_row) {
    highsLogUser(options_.log_options, HighsLogType::kError,
                 "Row index %" HIGHSINT_FORMAT
                 " out of range [0, %" HIGHSINT_FORMAT
                 "] in getBasisInverseRow\n",
                 row, num_row - 1);
    return HighsStatus::kError;
  }
  if (!ekk_instance_.status_.has_invert)
    return invertRequirementError("getBasisInverseRow");
  // Compute a row i of the inverse of the basis matrix by solving B^Tx=e_i
  vector<double> rhs;
  rhs.assign(num_row, 0);
  rhs[row] = 1;
  basisSolveInterface(rhs, row_vector, row_num_nz, row_indices, true);
  return HighsStatus::kOk;
}

HighsStatus Highs::getBasisInverseCol(const HighsInt col, double* col_vector,
                                      HighsInt* col_num_nz,
                                      HighsInt* col_indices) {
  if (col_vector == NULL) {
    highsLogUser(options_.log_options, HighsLogType::kError,
                 "getBasisInverseCol: col_vector is NULL\n");
    return HighsStatus::kError;
  }
  // col_indices can be NULL - it's the trigger that determines
  // whether they are identified or not
  HighsInt num_row = model_.lp_.num_row_;
  if (col < 0 || col >= num_row) {
    highsLogUser(options_.log_options, HighsLogType::kError,
                 "Column index %" HIGHSINT_FORMAT
                 " out of range [0, %" HIGHSINT_FORMAT
                 "] in getBasisInverseCol\n",
                 col, num_row - 1);
    return HighsStatus::kError;
  }
  if (!ekk_instance_.status_.has_invert)
    return invertRequirementError("getBasisInverseCol");
  // Compute a col i of the inverse of the basis matrix by solving Bx=e_i
  vector<double> rhs;
  rhs.assign(num_row, 0);
  rhs[col] = 1;
  basisSolveInterface(rhs, col_vector, col_num_nz, col_indices, false);
  return HighsStatus::kOk;
}

HighsStatus Highs::getBasisSolve(const double* Xrhs, double* solution_vector,
                                 HighsInt* solution_num_nz,
                                 HighsInt* solution_indices) {
  if (Xrhs == NULL) {
    highsLogUser(options_.log_options, HighsLogType::kError,
                 "getBasisSolve: Xrhs is NULL\n");
    return HighsStatus::kError;
  }
  if (solution_vector == NULL) {
    highsLogUser(options_.log_options, HighsLogType::kError,
                 "getBasisSolve: solution_vector is NULL\n");
    return HighsStatus::kError;
  }
  // solution_indices can be NULL - it's the trigger that determines
  // whether they are identified or not
  if (!ekk_instance_.status_.has_invert)
    return invertRequirementError("getBasisSolve");
  HighsInt num_row = model_.lp_.num_row_;
  vector<double> rhs;
  rhs.assign(num_row, 0);
  for (HighsInt row = 0; row < num_row; row++) rhs[row] = Xrhs[row];
  basisSolveInterface(rhs, solution_vector, solution_num_nz, solution_indices,
                      false);
  return HighsStatus::kOk;
}

HighsStatus Highs::getBasisTransposeSolve(const double* Xrhs,
                                          double* solution_vector,
                                          HighsInt* solution_num_nz,
                                          HighsInt* solution_indices) {
  if (Xrhs == NULL) {
    highsLogUser(options_.log_options, HighsLogType::kError,
                 "getBasisTransposeSolve: Xrhs is NULL\n");
    return HighsStatus::kError;
  }
  if (solution_vector == NULL) {
    highsLogUser(options_.log_options, HighsLogType::kError,
                 "getBasisTransposeSolve: solution_vector is NULL\n");
    return HighsStatus::kError;
  }
  // solution_indices can be NULL - it's the trigger that determines
  // whether they are identified or not
  if (!ekk_instance_.status_.has_invert)
    return invertRequirementError("getBasisTransposeSolve");
  HighsInt num_row = model_.lp_.num_row_;
  vector<double> rhs;
  rhs.assign(num_row, 0);
  for (HighsInt row = 0; row < num_row; row++) rhs[row] = Xrhs[row];
  basisSolveInterface(rhs, solution_vector, solution_num_nz, solution_indices,
                      true);
  return HighsStatus::kOk;
}

HighsStatus Highs::getReducedRow(const HighsInt row, double* row_vector,
                                 HighsInt* row_num_nz, HighsInt* row_indices,
                                 const double* pass_basis_inverse_row_vector) {
  HighsStatus return_status = HighsStatus::kOk;
  HighsLp& lp = model_.lp_;
  // Ensure that the LP is column-wise
  lp.ensureColwise();
  if (row_vector == NULL) {
    highsLogUser(options_.log_options, HighsLogType::kError,
                 "getReducedRow: row_vector is NULL\n");
    return HighsStatus::kError;
  }
  // row_indices can be NULL - it's the trigger that determines
  // whether they are identified or not pass_basis_inverse_row_vector
  // NULL - it's the trigger to determine whether it's computed or not
  if (row < 0 || row >= lp.num_row_) {
    highsLogUser(options_.log_options, HighsLogType::kError,
                 "Row index %" HIGHSINT_FORMAT
                 " out of range [0, %" HIGHSINT_FORMAT "] in getReducedRow\n",
                 row, lp.num_row_ - 1);
    return HighsStatus::kError;
  }
  if (!ekk_instance_.status_.has_invert)
    return invertRequirementError("getReducedRow");
  HighsInt num_row = lp.num_row_;
  vector<double> basis_inverse_row;
  double* basis_inverse_row_vector = (double*)pass_basis_inverse_row_vector;
  if (basis_inverse_row_vector == NULL) {
    vector<double> rhs;
    vector<HighsInt> col_indices;
    rhs.assign(num_row, 0);
    rhs[row] = 1;
    basis_inverse_row.resize(num_row, 0);
    // Form B^{-T}e_{row}
    basisSolveInterface(rhs, &basis_inverse_row[0], NULL, NULL, true);
    basis_inverse_row_vector = &basis_inverse_row[0];
  }
  bool return_indices = row_num_nz != NULL;
  if (return_indices) *row_num_nz = 0;
  for (HighsInt col = 0; col < lp.num_col_; col++) {
    double value = 0;
    for (HighsInt el = lp.a_matrix_.start_[col];
         el < lp.a_matrix_.start_[col + 1]; el++) {
      HighsInt row = lp.a_matrix_.index_[el];
      value += lp.a_matrix_.value_[el] * basis_inverse_row_vector[row];
    }
    row_vector[col] = 0;
    if (fabs(value) > kHighsTiny) {
      if (return_indices) row_indices[(*row_num_nz)++] = col;
      row_vector[col] = value;
    }
  }
  return HighsStatus::kOk;
}

HighsStatus Highs::getReducedColumn(const HighsInt col, double* col_vector,
                                    HighsInt* col_num_nz,
                                    HighsInt* col_indices) {
  HighsStatus return_status = HighsStatus::kOk;
  HighsLp& lp = model_.lp_;
  // Ensure that the LP is column-wise
  lp.ensureColwise();
  if (col_vector == NULL) {
    highsLogUser(options_.log_options, HighsLogType::kError,
                 "getReducedColumn: col_vector is NULL\n");
    return HighsStatus::kError;
  }
  // col_indices can be NULL - it's the trigger that determines
  // whether they are identified or not
  if (col < 0 || col >= lp.num_col_) {
    highsLogUser(options_.log_options, HighsLogType::kError,
                 "Column index %" HIGHSINT_FORMAT
                 " out of range [0, %" HIGHSINT_FORMAT
                 "] in getReducedColumn\n",
                 col, lp.num_col_ - 1);
    return HighsStatus::kError;
  }
  if (!ekk_instance_.status_.has_invert)
    return invertRequirementError("getReducedColumn");
  HighsInt num_row = lp.num_row_;
  vector<double> rhs;
  rhs.assign(num_row, 0);
  for (HighsInt el = lp.a_matrix_.start_[col];
       el < lp.a_matrix_.start_[col + 1]; el++)
    rhs[lp.a_matrix_.index_[el]] = lp.a_matrix_.value_[el];
  basisSolveInterface(rhs, col_vector, col_num_nz, col_indices, false);
  return HighsStatus::kOk;
}

HighsStatus Highs::setSolution(const HighsSolution& solution) {
  HighsStatus return_status = HighsStatus::kOk;
  // Determine whether a new solution will be defined. If so,
  // the old solution and any basis are cleared
  const bool new_primal_solution =
      model_.lp_.num_col_ > 0 &&
      solution.col_value.size() >= model_.lp_.num_col_;
  const bool new_dual_solution =
      model_.lp_.num_row_ > 0 &&
      solution.row_dual.size() >= model_.lp_.num_row_;
  const bool new_solution = new_primal_solution || new_dual_solution;

  if (new_solution) invalidateUserSolverData();

  if (new_primal_solution) {
    solution_.col_value = solution.col_value;
    if (model_.lp_.num_row_ > 0) {
      // Worth computing the row values
      solution_.row_value.resize(model_.lp_.num_row_);
      return_status = interpretCallStatus(
          options_.log_options, calculateRowValues(model_.lp_, solution_),
          return_status, "calculateRowValues");
      if (return_status == HighsStatus::kError) return return_status;
    }
    solution_.value_valid = true;
  }
  if (new_dual_solution) {
    solution_.row_dual = solution.row_dual;
    if (model_.lp_.num_col_ > 0) {
      // Worth computing the column duals
      solution_.col_dual.resize(model_.lp_.num_col_);
      return_status = interpretCallStatus(
          options_.log_options, calculateColDuals(model_.lp_, solution_),
          return_status, "calculateColDuals");
      if (return_status == HighsStatus::kError) return return_status;
    }
    solution_.dual_valid = true;
  }
  return returnFromHighs(return_status);
}

HighsStatus Highs::setLogCallback(void (*log_callback)(HighsLogType,
                                                       const char*, void*),
                                  void* log_callback_data) {
  options_.log_options.log_callback = log_callback;
  options_.log_options.log_callback_data = log_callback_data;
  return HighsStatus::kOk;
}

HighsStatus Highs::setBasis(const HighsBasis& basis,
                            const std::string& origin) {
  if (basis.alien) {
    // An alien basis needs to be checked properly, since it may be
    // singular, or even incomplete.
    HighsBasis modifiable_basis = basis;
    modifiable_basis.was_alien = true;
    HighsLpSolverObject solver_object(model_.lp_, modifiable_basis, solution_,
                                      info_, ekk_instance_, options_, timer_);
    HighsStatus return_status = formSimplexLpBasisAndFactor(solver_object);
    if (return_status != HighsStatus::kOk) return HighsStatus::kError;
    // Update the HiGHS basis
    basis_ = std::move(modifiable_basis);
  } else {
    // Check the user-supplied basis
    if (!isBasisConsistent(model_.lp_, basis)) {
      highsLogUser(options_.log_options, HighsLogType::kError,
                   "setBasis: invalid basis\n");
      return HighsStatus::kError;
    }
    // Update the HiGHS basis
    basis_ = basis;
  }
  basis_.valid = true;
  if (origin != "") basis_.debug_origin_name = origin;
  assert(basis_.debug_origin_name != "");
  assert(!basis_.alien);
  if (basis_.was_alien) {
    highsLogDev(
        options_.log_options, HighsLogType::kInfo,
        "Highs::setBasis Was alien = %-5s; Id = %9d; UpdateCount = %4d; Origin "
        "(%s)\n",
        highsBoolToString(basis_.was_alien).c_str(), (int)basis_.debug_id,
        (int)basis_.debug_update_count, basis_.debug_origin_name.c_str());
  }

  // Follow implications of a new HiGHS basis
  newHighsBasis();
  // Can't use returnFromHighs since...
  return HighsStatus::kOk;
}

HighsStatus Highs::setBasis() {
  // Invalidate the basis for HiGHS
  //
  // Don't set to logical basis since that causes presolve to be
  // skipped
  basis_.invalidate();
  // Follow implications of a new HiGHS basis
  newHighsBasis();
  // Can't use returnFromHighs since...
  return HighsStatus::kOk;
}

HighsStatus Highs::setHotStart(const HotStart& hot_start) {
  // Check that the user-supplied hot start is valid
  if (!hot_start.valid) {
    highsLogUser(options_.log_options, HighsLogType::kError,
                 "setHotStart: invalid hot start\n");
    return HighsStatus::kError;
  }
  HighsStatus return_status = setHotStartInterface(hot_start);
  return returnFromHighs(return_status);
}

HighsStatus Highs::freezeBasis(HighsInt& frozen_basis_id) {
  frozen_basis_id = kNoLink;
  // Check that there is a simplex basis to freeze
  if (!ekk_instance_.status_.has_invert) {
    highsLogUser(options_.log_options, HighsLogType::kError,
                 "freezeBasis: no simplex factorization to freeze\n");
    return HighsStatus::kError;
  }
  ekk_instance_.freezeBasis(frozen_basis_id);
  return returnFromHighs(HighsStatus::kOk);
}

HighsStatus Highs::unfreezeBasis(const HighsInt frozen_basis_id) {
  // Check that there is a simplex basis to unfreeze
  if (!ekk_instance_.status_.initialised_for_new_lp) {
    highsLogUser(options_.log_options, HighsLogType::kError,
                 "unfreezeBasis: no simplex information to unfreeze\n");
    return HighsStatus::kError;
  }
  HighsStatus call_status = ekk_instance_.unfreezeBasis(frozen_basis_id);
  if (call_status != HighsStatus::kOk) return call_status;
  // Reset simplex NLA pointers
  ekk_instance_.setNlaPointersForTrans(model_.lp_);
  // Get the corresponding HiGHS basis
  basis_ = ekk_instance_.getHighsBasis(model_.lp_);
  // Clear everything else
  invalidateModelStatusSolutionAndInfo();
  return returnFromHighs(HighsStatus::kOk);
}

HighsStatus Highs::putIterate() {
  // Check that there is a simplex iterate to put
  if (!ekk_instance_.status_.has_invert) {
    highsLogUser(options_.log_options, HighsLogType::kError,
                 "putIterate: no simplex iterate to put\n");
    return HighsStatus::kError;
  }
  ekk_instance_.putIterate();
  return returnFromHighs(HighsStatus::kOk);
}

HighsStatus Highs::getIterate() {
  // Check that there is a simplex iterate to get
  if (!ekk_instance_.status_.initialised_for_new_lp) {
    highsLogUser(options_.log_options, HighsLogType::kError,
                 "getIterate: no simplex iterate to get\n");
    return HighsStatus::kError;
  }
  HighsStatus call_status = ekk_instance_.getIterate();
  if (call_status != HighsStatus::kOk) return call_status;
  // Get the corresponding HiGHS basis
  basis_ = ekk_instance_.getHighsBasis(model_.lp_);
  // Clear everything else
  invalidateModelStatusSolutionAndInfo();
  return returnFromHighs(HighsStatus::kOk);
}

HighsStatus Highs::addCol(const double cost, const double lower_bound,
                          const double upper_bound, const HighsInt num_new_nz,
                          const HighsInt* indices, const double* values) {
  this->logHeader();
  HighsInt starts = 0;
  return addCols(1, &cost, &lower_bound, &upper_bound, num_new_nz, &starts,
                 indices, values);
}

HighsStatus Highs::addCols(const HighsInt num_new_col, const double* costs,
                           const double* lower_bounds,
                           const double* upper_bounds,
                           const HighsInt num_new_nz, const HighsInt* starts,
                           const HighsInt* indices, const double* values) {
  this->logHeader();
  HighsStatus return_status = HighsStatus::kOk;
  clearPresolve();
  return_status = interpretCallStatus(
      options_.log_options,
      addColsInterface(num_new_col, costs, lower_bounds, upper_bounds,
                       num_new_nz, starts, indices, values),
      return_status, "addCols");
  if (return_status == HighsStatus::kError) return HighsStatus::kError;
  return returnFromHighs(return_status);
}

HighsStatus Highs::addVars(const HighsInt num_new_var, const double* lower,
                           const double* upper) {
  this->logHeader();
  HighsStatus return_status = HighsStatus::kOk;
  // Avoid touching entry [0] of a vector of size 0
  if (num_new_var <= 0) returnFromHighs(return_status);
  std::vector<double> cost;
  cost.assign(num_new_var, 0);
  return addCols(num_new_var, &cost[0], lower, upper, 0, nullptr, nullptr,
                 nullptr);
}

HighsStatus Highs::addRow(const double lower_bound, const double upper_bound,
                          const HighsInt num_new_nz, const HighsInt* indices,
                          const double* values) {
  this->logHeader();
  HighsInt starts = 0;
  return addRows(1, &lower_bound, &upper_bound, num_new_nz, &starts, indices,
                 values);
}

HighsStatus Highs::addRows(const HighsInt num_new_row,
                           const double* lower_bounds,
                           const double* upper_bounds,
                           const HighsInt num_new_nz, const HighsInt* starts,
                           const HighsInt* indices, const double* values) {
  this->logHeader();
  HighsStatus return_status = HighsStatus::kOk;
  clearPresolve();
  return_status = interpretCallStatus(
      options_.log_options,
      addRowsInterface(num_new_row, lower_bounds, upper_bounds, num_new_nz,
                       starts, indices, values),
      return_status, "addRows");
  if (return_status == HighsStatus::kError) return HighsStatus::kError;
  return returnFromHighs(return_status);
}

HighsStatus Highs::changeObjectiveSense(const ObjSense sense) {
  if ((sense == ObjSense::kMinimize) !=
      (model_.lp_.sense_ == ObjSense::kMinimize)) {
    model_.lp_.sense_ = sense;
    // Nontrivial change
    clearPresolve();
    invalidateModelStatusSolutionAndInfo();
  }
  return returnFromHighs(HighsStatus::kOk);
}

HighsStatus Highs::changeObjectiveOffset(const double offset) {
  // Update the objective value
  info_.objective_function_value += (offset - model_.lp_.offset_);
  model_.lp_.offset_ = offset;
  presolved_model_.lp_.offset_ += offset;
  return returnFromHighs(HighsStatus::kOk);
}

HighsStatus Highs::changeColIntegrality(const HighsInt col,
                                        const HighsVarType integrality) {
  return changeColsIntegrality(1, &col, &integrality);
}

HighsStatus Highs::changeColsIntegrality(const HighsInt from_col,
                                         const HighsInt to_col,
                                         const HighsVarType* integrality) {
  clearPresolve();
  HighsIndexCollection index_collection;
  if (!create(index_collection, from_col, to_col, model_.lp_.num_col_)) {
    highsLogUser(
        options_.log_options, HighsLogType::kError,
        "Interval supplied to Highs::changeColsIntegrality is out of range\n");
    return HighsStatus::kError;
  }
  HighsStatus call_status =
      changeIntegralityInterface(index_collection, integrality);
  HighsStatus return_status = HighsStatus::kOk;
  return_status = interpretCallStatus(options_.log_options, call_status,
                                      return_status, "changeIntegrality");
  if (return_status == HighsStatus::kError) return HighsStatus::kError;
  return returnFromHighs(return_status);
}

HighsStatus Highs::changeColsIntegrality(const HighsInt num_set_entries,
                                         const HighsInt* set,
                                         const HighsVarType* integrality) {
  if (num_set_entries <= 0) return HighsStatus::kOk;
  clearPresolve();
  // Ensure that the set and data are in ascending order
  std::vector<HighsVarType> local_integrality{integrality,
                                              integrality + num_set_entries};
  std::vector<HighsInt> local_set{set, set + num_set_entries};
  sortSetData(num_set_entries, local_set, integrality, &local_integrality[0]);
  HighsIndexCollection index_collection;
  const bool create_ok = create(index_collection, num_set_entries,
                                &local_set[0], model_.lp_.num_col_);
  assert(create_ok);
  HighsStatus call_status =
      changeIntegralityInterface(index_collection, &local_integrality[0]);
  HighsStatus return_status = HighsStatus::kOk;
  return_status = interpretCallStatus(options_.log_options, call_status,
                                      return_status, "changeIntegrality");
  if (return_status == HighsStatus::kError) return HighsStatus::kError;
  return returnFromHighs(return_status);
}

HighsStatus Highs::changeColsIntegrality(const HighsInt* mask,
                                         const HighsVarType* integrality) {
  clearPresolve();
  HighsIndexCollection index_collection;
  create(index_collection, mask, model_.lp_.num_col_);
  HighsStatus call_status =
      changeIntegralityInterface(index_collection, integrality);
  HighsStatus return_status = HighsStatus::kOk;
  return_status = interpretCallStatus(options_.log_options, call_status,
                                      return_status, "changeIntegrality");
  if (return_status == HighsStatus::kError) return HighsStatus::kError;
  return returnFromHighs(return_status);
}

HighsStatus Highs::changeColCost(const HighsInt col, const double cost) {
  return changeColsCost(1, &col, &cost);
}

HighsStatus Highs::changeColsCost(const HighsInt from_col,
                                  const HighsInt to_col, const double* cost) {
  clearPresolve();
  HighsIndexCollection index_collection;
  if (!create(index_collection, from_col, to_col, model_.lp_.num_col_)) {
    highsLogUser(
        options_.log_options, HighsLogType::kError,
        "Interval supplied to Highs::changeColsCost is out of range\n");
    return HighsStatus::kError;
  }
  HighsStatus call_status = changeCostsInterface(index_collection, cost);
  HighsStatus return_status = HighsStatus::kOk;
  return_status = interpretCallStatus(options_.log_options, call_status,
                                      return_status, "changeCosts");
  if (return_status == HighsStatus::kError) return HighsStatus::kError;
  return returnFromHighs(return_status);
}

HighsStatus Highs::changeColsCost(const HighsInt num_set_entries,
                                  const HighsInt* set, const double* cost) {
  if (num_set_entries <= 0) return HighsStatus::kOk;
  // Check for NULL data in "set" version of changeColsCost since
  // values are sorted with set
  if (doubleUserDataNotNull(options_.log_options, cost, "column costs"))
    return HighsStatus::kError;
  clearPresolve();
  // Ensure that the set and data are in ascending order
  std::vector<double> local_cost{cost, cost + num_set_entries};
  std::vector<HighsInt> local_set{set, set + num_set_entries};
  sortSetData(num_set_entries, local_set, cost, NULL, NULL, &local_cost[0],
              NULL, NULL);
  HighsIndexCollection index_collection;
  const bool create_ok = create(index_collection, num_set_entries,
                                &local_set[0], model_.lp_.num_col_);
  assert(create_ok);
  HighsStatus call_status =
      changeCostsInterface(index_collection, &local_cost[0]);
  HighsStatus return_status = HighsStatus::kOk;
  return_status = interpretCallStatus(options_.log_options, call_status,
                                      return_status, "changeCosts");
  if (return_status == HighsStatus::kError) return HighsStatus::kError;
  return returnFromHighs(return_status);
}

HighsStatus Highs::changeColsCost(const HighsInt* mask, const double* cost) {
  clearPresolve();
  HighsIndexCollection index_collection;
  create(index_collection, mask, model_.lp_.num_col_);
  HighsStatus call_status = changeCostsInterface(index_collection, cost);
  HighsStatus return_status = HighsStatus::kOk;
  return_status = interpretCallStatus(options_.log_options, call_status,
                                      return_status, "changeCosts");
  if (return_status == HighsStatus::kError) return HighsStatus::kError;
  return returnFromHighs(return_status);
}

HighsStatus Highs::changeColBounds(const HighsInt col, const double lower,
                                   const double upper) {
  return changeColsBounds(1, &col, &lower, &upper);
}

HighsStatus Highs::changeColsBounds(const HighsInt from_col,
                                    const HighsInt to_col, const double* lower,
                                    const double* upper) {
  clearPresolve();
  HighsIndexCollection index_collection;
  if (!create(index_collection, from_col, to_col, model_.lp_.num_col_)) {
    highsLogUser(
        options_.log_options, HighsLogType::kError,
        "Interval supplied to Highs::changeColsBounds is out of range\n");
    return HighsStatus::kError;
  }
  HighsStatus call_status =
      changeColBoundsInterface(index_collection, lower, upper);
  HighsStatus return_status = HighsStatus::kOk;
  return_status = interpretCallStatus(options_.log_options, call_status,
                                      return_status, "changeColBounds");
  if (return_status == HighsStatus::kError) return HighsStatus::kError;
  return returnFromHighs(return_status);
}

HighsStatus Highs::changeColsBounds(const HighsInt num_set_entries,
                                    const HighsInt* set, const double* lower,
                                    const double* upper) {
  if (num_set_entries <= 0) return HighsStatus::kOk;
  // Check for NULL data in "set" version of changeColsBounds since
  // values are sorted with set
  bool null_data = false;
  null_data = doubleUserDataNotNull(options_.log_options, lower,
                                    "column lower bounds") ||
              null_data;
  null_data = doubleUserDataNotNull(options_.log_options, upper,
                                    "column upper bounds") ||
              null_data;
  if (null_data) return HighsStatus::kError;
  clearPresolve();
  // Ensure that the set and data are in ascending order
  std::vector<double> local_lower{lower, lower + num_set_entries};
  std::vector<double> local_upper{upper, upper + num_set_entries};
  std::vector<HighsInt> local_set{set, set + num_set_entries};
  sortSetData(num_set_entries, local_set, lower, upper, NULL, &local_lower[0],
              &local_upper[0], NULL);
  HighsIndexCollection index_collection;
  const bool create_ok = create(index_collection, num_set_entries,
                                &local_set[0], model_.lp_.num_col_);
  assert(create_ok);
  HighsStatus call_status = changeColBoundsInterface(
      index_collection, &local_lower[0], &local_upper[0]);
  HighsStatus return_status = HighsStatus::kOk;
  return_status = interpretCallStatus(options_.log_options, call_status,
                                      return_status, "changeColBounds");
  if (return_status == HighsStatus::kError) return HighsStatus::kError;
  return returnFromHighs(return_status);
}

HighsStatus Highs::changeColsBounds(const HighsInt* mask, const double* lower,
                                    const double* upper) {
  clearPresolve();
  HighsIndexCollection index_collection;
  create(index_collection, mask, model_.lp_.num_col_);
  HighsStatus call_status =
      changeColBoundsInterface(index_collection, lower, upper);
  HighsStatus return_status = HighsStatus::kOk;
  return_status = interpretCallStatus(options_.log_options, call_status,
                                      return_status, "changeColBounds");
  if (return_status == HighsStatus::kError) return HighsStatus::kError;
  return returnFromHighs(return_status);
}

HighsStatus Highs::changeRowBounds(const HighsInt row, const double lower,
                                   const double upper) {
  return changeRowsBounds(1, &row, &lower, &upper);
}

HighsStatus Highs::changeRowsBounds(const HighsInt from_row,
                                    const HighsInt to_row, const double* lower,
                                    const double* upper) {
  clearPresolve();
  HighsIndexCollection index_collection;
  if (!create(index_collection, from_row, to_row, model_.lp_.num_row_)) {
    highsLogUser(
        options_.log_options, HighsLogType::kError,
        "Interval supplied to Highs::changeRowsBounds is out of range\n");
    return HighsStatus::kError;
  }
  HighsStatus call_status =
      changeRowBoundsInterface(index_collection, lower, upper);
  HighsStatus return_status = HighsStatus::kOk;
  return_status = interpretCallStatus(options_.log_options, call_status,
                                      return_status, "changeRowBounds");
  if (return_status == HighsStatus::kError) return HighsStatus::kError;
  return returnFromHighs(return_status);
}

HighsStatus Highs::changeRowsBounds(const HighsInt num_set_entries,
                                    const HighsInt* set, const double* lower,
                                    const double* upper) {
  if (num_set_entries <= 0) return HighsStatus::kOk;
  // Check for NULL data in "set" version of changeRowsBounds since
  // values are sorted with set
  bool null_data = false;
  null_data =
      doubleUserDataNotNull(options_.log_options, lower, "row lower bounds") ||
      null_data;
  null_data =
      doubleUserDataNotNull(options_.log_options, upper, "row upper bounds") ||
      null_data;
  if (null_data) return HighsStatus::kError;
  clearPresolve();
  // Ensure that the set and data are in ascending order
  std::vector<double> local_lower{lower, lower + num_set_entries};
  std::vector<double> local_upper{upper, upper + num_set_entries};
  std::vector<HighsInt> local_set{set, set + num_set_entries};
  sortSetData(num_set_entries, local_set, lower, upper, NULL, &local_lower[0],
              &local_upper[0], NULL);
  HighsIndexCollection index_collection;
  const bool create_ok = create(index_collection, num_set_entries,
                                &local_set[0], model_.lp_.num_row_);
  assert(create_ok);
  HighsStatus call_status = changeRowBoundsInterface(
      index_collection, &local_lower[0], &local_upper[0]);
  HighsStatus return_status = HighsStatus::kOk;
  return_status = interpretCallStatus(options_.log_options, call_status,
                                      return_status, "changeRowBounds");
  if (return_status == HighsStatus::kError) return HighsStatus::kError;
  return returnFromHighs(return_status);
}

HighsStatus Highs::changeRowsBounds(const HighsInt* mask, const double* lower,
                                    const double* upper) {
  clearPresolve();
  HighsIndexCollection index_collection;
  create(index_collection, mask, model_.lp_.num_row_);
  HighsStatus call_status =
      changeRowBoundsInterface(index_collection, lower, upper);
  HighsStatus return_status = HighsStatus::kOk;
  return_status = interpretCallStatus(options_.log_options, call_status,
                                      return_status, "changeRowBounds");
  if (return_status == HighsStatus::kError) return HighsStatus::kError;
  return returnFromHighs(return_status);
}

HighsStatus Highs::changeCoeff(const HighsInt row, const HighsInt col,
                               const double value) {
  if (row < 0 || row >= model_.lp_.num_row_) {
    highsLogUser(options_.log_options, HighsLogType::kError,
                 "Row %" HIGHSINT_FORMAT
                 " supplied to Highs::changeCoeff is not in the range [0, "
                 "%" HIGHSINT_FORMAT "]\n",
                 row, model_.lp_.num_row_);
    return HighsStatus::kError;
  }
  if (col < 0 || col >= model_.lp_.num_col_) {
    highsLogUser(options_.log_options, HighsLogType::kError,
                 "Col %" HIGHSINT_FORMAT
                 " supplied to Highs::changeCoeff is not in the range [0, "
                 "%" HIGHSINT_FORMAT "]\n",
                 col, model_.lp_.num_col_);
    return HighsStatus::kError;
  }
  const double abs_value = std::fabs(value);
  if (0 < abs_value && abs_value <= options_.small_matrix_value) {
    highsLogUser(options_.log_options, HighsLogType::kWarning,
                 "|Value| of %g supplied to Highs::changeCoeff is in (0, %g]: "
                 "zeroes any existing coefficient, otherwise ignored\n",
                 abs_value, options_.small_matrix_value);
  }
  changeCoefficientInterface(row, col, value);
  return returnFromHighs(HighsStatus::kOk);
}

HighsStatus Highs::getObjectiveSense(ObjSense& sense) const {
  sense = model_.lp_.sense_;
  return HighsStatus::kOk;
}

HighsStatus Highs::getObjectiveOffset(double& offset) const {
  offset = model_.lp_.offset_;
  return HighsStatus::kOk;
}

HighsStatus Highs::getCols(const HighsInt from_col, const HighsInt to_col,
                           HighsInt& num_col, double* costs, double* lower,
                           double* upper, HighsInt& num_nz, HighsInt* start,
                           HighsInt* index, double* value) {
  HighsIndexCollection index_collection;
  if (!create(index_collection, from_col, to_col, model_.lp_.num_col_)) {
    highsLogUser(options_.log_options, HighsLogType::kError,
                 "Interval supplied to Highs::getCols is out of range\n");
    return HighsStatus::kError;
  }
  getColsInterface(index_collection, num_col, costs, lower, upper, num_nz,
                   start, index, value);
  return returnFromHighs(HighsStatus::kOk);
}

HighsStatus Highs::getCols(const HighsInt num_set_entries, const HighsInt* set,
                           HighsInt& num_col, double* costs, double* lower,
                           double* upper, HighsInt& num_nz, HighsInt* start,
                           HighsInt* index, double* value) {
  if (num_set_entries <= 0) return HighsStatus::kOk;
  HighsIndexCollection index_collection;
  if (!create(index_collection, num_set_entries, set, model_.lp_.num_col_)) {
    highsLogUser(options_.log_options, HighsLogType::kError,
                 "Set supplied to Highs::getCols not ordered\n");
    return HighsStatus::kError;
  }
  getColsInterface(index_collection, num_col, costs, lower, upper, num_nz,
                   start, index, value);
  return returnFromHighs(HighsStatus::kOk);
}

HighsStatus Highs::getCols(const HighsInt* mask, HighsInt& num_col,
                           double* costs, double* lower, double* upper,
                           HighsInt& num_nz, HighsInt* start, HighsInt* index,
                           double* value) {
  HighsIndexCollection index_collection;
  create(index_collection, mask, model_.lp_.num_col_);
  getColsInterface(index_collection, num_col, costs, lower, upper, num_nz,
                   start, index, value);
  return returnFromHighs(HighsStatus::kOk);
}

HighsStatus Highs::getRows(const HighsInt from_row, const HighsInt to_row,
                           HighsInt& num_row, double* lower, double* upper,
                           HighsInt& num_nz, HighsInt* start, HighsInt* index,
                           double* value) {
  HighsIndexCollection index_collection;
  if (!create(index_collection, from_row, to_row, model_.lp_.num_row_)) {
    highsLogUser(options_.log_options, HighsLogType::kError,
                 "Interval supplied to Highs::getRows is out of range\n");
    return HighsStatus::kError;
  }
  getRowsInterface(index_collection, num_row, lower, upper, num_nz, start,
                   index, value);
  return returnFromHighs(HighsStatus::kOk);
}

HighsStatus Highs::getRows(const HighsInt num_set_entries, const HighsInt* set,
                           HighsInt& num_row, double* lower, double* upper,
                           HighsInt& num_nz, HighsInt* start, HighsInt* index,
                           double* value) {
  if (num_set_entries <= 0) return HighsStatus::kOk;
  HighsIndexCollection index_collection;
  if (!create(index_collection, num_set_entries, set, model_.lp_.num_row_)) {
    highsLogUser(options_.log_options, HighsLogType::kError,
                 "Set supplied to Highs::getRows is not ordered\n");
    return HighsStatus::kError;
  }
  getRowsInterface(index_collection, num_row, lower, upper, num_nz, start,
                   index, value);
  return returnFromHighs(HighsStatus::kOk);
}

HighsStatus Highs::getRows(const HighsInt* mask, HighsInt& num_row,
                           double* lower, double* upper, HighsInt& num_nz,
                           HighsInt* start, HighsInt* index, double* value) {
  HighsIndexCollection index_collection;
  create(index_collection, mask, model_.lp_.num_row_);
  getRowsInterface(index_collection, num_row, lower, upper, num_nz, start,
                   index, value);
  return returnFromHighs(HighsStatus::kOk);
}

HighsStatus Highs::getCoeff(const HighsInt row, const HighsInt col,
                            double& value) {
  if (row < 0 || row >= model_.lp_.num_row_) {
    highsLogUser(
        options_.log_options, HighsLogType::kError,
        "Row %" HIGHSINT_FORMAT
        " supplied to Highs::getCoeff is not in the range [0, %" HIGHSINT_FORMAT
        "]\n",
        row, model_.lp_.num_row_);
    return HighsStatus::kError;
  }
  if (col < 0 || col >= model_.lp_.num_col_) {
    highsLogUser(
        options_.log_options, HighsLogType::kError,
        "Col %" HIGHSINT_FORMAT
        " supplied to Highs::getCoeff is not in the range [0, %" HIGHSINT_FORMAT
        "]\n",
        col, model_.lp_.num_col_);
    return HighsStatus::kError;
  }
  getCoefficientInterface(row, col, value);
  return returnFromHighs(HighsStatus::kOk);
}

HighsStatus Highs::deleteCols(const HighsInt from_col, const HighsInt to_col) {
  clearPresolve();
  HighsIndexCollection index_collection;
  if (!create(index_collection, from_col, to_col, model_.lp_.num_col_)) {
    highsLogUser(options_.log_options, HighsLogType::kError,
                 "Interval supplied to Highs::deleteCols is out of range\n");
    return HighsStatus::kError;
  }
  deleteColsInterface(index_collection);
  return returnFromHighs(HighsStatus::kOk);
}

HighsStatus Highs::deleteCols(const HighsInt num_set_entries,
                              const HighsInt* set) {
  if (num_set_entries <= 0) return HighsStatus::kOk;
  clearPresolve();
  HighsIndexCollection index_collection;
  if (!create(index_collection, num_set_entries, set, model_.lp_.num_col_)) {
    highsLogUser(options_.log_options, HighsLogType::kError,
                 "Set supplied to Highs::deleteCols is not ordered\n");
    return HighsStatus::kError;
  }
  deleteColsInterface(index_collection);
  return returnFromHighs(HighsStatus::kOk);
}

HighsStatus Highs::deleteCols(HighsInt* mask) {
  clearPresolve();
  const HighsInt original_num_col = model_.lp_.num_col_;
  HighsIndexCollection index_collection;
  create(index_collection, mask, original_num_col);
  deleteColsInterface(index_collection);
  for (HighsInt iCol = 0; iCol < original_num_col; iCol++)
    mask[iCol] = index_collection.mask_[iCol];
  return returnFromHighs(HighsStatus::kOk);
}

HighsStatus Highs::deleteRows(const HighsInt from_row, const HighsInt to_row) {
  clearPresolve();
  HighsIndexCollection index_collection;
  if (!create(index_collection, from_row, to_row, model_.lp_.num_row_)) {
    highsLogUser(options_.log_options, HighsLogType::kError,
                 "Interval supplied to Highs::deleteRows is out of range\n");
    return HighsStatus::kError;
  }
  deleteRowsInterface(index_collection);
  return returnFromHighs(HighsStatus::kOk);
}

HighsStatus Highs::deleteRows(const HighsInt num_set_entries,
                              const HighsInt* set) {
  if (num_set_entries <= 0) return HighsStatus::kOk;
  clearPresolve();
  HighsIndexCollection index_collection;
  if (!create(index_collection, num_set_entries, set, model_.lp_.num_row_)) {
    highsLogUser(options_.log_options, HighsLogType::kError,
                 "Set supplied to Highs::deleteRows is not ordered\n");
    return HighsStatus::kError;
  }
  deleteRowsInterface(index_collection);
  return returnFromHighs(HighsStatus::kOk);
}

HighsStatus Highs::deleteRows(HighsInt* mask) {
  clearPresolve();
  const HighsInt original_num_row = model_.lp_.num_row_;
  HighsIndexCollection index_collection;
  create(index_collection, mask, original_num_row);
  deleteRowsInterface(index_collection);
  for (HighsInt iRow = 0; iRow < original_num_row; iRow++)
    mask[iRow] = index_collection.mask_[iRow];
  return returnFromHighs(HighsStatus::kOk);
}

HighsStatus Highs::scaleCol(const HighsInt col, const double scale_value) {
  HighsStatus return_status = HighsStatus::kOk;
  clearPresolve();
  HighsStatus call_status = scaleColInterface(col, scale_value);
  return_status = interpretCallStatus(options_.log_options, call_status,
                                      return_status, "scaleCol");
  if (return_status == HighsStatus::kError) return HighsStatus::kError;
  return returnFromHighs(return_status);
}

HighsStatus Highs::scaleRow(const HighsInt row, const double scale_value) {
  HighsStatus return_status = HighsStatus::kOk;
  clearPresolve();
  HighsStatus call_status = scaleRowInterface(row, scale_value);
  return_status = interpretCallStatus(options_.log_options, call_status,
                                      return_status, "scaleRow");
  if (return_status == HighsStatus::kError) return HighsStatus::kError;
  return returnFromHighs(return_status);
}

HighsStatus Highs::postsolve(const HighsSolution& solution,
                             const HighsBasis& basis) {
  const bool can_run_postsolve =
      model_presolve_status_ == HighsPresolveStatus::kNotPresolved ||
      model_presolve_status_ == HighsPresolveStatus::kReduced ||
      model_presolve_status_ == HighsPresolveStatus::kReducedToEmpty ||
      model_presolve_status_ == HighsPresolveStatus::kTimeout;
  if (!can_run_postsolve) {
    highsLogUser(
        options_.log_options, HighsLogType::kWarning,
        "Cannot run postsolve with presolve status: %s\n",
        presolve_.presolveStatusToString(model_presolve_status_).c_str());
    return HighsStatus::kWarning;
  }
  HighsStatus return_status = callRunPostsolve(solution, basis);
  return returnFromHighs(return_status);
}

HighsStatus Highs::writeSolution(const std::string& filename,
                                 const HighsInt style) {
  HighsStatus return_status = HighsStatus::kOk;
  HighsStatus call_status;
  FILE* file;
  bool html;
  call_status = openWriteFile(filename, "writeSolution", file, html);
  return_status = interpretCallStatus(options_.log_options, call_status,
                                      return_status, "openWriteFile");
  if (return_status == HighsStatus::kError) return return_status;
  writeSolutionFile(file, options_, model_, basis_, solution_, info_,
                    model_status_, style);
  if (style == kSolutionStyleRaw) {
    fprintf(file, "\n# Basis\n");
    writeBasisFile(file, basis_);
  }
  if (options_.ranging == kHighsOnString) {
    if (model_.isMip() || model_.isQp()) {
      highsLogUser(options_.log_options, HighsLogType::kError,
                   "Cannot determing ranging information for MIP or QP\n");
      return HighsStatus::kError;
    }
    return_status = interpretCallStatus(
        options_.log_options, this->getRanging(), return_status, "getRanging");
    if (return_status == HighsStatus::kError) return return_status;
    fprintf(file, "\n# Ranging\n");
    writeRangingFile(file, model_.lp_, info_.objective_function_value, basis_,
                     solution_, ranging_, style);
  }
  if (file != stdout) fclose(file);
  return HighsStatus::kOk;
}

HighsStatus Highs::readSolution(const std::string& filename,
                                const HighsInt style) {
  return readSolutionFile(filename, options_, model_.lp_, basis_, solution_,
                          style);
}

HighsStatus Highs::checkSolutionFeasibility() {
  checkLpSolutionFeasibility(options_, model_.lp_, solution_);
  return HighsStatus::kOk;
}

std::string Highs::modelStatusToString(
    const HighsModelStatus model_status) const {
  return utilModelStatusToString(model_status);
}

std::string Highs::solutionStatusToString(
    const HighsInt solution_status) const {
  return utilSolutionStatusToString(solution_status);
}

std::string Highs::basisStatusToString(
    const HighsBasisStatus basis_status) const {
  return utilBasisStatusToString(basis_status);
}

std::string Highs::basisValidityToString(const HighsInt basis_validity) const {
  return utilBasisValidityToString(basis_validity);
}

// Private methods
void Highs::deprecationMessage(const std::string& method_name,
                               const std::string& alt_method_name) const {
  if (alt_method_name.compare("None") == 0) {
    highsLogUser(options_.log_options, HighsLogType::kWarning,
                 "Method %s is deprecated: no alternative method\n",
                 method_name.c_str());
  } else {
    highsLogUser(options_.log_options, HighsLogType::kWarning,
                 "Method %s is deprecated: alternative method is %s\n",
                 method_name.c_str(), alt_method_name.c_str());
  }
}

HighsPresolveStatus Highs::runPresolve(const bool force_presolve) {
  presolve_.clear();
  // Exit if presolve is set to off (unless presolve is forced)
  if (options_.presolve == kHighsOffString && !force_presolve)
    return HighsPresolveStatus::kNotPresolved;

  if (model_.isEmpty()) {
    // Empty models shouldn't reach here, but this status would cause
    // no harm if one did
    assert(1 == 0);
    return HighsPresolveStatus::kNotReduced;
  }

  // Ensure that the LP is column-wise
  HighsLp& original_lp = model_.lp_;
  original_lp.ensureColwise();

  if (original_lp.num_col_ == 0 && original_lp.num_row_ == 0)
    return HighsPresolveStatus::kNullError;

  // Clear info from previous runs if original_lp has been modified.
  double start_presolve = timer_.readRunHighsClock();

  // Set time limit.
  if (options_.time_limit > 0 && options_.time_limit < kHighsInf) {
    double left = options_.time_limit - start_presolve;
    if (left <= 0) {
      highsLogDev(options_.log_options, HighsLogType::kError,
                  "Time limit reached while reading in matrix\n");
      return HighsPresolveStatus::kTimeout;
    }

    highsLogDev(options_.log_options, HighsLogType::kVerbose,
                "Time limit set: reading matrix took %.2g, presolve "
                "time left: %.2g\n",
                start_presolve, left);
  }

  // Presolve.
  presolve_.init(original_lp, timer_);
  presolve_.options_ = &options_;
  if (options_.time_limit > 0 && options_.time_limit < kHighsInf) {
    double current = timer_.readRunHighsClock();
    double time_init = current - start_presolve;
    double left = presolve_.options_->time_limit - time_init;
    if (left <= 0) {
      highsLogDev(options_.log_options, HighsLogType::kError,
                  "Time limit reached while copying matrix into presolve.\n");
      return HighsPresolveStatus::kTimeout;
    }
    highsLogDev(options_.log_options, HighsLogType::kVerbose,
                "Time limit set: copying matrix took %.2g, presolve "
                "time left: %.2g\n",
                time_init, left);
  }

  HighsPresolveStatus presolve_return_status = presolve_.run();

  highsLogDev(options_.log_options, HighsLogType::kVerbose,
              "presolve_.run() returns status: %s\n",
              presolve_.presolveStatusToString(presolve_return_status).c_str());

  // Update reduction counts.
  assert(presolve_return_status == presolve_.presolve_status_);
  switch (presolve_.presolve_status_) {
    case HighsPresolveStatus::kReduced: {
      HighsLp& reduced_lp = presolve_.getReducedProblem();
      presolve_.info_.n_cols_removed =
          original_lp.num_col_ - reduced_lp.num_col_;
      presolve_.info_.n_rows_removed =
          original_lp.num_row_ - reduced_lp.num_row_;
      presolve_.info_.n_nnz_removed = (HighsInt)original_lp.a_matrix_.numNz() -
                                      (HighsInt)reduced_lp.a_matrix_.numNz();
      // Clear any scaling information inherited by the reduced LP
      reduced_lp.clearScale();
      assert(lpDimensionsOk("RunPresolve: reduced_lp", reduced_lp,
                            options_.log_options));
      break;
    }
    case HighsPresolveStatus::kReducedToEmpty: {
      presolve_.info_.n_cols_removed = original_lp.num_col_;
      presolve_.info_.n_rows_removed = original_lp.num_row_;
      presolve_.info_.n_nnz_removed = (HighsInt)original_lp.a_matrix_.numNz();
      break;
    }
    default:
      break;
  }
  return presolve_return_status;
}

HighsPostsolveStatus Highs::runPostsolve() {
  // assert(presolve_.has_run_);
  const bool have_primal_solution =
      presolve_.data_.recovered_solution_.value_valid;
  // Need at least a primal solution
  if (!have_primal_solution)
    return HighsPostsolveStatus::kNoPrimalSolutionError;
  const bool have_dual_solution =
      presolve_.data_.recovered_solution_.dual_valid;
  presolve_.data_.postSolveStack.undo(options_,
                                      presolve_.data_.recovered_solution_,
                                      presolve_.data_.recovered_basis_);
  // Compute the row activities
  calculateRowValuesQuad(model_.lp_, presolve_.data_.recovered_solution_);

  if (have_dual_solution && model_.lp_.sense_ == ObjSense::kMaximize)
    presolve_.negateReducedLpColDuals(true);

  // Ensure that the postsolve status is used to set
  // presolve_.postsolve_status_, as well as being returned
  HighsPostsolveStatus postsolve_status =
      HighsPostsolveStatus::kSolutionRecovered;
  presolve_.postsolve_status_ = postsolve_status;
  return postsolve_status;
}

void Highs::clearPresolve() {
  model_presolve_status_ = HighsPresolveStatus::kNotPresolved;
  presolved_model_.clear();
  presolve_.clear();
}

void Highs::invalidateUserSolverData() {
  invalidateModelStatus();
  invalidateSolution();
  invalidateBasis();
  invalidateRanging();
  invalidateInfo();
  invalidateEkk();
}

void Highs::invalidateModelStatusSolutionAndInfo() {
  invalidateModelStatus();
  invalidateSolution();
  invalidateInfo();
}

void Highs::invalidateModelStatus() {
  model_status_ = HighsModelStatus::kNotset;
}

void Highs::invalidateSolution() {
  info_.primal_solution_status = kSolutionStatusNone;
  info_.dual_solution_status = kSolutionStatusNone;
  info_.num_primal_infeasibilities = kHighsIllegalInfeasibilityCount;
  info_.max_primal_infeasibility = kHighsIllegalInfeasibilityMeasure;
  info_.sum_primal_infeasibilities = kHighsIllegalInfeasibilityMeasure;
  info_.num_dual_infeasibilities = kHighsIllegalInfeasibilityCount;
  info_.max_dual_infeasibility = kHighsIllegalInfeasibilityMeasure;
  info_.sum_dual_infeasibilities = kHighsIllegalInfeasibilityMeasure;
  this->solution_.invalidate();
}

void Highs::invalidateBasis() {
  info_.basis_validity = kBasisValidityInvalid;
  this->basis_.invalidate();
}

void Highs::invalidateInfo() { info_.invalidate(); }

void Highs::invalidateRanging() { ranging_.invalidate(); }

void Highs::invalidateEkk() { ekk_instance_.invalidate(); }

// The method below runs calls solveLp for the given LP
HighsStatus Highs::callSolveLp(HighsLp& lp, const string message) {
  HighsStatus return_status = HighsStatus::kOk;
  HighsStatus call_status;

  // Create a HighsLpSolverObject of references to data in the Highs
  // class, and the scaled/unscaled model status
  HighsLpSolverObject solver_object(lp, basis_, solution_, info_, ekk_instance_,
                                    options_, timer_);

  // Check that the model is column-wise
  assert(model_.lp_.a_matrix_.isColwise());

  // Solve the LP
  return_status = solveLp(solver_object, message);
  // Extract the model status
  model_status_ = solver_object.model_status_;
  if (model_status_ == HighsModelStatus::kOptimal)
    checkOptimality("LP", return_status);
  return return_status;
}

HighsStatus Highs::callSolveQp() {
  // Check that the model is column-wise
  HighsLp& lp = model_.lp_;
  HighsHessian& hessian = model_.hessian_;
  assert(model_.lp_.a_matrix_.isColwise());
  if (hessian.dim_ != lp.num_col_) {
    highsLogDev(options_.log_options, HighsLogType::kError,
                "Hessian dimension = %" HIGHSINT_FORMAT
                " incompatible with matrix dimension = %" HIGHSINT_FORMAT "\n",
                hessian.dim_, lp.num_col_);
    model_status_ = HighsModelStatus::kModelError;
    solution_.value_valid = false;
    solution_.dual_valid = false;
    return HighsStatus::kError;
  }
  //
  // Run the QP solver
  Instance instance(lp.num_col_, lp.num_row_);

  instance.num_con = lp.num_row_;
  instance.num_var = lp.num_col_;

  instance.A.mat.num_col = lp.num_col_;
  instance.A.mat.num_row = lp.num_row_;
  instance.A.mat.start = lp.a_matrix_.start_;
  instance.A.mat.index = lp.a_matrix_.index_;
  instance.A.mat.value = lp.a_matrix_.value_;
  instance.c.value = lp.col_cost_;
  instance.offset = lp.offset_;
  instance.con_lo = lp.row_lower_;
  instance.con_up = lp.row_upper_;
  instance.var_lo = lp.col_lower_;
  instance.var_up = lp.col_upper_;
  instance.Q.mat.num_col = lp.num_col_;
  instance.Q.mat.num_row = lp.num_col_;
  triangularToSquareHessian(hessian, instance.Q.mat.start, instance.Q.mat.index,
                            instance.Q.mat.value);

  for (HighsInt i = 0; i < instance.c.value.size(); i++) {
    if (instance.c.value[i] != 0.0) {
      instance.c.index[instance.c.num_nz++] = i;
    }
  }

  if (lp.sense_ == ObjSense::kMaximize) {
    // Negate the vector and Hessian
    for (double& i : instance.c.value) {
      i *= -1.0;
    }
    for (double& i : instance.Q.mat.value) {
      i *= -1.0;
    }
  }

  Runtime runtime(instance, timer_);

  runtime.settings.reportingfequency = 1000;
  runtime.endofiterationevent.subscribe([this](Runtime& rt) {
    int rep = rt.statistics.iteration.size() - 1;

    highsLogUser(options_.log_options, HighsLogType::kInfo,
                 "%" HIGHSINT_FORMAT ", %lf, %" HIGHSINT_FORMAT
                 ", %lf, %lf, %" HIGHSINT_FORMAT ", %lf, %lf\n",
                 rt.statistics.iteration[rep], rt.statistics.objval[rep],
                 rt.statistics.nullspacedimension[rep], rt.statistics.time[rep],
                 rt.statistics.sum_primal_infeasibilities[rep],
                 rt.statistics.num_primal_infeasibilities[rep],
                 rt.statistics.density_nullspace[rep],
                 rt.statistics.density_factor[rep]);
  });

  runtime.settings.timelimit = options_.time_limit;
  runtime.settings.iterationlimit = std::numeric_limits<int>::max();
  Quass qpsolver(runtime);
  qpsolver.solve();

  HighsStatus call_status = HighsStatus::kOk;
  HighsStatus return_status = HighsStatus::kOk;
  return_status = interpretCallStatus(options_.log_options, call_status,
                                      return_status, "QpSolver");
  if (return_status == HighsStatus::kError) return return_status;
  model_status_ =
      runtime.status == ProblemStatus::OPTIMAL
          ? HighsModelStatus::kOptimal
          : runtime.status == ProblemStatus::UNBOUNDED
                ? HighsModelStatus::kUnbounded
                : runtime.status == ProblemStatus::INFEASIBLE
                      ? HighsModelStatus::kInfeasible
                      : runtime.status == ProblemStatus::ITERATIONLIMIT
                            ? HighsModelStatus::kIterationLimit
                            : runtime.status == ProblemStatus::TIMELIMIT
                                  ? HighsModelStatus::kTimeLimit
                                  : HighsModelStatus::kNotset;
  solution_.col_value.resize(lp.num_col_);
  solution_.col_dual.resize(lp.num_col_);
  const double objective_multiplier = lp.sense_ == ObjSense::kMinimize ? 1 : -1;
  for (HighsInt iCol = 0; iCol < lp.num_col_; iCol++) {
    solution_.col_value[iCol] = runtime.primal.value[iCol];
    solution_.col_dual[iCol] =
        objective_multiplier * runtime.dualvar.value[iCol];
  }
  solution_.row_value.resize(lp.num_row_);
  solution_.row_dual.resize(lp.num_row_);
  // Negate the vector and Hessian
  for (HighsInt iRow = 0; iRow < lp.num_row_; iRow++) {
    solution_.row_value[iRow] = runtime.rowactivity.value[iRow];
    solution_.row_dual[iRow] =
        objective_multiplier * runtime.dualcon.value[iRow];
  }
  solution_.value_valid = true;
  solution_.dual_valid = true;
  // Get the objective and any KKT failures
  info_.objective_function_value = model_.objectiveValue(solution_.col_value);
  getKktFailures(options_, model_, solution_, basis_, info_);
  // Set the QP-specific values of info_
  info_.simplex_iteration_count += runtime.statistics.phase1_iterations;
  info_.qp_iteration_count += runtime.statistics.num_iterations;
  info_.valid = true;
  if (model_status_ == HighsModelStatus::kOptimal)
    checkOptimality("QP", return_status);
  return return_status;
}

HighsStatus Highs::callSolveMip() {
  // Record whether there is a valid primal solution on entry
  const bool user_solution = solution_.value_valid;
  std::vector<double> user_solution_col_value;
  std::vector<double> user_solution_row_value;
  if (user_solution) {
    // Save the col and row values
    user_solution_col_value = std::move(solution_.col_value);
    user_solution_row_value = std::move(solution_.row_value);
  }
  // Ensure that any solver data for users in Highs class members are
  // cleared
  invalidateUserSolverData();
  if (user_solution) {
    // Recover the col and row values
    solution_.col_value = std::move(user_solution_col_value);
    solution_.row_value = std::move(user_solution_row_value);
    solution_.value_valid = true;
  }
  // Run the MIP solver
  HighsInt log_dev_level = options_.log_dev_level;
  //  options_.log_dev_level = kHighsLogDevLevelInfo;
  // Check that the model isn't row-wise
  assert(model_.lp_.a_matrix_.format_ != MatrixFormat::kRowwise);
  const bool has_semi_variables = model_.lp_.hasSemiVariables();
  HighsLp use_lp;
  if (has_semi_variables) {
    // Replace any semi-variables by a continuous/integer variable and
    // a (temporary) binary. Any initial solution must accommodate this.
    use_lp = withoutSemiVariables(model_.lp_, solution_,
                                  options_.primal_feasibility_tolerance);
  }
  HighsLp& lp = has_semi_variables ? use_lp : model_.lp_;
  HighsMipSolver solver(options_, lp, solution_);
  solver.run();
  options_.log_dev_level = log_dev_level;
  // Set the return_status, model status and, for completeness, scaled
  // model status
  HighsStatus return_status =
      highsStatusFromHighsModelStatus(solver.modelstatus_);
  model_status_ = solver.modelstatus_;
  // Extract the solution
  if (solver.solution_objective_ != kHighsInf) {
    // There is a primal solution
    HighsInt solver_solution_size = solver.solution_.size();
    assert(solver_solution_size >= lp.num_col_);
    // If the original model has semi-variables, its solution is
    // (still) given by the first model_.lp_.num_col_ entries of the
    // solution from the MIP solver
    solution_.col_value.resize(model_.lp_.num_col_);
    solution_.col_value = solver.solution_;
    model_.lp_.a_matrix_.productQuad(solution_.row_value, solution_.col_value);
    solution_.value_valid = true;
  } else {
    // There is no primal solution: should be so by default
    assert(!solution_.value_valid);
  }
  // Check that no modified upper bounds for semi-variables are active
  if (solution_.value_valid &&
      activeModifiedUpperBounds(options_, model_.lp_, solution_.col_value)) {
    solution_.value_valid = false;
    model_status_ = HighsModelStatus::kSolveError;
    return_status = HighsStatus::kError;
  }
  // There is no dual solution: should be so by default
  assert(!solution_.dual_valid);
  // There is no basis: should be so by default
  assert(!basis_.valid);
  // Get the objective and any KKT failures
  info_.objective_function_value = solver.solution_objective_;
  const bool use_mip_feasibility_tolerance = true;
  double primal_feasibility_tolerance = options_.primal_feasibility_tolerance;
  if (use_mip_feasibility_tolerance) {
    options_.primal_feasibility_tolerance = options_.mip_feasibility_tolerance;
  }
  // NB getKktFailures sets the primal and dual solution status
  getKktFailures(options_, model_, solution_, basis_, info_);
  // Set the MIP-specific values of info_
  info_.mip_node_count = solver.node_count_;
  info_.mip_dual_bound = solver.dual_bound_;
  info_.mip_gap = solver.gap_;
  info_.valid = true;
  if (model_status_ == HighsModelStatus::kOptimal)
    checkOptimality("MIP", return_status);
  if (use_mip_feasibility_tolerance) {
    // Overwrite max infeasibility to include integrality if there is a solution
    if (solver.solution_objective_ != kHighsInf) {
      const double mip_max_bound_violation =
          std::max(solver.row_violation_, solver.bound_violation_);
      const double mip_max_infeasibility =
          std::max(mip_max_bound_violation, solver.integrality_violation_);
      const double delta_max_bound_violation =
          std::abs(mip_max_bound_violation - info_.max_primal_infeasibility);
      // Possibly report a mis-match between the max bound violation
      // returned by the MIP solver, and the value obtained from the
      // solution
      if (delta_max_bound_violation > 1e-12)
        highsLogDev(options_.log_options, HighsLogType::kWarning,
                    "Inconsistent max bound violation: MIP solver (%10.4g); LP "
                    "(%10.4g); Difference of %10.4g\n",
                    mip_max_bound_violation, info_.max_primal_infeasibility,
                    delta_max_bound_violation);
      info_.max_integrality_violation = solver.integrality_violation_;
      if (info_.max_integrality_violation >
          options_.mip_feasibility_tolerance) {
        info_.primal_solution_status = kSolutionStatusInfeasible;
        assert(model_status_ == HighsModelStatus::kInfeasible);
      }
    }
    // Recover the primal feasibility tolerance
    options_.primal_feasibility_tolerance = primal_feasibility_tolerance;
  }
  return return_status;
}

HighsStatus Highs::callRunPostsolve(const HighsSolution& solution,
                                    const HighsBasis& basis) {
  HighsStatus return_status = HighsStatus::kOk;
  HighsStatus call_status;
  const HighsLp& presolved_lp = presolve_.getReducedProblem();

  const bool solution_ok = isSolutionRightSize(presolved_lp, solution);
  if (!solution_ok) {
    highsLogUser(options_.log_options, HighsLogType::kError,
                 "Solution provided to postsolve is incorrect size\n");
    return HighsStatus::kError;
  }
  const bool basis_ok = isBasisConsistent(presolved_lp, basis);
  if (!basis_ok) {
    highsLogUser(options_.log_options, HighsLogType::kError,
                 "Basis provided to postsolve is incorrect size\n");
    return HighsStatus::kError;
  }
  presolve_.data_.recovered_solution_ = solution;
  presolve_.data_.recovered_basis_ = basis;

  HighsPostsolveStatus postsolve_status = runPostsolve();
  if (postsolve_status == HighsPostsolveStatus::kSolutionRecovered) {
    highsLogDev(options_.log_options, HighsLogType::kVerbose,
                "Postsolve finished\n");
    // Set solution and its status
    solution_.clear();
    solution_ = presolve_.data_.recovered_solution_;
    solution_.value_valid = true;
    solution_.dual_valid = true;
    // Set basis and its status
    basis_.valid = true;
    basis_.col_status = presolve_.data_.recovered_basis_.col_status;
    basis_.row_status = presolve_.data_.recovered_basis_.row_status;
    basis_.debug_origin_name += ": after postsolve";
    // Save the options to allow the best simplex strategy to
    // be used
    HighsOptions save_options = options_;
    options_.simplex_strategy = kSimplexStrategyChoose;
    // Ensure that the parallel solver isn't used
    options_.simplex_min_concurrency = 1;
    options_.simplex_max_concurrency = 1;
    // Use any pivot threshold resulting from solving the presolved LP
    // if (factor_pivot_threshold > 0)
    //    options_.factor_pivot_threshold = factor_pivot_threshold;
    // The basis returned from postsolve is just basic/nonbasic
    // and EKK expects a refined basis, so set it up now
    HighsLp& incumbent_lp = model_.lp_;
    refineBasis(incumbent_lp, solution_, basis_);
    // Scrap the EKK data from solving the presolved LP
    ekk_instance_.invalidate();
    ekk_instance_.lp_name_ = "Postsolve LP";
    // Set up the timing record so that adding the corresponding
    // values after callSolveLp gives difference
    timer_.start(timer_.solve_clock);
    call_status = callSolveLp(
        incumbent_lp,
        "Solving the original LP from the solution after postsolve");
    // Determine the timing record
    timer_.stop(timer_.solve_clock);
    return_status = interpretCallStatus(options_.log_options, call_status,
                                        return_status, "callSolveLp");
    // Recover the options
    options_ = save_options;
    if (return_status == HighsStatus::kError)
      return returnFromRun(return_status);
  } else {
    highsLogUser(options_.log_options, HighsLogType::kError,
                 "Postsolve return status is %d\n", (int)postsolve_status);
    setHighsModelStatusAndClearSolutionAndBasis(
        HighsModelStatus::kPostsolveError);
    return returnFromRun(HighsStatus::kError);
  }
  call_status = highsStatusFromHighsModelStatus(model_status_);
  return_status =
      interpretCallStatus(options_.log_options, call_status, return_status,
                          "highsStatusFromHighsModelStatus");
  return return_status;
}

// End of public methods
void Highs::logHeader() {
  if (written_log_header) return;
  highsLogHeader(options_.log_options);
  written_log_header = true;
  return;
}

void Highs::reportModel() {
  reportLp(options_.log_options, model_.lp_, HighsLogType::kVerbose);
  if (model_.hessian_.dim_) {
    const HighsInt dim = model_.hessian_.dim_;
    reportHessian(options_.log_options, dim, model_.hessian_.start_[dim],
                  &model_.hessian_.start_[0], &model_.hessian_.index_[0],
                  &model_.hessian_.value_[0]);
  }
}

// Actions to take if there is a new Highs basis
void Highs::newHighsBasis() {
  // Clear any simplex basis
  ekk_instance_.updateStatus(LpAction::kNewBasis);
}

// Ensure that the HiGHS solution and basis have the same size as the
// model, and that the HiGHS basis is kept up-to-date with any solved
// basis
void Highs::forceHighsSolutionBasisSize() {
  // Ensure that the HiGHS solution vectors are the right size
  solution_.col_value.resize(model_.lp_.num_col_);
  solution_.row_value.resize(model_.lp_.num_row_);
  solution_.col_dual.resize(model_.lp_.num_col_);
  solution_.row_dual.resize(model_.lp_.num_row_);
  // Ensure that the HiGHS basis vectors are the right size,
  // invalidating the basis if they aren't
  if ((HighsInt)basis_.col_status.size() != model_.lp_.num_col_) {
    basis_.col_status.resize(model_.lp_.num_col_);
    basis_.valid = false;
  }
  if ((HighsInt)basis_.row_status.size() != model_.lp_.num_row_) {
    basis_.row_status.resize(model_.lp_.num_row_);
    basis_.valid = false;
  }
}

void Highs::setHighsModelStatusAndClearSolutionAndBasis(
    const HighsModelStatus model_status) {
  model_status_ = model_status;
  invalidateSolution();
  invalidateBasis();
  info_.valid = true;
}

void Highs::setBasisValidity() {
  if (basis_.valid) {
    assert(info_.basis_validity == kBasisValidityValid);
    info_.basis_validity = kBasisValidityValid;
  } else {
    assert(info_.basis_validity == kBasisValidityInvalid);
    info_.basis_validity = kBasisValidityInvalid;
  }
  info_.valid = true;
}

HighsStatus Highs::openWriteFile(const string filename,
                                 const string method_name, FILE*& file,
                                 bool& html) const {
  html = false;
  if (filename == "") {
    // Empty file name: use stdout
    file = stdout;
  } else {
    file = fopen(filename.c_str(), "w");
    if (file == 0) {
      highsLogUser(options_.log_options, HighsLogType::kError,
                   "Cannot open writeable file \"%s\" in %s\n",
                   filename.c_str(), method_name.c_str());
      return HighsStatus::kError;
    }
    const char* dot = strrchr(filename.c_str(), '.');
    if (dot && dot != filename) html = strcmp(dot + 1, "html") == 0;
  }
  return HighsStatus::kOk;
}

// Applies checks before returning from run()
HighsStatus Highs::returnFromRun(const HighsStatus run_return_status) {
  assert(!called_return_from_run);
  HighsStatus return_status = highsStatusFromHighsModelStatus(model_status_);
  assert(return_status == run_return_status);
  //  return_status = run_return_status;
  switch (model_status_) {
      // First consider the error returns
    case HighsModelStatus::kNotset:
    case HighsModelStatus::kLoadError:
    case HighsModelStatus::kModelError:
    case HighsModelStatus::kPresolveError:
    case HighsModelStatus::kSolveError:
    case HighsModelStatus::kPostsolveError:
      // Don't clear the model status!
      //      invalidateUserSolverData();
      invalidateInfo();
      invalidateSolution();
      invalidateBasis();
      assert(return_status == HighsStatus::kError);
      break;

      // Then consider the OK returns
    case HighsModelStatus::kModelEmpty:
      invalidateInfo();
      invalidateSolution();
      invalidateBasis();
      assert(return_status == HighsStatus::kOk);
      break;

    case HighsModelStatus::kOptimal:
      // The following is an aspiration
      //
      // assert(info_.primal_solution_status == kSolutionStatusFeasible);
      //
      // assert(info_.dual_solution_status == kSolutionStatusFeasible);
      assert(model_status_ == HighsModelStatus::kNotset ||
             model_status_ == HighsModelStatus::kOptimal);
      assert(return_status == HighsStatus::kOk);
      break;

    case HighsModelStatus::kInfeasible:
    case HighsModelStatus::kUnbounded:
    case HighsModelStatus::kObjectiveBound:
    case HighsModelStatus::kObjectiveTarget:
      // For kInfeasible, will not have a basis, if infeasibility was
      // detected in presolve or by IPX without crossover
      assert(return_status == HighsStatus::kOk);
      break;

    case HighsModelStatus::kUnboundedOrInfeasible:
      if (options_.allow_unbounded_or_infeasible ||
          (options_.solver == kIpmString && options_.run_crossover) ||
          model_.isMip()) {
        assert(return_status == HighsStatus::kOk);
      } else {
        // This model status is not permitted unless IPM is run without
        // crossover
        highsLogUser(
            options_.log_options, HighsLogType::kError,
            "returnFromHighs: HighsModelStatus::kUnboundedOrInfeasible is not "
            "permitted\n");
        assert(options_.allow_unbounded_or_infeasible);
        return_status = HighsStatus::kError;
      }
      break;

      // Finally consider the warning returns
    case HighsModelStatus::kTimeLimit:
    case HighsModelStatus::kIterationLimit:
    case HighsModelStatus::kUnknown:
      assert(return_status == HighsStatus::kWarning);
      break;
    default:
      // All cases should have been considered so assert on reaching here
      assert(1 == 0);
  }
  // Now to check what's available with each model status
  //
  const bool have_info = info_.valid;
  const bool have_primal_solution = solution_.value_valid;
  const bool have_dual_solution = solution_.dual_valid;
  // Can't have a dual solution without a primal solution
  assert(have_primal_solution || !have_dual_solution);
  //  const bool have_solution = have_primal_solution && have_dual_solution;
  const bool have_basis = basis_.valid;
  switch (model_status_) {
    case HighsModelStatus::kNotset:
    case HighsModelStatus::kLoadError:
    case HighsModelStatus::kModelError:
    case HighsModelStatus::kPresolveError:
    case HighsModelStatus::kSolveError:
    case HighsModelStatus::kPostsolveError:
    case HighsModelStatus::kModelEmpty:
      // No info, primal solution or basis
      assert(have_info == false);
      assert(have_primal_solution == false);
      assert(have_basis == false);
      break;
    case HighsModelStatus::kOptimal:
    case HighsModelStatus::kInfeasible:
    case HighsModelStatus::kUnbounded:
    case HighsModelStatus::kObjectiveBound:
    case HighsModelStatus::kObjectiveTarget:
    case HighsModelStatus::kUnboundedOrInfeasible:
    case HighsModelStatus::kTimeLimit:
    case HighsModelStatus::kIterationLimit:
    case HighsModelStatus::kUnknown:
      // Have info and primal solution (unless infeasible). No primal solution
      // in some other case, too!
      assert(have_info == true);
      break;
    default:
      // All cases should have been considered so assert on reaching here
      assert(1 == 0);
  }
  if (have_primal_solution) {
    if (debugPrimalSolutionRightSize(options_, model_.lp_, solution_) ==
        HighsDebugStatus::kLogicalError)
      return_status = HighsStatus::kError;
  }
  if (have_dual_solution) {
    if (debugDualSolutionRightSize(options_, model_.lp_, solution_) ==
        HighsDebugStatus::kLogicalError)
      return_status = HighsStatus::kError;
  }
  if (have_basis) {
    if (debugBasisRightSize(options_, model_.lp_, basis_) ==
        HighsDebugStatus::kLogicalError)
      return_status = HighsStatus::kError;
  }
  if (have_primal_solution) {
    // Debug the Highs solution - needs primal values at least
    if (debugHighsSolution("Return from run()", options_, model_, solution_,
                           basis_, model_status_,
                           info_) == HighsDebugStatus::kLogicalError)
      return_status = HighsStatus::kError;
  }
  if (debugInfo(options_, model_.lp_, basis_, solution_, info_,
                model_status_) == HighsDebugStatus::kLogicalError)
    return_status = HighsStatus::kError;

  // Record that returnFromRun() has been called
  called_return_from_run = true;
  // Unapply any modifications that have not yet been unapplied
  this->model_.lp_.unapplyMods();

  // Unless solved as a MIP, report on the solution
  const bool solved_as_mip =
      !options_.solver.compare(kHighsChooseString) && model_.isMip();
  if (!solved_as_mip) reportSolvedLpQpStats();

  return returnFromHighs(return_status);
}

HighsStatus Highs::returnFromHighs(HighsStatus highs_return_status) {
  // Applies checks before returning from HiGHS
  HighsStatus return_status = highs_return_status;

  forceHighsSolutionBasisSize();

  const bool consistent =
      debugHighsBasisConsistent(options_, model_.lp_, basis_) !=
      HighsDebugStatus::kLogicalError;
  if (!consistent) {
    highsLogUser(
        options_.log_options, HighsLogType::kError,
        "returnFromHighs: Supposed to be a HiGHS basis, but not consistent\n");
    assert(consistent);
    return_status = HighsStatus::kError;
  }
  // Check that any retained Ekk data - basis and NLA - are OK
  bool retained_ekk_data_ok = ekk_instance_.debugRetainedDataOk(model_.lp_) !=
                              HighsDebugStatus::kLogicalError;
  if (!retained_ekk_data_ok) {
    highsLogUser(options_.log_options, HighsLogType::kError,
                 "returnFromHighs: Retained Ekk data not OK\n");
    assert(retained_ekk_data_ok);
    return_status = HighsStatus::kError;
  }
  // Check that returnFromRun() has been called
  if (!called_return_from_run) {
    highsLogDev(
        options_.log_options, HighsLogType::kError,
        "Highs::returnFromHighs() called with called_return_from_run false\n");
    assert(called_return_from_run);
  }
  // Stop the HiGHS run clock if it is running
  if (timer_.runningRunHighsClock()) timer_.stopRunHighsClock();
  const bool dimensions_ok =
      lpDimensionsOk("returnFromHighs", model_.lp_, options_.log_options);
  if (!dimensions_ok) {
    printf("LP Dimension error in returnFromHighs()\n");
  }
  assert(dimensions_ok);
  if (ekk_instance_.status_.has_nla) {
    if (!ekk_instance_.lpFactorRowCompatible(model_.lp_.num_row_)) {
      highsLogDev(options_.log_options, HighsLogType::kWarning,
                  "Highs::returnFromHighs(): LP and HFactor have inconsistent "
                  "numbers of rows\n");
      // Clear Ekk entirely
      ekk_instance_.clear();
    }
  }
  return return_status;
}

void Highs::reportSolvedLpQpStats() {
  HighsLogOptions& log_options = options_.log_options;
  highsLogUser(log_options, HighsLogType::kInfo, "Model   status      : %s\n",
               modelStatusToString(model_status_).c_str());
  if (info_.valid) {
    if (info_.simplex_iteration_count)
      highsLogUser(log_options, HighsLogType::kInfo,
                   "Simplex   iterations: %" HIGHSINT_FORMAT "\n",
                   info_.simplex_iteration_count);
    if (info_.ipm_iteration_count)
      highsLogUser(log_options, HighsLogType::kInfo,
                   "IPM       iterations: %" HIGHSINT_FORMAT "\n",
                   info_.ipm_iteration_count);
    if (info_.crossover_iteration_count)
      highsLogUser(log_options, HighsLogType::kInfo,
                   "Crossover iterations: %" HIGHSINT_FORMAT "\n",
                   info_.crossover_iteration_count);
    if (info_.qp_iteration_count)
      highsLogUser(log_options, HighsLogType::kInfo,
                   "QP ASM    iterations: %" HIGHSINT_FORMAT "\n",
                   info_.qp_iteration_count);
    highsLogUser(log_options, HighsLogType::kInfo,
                 "Objective value     : %17.10e\n",
                 info_.objective_function_value);
  }
  double run_time = timer_.readRunHighsClock();
  highsLogUser(log_options, HighsLogType::kInfo,
               "HiGHS run time      : %13.2f\n", run_time);
}

void Highs::underDevelopmentLogMessage(const std::string& method_name) {
  highsLogUser(options_.log_options, HighsLogType::kWarning,
               "Method %s is still under development and behaviour may be "
               "unpredictable\n",
               method_name.c_str());
}

HighsStatus Highs::crossover(const HighsSolution& user_solution) {
  HighsStatus return_status = HighsStatus::kOk;
  HighsLogOptions& log_options = options_.log_options;
  HighsLp& lp = model_.lp_;
  if (lp.isMip()) {
    highsLogUser(log_options, HighsLogType::kError,
                 "Cannot apply crossover to solve MIP\n");
    return_status = HighsStatus::kError;
  } else if (model_.isQp()) {
    highsLogUser(log_options, HighsLogType::kError,
                 "Cannot apply crossover to solve QP\n");
    return_status = HighsStatus::kError;
  } else {
    clearSolver();
    solution_ = user_solution;
    // Use IPX crossover to try to form a basic solution
    return_status = callCrossover(options_, model_.lp_, basis_, solution_,
                                  model_status_, info_);
    if (return_status == HighsStatus::kError) return return_status;
    // Get the objective and any KKT failures
    info_.objective_function_value =
        model_.lp_.objectiveValue(solution_.col_value);
    getLpKktFailures(options_, model_.lp_, solution_, basis_, info_);
  }
  return returnFromHighs(return_status);
}

HighsStatus Highs::openLogFile(const std::string& log_file) {
  highsOpenLogFile(options_.log_options, options_.records, log_file);
  return HighsStatus::kOk;
}

void Highs::resetGlobalScheduler(bool blocking) {
  HighsTaskExecutor::shutdown(blocking);
}
