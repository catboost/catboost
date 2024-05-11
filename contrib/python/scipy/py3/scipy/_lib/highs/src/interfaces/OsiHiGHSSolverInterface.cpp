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
/**@file interfaces/OsiHiGHSInterface.cpp
 * @brief Osi/HiGHS interface implementation
 */
#include "OsiHiGHSSolverInterface.hpp"

#include <cmath>

#include "CoinWarmStartBasis.hpp"
#include "Highs.h"
#include "HighsLp.h"
#include "HighsOptions.h"
#include "HighsStatus.h"
#include "io/FilereaderMps.h"
#include "io/HighsIO.h"
#include "lp_data/HConst.h"

static void logtomessagehandler(HighsLogType type, const char* msg,
                                void* log_callback_data) {
  assert(log_callback_data != NULL);

  CoinMessageHandler* handler = (CoinMessageHandler*)log_callback_data;

  // we know log message end with a newline, replace by coin-eol
  HighsInt len = strlen(msg);
  assert(len > 0);
  assert(msg[len - 1] == '\n');
  const_cast<char*>(msg)[len - 1] = '\0';

  handler->message(0, "HiGHS", msg, ' ') << CoinMessageEol;

  const_cast<char*>(msg)[len - 1] = '\n';
}

OsiHiGHSSolverInterface::OsiHiGHSSolverInterface()
    //  : status(HighsStatus::Init) {
    : status(HighsStatus::kOk) {
  this->highs = new Highs();

  this->highs->setLogCallback(logtomessagehandler, (void*)handler_);

  HighsOptions& options = this->highs->options_;
  highsLogDev(options.log_options, HighsLogType::kInfo,
              "Calling OsiHiGHSSolverInterface::OsiHiGHSSolverInterface()\n");
  this->dummy_solution = new HighsSolution;

  setStrParam(OsiSolverName, "HiGHS");
}

OsiHiGHSSolverInterface::OsiHiGHSSolverInterface(
    const OsiHiGHSSolverInterface& original)
    : OsiSolverInterface(original),
      //      status(HighsStatus::Init)
      status(HighsStatus::kOk) {
  this->highs = new Highs();

  this->highs->setLogCallback(logtomessagehandler, (void*)handler_);

  HighsOptions& options = this->highs->options_;
  highsLogDev(options.log_options, HighsLogType::kInfo,
              "Calling OsiHiGHSSolverInterface::OsiHiGHSSolverInterface()\n");
  this->dummy_solution = new HighsSolution;

  this->highs->passModel(original.highs->getLp());
  setStrParam(OsiSolverName, "HiGHS");
}

OsiHiGHSSolverInterface::~OsiHiGHSSolverInterface() {
  HighsOptions& options = this->highs->options_;
  highsLogDev(options.log_options, HighsLogType::kInfo,
              "Calling OsiHiGHSSolverInterface::~OsiHiGHSSolverInterface()\n");

  this->highs->setLogCallback(NULL, NULL);

  delete this->highs;

  if (this->rowRange != NULL) {
    delete[] this->rowRange;
  }

  if (this->rhs != NULL) {
    delete[] this->rhs;
  }

  if (this->rowSense != NULL) {
    delete[] this->rowSense;
  }

  if (this->matrixByCol != NULL) {
    delete this->matrixByCol;
  }
}

OsiSolverInterface* OsiHiGHSSolverInterface::clone(bool copyData) const {
  HighsOptions& options = this->highs->options_;
  highsLogDev(options.log_options, HighsLogType::kInfo,
              "Calling OsiHiGHSSolverInterface::clone()\n");
  if (!copyData) {
    OsiHiGHSSolverInterface* cln = new OsiHiGHSSolverInterface();
    return cln;

  } else {
    OsiHiGHSSolverInterface* cln = new OsiHiGHSSolverInterface(*this);
    cln->objOffset = this->objOffset;
    return cln;
  }
}

bool OsiHiGHSSolverInterface::setIntParam(OsiIntParam key, HighsInt value) {
  HighsOptions& options = this->highs->options_;
  highsLogDev(options.log_options, HighsLogType::kInfo,
              "Calling OsiHiGHSSolverInterface::setIntParam()\n");
  switch (key) {
    case OsiMaxNumIteration:
    case OsiMaxNumIterationHotStart:
      this->highs->options_.simplex_iteration_limit = value;
      return true;
    case OsiNameDiscipline:
      // TODO
      return false;
    case OsiLastIntParam:
    default:
      return false;
  }
}

bool OsiHiGHSSolverInterface::setDblParam(OsiDblParam key, double value) {
  HighsOptions& options = this->highs->options_;
  highsLogDev(options.log_options, HighsLogType::kInfo,
              "Calling OsiHiGHSSolverInterface::setDblParam()\n");
  switch (key) {
    case OsiDualObjectiveLimit:
      this->highs->options_.objective_bound = value;
      return true;
    case OsiPrimalObjectiveLimit:
      return false;
    case OsiDualTolerance:
      this->highs->options_.dual_feasibility_tolerance = value;
      return true;
    case OsiPrimalTolerance:
      this->highs->options_.primal_feasibility_tolerance = value;
      return true;
    case OsiObjOffset:
      this->objOffset = value;
      return true;
    case OsiLastDblParam:
    default:
      return false;
  }
}

bool OsiHiGHSSolverInterface::setStrParam(OsiStrParam key,
                                          const std::string& value) {
  HighsOptions& options = this->highs->options_;
  highsLogDev(options.log_options, HighsLogType::kInfo,
              "Calling OsiHiGHSSolverInterface::setStrParam(%" HIGHSINT_FORMAT
              ", %s)\n",
              key, value.c_str());
  switch (key) {
    case OsiProbName:
      return OsiSolverInterface::setStrParam(key, value);
    case OsiSolverName:
      return OsiSolverInterface::setStrParam(key, value);
    case OsiLastStrParam:
    default:
      return false;
  }
}

bool OsiHiGHSSolverInterface::getIntParam(OsiIntParam key,
                                          HighsInt& value) const {
  HighsOptions& options = this->highs->options_;
  highsLogDev(options.log_options, HighsLogType::kInfo,
              "Calling OsiHiGHSSolverInterface::getIntParam()\n");
  switch (key) {
    case OsiMaxNumIteration:
    case OsiMaxNumIterationHotStart:
      value = this->highs->options_.simplex_iteration_limit;
      return true;
    case OsiNameDiscipline:
      // TODO
      return false;
    case OsiLastIntParam:
    default:
      return false;
  }
}

bool OsiHiGHSSolverInterface::getDblParam(OsiDblParam key,
                                          double& value) const {
  HighsOptions& options = this->highs->options_;
  highsLogDev(options.log_options, HighsLogType::kInfo,
              "Calling OsiHiGHSSolverInterface::getDblParam()\n");
  switch (key) {
    case OsiDualObjectiveLimit:
      value = this->highs->options_.objective_bound;
      return true;
    case OsiPrimalObjectiveLimit:
      return false;
    case OsiDualTolerance:
      value = this->highs->options_.dual_feasibility_tolerance;
      return true;
    case OsiPrimalTolerance:
      value = this->highs->options_.primal_feasibility_tolerance;
      return true;
    case OsiObjOffset:
      value = this->objOffset;
      return true;
    case OsiLastDblParam:
    default:
      return false;
  }
}

bool OsiHiGHSSolverInterface::getStrParam(OsiStrParam key,
                                          std::string& value) const {
  HighsOptions& options = this->highs->options_;
  highsLogDev(options.log_options, HighsLogType::kInfo,
              "Calling OsiHiGHSSolverInterface::getStrParam(%" HIGHSINT_FORMAT
              ", %s)\n",
              key, value.c_str());
  switch (key) {
    case OsiProbName:
      return OsiSolverInterface::getStrParam(key, value);
    case OsiSolverName:
      return OsiSolverInterface::getStrParam(key, value);
    case OsiLastStrParam:
    default:
      return false;
  }
}

void OsiHiGHSSolverInterface::initialSolve() {
  HighsOptions& options = this->highs->options_;
  highsLogDev(options.log_options, HighsLogType::kInfo,
              "Calling OsiHiGHSSolverInterface::initialSolve()\n");
  this->status = this->highs->run();
}

bool OsiHiGHSSolverInterface::isAbandoned() const {
  HighsOptions& options = this->highs->options_;
  highsLogDev(options.log_options, HighsLogType::kInfo,
              "Calling OsiHiGHSSolverInterface::isAbandoned()\n");
  //  return this->status == HighsStatus::NumericalDifficulties;
  return false;
}

bool OsiHiGHSSolverInterface::isProvenOptimal() const {
  HighsOptions& options = this->highs->options_;
  highsLogDev(options.log_options, HighsLogType::kInfo,
              "Calling OsiHiGHSSolverInterface::isProvenOptimal()\n");
  //  return (this->status == HighsStatus::kOptimal) ||
  //         (this->status == HighsStatus::kOk);
  return false;
}

bool OsiHiGHSSolverInterface::isProvenPrimalInfeasible() const {
  HighsOptions& options = this->highs->options_;
  highsLogDev(options.log_options, HighsLogType::kInfo,
              "Calling OsiHiGHSSolverInterface::isProvenPrimalInfeasible()\n");
  //  return this->status == HighsStatus::kInfeasible;
  return false;
}

bool OsiHiGHSSolverInterface::isProvenDualInfeasible() const {
  HighsOptions& options = this->highs->options_;
  highsLogDev(options.log_options, HighsLogType::kInfo,
              "Calling OsiHiGHSSolverInterface::isProvenDualInfeasible()\n");
  //  return this->status == HighsStatus::Unbounded;
  return false;
}

bool OsiHiGHSSolverInterface::isPrimalObjectiveLimitReached() const {
  HighsOptions& options = this->highs->options_;
  highsLogDev(
      options.log_options, HighsLogType::kInfo,
      "Calling OsiHiGHSSolverInterface::isPrimalObjectiveLimitReached()\n");
  return false;
}

bool OsiHiGHSSolverInterface::isDualObjectiveLimitReached() const {
  HighsOptions& options = this->highs->options_;
  highsLogDev(
      options.log_options, HighsLogType::kInfo,
      "Calling OsiHiGHSSolverInterface::isDualObjectiveLimitReached()\n");
  //  return this->status == HighsStatus::ReachedDualObjectiveUpperBound;
  return false;
}

bool OsiHiGHSSolverInterface::isIterationLimitReached() const {
  HighsOptions& options = this->highs->options_;
  highsLogDev(options.log_options, HighsLogType::kInfo,
              "Calling OsiHiGHSSolverInterface::isIterationLimitReached()\n");
  //  return this->status == HighsStatus::ReachedIterationLimit;
  return false;
}

HighsInt OsiHiGHSSolverInterface::getNumCols() const {
  HighsOptions& options = this->highs->options_;
  highsLogDev(options.log_options, HighsLogType::kInfo,
              "Calling OsiHiGHSSolverInterface::getNumCols()\n");
  return this->highs->getNumCol();
}

HighsInt OsiHiGHSSolverInterface::getNumRows() const {
  HighsOptions& options = this->highs->options_;
  highsLogDev(options.log_options, HighsLogType::kInfo,
              "Calling OsiHiGHSSolverInterface::getNumRows()\n");
  return this->highs->getNumRow();
}

HighsInt OsiHiGHSSolverInterface::getNumElements() const {
  HighsOptions& options = this->highs->options_;
  highsLogDev(options.log_options, HighsLogType::kInfo,
              "Calling OsiHiGHSSolverInterface::getNumElements()\n");
  return this->highs->getNumNz();
}

const double* OsiHiGHSSolverInterface::getColLower() const {
  HighsOptions& options = this->highs->options_;
  highsLogDev(options.log_options, HighsLogType::kInfo,
              "Calling OsiHiGHSSolverInterface::getColLower()\n");
  return &(this->highs->getLp().col_lower_[0]);
}

const double* OsiHiGHSSolverInterface::getColUpper() const {
  HighsOptions& options = this->highs->options_;
  highsLogDev(options.log_options, HighsLogType::kInfo,
              "Calling OsiHiGHSSolverInterface::getColUpper()\n");
  return &(this->highs->getLp().col_upper_[0]);
}

const double* OsiHiGHSSolverInterface::getRowLower() const {
  HighsOptions& options = this->highs->options_;
  highsLogDev(options.log_options, HighsLogType::kInfo,
              "Calling OsiHiGHSSolverInterface::getRowLower()\n");
  return &(this->highs->getLp().row_lower_[0]);
}

const double* OsiHiGHSSolverInterface::getRowUpper() const {
  HighsOptions& options = this->highs->options_;
  highsLogDev(options.log_options, HighsLogType::kInfo,
              "Calling OsiHiGHSSolverInterface::getRowUpper()\n");
  return &(this->highs->getLp().row_upper_[0]);
}

const double* OsiHiGHSSolverInterface::getObjCoefficients() const {
  HighsOptions& options = this->highs->options_;
  highsLogDev(options.log_options, HighsLogType::kInfo,
              "Calling OsiHiGHSSolverInterface::getObjCoefficients()\n");
  return &(this->highs->getLp().col_cost_[0]);
}

// TODO: review: 10^20?
double OsiHiGHSSolverInterface::getInfinity() const {
  HighsOptions& options = this->highs->options_;
  highsLogDev(options.log_options, HighsLogType::kInfo,
              "Calling OsiHiGHSSolverInterface::getInfinity()\n");
  return kHighsInf;
}

const double* OsiHiGHSSolverInterface::getRowRange() const {
  HighsOptions& options = this->highs->options_;
  highsLogDev(options.log_options, HighsLogType::kInfo,
              "Calling OsiHiGHSSolverInterface::getRowRange()\n");
  if (this->rowRange != NULL) {
    delete[] this->rowRange;
  }

  HighsInt nrows = this->getNumRows();

  if (nrows == 0) {
    return this->rowRange;
  }

  this->rowRange = new double[nrows];

  for (HighsInt i = 0; i < nrows; i++) {
    // compute range for row i
    double lo = this->highs->getLp().row_lower_[i];
    double hi = this->highs->getLp().row_upper_[i];
    double t1;
    char t2;
    this->convertBoundToSense(lo, hi, t2, t1, this->rowRange[i]);
  }

  return this->rowRange;
}

const double* OsiHiGHSSolverInterface::getRightHandSide() const {
  HighsOptions& options = this->highs->options_;
  highsLogDev(options.log_options, HighsLogType::kInfo,
              "Calling OsiHiGHSSolverInterface::getRightHandSide()\n");
  if (this->rhs != NULL) {
    delete[] this->rhs;
  }

  HighsInt nrows = this->getNumRows();

  if (nrows == 0) {
    return this->rhs;
  }

  this->rhs = new double[nrows];

  for (HighsInt i = 0; i < nrows; i++) {
    // compute rhs for row i
    double lo = this->highs->getLp().row_lower_[i];
    double hi = this->highs->getLp().row_upper_[i];
    double t1;
    char t2;
    this->convertBoundToSense(lo, hi, t2, this->rhs[i], t1);
  }

  return this->rhs;
}

const char* OsiHiGHSSolverInterface::getRowSense() const {
  HighsOptions& options = this->highs->options_;
  highsLogDev(options.log_options, HighsLogType::kInfo,
              "Calling OsiHiGHSSolverInterface::getRowSense()\n");
  if (this->rowSense != NULL) {
    delete[] this->rowSense;
  }

  HighsInt nrows = this->getNumRows();

  if (nrows == 0) {
    return this->rowSense;
  }

  this->rowSense = new char[nrows];

  for (HighsInt i = 0; i < nrows; i++) {
    // compute sense for row i
    double lo = this->highs->getLp().row_lower_[i];
    double hi = this->highs->getLp().row_upper_[i];
    double t1, t2;
    this->convertBoundToSense(lo, hi, this->rowSense[i], t1, t2);
  }

  return this->rowSense;
}

const CoinPackedMatrix* OsiHiGHSSolverInterface::getMatrixByCol() const {
  HighsOptions& options = this->highs->options_;
  highsLogDev(options.log_options, HighsLogType::kInfo,
              "Calling OsiHiGHSSolverInterface::getMatrixByCol()\n");
  if (this->matrixByCol != NULL) {
    delete this->matrixByCol;
  }

  HighsInt nrows = this->getNumRows();
  HighsInt ncols = this->getNumCols();
  HighsInt nelements = this->getNumElements();

  HighsInt* len = new int[ncols];
  HighsInt* start = new int[ncols + 1];
  HighsInt* index = new int[nelements];
  double* value = new double[nelements];

  // copy data
  memcpy(start, &(this->highs->getLp().a_matrix_.start_[0]),
         (ncols + 1) * sizeof(HighsInt));
  memcpy(index, &(this->highs->getLp().a_matrix_.index_[0]),
         nelements * sizeof(HighsInt));
  memcpy(value, &(this->highs->getLp().a_matrix_.value_[0]),
         nelements * sizeof(double));

  for (HighsInt i = 0; i < ncols; i++) {
    len[i] = start[i + 1] - start[i];
  }

  this->matrixByCol = new CoinPackedMatrix();

  this->matrixByCol->assignMatrix(true, nrows, ncols, nelements, value, index,
                                  start, len);
  assert(this->matrixByCol->getNumCols() == ncols);
  assert(this->matrixByCol->getNumRows() == nrows);

  return this->matrixByCol;
}

const CoinPackedMatrix* OsiHiGHSSolverInterface::getMatrixByRow() const {
  HighsOptions& options = this->highs->options_;
  highsLogDev(options.log_options, HighsLogType::kInfo,
              "Calling OsiHiGHSSolverInterface::getMatrixByRow()\n");
  if (this->matrixByRow != NULL) {
    delete this->matrixByRow;
  }
  this->matrixByRow = new CoinPackedMatrix();
  this->matrixByRow->reverseOrderedCopyOf(*this->getMatrixByCol());

  return this->matrixByRow;
}

double OsiHiGHSSolverInterface::getObjSense() const {
  HighsOptions& options = this->highs->options_;
  highsLogDev(options.log_options, HighsLogType::kInfo,
              "Calling OsiHiGHSSolverInterface::getObjSense()\n");
  ObjSense sense;
  this->highs->getObjectiveSense(sense);
  return (double)sense;
}

void OsiHiGHSSolverInterface::setObjSense(double s) {
  HighsOptions& options = this->highs->options_;
  highsLogDev(options.log_options, HighsLogType::kInfo,
              "Calling OsiHiGHSSolverInterface::setObjSense()\n");
  ObjSense pass_sense = ObjSense::kMinimize;
  if (s == (double)ObjSense::kMaximize) pass_sense = ObjSense::kMaximize;
  this->highs->changeObjectiveSense(pass_sense);
}

void OsiHiGHSSolverInterface::addRow(const CoinPackedVectorBase& vec,
                                     const double rowlb, const double rowub) {
  HighsOptions& options = this->highs->options_;
  highsLogDev(options.log_options, HighsLogType::kInfo,
              "Calling OsiHiGHSSolverInterface::addRow()\n");
  HighsStatus status = this->highs->addRow(rowlb, rowub, vec.getNumElements(),
                                           vec.getIndices(), vec.getElements());
  assert(status == HighsStatus::kOk);
  if (status != HighsStatus::kOk) {
    highsLogDev(options.log_options, HighsLogType::kInfo,
                "Return from OsiHiGHSSolverInterface::addRow() is not ok\n");
  }
}

void OsiHiGHSSolverInterface::addRow(const CoinPackedVectorBase& vec,
                                     const char rowsen, const double rowrhs,
                                     const double rowrng) {
  HighsOptions& options = this->highs->options_;
  highsLogDev(options.log_options, HighsLogType::kInfo,
              "Calling OsiHiGHSSolverInterface::addRow()\n");
  // Assign arbitrary values so that compilation is clean
  double lb = 0;
  double ub = 1e200;
  this->convertSenseToBound(rowsen, rowrhs, rowrng, lb, ub);
  this->addRow(vec, lb, ub);
}

void OsiHiGHSSolverInterface::addCol(const CoinPackedVectorBase& vec,
                                     const double collb, const double colub,
                                     const double obj) {
  HighsOptions& options = this->highs->options_;
  highsLogDev(options.log_options, HighsLogType::kInfo,
              "Calling OsiHiGHSSolverInterface::addCol()\n");
  HighsStatus status =
      this->highs->addCol(obj, collb, colub, vec.getNumElements(),
                          vec.getIndices(), vec.getElements());
  assert(status == HighsStatus::kOk);
  if (status != HighsStatus::kOk) {
    highsLogDev(options.log_options, HighsLogType::kInfo,
                "Return from OsiHiGHSSolverInterface::addCol() is not ok\n");
  }
}

void OsiHiGHSSolverInterface::deleteCols(const HighsInt num,
                                         const HighsInt* colIndices) {
  HighsOptions& options = this->highs->options_;
  highsLogDev(options.log_options, HighsLogType::kInfo,
              "Calling OsiHiGHSSolverInterface::deleteCols()\n");
  this->highs->deleteCols(num, colIndices);
}

void OsiHiGHSSolverInterface::deleteRows(const HighsInt num,
                                         const HighsInt* rowIndices) {
  HighsOptions& options = this->highs->options_;
  highsLogDev(options.log_options, HighsLogType::kInfo,
              "Calling OsiHiGHSSolverInterface::deleteRows()\n");
  this->highs->deleteRows(num, rowIndices);
}

void OsiHiGHSSolverInterface::assignProblem(CoinPackedMatrix*& matrix,
                                            double*& collb, double*& colub,
                                            double*& obj, double*& rowlb,
                                            double*& rowub) {
  HighsOptions& options = this->highs->options_;
  highsLogDev(options.log_options, HighsLogType::kInfo,
              "Calling OsiHiGHSSolverInterface::assignProblem()\n");
  loadProblem(*matrix, collb, colub, obj, rowlb, rowub);
  delete matrix;
  matrix = 0;
  delete[] collb;
  collb = 0;
  delete[] colub;
  colub = 0;
  delete[] obj;
  obj = 0;
  delete[] rowlb;
  rowlb = 0;
  delete[] rowub;
  rowub = 0;
}

void OsiHiGHSSolverInterface::loadProblem(const CoinPackedMatrix& matrix,
                                          const double* collb,
                                          const double* colub,
                                          const double* obj, const char* rowsen,
                                          const double* rowrhs,
                                          const double* rowrng) {
  HighsOptions& options = this->highs->options_;
  highsLogDev(options.log_options, HighsLogType::kInfo,
              "Calling OsiHiGHSSolverInterface::loadProblem()\n");
  HighsInt numRow = matrix.getNumRows();

  double* rowlb = new double[numRow];
  double* rowub = new double[numRow];

  char* myrowsen = (char*)rowsen;
  bool rowsennull = false;
  double* myrowrhs = (double*)rowrhs;
  bool rowrhsnull = false;
  double* myrowrng = (double*)rowrng;
  bool rowrngnull = false;

  if (rowsen == NULL) {
    rowsennull = true;
    myrowsen = new char[numRow];
    for (HighsInt i = 0; i < numRow; i++) {
      myrowsen[i] = 'G';
    }
  }

  if (rowrhs == NULL) {
    rowsennull = true;
    myrowrhs = new double[numRow];
    for (HighsInt i = 0; i < numRow; i++) {
      myrowrhs[i] = 0.0;
    }
  }

  if (rowrng == NULL) {
    rowrngnull = true;
    myrowrng = new double[numRow];
    for (HighsInt i = 0; i < numRow; i++) {
      myrowrng[i] = 0.0;
    }
  }

  for (HighsInt i = 0; i < numRow; i++) {
    this->convertSenseToBound(myrowsen[i], myrowrhs[i], myrowrng[i], rowlb[i],
                              rowub[i]);
  }

  this->loadProblem(matrix, collb, colub, obj, rowlb, rowub);

  delete[] rowlb;
  delete[] rowub;

  if (rowsennull) {
    delete[] myrowsen;
  }

  if (rowrhsnull) {
    delete[] myrowrhs;
  }

  if (rowrngnull) {
    delete[] myrowrng;
  }
}

void OsiHiGHSSolverInterface::assignProblem(CoinPackedMatrix*& matrix,
                                            double*& collb, double*& colub,
                                            double*& obj, char*& rowsen,
                                            double*& rowrhs, double*& rowrng) {
  HighsOptions& options = this->highs->options_;
  highsLogDev(options.log_options, HighsLogType::kInfo,
              "Calling OsiHiGHSSolverInterface::assignProblem()\n");
  loadProblem(*matrix, collb, colub, obj, rowsen, rowrhs, rowrng);
  delete matrix;
  matrix = 0;
  delete[] collb;
  collb = 0;
  delete[] colub;
  colub = 0;
  delete[] obj;
  obj = 0;
  delete[] rowsen;
  rowsen = 0;
  delete[] rowrhs;
  rowrhs = 0;
  delete[] rowrng;
  rowrng = 0;
}

void OsiHiGHSSolverInterface::loadProblem(
    const HighsInt numcols, const HighsInt numrows, const CoinBigIndex* start,
    const HighsInt* index, const double* value, const double* collb,
    const double* colub, const double* obj, const double* rowlb,
    const double* rowub) {
  HighsOptions& options = this->highs->options_;
  highsLogDev(options.log_options, HighsLogType::kInfo,
              "Calling OsiHiGHSSolverInterface::loadProblem()\n");
  double oldObjSense = this->getObjSense();

  HighsLp lp;

  lp.num_row_ = numrows;
  lp.num_col_ = numcols;

  // setup HighsLp data structures
  lp.col_cost_.resize(numcols);
  lp.col_upper_.resize(numcols);
  lp.col_lower_.resize(numcols);

  lp.row_lower_.resize(numrows);
  lp.row_upper_.resize(numrows);

  lp.a_matrix_.start_.resize(numcols + 1);
  lp.a_matrix_.index_.resize(start[numcols]);
  lp.a_matrix_.value_.resize(start[numcols]);

  // copy data
  if (obj != NULL) {
    lp.col_cost_.assign(obj, obj + numcols);
  } else {
    lp.col_cost_.assign(numcols, 0.0);
  }

  if (collb != NULL) {
    lp.col_lower_.assign(collb, collb + numcols);
  } else {
    lp.col_lower_.assign(numcols, 0.0);
  }

  if (colub != NULL) {
    lp.col_upper_.assign(colub, colub + numcols);
  } else {
    lp.col_upper_.assign(numcols, kHighsInf);
  }

  if (rowlb != NULL) {
    lp.row_lower_.assign(rowlb, rowlb + numrows);
  } else {
    lp.row_lower_.assign(numrows, -kHighsInf);
  }

  if (rowub != NULL) {
    lp.row_upper_.assign(rowub, rowub + numrows);
  } else {
    lp.row_upper_.assign(numrows, kHighsInf);
  }

  lp.a_matrix_.start_.assign(start, start + numcols + 1);
  lp.a_matrix_.index_.assign(index, index + start[numcols]);
  lp.a_matrix_.value_.assign(value, value + start[numcols]);
  this->highs->passModel(lp);
  this->setObjSense(oldObjSense);
}

void OsiHiGHSSolverInterface::loadProblem(
    const HighsInt numcols, const HighsInt numrows, const CoinBigIndex* start,
    const HighsInt* index, const double* value, const double* collb,
    const double* colub, const double* obj, const char* rowsen,
    const double* rowrhs, const double* rowrng) {
  HighsOptions& options = this->highs->options_;
  highsLogDev(options.log_options, HighsLogType::kInfo,
              "Calling OsiHiGHSSolverInterface::loadProblem()\n");
  double* rowlb = new double[numrows];
  double* rowub = new double[numrows];

  for (HighsInt i = 0; i < numrows; i++) {
    this->convertSenseToBound(rowsen[i], rowrhs[i], rowrng[i], rowlb[i],
                              rowub[i]);
  }

  this->loadProblem(numcols, numrows, start, index, value, collb, colub, obj,
                    rowlb, rowub);

  delete[] rowlb;
  delete[] rowub;
}

void OsiHiGHSSolverInterface::loadProblem(
    const CoinPackedMatrix& matrix, const double* collb, const double* colub,
    const double* obj, const double* rowlb, const double* rowub) {
  HighsOptions& options = this->highs->options_;
  highsLogDev(options.log_options, HighsLogType::kInfo,
              "Calling OsiHiGHSSolverInterface::loadProblem()\n");
  bool transpose = false;
  if (!matrix.isColOrdered()) {
    transpose = true;
    // ToDo: remove this hack
    //((CoinPackedMatrix *)&matrix)->transpose();
    ((CoinPackedMatrix*)&matrix)->reverseOrdering();
  }

  HighsInt numCol = matrix.getNumCols();
  HighsInt numRow = matrix.getNumRows();
  HighsInt num_nz = matrix.getNumElements();

  HighsInt* start = new int[numCol + 1];
  HighsInt* index = new int[num_nz];
  double* value = new double[num_nz];

  // get matrix data
  // const CoinBigIndex *vectorStarts = matrix.getVectorStarts();
  const HighsInt* vectorLengths = matrix.getVectorLengths();
  const double* elements = matrix.getElements();
  const HighsInt* indices = matrix.getIndices();

  // set matrix in HighsLp
  start[0] = 0;
  HighsInt nz = 0;
  for (HighsInt i = 0; i < numCol; i++) {
    start[i + 1] = start[i] + vectorLengths[i];
    CoinBigIndex first = matrix.getVectorFirst(i);
    for (HighsInt j = 0; j < vectorLengths[i]; j++) {
      index[nz] = indices[first + j];
      value[nz] = elements[first + j];
      nz++;
    }
  }
  assert(num_nz == nz);

  this->loadProblem(numCol, numRow, start, index, value, collb, colub, obj,
                    rowlb, rowub);

  if (transpose) {
    //((CoinPackedMatrix)matrix).transpose();
    ((CoinPackedMatrix*)&matrix)->reverseOrdering();
  }

  delete[] start;
  delete[] index;
  delete[] value;
}

/// Read a problem in MPS format from the given filename.
// HighsInt OsiHiGHSSolverInterface::readMps(const char *filename,
//   const char *extension)
// {
//   HighsOptions& options = this->highs->options_;
//   highsLogDev(options.log_options, HighsLogType::kInfo,
//                     "Calling OsiHiGHSSolverInterface::readMps()\n");

//   HighsModel model;

//   highs->options_.filename = std::string(filename) + "." +
//   std::string(extension);

//   FilereaderRetcode rc = FilereaderMps().readModelFromFile(highs->options_,
//   model); if (rc != FilereaderRetcode::kOk)
// 	  return (HighsInt)rc;
//   this->setDblParam(OsiDblParam::OsiObjOffset, model.lp_.offset_);
//   highs->passModel(model);

//   return 0;
// }

/// Write the problem into an mps file of the given filename.
void OsiHiGHSSolverInterface::writeMps(const char* filename,
                                       const char* extension,
                                       double objSense) const {
  HighsOptions& options = this->highs->options_;
  highsLogDev(options.log_options, HighsLogType::kInfo,
              "Calling OsiHiGHSSolverInterface::writeMps()\n");

  std::string fullname = std::string(filename) + "." + std::string(extension);

  if (objSense != 0.0) {
    // HiGHS doesn't do funny stuff with the objective sense, so use Osi's
    // method if something strange is requested
    OsiSolverInterface::writeMpsNative(fullname.c_str(), NULL, NULL, 0, 2,
                                       objSense);
    return;
  }

  FilereaderMps frmps;
  HighsStatus rc =
      frmps.writeModelToFile(highs->options_, fullname, highs->model_);

  if (rc != HighsStatus::kOk)
    throw CoinError("Creating MPS file failed", "writeMps",
                    "OsiHiGHSSolverInterface", __FILE__, __LINE__);
}

void OsiHiGHSSolverInterface::passInMessageHandler(
    CoinMessageHandler* handler) {
  OsiSolverInterface::passInMessageHandler(handler);

  this->highs->setLogCallback(logtomessagehandler, (void*)handler);
}

const double* OsiHiGHSSolverInterface::getColSolution() const {
  HighsOptions& options = this->highs->options_;
  highsLogDev(options.log_options, HighsLogType::kInfo,
              "Calling OsiHiGHSSolverInterface::getColSolution()\n");
  if (!highs) {
    return nullptr;
  } else {
    if (highs->solution_.col_value.size() == 0) {
      double num_cols = highs->getNumCol();
      this->dummy_solution->col_value.resize(num_cols);
      for (HighsInt col = 0; col < highs->getNumCol(); col++) {
        if (highs->getLp().col_lower_[col] <= 0 &&
            highs->getLp().col_upper_[col] >= 0)
          dummy_solution->col_value[col] = 0;
        else if (std::fabs(highs->getLp().col_lower_[col] <
                           std::fabs(highs->getLp().col_upper_[col])))
          dummy_solution->col_value[col] = highs->getLp().col_lower_[col];
        else
          dummy_solution->col_value[col] = highs->getLp().col_upper_[col];
      }
      return &dummy_solution->col_value[0];
    }
  }

  return &highs->solution_.col_value[0];
}

const double* OsiHiGHSSolverInterface::getRowPrice() const {
  HighsOptions& options = this->highs->options_;
  highsLogDev(options.log_options, HighsLogType::kInfo,
              "Calling OsiHiGHSSolverInterface::getRowPrice()\n");
  if (!highs)
    return nullptr;
  else {
    if (highs->solution_.row_dual.size() == 0) {
      double num_cols = highs->getNumCol();
      this->dummy_solution->row_dual.resize(num_cols);
      for (HighsInt col = 0; col < highs->getNumCol(); col++) {
        if (highs->getLp().col_lower_[col] <= 0 &&
            highs->getLp().col_upper_[col] >= 0)
          dummy_solution->row_dual[col] = 0;
        else if (std::fabs(highs->getLp().col_lower_[col] <
                           std::fabs(highs->getLp().col_upper_[col])))
          dummy_solution->row_dual[col] = highs->getLp().col_lower_[col];
        else
          dummy_solution->row_dual[col] = highs->getLp().col_upper_[col];
      }
      return &dummy_solution->row_dual[0];
    }
  }

  return &highs->solution_.row_dual[0];
}

const double* OsiHiGHSSolverInterface::getReducedCost() const {
  HighsOptions& options = this->highs->options_;
  highsLogDev(options.log_options, HighsLogType::kInfo,
              "Calling OsiHiGHSSolverInterface::getReducedCost()\n");
  if (!highs)
    return nullptr;
  else {
    if (highs->solution_.col_dual.size() == 0) {
      const HighsLp& lp = highs->getLp();
      double num_cols = lp.num_col_;
      this->dummy_solution->col_dual.resize(num_cols);
      for (HighsInt col = 0; col < num_cols; col++) {
        dummy_solution->col_dual[col] = lp.col_cost_[col];
        for (HighsInt i = lp.a_matrix_.start_[col];
             i < lp.a_matrix_.start_[col + 1]; i++) {
          const HighsInt row = lp.a_matrix_.index_[i];
          assert(row >= 0);
          assert(row < lp.num_row_);

          dummy_solution->col_dual[col] -=
              dummy_solution->row_dual[row] * lp.a_matrix_.value_[i];
        }
      }
      return &dummy_solution->col_dual[0];
    }
  }

  return &highs->solution_.col_dual[0];
}

const double* OsiHiGHSSolverInterface::getRowActivity() const {
  HighsOptions& options = this->highs->options_;
  highsLogDev(options.log_options, HighsLogType::kInfo,
              "Calling OsiHiGHSSolverInterface::getRowActivity()\n");
  if (!highs)
    return nullptr;
  else {
    if (highs->solution_.row_value.size() == 0) {
      double num_cols = highs->getNumCol();
      this->dummy_solution->row_value.resize(num_cols);
      for (HighsInt col = 0; col < highs->getNumCol(); col++) {
        if (highs->getLp().col_lower_[col] <= 0 &&
            highs->getLp().col_upper_[col] >= 0)
          dummy_solution->row_value[col] = 0;
        else if (std::fabs(highs->getLp().col_lower_[col] <
                           std::fabs(highs->getLp().col_upper_[col])))
          dummy_solution->row_value[col] = highs->getLp().col_lower_[col];
        else
          dummy_solution->row_value[col] = highs->getLp().col_upper_[col];
      }
      return &dummy_solution->row_value[0];
    }
  }

  return &highs->solution_.row_value[0];
}

double OsiHiGHSSolverInterface::getObjValue() const {
  HighsOptions& options = this->highs->options_;
  highsLogDev(options.log_options, HighsLogType::kInfo,
              "Calling OsiHiGHSSolverInterface::getObjValue()\n");
  double objVal = 0.0;
  if (true || highs->solution_.col_value.size() == 0) {
    const double* sol = this->getColSolution();
    const double* cost = this->getObjCoefficients();
    HighsInt ncols = this->getNumCols();

    objVal = -this->objOffset;
    for (HighsInt i = 0; i < ncols; i++) {
      objVal += sol[i] * cost[i];
    }
  } else {
    this->highs->getInfoValue("objective_function_value", objVal);
  }

  return objVal;
}

HighsInt OsiHiGHSSolverInterface::getIterationCount() const {
  HighsOptions& options = this->highs->options_;
  highsLogDev(options.log_options, HighsLogType::kInfo,
              "Calling OsiHiGHSSolverInterface::getIterationCount()\n");
  if (!highs) {
    return 0;
  }
  HighsInt iteration_count;
  this->highs->getInfoValue("simplex_iteration_count", iteration_count);
  return iteration_count;
}

void OsiHiGHSSolverInterface::setRowPrice(const double* rowprice) {
  HighsOptions& options = this->highs->options_;
  highsLogDev(options.log_options, HighsLogType::kInfo,

              "Calling OsiHiGHSSolverInterface::setRowPrice()\n");
  if (!rowprice) return;
  HighsSolution solution;
  solution.row_dual.resize(highs->getNumRow());
  for (HighsInt row = 0; row < highs->getNumRow(); row++)
    solution.row_dual[row] = rowprice[row];

  /*HighsStatus result =*/highs->setSolution(solution);
}

void OsiHiGHSSolverInterface::setColSolution(const double* colsol) {
  HighsOptions& options = this->highs->options_;
  highsLogDev(options.log_options, HighsLogType::kInfo,
              "Calling OsiHiGHSSolverInterface::setColSolution()\n");
  if (!colsol) return;
  HighsSolution solution;
  solution.col_value.resize(highs->getNumCol());
  for (HighsInt col = 0; col < highs->getNumCol(); col++)
    solution.col_value[col] = colsol[col];

  /*HighsStatus result =*/highs->setSolution(solution);
}

void OsiHiGHSSolverInterface::applyRowCut(const OsiRowCut& rc) {
  HighsOptions& options = this->highs->options_;
  highsLogDev(options.log_options, HighsLogType::kInfo,
              "Calling OsiHiGHSSolverInterface::applyRowCut()\n");
}

void OsiHiGHSSolverInterface::applyColCut(const OsiColCut& cc) {
  HighsOptions& options = this->highs->options_;
  highsLogDev(options.log_options, HighsLogType::kInfo,
              "Calling OsiHiGHSSolverInterface::applyColCut()\n");
}

void OsiHiGHSSolverInterface::setContinuous(HighsInt index) {
  HighsOptions& options = this->highs->options_;
  highsLogDev(options.log_options, HighsLogType::kInfo,
              "Calling OsiHiGHSSolverInterface::setContinuous()\n");
}

void OsiHiGHSSolverInterface::setInteger(HighsInt index) {
  HighsOptions& options = this->highs->options_;
  highsLogDev(options.log_options, HighsLogType::kInfo,
              "Calling OsiHiGHSSolverInterface::setInteger()\n");
}

bool OsiHiGHSSolverInterface::isContinuous(HighsInt colNumber) const {
  HighsOptions& options = this->highs->options_;
  highsLogDev(options.log_options, HighsLogType::kInfo,
              "Calling OsiHiGHSSolverInterface::isContinuous()\n");
  return true;
}

void OsiHiGHSSolverInterface::setRowType(HighsInt index, char sense,
                                         double rightHandSide, double range) {
  HighsOptions& options = this->highs->options_;
  highsLogDev(options.log_options, HighsLogType::kInfo,
              "Calling OsiHiGHSSolverInterface::setRowType()\n");
  // Assign arbitrary values so that compilation is clean
  double lo = 0;
  double hi = 1e200;
  this->convertSenseToBound(sense, rightHandSide, range, lo, hi);
  this->setRowBounds(index, lo, hi);
}

void OsiHiGHSSolverInterface::setRowLower(HighsInt elementIndex,
                                          double elementValue) {
  HighsOptions& options = this->highs->options_;
  highsLogDev(options.log_options, HighsLogType::kInfo,
              "Calling OsiHiGHSSolverInterface::setRowLower()\n");

  double upper = this->getRowUpper()[elementIndex];

  this->highs->changeRowBounds(elementIndex, elementValue, upper);
}

void OsiHiGHSSolverInterface::setRowUpper(HighsInt elementIndex,
                                          double elementValue) {
  HighsOptions& options = this->highs->options_;
  highsLogDev(options.log_options, HighsLogType::kInfo,
              "Calling OsiHiGHSSolverInterface::setRowUpper()\n");
  double lower = this->getRowLower()[elementIndex];
  this->highs->changeRowBounds(elementIndex, lower, elementValue);
}

void OsiHiGHSSolverInterface::setColLower(HighsInt elementIndex,
                                          double elementValue) {
  HighsOptions& options = this->highs->options_;
  highsLogDev(options.log_options, HighsLogType::kInfo,
              "Calling OsiHiGHSSolverInterface::setColLower()\n");
  double upper = this->getColUpper()[elementIndex];
  this->highs->changeColBounds(elementIndex, elementValue, upper);
}

void OsiHiGHSSolverInterface::setColUpper(HighsInt elementIndex,
                                          double elementValue) {
  HighsOptions& options = this->highs->options_;
  highsLogDev(options.log_options, HighsLogType::kInfo,
              "Calling OsiHiGHSSolverInterface::setColUpper()\n");
  double lower = this->getColLower()[elementIndex];
  this->highs->changeColBounds(elementIndex, lower, elementValue);
}

void OsiHiGHSSolverInterface::setObjCoeff(HighsInt elementIndex,
                                          double elementValue) {
  HighsOptions& options = this->highs->options_;
  highsLogDev(options.log_options, HighsLogType::kInfo,
              "Calling OsiHiGHSSolverInterface::setObjCoeff()\n");
  this->highs->changeColCost(elementIndex, elementValue);
}

std::vector<double*> OsiHiGHSSolverInterface::getDualRays(HighsInt maxNumRays,
                                                          bool fullRay) const {
  HighsOptions& options = this->highs->options_;
  highsLogDev(options.log_options, HighsLogType::kInfo,
              "Calling OsiHiGHSSolverInterface::getDualRays()\n");
  return std::vector<double*>(0);
}

std::vector<double*> OsiHiGHSSolverInterface::getPrimalRays(
    HighsInt maxNumRays) const {
  HighsOptions& options = this->highs->options_;
  highsLogDev(options.log_options, HighsLogType::kInfo,
              "Calling OsiHiGHSSolverInterface::getPrimalRays()\n");
  return std::vector<double*>(0);
}

CoinWarmStart* OsiHiGHSSolverInterface::getEmptyWarmStart() const {
  HighsOptions& options = this->highs->options_;
  highsLogDev(options.log_options, HighsLogType::kInfo,
              "Calling OsiHiGHSSolverInterface::getEmptyWarmStart()\n");
  return (dynamic_cast<CoinWarmStart*>(new CoinWarmStartBasis()));
}

CoinWarmStart* OsiHiGHSSolverInterface::getWarmStart() const {
  HighsOptions& options = this->highs->options_;
  highsLogDev(options.log_options, HighsLogType::kInfo,
              "Calling OsiHiGHSSolverInterface::getWarmStart()\n");
  if (!highs) return NULL;

  if (highs->basis_.col_status.size() == 0 ||
      highs->basis_.row_status.size() == 0)
    return NULL;

  HighsInt num_cols = highs->getNumCol();
  HighsInt num_rows = highs->getNumRow();

  HighsInt* cstat = new int[num_cols];
  HighsInt* rstat = new int[num_rows];

  getBasisStatus(cstat, rstat);

  CoinWarmStartBasis* warm_start = new CoinWarmStartBasis();
  warm_start->setSize(num_cols, num_rows);

  for (HighsInt i = 0; i < num_rows; ++i)
    warm_start->setArtifStatus(i, CoinWarmStartBasis::Status(rstat[i]));
  for (HighsInt i = 0; i < num_cols; ++i)
    warm_start->setStructStatus(i, CoinWarmStartBasis::Status(cstat[i]));

  return warm_start;
}

bool OsiHiGHSSolverInterface::setWarmStart(const CoinWarmStart* warmstart) {
  HighsOptions& options = this->highs->options_;
  highsLogDev(options.log_options, HighsLogType::kInfo,
              "Calling OsiHiGHSSolverInterface::setWarmStart()\n");
  return false;
}

void OsiHiGHSSolverInterface::resolve() {
  HighsOptions& options = this->highs->options_;
  highsLogDev(options.log_options, HighsLogType::kInfo,
              "Calling OsiHiGHSSolverInterface::resolve()\n");
  this->status = this->highs->run();
}

void OsiHiGHSSolverInterface::setRowBounds(HighsInt elementIndex, double lower,
                                           double upper) {
  HighsOptions& options = this->highs->options_;
  highsLogDev(options.log_options, HighsLogType::kInfo,
              "Calling OsiHiGHSSolverInterface::setRowBounds()\n");

  this->highs->changeRowBounds(elementIndex, lower, upper);
}

void OsiHiGHSSolverInterface::setColBounds(HighsInt elementIndex, double lower,
                                           double upper) {
  HighsOptions& options = this->highs->options_;
  highsLogDev(options.log_options, HighsLogType::kInfo,
              "Calling OsiHiGHSSolverInterface::setColBounds()\n");

  this->highs->changeColBounds(elementIndex, lower, upper);
}

void OsiHiGHSSolverInterface::setRowSetBounds(const HighsInt* indexFirst,
                                              const HighsInt* indexLast,
                                              const double* boundList) {
  HighsOptions& options = this->highs->options_;
  highsLogDev(options.log_options, HighsLogType::kInfo,
              "Calling OsiHiGHSSolverInterface::setRowSetBounds()\n");
  OsiSolverInterface::setRowSetBounds(indexFirst, indexLast - 1, boundList);
}

void OsiHiGHSSolverInterface::setColSetBounds(const HighsInt* indexFirst,
                                              const HighsInt* indexLast,
                                              const double* boundList) {
  HighsOptions& options = this->highs->options_;
  highsLogDev(options.log_options, HighsLogType::kInfo,
              "Calling OsiHiGHSSolverInterface::setColSetBounds()\n");
  OsiSolverInterface::setColSetBounds(indexFirst, indexLast - 1, boundList);
}

void OsiHiGHSSolverInterface::branchAndBound() {
  HighsOptions& options = this->highs->options_;
  highsLogDev(options.log_options, HighsLogType::kInfo,
              "Calling OsiHiGHSSolverInterface::branchAndBound()\n");
  // TODO
}

void OsiHiGHSSolverInterface::setObjCoeffSet(const HighsInt* indexFirst,
                                             const HighsInt* indexLast,
                                             const double* coeffList) {
  HighsOptions& options = this->highs->options_;
  highsLogDev(options.log_options, HighsLogType::kInfo,
              "Calling OsiHiGHSSolverInterface::setObjCoeffSet()\n");
  OsiSolverInterface::setObjCoeffSet(indexFirst, indexLast - 1, coeffList);
}

HighsInt OsiHiGHSSolverInterface::canDoSimplexInterface() const { return 0; }

/* Osi return codes:
0: free
1: basic
2: upper
3: lower
*/
void OsiHiGHSSolverInterface::getBasisStatus(HighsInt* cstat,
                                             HighsInt* rstat) const {
  if (!highs) return;

  if (highs->basis_.col_status.size() == 0 ||
      highs->basis_.row_status.size() == 0)
    return;

  for (size_t i = 0; i < highs->basis_.col_status.size(); ++i)
    switch (highs->basis_.col_status[i]) {
      case HighsBasisStatus::kBasic:
        cstat[i] = 1;
        break;
      case HighsBasisStatus::kLower:
        cstat[i] = 3;
        break;
      case HighsBasisStatus::kUpper:
        cstat[i] = 2;
        break;
      case HighsBasisStatus::kZero:
        cstat[i] = 0;
        break;
      case HighsBasisStatus::kNonbasic:
        cstat[i] = 3;
        break;
    }

  for (size_t i = 0; i < highs->basis_.row_status.size(); ++i)
    switch (highs->basis_.row_status[i]) {
      case HighsBasisStatus::kBasic:
        rstat[i] = 1;
        break;
      case HighsBasisStatus::kLower:
        rstat[i] = 3;
        break;
      case HighsBasisStatus::kUpper:
        rstat[i] = 2;
        break;
      case HighsBasisStatus::kZero:
        rstat[i] = 0;
        break;
      case HighsBasisStatus::kNonbasic:
        rstat[i] = 3;
        break;
    }
}

void OsiHiGHSSolverInterface::setRowNames(OsiNameVec& srcNames,
                                          HighsInt srcStart, HighsInt len,
                                          HighsInt tgtStart) {}

void OsiHiGHSSolverInterface::setColNames(OsiNameVec& srcNames,
                                          HighsInt srcStart, HighsInt len,
                                          HighsInt tgtStart) {}

void OsiSolverInterfaceMpsUnitTest(
    const std::vector<OsiSolverInterface*>& vecSiP, const std::string& mpsDir) {
}
