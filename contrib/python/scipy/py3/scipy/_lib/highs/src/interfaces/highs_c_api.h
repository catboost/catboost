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
#ifndef HIGHS_C_API
#define HIGHS_C_API

#include "util/HighsInt.h"

const HighsInt kHighsStatusError = -1;
const HighsInt kHighsStatusOk = 0;
const HighsInt kHighsStatusWarning = 1;

const HighsInt kHighsVarTypeContinuous = 0;
const HighsInt kHighsVarTypeInteger = 1;
const HighsInt kHighsVarTypeSemiContinuous = 2;
const HighsInt kHighsVarTypeSemiInteger = 3;
const HighsInt kHighsVarTypeImplicitInteger = 4;

const HighsInt kHighsOptionTypeBool = 0;
const HighsInt kHighsOptionTypeInt = 1;
const HighsInt kHighsOptionTypeDouble = 2;
const HighsInt kHighsOptionTypeString = 3;

const HighsInt kHighsInfoTypeInt64 = -1;
const HighsInt kHighsInfoTypeInt = 1;
const HighsInt kHighsInfoTypeDouble = 2;

const HighsInt kHighsObjSenseMinimize = 1;
const HighsInt kHighsObjSenseMaximize = -1;

const HighsInt kHighsMatrixFormatColwise = 1;
const HighsInt kHighsMatrixFormatRowwise = 2;

const HighsInt kHighsHessianFormatTriangular = 1;
const HighsInt kHighsHessianFormatSquare = 2;

const HighsInt kHighsSolutionStatusNone = 0;
const HighsInt kHighsSolutionStatusInfeasible = 1;
const HighsInt kHighsSolutionStatusFeasible = 2;

const HighsInt kHighsBasisValidityInvalid = 0;
const HighsInt kHighsBasisValidityValid = 1;

const HighsInt kHighsPresolveStatusNotPresolved = -1;
const HighsInt kHighsPresolveStatusNotReduced = 0;
const HighsInt kHighsPresolveStatusInfeasible = 1;
const HighsInt kHighsPresolveStatusUnboundedOrInfeasible = 2;
const HighsInt kHighsPresolveStatusReduced = 3;
const HighsInt kHighsPresolveStatusReducedToEmpty = 4;
const HighsInt kHighsPresolveStatusTimeout = 5;
const HighsInt kHighsPresolveStatusNullError = 6;
const HighsInt kHighsPresolveStatusOptionsError = 7;

const HighsInt kHighsModelStatusNotset = 0;
const HighsInt kHighsModelStatusLoadError = 1;
const HighsInt kHighsModelStatusModelError = 2;
const HighsInt kHighsModelStatusPresolveError = 3;
const HighsInt kHighsModelStatusSolveError = 4;
const HighsInt kHighsModelStatusPostsolveError = 5;
const HighsInt kHighsModelStatusModelEmpty = 6;
const HighsInt kHighsModelStatusOptimal = 7;
const HighsInt kHighsModelStatusInfeasible = 8;
const HighsInt kHighsModelStatusUnboundedOrInfeasible = 9;
const HighsInt kHighsModelStatusUnbounded = 10;
const HighsInt kHighsModelStatusObjectiveBound = 11;
const HighsInt kHighsModelStatusObjectiveTarget = 12;
const HighsInt kHighsModelStatusTimeLimit = 13;
const HighsInt kHighsModelStatusIterationLimit = 14;
const HighsInt kHighsModelStatusUnknown = 15;

const HighsInt kHighsBasisStatusLower = 0;
const HighsInt kHighsBasisStatusBasic = 1;
const HighsInt kHighsBasisStatusUpper = 2;
const HighsInt kHighsBasisStatusZero = 3;
const HighsInt kHighsBasisStatusNonbasic = 4;

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Formulate and solve a linear program using HiGHS.
 *
 * @param num_col   the number of columns
 * @param num_row   the number of rows
 * @param num_nz    the number of nonzeros in the constraint matrix
 * @param a_format  the format of the constraint matrix as a
 *                  `kHighsMatrixFormat` constant
 * @param sense     the optimization sense as a `kHighsObjSense` constant
 * @param offset    the objective constant
 * @param col_cost  array of length [num_col] with the column costs
 * @param col_lower array of length [num_col] with the column lower bounds
 * @param col_upper array of length [num_col] with the column upper bounds
 * @param row_lower array of length [num_row] with the row lower bounds
 * @param row_upper array of length [num_row] with the row upper bounds
 * @param a_start   the constraint matrix is provided to HiGHS in compressed
 *                  sparse column form (if `a_format` is
 *                  `kHighsMatrixFormatColwise`, otherwise compressed sparse row
 *                  form). The sparse matrix consists of three arrays,
 *                  `a_start`, `a_index`, and `a_value`. `a_start` is an array
 *                  of length [num_col] containing the starting index of each
 *                  column in `a_index`. If `a_format` is
 *                  `kHighsMatrixFormatRowwise` the array is of length [num_row]
 *                  corresponding to each row.
 * @param a_index   array of length [num_nz] with indices of matrix entries
 * @param a_value   array of length [num_nz] with values of matrix entries
 *
 * @param col_value      array of length [num_col], filled with the primal
 *                       column solution
 * @param col_dual       array of length [num_col], filled with the dual column
 *                       solution
 * @param row_value      array of length [num_row], filled with the primal row
 *                       solution
 * @param row_dual       array of length [num_row], filled with the dual row
 *                       solution
 * @param col_basis_status  array of length [num_col], filled with the basis
 *                          status of the columns in the form of a
 *                          `kHighsBasisStatus` constant
 * @param row_basis_status  array of length [num_row], filled with the basis
 *                          status of the rows in the form of a
 *                          `kHighsBasisStatus` constant
 * @param model_status      termination status of the model after the solve in
 *                          the form of a `kHighsModelStatus` constant
 *
 * @returns a `kHighsStatus` constant indicating whether the call succeeded
 */
HighsInt Highs_lpCall(const HighsInt num_col, const HighsInt num_row,
                      const HighsInt num_nz, const HighsInt a_format,
                      const HighsInt sense, const double offset,
                      const double* col_cost, const double* col_lower,
                      const double* col_upper, const double* row_lower,
                      const double* row_upper, const HighsInt* a_start,
                      const HighsInt* a_index, const double* a_value,
                      double* col_value, double* col_dual, double* row_value,
                      double* row_dual, HighsInt* col_basis_status,
                      HighsInt* row_basis_status, HighsInt* model_status);

/**
 * Formulate and solve a mixed-integer linear program using HiGHS.
 *
 * The signature of this method is identical to `Highs_lpCall`, except that it
 * has an additional `integrality` argument, and that it is missing the
 * `col_dual`, `row_dual`, `col_basis_status` and `row_basis_status` arguments.
 *
 * @param integrality   array of length [num_col] containing a `kHighsVarType`
 *                      constant for each column
 *
 * @returns a `kHighsStatus` constant indicating whether the call succeeded
 */
HighsInt Highs_mipCall(const HighsInt num_col, const HighsInt num_row,
                       const HighsInt num_nz, const HighsInt a_format,
                       const HighsInt sense, const double offset,
                       const double* col_cost, const double* col_lower,
                       const double* col_upper, const double* row_lower,
                       const double* row_upper, const HighsInt* a_start,
                       const HighsInt* a_index, const double* a_value,
                       const HighsInt* integrality, double* col_value,
                       double* row_value, HighsInt* model_status);

/**
 * Formulate and solve a quadratic program using HiGHS.
 *
 * The signature of this method is identical to `Highs_lpCall`, except that it
 * has additional arguments for specifying the Hessian matrix.

 * @param q_num_nz  the number of nonzeros in the Hessian matrix
 * @param q_format  the format of the Hessian matrix in the form of a
 *                  `kHighsHessianStatus` constant. If q_num_nz > 0, this must
                    be `kHighsHessianFormatTriangular`
 * @param q_start   the Hessian matrix is provided in the same format as the
 *                  constraint matrix, using `q_start`, `q_index`, and `q_value`
 *                  in the place of `a_start`, `a_index`, and `a_value`
 * @param q_index   array of length [q_num_nz] with indices of matrix entries
 * @param q_value   array of length [q_num_nz] with values of matrix entries
  *
 * @returns a `kHighsStatus` constant indicating whether the call succeeded
 */
HighsInt Highs_qpCall(
    const HighsInt num_col, const HighsInt num_row, const HighsInt num_nz,
    const HighsInt q_num_nz, const HighsInt a_format, const HighsInt q_format,
    const HighsInt sense, const double offset, const double* col_cost,
    const double* col_lower, const double* col_upper, const double* row_lower,
    const double* row_upper, const HighsInt* a_start, const HighsInt* a_index,
    const double* a_value, const HighsInt* q_start, const HighsInt* q_index,
    const double* q_value, double* col_value, double* col_dual,
    double* row_value, double* row_dual, HighsInt* col_basis_status,
    HighsInt* row_basis_status, HighsInt* model_status);

/**
 * Create a Highs instance and return the reference.
 *
 * Call `Highs_destroy` on the returned reference to clean up allocated memory.
 *
 * @returns A pointer to the Highs instance
 */
void* Highs_create(void);

/**
 * Destroy the model `highs` created by `Highs_create` and free all
 * corresponding memory. Future calls using `highs` are not allowed.
 *
 * To empty a model without invalidating `highs`, see `Highs_clearModel`.
 *
 * @param highs     a pointer to the Highs instance
 */
void Highs_destroy(void* highs);

/**
 * Read a model from `filename` into `highs`.
 *
 * @param highs     a pointer to the Highs instance
 * @param filename  the filename to read
 *
 * @returns a `kHighsStatus` constant indicating whether the call succeeded
 */
HighsInt Highs_readModel(void* highs, const char* filename);

/**
 * Write the model in `highs` to `filename`.
 *
 * @param highs     a pointer to the Highs instance
 * @param filename  the filename to write.
 *
 * @returns a `kHighsStatus` constant indicating whether the call succeeded
 */
HighsInt Highs_writeModel(void* highs, const char* filename);

/**
 * Reset the options and then calls clearModel()
 *
 * See `Highs_destroy` to free all associated memory.
 *
 * @param highs     a pointer to the Highs instance
 *
 * @returns a `kHighsStatus` constant indicating whether the call succeeded
 */
HighsInt Highs_clear(void* highs);

/**
 * Remove all variables and constraints from the model `highs`, but do not
 * invalidate the pointer `highs`. Future calls (for example, adding new
 * variables and constraints) are allowed.
 *
 * @param highs     a pointer to the Highs instance
 *
 * @returns a `kHighsStatus` constant indicating whether the call succeeded
 */
HighsInt Highs_clearModel(void* highs);

/**
 * Clear all solution data associated with the model
 *
 * See `Highs_destroy` to clear the model and free all associated memory.
 *
 * @param highs     a pointer to the Highs instance
 *
 * @returns a `kHighsStatus` constant indicating whether the call succeeded
 */
HighsInt Highs_clearSolver(void* highs);

/**
 * Optimize a model. The algorithm used by HiGHS depends on the options that
 * have been set.
 *
 * @param highs     a pointer to the Highs instance
 *
 * @returns a `kHighsStatus` constant indicating whether the call succeeded
 */
HighsInt Highs_run(void* highs);

/**
 * Write the solution information (including dual and basis status, if
 * available) to a file.
 *
 * See also: `Highs_writeSolutionPretty`.
 *
 * @param highs     a pointer to the Highs instance
 * @param filename  the name of the file to write the results to
 *
 * @returns a `kHighsStatus` constant indicating whether the call succeeded
 */
HighsInt Highs_writeSolution(const void* highs, const char* filename);

/**
 * Write the solution information (including dual and basis status, if
 * available) to a file in a human-readable format.
 *
 * The method identical to `Highs_writeSolution`, except that the
 * printout is in a human-readiable format.
 *
 * @param highs     a pointer to the Highs instance
 * @param filename  the name of the file to write the results to
 *
 * @returns a `kHighsStatus` constant indicating whether the call succeeded
 */
HighsInt Highs_writeSolutionPretty(const void* highs, const char* filename);

/**
 * Pass a linear program (LP) to HiGHS in a single function call.
 *
 * The signature of this function is identical to `Highs_passModel`, without the
 * arguments for passing the Hessian matrix of a quadratic program and the
 * integrality vector.
 *
 * @returns a `kHighsStatus` constant indicating whether the call succeeded
 */
HighsInt Highs_passLp(void* highs, const HighsInt num_col,
                      const HighsInt num_row, const HighsInt num_nz,
                      const HighsInt a_format, const HighsInt sense,
                      const double offset, const double* col_cost,
                      const double* col_lower, const double* col_upper,
                      const double* row_lower, const double* row_upper,
                      const HighsInt* a_start, const HighsInt* a_index,
                      const double* a_value);

/**
 * Pass a mixed-integer linear program (MILP) to HiGHS in a single function
 * call.
 *
 * The signature of function is identical to `Highs_passModel`, without the
 * arguments for passing the Hessian matrix of a quadratic program.
 *
 * @returns a `kHighsStatus` constant indicating whether the call succeeded
 */
HighsInt Highs_passMip(void* highs, const HighsInt num_col,
                       const HighsInt num_row, const HighsInt num_nz,
                       const HighsInt a_format, const HighsInt sense,
                       const double offset, const double* col_cost,
                       const double* col_lower, const double* col_upper,
                       const double* row_lower, const double* row_upper,
                       const HighsInt* a_start, const HighsInt* a_index,
                       const double* a_value, const HighsInt* integrality);

/**
 * Pass a model to HiGHS in a single function call. This is faster than
 * constructing the model using `Highs_addRow` and `Highs_addCol`.
 *
 * @param highs       a pointer to the Highs instance
 * @param num_col     the number of columns
 * @param num_row     the number of rows
 * @param num_nz      the number of elements in the constraint matrix
 * @param q_num_nz    the number of elements in the Hessian matrix
 * @param a_format    the format of the constraint matrix to use in th form of a
 *                    `kHighsMatrixFormat` constant
 * @param q_format    the format of the Hessian matrix to use in the form of a
 *                    `kHighsHessianFormat` constant
 * @param sense       the optimization sense in the form of a `kHighsObjSense`
 *                    constant
 * @param offset      the constant term in the objective function
 * @param col_cost    array of length [num_col] with the objective coefficients
 * @param col_lower   array of length [num_col] with the lower column bounds
 * @param col_upper   array of length [num_col] with the upper column bounds
 * @param row_lower   array of length [num_row] with the upper row bounds
 * @param row_upper   array of length [num_row] with the upper row bounds
 * @param a_start     the constraint matrix is provided to HiGHS in compressed
 *                    sparse column form (if `a_format` is
 *                    `kHighsMatrixFormatColwise`, otherwise compressed sparse
 *                    row form). The sparse matrix consists of three arrays,
 *                    `a_start`, `a_index`, and `a_value`. `a_start` is an array
 *                    of length [num_col] containing the starting index of each
 *                    column in `a_index`. If `a_format` is
 *                    `kHighsMatrixFormatRowwise` the array is of length
 *                    [num_row] corresponding to each row.
 * @param a_index     array of length [num_nz] with indices of matrix entries
 * @param a_value     array of length [num_nz] with values of matrix entries
 * @param q_start     the Hessian matrix is provided in the same format as the
 *                    constraint matrix, using `q_start`, `q_index`, and
 *                    `q_value` in the place of `a_start`, `a_index`, and
 *                    `a_value`. If the model is linear, pass NULL.
 * @param q_index     array of length [q_num_nz] with indices of matrix entries.
 *                    If the model is linear, pass NULL.
 * @param q_value     array of length [q_num_nz] with values of matrix entries.
 *                    If the model is linear, pass NULL.
 * @param integrality an array of length [num_col] containing a `kHighsVarType`
 *                    consatnt for each column
 *
 * @returns a `kHighsStatus` constant indicating whether the call succeeded
 */
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
                         const HighsInt* integrality);

/**
 * Set the Hessian matrix for a quadratic objective.
 *
 * @param highs     a pointer to the Highs instance
 * @param dim       the dimension of the Hessian matrix. Should be [num_col].
 * @param num_nz    the number of non-zero elements in the Hessian matrix
 * @param format    the format of the Hessian matrix as a `kHighsHessianFormat`
 *                  constant. This must be `kHighsHessianFormatTriangular`.
 * @param start     the Hessian matrix is provided to HiGHS as the upper
 *                  triangular component in compressed sparse column form. The
 *                  sparse matrix consists of three arrays, `start`, `index`,
 *                  and `value`. `start` is an array of length [num_col]
 *                  containing the starting index of each column in `index`.
 * @param index     array of length [num_nz] with indices of matrix entries
 * @param value     array of length [num_nz] with values of matrix entries
 *
 * @returns a `kHighsStatus` constant indicating whether the call succeeded
 */
HighsInt Highs_passHessian(void* highs, const HighsInt dim,
                           const HighsInt num_nz, const HighsInt format,
                           const HighsInt* start, const HighsInt* index,
                           const double* value);

/**
 * Set a boolean-valued option.
 *
 * @param highs     a pointer to the Highs instance
 * @param option    the name of the option
 * @param value     the value of the option
 *
 * @returns a `kHighsStatus` constant indicating whether the call succeeded
 */
HighsInt Highs_setBoolOptionValue(void* highs, const char* option,
                                  const HighsInt value);

/**
 * Set an int-valued option.
 *
 * @param highs     a pointer to the Highs instance
 * @param option    the name of the option
 * @param value     the value of the option
 *
 * @returns a `kHighsStatus` constant indicating whether the call succeeded
 */
HighsInt Highs_setIntOptionValue(void* highs, const char* option,
                                 const HighsInt value);

/**
 * Set a double-valued option.
 *
 * @param highs     a pointer to the Highs instance
 * @param option    the name of the option
 * @param value     the value of the option
 *
 * @returns a `kHighsStatus` constant indicating whether the call succeeded
 */
HighsInt Highs_setDoubleOptionValue(void* highs, const char* option,
                                    const double value);

/**
 * Set a string-valued option.
 *
 * @param highs     a pointer to the Highs instance
 * @param option    the name of the option
 * @param value     the value of the option
 *
 * @returns a `kHighsStatus` constant indicating whether the call succeeded
 */
HighsInt Highs_setStringOptionValue(void* highs, const char* option,
                                    const char* value);

/**
 * Get a boolean-valued option.
 *
 * @param highs     a pointer to the Highs instance
 * @param option    the name of the option
 * @param value     storage for the value of the option
 *
 * @returns a `kHighsStatus` constant indicating whether the call succeeded
 */
HighsInt Highs_getBoolOptionValue(const void* highs, const char* option,
                                  HighsInt* value);

/**
 * Get an int-valued option.
 *
 * @param highs     a pointer to the Highs instance
 * @param option    the name of the option
 * @param value     storage for the value of the option
 *
 * @returns a `kHighsStatus` constant indicating whether the call succeeded
 */
HighsInt Highs_getIntOptionValue(const void* highs, const char* option,
                                 HighsInt* value);

/**
 * Get a double-valued option.
 *
 * @param highs     a pointer to the Highs instance
 * @param option    the name of the option
 * @param value     storage for the value of the option
 *
 * @returns a `kHighsStatus` constant indicating whether the call succeeded
 */
HighsInt Highs_getDoubleOptionValue(const void* highs, const char* option,
                                    double* value);

/**
 * Get a string-valued option.
 *
 * @param highs     a pointer to the Highs instance
 * @param option    the name of the option
 * @param value     pointer to allocated memory to store the value of the option
 *
 * @returns a `kHighsStatus` constant indicating whether the call succeeded
 */
HighsInt Highs_getStringOptionValue(const void* highs, const char* option,
                                    char* value);

/**
 * Get the type expected by an option.
 *
 * @param highs     a pointer to the Highs instance
 * @param option    the name of the option
 * @param type      int in which the corresponding `kHighsOptionType` constant
 *                  is stored
 *
 * @returns a `kHighsStatus` constant indicating whether the call succeeded
 */
HighsInt Highs_getOptionType(const void* highs, const char* option,
                             HighsInt* type);

/**
 * Reset all options to their default value.
 *
 * @param highs     a pointer to the Highs instance
 *
 * @returns a `kHighsStatus` constant indicating whether the call succeeded
 */
HighsInt Highs_resetOptions(void* highs);

/**
 * Write the current options to file.
 *
 * @param highs     a pointer to the Highs instance
 * @param filename  the filename to write the options to
 *
 * @returns a `kHighsStatus` constant indicating whether the call succeeded
 */
HighsInt Highs_writeOptions(const void* highs, const char* filename);

/**
 * Write the value of non-default options to file.
 *
 * This is similar to `Highs_writeOptions`, except only options with
 * non-default value are written to `filename`.
 *
 * @param highs     a pointer to the Highs instance
 * @param filename  the filename to write the options to
 *
 * @returns a `kHighsStatus` constant indicating whether the call succeeded
 */
HighsInt Highs_writeOptionsDeviations(const void* highs, const char* filename);

/**
 * Get an int-valued info value.
 *
 * @param highs     a pointer to the Highs instance
 * @param info      the name of the info item
 * @param value     a reference to an integer that the result will be stored in
 *
 * @returns a `kHighsStatus` constant indicating whether the call succeeded
 */
HighsInt Highs_getIntInfoValue(const void* highs, const char* info,
                               HighsInt* value);

/**
 * Get a double-valued info value.
 *
 * @param highs     a pointer to the Highs instance
 * @param info      the name of the info item
 * @param value     a reference to an double that the result will be stored in
 *
 * @returns a `kHighsStatus` constant indicating whether the call succeeded
 */
HighsInt Highs_getDoubleInfoValue(const void* highs, const char* info,
                                  double* value);

/**
 * Get an int64-valued info value.
 *
 * @param highs     a pointer to the Highs instance
 * @param info      the name of the info item
 * @param value     a reference to a int64 that the result will be stored in
 *
 * @returns a `kHighsStatus` constant indicating whether the call succeeded
 */
HighsInt Highs_getInt64InfoValue(const void* highs, const char* info,
                                 int64_t* value);

/**
 * Get the primal and dual solution from an optimized model.
 *
 * @param highs      a pointer to the Highs instance
 * @param col_value  array of length [num_col], filled with primal column values
 * @param col_dual   array of length [num_col], filled with dual column values
 * @param row_value  array of length [num_row], filled with primal row values
 * @param row_dual   array of length [num_row], filled with dual row values
 *
 * @returns a `kHighsStatus` constant indicating whether the call succeeded
 */
HighsInt Highs_getSolution(const void* highs, double* col_value,
                           double* col_dual, double* row_value,
                           double* row_dual);

/**
 * Given a linear program with a basic feasible solution, get the column and row
 * basis statuses.
 *
 * @param highs       a pointer to the Highs instance
 * @param col_status  array of length [num_col], to be filled with the column
 *                    basis statuses in the form of a `kHighsBasisStatus`
 *                    constant
 * @param row_status  array of length [num_row], to be filled with the row
 *                    basis statuses in the form of a `kHighsBasisStatus`
 *                    constant
 *
 * @returns a `kHighsStatus` constant indicating whether the call succeeded
 */
HighsInt Highs_getBasis(const void* highs, HighsInt* col_status,
                        HighsInt* row_status);

/**
 * Return the optimization status of the model in the form of a
 * `kHighsModelStatus` constant.
 *
 * @param highs     a pointer to the Highs instance
 *
 * @returns an integer corresponding to the `kHighsModelStatus` constant
 */
HighsInt Highs_getModelStatus(const void* highs);

/**
 * Get an unbounded dual ray that is a certificate of primal infeasibility.
 *
 * @param highs             a pointer to the Highs instance
 * @param has_dual_ray      a pointer to an int to store 1 if the dual ray
 *                          exists
 * @param dual_ray_value    an array of length [num_row] filled with the
 *                          unbounded ray
 *
 * @returns a `kHighsStatus` constant indicating whether the call succeeded
 */
HighsInt Highs_getDualRay(const void* highs, HighsInt* has_dual_ray,
                          double* dual_ray_value);

/**
 * Get an unbounded primal ray that is a certificate of dual infeasibility.
 *
 * @param highs             a pointer to the Highs instance
 * @param has_primal_ray    a pointer to an int to store 1 if the primal ray
 *                          exists
 * @param primal_ray_value  an array of length [num_col] filled with the
 *                          unbounded ray
 *
 * @returns a `kHighsStatus` constant indicating whether the call succeeded
 */
HighsInt Highs_getPrimalRay(const void* highs, HighsInt* has_primal_ray,
                            double* primal_ray_value);

/**
 * Get the primal objective function value.
 *
 * @param highs     a pointer to the Highs instance
 *
 * @returns the primal objective function value
 */
double Highs_getObjectiveValue(const void* highs);

/**
 * Get the indices of the rows and columns that make up the basis matrix of a
 * basic feasible solution.
 *
 * Non-negative entries are indices of columns, and negative entries are
 * `-row_index - 1`. For example, `{1, -1}` would be the second column and first
 * row.
 *
 * The order of these rows and columns is important for calls to the functions:
 *  - `Highs_getBasisInverseRow`
 *  - `Highs_getBasisInverseCol`
 *  - `Highs_getBasisSolve`
 *  - `Highs_getBasisTransposeSolve`
 *  - `Highs_getReducedRow`
 *  - `Highs_getReducedColumn`
 *
 * @param highs             a pointer to the Highs instance
 * @param basic_variables   array of size [num_rows], filled with the indices of
 *                          the basic variables
 *
 * @returns a `kHighsStatus` constant indicating whether the call succeeded
 */
HighsInt Highs_getBasicVariables(const void* highs, HighsInt* basic_variables);

/**
 * Get a row of the inverse basis matrix \f$B^{-1}\f$.
 *
 * See `Highs_getBasicVariables` for a description of the `B` matrix.
 *
 * The arrays `row_vector` and `row_index` must have an allocated length of
 * [num_row]. However, check `row_num_nz` to see how many non-zero elements are
 * actually stored.
 *
 * @param highs         a pointer to the Highs instance
 * @param row           index of the row to compute
 * @param row_vector    values of the non-zero elements
 * @param row_num_nz    the number of non-zeros in the row
 * @param row_index     indices of the non-zero elements
 *
 * @returns a `kHighsStatus` constant indicating whether the call succeeded
 */
HighsInt Highs_getBasisInverseRow(const void* highs, const HighsInt row,
                                  double* row_vector, HighsInt* row_num_nz,
                                  HighsInt* row_index);

/**
 * Get a column of the inverse basis matrix \f$B^{-1}\f$.
 *
 * See `Highs_getBasicVariables` for a description of the `B` matrix.
 *
 * The arrays `col_vector` and `col_index` must have an allocated length of
 * [num_row]. However, check `col_num_nz` to see how many non-zero elements are
 * actually stored.
 *
 * @param highs         a pointer to the Highs instance
 * @param col           index of the column to compute
 * @param col_vector    values of the non-zero elements
 * @param col_num_nz    the number of non-zeros in the column
 * @param col_index     indices of the non-zero elements

 * @returns a `kHighsStatus` constant indicating whether the call succeeded
 */
HighsInt Highs_getBasisInverseCol(const void* highs, const HighsInt col,
                                  double* col_vector, HighsInt* col_num_nz,
                                  HighsInt* col_index);

/**
 * Compute \f$\mathbf{x}=B^{-1}\mathbf{b}\f$ for a given vector
 * \f$\mathbf{b}\f$.
 *
 * See `Highs_getBasicVariables` for a description of the `B` matrix.
 *
 * The arrays `solution_vector` and `solution_index` must have an allocated
 * length of [num_row]. However, check `solution_num_nz` to see how many
 * non-zero elements are actually stored.
 *
 * @param highs             a pointer to the Highs instance
 * @param rhs               the right-hand side vector `b`
 * @param solution_vector   values of the non-zero elements
 * @param solution_num_nz   the number of non-zeros in the solution
 * @param solution_index    indices of the non-zero elements
 *
 * @returns a `kHighsStatus` constant indicating whether the call succeeded
 */
HighsInt Highs_getBasisSolve(const void* highs, const double* rhs,
                             double* solution_vector, HighsInt* solution_num_nz,
                             HighsInt* solution_index);

/**
 * Compute \f$\mathbf{x}=B^{-T}\mathbf{b}\f$ for a given vector
 * \f$\mathbf{b}\f$.
 *
 * See `Highs_getBasicVariables` for a description of the `B` matrix.
 *
 * The arrays `solution_vector` and `solution_index` must have an allocated
 * length of [num_row]. However, check `solution_num_nz` to see how many
 * non-zero elements are actually stored.
 *
 * @param highs             a pointer to the Highs instance
 * @param rhs               the right-hand side vector `b`
 * @param solution_vector   values of the non-zero elements
 * @param solution_num_nz   the number of non-zeros in the solution
 * @param solution_index    indices of the non-zero elements
 *
 * @returns a `kHighsStatus` constant indicating whether the call succeeded
 */
HighsInt Highs_getBasisTransposeSolve(const void* highs, const double* rhs,
                                      double* solution_vector,
                                      HighsInt* solution_nz,
                                      HighsInt* solution_index);

/**
 * Compute a row of \f$B^{-1}A\f$.
 *
 * See `Highs_getBasicVariables` for a description of the `B` matrix.
 *
 * The arrays `row_vector` and `row_index` must have an allocated length of
 * [num_row]. However, check `row_num_nz` to see how many non-zero elements are
 * actually stored.
 *
 * @param highs         a pointer to the Highs instance
 * @param row           index of the row to compute
 * @param row_vector    values of the non-zero elements
 * @param row_num_nz    the number of non-zeros in the row
 * @param row_index     indices of the non-zero elements
 *
 * @returns a `kHighsStatus` constant indicating whether the call succeeded
 */
HighsInt Highs_getReducedRow(const void* highs, const HighsInt row,
                             double* row_vector, HighsInt* row_num_nz,
                             HighsInt* row_index);

/**
 * Compute a column of \f$B^{-1}A\f$.
 *
 * See `Highs_getBasicVariables` for a description of the `B` matrix.
 *
 * The arrays `col_vector` and `col_index` must have an allocated length of
 * [num_row]. However, check `col_num_nz` to see how many non-zero elements are
 * actually stored.
 *
 * @param highs         a pointer to the Highs instance
 * @param col           index of the column to compute
 * @param col_vector    values of the non-zero elements
 * @param col_num_nz    the number of non-zeros in the column
 * @param col_index     indices of the non-zero elements

 * @returns a `kHighsStatus` constant indicating whether the call succeeded
 */
HighsInt Highs_getReducedColumn(const void* highs, const HighsInt col,
                                double* col_vector, HighsInt* col_num_nz,
                                HighsInt* col_index);

/**
 * Set a basic feasible solution by passing the column and row basis statuses to
 * the model.
 *
 * @param highs       a pointer to the Highs instance
 * @param col_status  an array of length [num_col] with the column basis status
 *                    in the form of `kHighsBasisStatus` constants
 * @param row_status  an array of length [num_row] with the row basis status
 *                    in the form of `kHighsBasisStatus` constants
 *
 * @returns a `kHighsStatus` constant indicating whether the call succeeded
 */
HighsInt Highs_setBasis(void* highs, const HighsInt* col_status,
                        const HighsInt* row_status);

/**
 * Set a logical basis in the model.
 *
 * @param highs     a pointer to the Highs instance
 *
 * @returns a `kHighsStatus` constant indicating whether the call succeeded
 */
HighsInt Highs_setLogicalBasis(void* highs);

/**
 * Set a solution by passing the column and row primal and dual
 * solution values. For any values that are unavailable pass NULL.
 *
 * @param highs       a pointer to the Highs instance
 * @param col_value   an array of length [num_col] with the column solution
 *                    values
 * @param row_value   an array of length [num_row] with the row solution
 *                    values
 * @param col_dual    an array of length [num_col] with the column dual values
 * @param row_dual    an array of length [num_row] with the row dual values
 *
 * @returns a `kHighsStatus` constant indicating whether the call succeeded
 */
HighsInt Highs_setSolution(void* highs, const double* col_value,
                           const double* row_value, const double* col_dual,
                           const double* row_dual);

/**
 * Return the cumulative wall-clock time spent in `Highs_run`.
 *
 * @param highs     a pointer to the Highs instance
 *
 * @returns the cumulative wall-clock time spent in `Highs_run`
 */
double Highs_getRunTime(const void* highs);

/**
 * Add a new column (variable) to the model.
 *
 * @param highs         a pointer to the Highs instance
 * @param cost          objective coefficient of the column
 * @param lower         lower bound of the column
 * @param upper         upper bound of the column
 * @param num_new_nz    number of non-zeros in the column
 * @param index         array of size [num_new_nz] with the row indices
 * @param value         array of size [num_new_nz] with row values
 *
 * @returns a `kHighsStatus` constant indicating whether the call succeeded
 */
HighsInt Highs_addCol(void* highs, const double cost, const double lower,
                      const double upper, const HighsInt num_new_nz,
                      const HighsInt* index, const double* value);

/**
 * Add multiple columns (variables) to the model.
 *
 * @param highs         a pointer to the Highs instance
 * @param num_new_col   number of new columns to add
 * @param costs         array of size [num_new_col] with objective coefficients
 * @param lower         array of size [num_new_col] with lower bounds
 * @param upper         array of size [num_new_col] with upper bounds
 * @param num_new_nz    number of new nonzeros in the constraint matrix
 * @param starts        the constraint coefficients are given as a matrix in
 *                      compressed sparse column form by the arrays `starts`,
 *                      `index`, and `value`. `starts` is an array of size
 *                      [num_new_cols] with the start index of each row in
 *                      indices and values.
 * @param index         array of size [num_new_nz] with row indices
 * @param value         array of size [num_new_nz] with row values
 *
 * @returns a `kHighsStatus` constant indicating whether the call succeeded
 */
HighsInt Highs_addCols(void* highs, const HighsInt num_new_col,
                       const double* costs, const double* lower,
                       const double* upper, const HighsInt num_new_nz,
                       const HighsInt* starts, const HighsInt* index,
                       const double* value);

/**
 * Add a new variable to the model.
 *
 * @param highs         a pointer to the Highs instance
 * @param lower         lower bound of the column
 * @param upper         upper bound of the column
 *
 * @returns a `kHighsStatus` constant indicating whether the call succeeded
 */
HighsInt Highs_addVar(void* highs, const double lower, const double upper);

/**
 * Add multiple variables to the model.
 *
 * @param highs         a pointer to the Highs instance
 * @param num_new_var   number of new variables to add
 * @param lower         array of size [num_new_var] with lower bounds
 * @param upper         array of size [num_new_var] with upper bounds
 *
 * @returns a `kHighsStatus` constant indicating whether the call succeeded
 */
HighsInt Highs_addVars(void* highs, const HighsInt num_new_var,
                       const double* lower, const double* upper);

/**
 * Add a new row (a linear constraint) to the model.
 *
 * @param highs         a pointer to the Highs instance
 * @param lower         lower bound of the row
 * @param upper         upper bound of the row
 * @param num_new_nz    number of non-zeros in the row
 * @param index         array of size [num_new_nz] with column indices
 * @param value         array of size [num_new_nz] with column values
 *
 * @returns a `kHighsStatus` constant indicating whether the call succeeded
 */
HighsInt Highs_addRow(void* highs, const double lower, const double upper,
                      const HighsInt num_new_nz, const HighsInt* index,
                      const double* value);

/**
 * Add multiple rows (linear constraints) to the model.
 *
 * @param highs         a pointer to the Highs instance
 * @param num_new_row   the number of new rows to add
 * @param lower         array of size [num_new_row] with the lower bounds of the
 *                      rows
 * @param upper         array of size [num_new_row] with the upper bounds of the
 *                      rows
 * @param num_new_nz    number of non-zeros in the rows
 * @param starts        the constraint coefficients are given as a matrix in
 *                      compressed sparse row form by the arrays `starts`,
 *                      `index`, and `value`. `starts` is an array of size
 *                      [num_new_rows] with the start index of each row in
 *                      indices and values.
 * @param index         array of size [num_new_nz] with column indices
 * @param value         array of size [num_new_nz] with column values
 *
 * @returns a `kHighsStatus` constant indicating whether the call succeeded
 */
HighsInt Highs_addRows(void* highs, const HighsInt num_new_row,
                       const double* lower, const double* upper,
                       const HighsInt num_new_nz, const HighsInt* starts,
                       const HighsInt* index, const double* value);

/**
 * Change the objective sense of the model.
 *
 * @param highs     a pointer to the Highs instance
 * @param sense     the new optimization sense in the form of a `kHighsObjSense`
 *                  constant
 *
 * @returns a `kHighsStatus` constant indicating whether the call succeeded
 */
HighsInt Highs_changeObjectiveSense(void* highs, const HighsInt sense);

/**
 * Change the objective offset of the model.
 *
 * @param highs     a pointer to the Highs instance
 * @param offset    the new objective offset
 *
 * @returns a `kHighsStatus` constant indicating whether the call succeeded
 */
HighsInt Highs_changeObjectiveOffset(void* highs, const double offset);

/**
 * Change the integrality of a column.
 *
 * @param highs         a pointer to the Highs instance
 * @param col           the column index to change
 * @param integrality   the new integrality of the column in the form of a
 *                      `kHighsVarType` constant
 *
 * @returns a `kHighsStatus` constant indicating whether the call succeeded
 */
HighsInt Highs_changeColIntegrality(void* highs, const HighsInt col,
                                    const HighsInt integrality);

/**
 * Change the integrality of multiple adjacent columns.
 *
 * @param highs         a pointer to the Highs instance
 * @param from_col      the index of the first column whose integrality changes
 * @param to_col        the index of the last column whose integrality
 *                      changes
 * @param integrality   an array of length [to_col - from_col + 1] with the new
 *                      integralities of the columns in the form of
 *                      `kHighsVarType` constants
 *
 * @returns a `kHighsStatus` constant indicating whether the call succeeded
 */
HighsInt Highs_changeColsIntegralityByRange(void* highs,
                                            const HighsInt from_col,
                                            const HighsInt to_col,
                                            const HighsInt* integrality);

/**
 * Change the integrality of multiple columns given by an array of indices.
 *
 * @param highs             a pointer to the Highs instance
 * @param num_set_entries   the number of columns to change
 * @param set               an array of size [num_set_entries] with the indices
 *                          of the columns to change
 * @param integrality       an array of length [num_set_entries] with the new
 *                          integralities of the columns in the form of
 *                          `kHighsVarType` constants
 *
 * @returns a `kHighsStatus` constant indicating whether the call succeeded
 */
HighsInt Highs_changeColsIntegralityBySet(void* highs,
                                          const HighsInt num_set_entries,
                                          const HighsInt* set,
                                          const HighsInt* integrality);

/**
 * Change the integrality of multiple columns given by a mask.
 *
 * @param highs         a pointer to the Highs instance
 * @param mask          an array of length [num_col] with 1 if the column
 *                      integrality should be changed and 0 otherwise
 * @param integrality   an array of length [num_col] with the new
 *                      integralities of the columns in the form of
 *                      `kHighsVarType` constants
 *
 * @returns a `kHighsStatus` constant indicating whether the call succeeded
 */
HighsInt Highs_changeColsIntegralityByMask(void* highs, const HighsInt* mask,
                                           const HighsInt* integrality);

/**
 * Change the objective coefficient of a column.
 *
 * @param highs     a pointer to the Highs instance
 * @param col       the index of the column fo change
 * @param cost      the new objective coefficient
 *
 * @returns a `kHighsStatus` constant indicating whether the call succeeded
 */
HighsInt Highs_changeColCost(void* highs, const HighsInt col,
                             const double cost);

/**
 * Change the cost coefficients of multiple adjacent columns.
 *
 * @param highs     a pointer to the Highs instance
 * @param from_col  the index of the first column whose cost changes
 * @param to_col    the index of the last column whose cost changes
 * @param cost      an array of length [to_col - from_col + 1] with the new
 *                  objective coefficients
 *
 * @returns a `kHighsStatus` constant indicating whether the call succeeded
 */
HighsInt Highs_changeColsCostByRange(void* highs, const HighsInt from_col,
                                     const HighsInt to_col, const double* cost);

/**
 * Change the cost of multiple columns given by an array of indices.
 *
 * @param highs             a pointer to the Highs instance
 * @param num_set_entries   the number of columns to change
 * @param set               an array of size [num_set_entries] with the indices
 *                          of the columns to change
 * @param cost              an array of length [num_set_entries] with the new
 *                          costs of the columns.
 *
 * @returns a `kHighsStatus` constant indicating whether the call succeeded
 */
HighsInt Highs_changeColsCostBySet(void* highs, const HighsInt num_set_entries,
                                   const HighsInt* set, const double* cost);

/**
 * Change the cost of multiple columns given by a mask.
 *
 * @param highs     a pointer to the Highs instance
 * @param mask      an array of length [num_col] with 1 if the column
 *                  cost should be changed and 0 otherwise
 * @param cost      an array of length [num_col] with the new costs
 *
 * @returns a `kHighsStatus` constant indicating whether the call succeeded
 */
HighsInt Highs_changeColsCostByMask(void* highs, const HighsInt* mask,
                                    const double* cost);

/**
 * Change the variable bounds of a column.
 *
 * @param highs     a pointer to the Highs instance
 * @param col       the index of the column whose bounds are to change
 * @param lower     the new lower bound
 * @param upper     the new upper bound
 *
 * @returns a `kHighsStatus` constant indicating whether the call succeeded
 */
HighsInt Highs_changeColBounds(void* highs, const HighsInt col,
                               const double lower, const double upper);

/**
 * Change the variable bounds of multiple adjacent columns.
 *
 * @param highs     a pointer to the Highs instance
 * @param from_col  the index of the first column whose bound changes
 * @param to_col    the index of the last column whose bound changes
 * @param lower     an array of length [to_col - from_col + 1] with the new
 *                  lower bounds
 * @param upper     an array of length [to_col - from_col + 1] with the new
 *                  upper bounds
 *
 * @returns a `kHighsStatus` constant indicating whether the call succeeded
 */
HighsInt Highs_changeColsBoundsByRange(void* highs, const HighsInt from_col,
                                       const HighsInt to_col,
                                       const double* lower,
                                       const double* upper);

/**
 * Change the bounds of multiple columns given by an array of indices.
 *
 * @param highs             a pointer to the Highs instance
 * @param num_set_entries   the number of columns to change
 * @param set               an array of size [num_set_entries] with the indices
 *                          of the columns to change
 * @param lower             an array of length [num_set_entries] with the new
 *                          lower bounds
 * @param upper             an array of length [num_set_entries] with the new
 *                          upper bounds
 *
 * @returns a `kHighsStatus` constant indicating whether the call succeeded
 */
HighsInt Highs_changeColsBoundsBySet(void* highs,
                                     const HighsInt num_set_entries,
                                     const HighsInt* set, const double* lower,
                                     const double* upper);

/**
 * Change the variable bounds of multiple columns given by a mask.
 *
 * @param highs     a pointer to the Highs instance
 * @param mask      an array of length [num_col] with 1 if the column
 *                  bounds should be changed and 0 otherwise
 * @param lower     an array of length [num_col] with the new lower bounds
 * @param upper     an array of length [num_col] with the new upper bounds
 *
 * @returns a `kHighsStatus` constant indicating whether the call succeeded
 */
HighsInt Highs_changeColsBoundsByMask(void* highs, const HighsInt* mask,
                                      const double* lower, const double* upper);

/**
 * Change the bounds of a row.
 *
 * @param highs     a pointer to the Highs instance
 * @param row       the index of the row whose bounds are to change
 * @param lower     the new lower bound
 * @param upper     the new upper bound
 *
 * @returns a `kHighsStatus` constant indicating whether the call succeeded
 */
HighsInt Highs_changeRowBounds(void* highs, const HighsInt row,
                               const double lower, const double upper);

/**
 * Change the bounds of multiple rows given by an array of indices.
 *
 * @param highs             a pointer to the Highs instance
 * @param num_set_entries   the number of rows to change
 * @param set               an array of size [num_set_entries] with the indices
 *                          of the rows to change
 * @param lower             an array of length [num_set_entries] with the new
 *                          lower bounds
 * @param upper             an array of length [num_set_entries] with the new
 *                          upper bounds
 *
 * @returns a `kHighsStatus` constant indicating whether the call succeeded
 */
HighsInt Highs_changeRowsBoundsBySet(void* highs,
                                     const HighsInt num_set_entries,
                                     const HighsInt* set, const double* lower,
                                     const double* upper);

/**
 * Change the bounds of multiple rows given by a mask.
 *
 * @param highs     a pointer to the Highs instance
 * @param mask      an array of length [num_row] with 1 if the row
 *                  bounds should be changed and 0 otherwise
 * @param lower     an array of length [num_row] with the new lower bounds
 * @param upper     an array of length [num_row] with the new upper bounds
 *
 * @returns a `kHighsStatus` constant indicating whether the call succeeded
 */
HighsInt Highs_changeRowsBoundsByMask(void* highs, const HighsInt* mask,
                                      const double* lower, const double* upper);

/**
 * Change a coefficient in the constraint matrix.
 *
 * @param highs     a pointer to the Highs instance
 * @param row       the index of the row to change
 * @param col       the index of the col to change
 * @param value     the new constraint coefficient
 *
 * @returns a `kHighsStatus` constant indicating whether the call succeeded
 */
HighsInt Highs_changeCoeff(void* highs, const HighsInt row, const HighsInt col,
                           const double value);

/**
 * Get the objective sense.
 *
 * @param highs     a pointer to the Highs instance
 * @param sense     stores the current objective sense as a `kHighsObjSense`
 *                  constant
 *
 * @returns a `kHighsStatus` constant indicating whether the call succeeded
 */
HighsInt Highs_getObjectiveSense(const void* highs, HighsInt* sense);

/**
 * Get the objective offset.
 *
 * @param highs     a pointer to the Highs instance
 * @param offset    stores the current objective offset
 *
 * @returns a `kHighsStatus` constant indicating whether the call succeeded
 */
HighsInt Highs_getObjectiveOffset(const void* highs, double* offset);

/**
 * Get data associated with multiple adjacent columns from the model.
 *
 * To query the constraint coefficients, this function should be called twice:
 *  - First, call this function with `matrix_start`, `matrix_index`, and
 *    `matrix_value` as `NULL`. This call will populate `num_nz` with the
 *    number of nonzero elements in the corresponding section of the constraint
 *    matrix.
 *  - Second, allocate new `matrix_index` and `matrix_value` arrays of length
 *    `num_nz` and call this function again to populate the new arrays with
 *    their contents.
 *
 * @param highs         a pointer to the Highs instance
 * @param from_col      the first column for which to query data for
 * @param to_col        the last column (inclusive) for which to query data for
 * @param num_col       an integer populated with the number of columns got from
 *                      the model (this should equal `to_col - from_col + 1`)
 * @param costs         array of size [to_col - from_col + 1] for the column
 *                      cost coefficients
 * @param lower         array of size [to_col - from_col + 1] for the column
 *                      lower bounds
 * @param upper         array of size [to_col - from_col + 1] for the column
 *                      upper bounds
 * @param num_nz        an integer populated with the number of non-zero
 *                      elements in the constraint matrix
 * @param matrix_start  array of size [to_col - from_col + 1] with the start
 *                      indices of each
 *                      column in `matrix_index` and `matrix_value`
 * @param matrix_index  array of size [num_nz] with the row indices of each
 *                      element in the constraint matrix
 * @param matrix_value  array of size [num_nz] with the non-zero elements of the
 *                      constraint matrix.
 *
 * @returns a `kHighsStatus` constant indicating whether the call succeeded
 */
HighsInt Highs_getColsByRange(const void* highs, const HighsInt from_col,
                              const HighsInt to_col, HighsInt* num_col,
                              double* costs, double* lower, double* upper,
                              HighsInt* num_nz, HighsInt* matrix_start,
                              HighsInt* matrix_index, double* matrix_value);

/**
 * Get data associated with multiple columns given by an array.
 *
 * This function is identical to `Highs_getColsByRange`, except for how the
 * columns are specified.
 *
 * @param num_set_indices   the number of indices in the set
 * @param set               array of size [num_set_entries] with the column
 *                          indices to get
 *
 * @returns a `kHighsStatus` constant indicating whether the call succeeded
 */
HighsInt Highs_getColsBySet(const void* highs, const HighsInt num_set_entries,
                            const HighsInt* set, HighsInt* num_col,
                            double* costs, double* lower, double* upper,
                            HighsInt* num_nz, HighsInt* matrix_start,
                            HighsInt* matrix_index, double* matrix_value);

/**
 * Get data associated with multiple columns given by a mask.
 *
 * This function is identical to `Highs_getColsByRange`, except for how the
 * columns are specified.
 *
 * @param mask  array of length [num_col] containing a 1 to get the column and 0
 *              otherwise
 *
 * @returns a `kHighsStatus` constant indicating whether the call succeeded
 */
HighsInt Highs_getColsByMask(const void* highs, const HighsInt* mask,
                             HighsInt* num_col, double* costs, double* lower,
                             double* upper, HighsInt* num_nz,
                             HighsInt* matrix_start, HighsInt* matrix_index,
                             double* matrix_value);

/**
 * Get data associated with multiple adjacent rows from the model.
 *
 * To query the constraint coefficients, this function should be called twice:
 *  - First, call this function with `matrix_start`, `matrix_index`, and
 *    `matrix_value` as `NULL`. This call will populate `num_nz` with the
 *    number of nonzero elements in the corresponding section of the constraint
 *    matrix.
 *  - Second, allocate new `matrix_index` and `matrix_value` arrays of length
 *    `num_nz` and call this function again to populate the new arrays with
 *    their contents.
 *
 * @param highs         a pointer to the Highs instance
 * @param from_row      the first row for which to query data for
 * @param to_row        the last row (inclusive) for which to query data for
 * @param num_row       an integer populated with the number of row got from the
 *                      model
 * @param lower         array of size [to_row - from_row + 1] for the row lower
 *                      bounds
 * @param upper         array of size [to_row - from_row + 1] for the row upper
 *                      bounds
 * @param num_nz        an integer populated with the number of non-zero
 *                      elements in the constraint matrix
 * @param matrix_start  array of size [to_row - from_row + 1] with the start
 *                      indices of each row in `matrix_index` and `matrix_value`
 * @param matrix_index  array of size [num_nz] with the column indices of each
 *                      element in the constraint matrix
 * @param matrix_value  array of size [num_nz] with the non-zero elements of the
 *                      constraint matrix.
 *
 * @returns a `kHighsStatus` constant indicating whether the call succeeded
 */
HighsInt Highs_getRowsByRange(const void* highs, const HighsInt from_row,
                              const HighsInt to_row, HighsInt* num_row,
                              double* lower, double* upper, HighsInt* num_nz,
                              HighsInt* matrix_start, HighsInt* matrix_index,
                              double* matrix_value);

/**
 * Get data associated with multiple rows given by an array.
 *
 * This function is identical to `Highs_getRowsByRange`, except for how the
 * rows are specified.
 *
 * @param num_set_indices   the number of indices in the set
 * @param set               array of size [num_set_entries] with the row indices
 *                          to get
 *
 * @returns a `kHighsStatus` constant indicating whether the call succeeded
 */
HighsInt Highs_getRowsBySet(const void* highs, const HighsInt num_set_entries,
                            const HighsInt* set, HighsInt* num_row,
                            double* lower, double* upper, HighsInt* num_nz,
                            HighsInt* matrix_start, HighsInt* matrix_index,
                            double* matrix_value);

/**
 * Get data associated with multiple rows given by a mask.
 *
 * This function is identical to `Highs_getRowsByRange`, except for how the
 * rows are specified.
 *
 * @param mask  array of length [num_row] containing a 1 to get the row and 0
 *              otherwise
 *
 * @returns a `kHighsStatus` constant indicating whether the call succeeded
 */
HighsInt Highs_getRowsByMask(const void* highs, const HighsInt* mask,
                             HighsInt* num_row, double* lower, double* upper,
                             HighsInt* num_nz, HighsInt* matrix_start,
                             HighsInt* matrix_index, double* matrix_value);

/**
 * Delete multiple adjacent columns.
 *
 * @param highs     a pointer to the Highs instance
 * @param from_col  the index of the first column to delete
 * @param to_col    the index of the last column to delete
 *
 * @returns a `kHighsStatus` constant indicating whether the call succeeded
 */
HighsInt Highs_deleteColsByRange(void* highs, const HighsInt from_col,
                                 const HighsInt to_col);

/**
 * Delete multiple columns given by an array of indices.
 *
 * @param highs             a pointer to the Highs instance
 * @param num_set_entries   the number of columns to delete
 * @param set               an array of size [num_set_entries] with the indices
 *                          of the columns to delete
 *
 * @returns a `kHighsStatus` constant indicating whether the call succeeded
 */
HighsInt Highs_deleteColsBySet(void* highs, const HighsInt num_set_entries,
                               const HighsInt* set);

/**
 * Delete multiple columns given by a mask.
 *
 * @param highs     a pointer to the Highs instance
 * @param mask      an array of length [num_col] with 1 if the column
 *                  should be deleted and 0 otherwise
 *
 * @returns a `kHighsStatus` constant indicating whether the call succeeded
 */
HighsInt Highs_deleteColsByMask(void* highs, HighsInt* mask);

/**
 * Delete multiple adjacent rows.
 *
 * @param highs     a pointer to the Highs instance
 * @param from_row  the index of the first row to delete
 * @param to_row    the index of the last row to delete
 *
 * @returns a `kHighsStatus` constant indicating whether the call succeeded
 */
HighsInt Highs_deleteRowsByRange(void* highs, const int from_row,
                                 const HighsInt to_row);

/**
 * Delete multiple rows given by an array of indices.
 *
 * @param highs             a pointer to the Highs instance
 * @param num_set_entries   the number of rows to delete
 * @param set               an array of size [num_set_entries] with the indices
 *                          of the rows to delete
 *
 * @returns a `kHighsStatus` constant indicating whether the call succeeded
 */
HighsInt Highs_deleteRowsBySet(void* highs, const HighsInt num_set_entries,
                               const HighsInt* set);

/**
 * Delete multiple rows given by a mask.
 *
 * @param highs     a pointer to the Highs instance
 * @param mask      an array of length [num_row] with 1 if the row should be
 *                  deleted and 0 otherwise. New index of any column not
 *                  deleted is returned in place of the value 0.
 *
 * @returns a `kHighsStatus` constant indicating whether the call succeeded
 */
HighsInt Highs_deleteRowsByMask(void* highs, HighsInt* mask);

/**
 * Scale a column by a constant.
 *
 * Scaling a column modifies the elements in the constraint matrix, the variable
 * bounds, and the objective coefficient.
 *
 * If scaleval < 0, the variable bounds flipped.
 *
 * @param highs     a pointer to the Highs instance
 * @param col       the index of the column to scale
 * @param scaleval  the value by which to scale the column
 *
 * @returns a `kHighsStatus` constant indicating whether the call succeeded
 */
HighsInt Highs_scaleCol(void* highs, const HighsInt col, const double scaleval);

/**
 * Scale a row by a constant.
 *
 * If scaleval < 0, the row bounds are flipped.
 *
 * @param highs     a pointer to the Highs instance
 * @param row       the index of the row to scale
 * @param scaleval  the value by which to scale the row
 *
 * @returns a `kHighsStatus` constant indicating whether the call succeeded
 */
HighsInt Highs_scaleRow(void* highs, const HighsInt row, const double scaleval);

/**
 * Return the value of infinity used by HiGHS.
 *
 * @param highs     a pointer to the Highs instance
 *
 * @returns the value of infinity used by HiGHS
 */
double Highs_getInfinity(const void* highs);

/**
 * Return the number of columns in the model.
 *
 * @param highs     a pointer to the Highs instance
 *
 * @returns the number of columns in the model
 */
HighsInt Highs_getNumCol(const void* highs);

/**
 * Return the number of rows in the model.
 *
 * @param highs     a pointer to the Highs instance
 *
 * @returns the number of rows in the model.
 */
HighsInt Highs_getNumRow(const void* highs);

/**
 * Return the number of nonzeros in the constraint matrix of the model.
 *
 * @param highs     a pointer to the Highs instance
 *
 * @returns the number of nonzeros in the constraint matrix of the model.
 */
HighsInt Highs_getNumNz(const void* highs);

/**
 * Return the number of nonzeroes in the Hessian matrix of the model.
 *
 * @param highs     a pointer to the Highs instance
 *
 * @returns the number of nonzeroes in the Hessian matrix of the model.
 */
HighsInt Highs_getHessianNumNz(const void* highs);

/**
 * Get the data from a HiGHS model.
 *
 * The input arguments have the same meaning (in a different order) to those
 * used in `Highs_passModel`.
 *
 * Note that all arrays must be pre-allocated to the correct size before calling
 * `Highs_getModel`. Use the following query methods to check the appropriate
 * size:
 *  - `Highs_getNumCol`
 *  - `Highs_getNumRow`
 *  - `Highs_getNumNz`
 *  - `Highs_getHessianNumNz`
 *
 * @returns a `kHighsStatus` constant indicating whether the call succeeded
 */
HighsInt Highs_getModel(const void* highs, const HighsInt a_format,
                        const HighsInt q_format, HighsInt* num_col,
                        HighsInt* num_row, HighsInt* num_nz,
                        HighsInt* hessian_num_nz, HighsInt* sense,
                        double* offset, double* col_cost, double* col_lower,
                        double* col_upper, double* row_lower, double* row_upper,
                        HighsInt* a_start, HighsInt* a_index, double* a_value,
                        HighsInt* q_start, HighsInt* q_index, double* q_value,
                        HighsInt* integrality);

/**
 * Set a primal (and possibly dual) solution as a starting point, then run
 * crossover to compute a basic feasible solution. If there is no dual solution,
 * pass col_dual and row_dual as nullptr.
 *
 * @param highs      a pointer to the Highs instance
 * @param num_col    the number of variables
 * @param num_row    the number of rows
 * @param col_value  array of length [num_col] with optimal primal solution for
 *                   each column
 * @param col_dual   array of length [num_col] with optimal dual solution for
 *                   each column
 * @param row_dual   array of length [num_row] with optimal dual solution for
 *                   each row
 *
 * @returns a `kHighsStatus` constant indicating whether the call succeeded
 */
HighsInt Highs_crossover(void* highs, const int num_col, const int num_row,
                         const double* col_value, const double* col_dual,
                         const double* row_dual);

/**
 * Releases all resources held by the global scheduler instance. It is
 * not thread-safe to call this function while calling Highs_run()/Highs_*call()
 * on any other Highs instance in any thread. After this function has terminated
 * it is guaranteed that eventually all previously created scheduler threads
 * will terminate and allocated memory will be released. After this function
 * has returned the option value for the number of threads may be altered to a
 * new value before the next call to Highs_run()/Highs_*call(). If the given
 * parameter has a nonzero value, then the function will not return until all
 * memory is freed, which might be desirable when debugging heap memory but
 * requires the calling thread to wait for all scheduler threads to wake-up
 * which is usually not necessary.
 *
 * @returns No status is returned since the function call cannot fail. Calling
 * this function while any Highs instance is in use on any thread is
 * undefined behavior and may cause crashes, but cannot be detected and hence
 * is fully in the callers responsibility.
 */
void Highs_resetGlobalScheduler(const HighsInt blocking);

// *********************
// * Deprecated methods*
// *********************

// These are deprecated because they don't follow the style guide. Constants
// must begin with `k`.
const HighsInt HighsStatuskError = -1;
const HighsInt HighsStatuskOk = 0;
const HighsInt HighsStatuskWarning = 1;

HighsInt Highs_call(const HighsInt num_col, const HighsInt num_row,
                    const HighsInt num_nz, const double* col_cost,
                    const double* col_lower, const double* col_upper,
                    const double* row_lower, const double* row_upper,
                    const HighsInt* a_start, const HighsInt* a_index,
                    const double* a_value, double* col_value, double* col_dual,
                    double* row_value, double* row_dual,
                    HighsInt* col_basis_status, HighsInt* row_basis_status,
                    HighsInt* model_status);

HighsInt Highs_runQuiet(void* highs);

HighsInt Highs_setHighsLogfile(void* highs, const void* logfile);

HighsInt Highs_setHighsOutput(void* highs, const void* outputfile);

HighsInt Highs_getIterationCount(const void* highs);

HighsInt Highs_getSimplexIterationCount(const void* highs);

HighsInt Highs_setHighsBoolOptionValue(void* highs, const char* option,
                                       const HighsInt value);

HighsInt Highs_setHighsIntOptionValue(void* highs, const char* option,
                                      const HighsInt value);

HighsInt Highs_setHighsDoubleOptionValue(void* highs, const char* option,
                                         const double value);

HighsInt Highs_setHighsStringOptionValue(void* highs, const char* option,
                                         const char* value);

HighsInt Highs_setHighsOptionValue(void* highs, const char* option,
                                   const char* value);

HighsInt Highs_getHighsBoolOptionValue(const void* highs, const char* option,
                                       HighsInt* value);

HighsInt Highs_getHighsIntOptionValue(const void* highs, const char* option,
                                      HighsInt* value);

HighsInt Highs_getHighsDoubleOptionValue(const void* highs, const char* option,
                                         double* value);

HighsInt Highs_getHighsStringOptionValue(const void* highs, const char* option,
                                         char* value);

HighsInt Highs_getHighsOptionType(const void* highs, const char* option,
                                  HighsInt* type);

HighsInt Highs_resetHighsOptions(void* highs);

HighsInt Highs_getHighsIntInfoValue(const void* highs, const char* info,
                                    HighsInt* value);

HighsInt Highs_getHighsDoubleInfoValue(const void* highs, const char* info,
                                       double* value);

HighsInt Highs_getNumCols(const void* highs);

HighsInt Highs_getNumRows(const void* highs);

double Highs_getHighsInfinity(const void* highs);

double Highs_getHighsRunTime(const void* highs);

HighsInt Highs_setOptionValue(void* highs, const char* option,
                              const char* value);

HighsInt Highs_getScaledModelStatus(const void* highs);

#ifdef __cplusplus
}
#endif

#endif
