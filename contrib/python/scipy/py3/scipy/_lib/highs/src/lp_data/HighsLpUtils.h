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
/**@file lp_data/HighsLpUtils.h
 * @brief Class-independent utilities for HiGHS
 */
#ifndef LP_DATA_HIGHSLPUTILS_H_
#define LP_DATA_HIGHSLPUTILS_H_

#include <vector>

#include "lp_data/HConst.h"
#include "lp_data/HighsInfo.h"
#include "lp_data/HighsLp.h"
#include "lp_data/HighsStatus.h"
#include "util/HighsUtils.h"

// class HighsLp;
struct SimplexScale;
struct HighsBasis;
struct HighsSolution;
class HighsOptions;

using std::vector;

void writeBasisFile(FILE*& file, const HighsBasis& basis);

HighsStatus readBasisFile(const HighsLogOptions& log_options, HighsBasis& basis,
                          const std::string filename);
HighsStatus readBasisStream(const HighsLogOptions& log_options,
                            HighsBasis& basis, std::ifstream& in_file);

// Methods taking HighsLp as an argument
HighsStatus assessLp(HighsLp& lp, const HighsOptions& options);

bool lpDimensionsOk(std::string message, const HighsLp& lp,
                    const HighsLogOptions& log_options);

HighsStatus assessCosts(const HighsOptions& options, const HighsInt ml_col_os,
                        const HighsIndexCollection& index_collection,
                        vector<double>& cost, const double infinite_cost);

HighsStatus assessBounds(const HighsOptions& options, const char* type,
                         const HighsInt ml_ix_os,
                         const HighsIndexCollection& index_collection,
                         vector<double>& lower, vector<double>& upper,
                         const double infinite_bound);

HighsStatus cleanBounds(const HighsOptions& options, HighsLp& lp);

HighsStatus assessIntegrality(HighsLp& lp, const HighsOptions& options);
bool activeModifiedUpperBounds(const HighsOptions& options, const HighsLp& lp,
                               const std::vector<double> col_value);

bool considerScaling(const HighsOptions& options, HighsLp& lp);
void scaleLp(const HighsOptions& options, HighsLp& lp);
bool equilibrationScaleMatrix(const HighsOptions& options, HighsLp& lp,
                              const HighsInt use_scale_strategy);
bool maxValueScaleMatrix(const HighsOptions& options, HighsLp& lp,
                         const HighsInt use_scale_strategy);

HighsStatus applyScalingToLpCol(HighsLp& lp, const HighsInt col,
                                const double colScale);

HighsStatus applyScalingToLpRow(HighsLp& lp, const HighsInt row,
                                const double rowScale);

void appendColsToLpVectors(HighsLp& lp, const HighsInt num_new_col,
                           const vector<double>& colCost,
                           const vector<double>& colLower,
                           const vector<double>& colUpper);

void appendRowsToLpVectors(HighsLp& lp, const HighsInt num_new_row,
                           const vector<double>& rowLower,
                           const vector<double>& rowUpper);

void deleteLpCols(HighsLp& lp, const HighsIndexCollection& index_collection);

void deleteColsFromLpVectors(HighsLp& lp, HighsInt& new_num_col,
                             const HighsIndexCollection& index_collection);

void deleteLpRows(HighsLp& lp, const HighsIndexCollection& index_collection);

void deleteRowsFromLpVectors(HighsLp& lp, HighsInt& new_num_row,
                             const HighsIndexCollection& index_collection);

void deleteScale(vector<double>& scale,
                 const HighsIndexCollection& index_collection);

void changeLpMatrixCoefficient(HighsLp& lp, const HighsInt row,
                               const HighsInt col, const double new_value,
                               const bool zero_new_value);

void changeLpIntegrality(HighsLp& lp,
                         const HighsIndexCollection& index_collection,
                         const vector<HighsVarType>& new_integrality);

void changeLpCosts(HighsLp& lp, const HighsIndexCollection& index_collection,
                   const vector<double>& new_col_cost);

void changeLpColBounds(HighsLp& lp,
                       const HighsIndexCollection& index_collection,
                       const vector<double>& new_col_lower,
                       const vector<double>& new_col_upper);

void changeLpRowBounds(HighsLp& lp,
                       const HighsIndexCollection& index_collection,
                       const vector<double>& new_row_lower,
                       const vector<double>& new_row_upper);

void changeBounds(vector<double>& lower, vector<double>& upper,
                  const HighsIndexCollection& index_collection,
                  const vector<double>& new_lower,
                  const vector<double>& new_upper);

/**
 * @brief Report the data of an LP
 */
void reportLp(const HighsLogOptions& log_options,
              const HighsLp& lp,  //!< LP whose data are to be reported
              const HighsLogType report_level = HighsLogType::kInfo
              //!< INFO => scalar [dimensions];
              //!< DETAILED => vector[costs/bounds];
              //!< VERBOSE => vector+matrix
);
/**
 * @brief Report the brief data of an LP
 */
void reportLpBrief(const HighsLogOptions& log_options,
                   const HighsLp& lp  //!< LP whose data are to be reported
);
/**
 * @brief Report the data of an LP
 */
void reportLpDimensions(const HighsLogOptions& log_options,
                        const HighsLp& lp  //!< LP whose data are to be reported
);
/**
 * @brief Report the data of an LP
 */
void reportLpObjSense(const HighsLogOptions& log_options,
                      const HighsLp& lp  //!< LP whose data are to be reported
);
/**
 * @brief Report the data of an LP
 */
void reportLpColVectors(const HighsLogOptions& log_options,
                        const HighsLp& lp  //!< LP whose data are to be reported
);
/**
 * @brief Report the data of an LP
 */
void reportLpRowVectors(const HighsLogOptions& log_options,
                        const HighsLp& lp  //!< LP whose data are to be reported
);
/**
 * @brief Report the data of an LP
 */
void reportLpColMatrix(const HighsLogOptions& log_options,
                       const HighsLp& lp  //!< LP whose data are to be reported
);

void reportMatrix(const HighsLogOptions& log_options, const std::string message,
                  const HighsInt num_col, const HighsInt num_nz,
                  const HighsInt* start, const HighsInt* index,
                  const double* value);

// Get the number of integer-valued columns in the LP
HighsInt getNumInt(const HighsLp& lp);

// Get the costs for a contiguous set of columns
void getLpCosts(const HighsLp& lp, const HighsInt from_col,
                const HighsInt to_col, double* XcolCost);

// Get the bounds for a contiguous set of columns
void getLpColBounds(const HighsLp& lp, const HighsInt from_col,
                    const HighsInt to_col, double* XcolLower,
                    double* XcolUpper);

// Get the bounds for a contiguous set of rows
void getLpRowBounds(const HighsLp& lp, const HighsInt from_row,
                    const HighsInt to_row, double* XrowLower,
                    double* XrowUpper);

void getLpMatrixCoefficient(const HighsLp& lp, const HighsInt row,
                            const HighsInt col, double* val);
// Analyse the data in an LP problem
void analyseLp(const HighsLogOptions& log_options, const HighsLp& lp);

HighsStatus readSolutionFile(const std::string filename,
                             const HighsOptions& options, const HighsLp& lp,
                             HighsBasis& basis, HighsSolution& solution,
                             const HighsInt style);

void checkLpSolutionFeasibility(const HighsOptions& options, const HighsLp& lp,
                                const HighsSolution& solution);

HighsStatus calculateRowValues(const HighsLp& lp, HighsSolution& solution);
HighsStatus calculateRowValuesQuad(const HighsLp& lp, HighsSolution& solution);
HighsStatus calculateColDuals(const HighsLp& lp, HighsSolution& solution);

bool isBoundInfeasible(const HighsLogOptions& log_options, const HighsLp& lp);

bool isColDataNull(const HighsLogOptions& log_options,
                   const double* usr_col_cost, const double* usr_col_lower,
                   const double* usr_col_upper);
bool isRowDataNull(const HighsLogOptions& log_options,
                   const double* usr_row_lower, const double* usr_row_upper);
bool isMatrixDataNull(const HighsLogOptions& log_options,
                      const HighsInt* usr_matrix_start,
                      const HighsInt* usr_matrix_index,
                      const double* usr_matrix_value);

void reportPresolveReductions(const HighsLogOptions& log_options,
                              const HighsLp& lp, const HighsLp& presolve_lp);

void reportPresolveReductions(const HighsLogOptions& log_options,
                              const HighsLp& lp, const bool presolve_to_empty);

bool isLessInfeasibleDSECandidate(const HighsLogOptions& log_options,
                                  const HighsLp& lp);

HighsLp withoutSemiVariables(const HighsLp& lp, HighsSolution& solution,
                             const double primal_feasibility_tolerance);

void removeRowsOfCountOne(const HighsLogOptions& log_options, HighsLp& lp);

#endif  // LP_DATA_HIGHSLPUTILS_H_
