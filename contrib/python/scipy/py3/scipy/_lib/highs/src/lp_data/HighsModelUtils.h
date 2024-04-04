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
/**@file lp_data/HighsModelUtils.h
 * @brief Class-independent utilities for HiGHS
 */
#ifndef LP_DATA_HIGHSMODELUTILS_H_
#define LP_DATA_HIGHSMODELUTILS_H_

//#include "Highs.h"
//#include "lp_data/HighsStatus.h"
#include "lp_data/HighsInfo.h"
#include "model/HighsModel.h"
//#include "lp_data/HStruct.h"
//#include "lp_data/HighsInfo.h"
//#include "lp_data/HighsLp.h"
//#include "lp_data/HighsOptions.h"

// Analyse lower and upper bounds of a model
void analyseModelBounds(const HighsLogOptions& log_options, const char* message,
                        HighsInt numBd, const std::vector<double>& lower,
                        const std::vector<double>& upper);
bool hasNamesWithSpaces(const HighsLogOptions& log_options,
                        const HighsInt num_name,
                        const std::vector<std::string>& names);
void writeModelBoundSolution(
    FILE* file, const bool columns, const HighsInt dim,
    const std::vector<double>& lower, const std::vector<double>& upper,
    const std::vector<std::string>& names, const bool have_primal,
    const std::vector<double>& primal, const bool have_dual,
    const std::vector<double>& dual, const bool have_basis,
    const std::vector<HighsBasisStatus>& status,
    const HighsVarType* integrality = NULL);
void writeModelSolution(FILE* file, const HighsLp& lp,
                        const HighsSolution& solution, const HighsInfo& info);

HighsInt maxNameLength(const HighsInt num_name,
                       const std::vector<std::string>& names);
HighsStatus normaliseNames(const HighsLogOptions& log_options,
                           const std::string name_type, const HighsInt num_name,
                           std::vector<std::string>& names,
                           HighsInt& max_name_length);

void writeSolutionFile(FILE* file, const HighsOptions& options,
                       const HighsModel& model, const HighsBasis& basis,
                       const HighsSolution& solution, const HighsInfo& info,
                       const HighsModelStatus model_status,
                       const HighsInt style);

void writeGlpsolCostRow(FILE* file, const bool raw, const bool is_mip,
                        const HighsInt row_id, const std::string objective_name,
                        const double objective_function_value);

void writeGlpsolSolution(FILE* file, const HighsOptions& options,
                         const HighsModel& model, const HighsBasis& basis,
                         const HighsSolution& solution,
                         const HighsModelStatus model_status,
                         const HighsInfo& info, const bool raw);

void writeOldRawSolution(FILE* file, const HighsLp& lp, const HighsBasis& basis,
                         const HighsSolution& solution);

HighsBasisStatus checkedVarHighsNonbasicStatus(
    const HighsBasisStatus ideal_status, const double lower,
    const double upper);

std::string utilModelStatusToString(const HighsModelStatus model_status);

std::string utilSolutionStatusToString(const HighsInt solution_status);

std::string utilBasisStatusToString(const HighsBasisStatus basis_status);

std::string utilBasisValidityToString(const HighsInt basis_validity);

HighsStatus highsStatusFromHighsModelStatus(HighsModelStatus model_status);

std::string statusToString(const HighsBasisStatus status, const double lower,
                           const double upper);
std::string typeToString(const HighsVarType type);

std::string findModelObjectiveName(const HighsLp* lp,
                                   const HighsHessian* hessian = nullptr);

#endif
