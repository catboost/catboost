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
/**@file lp_data/HConst.h
 * @brief Constants for HiGHS
 */
#ifndef LP_DATA_HCONST_H_
#define LP_DATA_HCONST_H_

#include <cmath>
#include <limits>
#include <string>

#include "util/HighsInt.h"

const HighsInt kHighsIInf = std::numeric_limits<HighsInt>::max();
const double kHighsInf = std::numeric_limits<double>::infinity();
const double kHighsTiny = 1e-14;
const double kHighsMacheps = std::ldexp(1, -52);
const double kHighsZero = 1e-50;
const std::string kHighsOffString = "off";
const std::string kHighsChooseString = "choose";
const std::string kHighsOnString = "on";
const HighsInt kSimplexConcurrencyLimit = 8;
const double kRunningAverageMultiplier = 0.05;

const bool kExtendInvertWhenAddingRows = false;

enum SimplexScaleStrategy {
  kSimplexScaleStrategyMin = 0,
  kSimplexScaleStrategyOff = kSimplexScaleStrategyMin,  // 0
  kSimplexScaleStrategyChoose,                          // 1
  kSimplexScaleStrategyEquilibration,                   // 2
  kSimplexScaleStrategyForcedEquilibration,             // 3
  kSimplexScaleStrategyMaxValue015,                     // 4
  kSimplexScaleStrategyMaxValue0157,                    // 5
  kSimplexScaleStrategyMax = kSimplexScaleStrategyMaxValue0157
};

enum HighsDebugLevel {
  kHighsDebugLevelNone = 0,
  kHighsDebugLevelCheap,
  kHighsDebugLevelCostly,
  kHighsDebugLevelExpensive,
  kHighsDebugLevelMin = kHighsDebugLevelNone,
  kHighsDebugLevelMax = kHighsDebugLevelExpensive
};

enum class HighsDebugStatus {
  kNotChecked = -1,
  kOk,
  kSmallError,
  kWarning,
  kLargeError,
  kError,
  kExcessiveError,
  kLogicalError,
};

enum HighsAnalysisLevel {
  kHighsAnalysisLevelNone = 0,
  kHighsAnalysisLevelModelData = 1,
  kHighsAnalysisLevelSolverSummaryData = 2,
  kHighsAnalysisLevelSolverRuntimeData = 4,
  kHighsAnalysisLevelSolverTime = 8,
  kHighsAnalysisLevelNlaData = 16,
  kHighsAnalysisLevelNlaTime = 32,
  kHighsAnalysisLevelMin = kHighsAnalysisLevelNone,
  kHighsAnalysisLevelMax =
      kHighsAnalysisLevelModelData + kHighsAnalysisLevelSolverSummaryData +
      kHighsAnalysisLevelSolverRuntimeData + kHighsAnalysisLevelSolverTime +
      kHighsAnalysisLevelNlaData + kHighsAnalysisLevelNlaTime
};

enum class HighsVarType : uint8_t {
  kContinuous = 0,
  kInteger = 1,
  kSemiContinuous = 2,
  kSemiInteger = 3,
  kImplicitInteger = 4,
};

enum class HighsOptionType { kBool = 0, kInt, kDouble, kString };

enum class HighsInfoType { kInt64 = -1, kInt = 1, kDouble };

enum OptionOffChooseOn {
  kHighsOptionOff = -1,
  kHighsOptionChoose,
  kHighsOptionOn
};

/** SCIP/HiGHS Objective sense */
enum class ObjSense { kMinimize = 1, kMaximize = -1 };

enum class MatrixFormat { kColwise = 1, kRowwise, kRowwisePartitioned };

enum class HessianFormat { kTriangular = 1, kSquare };

enum SolutionStatus {
  kSolutionStatusNone = 0,
  kSolutionStatusInfeasible,
  kSolutionStatusFeasible,
  kSolutionStatusMin = kSolutionStatusNone,
  kSolutionStatusMax = kSolutionStatusFeasible
};

enum BasisValidity {
  kBasisValidityInvalid = 0,
  kBasisValidityValid,
  kBasisValidityMin = kBasisValidityInvalid,
  kBasisValidityMax = kBasisValidityValid
};

enum SolutionStyle {
  kSolutionStyleOldRaw = -1,
  kSolutionStyleRaw = 0,
  kSolutionStylePretty,        // 1;
  kSolutionStyleGlpsolRaw,     // 2;
  kSolutionStyleGlpsolPretty,  // 3;
  kSolutionStyleMin = kSolutionStyleOldRaw,
  kSolutionStyleMax = kSolutionStyleGlpsolPretty
};

enum GlpsolCostRowLocation {
  kGlpsolCostRowLocationLast = -2,
  kGlpsolCostRowLocationNone,         // -1
  kGlpsolCostRowLocationNoneIfEmpty,  // 0
  kGlpsolCostRowLocationMin = kGlpsolCostRowLocationLast
};

const std::string kHighsFilenameDefault = "";

// Need to allow infinite costs to pass SCIP LPI unit tests
const bool kHighsAllowInfiniteCosts = true;

enum class HighsPresolveStatus {
  kNotPresolved = -1,
  kNotReduced,
  kInfeasible,
  kUnboundedOrInfeasible,
  kReduced,
  kReducedToEmpty,
  kTimeout,
  kNullError,
  kOptionsError,
};

enum class HighsModelStatus {
  // NB Add new status values to the end so that int cast of status
  // values is unchanged, since enums are not preserved in some
  // interfaces
  kNotset = 0,
  kLoadError,
  kModelError,
  kPresolveError,
  kSolveError,
  kPostsolveError,
  kModelEmpty,
  kOptimal,
  kInfeasible,
  kUnboundedOrInfeasible,
  kUnbounded,
  kObjectiveBound,
  kObjectiveTarget,
  kTimeLimit,
  kIterationLimit,
  kUnknown,
  kMin = kNotset,
  kMax = kUnknown
};

/** SCIP/CPLEX-like HiGHS basis status for columns and rows. */
enum class HighsBasisStatus : uint8_t {
  kLower =
      0,   // (slack) variable is at its lower bound [including fixed variables]
  kBasic,  // (slack) variable is basic
  kUpper,  // (slack) variable is at its upper bound
  kZero,   // free variable is non-basic and set to zero
  kNonbasic  // nonbasic with no specific bound information - useful for users
             // and postsolve
};

// Default and max allowed power-of-two matrix scale factor
const HighsInt kDefaultAllowedMatrixPow2Scale = 20;
const HighsInt kMaxAllowedMatrixPow2Scale = 30;

// Illegal values of num/max/sum infeasibility - used to indicate that true
// values aren't known
const double kHighsIllegalInfeasibilityMeasure = kHighsInf;
const HighsInt kHighsIllegalInfeasibilityCount = -1;

// Illegal values for HighsError - used to indicate that true
// values aren't known
const double kHighsIllegalErrorValue = kHighsInf;
const HighsInt kHighsIllegalErrorIndex = -1;

// Maximum upper bound on semi-variables
const double kMaxSemiVariableUpper = 1e5;

// Termination link in linked lists
const HighsInt kNoLink = -1;

const int8_t kPivotIllegal = -1;
const int8_t kPivotLogical = 0;
const int8_t kPivotUnit = 1;
const int8_t kPivotRowSingleton = 2;
const int8_t kPivotColSingleton = 3;
const int8_t kPivotMarkowitz = 4;
#endif /* LP_DATA_HCONST_H_ */
