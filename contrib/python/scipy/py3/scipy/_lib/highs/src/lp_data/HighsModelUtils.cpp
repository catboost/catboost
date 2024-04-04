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
/**@file lp_data/HighsUtils.cpp
 * @brief Class-independent utilities for HiGHS
 */

#include "lp_data/HighsModelUtils.h"

#include <algorithm>
#include <cfloat>
#include <sstream>
#include <vector>

//#include "model/HighsModel.h"
//#include "HConfig.h"
//#include "io/HighsIO.h"
//#include "lp_data/HConst.h"
#include "lp_data/HighsSolution.h"
#include "util/stringutil.h"

const double kHighsDoubleTolerance = 1e-13;
const double kGlpsolDoubleTolerance = 1e-12;

void analyseModelBounds(const HighsLogOptions& log_options, const char* message,
                        HighsInt numBd, const std::vector<double>& lower,
                        const std::vector<double>& upper) {
  if (numBd == 0) return;
  HighsInt numFr = 0;
  HighsInt numLb = 0;
  HighsInt numUb = 0;
  HighsInt numBx = 0;
  HighsInt numFx = 0;
  for (HighsInt ix = 0; ix < numBd; ix++) {
    if (highs_isInfinity(-lower[ix])) {
      // Infinite lower bound
      if (highs_isInfinity(upper[ix])) {
        // Infinite lower bound and infinite upper bound: Fr
        numFr++;
      } else {
        // Infinite lower bound and   finite upper bound: Ub
        numUb++;
      }
    } else {
      // Finite lower bound
      if (highs_isInfinity(upper[ix])) {
        // Finite lower bound and infinite upper bound: Lb
        numLb++;
      } else {
        // Finite lower bound and   finite upper bound:
        if (lower[ix] < upper[ix]) {
          // Distinct finite bounds: Bx
          numBx++;
        } else {
          // Equal finite bounds: Fx
          numFx++;
        }
      }
    }
  }
  highsLogDev(log_options, HighsLogType::kInfo,
              "Analysing %" HIGHSINT_FORMAT " %s bounds\n", numBd, message);
  if (numFr > 0)
    highsLogDev(log_options, HighsLogType::kInfo,
                "   Free:  %7" HIGHSINT_FORMAT " (%3" HIGHSINT_FORMAT "%%)\n",
                numFr, (100 * numFr) / numBd);
  if (numLb > 0)
    highsLogDev(log_options, HighsLogType::kInfo,
                "   LB:    %7" HIGHSINT_FORMAT " (%3" HIGHSINT_FORMAT "%%)\n",
                numLb, (100 * numLb) / numBd);
  if (numUb > 0)
    highsLogDev(log_options, HighsLogType::kInfo,
                "   UB:    %7" HIGHSINT_FORMAT " (%3" HIGHSINT_FORMAT "%%)\n",
                numUb, (100 * numUb) / numBd);
  if (numBx > 0)
    highsLogDev(log_options, HighsLogType::kInfo,
                "   Boxed: %7" HIGHSINT_FORMAT " (%3" HIGHSINT_FORMAT "%%)\n",
                numBx, (100 * numBx) / numBd);
  if (numFx > 0)
    highsLogDev(log_options, HighsLogType::kInfo,
                "   Fixed: %7" HIGHSINT_FORMAT " (%3" HIGHSINT_FORMAT "%%)\n",
                numFx, (100 * numFx) / numBd);
  highsLogDev(log_options, HighsLogType::kInfo,
              "grep_CharMl,%s,Free,LB,UB,Boxed,Fixed\n", message);
  highsLogDev(log_options, HighsLogType::kInfo,
              "grep_CharMl,%" HIGHSINT_FORMAT ",%" HIGHSINT_FORMAT
              ",%" HIGHSINT_FORMAT ",%" HIGHSINT_FORMAT ",%" HIGHSINT_FORMAT
              ",%" HIGHSINT_FORMAT "\n",
              numBd, numFr, numLb, numUb, numBx, numFx);
}

std::string statusToString(const HighsBasisStatus status, const double lower,
                           const double upper) {
  switch (status) {
    case HighsBasisStatus::kLower:
      if (lower == upper) {
        return "FX";
      } else {
        return "LB";
      }
      break;
    case HighsBasisStatus::kBasic:
      return "BS";
      break;
    case HighsBasisStatus::kUpper:
      return "UB";
      break;
    case HighsBasisStatus::kZero:
      return "FR";
      break;
    case HighsBasisStatus::kNonbasic:
      return "NB";
      break;
  }
  return "";
}

std::string typeToString(const HighsVarType type) {
  switch (type) {
    case HighsVarType::kContinuous:
      return "Continuous";
    case HighsVarType::kInteger:
      return "Integer   ";
    case HighsVarType::kSemiContinuous:
      return "Semi-conts";
    case HighsVarType::kSemiInteger:
      return "Semi-int  ";
    case HighsVarType::kImplicitInteger:
      return "ImpliedInt";
  }
  return "";
}

void writeModelBoundSolution(
    FILE* file, const bool columns, const HighsInt dim,
    const std::vector<double>& lower, const std::vector<double>& upper,
    const std::vector<std::string>& names, const bool have_primal,
    const std::vector<double>& primal, const bool have_dual,
    const std::vector<double>& dual, const bool have_basis,
    const std::vector<HighsBasisStatus>& status,
    const HighsVarType* integrality) {
  const bool have_names = names.size() > 0;
  if (have_names) assert((int)names.size() >= dim);
  if (have_primal) assert((int)primal.size() >= dim);
  if (have_dual) assert((int)dual.size() >= dim);
  if (have_basis) assert((int)status.size() >= dim);
  const bool have_integrality = integrality != NULL;
  std::string var_status_string;
  if (columns) {
    fprintf(file, "Columns\n");
  } else {
    fprintf(file, "Rows\n");
  }
  fprintf(
      file,
      "    Index Status        Lower        Upper       Primal         Dual");
  if (have_integrality) fprintf(file, "  Type      ");
  if (have_names) {
    fprintf(file, "  Name\n");
  } else {
    fprintf(file, "\n");
  }
  for (HighsInt ix = 0; ix < dim; ix++) {
    if (have_basis) {
      var_status_string = statusToString(status[ix], lower[ix], upper[ix]);
    } else {
      var_status_string = "";
    }
    fprintf(file, "%9" HIGHSINT_FORMAT "   %4s %12g %12g", ix,
            var_status_string.c_str(), lower[ix], upper[ix]);
    if (have_primal) {
      fprintf(file, " %12g", primal[ix]);
    } else {
      fprintf(file, "             ");
    }
    if (have_dual) {
      fprintf(file, " %12g", dual[ix]);
    } else {
      fprintf(file, "             ");
    }
    if (have_integrality)
      fprintf(file, "  %s", typeToString(integrality[ix]).c_str());
    if (have_names) {
      fprintf(file, "  %-s\n", names[ix].c_str());
    } else {
      fprintf(file, "\n");
    }
  }
}

void writeModelSolution(FILE* file, const HighsLp& lp,
                        const HighsSolution& solution, const HighsInfo& info) {
  const bool have_col_names = lp.col_names_.size() > 0;
  const bool have_row_names = lp.row_names_.size() > 0;
  const bool have_primal = solution.value_valid;
  const bool have_dual = solution.dual_valid;
  std::stringstream ss;
  if (have_col_names) assert((int)lp.col_names_.size() >= lp.num_col_);
  if (have_row_names) assert((int)lp.row_names_.size() >= lp.num_row_);
  if (have_primal) {
    assert((int)solution.col_value.size() >= lp.num_col_);
    assert((int)solution.row_value.size() >= lp.num_row_);
    assert(info.primal_solution_status != kSolutionStatusNone);
  }
  if (have_dual) {
    assert((int)solution.col_dual.size() >= lp.num_col_);
    assert((int)solution.row_dual.size() >= lp.num_row_);
    assert(info.dual_solution_status != kSolutionStatusNone);
  }
  fprintf(file, "\n# Primal solution values\n");
  if (!have_primal || info.primal_solution_status == kSolutionStatusNone) {
    fprintf(file, "None\n");
  } else {
    if (info.primal_solution_status == kSolutionStatusFeasible) {
      fprintf(file, "Feasible\n");
    } else {
      assert(info.primal_solution_status == kSolutionStatusInfeasible);
      fprintf(file, "Infeasible\n");
    }
    HighsCDouble objective_function_value = lp.offset_;
    for (HighsInt i = 0; i < lp.num_col_; ++i)
      objective_function_value += lp.col_cost_[i] * solution.col_value[i];
    std::array<char, 32> objStr = highsDoubleToString(
        (double)objective_function_value, kHighsDoubleTolerance);
    fprintf(file, "Objective %s\n", objStr.data());
    fprintf(file, "# Columns %" HIGHSINT_FORMAT "\n", lp.num_col_);
    for (HighsInt ix = 0; ix < lp.num_col_; ix++) {
      std::array<char, 32> valStr =
          highsDoubleToString(solution.col_value[ix], kHighsDoubleTolerance);
      // Create a column name
      ss.str(std::string());
      ss << "C" << ix;
      const std::string name = have_col_names ? lp.col_names_[ix] : ss.str();
      fprintf(file, "%-s %s\n", name.c_str(), valStr.data());
    }
    fprintf(file, "# Rows %" HIGHSINT_FORMAT "\n", lp.num_row_);
    for (HighsInt ix = 0; ix < lp.num_row_; ix++) {
      std::array<char, 32> valStr =
          highsDoubleToString(solution.row_value[ix], kHighsDoubleTolerance);
      // Create a row name
      ss.str(std::string());
      ss << "R" << ix;
      const std::string name = have_row_names ? lp.row_names_[ix] : ss.str();
      fprintf(file, "%-s %s\n", name.c_str(), valStr.data());
    }
  }
  fprintf(file, "\n# Dual solution values\n");
  if (!have_dual || info.dual_solution_status == kSolutionStatusNone) {
    fprintf(file, "None\n");
  } else {
    if (info.dual_solution_status == kSolutionStatusFeasible) {
      fprintf(file, "Feasible\n");
    } else {
      assert(info.dual_solution_status == kSolutionStatusInfeasible);
      fprintf(file, "Infeasible\n");
    }
    fprintf(file, "# Columns %" HIGHSINT_FORMAT "\n", lp.num_col_);
    for (HighsInt ix = 0; ix < lp.num_col_; ix++) {
      std::array<char, 32> valStr =
          highsDoubleToString(solution.col_dual[ix], kHighsDoubleTolerance);
      ss.str(std::string());
      ss << "C" << ix;
      const std::string name = have_col_names ? lp.col_names_[ix] : ss.str();
      fprintf(file, "%-s %s\n", name.c_str(), valStr.data());
    }
    fprintf(file, "# Rows %" HIGHSINT_FORMAT "\n", lp.num_row_);
    for (HighsInt ix = 0; ix < lp.num_row_; ix++) {
      std::array<char, 32> valStr =
          highsDoubleToString(solution.row_dual[ix], kHighsDoubleTolerance);
      ss.str(std::string());
      ss << "R" << ix;
      const std::string name = have_row_names ? lp.row_names_[ix] : ss.str();
      fprintf(file, "%-s %s\n", name.c_str(), valStr.data());
    }
  }
}

bool hasNamesWithSpaces(const HighsLogOptions& log_options,
                        const HighsInt num_name,
                        const std::vector<std::string>& names) {
  HighsInt num_names_with_spaces = 0;
  for (HighsInt ix = 0; ix < num_name; ix++) {
    HighsInt space_pos = names[ix].find(" ");
    if (space_pos >= 0) {
      if (num_names_with_spaces == 0) {
        highsLogDev(
            log_options, HighsLogType::kInfo,
            "Name |%s| contains a space character in position %" HIGHSINT_FORMAT
            "\n",
            names[ix].c_str(), space_pos);
        num_names_with_spaces++;
      }
    }
  }
  if (num_names_with_spaces)
    highsLogDev(log_options, HighsLogType::kInfo,
                "There are %" HIGHSINT_FORMAT " names with spaces\n",
                num_names_with_spaces);
  return num_names_with_spaces > 0;
}

HighsInt maxNameLength(const HighsInt num_name,
                       const std::vector<std::string>& names) {
  HighsInt max_name_length = 0;
  for (HighsInt ix = 0; ix < num_name; ix++)
    max_name_length = std::max((HighsInt)names[ix].length(), max_name_length);
  return max_name_length;
}

HighsStatus normaliseNames(const HighsLogOptions& log_options,
                           const std::string name_type, const HighsInt num_name,
                           std::vector<std::string>& names,
                           HighsInt& max_name_length) {
  // Record the desired maximum name length
  HighsInt desired_max_name_length = max_name_length;
  // First look for empty names
  HighsInt num_empty_name = 0;
  std::string name_prefix = name_type.substr(0, 1);
  bool names_with_spaces = false;
  for (HighsInt ix = 0; ix < num_name; ix++) {
    if ((HighsInt)names[ix].length() == 0) num_empty_name++;
  }
  // If there are no empty names - in which case they will all be
  // replaced - find the maximum name length
  if (!num_empty_name) max_name_length = maxNameLength(num_name, names);
  bool construct_names =
      num_empty_name || max_name_length > desired_max_name_length;
  if (construct_names) {
    // Construct names, either because they are empty names, or
    // because the existing names are too long

    highsLogUser(log_options, HighsLogType::kWarning,
                 "There are empty or excessively-long %s names: using "
                 "constructed names with prefix \"%s\"\n",
                 name_type.c_str(), name_prefix.c_str());
    for (HighsInt ix = 0; ix < num_name; ix++)
      names[ix] = name_prefix + std::to_string(ix);
  } else {
    // Using original names, so look to see whether there are names with spaces
    names_with_spaces = hasNamesWithSpaces(log_options, num_name, names);
  }
  // Find the final maximum name length
  max_name_length = maxNameLength(num_name, names);
  // Can't have names with spaces and more than 8 characters
  if (max_name_length > 8 && names_with_spaces) return HighsStatus::kError;
  if (construct_names) return HighsStatus::kWarning;
  return HighsStatus::kOk;
}

void writeSolutionFile(FILE* file, const HighsOptions& options,
                       const HighsModel& model, const HighsBasis& basis,
                       const HighsSolution& solution, const HighsInfo& info,
                       const HighsModelStatus model_status,
                       const HighsInt style) {
  const bool have_primal = solution.value_valid;
  const bool have_dual = solution.dual_valid;
  const bool have_basis = basis.valid;
  const HighsLp& lp = model.lp_;
  if (style == kSolutionStyleOldRaw) {
    writeOldRawSolution(file, lp, basis, solution);
  } else if (style == kSolutionStylePretty) {
    const HighsVarType* integrality_ptr =
        lp.integrality_.size() > 0 ? &lp.integrality_[0] : NULL;
    writeModelBoundSolution(file, true, lp.num_col_, lp.col_lower_,
                            lp.col_upper_, lp.col_names_, have_primal,
                            solution.col_value, have_dual, solution.col_dual,
                            have_basis, basis.col_status, integrality_ptr);
    writeModelBoundSolution(file, false, lp.num_row_, lp.row_lower_,
                            lp.row_upper_, lp.row_names_, have_primal,
                            solution.row_value, have_dual, solution.row_dual,
                            have_basis, basis.row_status);
    fprintf(file, "\nModel status: %s\n",
            utilModelStatusToString(model_status).c_str());
    std::array<char, 32> objStr = highsDoubleToString(
        (double)info.objective_function_value, kHighsDoubleTolerance);
    fprintf(file, "\nObjective value: %s\n", objStr.data());
  } else if (style == kSolutionStyleGlpsolRaw ||
             style == kSolutionStyleGlpsolPretty) {
    const bool raw = style == kSolutionStyleGlpsolRaw;
    writeGlpsolSolution(file, options, model, basis, solution, model_status,
                        info, raw);
  } else {
    fprintf(file, "Model status\n");
    fprintf(file, "%s\n", utilModelStatusToString(model_status).c_str());
    writeModelSolution(file, lp, solution, info);
  }
}

void writeGlpsolCostRow(FILE* file, const bool raw, const bool is_mip,
                        const HighsInt row_id, const std::string objective_name,
                        const double objective_function_value) {
  if (raw) {
    double double_value = objective_function_value;
    std::array<char, 32> double_string =
        highsDoubleToString(double_value, kGlpsolDoubleTolerance);
    // Last term of 0 for dual should (also) be blank when not MIP
    fprintf(file, "i %d %s%s%s\n", (int)row_id, is_mip ? "" : "b ",
            double_string.data(), is_mip ? "" : " 0");
  } else {
    fprintf(file, "%6d ", (int)row_id);
    if (objective_name.length() <= 12) {
      fprintf(file, "%-12s ", objective_name.c_str());
    } else {
      fprintf(file, "%s\n%20s", objective_name.c_str(), "");
    }
    if (is_mip) {
      fprintf(file, "   ");
    } else {
      fprintf(file, "B  ");
    }
    fprintf(file, "%13.6g %13s %13s \n", objective_function_value, "", "");
  }
}

void writeGlpsolSolution(FILE* file, const HighsOptions& options,
                         const HighsModel& model, const HighsBasis& basis,
                         const HighsSolution& solution,
                         const HighsModelStatus model_status,
                         const HighsInfo& info, const bool raw) {
  const bool have_value = solution.value_valid;
  const bool have_dual = solution.dual_valid;
  const bool have_basis = basis.valid;
  if (!have_value) return;
  const double kGlpsolHighQuality = 1e-9;
  const double kGlpsolMediumQuality = 1e-6;
  const double kGlpsolLowQuality = 1e-3;
  const double kGlpsolPrintAsZero = 1e-9;
  const HighsLp& lp = model.lp_;
  const bool have_col_names = lp.col_names_.size();
  const bool have_row_names = lp.row_names_.size();
  // Determine number of nonzeros including the objective function
  // and, hence, determine whether there is an objective function
  HighsInt num_nz = lp.a_matrix_.numNz();
  for (HighsInt iCol = 0; iCol < lp.num_col_; iCol++)
    if (lp.col_cost_[iCol]) num_nz++;
  const bool empty_cost_row = num_nz == lp.a_matrix_.numNz();
  const bool has_objective = !empty_cost_row || model.hessian_.dim_;
  // When writing out the row information (and hence the number of
  // rows and nonzeros), the case of the cost row is tricky
  // (particularly if it's empty) if HiGHS is to be able to reproduce
  // the (inconsistent) behaviour of Glpsol.
  //
  // If Glpsol is run from a .mod file then the cost row is reported
  // unless there is no objecive [minimize/maximize "objname"]
  // statement in the .mod file. In this case, the N-row in the MPS
  // file is called "R0000000" and referred to below as being artificial.
  //
  // However, the position of a defined cost row depends on where the
  // objecive appears in the .mod file. If Glpsol is run from a .mod
  // file, and reads a .sol file, it must be in the right format.
  //
  // HiGHS can't read ..mod files, so works from an MPS or LP file
  // generated by glpsol.
  //
  // An MPS file generated by glpsol will have the cost row in the
  // same position as it was in the .mod file
  //
  // An LP file generated by glpsol will have the objective defined
  // first, so the desired position of the cost row in the .sol file
  // is unavailable. The only option with this route is to define the
  // cost row location "by hand" using glpsol_cost_row_location
  //
  // If Glpsol is run from an LP or MPS file then the cost row is not
  // reported. This behaviour is defined by setting
  // glpsol_cost_row_location = -1;
  //
  // This inconsistent behaviour means that it must be possible to
  // tell HiGHS to suppress the cost row
  //
  const HighsInt cost_row_option = options.glpsol_cost_row_location;
  // Define cost_row_location
  //
  // It is indexed from 1 so that it matches the index printed on that
  // row...
  //
  // ... hence a location of zero means that the cost row isn't
  // reported
  HighsInt cost_row_location = 0;
  std::string artificial_cost_row_name = "R0000000";
  const bool artificial_cost_row =
      lp.objective_name_ == artificial_cost_row_name;
  if (artificial_cost_row)
    highsLogUser(options.log_options, HighsLogType::kWarning,
                 "The cost row name of \"%s\" is assumed to be artificial and "
                 "will not be reported in the Glpsol solution file\n",
                 lp.objective_name_.c_str());

  if (cost_row_option <= kGlpsolCostRowLocationLast ||
      cost_row_option > lp.num_row_) {
    // Place the cost row last
    cost_row_location = lp.num_row_ + 1;
  } else if (cost_row_option == kGlpsolCostRowLocationNone) {
    // Don't report the cost row
    assert(cost_row_location == 0);
  } else if (cost_row_option == kGlpsolCostRowLocationNoneIfEmpty) {
    // This option allows the cost row to be omitted if it's empty.
    if (empty_cost_row && artificial_cost_row) {
      // The cost row is empty and artificial, so don't report it
      assert(cost_row_location == 0);
    } else {
      // Place the cost row according to lp.cost_row_location_
      if (lp.cost_row_location_ >= 0) {
        // The cost row location is known from the MPS file. NB To
        // index from zero whenever possible, lp.cost_row_location_ =
        // 0 if the cost row came first
        assert(lp.cost_row_location_ <= lp.num_row_);
        cost_row_location = lp.cost_row_location_ + 1;
      } else {
        // The location isn't known from an MPS file, so place it
        // last, giving a warning
        cost_row_location = lp.num_row_ + 1;
        highsLogUser(
            options.log_options, HighsLogType::kWarning,
            "The cost row for the Glpsol solution file is reported last since "
            "there is no indication of where it should be\n");
      }
    }
  } else {
    // Place the cost row according to the option value
    cost_row_location = cost_row_option;
  }
  assert(0 <= cost_row_location && cost_row_location <= lp.num_row_ + 1);
  // Despite being written in C, GLPSOL indexes rows (columns) from
  // 1..m (1..n) with - bizarrely! - m being one more than the number
  // of constraints if the cost vector is reported.
  const HighsInt num_row = lp.num_row_;
  const HighsInt num_col = lp.num_col_;
  // There's one more row and more nonzeros if the cost row is
  // reported
  const HighsInt delta_num_row = cost_row_location > 0;
  const HighsInt glpsol_num_row = num_row + delta_num_row;
  // If the cost row isn't reported, then the number of nonzeros is
  // just the number in the constraint matrix
  if (cost_row_location <= 0) num_nz = lp.a_matrix_.numNz();
  // Record the discrete nature of the model
  HighsInt num_integer = 0;
  HighsInt num_binary = 0;
  bool is_mip = false;
  if (lp.integrality_.size() == lp.num_col_) {
    for (HighsInt iCol = 0; iCol < lp.num_col_; iCol++) {
      if (lp.integrality_[iCol] != HighsVarType::kContinuous) {
        is_mip = true;
        num_integer++;
        if (lp.col_lower_[iCol] == 0 && lp.col_upper_[iCol] == 1) num_binary++;
      }
    }
  }
  // Raw and pretty are the initially the same, but for the "c "
  // prefix to raw lines
  std::string line_prefix = "";
  if (raw) line_prefix = "c ";
  fprintf(file, "%s%-12s%s\n", line_prefix.c_str(),
          "Problem:", lp.model_name_.c_str());
  fprintf(file, "%s%-12s%d\n", line_prefix.c_str(),
          "Rows:", (int)glpsol_num_row);
  fprintf(file, "%s%-12s%d", line_prefix.c_str(), "Columns:", (int)num_col);
  if (!raw && is_mip)
    fprintf(file, " (%d integer, %d binary)", (int)num_integer,
            (int)num_binary);
  fprintf(file, "\n");
  fprintf(file, "%s%-12s%d\n", line_prefix.c_str(), "Non-zeros:", (int)num_nz);
  std::string model_status_text = "";
  std::string model_status_char = "";  // Just for raw MIP solution
  switch (model_status) {
    case HighsModelStatus::kOptimal:
      if (is_mip) model_status_text = "INTEGER ";
      model_status_text += "OPTIMAL";
      model_status_char = "o";
      break;
    case HighsModelStatus::kInfeasible:
      if (is_mip) {
        model_status_text = "INTEGER INFEASIBLE";
      } else {
        model_status_text = "INFEASIBLE (FINAL)";
      }
      model_status_char = "n";
      break;
    case HighsModelStatus::kUnboundedOrInfeasible:
      model_status_text += "INFEASIBLE (INTERMEDIATE)";
      model_status_char = "?";  // Glpk has no equivalent
      break;
    case HighsModelStatus::kUnbounded:
      if (is_mip) model_status_text = "INTEGER ";
      model_status_text += "UNBOUNDED";
      model_status_char = "u";  // Glpk has no equivalent
      break;
    default:
      model_status_text = "????";
      model_status_char = "?";
      break;
  }
  assert(model_status_text != "");
  assert(model_status_char != "");
  fprintf(file, "%s%-12s%s\n", line_prefix.c_str(),
          "Status:", model_status_text.c_str());
  // If info is not valid, then cannot write more
  if (!info.valid) return;
  // Now write out the numerical information
  //
  // Determine the objective name to write out
  std::string objective_name = lp.objective_name_;
  // Make sure that no objective name is written out if there are rows
  // and no row names
  if (lp.num_row_ && !have_row_names) objective_name = "";
  // if there are row names to be written out, there must be a
  // non-trivial objective name
  if (have_row_names) assert(lp.objective_name_ != "");
  const bool has_objective_name = lp.objective_name_ != "";
  fprintf(file, "%s%-12s%s%.10g (%s)\n", line_prefix.c_str(), "Objective:",
          !(has_objective && has_objective_name)
              ? ""
              : (objective_name + " = ").c_str(),
          has_objective ? info.objective_function_value : 0,
          lp.sense_ == ObjSense::kMinimize ? "MINimum" : "MAXimum");
  // No space after "c" on blank line!
  if (raw) line_prefix = "c";
  fprintf(file, "%s\n", line_prefix.c_str());
  // Detailed lines are rather different
  if (raw) {
    fprintf(file, "s %s %d %d ", is_mip ? "mip" : "bas", (int)glpsol_num_row,
            (int)num_col);
    if (is_mip) {
      fprintf(file, "%s", model_status_char.c_str());
    } else {
      if (info.primal_solution_status == kSolutionStatusNone) {
        fprintf(file, "u");
      } else if (info.primal_solution_status == kSolutionStatusInfeasible) {
        fprintf(file, "i");
      } else if (info.primal_solution_status == kSolutionStatusFeasible) {
        fprintf(file, "f");
      } else {
        fprintf(file, "?");
      }
      fprintf(file, " ");
      if (info.dual_solution_status == kSolutionStatusNone) {
        fprintf(file, "u");
      } else if (info.dual_solution_status == kSolutionStatusInfeasible) {
        fprintf(file, "i");
      } else if (info.dual_solution_status == kSolutionStatusFeasible) {
        fprintf(file, "f");
      } else {
        fprintf(file, "?");
      }
    }
    double double_value = has_objective ? info.objective_function_value : 0;
    std::array<char, 32> double_string =
        highsDoubleToString(double_value, kHighsDoubleTolerance);
    fprintf(file, " %s\n", double_string.data());
  }
  if (!raw) {
    fprintf(file,
            "   No.   Row name   %s   Activity     Lower bound  "
            " Upper bound",
            have_basis ? "St" : "  ");
    if (have_dual) fprintf(file, "    Marginal");
    fprintf(file, "\n");

    fprintf(file,
            "------ ------------ %s ------------- ------------- "
            "-------------",
            have_basis ? "--" : "  ");
    if (have_dual) fprintf(file, " -------------");
    fprintf(file, "\n");
  }

  HighsInt row_id = 0;
  for (HighsInt iRow = 0; iRow < lp.num_row_; iRow++) {
    row_id++;
    if (row_id == cost_row_location) {
      writeGlpsolCostRow(file, raw, is_mip, row_id, objective_name,
                         info.objective_function_value);
      row_id++;
    }
    if (raw) {
      fprintf(file, "i %d ", (int)row_id);
      if (is_mip) {
        // Complete the line if for a MIP
        double double_value = solution.row_value[iRow];
        std::array<char, 32> double_string =
            highsDoubleToString(double_value, kHighsDoubleTolerance);
        fprintf(file, "%s\n", double_string.data());
        continue;
      }
    } else {
      fprintf(file, "%6d ", (int)row_id);
      std::string row_name = "";
      if (have_row_names) row_name = lp.row_names_[iRow];
      if (row_name.length() <= 12) {
        fprintf(file, "%-12s ", row_name.c_str());
      } else {
        fprintf(file, "%s\n%20s", row_name.c_str(), "");
      }
    }
    const double lower = lp.row_lower_[iRow];
    const double upper = lp.row_upper_[iRow];
    const double value = solution.row_value[iRow];
    const double dual = have_dual ? solution.row_dual[iRow] : 0;
    std::string status_text = "  ";
    std::string status_char = "";
    if (have_basis) {
      const HighsBasisStatus status = basis.row_status[iRow];
      switch (basis.row_status[iRow]) {
        case HighsBasisStatus::kBasic:
          status_text = "B ";
          status_char = "b";
          break;
        case HighsBasisStatus::kLower:
          status_text = lower == upper ? "NS" : "NL";
          status_char = lower == upper ? "s" : "l";
          break;
        case HighsBasisStatus::kUpper:
          status_text = lower == upper ? "NS" : "NU";
          status_char = lower == upper ? "s" : "u";
          break;
        case HighsBasisStatus::kZero:
          status_text = "NF";
          status_char = "f";
          break;
        default:
          status_text = "??";
          status_char = "?";
          break;
      }
    }
    if (raw) {
      fprintf(file, "%s ", status_char.c_str());
      double double_value = solution.row_value[iRow];
      std::array<char, 32> double_string =
          highsDoubleToString(double_value, kHighsDoubleTolerance);
      fprintf(file, "%s ", double_string.data());
    } else {
      fprintf(file, "%s ", status_text.c_str());
      fprintf(file, "%13.6g ", fabs(value) <= kGlpsolPrintAsZero ? 0.0 : value);
      if (lower > -kHighsInf)
        fprintf(file, "%13.6g ", lower);
      else
        fprintf(file, "%13s ", "");
      if (lower != upper && upper < kHighsInf)
        fprintf(file, "%13.6g ", upper);
      else
        fprintf(file, "%13s ", lower == upper ? "=" : "");
    }
    if (have_dual) {
      if (raw) {
        double double_value = solution.row_dual[iRow];
        std::array<char, 32> double_string =
            highsDoubleToString(double_value, kHighsDoubleTolerance);
        fprintf(file, "%s", double_string.data());
      } else {
        // If the row is known to be basic, don't print the dual
        // value. If there's no basis, row cannot be known to be basic
        bool not_basic = have_basis;
        if (have_basis)
          not_basic = basis.row_status[iRow] != HighsBasisStatus::kBasic;
        if (not_basic) {
          if (fabs(dual) <= kGlpsolPrintAsZero)
            fprintf(file, "%13s", "< eps");
          else
            fprintf(file, "%13.6g ", dual);
        }
      }
    }
    fprintf(file, "\n");
  }

  if (cost_row_location == lp.num_row_ + 1) {
    row_id++;
    writeGlpsolCostRow(file, raw, is_mip, row_id, objective_name,
                       info.objective_function_value);
  }
  if (!raw) fprintf(file, "\n");

  if (!raw) {
    fprintf(file,
            "   No. Column name  %s   Activity     Lower bound  "
            " Upper bound",
            have_basis ? "St" : "  ");
    if (have_dual) fprintf(file, "    Marginal");
    fprintf(file, "\n");
    fprintf(file,
            "------ ------------ %s ------------- ------------- "
            "-------------",
            have_basis ? "--" : "  ");
    if (have_dual) fprintf(file, " -------------");
    fprintf(file, "\n");
  }

  if (raw) line_prefix = "j ";
  for (HighsInt iCol = 0; iCol < lp.num_col_; iCol++) {
    if (raw) {
      fprintf(file, "%s%d ", line_prefix.c_str(), (int)(iCol + 1));
      if (is_mip) {
        double double_value = solution.col_value[iCol];
        std::array<char, 32> double_string =
            highsDoubleToString(double_value, kHighsDoubleTolerance);
        fprintf(file, "%s\n", double_string.data());
        continue;
      }
    } else {
      fprintf(file, "%6d ", (int)(iCol + 1));
      std::string col_name = "";
      if (have_col_names) col_name = lp.col_names_[iCol];
      if (!have_col_names || col_name.length() <= 12) {
        fprintf(file, "%-12s ", !have_col_names ? "" : col_name.c_str());
      } else {
        fprintf(file, "%s\n%20s", col_name.c_str(), "");
      }
    }
    const double lower = lp.col_lower_[iCol];
    const double upper = lp.col_upper_[iCol];
    const double value = solution.col_value[iCol];
    const double dual = have_dual ? solution.col_dual[iCol] : 0;
    std::string status_text = "  ";
    std::string status_char = "";
    if (have_basis) {
      const HighsBasisStatus status = basis.col_status[iCol];
      switch (basis.col_status[iCol]) {
        case HighsBasisStatus::kBasic:
          status_text = "B ";
          status_char = "b";
          break;
        case HighsBasisStatus::kLower:
          status_text = lower == upper ? "NS" : "NL";
          status_char = lower == upper ? "s" : "l";
          break;
        case HighsBasisStatus::kUpper:
          status_text = lower == upper ? "NS" : "NU";
          status_char = lower == upper ? "s" : "u";
          break;
        case HighsBasisStatus::kZero:
          status_text = "NF";
          status_char = "f";
          break;
        default:
          status_text = "??";
          status_char = "?";
          break;
      }
    } else if (is_mip) {
      if (lp.integrality_[iCol] != HighsVarType::kContinuous)
        status_text = "* ";
    }
    if (raw) {
      fprintf(file, "%s ", status_char.c_str());
      double double_value = solution.col_value[iCol];
      std::array<char, 32> double_string =
          highsDoubleToString(double_value, kHighsDoubleTolerance);
      fprintf(file, "%s ", double_string.data());
    } else {
      fprintf(file, "%s ", status_text.c_str());
      fprintf(file, "%13.6g ", fabs(value) <= kGlpsolPrintAsZero ? 0.0 : value);
      if (lower > -kHighsInf)
        fprintf(file, "%13.6g ", lower);
      else
        fprintf(file, "%13s ", "");
      if (lower != upper && upper < kHighsInf)
        fprintf(file, "%13.6g ", upper);
      else
        fprintf(file, "%13s ", lower == upper ? "=" : "");
    }
    if (have_dual) {
      if (raw) {
        double double_value = solution.col_dual[iCol];
        std::array<char, 32> double_string =
            highsDoubleToString(double_value, kHighsDoubleTolerance);
        fprintf(file, "%s", double_string.data());
      } else {
        // If the column is known to be basic, don't print the dual
        // value. If there's no basis, column cannot be known to be
        // basic
        bool not_basic = have_basis;
        if (have_basis)
          not_basic = basis.col_status[iCol] != HighsBasisStatus::kBasic;
        if (not_basic) {
          if (fabs(dual) <= kGlpsolPrintAsZero)
            fprintf(file, "%13s", "< eps");
          else
            fprintf(file, "%13.6g ", dual);
        }
      }
    }
    fprintf(file, "\n");
  }
  if (raw) {
    fprintf(file, "e o f\n");
    return;
  }
  HighsPrimalDualErrors errors;
  HighsInfo local_info;
  HighsInt absolute_error_index;
  double absolute_error_value;
  HighsInt relative_error_index;
  double relative_error_value;
  getKktFailures(options, model, solution, basis, local_info, errors, true);
  fprintf(file, "\n");
  if (is_mip) {
    fprintf(file, "Integer feasibility conditions:\n");
  } else {
    fprintf(file, "Karush-Kuhn-Tucker optimality conditions:\n");
  }
  fprintf(file, "\n");
  // Primal residual
  absolute_error_value = errors.max_primal_residual.absolute_value;
  absolute_error_index = errors.max_primal_residual.absolute_index + 1;
  relative_error_value = errors.max_primal_residual.relative_value;
  relative_error_index = errors.max_primal_residual.relative_index + 1;
  if (!absolute_error_value) absolute_error_index = 0;
  if (!relative_error_value) relative_error_index = 0;
  fprintf(file, "KKT.PE: max.abs.err = %.2e on row %d\n", absolute_error_value,
          absolute_error_index == 0 ? 0 : (int)absolute_error_index);
  fprintf(file, "        max.rel.err = %.2e on row %d\n", relative_error_value,
          absolute_error_index == 0 ? 0 : (int)relative_error_index);
  fprintf(file, "%8s%s\n", "",
          relative_error_value <= kGlpsolHighQuality
              ? "High quality"
              : relative_error_value <= kGlpsolMediumQuality
                    ? "Medium quality"
                    : relative_error_value <= kGlpsolLowQuality
                          ? "Low quality"
                          : "PRIMAL SOLUTION IS WRONG");
  fprintf(file, "\n");

  // Primal infeasibility
  absolute_error_value = errors.max_primal_infeasibility.absolute_value;
  absolute_error_index = errors.max_primal_infeasibility.absolute_index + 1;
  relative_error_value = errors.max_primal_infeasibility.relative_value;
  relative_error_index = errors.max_primal_infeasibility.relative_index + 1;
  if (!absolute_error_value) absolute_error_index = 0;
  if (!relative_error_value) relative_error_index = 0;
  bool on_col = absolute_error_index > 0 && absolute_error_index <= lp.num_col_;
  fprintf(file, "KKT.PB: max.abs.err = %.2e on %s %d\n", absolute_error_value,
          on_col ? "column" : "row",
          absolute_error_index <= lp.num_col_
              ? (int)absolute_error_index
              : (int)(absolute_error_index - lp.num_col_));
  on_col = relative_error_index > 0 && relative_error_index <= lp.num_col_;
  fprintf(file, "        max.rel.err = %.2e on %s %d\n", relative_error_value,
          on_col ? "column" : "row",
          relative_error_index <= lp.num_col_
              ? (int)relative_error_index
              : (int)(relative_error_index - lp.num_col_));
  fprintf(file, "%8s%s\n", "",
          relative_error_value <= kGlpsolHighQuality
              ? "High quality"
              : relative_error_value <= kGlpsolMediumQuality
                    ? "Medium quality"
                    : relative_error_value <= kGlpsolLowQuality
                          ? "Low quality"
                          : "PRIMAL SOLUTION IS INFEASIBLE");
  fprintf(file, "\n");

  if (have_dual) {
    // Dual residual
    absolute_error_value = errors.max_dual_residual.absolute_value;
    absolute_error_index = errors.max_dual_residual.absolute_index + 1;
    relative_error_value = errors.max_dual_residual.relative_value;
    relative_error_index = errors.max_dual_residual.relative_index + 1;
    if (!absolute_error_value) absolute_error_index = 0;
    if (!relative_error_value) relative_error_index = 0;
    fprintf(file, "KKT.DE: max.abs.err = %.2e on column %d\n",
            absolute_error_value, (int)absolute_error_index);
    fprintf(file, "        max.rel.err = %.2e on column %d\n",
            relative_error_value, (int)relative_error_index);
    fprintf(file, "%8s%s\n", "",
            relative_error_value <= kGlpsolHighQuality
                ? "High quality"
                : relative_error_value <= kGlpsolMediumQuality
                      ? "Medium quality"
                      : relative_error_value <= kGlpsolLowQuality
                            ? "Low quality"
                            : "DUAL SOLUTION IS WRONG");
    fprintf(file, "\n");

    // Dual infeasibility
    absolute_error_value = errors.max_dual_infeasibility.absolute_value;
    absolute_error_index = errors.max_dual_infeasibility.absolute_index + 1;
    relative_error_value = errors.max_dual_infeasibility.relative_value;
    relative_error_index = errors.max_dual_infeasibility.relative_index + 1;
    if (!absolute_error_value) absolute_error_index = 0;
    if (!relative_error_value) relative_error_index = 0;
    bool on_col =
        absolute_error_index > 0 && absolute_error_index <= lp.num_col_;
    fprintf(file, "KKT.DB: max.abs.err = %.2e on %s %d\n", absolute_error_value,
            on_col ? "column" : "row",
            absolute_error_index <= lp.num_col_
                ? (int)absolute_error_index
                : (int)(absolute_error_index - lp.num_col_));
    on_col = relative_error_index > 0 && relative_error_index <= lp.num_col_;
    fprintf(file, "        max.rel.err = %.2e on %s %d\n", relative_error_value,
            on_col ? "column" : "row",
            relative_error_index <= lp.num_col_
                ? (int)relative_error_index
                : (int)(relative_error_index - lp.num_col_));
    fprintf(file, "%8s%s\n", "",
            relative_error_value <= kGlpsolHighQuality
                ? "High quality"
                : relative_error_value <= kGlpsolMediumQuality
                      ? "Medium quality"
                      : relative_error_value <= kGlpsolLowQuality
                            ? "Low quality"
                            : "DUAL SOLUTION IS INFEASIBLE");
    fprintf(file, "\n");
  }
  fprintf(file, "End of output\n");
}

void writeOldRawSolution(FILE* file, const HighsLp& lp, const HighsBasis& basis,
                         const HighsSolution& solution) {
  const bool have_value = solution.value_valid;
  const bool have_dual = solution.dual_valid;
  const bool have_basis = basis.valid;
  vector<double> use_col_value;
  vector<double> use_row_value;
  vector<double> use_col_dual;
  vector<double> use_row_dual;
  vector<HighsBasisStatus> use_col_status;
  vector<HighsBasisStatus> use_row_status;
  if (have_value) {
    use_col_value = solution.col_value;
    use_row_value = solution.row_value;
  }
  if (have_dual) {
    use_col_dual = solution.col_dual;
    use_row_dual = solution.row_dual;
  }
  if (have_basis) {
    use_col_status = basis.col_status;
    use_row_status = basis.row_status;
  }
  if (!have_value && !have_dual && !have_basis) return;
  fprintf(file,
          "%" HIGHSINT_FORMAT " %" HIGHSINT_FORMAT
          " : Number of columns and rows for primal or dual solution "
          "or basis\n",
          lp.num_col_, lp.num_row_);
  if (have_value) {
    fprintf(file, "T");
  } else {
    fprintf(file, "F");
  }
  fprintf(file, " Primal solution\n");
  if (have_dual) {
    fprintf(file, "T");
  } else {
    fprintf(file, "F");
  }
  fprintf(file, " Dual solution\n");
  if (have_basis) {
    fprintf(file, "T");
  } else {
    fprintf(file, "F");
  }
  fprintf(file, " Basis\n");
  fprintf(file, "Columns\n");
  for (HighsInt iCol = 0; iCol < lp.num_col_; iCol++) {
    if (have_value) fprintf(file, "%.15g ", use_col_value[iCol]);
    if (have_dual) fprintf(file, "%.15g ", use_col_dual[iCol]);
    if (have_basis)
      fprintf(file, "%" HIGHSINT_FORMAT "", (HighsInt)use_col_status[iCol]);
    fprintf(file, "\n");
  }
  fprintf(file, "Rows\n");
  for (HighsInt iRow = 0; iRow < lp.num_row_; iRow++) {
    if (have_value) fprintf(file, "%.15g ", use_row_value[iRow]);
    if (have_dual) fprintf(file, "%.15g ", use_row_dual[iRow]);
    if (have_basis)
      fprintf(file, "%" HIGHSINT_FORMAT "", (HighsInt)use_row_status[iRow]);
    fprintf(file, "\n");
  }
}

HighsBasisStatus checkedVarHighsNonbasicStatus(
    const HighsBasisStatus ideal_status, const double lower,
    const double upper) {
  HighsBasisStatus checked_status;
  if (ideal_status == HighsBasisStatus::kLower ||
      ideal_status == HighsBasisStatus::kZero) {
    // Looking to give status LOWER or ZERO
    if (highs_isInfinity(-lower)) {
      // Lower bound is infinite
      if (highs_isInfinity(upper)) {
        // Upper bound is infinite
        checked_status = HighsBasisStatus::kZero;
      } else {
        // Upper bound is finite
        checked_status = HighsBasisStatus::kUpper;
      }
    } else {
      checked_status = HighsBasisStatus::kLower;
    }
  } else {
    // Looking to give status UPPER
    if (highs_isInfinity(upper)) {
      // Upper bound is infinite
      if (highs_isInfinity(-lower)) {
        // Lower bound is infinite
        checked_status = HighsBasisStatus::kZero;
      } else {
        // Upper bound is finite
        checked_status = HighsBasisStatus::kLower;
      }
    } else {
      checked_status = HighsBasisStatus::kUpper;
    }
  }
  return checked_status;
}

// Return a string representation of SolutionStatus
std::string utilSolutionStatusToString(const HighsInt solution_status) {
  switch (solution_status) {
    case kSolutionStatusNone:
      return "None";
      break;
    case kSolutionStatusInfeasible:
      return "Infeasible";
      break;
    case kSolutionStatusFeasible:
      return "Feasible";
      break;
    default:
      assert(1 == 0);
      return "Unrecognised solution status";
  }
}

// Return a string representation of HighsBasisStatus
std::string utilBasisStatusToString(const HighsBasisStatus basis_status) {
  switch (basis_status) {
    case HighsBasisStatus::kLower:
      return "At lower/fixed bound";
      break;
    case HighsBasisStatus::kBasic:
      return "Basic";
      break;
    case HighsBasisStatus::kUpper:
      return "At upper bound";
      break;
    case HighsBasisStatus::kZero:
      return "Free at zero";
      break;
    case HighsBasisStatus::kNonbasic:
      return "Nonbasic";
      break;
    default:
      assert(1 == 0);
      return "Unrecognised solution status";
  }
}

// Return a string representation of basis validity
std::string utilBasisValidityToString(const HighsInt basis_validity) {
  if (basis_validity) {
    return "Valid";
  } else {
    return "Not valid";
  }
}

// Return a string representation of HighsModelStatus.
std::string utilModelStatusToString(const HighsModelStatus model_status) {
  switch (model_status) {
    case HighsModelStatus::kNotset:
      return "Not Set";
      break;
    case HighsModelStatus::kLoadError:
      return "Load error";
      break;
    case HighsModelStatus::kModelError:
      return "Model error";
      break;
    case HighsModelStatus::kPresolveError:
      return "Presolve error";
      break;
    case HighsModelStatus::kSolveError:
      return "Solve error";
      break;
    case HighsModelStatus::kPostsolveError:
      return "Postsolve error";
      break;
    case HighsModelStatus::kModelEmpty:
      return "Empty";
      break;
    case HighsModelStatus::kOptimal:
      return "Optimal";
      break;
    case HighsModelStatus::kInfeasible:
      return "Infeasible";
      break;
    case HighsModelStatus::kUnboundedOrInfeasible:
      return "Primal infeasible or unbounded";
      break;
    case HighsModelStatus::kUnbounded:
      return "Unbounded";
      break;
    case HighsModelStatus::kObjectiveBound:
      return "Bound on objective reached";
      break;
    case HighsModelStatus::kObjectiveTarget:
      return "Target for objective reached";
      break;
    case HighsModelStatus::kTimeLimit:
      return "Time limit reached";
      break;
    case HighsModelStatus::kIterationLimit:
      return "Iteration limit reached";
      break;
    case HighsModelStatus::kUnknown:
      return "Unknown";
      break;
    default:
      assert(1 == 0);
      return "Unrecognised HiGHS model status";
  }
}

// Deduce the HighsStatus value corresponding to a HighsModelStatus value.
HighsStatus highsStatusFromHighsModelStatus(HighsModelStatus model_status) {
  switch (model_status) {
    case HighsModelStatus::kNotset:
      return HighsStatus::kError;
    case HighsModelStatus::kLoadError:
      return HighsStatus::kError;
    case HighsModelStatus::kModelError:
      return HighsStatus::kError;
    case HighsModelStatus::kPresolveError:
      return HighsStatus::kError;
    case HighsModelStatus::kSolveError:
      return HighsStatus::kError;
    case HighsModelStatus::kPostsolveError:
      return HighsStatus::kError;
    case HighsModelStatus::kModelEmpty:
      return HighsStatus::kOk;
    case HighsModelStatus::kOptimal:
      return HighsStatus::kOk;
    case HighsModelStatus::kInfeasible:
      return HighsStatus::kOk;
    case HighsModelStatus::kUnboundedOrInfeasible:
      return HighsStatus::kOk;
    case HighsModelStatus::kUnbounded:
      return HighsStatus::kOk;
    case HighsModelStatus::kObjectiveBound:
      return HighsStatus::kOk;
    case HighsModelStatus::kObjectiveTarget:
      return HighsStatus::kOk;
    case HighsModelStatus::kTimeLimit:
      return HighsStatus::kWarning;
    case HighsModelStatus::kIterationLimit:
      return HighsStatus::kWarning;
    case HighsModelStatus::kUnknown:
      return HighsStatus::kWarning;
    default:
      return HighsStatus::kError;
  }
}

std::string findModelObjectiveName(const HighsLp* lp,
                                   const HighsHessian* hessian) {
  // Return any non-trivial current objective name
  if (lp->objective_name_ != "") return lp->objective_name_;

  std::string objective_name = "";
  // Determine whether there is a nonzero cost vector
  bool has_objective = false;
  for (HighsInt iCol = 0; iCol < lp->num_col_; iCol++) {
    if (lp->col_cost_[iCol]) {
      has_objective = true;
      break;
    }
  }
  if (!has_objective && hessian) {
    // Zero cost vector, so only chance of an objective comes from any
    // Hessian
    has_objective = hessian->dim_;
  }
  HighsInt pass = 0;
  for (;;) {
    // Loop until a valid name is found. Vanishingly unlikely to have
    // to pass more than once, since check for objective name
    // duplicating a row name is very unlikely to fail
    //
    // So set up an appropriate name (stem)
    if (has_objective) {
      objective_name = "Obj";
    } else {
      objective_name = "NoObj";
    }
    // If there are no row names, then the objective name is certainly
    // OK
    if (lp->row_names_.size() == 0) break;
    if (pass) objective_name += pass;
    // Ensure that the objective name doesn't clash with any row names
    bool ok_objective_name = true;
    for (HighsInt iRow = 0; iRow < lp->num_row_; iRow++) {
      std::string trimmed_name = lp->row_names_[iRow];
      trimmed_name = trim(trimmed_name);
      if (objective_name == trimmed_name) {
        ok_objective_name = false;
        break;
      }
    }
    if (ok_objective_name) break;
    pass++;
  }
  assert(objective_name != "");
  return objective_name;
}
