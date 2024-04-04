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
/**@file lp_data/HighsInfo.cpp
 * @brief
 */
#include "lp_data/HighsInfo.h"

#include <cassert>

#include "lp_data/HighsOptions.h"

void HighsInfo::invalidate() {
  valid = false;
  mip_node_count = -1;
  simplex_iteration_count = -1;
  ipm_iteration_count = -1;
  crossover_iteration_count = -1;
  qp_iteration_count = -1;
  primal_solution_status = kSolutionStatusNone;
  dual_solution_status = kSolutionStatusNone;
  basis_validity = kBasisValidityInvalid;
  objective_function_value = 0;
  mip_dual_bound = 0;
  mip_gap = kHighsInf;
  max_integrality_violation = kHighsIllegalInfeasibilityMeasure;
  num_primal_infeasibilities = kHighsIllegalInfeasibilityCount;
  max_primal_infeasibility = kHighsIllegalInfeasibilityMeasure;
  sum_primal_infeasibilities = kHighsIllegalInfeasibilityMeasure;
  num_dual_infeasibilities = kHighsIllegalInfeasibilityCount;
  max_dual_infeasibility = kHighsIllegalInfeasibilityMeasure;
  sum_dual_infeasibilities = kHighsIllegalInfeasibilityMeasure;
}

static std::string infoEntryTypeToString(const HighsInfoType type) {
  if (type == HighsInfoType::kInt64) {
    return "int64_t";
  } else if (type == HighsInfoType::kInt) {
    return "HighsInt";
  } else {
    return "double";
  }
}

InfoStatus getInfoIndex(const HighsOptions& options, const std::string& name,
                        const std::vector<InfoRecord*>& info_records,
                        HighsInt& index) {
  HighsInt num_info = info_records.size();
  for (index = 0; index < num_info; index++)
    if (info_records[index]->name == name) return InfoStatus::kOk;
  highsLogUser(options.log_options, HighsLogType::kError,
               "getInfoIndex: Info \"%s\" is unknown\n", name.c_str());
  return InfoStatus::kUnknownInfo;
}

InfoStatus checkInfo(const HighsOptions& options,
                     const std::vector<InfoRecord*>& info_records) {
  bool error_found = false;
  HighsInt num_info = info_records.size();
  for (HighsInt index = 0; index < num_info; index++) {
    std::string name = info_records[index]->name;
    HighsInfoType type = info_records[index]->type;
    // Check that there are no other info with the same name
    for (HighsInt check_index = 0; check_index < num_info; check_index++) {
      if (check_index == index) continue;
      std::string check_name = info_records[check_index]->name;
      if (check_name == name) {
        highsLogUser(options.log_options, HighsLogType::kError,
                     "checkInfo: Info %" HIGHSINT_FORMAT
                     " (\"%s\") has the same name as info %" HIGHSINT_FORMAT
                     " \"%s\"\n",
                     index, name.c_str(), check_index, check_name.c_str());
        error_found = true;
      }
    }
    if (type == HighsInfoType::kInt64) {
      // Check int64_t info
      InfoRecordInt64& info = ((InfoRecordInt64*)info_records[index])[0];
      // Check that there are no other info with the same value pointers
      int64_t* value_pointer = info.value;
      for (HighsInt check_index = 0; check_index < num_info; check_index++) {
        if (check_index == index) continue;
        InfoRecordInt64& check_info =
            ((InfoRecordInt64*)info_records[check_index])[0];
        if (check_info.type == HighsInfoType::kInt64) {
          if (check_info.value == value_pointer) {
            highsLogUser(options.log_options, HighsLogType::kError,
                         "checkInfo: Info %" HIGHSINT_FORMAT
                         " (\"%s\") has the same value "
                         "pointer as info %" HIGHSINT_FORMAT " (\"%s\")\n",
                         index, info.name.c_str(), check_index,
                         check_info.name.c_str());
            error_found = true;
          }
        }
      }
    } else if (type == HighsInfoType::kInt) {
      // Check HighsInt info
      InfoRecordInt& info = ((InfoRecordInt*)info_records[index])[0];
      // Check that there are no other info with the same value pointers
      HighsInt* value_pointer = info.value;
      for (HighsInt check_index = 0; check_index < num_info; check_index++) {
        if (check_index == index) continue;
        InfoRecordInt& check_info =
            ((InfoRecordInt*)info_records[check_index])[0];
        if (check_info.type == HighsInfoType::kInt) {
          if (check_info.value == value_pointer) {
            highsLogUser(options.log_options, HighsLogType::kError,
                         "checkInfo: Info %" HIGHSINT_FORMAT
                         " (\"%s\") has the same value "
                         "pointer as info %" HIGHSINT_FORMAT " (\"%s\")\n",
                         index, info.name.c_str(), check_index,
                         check_info.name.c_str());
            error_found = true;
          }
        }
      }
    } else if (type == HighsInfoType::kDouble) {
      // Check double info
      InfoRecordDouble& info = ((InfoRecordDouble*)info_records[index])[0];
      // Check that there are no other info with the same value pointers
      double* value_pointer = info.value;
      for (HighsInt check_index = 0; check_index < num_info; check_index++) {
        if (check_index == index) continue;
        InfoRecordDouble& check_info =
            ((InfoRecordDouble*)info_records[check_index])[0];
        if (check_info.type == HighsInfoType::kDouble) {
          if (check_info.value == value_pointer) {
            highsLogUser(options.log_options, HighsLogType::kError,
                         "checkInfo: Info %" HIGHSINT_FORMAT
                         " (\"%s\") has the same value "
                         "pointer as info %" HIGHSINT_FORMAT " (\"%s\")\n",
                         index, info.name.c_str(), check_index,
                         check_info.name.c_str());
            error_found = true;
          }
        }
      }
    }
  }
  if (error_found) return InfoStatus::kIllegalValue;
  highsLogUser(options.log_options, HighsLogType::kInfo,
               "checkInfo: Info are OK\n");
  return InfoStatus::kOk;
}

#ifndef HIGHSINT64
InfoStatus getLocalInfoValue(const HighsOptions& options,
                             const std::string& name, const bool valid,
                             const std::vector<InfoRecord*>& info_records,
                             int64_t& value) {
  HighsInt index;
  InfoStatus status = getInfoIndex(options, name, info_records, index);
  if (status != InfoStatus::kOk) return status;
  if (!valid) return InfoStatus::kUnavailable;
  HighsInfoType type = info_records[index]->type;
  if (type != HighsInfoType::kInt64) {
    highsLogUser(
        options.log_options, HighsLogType::kError,
        "getInfoValue: Info \"%s\" requires value of type %s, not int64_t\n",
        name.c_str(), infoEntryTypeToString(type).c_str());
    return InfoStatus::kIllegalValue;
  }
  InfoRecordInt64 info = ((InfoRecordInt64*)info_records[index])[0];
  value = *info.value;
  return InfoStatus::kOk;
}
#endif

InfoStatus getLocalInfoValue(const HighsOptions& options,
                             const std::string& name, const bool valid,
                             const std::vector<InfoRecord*>& info_records,
                             HighsInt& value) {
  HighsInt index;
  InfoStatus status = getInfoIndex(options, name, info_records, index);
  if (status != InfoStatus::kOk) return status;
  if (!valid) return InfoStatus::kUnavailable;
  HighsInfoType type = info_records[index]->type;
  bool type_ok = type == HighsInfoType::kInt;
  // When HIGHSINT64 is "on", value is int64_t and this method is used
  // get HighsInfo values of type HighsInfoType::kInt64
#ifdef HIGHSINT64
  type_ok = type_ok || type == HighsInfoType::kInt64;
#endif
  if (!type_ok) {
    std::string illegal_type = "HighsInt";
#ifdef HIGHSINT64
    illegal_type += " or int64_t";
#endif
    highsLogUser(
        options.log_options, HighsLogType::kError,
        "getInfoValue: Info \"%s\" requires value of type %s, not %s\n",
        name.c_str(), infoEntryTypeToString(type).c_str(),
        illegal_type.c_str());
    return InfoStatus::kIllegalValue;
  }
  if (type == HighsInfoType::kInt) {
    InfoRecordInt info = ((InfoRecordInt*)info_records[index])[0];
    value = *info.value;
  } else {
    assert(type == HighsInfoType::kInt64);
    InfoRecordInt64 info = ((InfoRecordInt64*)info_records[index])[0];
    value = *info.value;
  }
  return InfoStatus::kOk;
}

InfoStatus getLocalInfoValue(const HighsOptions& options,
                             const std::string& name, const bool valid,
                             const std::vector<InfoRecord*>& info_records,
                             double& value) {
  HighsInt index;
  InfoStatus status = getInfoIndex(options, name, info_records, index);
  if (status != InfoStatus::kOk) return status;
  if (!valid) return InfoStatus::kUnavailable;
  HighsInfoType type = info_records[index]->type;
  if (type != HighsInfoType::kDouble) {
    highsLogUser(
        options.log_options, HighsLogType::kError,
        "getInfoValue: Info \"%s\" requires value of type %s, not double\n",
        name.c_str(), infoEntryTypeToString(type).c_str());
    return InfoStatus::kIllegalValue;
  }
  InfoRecordDouble info = ((InfoRecordDouble*)info_records[index])[0];
  value = *info.value;
  return InfoStatus::kOk;
}

HighsStatus writeInfoToFile(FILE* file, const bool valid,
                            const std::vector<InfoRecord*>& info_records,
                            const bool html) {
  if (!html && !valid) return HighsStatus::kWarning;
  if (html) {
    fprintf(file, "<!DOCTYPE HTML>\n<html>\n\n<head>\n");
    fprintf(file, "  <title>HiGHS Info</title>\n");
    fprintf(file, "	<meta charset=\"utf-8\" />\n");
    fprintf(file,
            "	<meta name=\"viewport\" content=\"width=device-width, "
            "initial-scale=1, user-scalable=no\" />\n");
    fprintf(file,
            "	<link rel=\"stylesheet\" href=\"assets/css/main.css\" />\n");
    fprintf(file, "</head>\n");
    fprintf(file, "<body style=\"background-color:f5fafa;\"></body>\n\n");
    fprintf(file, "<h3>HiGHS Info</h3>\n\n");
    fprintf(file, "<ul>\n");
  }
  if (html || valid) reportInfo(file, info_records, html);
  if (html) {
    fprintf(file, "</ul>\n");
    fprintf(file, "</body>\n\n</html>\n");
  }
  return HighsStatus::kOk;
}

void reportInfo(FILE* file, const std::vector<InfoRecord*>& info_records,
                const bool html) {
  HighsInt num_info = info_records.size();
  for (HighsInt index = 0; index < num_info; index++) {
    HighsInfoType type = info_records[index]->type;
    // Skip the advanced info when creating HTML
    if (html && info_records[index]->advanced) continue;
    if (type == HighsInfoType::kInt64) {
      reportInfo(file, ((InfoRecordInt64*)info_records[index])[0], html);
    } else if (type == HighsInfoType::kInt) {
      reportInfo(file, ((InfoRecordInt*)info_records[index])[0], html);
    } else {
      reportInfo(file, ((InfoRecordDouble*)info_records[index])[0], html);
    }
  }
}

void reportInfo(FILE* file, const InfoRecordInt64& info, const bool html) {
  if (html) {
    fprintf(file,
            "<li><tt><font size=\"+2\"><strong>%s</strong></font></tt><br>\n",
            info.name.c_str());
    fprintf(file, "%s<br>\n", info.description.c_str());
    fprintf(file, "type: HighsInt, advanced: %s\n",
            highsBoolToString(info.advanced).c_str());
    fprintf(file, "</li>\n");
  } else {
    fprintf(file, "\n# %s\n", info.description.c_str());
    fprintf(file, "# [type: HighsInt, advanced: %s]\n",
            highsBoolToString(info.advanced).c_str());
    fprintf(file, "%s = %" PRId64 "\n", info.name.c_str(), *info.value);
  }
}

void reportInfo(FILE* file, const InfoRecordInt& info, const bool html) {
  if (html) {
    fprintf(file,
            "<li><tt><font size=\"+2\"><strong>%s</strong></font></tt><br>\n",
            info.name.c_str());
    fprintf(file, "%s<br>\n", info.description.c_str());
    fprintf(file, "type: HighsInt, advanced: %s\n",
            highsBoolToString(info.advanced).c_str());
    fprintf(file, "</li>\n");
  } else {
    fprintf(file, "\n# %s\n", info.description.c_str());
    fprintf(file, "# [type: HighsInt, advanced: %s]\n",
            highsBoolToString(info.advanced).c_str());
    fprintf(file, "%s = %" HIGHSINT_FORMAT "\n", info.name.c_str(),
            *info.value);
  }
}

void reportInfo(FILE* file, const InfoRecordDouble& info, const bool html) {
  if (html) {
    fprintf(file,
            "<li><tt><font size=\"+2\"><strong>%s</strong></font></tt><br>\n",
            info.name.c_str());
    fprintf(file, "%s<br>\n", info.description.c_str());
    fprintf(file, "type: double, advanced: %s\n",
            highsBoolToString(info.advanced).c_str());
    fprintf(file, "</li>\n");
  } else {
    fprintf(file, "\n# %s\n", info.description.c_str());
    fprintf(file, "# [type: double, advanced: %s]\n",
            highsBoolToString(info.advanced).c_str());
    fprintf(file, "%s = %g\n", info.name.c_str(), *info.value);
  }
}
