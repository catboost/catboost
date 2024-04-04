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
/**@file lp_data/HighsOptions.cpp
 * @brief
 */
#include "lp_data/HighsOptions.h"

#include <algorithm>
#include <cassert>

// void setLogOptions();

void highsOpenLogFile(HighsLogOptions& log_options,
                      std::vector<OptionRecord*>& option_records,
                      const std::string log_file) {
  HighsInt index;
  OptionStatus status =
      getOptionIndex(log_options, "log_file", option_records, index);
  assert(status == OptionStatus::kOk);
  if (log_options.log_file_stream != NULL) {
    // Current log file stream is not null, so flush and close it
    fflush(log_options.log_file_stream);
    fclose(log_options.log_file_stream);
  }
  if (log_file.compare("")) {
    // New log file name is not empty, so open it
    log_options.log_file_stream = fopen(log_file.c_str(), "w");
  } else {
    // New log file name is empty, so set the stream to null
    log_options.log_file_stream = NULL;
  }
  OptionRecordString& option = *(OptionRecordString*)option_records[index];
  option.assignvalue(log_file);
}

static std::string optionEntryTypeToString(const HighsOptionType type) {
  if (type == HighsOptionType::kBool) {
    return "bool";
  } else if (type == HighsOptionType::kInt) {
    return "HighsInt";
  } else if (type == HighsOptionType::kDouble) {
    return "double";
  } else {
    return "string";
  }
}

bool commandLineOffChooseOnOk(const HighsLogOptions& report_log_options,
                              const string& value) {
  if (value == kHighsOffString || value == kHighsChooseString ||
      value == kHighsOnString)
    return true;
  highsLogUser(report_log_options, HighsLogType::kWarning,
               "Value \"%s\" is not one of \"%s\", \"%s\" or \"%s\"\n",
               value.c_str(), kHighsOffString.c_str(),
               kHighsChooseString.c_str(), kHighsOnString.c_str());
  return false;
}

bool commandLineSolverOk(const HighsLogOptions& report_log_options,
                         const string& value) {
  if (value == kSimplexString || value == kHighsChooseString ||
      value == kIpmString)
    return true;
  highsLogUser(report_log_options, HighsLogType::kWarning,
               "Value \"%s\" is not one of \"%s\", \"%s\" or \"%s\"\n",
               value.c_str(), kSimplexString.c_str(),
               kHighsChooseString.c_str(), kIpmString.c_str());
  return false;
}

bool boolFromString(std::string value, bool& bool_value) {
  std::transform(value.begin(), value.end(), value.begin(),
                 [](unsigned char c) { return std::tolower(c); });
  if (value == "t" || value == "true" || value == "1" || value == "on") {
    bool_value = true;
  } else if (value == "f" || value == "false" || value == "0" ||
             value == "off") {
    bool_value = false;
  } else {
    return false;
  }
  return true;
}

OptionStatus getOptionIndex(const HighsLogOptions& report_log_options,
                            const std::string& name,
                            const std::vector<OptionRecord*>& option_records,
                            HighsInt& index) {
  HighsInt num_options = option_records.size();
  for (index = 0; index < num_options; index++)
    if (option_records[index]->name == name) return OptionStatus::kOk;
  highsLogUser(report_log_options, HighsLogType::kError,
               "getOptionIndex: Option \"%s\" is unknown\n", name.c_str());
  return OptionStatus::kUnknownOption;
}

OptionStatus checkOptions(const HighsLogOptions& report_log_options,
                          const std::vector<OptionRecord*>& option_records) {
  bool error_found = false;
  HighsInt num_options = option_records.size();
  for (HighsInt index = 0; index < num_options; index++) {
    std::string name = option_records[index]->name;
    HighsOptionType type = option_records[index]->type;
    // Check that there are no other options with the same name
    for (HighsInt check_index = 0; check_index < num_options; check_index++) {
      if (check_index == index) continue;
      std::string check_name = option_records[check_index]->name;
      if (check_name == name) {
        highsLogUser(report_log_options, HighsLogType::kError,
                     "checkOptions: Option %" HIGHSINT_FORMAT
                     " (\"%s\") has the same name as "
                     "option %" HIGHSINT_FORMAT " \"%s\"\n",
                     index, name.c_str(), check_index, check_name.c_str());
        error_found = true;
      }
    }
    if (type == HighsOptionType::kBool) {
      // Check bool option
      OptionRecordBool& option = ((OptionRecordBool*)option_records[index])[0];
      // Check that there are no other options with the same value pointers
      bool* value_pointer = option.value;
      for (HighsInt check_index = 0; check_index < num_options; check_index++) {
        if (check_index == index) continue;
        OptionRecordBool& check_option =
            ((OptionRecordBool*)option_records[check_index])[0];
        if (check_option.type == HighsOptionType::kBool) {
          if (check_option.value == value_pointer) {
            highsLogUser(report_log_options, HighsLogType::kError,
                         "checkOptions: Option %" HIGHSINT_FORMAT
                         " (\"%s\") has the same "
                         "value pointer as option %" HIGHSINT_FORMAT
                         " (\"%s\")\n",
                         index, option.name.c_str(), check_index,
                         check_option.name.c_str());
            error_found = true;
          }
        }
      }
    } else if (type == HighsOptionType::kInt) {
      // Check HighsInt option
      OptionRecordInt& option = ((OptionRecordInt*)option_records[index])[0];
      if (checkOption(report_log_options, option) != OptionStatus::kOk)
        error_found = true;
      // Check that there are no other options with the same value pointers
      HighsInt* value_pointer = option.value;
      for (HighsInt check_index = 0; check_index < num_options; check_index++) {
        if (check_index == index) continue;
        OptionRecordInt& check_option =
            ((OptionRecordInt*)option_records[check_index])[0];
        if (check_option.type == HighsOptionType::kInt) {
          if (check_option.value == value_pointer) {
            highsLogUser(report_log_options, HighsLogType::kError,
                         "checkOptions: Option %" HIGHSINT_FORMAT
                         " (\"%s\") has the same "
                         "value pointer as option %" HIGHSINT_FORMAT
                         " (\"%s\")\n",
                         index, option.name.c_str(), check_index,
                         check_option.name.c_str());
            error_found = true;
          }
        }
      }
    } else if (type == HighsOptionType::kDouble) {
      // Check double option
      OptionRecordDouble& option =
          ((OptionRecordDouble*)option_records[index])[0];
      if (checkOption(report_log_options, option) != OptionStatus::kOk)
        error_found = true;
      // Check that there are no other options with the same value pointers
      double* value_pointer = option.value;
      for (HighsInt check_index = 0; check_index < num_options; check_index++) {
        if (check_index == index) continue;
        OptionRecordDouble& check_option =
            ((OptionRecordDouble*)option_records[check_index])[0];
        if (check_option.type == HighsOptionType::kDouble) {
          if (check_option.value == value_pointer) {
            highsLogUser(report_log_options, HighsLogType::kError,
                         "checkOptions: Option %" HIGHSINT_FORMAT
                         " (\"%s\") has the same "
                         "value pointer as option %" HIGHSINT_FORMAT
                         " (\"%s\")\n",
                         index, option.name.c_str(), check_index,
                         check_option.name.c_str());
            error_found = true;
          }
        }
      }
    } else if (type == HighsOptionType::kString) {
      // Check string option
      OptionRecordString& option =
          ((OptionRecordString*)option_records[index])[0];
      // Check that there are no other options with the same value pointers
      std::string* value_pointer = option.value;
      for (HighsInt check_index = 0; check_index < num_options; check_index++) {
        if (check_index == index) continue;
        OptionRecordString& check_option =
            ((OptionRecordString*)option_records[check_index])[0];
        if (check_option.type == HighsOptionType::kString) {
          if (check_option.value == value_pointer) {
            highsLogUser(report_log_options, HighsLogType::kError,
                         "checkOptions: Option %" HIGHSINT_FORMAT
                         " (\"%s\") has the same "
                         "value pointer as option %" HIGHSINT_FORMAT
                         " (\"%s\")\n",
                         index, option.name.c_str(), check_index,
                         check_option.name.c_str());
            error_found = true;
          }
        }
      }
    }
  }
  if (error_found) return OptionStatus::kIllegalValue;
  highsLogUser(report_log_options, HighsLogType::kInfo,
               "checkOptions: Options are OK\n");
  return OptionStatus::kOk;
}

OptionStatus checkOption(const HighsLogOptions& report_log_options,
                         const OptionRecordInt& option) {
  if (option.lower_bound > option.upper_bound) {
    highsLogUser(
        report_log_options, HighsLogType::kError,
        "checkOption: Option \"%s\" has inconsistent bounds [%" HIGHSINT_FORMAT
        ", %" HIGHSINT_FORMAT "]\n",
        option.name.c_str(), option.lower_bound, option.upper_bound);
    return OptionStatus::kIllegalValue;
  }
  if (option.default_value < option.lower_bound ||
      option.default_value > option.upper_bound) {
    highsLogUser(
        report_log_options, HighsLogType::kError,
        "checkOption: Option \"%s\" has default value %" HIGHSINT_FORMAT
        " "
        "inconsistent with bounds [%" HIGHSINT_FORMAT ", %" HIGHSINT_FORMAT
        "]\n",
        option.name.c_str(), option.default_value, option.lower_bound,
        option.upper_bound);
    return OptionStatus::kIllegalValue;
  }
  HighsInt value = *option.value;
  if (value < option.lower_bound || value > option.upper_bound) {
    highsLogUser(report_log_options, HighsLogType::kError,
                 "checkOption: Option \"%s\" has value %" HIGHSINT_FORMAT
                 " inconsistent with "
                 "bounds [%" HIGHSINT_FORMAT ", %" HIGHSINT_FORMAT "]\n",
                 option.name.c_str(), value, option.lower_bound,
                 option.upper_bound);
    return OptionStatus::kIllegalValue;
  }
  return OptionStatus::kOk;
}

OptionStatus checkOption(const HighsLogOptions& report_log_options,
                         const OptionRecordDouble& option) {
  if (option.lower_bound > option.upper_bound) {
    highsLogUser(
        report_log_options, HighsLogType::kError,
        "checkOption: Option \"%s\" has inconsistent bounds [%g, %g]\n",
        option.name.c_str(), option.lower_bound, option.upper_bound);
    return OptionStatus::kIllegalValue;
  }
  if (option.default_value < option.lower_bound ||
      option.default_value > option.upper_bound) {
    highsLogUser(report_log_options, HighsLogType::kError,
                 "checkOption: Option \"%s\" has default value %g "
                 "inconsistent with bounds [%g, %g]\n",
                 option.name.c_str(), option.default_value, option.lower_bound,
                 option.upper_bound);
    return OptionStatus::kIllegalValue;
  }
  double value = *option.value;
  if (value < option.lower_bound || value > option.upper_bound) {
    highsLogUser(report_log_options, HighsLogType::kError,
                 "checkOption: Option \"%s\" has value %g inconsistent with "
                 "bounds [%g, %g]\n",
                 option.name.c_str(), value, option.lower_bound,
                 option.upper_bound);
    return OptionStatus::kIllegalValue;
  }
  return OptionStatus::kOk;
}

OptionStatus checkOptionValue(const HighsLogOptions& report_log_options,
                              OptionRecordInt& option, const HighsInt value) {
  if (value < option.lower_bound) {
    highsLogUser(report_log_options, HighsLogType::kWarning,
                 "checkOptionValue: Value %" HIGHSINT_FORMAT
                 " for option \"%s\" is below "
                 "lower bound of %" HIGHSINT_FORMAT "\n",
                 value, option.name.c_str(), option.lower_bound);
    return OptionStatus::kIllegalValue;
  } else if (value > option.upper_bound) {
    highsLogUser(report_log_options, HighsLogType::kWarning,
                 "checkOptionValue: Value %" HIGHSINT_FORMAT
                 " for option \"%s\" is above "
                 "upper bound of %" HIGHSINT_FORMAT "\n",
                 value, option.name.c_str(), option.upper_bound);
    return OptionStatus::kIllegalValue;
  }
  return OptionStatus::kOk;
}

OptionStatus checkOptionValue(const HighsLogOptions& report_log_options,
                              OptionRecordDouble& option, const double value) {
  if (value < option.lower_bound) {
    highsLogUser(report_log_options, HighsLogType::kWarning,
                 "checkOptionValue: Value %g for option \"%s\" is below "
                 "lower bound of %g\n",
                 value, option.name.c_str(), option.lower_bound);
    return OptionStatus::kIllegalValue;
  } else if (value > option.upper_bound) {
    highsLogUser(report_log_options, HighsLogType::kWarning,
                 "checkOptionValue: Value %g for option \"%s\" is above "
                 "upper bound of %g\n",
                 value, option.name.c_str(), option.upper_bound);
    return OptionStatus::kIllegalValue;
  }
  return OptionStatus::kOk;
}

OptionStatus checkOptionValue(const HighsLogOptions& report_log_options,
                              OptionRecordString& option,
                              const std::string value) {
  // Setting a string option. For some options only particular values
  // are permitted, so check them
  if (option.name == kPresolveString) {
    if (!commandLineOffChooseOnOk(report_log_options, value) && value != "mip")
      return OptionStatus::kIllegalValue;
  } else if (option.name == kSolverString) {
    if (!commandLineSolverOk(report_log_options, value))
      return OptionStatus::kIllegalValue;
  } else if (option.name == kParallelString) {
    if (!commandLineOffChooseOnOk(report_log_options, value))
      return OptionStatus::kIllegalValue;
  }
  return OptionStatus::kOk;
}

OptionStatus setLocalOptionValue(const HighsLogOptions& report_log_options,
                                 const std::string& name,
                                 std::vector<OptionRecord*>& option_records,
                                 const bool value) {
  HighsInt index;
  //  printf("setLocalOptionValue: \"%s\" with bool %" HIGHSINT_FORMAT "\n",
  //  name.c_str(), value);
  OptionStatus status =
      getOptionIndex(report_log_options, name, option_records, index);
  if (status != OptionStatus::kOk) return status;
  HighsOptionType type = option_records[index]->type;
  if (type != HighsOptionType::kBool) {
    highsLogUser(
        report_log_options, HighsLogType::kError,
        "setLocalOptionValue: Option \"%s\" cannot be assigned a bool\n",
        name.c_str());
    return OptionStatus::kIllegalValue;
  }
  return setLocalOptionValue(((OptionRecordBool*)option_records[index])[0],
                             value);
}

OptionStatus setLocalOptionValue(const HighsLogOptions& report_log_options,
                                 const std::string& name,
                                 std::vector<OptionRecord*>& option_records,
                                 const HighsInt value) {
  HighsInt index;
  //  printf("setLocalOptionValue: \"%s\" with HighsInt %" HIGHSINT_FORMAT "\n",
  //  name.c_str(), value);
  OptionStatus status =
      getOptionIndex(report_log_options, name, option_records, index);
  if (status != OptionStatus::kOk) return status;
  HighsOptionType type = option_records[index]->type;
  if (type != HighsOptionType::kInt) {
    highsLogUser(
        report_log_options, HighsLogType::kError,
        "setLocalOptionValue: Option \"%s\" cannot be assigned an int\n",
        name.c_str());
    return OptionStatus::kIllegalValue;
  }
  return setLocalOptionValue(
      report_log_options, ((OptionRecordInt*)option_records[index])[0], value);
}

OptionStatus setLocalOptionValue(const HighsLogOptions& report_log_options,
                                 const std::string& name,
                                 std::vector<OptionRecord*>& option_records,
                                 const double value) {
  HighsInt index;
  //  printf("setLocalOptionValue: \"%s\" with double %g\n", name.c_str(),
  //  value);
  OptionStatus status =
      getOptionIndex(report_log_options, name, option_records, index);
  if (status != OptionStatus::kOk) return status;
  HighsOptionType type = option_records[index]->type;
  if (type != HighsOptionType::kDouble) {
    highsLogUser(
        report_log_options, HighsLogType::kError,
        "setLocalOptionValue: Option \"%s\" cannot be assigned a double\n",
        name.c_str());
    return OptionStatus::kIllegalValue;
  }
  return setLocalOptionValue(report_log_options,
                             ((OptionRecordDouble*)option_records[index])[0],
                             value);
}

OptionStatus setLocalOptionValue(const HighsLogOptions& report_log_options,
                                 const std::string& name,
                                 HighsLogOptions& log_options,
                                 std::vector<OptionRecord*>& option_records,
                                 const std::string value) {
  HighsInt index;
  OptionStatus status =
      getOptionIndex(report_log_options, name, option_records, index);
  if (status != OptionStatus::kOk) return status;
  HighsOptionType type = option_records[index]->type;
  if (type == HighsOptionType::kBool) {
    bool value_bool;
    bool return_status = boolFromString(value, value_bool);
    if (!return_status) {
      highsLogUser(
          report_log_options, HighsLogType::kError,
          "setLocalOptionValue: Value \"%s\" cannot be interpreted as a bool\n",
          value.c_str());
      return OptionStatus::kIllegalValue;
    }
    return setLocalOptionValue(((OptionRecordBool*)option_records[index])[0],
                               value_bool);
  } else if (type == HighsOptionType::kInt) {
    HighsInt value_int;
    int scanned_num_char;
    const char* value_char = value.c_str();
    sscanf(value_char, "%" HIGHSINT_FORMAT "%n", &value_int, &scanned_num_char);
    const int value_num_char = strlen(value_char);
    const bool converted_ok = scanned_num_char == value_num_char;
    if (!converted_ok) {
      highsLogDev(report_log_options, HighsLogType::kError,
                  "setLocalOptionValue: Value = \"%s\" converts via sscanf as "
                  "%" HIGHSINT_FORMAT
                  " "
                  "by scanning %" HIGHSINT_FORMAT " of %" HIGHSINT_FORMAT
                  " characters\n",
                  value.c_str(), value_int, scanned_num_char, value_num_char);
      return OptionStatus::kIllegalValue;
    }
    return setLocalOptionValue(report_log_options,
                               ((OptionRecordInt*)option_records[index])[0],
                               value_int);
  } else if (type == HighsOptionType::kDouble) {
    HighsInt value_int = atoi(value.c_str());
    double value_double = atof(value.c_str());
    double value_int_double = value_int;
    if (value_double == value_int_double) {
      highsLogDev(report_log_options, HighsLogType::kInfo,
                  "setLocalOptionValue: Value = \"%s\" converts via atoi as "
                  "%" HIGHSINT_FORMAT
                  " "
                  "so is %g as double, and %g via atof\n",
                  value.c_str(), value_int, value_int_double, value_double);
    }
    return setLocalOptionValue(report_log_options,
                               ((OptionRecordDouble*)option_records[index])[0],
                               atof(value.c_str()));
  } else {
    // Setting a string option value
    if (!name.compare(kLogFileString)) {
      OptionRecordString& option = *(OptionRecordString*)option_records[index];
      std::string original_log_file = *(option.value);
      if (value.compare(original_log_file)) {
        // Changing the name of the log file
        highsOpenLogFile(log_options, option_records, value);
      }
    }
    if (!name.compare(kModelFileString)) {
      // Don't allow model filename to be changed - it's only an
      // option so that reading of run-time options works
      highsLogUser(report_log_options, HighsLogType::kError,
                   "setLocalOptionValue: model filename cannot be set\n");
      return OptionStatus::kUnknownOption;
    } else {
      return setLocalOptionValue(
          report_log_options, ((OptionRecordString*)option_records[index])[0],
          value);
    }
  }
}

OptionStatus setLocalOptionValue(const HighsLogOptions& report_log_options,
                                 const std::string& name,
                                 HighsLogOptions& log_options,
                                 std::vector<OptionRecord*>& option_records,
                                 const char* value) {
  // Handles values passed as explicit values in quotes
  std::string value_as_string(value);
  return setLocalOptionValue(report_log_options, name, log_options,
                             option_records, value_as_string);
}

OptionStatus setLocalOptionValue(OptionRecordBool& option, const bool value) {
  option.assignvalue(value);
  return OptionStatus::kOk;
}

OptionStatus setLocalOptionValue(const HighsLogOptions& report_log_options,
                                 OptionRecordInt& option,
                                 const HighsInt value) {
  OptionStatus return_status =
      checkOptionValue(report_log_options, option, value);
  if (return_status != OptionStatus::kOk) return return_status;
  option.assignvalue(value);
  return OptionStatus::kOk;
}

OptionStatus setLocalOptionValue(const HighsLogOptions& report_log_options,
                                 OptionRecordDouble& option,
                                 const double value) {
  OptionStatus return_status =
      checkOptionValue(report_log_options, option, value);
  if (return_status != OptionStatus::kOk) return return_status;
  option.assignvalue(value);
  return OptionStatus::kOk;
}

OptionStatus setLocalOptionValue(const HighsLogOptions& report_log_options,
                                 OptionRecordString& option,
                                 const std::string value) {
  OptionStatus return_status =
      checkOptionValue(report_log_options, option, value);
  if (return_status != OptionStatus::kOk) return return_status;
  option.assignvalue(value);
  return OptionStatus::kOk;
}

OptionStatus passLocalOptions(const HighsLogOptions& report_log_options,
                              const HighsOptions& from_options,
                              HighsOptions& to_options) {
  // (Attempt to) set option value from the HighsOptions passed in
  OptionStatus return_status;
  //  std::string empty_file = "";
  //  std::string from_log_file = from_options.log_file;
  //  std::string original_to_log_file = to_options.log_file;
  //  FILE* original_to_log_file_stream =
  //  to_options.log_options.log_file_stream;
  HighsInt num_options = to_options.records.size();
  // Check all the option values before setting any of them - in case
  // to_options are the main Highs options. Checks are only needed for
  // HighsInt, double and string since bool values can't be illegal
  for (HighsInt index = 0; index < num_options; index++) {
    HighsOptionType type = to_options.records[index]->type;
    if (type == HighsOptionType::kInt) {
      HighsInt value =
          *(((OptionRecordInt*)from_options.records[index])[0].value);
      return_status = checkOptionValue(
          report_log_options, ((OptionRecordInt*)to_options.records[index])[0],
          value);
      if (return_status != OptionStatus::kOk) return return_status;
    } else if (type == HighsOptionType::kDouble) {
      double value =
          *(((OptionRecordDouble*)from_options.records[index])[0].value);
      return_status = checkOptionValue(
          report_log_options,
          ((OptionRecordDouble*)to_options.records[index])[0], value);
      if (return_status != OptionStatus::kOk) return return_status;
    } else if (type == HighsOptionType::kString) {
      std::string value =
          *(((OptionRecordString*)from_options.records[index])[0].value);
      return_status = checkOptionValue(
          report_log_options,
          ((OptionRecordString*)to_options.records[index])[0], value);
      if (return_status != OptionStatus::kOk) return return_status;
    }
  }
  // Checked from_options and found it to be OK, so set all the values
  for (HighsInt index = 0; index < num_options; index++) {
    HighsOptionType type = to_options.records[index]->type;
    if (type == HighsOptionType::kBool) {
      bool value = *(((OptionRecordBool*)from_options.records[index])[0].value);
      return_status = setLocalOptionValue(
          ((OptionRecordBool*)to_options.records[index])[0], value);
      if (return_status != OptionStatus::kOk) return return_status;
    } else if (type == HighsOptionType::kInt) {
      HighsInt value =
          *(((OptionRecordInt*)from_options.records[index])[0].value);
      return_status = setLocalOptionValue(
          report_log_options, ((OptionRecordInt*)to_options.records[index])[0],
          value);
      if (return_status != OptionStatus::kOk) return return_status;
    } else if (type == HighsOptionType::kDouble) {
      double value =
          *(((OptionRecordDouble*)from_options.records[index])[0].value);
      return_status = setLocalOptionValue(
          report_log_options,
          ((OptionRecordDouble*)to_options.records[index])[0], value);
      if (return_status != OptionStatus::kOk) return return_status;
    } else {
      std::string value =
          *(((OptionRecordString*)from_options.records[index])[0].value);
      return_status = setLocalOptionValue(
          report_log_options,
          ((OptionRecordString*)to_options.records[index])[0], value);
      if (return_status != OptionStatus::kOk) return return_status;
    }
  }
  /*
  if (from_log_file.compare(original_to_log_file)) {
    // The log file name has changed
    if (from_options.log_options.log_file_stream &&
        !original_to_log_file.compare(empty_file)) {
      // The stream corresponding to from_log_file is non-null and the
      // original log file name was empty, so to_options inherits the
      // stream, but associated with the (necessarily) non-empty name
      // from_log_file
      //
      // This ensures that the stream to Highs.log opened in
      // RunHighs.cpp is retained unless the log file name is changed.
      assert(from_log_file.compare(empty_file));
      assert(!original_to_log_file_stream);
      to_options.log_options.log_file_stream =
          from_options.log_options.log_file_stream;
    } else {
      highsOpenLogFile(to_options, to_options.log_file);
    }
  }
  */
  return OptionStatus::kOk;
}

OptionStatus getLocalOptionValue(
    const HighsLogOptions& report_log_options, const std::string& name,
    const std::vector<OptionRecord*>& option_records, bool& value) {
  HighsInt index;
  OptionStatus status =
      getOptionIndex(report_log_options, name, option_records, index);
  if (status != OptionStatus::kOk) return status;
  HighsOptionType type = option_records[index]->type;
  if (type != HighsOptionType::kBool) {
    highsLogUser(report_log_options, HighsLogType::kError,
                 "getLocalOptionValue: Option \"%s\" requires value of type "
                 "%s, not bool\n",
                 name.c_str(), optionEntryTypeToString(type).c_str());
    return OptionStatus::kIllegalValue;
  }
  OptionRecordBool option = ((OptionRecordBool*)option_records[index])[0];
  value = *option.value;
  return OptionStatus::kOk;
}

OptionStatus getLocalOptionValue(
    const HighsLogOptions& report_log_options, const std::string& name,
    const std::vector<OptionRecord*>& option_records, HighsInt& value) {
  HighsInt index;
  OptionStatus status =
      getOptionIndex(report_log_options, name, option_records, index);
  if (status != OptionStatus::kOk) return status;
  HighsOptionType type = option_records[index]->type;
  if (type != HighsOptionType::kInt) {
    highsLogUser(report_log_options, HighsLogType::kError,
                 "getLocalOptionValue: Option \"%s\" requires value of type "
                 "%s, not HighsInt\n",
                 name.c_str(), optionEntryTypeToString(type).c_str());
    return OptionStatus::kIllegalValue;
  }
  OptionRecordInt option = ((OptionRecordInt*)option_records[index])[0];
  value = *option.value;
  return OptionStatus::kOk;
}

OptionStatus getLocalOptionValue(
    const HighsLogOptions& report_log_options, const std::string& name,
    const std::vector<OptionRecord*>& option_records, double& value) {
  HighsInt index;
  OptionStatus status =
      getOptionIndex(report_log_options, name, option_records, index);
  if (status != OptionStatus::kOk) return status;
  HighsOptionType type = option_records[index]->type;
  if (type != HighsOptionType::kDouble) {
    highsLogUser(report_log_options, HighsLogType::kError,
                 "getLocalOptionValue: Option \"%s\" requires value of type "
                 "%s, not double\n",
                 name.c_str(), optionEntryTypeToString(type).c_str());
    return OptionStatus::kIllegalValue;
  }
  OptionRecordDouble option = ((OptionRecordDouble*)option_records[index])[0];
  value = *option.value;
  return OptionStatus::kOk;
}

OptionStatus getLocalOptionValue(
    const HighsLogOptions& report_log_options, const std::string& name,
    const std::vector<OptionRecord*>& option_records, std::string& value) {
  HighsInt index;
  OptionStatus status =
      getOptionIndex(report_log_options, name, option_records, index);
  if (status != OptionStatus::kOk) return status;
  HighsOptionType type = option_records[index]->type;
  if (type != HighsOptionType::kString) {
    highsLogUser(report_log_options, HighsLogType::kError,
                 "getLocalOptionValue: Option \"%s\" requires value of type "
                 "%s, not string\n",
                 name.c_str(), optionEntryTypeToString(type).c_str());
    return OptionStatus::kIllegalValue;
  }
  OptionRecordString option = ((OptionRecordString*)option_records[index])[0];
  value = *option.value;
  return OptionStatus::kOk;
}

OptionStatus getLocalOptionType(
    const HighsLogOptions& report_log_options, const std::string& name,
    const std::vector<OptionRecord*>& option_records, HighsOptionType& type) {
  HighsInt index;
  OptionStatus status =
      getOptionIndex(report_log_options, name, option_records, index);
  if (status != OptionStatus::kOk) return status;
  type = option_records[index]->type;
  return OptionStatus::kOk;
}

void resetLocalOptions(std::vector<OptionRecord*>& option_records) {
  HighsInt num_options = option_records.size();
  for (HighsInt index = 0; index < num_options; index++) {
    HighsOptionType type = option_records[index]->type;
    if (type == HighsOptionType::kBool) {
      OptionRecordBool& option = ((OptionRecordBool*)option_records[index])[0];
      *(option.value) = option.default_value;
    } else if (type == HighsOptionType::kInt) {
      OptionRecordInt& option = ((OptionRecordInt*)option_records[index])[0];
      *(option.value) = option.default_value;
    } else if (type == HighsOptionType::kDouble) {
      OptionRecordDouble& option =
          ((OptionRecordDouble*)option_records[index])[0];
      *(option.value) = option.default_value;
    } else {
      OptionRecordString& option =
          ((OptionRecordString*)option_records[index])[0];
      *(option.value) = option.default_value;
    }
  }
}

HighsStatus writeOptionsToFile(FILE* file,
                               const std::vector<OptionRecord*>& option_records,
                               const bool report_only_deviations,
                               const bool html) {
  if (html) {
    fprintf(file, "<!DOCTYPE HTML>\n<html>\n\n<head>\n");
    fprintf(file, "  <title>HiGHS Options</title>\n");
    fprintf(file, "	<meta charset=\"utf-8\" />\n");
    fprintf(file,
            "	<meta name=\"viewport\" content=\"width=device-width, "
            "initial-scale=1, user-scalable=no\" />\n");
    fprintf(file,
            "	<link rel=\"stylesheet\" href=\"assets/css/main.css\" />\n");
    fprintf(file, "</head>\n");
    fprintf(file, "<body style=\"background-color:f5fafa;\"></body>\n\n");
    fprintf(file, "<h3>HiGHS Options</h3>\n\n");
    fprintf(file, "<ul>\n");
  }
  reportOptions(file, option_records, report_only_deviations, html);
  if (html) {
    fprintf(file, "</ul>\n");
    fprintf(file, "</body>\n\n</html>\n");
  }
  return HighsStatus::kOk;
}

void reportOptions(FILE* file, const std::vector<OptionRecord*>& option_records,
                   const bool report_only_deviations, const bool html) {
  HighsInt num_options = option_records.size();
  for (HighsInt index = 0; index < num_options; index++) {
    HighsOptionType type = option_records[index]->type;
    //    fprintf(file, "\n# Option %1" HIGHSINT_FORMAT "\n", index);
    // Skip the advanced options when creating HTML
    if (html && option_records[index]->advanced) continue;
    if (type == HighsOptionType::kBool) {
      reportOption(file, ((OptionRecordBool*)option_records[index])[0],
                   report_only_deviations, html);
    } else if (type == HighsOptionType::kInt) {
      reportOption(file, ((OptionRecordInt*)option_records[index])[0],
                   report_only_deviations, html);
    } else if (type == HighsOptionType::kDouble) {
      reportOption(file, ((OptionRecordDouble*)option_records[index])[0],
                   report_only_deviations, html);
    } else {
      reportOption(file, ((OptionRecordString*)option_records[index])[0],
                   report_only_deviations, html);
    }
  }
}

void reportOption(FILE* file, const OptionRecordBool& option,
                  const bool report_only_deviations, const bool html) {
  if (!report_only_deviations || option.default_value != *option.value) {
    if (html) {
      fprintf(file,
              "<li><tt><font size=\"+2\"><strong>%s</strong></font></tt><br>\n",
              option.name.c_str());
      fprintf(file, "%s<br>\n", option.description.c_str());
      fprintf(file,
              "type: bool, advanced: %s, range: {false, true}, default: %s\n",
              highsBoolToString(option.advanced).c_str(),
              highsBoolToString(option.default_value).c_str());
      fprintf(file, "</li>\n");
    } else {
      fprintf(file, "\n# %s\n", option.description.c_str());
      fprintf(
          file,
          "# [type: bool, advanced: %s, range: {false, true}, default: %s]\n",
          highsBoolToString(option.advanced).c_str(),
          highsBoolToString(option.default_value).c_str());
      fprintf(file, "%s = %s\n", option.name.c_str(),
              highsBoolToString(*option.value).c_str());
    }
  }
}

void reportOption(FILE* file, const OptionRecordInt& option,
                  const bool report_only_deviations, const bool html) {
  if (!report_only_deviations || option.default_value != *option.value) {
    if (html) {
      fprintf(file,
              "<li><tt><font size=\"+2\"><strong>%s</strong></font></tt><br>\n",
              option.name.c_str());
      fprintf(file, "%s<br>\n", option.description.c_str());
      fprintf(file,
              "type: HighsInt, advanced: %s, range: {%" HIGHSINT_FORMAT
              ", %" HIGHSINT_FORMAT "}, default: %" HIGHSINT_FORMAT "\n",
              highsBoolToString(option.advanced).c_str(), option.lower_bound,
              option.upper_bound, option.default_value);
      fprintf(file, "</li>\n");
    } else {
      fprintf(file, "\n# %s\n", option.description.c_str());
      fprintf(file,
              "# [type: HighsInt, advanced: %s, range: {%" HIGHSINT_FORMAT
              ", %" HIGHSINT_FORMAT "}, default: %" HIGHSINT_FORMAT "]\n",
              highsBoolToString(option.advanced).c_str(), option.lower_bound,
              option.upper_bound, option.default_value);
      fprintf(file, "%s = %" HIGHSINT_FORMAT "\n", option.name.c_str(),
              *option.value);
    }
  }
}

void reportOption(FILE* file, const OptionRecordDouble& option,
                  const bool report_only_deviations, const bool html) {
  if (!report_only_deviations || option.default_value != *option.value) {
    if (html) {
      fprintf(file,
              "<li><tt><font size=\"+2\"><strong>%s</strong></font></tt><br>\n",
              option.name.c_str());
      fprintf(file, "%s<br>\n", option.description.c_str());
      fprintf(file,
              "type: double, advanced: %s, range: [%g, %g], default: %g\n",
              highsBoolToString(option.advanced).c_str(), option.lower_bound,
              option.upper_bound, option.default_value);
      fprintf(file, "</li>\n");
    } else {
      fprintf(file, "\n# %s\n", option.description.c_str());
      fprintf(file,
              "# [type: double, advanced: %s, range: [%g, %g], default: %g]\n",
              highsBoolToString(option.advanced).c_str(), option.lower_bound,
              option.upper_bound, option.default_value);
      fprintf(file, "%s = %g\n", option.name.c_str(), *option.value);
    }
  }
}

void reportOption(FILE* file, const OptionRecordString& option,
                  const bool report_only_deviations, const bool html) {
  // Don't report for the options file if writing to an options file
  if (option.name == kOptionsFileString) return;
  if (!report_only_deviations || option.default_value != *option.value) {
    if (html) {
      fprintf(file,
              "<li><tt><font size=\"+2\"><strong>%s</strong></font></tt><br>\n",
              option.name.c_str());
      fprintf(file, "%s<br>\n", option.description.c_str());
      fprintf(file, "type: string, advanced: %s, default: \"%s\"\n",
              highsBoolToString(option.advanced).c_str(),
              option.default_value.c_str());
      fprintf(file, "</li>\n");
    } else {
      fprintf(file, "\n# %s\n", option.description.c_str());
      fprintf(file, "# [type: string, advanced: %s, default: \"%s\"]\n",
              highsBoolToString(option.advanced).c_str(),
              option.default_value.c_str());
      fprintf(file, "%s = %s\n", option.name.c_str(), (*option.value).c_str());
    }
  }
}

void HighsOptions::setLogOptions() {
  this->log_options.output_flag = &this->output_flag;
  this->log_options.log_to_console = &this->log_to_console;
  this->log_options.log_dev_level = &this->log_dev_level;
}
