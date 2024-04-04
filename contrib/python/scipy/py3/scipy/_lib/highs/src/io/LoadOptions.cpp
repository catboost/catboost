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
#include "io/LoadOptions.h"

#include <fstream>

#include "util/stringutil.h"

// For extended options to be parsed from a file. Assuming options file is
// specified.
bool loadOptionsFromFile(const HighsLogOptions& report_log_options,
                         HighsOptions& options, const std::string filename) {
  if (filename.size() == 0) return false;

  string line, option, value;
  HighsInt line_count = 0;
  // loadOptionsFromFile needs its own non-chars string since the
  // default setting in io/stringutil.h excludes \" and \' that can
  // appear in an MPS name - the only other place where trim() is used
  const std::string non_chars = "\t\n\v\f\r\"\' ";
  std::ifstream file(filename);
  if (file.is_open()) {
    while (file.good()) {
      getline(file, line);
      line_count++;
      if (line.size() == 0 || line[0] == '#') continue;

      HighsInt equals = line.find_first_of("=");
      if (equals < 0 || equals >= (HighsInt)line.size() - 1) {
        highsLogUser(report_log_options, HighsLogType::kError,
                     "Error on line %" HIGHSINT_FORMAT " of options file.\n",
                     line_count);
        return false;
      }
      option = line.substr(0, equals);
      value = line.substr(equals + 1, line.size() - equals);
      trim(option, non_chars);
      trim(value, non_chars);
      if (setLocalOptionValue(report_log_options, option, options.log_options,
                              options.records, value) != OptionStatus::kOk)
        return false;
    }
  } else {
    highsLogUser(report_log_options, HighsLogType::kError,
                 "Options file not found.\n");
    return false;
  }

  return true;
}
