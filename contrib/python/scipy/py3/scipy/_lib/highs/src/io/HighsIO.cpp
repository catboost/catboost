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
/**@file io/HighsIO.cpp
 * @brief IO methods for HiGHS - currently just print/log messages
 */
#include "io/HighsIO.h"

#include <cmath>
#include <cstdarg>
#include <cstdio>

#include "lp_data/HighsLp.h"
#include "lp_data/HighsOptions.h"

void highsLogHeader(const HighsLogOptions& log_options) {
  highsLogUser(log_options, HighsLogType::kInfo,
               "Running HiGHS %d.%d.%d [date: %s, git hash: %s]\n",
               (int)HIGHS_VERSION_MAJOR, (int)HIGHS_VERSION_MINOR,
               (int)HIGHS_VERSION_PATCH, HIGHS_COMPILATION_DATE, HIGHS_GITHASH);
  highsLogUser(log_options, HighsLogType::kInfo,
               "Copyright (c) 2022 ERGO-Code under MIT licence terms\n");
}

std::array<char, 32> highsDoubleToString(double val, double tolerance) {
  std::array<char, 32> printString;
  double l =
      std::abs(val) == kHighsInf
          ? 1.0
          : (1.0 - tolerance +
             std::log10(std::max(tolerance, std::abs(val)) / (tolerance)));
  switch (int(l)) {
    case 0:
      std::snprintf(printString.data(), 32, "%c", '0');
      break;
    case 1:
      std::snprintf(printString.data(), 32, "%.1g", val);
      break;
    case 2:
      std::snprintf(printString.data(), 32, "%.2g", val);
      break;
    case 3:
      std::snprintf(printString.data(), 32, "%.3g", val);
      break;
    case 4:
      std::snprintf(printString.data(), 32, "%.4g", val);
      break;
    case 5:
      std::snprintf(printString.data(), 32, "%.5g", val);
      break;
    case 6:
      std::snprintf(printString.data(), 32, "%.6g", val);
      break;
    case 7:
      std::snprintf(printString.data(), 32, "%.7g", val);
      break;
    case 8:
      std::snprintf(printString.data(), 32, "%.8g", val);
      break;
    case 9:
      std::snprintf(printString.data(), 32, "%.9g", val);
      break;
    case 10:
      std::snprintf(printString.data(), 32, "%.10g", val);
      break;
    case 11:
      std::snprintf(printString.data(), 32, "%.11g", val);
      break;
    case 12:
      std::snprintf(printString.data(), 32, "%.12g", val);
      break;
    case 13:
      std::snprintf(printString.data(), 32, "%.13g", val);
      break;
    case 14:
      std::snprintf(printString.data(), 32, "%.14g", val);
      break;
    case 15:
      std::snprintf(printString.data(), 32, "%.15g", val);
      break;
    default:
      std::snprintf(printString.data(), 32, "%.16g", val);
  }

  return printString;
}

void highsLogUser(const HighsLogOptions& log_options_, const HighsLogType type,
                  const char* format, ...) {
  if (!*log_options_.output_flag ||
      (log_options_.log_file_stream == NULL && !*log_options_.log_to_console))
    return;
  // highsLogUser should not be passed HighsLogType::kDetailed or
  // HighsLogType::kVerbose
  assert(type != HighsLogType::kDetailed);
  assert(type != HighsLogType::kVerbose);
  const bool prefix =
      type == HighsLogType::kWarning || type == HighsLogType::kError;
  va_list argptr;
  va_start(argptr, format);
  const bool flush_streams = true;
  if (!log_options_.log_callback) {
    // Write to log file stream unless it is NULL
    if (log_options_.log_file_stream) {
      if (prefix)
        fprintf(log_options_.log_file_stream, "%-9s",
                HighsLogTypeTag[(int)type]);
      vfprintf(log_options_.log_file_stream, format, argptr);
      if (flush_streams) fflush(log_options_.log_file_stream);
      va_start(argptr, format);
    }
    // Write to stdout unless log file stream is stdout
    if (*log_options_.log_to_console &&
        log_options_.log_file_stream != stdout) {
      if (prefix) fprintf(stdout, "%-9s", HighsLogTypeTag[(int)type]);
      vfprintf(stdout, format, argptr);
      if (flush_streams) fflush(stdout);
    }
  } else {
    int len = 0;
    char msgbuffer[kIoBufferSize];
    if (prefix)
      len = snprintf(msgbuffer, sizeof(msgbuffer), "%-9s",
                     HighsLogTypeTag[(int)type]);
    if (len < (int)sizeof(msgbuffer))
      len +=
          vsnprintf(msgbuffer + len, sizeof(msgbuffer) - len, format, argptr);
    if (len >= (int)sizeof(msgbuffer)) {
      // Output was truncated: for now just ensure string is null-terminated
      msgbuffer[sizeof(msgbuffer) - 1] = '\0';
    }
    log_options_.log_callback(type, msgbuffer, log_options_.log_callback_data);
  }
  va_end(argptr);
}

void highsLogDev(const HighsLogOptions& log_options_, const HighsLogType type,
                 const char* format, ...) {
  if (!*log_options_.output_flag ||
      (log_options_.log_file_stream == NULL && !*log_options_.log_to_console) ||
      !*log_options_.log_dev_level)
    return;
  // Always report HighsLogType::kInfo, HighsLogType::kWarning or
  // HighsLogType::kError
  //
  // Report HighsLogType::kDetailed if *log_options_.log_dev_level >=
  // kHighsLogDevLevelDetailed
  //
  // Report HighsLogType::kVerbose if *log_options_.log_dev_level >=
  // kHighsLogDevLevelVerbose
  if (type == HighsLogType::kDetailed &&
      *log_options_.log_dev_level < kHighsLogDevLevelDetailed)
    return;
  if (type == HighsLogType::kVerbose &&
      *log_options_.log_dev_level < kHighsLogDevLevelVerbose)
    return;
  va_list argptr;
  va_start(argptr, format);
  const bool flush_streams = true;
  if (!log_options_.log_callback) {
    // Write to log file stream unless it is NULL
    if (log_options_.log_file_stream) {
      // Write to log file stream
      vfprintf(log_options_.log_file_stream, format, argptr);
      if (flush_streams) fflush(log_options_.log_file_stream);
      va_start(argptr, format);
    }
    // Write to stdout unless log file stream is stdout
    if (*log_options_.log_to_console &&
        log_options_.log_file_stream != stdout) {
      vfprintf(stdout, format, argptr);
      if (flush_streams) fflush(stdout);
    }
  } else {
    int len;
    char msgbuffer[kIoBufferSize];
    len = vsnprintf(msgbuffer, sizeof(msgbuffer), format, argptr);
    if (len >= (int)sizeof(msgbuffer)) {
      // Output was truncated: for now just ensure string is null-terminated
      msgbuffer[sizeof(msgbuffer) - 1] = '\0';
    }
    log_options_.log_callback(type, msgbuffer, log_options_.log_callback_data);
  }
  va_end(argptr);
}

void highsReportDevInfo(const HighsLogOptions* log_options,
                        const std::string line) {
  if (log_options) {
    highsLogDev(*log_options, HighsLogType::kInfo, "%s", line.c_str());
  } else {
    printf("%s", line.c_str());
  }
}

void highsOpenLogFile(HighsOptions& options, const std::string log_file) {
  highsOpenLogFile(options.log_options, options.records, log_file);
}

void highsReportLogOptions(const HighsLogOptions& log_options_) {
  printf("\nHighs log options\n");
  if (log_options_.log_file_stream == NULL) {
    printf("   log_file_stream = NULL\n");
  } else {
    printf("   log_file_stream = Not NULL\n");
  }
  printf("   output_flag = %s\n",
         highsBoolToString(*log_options_.output_flag).c_str());
  printf("   log_to_console = %s\n",
         highsBoolToString(*log_options_.log_to_console).c_str());
  printf("   log_dev_level = %" HIGHSINT_FORMAT "\n\n",
         *log_options_.log_dev_level);
}

std::string highsFormatToString(const char* format, ...) {
  va_list argptr;
  va_start(argptr, format);
  int len;
  char msgbuffer[kIoBufferSize];
  len = vsnprintf(msgbuffer, sizeof(msgbuffer), format, argptr);

  if (len >= (int)sizeof(msgbuffer)) {
    // Output was truncated: for now just ensure string is null-terminated
    msgbuffer[sizeof(msgbuffer) - 1] = '\0';
  }
  va_end(argptr);
  return std::string(msgbuffer);
}

const std::string highsBoolToString(const bool b) {
  return b ? "true" : "false";
}
