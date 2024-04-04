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
/**@file io/HighsIO.h
 * @brief IO methods for HiGHS - currently just print/log messages
 */
#ifndef HIGHS_IO_H
#define HIGHS_IO_H

#include <array>
#include <iostream>

#include "util/HighsInt.h"

class HighsOptions;

const HighsInt kIoBufferSize = 1024;  // 65536;

/**
 * @brief IO methods for HiGHS - currently just print/log messages
 */
enum class HighsLogType { kInfo = 1, kDetailed, kVerbose, kWarning, kError };

const char* const HighsLogTypeTag[] = {"", "",          "",
                                       "", "WARNING: ", "ERROR:   "};
enum LogDevLevel {
  kHighsLogDevLevelMin = 0,
  kHighsLogDevLevelNone = kHighsLogDevLevelMin,  // 0
  kHighsLogDevLevelInfo,                         // 1
  kHighsLogDevLevelDetailed,                     // 2
  kHighsLogDevLevelVerbose,                      // 3
  kHighsLogDevLevelMax = kHighsLogDevLevelVerbose
};

struct HighsLogOptions {
  FILE* log_file_stream;
  bool* output_flag;
  bool* log_to_console;
  HighsInt* log_dev_level;
  void (*log_callback)(HighsLogType, const char*, void*) = nullptr;
  void* log_callback_data = nullptr;
};

/**
 * @brief Write the HiGHS version, compilation date, git hash and
 * copyright statement
 */
void highsLogHeader(const HighsLogOptions& log_options);

/**
 * @brief Convert a double number to a string using given tolerance
 */
std::array<char, 32> highsDoubleToString(double val, double tolerance);

/**
 * @brief For _single-line_ user logging with message type notification
 */
// Printing format: must contain exactly one "\n" at end of format
void highsLogUser(const HighsLogOptions& log_options_, const HighsLogType type,
                  const char* format, ...);

/**
 * @brief For development logging
 */
void highsLogDev(const HighsLogOptions& log_options_, const HighsLogType type,
                 const char* format, ...);

/**
 * @brief For development logging when true log_options may not be available -
 * indicated by null pointer
 */
void highsReportDevInfo(const HighsLogOptions* log_options,
                        const std::string line);

void highsOpenLogFile(HighsOptions& options, const std::string log_file);

void highsReportLogOptions(const HighsLogOptions& log_options_);

std::string highsFormatToString(const char* format, ...);

const std::string highsBoolToString(const bool b);

#endif
