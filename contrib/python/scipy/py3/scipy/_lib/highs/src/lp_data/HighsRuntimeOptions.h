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

#ifndef LP_DATA_HIGHSRUNTIMEOPTIONS_H_
#define LP_DATA_HIGHSRUNTIMEOPTIONS_H_

#include "cxxopts.hpp"
#include "io/HighsIO.h"
#include "io/LoadOptions.h"
#include "util/stringutil.h"

bool loadOptions(const HighsLogOptions& report_log_options, int argc,
                 char** argv, HighsOptions& options, std::string& model_file) {
  try {
    cxxopts::Options cxx_options(argv[0], "HiGHS options");
    cxx_options.positional_help("[file]").show_positional_help();

    std::string presolve, solver, parallel, ranging;

    // clang-format off
    cxx_options.add_options()
        (kModelFileString,
        "File of model to solve.",
        cxxopts::value<std::vector<std::string>>())
        (kPresolveString,
        "Presolve: \"choose\" by default - \"on\"/\"off\" are alternatives.",
        cxxopts::value<std::string>(presolve))
        (kSolverString,
        "Solver: \"choose\" by default - \"simplex\"/\"ipm\" are alternatives.",
        cxxopts::value<std::string>(solver))
        (kParallelString,
        "Parallel solve: \"choose\" by default - \"on\"/\"off\" are alternatives.",
        cxxopts::value<std::string>(parallel))
        (kTimeLimitString,
        "Run time limit (seconds - double).",
        cxxopts::value<double>())
        (kOptionsFileString,
        "File containing HiGHS options.",
        cxxopts::value<std::vector<std::string>>())
        (kSolutionFileString,
        "File for writing out model solution.",
        cxxopts::value<std::vector<std::string>>())
        (kWriteModelFileString,
        "File for writing out model.",
        cxxopts::value<std::vector<std::string>>())
        (kRandomSeedString,
        "Seed to initialize random number generation.",
        cxxopts::value<HighsInt>())
        (kRangingString,
        "Compute cost, bound, RHS and basic solution ranging.",
        cxxopts::value<std::string>(ranging))
        ("h, help", "Print help.");
    // clang-format on
    cxx_options.parse_positional("model_file");

    auto result = cxx_options.parse(argc, argv);

    if (result.count("help")) {
      std::cout << cxx_options.help({""}) << std::endl;
      exit(0);
    }
    if (result.count(kModelFileString)) {
      auto& v = result[kModelFileString].as<std::vector<std::string>>();
      if (v.size() > 1) {
        HighsInt nonEmpty = 0;
        for (HighsInt i = 0; i < (HighsInt)v.size(); i++) {
          std::string arg = v[i];
          if (trim(arg).size() > 0) {
            nonEmpty++;
            model_file = arg;
          }
        }
        if (nonEmpty > 1) {
          std::cout << "Multiple files not implemented.\n";
          return false;
        }
      } else {
        model_file = v[0];
      }
    }

    if (result.count(kPresolveString)) {
      std::string value = result[kPresolveString].as<std::string>();
      if (setLocalOptionValue(report_log_options, kPresolveString,
                              options.log_options, options.records,
                              value) != OptionStatus::kOk)
        return false;
    }

    if (result.count(kSolverString)) {
      std::string value = result[kSolverString].as<std::string>();
      if (setLocalOptionValue(report_log_options, kSolverString,
                              options.log_options, options.records,
                              value) != OptionStatus::kOk)
        return false;
    }

    if (result.count(kParallelString)) {
      std::string value = result[kParallelString].as<std::string>();
      if (setLocalOptionValue(report_log_options, kParallelString,
                              options.log_options, options.records,
                              value) != OptionStatus::kOk)
        return false;
    }

    if (result.count(kTimeLimitString)) {
      double value = result[kTimeLimitString].as<double>();
      if (setLocalOptionValue(report_log_options, kTimeLimitString,
                              options.records, value) != OptionStatus::kOk)
        return false;
    }

    if (result.count(kOptionsFileString)) {
      auto& v = result[kOptionsFileString].as<std::vector<std::string>>();
      if (v.size() > 1) {
        std::cout << "Multiple options files not implemented.\n";
        return false;
      }
      if (!loadOptionsFromFile(report_log_options, options, v[0])) return false;
    }

    if (result.count(kSolutionFileString)) {
      auto& v = result[kSolutionFileString].as<std::vector<std::string>>();
      if (v.size() > 1) {
        std::cout << "Multiple solution files not implemented.\n";
        return false;
      }
      if (setLocalOptionValue(report_log_options, kSolutionFileString,
                              options.log_options, options.records,
                              v[0]) != OptionStatus::kOk ||
          setLocalOptionValue(report_log_options, "write_solution_to_file",
                              options.records, true) != OptionStatus::kOk)
        return false;
    }

    if (result.count(kWriteModelFileString)) {
      auto& v = result[kWriteModelFileString].as<std::vector<std::string>>();
      if (v.size() > 1) {
        std::cout << "Multiple write model files not implemented.\n";
        return false;
      }
      if (setLocalOptionValue(report_log_options, kWriteModelFileString,
                              options.log_options, options.records,
                              v[0]) != OptionStatus::kOk ||
          setLocalOptionValue(report_log_options, "write_model_to_file",
                              options.records, true) != OptionStatus::kOk)
        return false;
    }

    if (result.count(kRandomSeedString)) {
      HighsInt value = result[kRandomSeedString].as<HighsInt>();
      if (setLocalOptionValue(report_log_options, kRandomSeedString,
                              options.records, value) != OptionStatus::kOk)
        return false;
    }

    if (result.count(kRangingString)) {
      std::string value = result[kRangingString].as<std::string>();
      if (setLocalOptionValue(report_log_options, kRangingString,
                              options.log_options, options.records,
                              value) != OptionStatus::kOk)
        return false;
    }

  } catch (const cxxopts::OptionException& e) {
    highsLogUser(report_log_options, HighsLogType::kError,
                 "Error parsing options: %s\n", e.what());
    return false;
  }

  const bool horrible_hack_for_windows_visual_studio = false;
  if (horrible_hack_for_windows_visual_studio) {
    // Until I know how to debug an executable using command line
    // arguments in Visual Studio on Windows, this is necessary!
    HighsInt random_seed = -3;
    if (random_seed >= 0) {
      if (setLocalOptionValue(report_log_options, kRandomSeedString,
                              options.records,
                              random_seed) != OptionStatus::kOk)
        return false;
    }
    model_file = "ml.mps";
  }

  if (model_file.size() == 0) {
    std::cout << "Please specify filename in .mps|.lp|.ems format.\n";
    return false;
  }

  return true;
}

#endif
