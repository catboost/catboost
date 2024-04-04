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
/**@file io/FilereaderMps.cpp
 * @brief
 */
#include "io/FilereaderMps.h"

#include "io/HMPSIO.h"
#include "io/HMpsFF.h"
#include "lp_data/HighsLp.h"
#include "lp_data/HighsLpUtils.h"
#include "lp_data/HighsModelUtils.h"

using free_format_parser::HMpsFF;

FilereaderRetcode FilereaderMps::readModelFromFile(const HighsOptions& options,
                                                   const std::string filename,
                                                   HighsModel& model) {
  HighsLp& lp = model.lp_;
  HighsHessian& hessian = model.hessian_;
  // if free format parser
  // Parse file and return status.
  if (options.mps_parser_type_free) {
    HMpsFF parser{};
    if (options.time_limit < kHighsInf && options.time_limit > 0)
      parser.time_limit = options.time_limit;

    FreeFormatParserReturnCode result =
        parser.loadProblem(options.log_options, filename, model);
    switch (result) {
      case FreeFormatParserReturnCode::kSuccess:
        lp.ensureColwise();
        assert(model.lp_.objective_name_ != "");
        return FilereaderRetcode::kOk;
      case FreeFormatParserReturnCode::kParserError:
        return FilereaderRetcode::kParserError;
      case FreeFormatParserReturnCode::kFileNotFound:
        return FilereaderRetcode::kFileNotFound;
      case FreeFormatParserReturnCode::kFixedFormat:
        highsLogUser(options.log_options, HighsLogType::kWarning,
                     "Free format reader has detected row/col names with "
                     "spaces: switching to fixed format parser\n");
        break;
      case FreeFormatParserReturnCode::kTimeout:
        highsLogUser(options.log_options, HighsLogType::kWarning,
                     "Free format reader reached time_limit while parsing "
                     "the input file\n");
        return FilereaderRetcode::kTimeout;
    }
  }

  // else use fixed format parser
  FilereaderRetcode return_code =
      readMps(options.log_options, filename, -1, -1, lp.num_row_, lp.num_col_,
              lp.sense_, lp.offset_, lp.a_matrix_.start_, lp.a_matrix_.index_,
              lp.a_matrix_.value_, lp.col_cost_, lp.col_lower_, lp.col_upper_,
              lp.row_lower_, lp.row_upper_, lp.integrality_, lp.objective_name_,
              lp.col_names_, lp.row_names_, hessian.dim_, hessian.start_,
              hessian.index_, hessian.value_, lp.cost_row_location_,
              options.keep_n_rows);
  if (return_code == FilereaderRetcode::kOk) lp.ensureColwise();
  // Comment on existence of names with spaces
  hasNamesWithSpaces(options.log_options, lp.num_col_, lp.col_names_);
  hasNamesWithSpaces(options.log_options, lp.num_row_, lp.row_names_);
  assert(model.lp_.objective_name_ != "");
  return return_code;
}

HighsStatus FilereaderMps::writeModelToFile(const HighsOptions& options,
                                            const std::string filename,
                                            const HighsModel& model) {
  assert(model.lp_.a_matrix_.isColwise());
  return writeModelAsMps(options, filename, model);
}
