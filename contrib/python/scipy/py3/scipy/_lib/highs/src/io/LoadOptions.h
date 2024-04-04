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
/**@file io/LoadOptions.h
 * @brief
 */

#ifndef IO_LOAD_OPTIONS_H_
#define IO_LOAD_OPTIONS_H_

#include "lp_data/HighsOptions.h"

// For extended options to be parsed from filename
bool loadOptionsFromFile(const HighsLogOptions& report_log_options,
                         HighsOptions& options, const std::string filename);

#endif
