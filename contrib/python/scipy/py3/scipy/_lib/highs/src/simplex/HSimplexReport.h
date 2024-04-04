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
/**@file lp_data/HSimplexReport.h
 * @brief
 */
#ifndef SIMPLEX_HSIMPLEXREPORT_H_
#define SIMPLEX_HSIMPLEXREPORT_H_

#include "lp_data/HighsOptions.h"
#include "simplex/HSimplex.h"

void reportSimplexPhaseIterations(const HighsLogOptions& log_options,
                                  const HighsInt iteration_count,
                                  const HighsSimplexInfo& info,
                                  const bool initialise = false);
#endif  // SIMPLEX_HSIMPLEXREPORT_H_
