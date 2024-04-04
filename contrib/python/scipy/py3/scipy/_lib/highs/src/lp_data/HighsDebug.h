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
/**@file lp_data/HighsDebug.h
 * @brief
 */
#ifndef SIMPLEX_HIGHSDEBUG_H_
#define SIMPLEX_HIGHSDEBUG_H_

#include <string>
#include <vector>

#include "io/HighsIO.h"
#include "lp_data/HConst.h"
#include "lp_data/HighsStatus.h"

//#include "lp_data/HighsOptions.h"

HighsStatus debugDebugToHighsStatus(const HighsDebugStatus debug_status);

HighsDebugStatus debugWorseStatus(const HighsDebugStatus status0,
                                  const HighsDebugStatus status1);

bool debugVectorRightSize(const std::vector<double> v,
                          const HighsInt right_size);

bool debugVectorRightSize(const std::vector<HighsInt> v,
                          const HighsInt right_size);

#endif  // SIMPLEX_HIGHSDEBUG_H_
