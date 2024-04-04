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
/**@file lp_data/HighsInfoDebug.h
 * @brief
 */
#ifndef LP_DATA_HIGHS_INFO_DEBUG_H_
#define LP_DATA_HIGHS_INFO_DEBUG_H_

#include "lp_data/HStruct.h"
#include "lp_data/HighsInfo.h"
#include "lp_data/HighsLp.h"
#include "lp_data/HighsOptions.h"
//#include "lp_data/HighsLp.h"

HighsDebugStatus debugInfo(const HighsOptions& options, const HighsLp& lp,
                           const HighsBasis& basis,
                           const HighsSolution& solution, const HighsInfo& info,
                           const HighsModelStatus model_status);

HighsDebugStatus debugNoInfo(const HighsInfo& info);

#endif
