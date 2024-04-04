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
#ifndef LP_DATA_HIGHS_STATUS_H_
#define LP_DATA_HIGHS_STATUS_H_

#include <string>

#include "io/HighsIO.h"

enum class HighsStatus { kError = -1, kOk = 0, kWarning = 1 };

// Return a string representation of HighsStatus.
std::string highsStatusToString(HighsStatus status);

// Return the maximum of two HighsStatus and possibly report on
// call_status not being HighsStatus::kOk
HighsStatus interpretCallStatus(const HighsLogOptions log_options,
                                const HighsStatus call_status,
                                const HighsStatus from_return_status,
                                const std::string& message = "");

// Return the maximum of two HighsStatus
HighsStatus worseStatus(HighsStatus status0, HighsStatus status1);
#endif
