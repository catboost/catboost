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
/**@file ipm/IpxSolution.h
 * @brief
 */
#ifndef IPM_IPX_SOLUTION_H_
#define IPM_IPX_SOLUTION_H_

#include <stdint.h>

#include <vector>

#include "util/HighsInt.h"
typedef HighsInt ipxint;

struct IpxSolution {
  ipxint num_col;
  ipxint num_row;
  std::vector<double> ipx_col_value;
  std::vector<double> ipx_row_value;
  std::vector<double> ipx_col_dual;
  std::vector<double> ipx_row_dual;
  std::vector<ipxint> ipx_col_status;
  std::vector<ipxint> ipx_row_status;
};

#endif
