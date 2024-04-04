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
/**@file test/KktChStep.h
 * @brief
 */
#ifndef TEST_KKTCH2_H_
#define TEST_KKTCH2_H_

#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stack>
#include <string>
#include <vector>

#include "lp_data/HConst.h"
#include "test/DevKkt.h"
#include "util/HighsInt.h"

namespace presolve {

namespace dev_kkt_check {

class KktCheck;

class KktChStep {
 public:
  KktChStep() {}
  virtual ~KktChStep() {}

  std::vector<double> RcolCost;
  std::vector<double> RcolLower;
  std::vector<double> RcolUpper;
  std::vector<double> RrowLower;
  std::vector<double> RrowUpper;

  int print = 1;

  std::stack<std::vector<std::pair<HighsInt, double> > > rLowers;
  std::stack<std::vector<std::pair<HighsInt, double> > > rUppers;
  std::stack<std::vector<std::pair<HighsInt, double> > > cLowers;
  std::stack<std::vector<std::pair<HighsInt, double> > > cUppers;
  std::stack<std::vector<std::pair<HighsInt, double> > > costs;

  // full matrix
  void setBoundsCostRHS(const std::vector<double>& colUpper_,
                        const std::vector<double>& colLower_,
                        const std::vector<double>& cost,
                        const std::vector<double>& rowLower_,
                        const std::vector<double>& rowUpper_);
  void addChange(int type, HighsInt row, HighsInt col, double valC,
                 double dualC, double dualR);
  void addCost(HighsInt col, double value);

  dev_kkt_check::State initState(
      const HighsInt numCol_, const HighsInt numRow_,
      const std::vector<HighsInt>& Astart_, const std::vector<HighsInt>& Aend_,
      const std::vector<HighsInt>& Aindex_, const std::vector<double>& Avalue_,
      const std::vector<HighsInt>& ARstart_,
      const std::vector<HighsInt>& ARindex_,
      const std::vector<double>& ARvalue_,
      const std::vector<HighsInt>& flagCol_,
      const std::vector<HighsInt>& flagRow_,
      const std::vector<double>& colValue_, const std::vector<double>& colDual_,
      const std::vector<double>& rowValue_, const std::vector<double>& rowDual_,
      const std::vector<HighsBasisStatus>& col_status_,
      const std::vector<HighsBasisStatus>& row_status_);
};

}  // namespace dev_kkt_check

}  // namespace presolve
#endif /* TEST_KKTCHSTEP_H_ */
