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
#ifndef HIGHS_SEPARATION_H_
#define HIGHS_SEPARATION_H_

#include <cstdint>
#include <vector>

#include "mip/HighsCutPool.h"
#include "mip/HighsLpRelaxation.h"
#include "mip/HighsSeparator.h"

class HighsMipSolver;
class HighsImplications;
class HighsCliqueTable;

class HighsSeparation {
 public:
  HighsInt separationRound(HighsDomain& propdomain,
                           HighsLpRelaxation::Status& status);

  void separate(HighsDomain& propdomain);

  void setLpRelaxation(HighsLpRelaxation* lp) { this->lp = lp; }

  HighsSeparation(const HighsMipSolver& mipsolver);

 private:
  HighsInt implBoundClock;
  HighsInt cliqueClock;
  std::vector<std::unique_ptr<HighsSeparator>> separators;
  HighsCutSet cutset;
  HighsLpRelaxation* lp;
};

#endif
