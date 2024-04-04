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
#include "mip/HighsSeparator.h"

#include <string>

#include "mip/HighsCutPool.h"
#include "mip/HighsLpRelaxation.h"
#include "mip/HighsMipSolver.h"

HighsSeparator::HighsSeparator(const HighsMipSolver& mipsolver,
                               const char* name, const char* ch3_name)
    : numCutsFound(0), numCalls(0) {
  clockIndex = mipsolver.timer_.clock_def(name, ch3_name);
}

void HighsSeparator::run(HighsLpRelaxation& lpRelaxation,
                         HighsLpAggregator& lpAggregator,
                         HighsTransformedLp& transLp, HighsCutPool& cutpool) {
  ++numCalls;
  HighsInt currNumCuts = cutpool.getNumCuts();

  lpRelaxation.getMipSolver().timer_.start(clockIndex);
  separateLpSolution(lpRelaxation, lpAggregator, transLp, cutpool);
  lpRelaxation.getMipSolver().timer_.stop(clockIndex);

  numCutsFound += cutpool.getNumCuts() - currNumCuts;
}
