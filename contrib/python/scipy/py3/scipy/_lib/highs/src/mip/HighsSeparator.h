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
/**@file mip/HighsSeparator.h
 * @brief Base class for separators
 *
 */

#ifndef MIP_HIGHS_SEPARATOR_H_
#define MIP_HIGHS_SEPARATOR_H_

#include "util/HighsInt.h"

class HighsLpRelaxation;
class HighsTransformedLp;
class HighsCutPool;
class HighsLpAggregator;
class HighsMipSolver;

/// Helper class to compute single-row relaxations from the current LP
/// relaxation by substituting bounds and aggregating rows
class HighsSeparator {
 private:
  HighsInt numCutsFound;
  HighsInt numCalls;
  int clockIndex;

 public:
  HighsSeparator(const HighsMipSolver& mipsolver, const char* name,
                 const char* ch3_name);

  virtual void separateLpSolution(HighsLpRelaxation& lpRelaxation,
                                  HighsLpAggregator& lpAggregator,
                                  HighsTransformedLp& transLp,
                                  HighsCutPool& cutpool) = 0;

  void run(HighsLpRelaxation& lpRelaxation, HighsLpAggregator& lpAggregator,
           HighsTransformedLp& transLp, HighsCutPool& cutpool);

  HighsInt getNumCutsFound() const { return numCutsFound; }

  HighsInt getNumCalls() const { return numCalls; }

  HighsInt getClockIndex() const { return clockIndex; }

  virtual ~HighsSeparator() {}
};

#endif
