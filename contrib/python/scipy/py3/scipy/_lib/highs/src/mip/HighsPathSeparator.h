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
/**@file mip/HighsPathSeparator.h
 * @brief Class for separating cuts from heuristically aggregating rows from the
 * LP to indetify path's in a network
 *
 */

#ifndef MIP_HIGHS_PATH_SEPARATOR_H_
#define MIP_HIGHS_PATH_SEPARATOR_H_

#include "mip/HighsMipSolver.h"
#include "mip/HighsSeparator.h"
#include "util/HighsRandom.h"

/// Helper class to compute single-row relaxations from the current LP
/// relaxation by substituting bounds and aggregating rows
class HighsPathSeparator : public HighsSeparator {
 private:
  HighsRandom randgen;

 public:
  void separateLpSolution(HighsLpRelaxation& lpRelaxation,
                          HighsLpAggregator& lpAggregator,
                          HighsTransformedLp& transLp,
                          HighsCutPool& cutpool) override;

  HighsPathSeparator(const HighsMipSolver& mipsolver)
      : HighsSeparator(mipsolver, "PathAggr sepa", "Agg") {
    randgen.initialise(mipsolver.options_mip_->random_seed);
  }
};

#endif
