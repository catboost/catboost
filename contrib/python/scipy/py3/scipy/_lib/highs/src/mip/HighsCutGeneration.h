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
/**@file mip/HighsCutGeneration.h
 * @brief Class that generates cuts from single row relaxations
 *
 *
 */

#ifndef MIP_HIGHS_CUT_GENERATION_H_
#define MIP_HIGHS_CUT_GENERATION_H_

#include <cstdint>
#include <vector>

#include "util/HighsCDouble.h"
#include "util/HighsInt.h"
#include "util/HighsRandom.h"

class HighsLpRelaxation;
class HighsTransformedLp;
class HighsCutPool;
class HighsDomain;

/// Helper class to compute single-row relaxations from the current LP
/// relaxation by substituting bounds and aggregating rows
class HighsCutGeneration {
 private:
  const HighsLpRelaxation& lpRelaxation;
  HighsCutPool& cutpool;
  HighsRandom randgen;
  std::vector<HighsInt> cover;
  HighsCDouble coverweight;
  HighsCDouble lambda;
  std::vector<double> upper;
  std::vector<double> solval;
  std::vector<uint8_t> complementation;
  std::vector<uint8_t> isintegral;
  const double feastol;
  const double epsilon;

  double* vals;
  HighsInt* inds;
  HighsCDouble rhs;
  bool integralSupport;
  bool integralCoefficients;
  HighsInt rowlen;
  double initialScale;

  std::vector<HighsInt> integerinds;
  std::vector<double> deltas;

  bool determineCover(bool lpSol = true);

  void separateLiftedKnapsackCover();

  bool separateLiftedMixedBinaryCover();

  bool separateLiftedMixedIntegerCover();

  bool cmirCutGenerationHeuristic(double minEfficacy,
                                  bool onlyInitialCMIRScale = false);

  bool postprocessCut();

  bool preprocessBaseInequality(bool& hasUnboundedInts, bool& hasGeneralInts,
                                bool& hasContinuous);

 public:
  HighsCutGeneration(const HighsLpRelaxation& lpRelaxation,
                     HighsCutPool& cutpool);

  /// separates the LP solution for the given single row relaxation
  bool generateCut(HighsTransformedLp& transLp, std::vector<HighsInt>& inds,
                   std::vector<double>& vals, double& rhs,
                   bool onlyInitialCMIRScale = false);

  /// generate a conflict from the given proof constraint which cuts of the
  /// given local domain
  bool generateConflict(HighsDomain& localdom, std::vector<HighsInt>& proofinds,
                        std::vector<double>& proofvals, double& proofrhs);

  /// applies postprocessing to an externally generated cut and adds it to the
  /// cutpool if it is violated enough
  bool finalizeAndAddCut(std::vector<HighsInt>& inds, std::vector<double>& vals,
                         double& rhs);
};

#endif
