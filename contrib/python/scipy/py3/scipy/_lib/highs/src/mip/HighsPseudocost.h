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

#ifndef HIGHS_PSEUDOCOST_H_
#define HIGHS_PSEUDOCOST_H_

#include <algorithm>
#include <cassert>
#include <cmath>
#include <limits>
#include <vector>

#include "util/HighsInt.h"

class HighsMipSolver;
namespace presolve {
class HighsPostsolveStack;
}

class HighsPseudocost;

struct HighsPseudocostInitialization {
  std::vector<double> pseudocostup;
  std::vector<double> pseudocostdown;
  std::vector<HighsInt> nsamplesup;
  std::vector<HighsInt> nsamplesdown;
  std::vector<double> inferencesup;
  std::vector<double> inferencesdown;
  std::vector<HighsInt> ninferencesup;
  std::vector<HighsInt> ninferencesdown;
  std::vector<double> conflictscoreup;
  std::vector<double> conflictscoredown;
  double cost_total;
  double inferences_total;
  double conflict_avg_score;
  int64_t nsamplestotal;
  int64_t ninferencestotal;

  HighsPseudocostInitialization(const HighsPseudocost& pscost,
                                HighsInt maxCount);
  HighsPseudocostInitialization(
      const HighsPseudocost& pscost, HighsInt maxCount,
      const presolve::HighsPostsolveStack& postsolveStack);
};
class HighsPseudocost {
  friend struct HighsPseudocostInitialization;
  std::vector<double> pseudocostup;
  std::vector<double> pseudocostdown;
  std::vector<HighsInt> nsamplesup;
  std::vector<HighsInt> nsamplesdown;
  std::vector<double> inferencesup;
  std::vector<double> inferencesdown;
  std::vector<HighsInt> ninferencesup;
  std::vector<HighsInt> ninferencesdown;
  std::vector<HighsInt> ncutoffsup;
  std::vector<HighsInt> ncutoffsdown;
  std::vector<double> conflictscoreup;
  std::vector<double> conflictscoredown;

  double conflict_weight;
  double conflict_avg_score;
  double cost_total;
  double inferences_total;
  int64_t nsamplestotal;
  int64_t ninferencestotal;
  int64_t ncutoffstotal;
  HighsInt minreliable;
  double degeneracyFactor;

 public:
  HighsPseudocost() = default;
  HighsPseudocost(const HighsMipSolver& mipsolver);

  void subtractBase(const HighsPseudocost& base) {
    HighsInt ncols = pseudocostup.size();

    for (HighsInt i = 0; i != ncols; ++i) {
      pseudocostup[i] -= base.pseudocostup[i];
      pseudocostdown[i] -= base.pseudocostdown[i];
      nsamplesup[i] -= base.nsamplesup[i];
      nsamplesdown[i] -= base.nsamplesdown[i];
    }
  }

  void increaseConflictWeight() {
    conflict_weight *= 1.02;

    if (conflict_weight > 1000.0) {
      double scale = 1.0 / conflict_weight;
      conflict_weight = 1.0;
      conflict_avg_score *= scale;

      HighsInt numCol = conflictscoreup.size();
      for (HighsInt i = 0; i < numCol; ++i) {
        conflictscoreup[i] *= scale;
        conflictscoredown[i] *= scale;
      }
    }
  }

  void setDegeneracyFactor(double degeneracyFactor) {
    assert(degeneracyFactor >= 1.0);
    this->degeneracyFactor = degeneracyFactor;
  }

  void increaseConflictScoreUp(HighsInt col) {
    conflictscoreup[col] += conflict_weight;
    conflict_avg_score += conflict_weight;
  }

  void increaseConflictScoreDown(HighsInt col) {
    conflictscoredown[col] += conflict_weight;
    conflict_avg_score += conflict_weight;
  }

  void setMinReliable(HighsInt minreliable) { this->minreliable = minreliable; }

  HighsInt getMinReliable() const { return minreliable; }

  HighsInt getNumObservations(HighsInt col) const {
    return nsamplesup[col] + nsamplesdown[col];
  }

  HighsInt getNumObservationsUp(HighsInt col) const { return nsamplesup[col]; }

  HighsInt getNumObservationsDown(HighsInt col) const {
    return nsamplesdown[col];
  }

  void addCutoffObservation(HighsInt col, bool upbranch) {
    ++ncutoffstotal;
    if (upbranch)
      ncutoffsup[col] += 1;
    else
      ncutoffsdown[col] += 1;
  }

  void addObservation(HighsInt col, double delta, double objdelta) {
    assert(delta != 0.0);
    assert(objdelta >= 0.0);
    if (delta > 0.0) {
      double unit_gain = objdelta / delta;
      double d = unit_gain - pseudocostup[col];
      nsamplesup[col] += 1;
      pseudocostup[col] += d / nsamplesup[col];

      d = unit_gain - cost_total;
      ++nsamplestotal;
      cost_total += d / nsamplestotal;
    } else {
      double unit_gain = -objdelta / delta;
      double d = unit_gain - pseudocostdown[col];
      nsamplesdown[col] += 1;
      pseudocostdown[col] += d / nsamplesdown[col];

      d = unit_gain - cost_total;
      ++nsamplestotal;
      cost_total += d / nsamplestotal;
    }
  }

  void addInferenceObservation(HighsInt col, HighsInt ninferences,
                               bool upbranch) {
    double d = ninferences - inferences_total;
    ++ninferencestotal;
    inferences_total += d / ninferencestotal;
    if (upbranch) {
      d = ninferences - inferencesup[col];
      ninferencesup[col] += 1;
      inferencesup[col] += d / ninferencesup[col];
    } else {
      d = ninferences - inferencesdown[col];
      ninferencesdown[col] += 1;
      inferencesdown[col] += d / ninferencesdown[col];
    }
  }

  bool isReliable(HighsInt col) const {
    return std::min(nsamplesup[col], nsamplesdown[col]) >= minreliable;
  }

  bool isReliableUp(HighsInt col) const {
    return nsamplesup[col] >= minreliable;
  }

  bool isReliableDown(HighsInt col) const {
    return nsamplesdown[col] >= minreliable;
  }

  double getAvgPseudocost() const { return cost_total; }

  double getPseudocostUp(HighsInt col, double frac, double offset) const {
    double up = std::ceil(frac) - frac;
    double cost;

    if (nsamplesup[col] == 0 || nsamplesup[col] < minreliable) {
      double weightPs = nsamplesup[col] == 0
                            ? 0
                            : 0.9 + 0.1 * nsamplesup[col] / (double)minreliable;
      cost = weightPs * pseudocostup[col];
      cost += (1.0 - weightPs) * getAvgPseudocost();
    } else
      cost = pseudocostup[col];
    return up * (offset + cost);
  }

  double getPseudocostDown(HighsInt col, double frac, double offset) const {
    double down = frac - std::floor(frac);
    double cost;

    if (nsamplesdown[col] == 0 || nsamplesdown[col] < minreliable) {
      double weightPs = nsamplesdown[col] == 0 ? 0
                                               : 0.9 + 0.1 * nsamplesdown[col] /
                                                           (double)minreliable;
      cost = weightPs * pseudocostdown[col];
      cost += (1.0 - weightPs) * getAvgPseudocost();
    } else
      cost = pseudocostdown[col];

    return down * (offset + cost);
  }

  double getPseudocostUp(HighsInt col, double frac) const {
    double up = std::ceil(frac) - frac;
    if (nsamplesup[col] == 0) return up * cost_total;
    return up * pseudocostup[col];
  }

  double getPseudocostDown(HighsInt col, double frac) const {
    double down = frac - std::floor(frac);
    if (nsamplesdown[col] == 0) return down * cost_total;
    return down * pseudocostdown[col];
  }

  double getConflictScoreUp(HighsInt col) const {
    return conflictscoreup[col] / conflict_weight;
  }

  double getConflictScoreDown(HighsInt col) const {
    return conflictscoredown[col] / conflict_weight;
  }

  double getScore(HighsInt col, double upcost, double downcost) const {
    double costScore = std::max(upcost, 1e-6) * std::max(downcost, 1e-6) /
                       std::max(1e-6, cost_total * cost_total);
    double inferenceScore = std::max(inferencesup[col], 1e-6) *
                            std::max(inferencesdown[col], 1e-6) /
                            std::max(1e-6, inferences_total * inferences_total);

    double cutOffScoreUp =
        ncutoffsup[col] /
        std::max(1.0, double(ncutoffsup[col] + nsamplesup[col]));
    double cutOffScoreDown =
        ncutoffsdown[col] /
        std::max(1.0, double(ncutoffsdown[col] + nsamplesdown[col]));
    double avgCutoffs =
        ncutoffstotal / std::max(1.0, double(ncutoffstotal + nsamplestotal));

    double cutoffScore = std::max(cutOffScoreUp, 1e-6) *
                         std::max(cutOffScoreDown, 1e-6) /
                         std::max(1e-6, avgCutoffs * avgCutoffs);

    double conflictScoreUp = conflictscoreup[col] / conflict_weight;
    double conflictScoreDown = conflictscoredown[col] / conflict_weight;
    double conflictScoreAvg =
        conflict_avg_score / (conflict_weight * conflictscoreup.size());
    double conflictScore = std::max(conflictScoreUp, 1e-6) *
                           std::max(conflictScoreDown, 1e-6) /
                           std::max(1e-6, conflictScoreAvg * conflictScoreAvg);

    auto mapScore = [](double score) { return 1.0 - 1.0 / (1.0 + score); };
    return mapScore(costScore) / degeneracyFactor +
           degeneracyFactor *
               (1e-2 * mapScore(conflictScore) +
                1e-4 * (mapScore(cutoffScore) + mapScore(inferenceScore)));
  }

  double getScore(HighsInt col, double frac) const {
    double upcost = getPseudocostUp(col, frac);
    double downcost = getPseudocostDown(col, frac);

    return getScore(col, upcost, downcost);
  }

  double getScoreUp(HighsInt col, double frac) const {
    double costScore = getPseudocostUp(col, frac) / std::max(1e-6, cost_total);
    double inferenceScore =
        inferencesup[col] / std::max(1e-6, inferences_total);

    double cutOffScoreUp =
        ncutoffsup[col] /
        std::max(1.0, double(ncutoffsup[col] + nsamplesup[col]));
    double avgCutoffs =
        ncutoffstotal / std::max(1.0, double(ncutoffstotal + nsamplestotal));

    double cutoffScore = cutOffScoreUp / std::max(1e-6, avgCutoffs);

    double conflictScoreUp = conflictscoreup[col] / conflict_weight;
    double conflictScoreAvg =
        conflict_avg_score / (conflict_weight * conflictscoreup.size());
    double conflictScore = conflictScoreUp / std::max(1e-6, conflictScoreAvg);

    auto mapScore = [](double score) { return 1.0 - 1.0 / (1.0 + score); };

    return mapScore(costScore) +
           (1e-2 * mapScore(conflictScore) +
            1e-4 * (mapScore(cutoffScore) + mapScore(inferenceScore)));
  }

  double getScoreDown(HighsInt col, double frac) const {
    double costScore =
        getPseudocostDown(col, frac) / std::max(1e-6, cost_total);
    double inferenceScore =
        inferencesdown[col] / std::max(1e-6, inferences_total);

    double cutOffScoreDown =
        ncutoffsdown[col] /
        std::max(1.0, double(ncutoffsdown[col] + nsamplesdown[col]));
    double avgCutoffs =
        ncutoffstotal / std::max(1.0, double(ncutoffstotal + nsamplestotal));

    double cutoffScore = cutOffScoreDown / std::max(1e-6, avgCutoffs);

    double conflictScoreDown = conflictscoredown[col] / conflict_weight;
    double conflictScoreAvg =
        conflict_avg_score / (conflict_weight * conflictscoredown.size());
    double conflictScore = conflictScoreDown / std::max(1e-6, conflictScoreAvg);

    auto mapScore = [](double score) { return 1.0 - 1.0 / (1.0 + score); };

    return mapScore(costScore) +
           (1e-2 * mapScore(conflictScore) +
            1e-4 * (mapScore(cutoffScore) + mapScore(inferenceScore)));
  }

  double getAvgInferencesUp(HighsInt col) const { return inferencesup[col]; }

  double getAvgInferencesDown(HighsInt col) const {
    return inferencesdown[col];
  }
};

#endif
