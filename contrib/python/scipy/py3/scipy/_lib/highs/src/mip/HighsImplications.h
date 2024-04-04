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
#ifndef HIGHS_IMPLICATIONS_H_
#define HIGHS_IMPLICATIONS_H_

#include <algorithm>
#include <cassert>
#include <utility>
#include <vector>

#include "mip/HighsDomain.h"
#include "mip/HighsDomainChange.h"

class HighsCliqueTable;
class HighsLpRelaxation;

class HighsImplications {
  HighsInt nextCleanupCall;

  struct Implics {
    std::vector<HighsDomainChange> implics;
    bool computed = false;
  };
  std::vector<Implics> implications;
  int64_t numImplications;

  bool computeImplications(HighsInt col, bool val);

 public:
  struct VarBound {
    double coef;
    double constant;

    double minValue() const { return constant + std::min(coef, 0.0); }
    double maxValue() const { return constant + std::max(coef, 0.0); }
  };

 private:
  std::vector<std::map<HighsInt, VarBound>> vubs;
  std::vector<std::map<HighsInt, VarBound>> vlbs;

 public:
  const HighsMipSolver& mipsolver;
  std::vector<HighsSubstitution> substitutions;
  std::vector<uint8_t> colsubstituted;
  HighsImplications(const HighsMipSolver& mipsolver) : mipsolver(mipsolver) {
    HighsInt numcol = mipsolver.numCol();
    implications.resize(2 * numcol);
    colsubstituted.resize(numcol);
    vubs.resize(numcol);
    vlbs.resize(numcol);
    nextCleanupCall = mipsolver.numNonzero();
    numImplications = 0;
  }

  void reset() {
    colsubstituted.clear();
    colsubstituted.shrink_to_fit();
    implications.clear();
    implications.shrink_to_fit();

    HighsInt numcol = mipsolver.numCol();
    implications.resize(2 * numcol);
    colsubstituted.resize(numcol);
    numImplications = 0;
    vubs.clear();
    vubs.shrink_to_fit();
    vubs.resize(numcol);
    vlbs.clear();
    vlbs.shrink_to_fit();
    vlbs.resize(numcol);

    nextCleanupCall = mipsolver.numNonzero();
  }

  HighsInt getNumImplications() const { return numImplications; }

  const std::vector<HighsDomainChange>& getImplications(HighsInt col, bool val,
                                                        bool& infeasible) {
    HighsInt loc = 2 * col + val;
    if (!implications[loc].computed)
      infeasible = computeImplications(col, val);
    else
      infeasible = false;

    assert(implications[loc].computed);

    return implications[loc].implics;
  }

  bool implicationsCached(HighsInt col, bool val) {
    HighsInt loc = 2 * col + val;
    return implications[loc].computed;
  }

  void addVUB(HighsInt col, HighsInt vubcol, double vubcoef,
              double vubconstant);

  void addVLB(HighsInt col, HighsInt vlbcol, double vlbcoef,
              double vlbconstant);

  const std::map<HighsInt, VarBound>& getVUBs(HighsInt col) const {
    return vubs[col];
  }

  const std::map<HighsInt, VarBound>& getVLBs(HighsInt col) const {
    return vlbs[col];
  }

  std::map<HighsInt, VarBound>& getVUBs(HighsInt col) { return vubs[col]; }

  std::map<HighsInt, VarBound>& getVLBs(HighsInt col) { return vlbs[col]; }

  bool runProbing(HighsInt col, HighsInt& numReductions);

  void rebuild(HighsInt ncols, const std::vector<HighsInt>& cIndex,
               const std::vector<HighsInt>& rIndex);

  void buildFrom(const HighsImplications& init);

  void separateImpliedBounds(const HighsLpRelaxation& lpRelaxation,
                             const std::vector<double>& sol,
                             HighsCutPool& cutpool, double feastol);

  void cleanupVarbounds(HighsInt col);
};

#endif
