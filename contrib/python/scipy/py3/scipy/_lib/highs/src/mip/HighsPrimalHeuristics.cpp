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
#include "mip/HighsPrimalHeuristics.h"

#include <numeric>
#include <unordered_set>

#include "io/HighsIO.h"
#include "lp_data/HConst.h"
#include "lp_data/HighsLpUtils.h"
#include "mip/HighsCutGeneration.h"
#include "mip/HighsDomainChange.h"
#include "mip/HighsLpRelaxation.h"
#include "mip/HighsMipSolverData.h"
#include "pdqsort/pdqsort.h"
#include "util/HighsHash.h"
#include "util/HighsIntegers.h"

HighsPrimalHeuristics::HighsPrimalHeuristics(HighsMipSolver& mipsolver)
    : mipsolver(mipsolver),
      lp_iterations(0),
      randgen(mipsolver.options_mip_->random_seed) {
  successObservations = 0;
  numSuccessObservations = 0;
  infeasObservations = 0;
  numInfeasObservations = 0;
}

void HighsPrimalHeuristics::setupIntCols() {
  intcols = mipsolver.mipdata_->integer_cols;

  pdqsort(intcols.begin(), intcols.end(), [&](HighsInt c1, HighsInt c2) {
    double lockScore1 =
        (mipsolver.mipdata_->feastol + mipsolver.mipdata_->uplocks[c1]) *
        (mipsolver.mipdata_->feastol + mipsolver.mipdata_->downlocks[c1]);

    double lockScore2 =
        (mipsolver.mipdata_->feastol + mipsolver.mipdata_->uplocks[c2]) *
        (mipsolver.mipdata_->feastol + mipsolver.mipdata_->downlocks[c2]);

    if (lockScore1 > lockScore2) return true;
    if (lockScore2 > lockScore1) return false;

    double cliqueScore1 =
        (mipsolver.mipdata_->feastol +
         mipsolver.mipdata_->cliquetable.getNumImplications(c1, 1)) *
        (mipsolver.mipdata_->feastol +
         mipsolver.mipdata_->cliquetable.getNumImplications(c1, 0));

    double cliqueScore2 =
        (mipsolver.mipdata_->feastol +
         mipsolver.mipdata_->cliquetable.getNumImplications(c2, 1)) *
        (mipsolver.mipdata_->feastol +
         mipsolver.mipdata_->cliquetable.getNumImplications(c2, 0));

    return std::make_tuple(cliqueScore1, HighsHashHelpers::hash(uint64_t(c1)),
                           c1) >
           std::make_tuple(cliqueScore2, HighsHashHelpers::hash(uint64_t(c2)),
                           c2);
  });
}

bool HighsPrimalHeuristics::solveSubMip(
    const HighsLp& lp, const HighsBasis& basis, double fixingRate,
    std::vector<double> colLower, std::vector<double> colUpper,
    HighsInt maxleaves, HighsInt maxnodes, HighsInt stallnodes) {
  HighsOptions submipoptions = *mipsolver.options_mip_;
  HighsLp submip = lp;

  // set bounds and restore integrality of the lp relaxation copy
  submip.col_lower_ = std::move(colLower);
  submip.col_upper_ = std::move(colUpper);
  submip.integrality_ = mipsolver.model_->integrality_;
  submip.offset_ = 0;

  // set limits
  submipoptions.mip_max_leaves = maxleaves;
  submipoptions.output_flag = false;
  submipoptions.mip_max_nodes = maxnodes;
  submipoptions.mip_max_stall_nodes = stallnodes;
  submipoptions.mip_pscost_minreliable = 0;
  submipoptions.time_limit -=
      mipsolver.timer_.read(mipsolver.timer_.solve_clock);
  submipoptions.objective_bound = mipsolver.mipdata_->upper_limit;

  if (!mipsolver.submip) {
    double curr_abs_gap =
        mipsolver.mipdata_->upper_limit - mipsolver.mipdata_->lower_bound;

    if (curr_abs_gap == kHighsInf) {
      curr_abs_gap = fabs(mipsolver.mipdata_->lower_bound);
      if (curr_abs_gap == kHighsInf) curr_abs_gap = 0.0;
    }

    submipoptions.mip_rel_gap = 0.0;
    submipoptions.mip_abs_gap =
        mipsolver.mipdata_->feastol * std::max(curr_abs_gap, 1000.0);
  }

  submipoptions.presolve = "on";
  submipoptions.mip_detect_symmetry = false;
  submipoptions.mip_heuristic_effort = 0.8;
  // setup solver and run it

  HighsSolution solution;
  solution.value_valid = false;
  solution.dual_valid = false;
  HighsMipSolver submipsolver(submipoptions, submip, solution, true);
  submipsolver.rootbasis = &basis;
  HighsPseudocostInitialization pscostinit(mipsolver.mipdata_->pseudocost, 1);
  submipsolver.pscostinit = &pscostinit;
  submipsolver.clqtableinit = &mipsolver.mipdata_->cliquetable;
  submipsolver.implicinit = &mipsolver.mipdata_->implications;
  submipsolver.run();
  if (submipsolver.mipdata_) {
    double numUnfixed = mipsolver.mipdata_->integral_cols.size() +
                        mipsolver.mipdata_->continuous_cols.size();
    double adjustmentfactor = submipsolver.numCol() / std::max(1.0, numUnfixed);
    // (double)mipsolver.orig_model_->a_matrix_.value_.size();
    int64_t adjusted_lp_iterations =
        (size_t)(adjustmentfactor * submipsolver.mipdata_->total_lp_iterations);
    lp_iterations += adjusted_lp_iterations;

    if (mipsolver.submip)
      mipsolver.mipdata_->num_nodes += std::max(
          int64_t{1}, int64_t(adjustmentfactor * submipsolver.node_count_));
  }

  if (submipsolver.modelstatus_ == HighsModelStatus::kInfeasible) {
    infeasObservations += fixingRate;
    ++numInfeasObservations;
  }
  if (submipsolver.node_count_ <= 1 &&
      submipsolver.modelstatus_ == HighsModelStatus::kInfeasible)
    return false;
  HighsInt oldNumImprovingSols = mipsolver.mipdata_->numImprovingSols;
  if (submipsolver.modelstatus_ != HighsModelStatus::kInfeasible &&
      !submipsolver.solution_.empty()) {
    mipsolver.mipdata_->trySolution(submipsolver.solution_, 'L');
  }

  if (mipsolver.mipdata_->numImprovingSols != oldNumImprovingSols) {
    // remember fixing rate as good
    successObservations += fixingRate;
    ++numSuccessObservations;
  }

  return true;
}

double HighsPrimalHeuristics::determineTargetFixingRate() {
  double lowFixingRate = 0.6;
  double highFixingRate = 0.6;

  if (numInfeasObservations != 0) {
    double infeasRate = infeasObservations / numInfeasObservations;
    highFixingRate = 0.9 * infeasRate;
    lowFixingRate = std::min(lowFixingRate, highFixingRate);
  }

  if (numSuccessObservations != 0) {
    double successFixingRate = successObservations / numSuccessObservations;
    lowFixingRate = std::min(lowFixingRate, 0.9 * successFixingRate);
    highFixingRate = std::max(successFixingRate * 1.1, highFixingRate);
  }

  double fixingRate = randgen.real(lowFixingRate, highFixingRate);
  // if (!mipsolver.submip) printf("fixing rate: %.2f\n", 100.0 * fixingRate);
  return fixingRate;
}

class HeuristicNeighborhood {
  HighsDomain& localdom;
  HighsInt numFixed;
  HighsHashTable<HighsInt> fixedCols;
  size_t startCheckedChanges;
  size_t nCheckedChanges;
  HighsInt numTotal;

 public:
  HeuristicNeighborhood(HighsMipSolver& mipsolver, HighsDomain& localdom)
      : localdom(localdom),
        numFixed(0),
        startCheckedChanges(localdom.getDomainChangeStack().size()),
        nCheckedChanges(startCheckedChanges) {
    for (HighsInt i : mipsolver.mipdata_->integral_cols)
      if (localdom.col_lower_[i] == localdom.col_upper_[i]) ++numFixed;

    numTotal = mipsolver.mipdata_->integral_cols.size() - numFixed;
  }

  double getFixingRate() {
    while (nCheckedChanges < localdom.getDomainChangeStack().size()) {
      HighsInt col = localdom.getDomainChangeStack()[nCheckedChanges++].column;
      if (localdom.variableType(col) == HighsVarType::kContinuous) continue;
      if (localdom.isFixed(col)) fixedCols.insert(col);
    }

    return numTotal ? fixedCols.size() / (double)numTotal : 0.0;
  }

  void backtracked() {
    nCheckedChanges = startCheckedChanges;
    if (fixedCols.size()) fixedCols.clear();
  }
};

void HighsPrimalHeuristics::rootReducedCost() {
  std::vector<std::pair<double, HighsDomainChange>> lurkingBounds =
      mipsolver.mipdata_->redcostfixing.getLurkingBounds(mipsolver);
  if (lurkingBounds.size() < 0.1 * mipsolver.mipdata_->integral_cols.size())
    return;
  pdqsort(lurkingBounds.begin(), lurkingBounds.end(),
          [](const std::pair<double, HighsDomainChange>& a,
             const std::pair<double, HighsDomainChange>& b) {
            return a.first > b.first;
          });

  auto localdom = mipsolver.mipdata_->domain;

  HeuristicNeighborhood neighborhood(mipsolver, localdom);

  double currCutoff = kHighsInf;
  double lower_bound;

  lower_bound = mipsolver.mipdata_->lower_bound + mipsolver.mipdata_->feastol;

  for (const std::pair<double, HighsDomainChange>& domchg : lurkingBounds) {
    currCutoff = domchg.first;

    if (currCutoff <= lower_bound) break;

    if (localdom.isActive(domchg.second)) continue;
    localdom.changeBound(domchg.second);

    while (true) {
      localdom.propagate();
      if (localdom.infeasible()) {
        localdom.conflictAnalysis(mipsolver.mipdata_->conflictPool);
        mipsolver.mipdata_->lower_bound =
            std::max(mipsolver.mipdata_->lower_bound, currCutoff);
        localdom.backtrack();
        if (localdom.getBranchDepth() == 0) break;
        neighborhood.backtracked();
        continue;
      }
      break;
    }
    double fixingRate = neighborhood.getFixingRate();
    if (fixingRate >= 0.5) break;
    // double gap = (currCutoff - mipsolver.mipdata_->lower_bound) /
    //             std::max(std::abs(mipsolver.mipdata_->lower_bound), 1.0);
    // if (gap < 0.001) break;
  }

  double fixingRate = neighborhood.getFixingRate();
  if (fixingRate < 0.3) return;

  solveSubMip(*mipsolver.model_, mipsolver.mipdata_->firstrootbasis, fixingRate,
              localdom.col_lower_, localdom.col_upper_,
              500,  // std::max(50, int(0.05 *
                    // (mipsolver.mipdata_->num_leaves))),
              200 + int(0.05 * (mipsolver.mipdata_->num_nodes)), 12);
}

void HighsPrimalHeuristics::RENS(const std::vector<double>& tmp) {
  HighsSearch heur(mipsolver, mipsolver.mipdata_->pseudocost);
  HighsDomain& localdom = heur.getLocalDomain();
  heur.setHeuristic(true);

  intcols.erase(std::remove_if(intcols.begin(), intcols.end(),
                               [&](HighsInt i) {
                                 return mipsolver.mipdata_->domain.isFixed(i);
                               }),
                intcols.end());

  HighsLpRelaxation heurlp(mipsolver.mipdata_->lp);
  // only use the global upper limit as LP limit so that dual proofs are valid
  heurlp.setObjectiveLimit(mipsolver.mipdata_->upper_limit);
  heurlp.setAdjustSymmetricBranchingCol(false);
  heur.setLpRelaxation(&heurlp);

  heurlp.getLpSolver().changeColsBounds(0, mipsolver.numCol() - 1,
                                        localdom.col_lower_.data(),
                                        localdom.col_upper_.data());
  localdom.clearChangedCols();
  heur.createNewNode();

  // determine the initial number of unfixed variables fixing rate to decide if
  // the problem is restricted enough to be considered for solving a submip
  double maxfixingrate = determineTargetFixingRate();
  double fixingrate = 0.0;
  bool stop = false;
  // heurlp.setIterationLimit(2 * mipsolver.mipdata_->maxrootlpiters);
  // printf("iterlimit: %" HIGHSINT_FORMAT "\n",
  //       heurlp.getLpSolver().getOptions().simplex_iteration_limit);
  HighsInt targetdepth = 1;
  HighsInt nbacktracks = -1;
  HeuristicNeighborhood neighborhood(mipsolver, localdom);
retry:
  ++nbacktracks;
  neighborhood.backtracked();
  // printf("current depth : %" HIGHSINT_FORMAT
  //        "   target depth : %" HIGHSINT_FORMAT "\n",
  //        heur.getCurrentDepth(), targetdepth);
  if (heur.getCurrentDepth() > targetdepth) {
    if (!heur.backtrackUntilDepth(targetdepth)) {
      lp_iterations += heur.getLocalLpIterations();
      return;
    }
  }

  // printf("fixingrate before loop is %g\n", fixingrate);
  assert(heur.hasNode());
  while (true) {
    // printf("evaluating node\n");
    heur.evaluateNode();
    // printf("done evaluating node\n");
    if (heur.currentNodePruned()) {
      ++nbacktracks;
      if (mipsolver.mipdata_->domain.infeasible()) {
        lp_iterations += heur.getLocalLpIterations();
        return;
      }

      if (!heur.backtrack()) break;
      neighborhood.backtracked();
      continue;
    }

    fixingrate = neighborhood.getFixingRate();
    // printf("after evaluating node current fixingrate is %g\n", fixingrate);
    if (fixingrate >= maxfixingrate) break;
    if (stop) break;
    if (nbacktracks >= 10) break;

    HighsInt numBranched = 0;
    double stopFixingRate = std::min(
        1.0 - (1.0 - neighborhood.getFixingRate()) * 0.9, maxfixingrate);
    const auto& relaxationsol = heurlp.getSolution().col_value;
    for (HighsInt i : intcols) {
      if (localdom.col_lower_[i] == localdom.col_upper_[i]) continue;

      double downval =
          std::floor(relaxationsol[i] + mipsolver.mipdata_->feastol);
      double upval = std::ceil(relaxationsol[i] - mipsolver.mipdata_->feastol);

      downval = std::min(downval, localdom.col_upper_[i]);
      upval = std::max(upval, localdom.col_lower_[i]);
      if (localdom.col_lower_[i] < downval) {
        ++numBranched;
        heur.branchUpwards(i, downval, downval - 0.5);
        localdom.propagate();
        if (localdom.infeasible()) {
          localdom.conflictAnalysis(mipsolver.mipdata_->conflictPool);
          break;
        }
      }
      if (localdom.col_upper_[i] > upval) {
        ++numBranched;
        heur.branchDownwards(i, upval, upval + 0.5);
        localdom.propagate();
        if (localdom.infeasible()) {
          localdom.conflictAnalysis(mipsolver.mipdata_->conflictPool);
          break;
        }
      }

      if (neighborhood.getFixingRate() >= stopFixingRate) break;
    }

    if (numBranched == 0) {
      auto getFixVal = [&](HighsInt col, double fracval) {
        double fixval;

        // reinforce direction of this solution away from root
        // solution if the change is at least 0.4
        // otherwise take the direction where the objective gets worse
        // if objective is zero round to nearest integer
        double rootchange = mipsolver.mipdata_->rootlpsol.empty()
                                ? 0.0
                                : fracval - mipsolver.mipdata_->rootlpsol[col];
        if (rootchange >= 0.4)
          fixval = std::ceil(fracval);
        else if (rootchange <= -0.4)
          fixval = std::floor(fracval);
        if (mipsolver.model_->col_cost_[col] > 0.0)
          fixval = std::ceil(fracval);
        else if (mipsolver.model_->col_cost_[col] < 0.0)
          fixval = std::floor(fracval);
        else
          fixval = std::floor(fracval + 0.5);
        // make sure we do not set an infeasible domain
        fixval = std::min(localdom.col_upper_[col], fixval);
        fixval = std::max(localdom.col_lower_[col], fixval);
        return fixval;
      };

      pdqsort(heurlp.getFractionalIntegers().begin(),
              heurlp.getFractionalIntegers().end(),
              [&](const std::pair<HighsInt, double>& a,
                  const std::pair<HighsInt, double>& b) {
                return std::make_pair(
                           std::abs(getFixVal(a.first, a.second) - a.second),
                           HighsHashHelpers::hash(
                               (uint64_t(a.first) << 32) +
                               heurlp.getFractionalIntegers().size())) <
                       std::make_pair(
                           std::abs(getFixVal(b.first, b.second) - b.second),
                           HighsHashHelpers::hash(
                               (uint64_t(b.first) << 32) +
                               heurlp.getFractionalIntegers().size()));
              });

      double change = 0.0;
      // select a set of fractional variables to fix
      for (auto fracint : heurlp.getFractionalIntegers()) {
        double fixval = getFixVal(fracint.first, fracint.second);

        if (localdom.col_lower_[fracint.first] < fixval) {
          ++numBranched;
          heur.branchUpwards(fracint.first, fixval, fracint.second);
          localdom.propagate();
          if (localdom.infeasible()) {
            localdom.conflictAnalysis(mipsolver.mipdata_->conflictPool);
            break;
          }

          fixingrate = neighborhood.getFixingRate();
        }

        if (localdom.col_upper_[fracint.first] > fixval) {
          ++numBranched;
          heur.branchDownwards(fracint.first, fixval, fracint.second);
          localdom.propagate();
          if (localdom.infeasible()) {
            localdom.conflictAnalysis(mipsolver.mipdata_->conflictPool);
            break;
          }

          fixingrate = neighborhood.getFixingRate();
        }

        if (fixingrate >= maxfixingrate) break;

        change += std::abs(fixval - fracint.second);
        if (change >= 0.5) break;
      }
    }

    if (numBranched == 0) break;
    heurlp.flushDomain(localdom);
  }

  // printf("stopped heur dive with fixing rate %g\n", fixingrate);
  // if there is no node left it means we backtracked to the global domain and
  // the subproblem was solved with the dive
  if (!heur.hasNode()) {
    lp_iterations += heur.getLocalLpIterations();
    return;
  }
  // determine the fixing rate to decide if the problem is restricted enough to
  // be considered for solving a submip

  fixingrate = neighborhood.getFixingRate();
  // printf("fixing rate is %g\n", fixingrate);
  if (fixingrate < 0.1 ||
      (mipsolver.submip && mipsolver.mipdata_->numImprovingSols != 0)) {
    // heur.childselrule = ChildSelectionRule::kBestCost;
    heur.setMinReliable(0);
    heur.solveDepthFirst(10);
    lp_iterations += heur.getLocalLpIterations();
    if (mipsolver.submip) mipsolver.mipdata_->num_nodes += heur.getLocalNodes();
    // lpiterations += heur.lpiterations;
    // pseudocost = heur.pseudocost;
    return;
  }

  heurlp.removeObsoleteRows(false);
  if (!solveSubMip(heurlp.getLp(), heurlp.getLpSolver().getBasis(), fixingrate,
                   localdom.col_lower_, localdom.col_upper_,
                   500,  // std::max(50, int(0.05 *
                         // (mipsolver.mipdata_->num_leaves))),
                   200 + int(0.05 * (mipsolver.mipdata_->num_nodes)), 12)) {
    int64_t new_lp_iterations = lp_iterations + heur.getLocalLpIterations();
    if (new_lp_iterations + mipsolver.mipdata_->heuristic_lp_iterations >
        100000 + ((mipsolver.mipdata_->total_lp_iterations -
                   mipsolver.mipdata_->heuristic_lp_iterations -
                   mipsolver.mipdata_->sb_lp_iterations) >>
                  1)) {
      lp_iterations = new_lp_iterations;
      return;
    }

    targetdepth = heur.getCurrentDepth() / 2;
    if (targetdepth <= 1 || mipsolver.mipdata_->checkLimits()) {
      lp_iterations = new_lp_iterations;
      return;
    }
    maxfixingrate = fixingrate * 0.5;
    // printf("infeasible in in root node, trying with lower fixing rate %g\n",
    //        maxfixingrate);
    goto retry;
  }

  lp_iterations += heur.getLocalLpIterations();
}

void HighsPrimalHeuristics::RINS(const std::vector<double>& relaxationsol) {
  if (int(relaxationsol.size()) != mipsolver.numCol()) return;

  intcols.erase(std::remove_if(intcols.begin(), intcols.end(),
                               [&](HighsInt i) {
                                 return mipsolver.mipdata_->domain.isFixed(i);
                               }),
                intcols.end());

  HighsSearch heur(mipsolver, mipsolver.mipdata_->pseudocost);
  HighsDomain& localdom = heur.getLocalDomain();
  heur.setHeuristic(true);

  HighsLpRelaxation heurlp(mipsolver.mipdata_->lp);
  // only use the global upper limit as LP limit so that dual proofs are valid
  heurlp.setObjectiveLimit(mipsolver.mipdata_->upper_limit);
  heurlp.setAdjustSymmetricBranchingCol(false);
  heur.setLpRelaxation(&heurlp);

  heurlp.getLpSolver().changeColsBounds(0, mipsolver.numCol() - 1,
                                        localdom.col_lower_.data(),
                                        localdom.col_upper_.data());
  localdom.clearChangedCols();
  heur.createNewNode();

  // determine the initial number of unfixed variables fixing rate to decide if
  // the problem is restricted enough to be considered for solving a submip
  double maxfixingrate = determineTargetFixingRate();
  double minfixingrate = 0.25;
  double fixingrate = 0.0;
  bool stop = false;
  HighsInt nbacktracks = -1;
  HighsInt targetdepth = 1;
  HeuristicNeighborhood neighborhood(mipsolver, localdom);
retry:
  ++nbacktracks;
  neighborhood.backtracked();
  // printf("current depth : %" HIGHSINT_FORMAT "   target depth : %"
  // HIGHSINT_FORMAT "\n", heur.getCurrentDepth(),
  //       targetdepth);
  if (heur.getCurrentDepth() > targetdepth) {
    if (!heur.backtrackUntilDepth(targetdepth)) {
      lp_iterations += heur.getLocalLpIterations();
      return;
    }
  }

  assert(heur.hasNode());

  while (true) {
    heur.evaluateNode();
    if (heur.currentNodePruned()) {
      ++nbacktracks;
      // printf("backtrack1\n");
      if (mipsolver.mipdata_->domain.infeasible()) {
        lp_iterations += heur.getLocalLpIterations();
        return;
      }

      if (!heur.backtrack()) break;
      neighborhood.backtracked();
      continue;
    }

    fixingrate = neighborhood.getFixingRate();

    if (stop) break;
    if (fixingrate >= maxfixingrate) break;
    if (nbacktracks >= 10) break;

    std::vector<std::pair<HighsInt, double>>::iterator fixcandend;

    // partition the fractional variables to consider which ones should we fix
    // in this dive first if there is an incumbent, we dive towards the RINS
    // neighborhood
    fixcandend = std::partition(
        heurlp.getFractionalIntegers().begin(),
        heurlp.getFractionalIntegers().end(),
        [&](const std::pair<HighsInt, double>& fracvar) {
          return std::abs(relaxationsol[fracvar.first] -
                          mipsolver.mipdata_->incumbent[fracvar.first]) <=
                 mipsolver.mipdata_->feastol;
        });

    bool fixtolpsol = true;

    auto getFixVal = [&](HighsInt col, double fracval) {
      double fixval;
      if (fixtolpsol) {
        // RINS neighborhood (with extension)
        fixval = std::floor(relaxationsol[col] + 0.5);
      } else {
        // reinforce direction of this solution away from root
        // solution if the change is at least 0.4
        // otherwise take the direction where the objective gets worse
        // if objcetive is zero round to nearest integer
        double rootchange = fracval - mipsolver.mipdata_->rootlpsol[col];
        if (rootchange >= 0.4)
          fixval = std::ceil(fracval);
        else if (rootchange <= -0.4)
          fixval = std::floor(fracval);
        if (mipsolver.model_->col_cost_[col] > 0.0)
          fixval = std::ceil(fracval);
        else if (mipsolver.model_->col_cost_[col] < 0.0)
          fixval = std::floor(fracval);
        else
          fixval = std::floor(fracval + 0.5);
      }
      // make sure we do not set an infeasible domain
      fixval = std::min(localdom.col_upper_[col], fixval);
      fixval = std::max(localdom.col_lower_[col], fixval);
      return fixval;
    };

    // no candidates left to fix for getting to the neighborhood, therefore we
    // switch to a different diving strategy until the minimal fixing rate is
    // reached
    HighsInt numBranched = 0;
    if (heurlp.getFractionalIntegers().begin() == fixcandend) {
      fixingrate = neighborhood.getFixingRate();
      double stopFixingRate =
          std::min(maxfixingrate, 1.0 - (1.0 - fixingrate) * 0.9);
      const auto& currlpsol = heurlp.getSolution().col_value;
      for (HighsInt i : intcols) {
        if (localdom.col_lower_[i] == localdom.col_upper_[i]) continue;

        if (std::abs(currlpsol[i] - mipsolver.mipdata_->incumbent[i]) <=
            mipsolver.mipdata_->feastol) {
          double fixval = HighsIntegers::nearestInteger(currlpsol[i]);
          HighsInt oldNumBranched = numBranched;
          if (localdom.col_lower_[i] < fixval) {
            ++numBranched;
            heur.branchUpwards(i, fixval, fixval - 0.5);
            localdom.propagate();
            if (localdom.infeasible()) {
              localdom.conflictAnalysis(mipsolver.mipdata_->conflictPool);
              break;
            }

            fixingrate = neighborhood.getFixingRate();
          }
          if (localdom.col_upper_[i] > fixval) {
            ++numBranched;
            heur.branchDownwards(i, fixval, fixval + 0.5);
            localdom.propagate();
            if (localdom.infeasible()) {
              localdom.conflictAnalysis(mipsolver.mipdata_->conflictPool);
              break;
            }

            fixingrate = neighborhood.getFixingRate();
          }

          if (fixingrate >= stopFixingRate) break;
        }
      }

      if (numBranched != 0) {
        // printf(
        //    "fixed %" HIGHSINT_FORMAT " additional cols, old fixing rate:
        //    %.2f%%, new fixing " "rate: %.2f%%\n", numBranched, fixingrate,
        //    getFixingRate());
        heurlp.flushDomain(localdom);
        continue;
      }

      if (fixingrate >= minfixingrate)
        break;  // if the RINS neigborhood achieved a high enough fixing rate
                // by itself we stop here
      fixcandend = heurlp.getFractionalIntegers().end();
      // now sort the variables by their distance towards the value they will
      // be fixed to
      fixtolpsol = false;
    }

    // now sort the variables by their distance towards the value they will be
    // fixed to
    pdqsort(heurlp.getFractionalIntegers().begin(), fixcandend,
            [&](const std::pair<HighsInt, double>& a,
                const std::pair<HighsInt, double>& b) {
              return std::make_pair(
                         std::abs(getFixVal(a.first, a.second) - a.second),
                         HighsHashHelpers::hash(
                             (uint64_t(a.first) << 32) +
                             heurlp.getFractionalIntegers().size())) <
                     std::make_pair(
                         std::abs(getFixVal(b.first, b.second) - b.second),
                         HighsHashHelpers::hash(
                             (uint64_t(b.first) << 32) +
                             heurlp.getFractionalIntegers().size()));
            });

    double change = 0.0;
    // select a set of fractional variables to fix
    for (auto fracint = heurlp.getFractionalIntegers().begin();
         fracint != fixcandend; ++fracint) {
      double fixval = getFixVal(fracint->first, fracint->second);

      if (localdom.col_lower_[fracint->first] < fixval) {
        ++numBranched;
        heur.branchUpwards(fracint->first, fixval, fracint->second);
        if (localdom.infeasible()) {
          localdom.conflictAnalysis(mipsolver.mipdata_->conflictPool);
          break;
        }

        fixingrate = neighborhood.getFixingRate();
      }

      if (localdom.col_upper_[fracint->first] > fixval) {
        ++numBranched;
        heur.branchDownwards(fracint->first, fixval, fracint->second);
        if (localdom.infeasible()) {
          localdom.conflictAnalysis(mipsolver.mipdata_->conflictPool);
          break;
        }

        fixingrate = neighborhood.getFixingRate();
      }

      if (fixingrate >= maxfixingrate) break;

      change += std::abs(fixval - fracint->second);
      if (change >= 0.5) break;
    }

    if (numBranched == 0) break;

    heurlp.flushDomain(localdom);

    // printf("%" HIGHSINT_FORMAT "/%" HIGHSINT_FORMAT " fixed, fixingrate is
    // %g\n", nfixed, ntotal, fixingrate);
  }

  // if there is no node left it means we backtracked to the global domain and
  // the subproblem was solved with the dive
  if (!heur.hasNode()) {
    lp_iterations += heur.getLocalLpIterations();
    return;
  }
  // determine the fixing rate to decide if the problem is restricted enough
  // to be considered for solving a submip

  // printf("fixing rate is %g\n", fixingrate);
  fixingrate = neighborhood.getFixingRate();
  if (fixingrate < 0.1 ||
      (mipsolver.submip && mipsolver.mipdata_->numImprovingSols != 0)) {
    // heur.childselrule = ChildSelectionRule::kBestCost;
    heur.setMinReliable(0);
    heur.solveDepthFirst(10);
    lp_iterations += heur.getLocalLpIterations();
    if (mipsolver.submip) mipsolver.mipdata_->num_nodes += heur.getLocalNodes();
    // lpiterations += heur.lpiterations;
    // pseudocost = heur.pseudocost;
    return;
  }

  heurlp.removeObsoleteRows(false);
  if (!solveSubMip(heurlp.getLp(), heurlp.getLpSolver().getBasis(), fixingrate,
                   localdom.col_lower_, localdom.col_upper_,
                   500,  // std::max(50, int(0.05 *
                         // (mipsolver.mipdata_->num_leaves))),
                   200 + int(0.05 * (mipsolver.mipdata_->num_nodes)), 12)) {
    int64_t new_lp_iterations = lp_iterations + heur.getLocalLpIterations();
    if (new_lp_iterations + mipsolver.mipdata_->heuristic_lp_iterations >
        100000 + ((mipsolver.mipdata_->total_lp_iterations -
                   mipsolver.mipdata_->heuristic_lp_iterations -
                   mipsolver.mipdata_->sb_lp_iterations) >>
                  1)) {
      lp_iterations = new_lp_iterations;
      return;
    }

    targetdepth = heur.getCurrentDepth() / 2;
    if (targetdepth <= 1 || mipsolver.mipdata_->checkLimits()) {
      lp_iterations = new_lp_iterations;
      return;
    }
    // printf("infeasible in in root node, trying with lower fixing rate\n");
    maxfixingrate = fixingrate * 0.5;
    goto retry;
  }

  lp_iterations += heur.getLocalLpIterations();
}

bool HighsPrimalHeuristics::tryRoundedPoint(const std::vector<double>& point,
                                            char source) {
  auto localdom = mipsolver.mipdata_->domain;

  HighsInt numintcols = intcols.size();
  for (HighsInt i = 0; i != numintcols; ++i) {
    HighsInt col = intcols[i];
    double intval = point[col];
    intval = std::min(localdom.col_upper_[col], intval);
    intval = std::max(localdom.col_lower_[col], intval);

    localdom.fixCol(col, intval, HighsDomain::Reason::branching());
    if (localdom.infeasible()) {
      localdom.conflictAnalysis(mipsolver.mipdata_->conflictPool);
      return false;
    }
    localdom.propagate();
    if (localdom.infeasible()) {
      localdom.conflictAnalysis(mipsolver.mipdata_->conflictPool);
      return false;
    }
  }

  if (numintcols != mipsolver.numCol()) {
    HighsLpRelaxation lprelax(mipsolver);
    lprelax.loadModel();
    lprelax.setIterationLimit(
        std::max(int64_t{10000}, 2 * mipsolver.mipdata_->firstrootlpiters));
    lprelax.getLpSolver().changeColsBounds(0, mipsolver.numCol() - 1,
                                           localdom.col_lower_.data(),
                                           localdom.col_upper_.data());

    if (numintcols / (double)mipsolver.numCol() >= 0.2)
      lprelax.getLpSolver().setOptionValue("presolve", "on");
    else
      lprelax.getLpSolver().setBasis(mipsolver.mipdata_->firstrootbasis,
                                     "HighsPrimalHeuristics::tryRoundedPoint");

    HighsLpRelaxation::Status st = lprelax.resolveLp();

    if (st == HighsLpRelaxation::Status::kInfeasible) {
      std::vector<HighsInt> inds;
      std::vector<double> vals;
      double rhs;
      if (lprelax.computeDualInfProof(mipsolver.mipdata_->domain, inds, vals,
                                      rhs)) {
        HighsCutGeneration cutGen(lprelax, mipsolver.mipdata_->cutpool);
        cutGen.generateConflict(localdom, inds, vals, rhs);
      }
      return false;
    } else if (lprelax.unscaledPrimalFeasible(st)) {
      mipsolver.mipdata_->addIncumbent(
          lprelax.getLpSolver().getSolution().col_value, lprelax.getObjective(),
          source);
      return true;
    }
  }

  return mipsolver.mipdata_->trySolution(localdom.col_lower_, source);
}

bool HighsPrimalHeuristics::linesearchRounding(
    const std::vector<double>& point1, const std::vector<double>& point2,
    char source) {
  std::vector<double> roundedpoint;

  HighsInt numintcols = intcols.size();
  roundedpoint.resize(mipsolver.numCol());

  double alpha = 0.0;
  assert(int(mipsolver.mipdata_->uplocks.size()) == mipsolver.numCol());
  assert(int(point1.size()) == mipsolver.numCol());
  assert(int(point2.size()) == mipsolver.numCol());

  while (alpha < 1.0) {
    double nextalpha = 1.0;
    bool reachedpoint2 = true;
    // printf("trying alpha = %g\n", alpha);
    for (HighsInt i = 0; i != numintcols; ++i) {
      HighsInt col = intcols[i];
      assert(col >= 0);
      assert(col < mipsolver.numCol());
      if (mipsolver.mipdata_->uplocks[col] == 0) {
        roundedpoint[col] = std::ceil(std::max(point1[col], point2[col]) -
                                      mipsolver.mipdata_->feastol);
        continue;
      }

      if (mipsolver.mipdata_->downlocks[col] == 0) {
        roundedpoint[col] = std::floor(std::min(point1[col], point2[col]) +
                                       mipsolver.mipdata_->feastol);
        continue;
      }

      double convexcomb = (1.0 - alpha) * point1[col] + alpha * point2[col];
      double intpoint2 = std::floor(point2[col] + 0.5);
      roundedpoint[col] = std::floor(convexcomb + 0.5);

      if (roundedpoint[col] == intpoint2) continue;

      reachedpoint2 = false;
      double tmpalpha = (roundedpoint[col] + 0.5 + mipsolver.mipdata_->feastol -
                         point1[col]) /
                        std::abs(point2[col] - point1[col]);
      if (tmpalpha < nextalpha && tmpalpha > alpha + 1e-2) nextalpha = tmpalpha;
    }

    if (tryRoundedPoint(roundedpoint, source)) return true;

    if (reachedpoint2) return false;

    alpha = nextalpha;
  }

  return false;
}

void HighsPrimalHeuristics::randomizedRounding(
    const std::vector<double>& relaxationsol) {
  if (int(relaxationsol.size()) != mipsolver.numCol()) return;

  auto localdom = mipsolver.mipdata_->domain;

  for (HighsInt i : intcols) {
    double intval;
    if (mipsolver.mipdata_->uplocks[i] == 0)
      intval = std::ceil(relaxationsol[i] - mipsolver.mipdata_->feastol);
    else if (mipsolver.mipdata_->downlocks[i] == 0)
      intval = std::floor(relaxationsol[i] + mipsolver.mipdata_->feastol);
    else
      intval = std::floor(relaxationsol[i] + randgen.real(0.1, 0.9));

    intval = std::min(localdom.col_upper_[i], intval);
    intval = std::max(localdom.col_lower_[i], intval);

    localdom.fixCol(i, intval, HighsDomain::Reason::branching());
    if (localdom.infeasible()) {
      localdom.conflictAnalysis(mipsolver.mipdata_->conflictPool);
      return;
    }
    localdom.propagate();
    if (localdom.infeasible()) {
      localdom.conflictAnalysis(mipsolver.mipdata_->conflictPool);
      return;
    }
  }

  if (int(mipsolver.mipdata_->integer_cols.size()) != mipsolver.numCol()) {
    HighsLpRelaxation lprelax(mipsolver);
    lprelax.loadModel();
    lprelax.setIterationLimit(
        std::max(int64_t{10000}, 2 * mipsolver.mipdata_->firstrootlpiters));
    lprelax.getLpSolver().changeColsBounds(0, mipsolver.numCol() - 1,
                                           localdom.col_lower_.data(),
                                           localdom.col_upper_.data());
    if (intcols.size() / (double)mipsolver.numCol() >= 0.2)
      lprelax.getLpSolver().setOptionValue("presolve", "on");
    else
      lprelax.getLpSolver().setBasis(
          mipsolver.mipdata_->firstrootbasis,
          "HighsPrimalHeuristics::randomizedRounding");
    HighsLpRelaxation::Status st = lprelax.resolveLp();

    if (st == HighsLpRelaxation::Status::kInfeasible) {
      std::vector<HighsInt> inds;
      std::vector<double> vals;
      double rhs;
      if (lprelax.computeDualInfProof(mipsolver.mipdata_->domain, inds, vals,
                                      rhs)) {
        HighsCutGeneration cutGen(lprelax, mipsolver.mipdata_->cutpool);
        cutGen.generateConflict(localdom, inds, vals, rhs);
      }

    } else if (lprelax.unscaledPrimalFeasible(st))
      mipsolver.mipdata_->addIncumbent(
          lprelax.getLpSolver().getSolution().col_value, lprelax.getObjective(),
          'R');
  } else {
    mipsolver.mipdata_->trySolution(localdom.col_lower_, 'R');
  }
}

void HighsPrimalHeuristics::feasibilityPump() {
  HighsLpRelaxation lprelax(mipsolver.mipdata_->lp);
  std::unordered_set<std::vector<HighsInt>, HighsVectorHasher, HighsVectorEqual>
      referencepoints;
  std::vector<double> roundedsol;
  HighsLpRelaxation::Status status = lprelax.resolveLp();
  lp_iterations += lprelax.getNumLpIterations();

  std::vector<double> fracintcost;
  std::vector<HighsInt> fracintset;

  std::vector<HighsInt> mask(mipsolver.model_->num_col_, 1);
  std::vector<double> cost(mipsolver.model_->num_col_, 0.0);

  lprelax.getLpSolver().setOptionValue("simplex_strategy",
                                       kSimplexStrategyPrimal);
  lprelax.setObjectiveLimit();
  lprelax.getLpSolver().setOptionValue(
      "primal_simplex_bound_perturbation_multiplier", 0.0);

  lprelax.setIterationLimit(5 * mipsolver.mipdata_->avgrootlpiters);

  while (!lprelax.getFractionalIntegers().empty()) {
    const auto& lpsol = lprelax.getLpSolver().getSolution().col_value;
    roundedsol = lprelax.getLpSolver().getSolution().col_value;

    std::vector<HighsInt> referencepoint;
    referencepoint.reserve(mipsolver.mipdata_->integer_cols.size());

    auto localdom = mipsolver.mipdata_->domain;
    for (HighsInt i : mipsolver.mipdata_->integer_cols) {
      assert(mipsolver.variableType(i) == HighsVarType::kInteger);
      double intval = std::floor(roundedsol[i] + randgen.real(0.4, 0.6));
      intval = std::max(intval, localdom.col_lower_[i]);
      intval = std::min(intval, localdom.col_upper_[i]);
      roundedsol[i] = intval;
      referencepoint.push_back((HighsInt)intval);
      if (!localdom.infeasible()) {
        localdom.fixCol(i, intval, HighsDomain::Reason::branching());
        if (localdom.infeasible()) {
          localdom.conflictAnalysis(mipsolver.mipdata_->conflictPool);
          continue;
        }
        localdom.propagate();
        if (localdom.infeasible()) {
          localdom.conflictAnalysis(mipsolver.mipdata_->conflictPool);
          continue;
        }
      }
    }

    bool havecycle = !referencepoints.emplace(referencepoint).second;
    for (HighsInt k = 0; havecycle && k < 2; ++k) {
      for (HighsInt i = 0; i != 10; ++i) {
        HighsInt flippos =
            randgen.integer(mipsolver.mipdata_->integer_cols.size());
        HighsInt col = mipsolver.mipdata_->integer_cols[flippos];
        if (roundedsol[col] > lpsol[col])
          roundedsol[col] = (HighsInt)std::floor(lpsol[col]);
        else if (roundedsol[col] < lpsol[col])
          roundedsol[col] = (HighsInt)std::ceil(lpsol[col]);
        else if (roundedsol[col] < mipsolver.mipdata_->domain.col_upper_[col])
          roundedsol[col] = mipsolver.mipdata_->domain.col_upper_[col];
        else
          roundedsol[col] = mipsolver.mipdata_->domain.col_lower_[col];

        referencepoint[flippos] = (HighsInt)roundedsol[col];
      }
      havecycle = !referencepoints.emplace(referencepoint).second;
    }

    if (havecycle) return;

    if (linesearchRounding(lpsol, roundedsol, 'F')) return;

    if (lprelax.getNumLpIterations() >=
        1000 + mipsolver.mipdata_->avgrootlpiters * 5)
      break;

    for (HighsInt i : mipsolver.mipdata_->integer_cols) {
      assert(mipsolver.variableType(i) == HighsVarType::kInteger);

      if (mipsolver.mipdata_->uplocks[i] == 0 ||
          mipsolver.mipdata_->downlocks[i] == 0)
        cost[i] = 0.0;
      else if (lpsol[i] > roundedsol[i] - mipsolver.mipdata_->feastol)
        cost[i] = -1.0 + randgen.real(-1e-4, 1e-4);
      else
        cost[i] = 1.0 + randgen.real(-1e-4, 1e-4);
    }

    lprelax.getLpSolver().changeColsCost(mask.data(), cost.data());
    int64_t niters = -lprelax.getNumLpIterations();
    status = lprelax.resolveLp();
    niters += lprelax.getNumLpIterations();
    if (niters == 0) break;
    lp_iterations += niters;
  }

  if (lprelax.getFractionalIntegers().empty() &&
      lprelax.unscaledPrimalFeasible(status))
    mipsolver.mipdata_->addIncumbent(
        lprelax.getLpSolver().getSolution().col_value, lprelax.getObjective(),
        'F');
}

void HighsPrimalHeuristics::centralRounding() {
  if (HighsInt(mipsolver.mipdata_->analyticCenter.size()) != mipsolver.numCol())
    return;

  if (!mipsolver.mipdata_->firstlpsol.empty())
    linesearchRounding(mipsolver.mipdata_->firstlpsol,
                       mipsolver.mipdata_->analyticCenter, 'C');
  else if (!mipsolver.mipdata_->rootlpsol.empty())
    linesearchRounding(mipsolver.mipdata_->rootlpsol,
                       mipsolver.mipdata_->analyticCenter, 'C');
  else
    linesearchRounding(mipsolver.mipdata_->analyticCenter,
                       mipsolver.mipdata_->analyticCenter, 'C');
}

#if 0
void HighsPrimalHeuristics::clique() {
  HighsHashTable<HighsInt, double> entries;
  double offset = 0.0;

  HighsDomain& globaldom = mipsolver.mipdata_->domain;
  for (HighsInt j = 0; j != mipsolver.numCol(); ++j) {
    HighsInt col = j;
    double val = mipsolver.colCost(col);
    if (val == 0.0) continue;

    if (!globaldom.isBinary(col)) {
      offset += val * globaldom.col_lower_[col];
      continue;
    }

    mipsolver.mipdata_->cliquetable.resolveSubstitution(col, val, offset);
    entries[col] += val;
  }

  std::vector<double> profits;
  std::vector<HighsCliqueTable::CliqueVar> objvars;

  for (const auto& entry : entries) {
    double objprofit = -entry.value();
    if (objprofit < 0) {
      offset += objprofit;
      profits.push_back(-objprofit);
      objvars.emplace_back(entry.key(), 0);
    } else {
      profits.push_back(objprofit);
      objvars.emplace_back(entry.key(), 1);
    }
  }

  std::vector<double> solution(mipsolver.numCol());

  HighsInt nobjvars = profits.size();
  for (HighsInt i = 0; i != nobjvars; ++i) solution[objvars[i].col] = objvars[i].val;

  std::vector<std::vector<HighsCliqueTable::CliqueVar>> cliques;
  double bestviol;
  HighsInt bestviolpos;
  HighsInt numcliques;

  cliques = mipsolver.mipdata_->cliquetable.separateCliques(
      solution, mipsolver.mipdata_->domain, mipsolver.mipdata_->feastol);
  numcliques = cliques.size();
  while (numcliques != 0) {
    bestviol = 0.5;
    bestviolpos = -1;

    for (HighsInt c = 0; c != numcliques; ++c) {
      double viol = -1.0;
      for (HighsCliqueTable::CliqueVar clqvar : cliques[c])
        viol += clqvar.weight(solution);

      if (viol > bestviolpos) {
        bestviolpos = c;
        bestviol = viol;
      }
    }

    cliques = mipsolver.mipdata_->cliquetable.separateCliques(
        solution, mipsolver.mipdata_->domain, mipsolver.mipdata_->feastol);
    numcliques = cliques.size();
  }
}
#endif

void HighsPrimalHeuristics::flushStatistics() {
  mipsolver.mipdata_->heuristic_lp_iterations += lp_iterations;
  mipsolver.mipdata_->total_lp_iterations += lp_iterations;
  lp_iterations = 0;
}
