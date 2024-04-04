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
#include "mip/HighsSearch.h"

#include <numeric>

#include "lp_data/HConst.h"
#include "mip/HighsCutGeneration.h"
#include "mip/HighsDomainChange.h"
#include "mip/HighsMipSolverData.h"

HighsSearch::HighsSearch(HighsMipSolver& mipsolver,
                         const HighsPseudocost& pseudocost)
    : mipsolver(mipsolver),
      lp(nullptr),
      localdom(mipsolver.mipdata_->domain),
      pseudocost(pseudocost) {
  nnodes = 0;
  treeweight = 0.0;
  depthoffset = 0;
  lpiterations = 0;
  heurlpiterations = 0;
  sblpiterations = 0;
  upper_limit = kHighsInf;
  inheuristic = false;
  inbranching = false;
  countTreeWeight = true;
  childselrule = mipsolver.submip ? ChildSelectionRule::kHybridInferenceCost
                                  : ChildSelectionRule::kRootSol;
  this->localdom.setDomainChangeStack(std::vector<HighsDomainChange>());
}

double HighsSearch::checkSol(const std::vector<double>& sol,
                             bool& integerfeasible) const {
  HighsCDouble objval = 0.0;
  integerfeasible = true;
  for (HighsInt i = 0; i != mipsolver.numCol(); ++i) {
    objval += sol[i] * mipsolver.colCost(i);
    assert(std::isfinite(sol[i]));

    if (!integerfeasible || mipsolver.variableType(i) != HighsVarType::kInteger)
      continue;

    double intval = std::floor(sol[i] + 0.5);
    if (std::abs(sol[i] - intval) > mipsolver.mipdata_->feastol) {
      integerfeasible = false;
    }
  }

  return double(objval);
}

bool HighsSearch::orbitsValidInChildNode(
    const HighsDomainChange& branchChg) const {
  HighsInt branchCol = branchChg.column;
  // if the variable is integral or we are in an up branch the stabilizer only
  // stays valid if the column has been stabilized
  const NodeData& currNode = nodestack.back();
  if (!currNode.stabilizerOrbits ||
      currNode.stabilizerOrbits->orbitCols.empty() ||
      currNode.stabilizerOrbits->isStabilized(branchCol))
    return true;

  // a down branch stays valid if the variable is binary
  if (branchChg.boundtype == HighsBoundType::kUpper &&
      localdom.isGlobalBinary(branchChg.column))
    return true;

  return false;
}

double HighsSearch::getCutoffBound() const {
  return std::min(mipsolver.mipdata_->upper_limit, upper_limit);
}

void HighsSearch::setRINSNeighbourhood(const std::vector<double>& basesol,
                                       const std::vector<double>& relaxsol) {
  for (HighsInt i = 0; i != mipsolver.numCol(); ++i) {
    if (mipsolver.variableType(i) != HighsVarType::kInteger) continue;
    if (localdom.col_lower_[i] == localdom.col_upper_[i]) continue;

    double intval = std::floor(basesol[i] + 0.5);
    if (std::abs(relaxsol[i] - intval) < mipsolver.mipdata_->feastol) {
      if (localdom.col_lower_[i] < intval)
        localdom.changeBound(HighsBoundType::kLower, i,
                             std::min(intval, localdom.col_upper_[i]),
                             HighsDomain::Reason::unspecified());
      if (localdom.col_upper_[i] > intval)
        localdom.changeBound(HighsBoundType::kUpper, i,
                             std::max(intval, localdom.col_lower_[i]),
                             HighsDomain::Reason::unspecified());
    }
  }
}

void HighsSearch::setRENSNeighbourhood(const std::vector<double>& lpsol) {
  for (HighsInt i = 0; i != mipsolver.numCol(); ++i) {
    if (mipsolver.variableType(i) != HighsVarType::kInteger) continue;
    if (localdom.col_lower_[i] == localdom.col_upper_[i]) continue;

    double downval = std::floor(lpsol[i] + mipsolver.mipdata_->feastol);
    double upval = std::ceil(lpsol[i] - mipsolver.mipdata_->feastol);

    if (localdom.col_lower_[i] < downval) {
      localdom.changeBound(HighsBoundType::kLower, i,
                           std::min(downval, localdom.col_upper_[i]),
                           HighsDomain::Reason::unspecified());
      if (localdom.infeasible()) return;
    }
    if (localdom.col_upper_[i] > upval) {
      localdom.changeBound(HighsBoundType::kUpper, i,
                           std::max(upval, localdom.col_lower_[i]),
                           HighsDomain::Reason::unspecified());
      if (localdom.infeasible()) return;
    }
  }
}

void HighsSearch::createNewNode() {
  nodestack.emplace_back();
  nodestack.back().domgchgStackPos = localdom.getDomainChangeStack().size();
}

void HighsSearch::cutoffNode() { nodestack.back().opensubtrees = 0; }

void HighsSearch::setMinReliable(HighsInt minreliable) {
  pseudocost.setMinReliable(minreliable);
}

void HighsSearch::branchDownwards(HighsInt col, double newub,
                                  double branchpoint) {
  NodeData& currnode = nodestack.back();

  assert(currnode.opensubtrees == 2);
  assert(mipsolver.variableType(col) != HighsVarType::kContinuous);

  currnode.opensubtrees = 1;
  currnode.branching_point = branchpoint;
  currnode.branchingdecision.column = col;
  currnode.branchingdecision.boundval = newub;
  currnode.branchingdecision.boundtype = HighsBoundType::kUpper;

  HighsInt domchgPos = localdom.getDomainChangeStack().size();
  bool passStabilizerToChildNode =
      orbitsValidInChildNode(currnode.branchingdecision);
  localdom.changeBound(currnode.branchingdecision);
  nodestack.emplace_back(
      currnode.lower_bound, currnode.estimate, currnode.nodeBasis,
      passStabilizerToChildNode ? currnode.stabilizerOrbits : nullptr);
  nodestack.back().domgchgStackPos = domchgPos;
}

void HighsSearch::branchUpwards(HighsInt col, double newlb,
                                double branchpoint) {
  NodeData& currnode = nodestack.back();

  assert(currnode.opensubtrees == 2);
  assert(mipsolver.variableType(col) != HighsVarType::kContinuous);

  currnode.opensubtrees = 1;
  currnode.branching_point = branchpoint;
  currnode.branchingdecision.column = col;
  currnode.branchingdecision.boundval = newlb;
  currnode.branchingdecision.boundtype = HighsBoundType::kLower;

  HighsInt domchgPos = localdom.getDomainChangeStack().size();
  bool passStabilizerToChildNode =
      orbitsValidInChildNode(currnode.branchingdecision);
  localdom.changeBound(currnode.branchingdecision);
  nodestack.emplace_back(
      currnode.lower_bound, currnode.estimate, currnode.nodeBasis,
      passStabilizerToChildNode ? currnode.stabilizerOrbits : nullptr);
  nodestack.back().domgchgStackPos = domchgPos;
}

void HighsSearch::addBoundExceedingConflict() {
  if (mipsolver.mipdata_->upper_limit != kHighsInf) {
    double rhs;
    if (lp->computeDualProof(mipsolver.mipdata_->domain,
                             mipsolver.mipdata_->upper_limit, inds, vals,
                             rhs)) {
      if (mipsolver.mipdata_->domain.infeasible()) return;
      localdom.conflictAnalysis(inds.data(), vals.data(), inds.size(), rhs,
                                mipsolver.mipdata_->conflictPool);

      HighsCutGeneration cutGen(*lp, mipsolver.mipdata_->cutpool);
      mipsolver.mipdata_->debugSolution.checkCut(inds.data(), vals.data(),
                                                 inds.size(), rhs);
      cutGen.generateConflict(localdom, inds, vals, rhs);
    }
  }
}

void HighsSearch::addInfeasibleConflict() {
  double rhs;
  if (lp->getLpSolver().getModelStatus() == HighsModelStatus::kObjectiveBound)
    lp->performAging();

  if (lp->computeDualInfProof(mipsolver.mipdata_->domain, inds, vals, rhs)) {
    if (mipsolver.mipdata_->domain.infeasible()) return;
    // double minactlocal = 0.0;
    // double minactglobal = 0.0;
    // for (HighsInt i = 0; i < int(inds.size()); ++i) {
    //  if (vals[i] > 0.0) {
    //    minactlocal += localdom.col_lower_[inds[i]] * vals[i];
    //    minactglobal += globaldom.col_lower_[inds[i]] * vals[i];
    //  } else {
    //    minactlocal += localdom.col_upper_[inds[i]] * vals[i];
    //    minactglobal += globaldom.col_upper_[inds[i]] * vals[i];
    //  }
    //}
    // HighsInt oldnumcuts = cutpool.getNumCuts();
    localdom.conflictAnalysis(inds.data(), vals.data(), inds.size(), rhs,
                              mipsolver.mipdata_->conflictPool);

    HighsCutGeneration cutGen(*lp, mipsolver.mipdata_->cutpool);
    mipsolver.mipdata_->debugSolution.checkCut(inds.data(), vals.data(),
                                               inds.size(), rhs);
    cutGen.generateConflict(localdom, inds, vals, rhs);

    // if (cutpool.getNumCuts() > oldnumcuts) {
    //  printf(
    //      "added cut from infeasibility proof with local min activity %g, "
    //      "global min activity %g, and rhs %g\n",
    //      minactlocal, minactglobal, rhs);
    //} else {
    //  printf(
    //      "no cut found for infeasibility proof with local min activity %g, "
    //      "global min "
    //      " activity %g, and rhs % g\n ",
    //      minactlocal, minactglobal, rhs);
    //}
    // HighsInt cutind = cutpool.addCut(inds.data(), vals.data(), inds.size(),
    // rhs); localdom.cutAdded(cutind);
  }
}

HighsInt HighsSearch::selectBranchingCandidate(int64_t maxSbIters,
                                               double& downNodeLb,
                                               double& upNodeLb) {
  assert(!lp->getFractionalIntegers().empty());

  static constexpr HighsInt basisstart_threshold = 20;
  std::vector<double> upscore;
  std::vector<double> downscore;
  std::vector<uint8_t> upscorereliable;
  std::vector<uint8_t> downscorereliable;
  std::vector<double> upbound;
  std::vector<double> downbound;

  HighsInt numfrac = lp->getFractionalIntegers().size();
  const auto& fracints = lp->getFractionalIntegers();

  upscore.resize(numfrac, kHighsInf);
  downscore.resize(numfrac, kHighsInf);
  upbound.resize(numfrac, getCurrentLowerBound());
  downbound.resize(numfrac, getCurrentLowerBound());

  upscorereliable.resize(numfrac, 0);
  downscorereliable.resize(numfrac, 0);

  // initialize up and down scores of variables that have a
  // reliable pseudocost so that they do not get evaluated
  for (HighsInt k = 0; k != numfrac; ++k) {
    HighsInt col = fracints[k].first;
    double fracval = fracints[k].second;

    assert(fracval > localdom.col_lower_[col] + mipsolver.mipdata_->feastol);
    assert(fracval < localdom.col_upper_[col] - mipsolver.mipdata_->feastol);

    if (pseudocost.isReliable(col)) {
      upscore[k] = pseudocost.getPseudocostUp(col, fracval);
      downscore[k] = pseudocost.getPseudocostDown(col, fracval);
      upscorereliable[k] = true;
      downscorereliable[k] = true;
    } else {
      int flags = branchingVarReliableAtNodeFlags(col);
      if (flags & kUpReliable) {
        upscore[k] = pseudocost.getPseudocostUp(col, fracval);
        upscorereliable[k] = true;
      }

      if (flags & kDownReliable) {
        downscore[k] = pseudocost.getPseudocostDown(col, fracval);
        downscorereliable[k] = true;
      }
    }
  }

  std::vector<HighsInt> evalqueue;
  evalqueue.resize(numfrac);
  std::iota(evalqueue.begin(), evalqueue.end(), 0);

  auto numNodesUp = [&](HighsInt k) {
    return mipsolver.mipdata_->nodequeue.numNodesUp(fracints[k].first);
  };

  auto numNodesDown = [&](HighsInt k) {
    return mipsolver.mipdata_->nodequeue.numNodesDown(fracints[k].first);
  };

  double minScore = mipsolver.mipdata_->feastol;

  auto selectBestScore = [&](bool finalSelection) {
    HighsInt best = -1;
    double bestscore = -1.0;
    double bestnodes = -1.0;
    int64_t bestnumnodes = 0;

    double oldminscore = minScore;
    for (HighsInt k : evalqueue) {
      double score;

      if (upscore[k] <= oldminscore) upscorereliable[k] = true;
      if (downscore[k] <= oldminscore) downscorereliable[k] = true;

      double s = 1e-3 * std::min(upscorereliable[k] ? upscore[k] : 0,
                                 downscorereliable[k] ? downscore[k] : 0);
      minScore = std::max(s, minScore);

      if (upscore[k] <= oldminscore || downscore[k] <= oldminscore)
        score = pseudocost.getScore(fracints[k].first,
                                    std::min(upscore[k], oldminscore),
                                    std::min(downscore[k], oldminscore));
      else {
        score = upscore[k] == kHighsInf || downscore[k] == kHighsInf
                    ? finalSelection ? pseudocost.getScore(fracints[k].first,
                                                           fracints[k].second)
                                     : kHighsInf
                    : pseudocost.getScore(fracints[k].first, upscore[k],
                                          downscore[k]);
      }

      assert(score >= 0.0);
      int64_t upnodes = numNodesUp(k);
      int64_t downnodes = numNodesDown(k);
      double nodes = 0;
      int64_t numnodes = upnodes + downnodes;
      if (upnodes != 0 || downnodes != 0)
        nodes =
            (downnodes / (double)(numnodes)) * (upnodes / (double)(numnodes));
      if (score > bestscore ||
          (score > bestscore - mipsolver.mipdata_->feastol &&
           std::make_pair(nodes, numnodes) >
               std::make_pair(bestnodes, bestnumnodes))) {
        bestscore = score;
        best = k;
        bestnodes = nodes;
        bestnumnodes = numnodes;
      }
    }

    return best;
  };

  HighsLpRelaxation::Playground playground = lp->playground();
  //  HighsLpRelaxation::ResolveGuard resolveGuard = lp->resolveGuard();

  while (true) {
    bool mustStop = getStrongBranchingLpIterations() >= maxSbIters ||
                    mipsolver.mipdata_->checkLimits();

    HighsInt candidate = selectBestScore(mustStop);

    if ((upscorereliable[candidate] && downscorereliable[candidate]) ||
        mustStop) {
      downNodeLb = downbound[candidate];
      upNodeLb = upbound[candidate];
      return candidate;
    }

    lp->setObjectiveLimit(mipsolver.mipdata_->upper_limit);

    HighsInt col = fracints[candidate].first;
    double fracval = fracints[candidate].second;
    double upval = std::ceil(fracval);
    double downval = std::floor(fracval);

    auto analyzeSolution = [&](double objdelta,
                               const std::vector<double>& sol) {
      HighsInt numChangedCols = localdom.getChangedCols().size();
      HighsInt domchgStackSize = localdom.getDomainChangeStack().size();
      const auto& domchgstack = localdom.getDomainChangeStack();

      for (HighsInt k = 0; k != numfrac; ++k) {
        if (fracints[k].first == col) continue;
        double otherfracval = fracints[k].second;
        double otherdownval = std::floor(fracints[k].second);
        double otherupval = std::ceil(fracints[k].second);
        if (sol[fracints[k].first] <=
            otherdownval + mipsolver.mipdata_->feastol) {
          if (localdom.col_upper_[fracints[k].first] >
              otherdownval + mipsolver.mipdata_->feastol) {
            localdom.changeBound(HighsBoundType::kUpper, fracints[k].first,
                                 otherdownval);
            if (localdom.infeasible()) {
              localdom.conflictAnalysis(mipsolver.mipdata_->conflictPool);
              localdom.backtrack();
              localdom.clearChangedCols(numChangedCols);
              continue;
            }
            localdom.propagate();
            if (localdom.infeasible()) {
              localdom.conflictAnalysis(mipsolver.mipdata_->conflictPool);
              localdom.backtrack();
              localdom.clearChangedCols(numChangedCols);
              continue;
            }

            HighsInt newStackSize = localdom.getDomainChangeStack().size();

            bool solutionValid = true;
            for (HighsInt j = domchgStackSize + 1; j < newStackSize; ++j) {
              if (domchgstack[j].boundtype == HighsBoundType::kLower) {
                if (domchgstack[j].boundval >
                    sol[domchgstack[j].column] + mipsolver.mipdata_->feastol) {
                  solutionValid = false;
                  break;
                }
              } else {
                if (domchgstack[j].boundval <
                    sol[domchgstack[j].column] - mipsolver.mipdata_->feastol) {
                  solutionValid = false;
                  break;
                }
              }
            }

            localdom.backtrack();
            localdom.clearChangedCols(numChangedCols);
            if (!solutionValid) continue;
          }

          if (objdelta <= mipsolver.mipdata_->feastol) {
            pseudocost.addObservation(fracints[k].first,
                                      otherdownval - otherfracval, objdelta);
            markBranchingVarDownReliableAtNode(fracints[k].first);
          }

          downscore[k] = std::min(downscore[k], objdelta);
        } else if (sol[fracints[k].first] >=
                   otherupval - mipsolver.mipdata_->feastol) {
          if (localdom.col_lower_[fracints[k].first] <
              otherupval - mipsolver.mipdata_->feastol) {
            localdom.changeBound(HighsBoundType::kLower, fracints[k].first,
                                 otherupval);

            if (localdom.infeasible()) {
              localdom.conflictAnalysis(mipsolver.mipdata_->conflictPool);
              localdom.backtrack();
              localdom.clearChangedCols(numChangedCols);
              continue;
            }
            localdom.propagate();
            if (localdom.infeasible()) {
              localdom.conflictAnalysis(mipsolver.mipdata_->conflictPool);
              localdom.backtrack();
              localdom.clearChangedCols(numChangedCols);
              continue;
            }

            HighsInt newStackSize = localdom.getDomainChangeStack().size();

            bool solutionValid = true;
            for (HighsInt j = domchgStackSize + 1; j < newStackSize; ++j) {
              if (domchgstack[j].boundtype == HighsBoundType::kLower) {
                if (domchgstack[j].boundval >
                    sol[domchgstack[j].column] + mipsolver.mipdata_->feastol) {
                  solutionValid = false;
                  break;
                }
              } else {
                if (domchgstack[j].boundval <
                    sol[domchgstack[j].column] - mipsolver.mipdata_->feastol) {
                  solutionValid = false;
                  break;
                }
              }
            }

            localdom.backtrack();
            localdom.clearChangedCols(numChangedCols);

            if (!solutionValid) continue;
          }

          if (objdelta <= mipsolver.mipdata_->feastol) {
            pseudocost.addObservation(fracints[k].first,
                                      otherupval - otherfracval, objdelta);
            markBranchingVarUpReliableAtNode(fracints[k].first);
          }

          upscore[k] = std::min(upscore[k], objdelta);
        }
      }
    };

    if (!downscorereliable[candidate] &&
        (upscorereliable[candidate] ||
         std::make_pair(downscore[candidate],
                        pseudocost.getAvgInferencesDown(col)) >=
             std::make_pair(upscore[candidate],
                            pseudocost.getAvgInferencesUp(col)))) {
      // evaluate down branch
      // if (!mipsolver.submip)
      //   printf("down eval col=%d fracval=%g\n", col, fracval);
      int64_t inferences = -(int64_t)localdom.getDomainChangeStack().size() - 1;

      HighsDomainChange domchg{downval, col, HighsBoundType::kUpper};
      bool orbitalFixing =
          nodestack.back().stabilizerOrbits && orbitsValidInChildNode(domchg);
      localdom.changeBound(domchg);
      localdom.propagate();

      if (!localdom.infeasible()) {
        if (orbitalFixing)
          nodestack.back().stabilizerOrbits->orbitalFixing(localdom);
        else
          mipsolver.mipdata_->symmetries.propagateOrbitopes(localdom);
      }

      inferences += localdom.getDomainChangeStack().size();
      if (localdom.infeasible()) {
        localdom.conflictAnalysis(mipsolver.mipdata_->conflictPool);
        pseudocost.addCutoffObservation(col, false);
        localdom.backtrack();
        localdom.clearChangedCols();

        branchUpwards(col, upval, fracval);
        nodestack[nodestack.size() - 2].opensubtrees = 0;
        nodestack[nodestack.size() - 2].skipDepthCount = 1;
        depthoffset -= 1;

        return -1;
      }

      pseudocost.addInferenceObservation(col, inferences, false);

      int64_t numiters = lp->getNumLpIterations();
      HighsLpRelaxation::Status status = playground.solveLp(localdom);
      numiters = lp->getNumLpIterations() - numiters;
      lpiterations += numiters;
      sblpiterations += numiters;

      if (lp->scaledOptimal(status)) {
        lp->performAging();

        double delta = downval - fracval;
        bool integerfeasible;
        const std::vector<double>& sol = lp->getSolution().col_value;
        double solobj = checkSol(sol, integerfeasible);

        double objdelta = std::max(solobj - lp->getObjective(), 0.0);
        if (objdelta <= mipsolver.mipdata_->epsilon) objdelta = 0.0;

        downscore[candidate] = objdelta;
        downscorereliable[candidate] = true;

        markBranchingVarDownReliableAtNode(col);
        pseudocost.addObservation(col, delta, objdelta);
        analyzeSolution(objdelta, sol);

        if (lp->unscaledPrimalFeasible(status) && integerfeasible) {
          double cutoffbnd = getCutoffBound();
          mipsolver.mipdata_->addIncumbent(
              lp->getLpSolver().getSolution().col_value, solobj,
              inheuristic ? 'H' : 'B');

          if (mipsolver.mipdata_->upper_limit < cutoffbnd)
            lp->setObjectiveLimit(mipsolver.mipdata_->upper_limit);
        }

        if (lp->unscaledDualFeasible(status)) {
          downbound[candidate] = solobj;
          if (solobj > mipsolver.mipdata_->optimality_limit) {
            addBoundExceedingConflict();

            bool pruned = solobj > getCutoffBound();
            if (pruned) mipsolver.mipdata_->debugSolution.nodePruned(localdom);

            localdom.backtrack();
            lp->flushDomain(localdom);

            branchUpwards(col, upval, fracval);
            nodestack[nodestack.size() - 2].opensubtrees = pruned ? 0 : 1;
            nodestack[nodestack.size() - 2].other_child_lb = solobj;
            nodestack[nodestack.size() - 2].skipDepthCount = 1;
            depthoffset -= 1;

            return -1;
          }
        } else if (solobj > getCutoffBound()) {
          addBoundExceedingConflict();
          localdom.propagate();
          bool infeas = localdom.infeasible();
          if (infeas) {
            localdom.backtrack();
            lp->flushDomain(localdom);

            branchUpwards(col, upval, fracval);
            nodestack[nodestack.size() - 2].opensubtrees = 0;
            nodestack[nodestack.size() - 2].skipDepthCount = 1;
            depthoffset -= 1;

            return -1;
          }
        }
      } else if (status == HighsLpRelaxation::Status::kInfeasible) {
        mipsolver.mipdata_->debugSolution.nodePruned(localdom);
        addInfeasibleConflict();
        pseudocost.addCutoffObservation(col, false);
        localdom.backtrack();
        lp->flushDomain(localdom);

        branchUpwards(col, upval, fracval);
        nodestack[nodestack.size() - 2].opensubtrees = 0;
        nodestack[nodestack.size() - 2].skipDepthCount = 1;
        depthoffset -= 1;

        return -1;
      } else {
        // printf("todo2\n");
        // in case of an LP error we set the score of this variable to zero to
        // avoid choosing it as branching candidate if possible
        downscore[candidate] = 0.0;
        upscore[candidate] = 0.0;
        downscorereliable[candidate] = 1;
        upscorereliable[candidate] = 1;
        markBranchingVarUpReliableAtNode(col);
        markBranchingVarDownReliableAtNode(col);
      }

      localdom.backtrack();
      lp->flushDomain(localdom);
    } else {
      // if (!mipsolver.submip)
      //  printf("up eval col=%d fracval=%g\n", col, fracval);
      // evaluate up branch
      int64_t inferences = -(int64_t)localdom.getDomainChangeStack().size() - 1;
      HighsDomainChange domchg{upval, col, HighsBoundType::kLower};
      bool orbitalFixing =
          nodestack.back().stabilizerOrbits && orbitsValidInChildNode(domchg);
      localdom.changeBound(domchg);
      localdom.propagate();

      if (!localdom.infeasible()) {
        if (orbitalFixing)
          nodestack.back().stabilizerOrbits->orbitalFixing(localdom);
        else
          mipsolver.mipdata_->symmetries.propagateOrbitopes(localdom);
      }

      inferences += localdom.getDomainChangeStack().size();
      if (localdom.infeasible()) {
        localdom.conflictAnalysis(mipsolver.mipdata_->conflictPool);
        pseudocost.addCutoffObservation(col, true);
        localdom.backtrack();
        localdom.clearChangedCols();

        branchDownwards(col, downval, fracval);
        nodestack[nodestack.size() - 2].opensubtrees = 0;
        nodestack[nodestack.size() - 2].skipDepthCount = 1;
        depthoffset -= 1;

        return -1;
      }

      pseudocost.addInferenceObservation(col, inferences, true);

      int64_t numiters = lp->getNumLpIterations();
      HighsLpRelaxation::Status status = playground.solveLp(localdom);
      numiters = lp->getNumLpIterations() - numiters;
      lpiterations += numiters;
      sblpiterations += numiters;

      if (lp->scaledOptimal(status)) {
        lp->performAging();

        double delta = upval - fracval;
        bool integerfeasible;

        const std::vector<double>& sol =
            lp->getLpSolver().getSolution().col_value;
        double solobj = checkSol(sol, integerfeasible);

        double objdelta = std::max(solobj - lp->getObjective(), 0.0);
        if (objdelta <= mipsolver.mipdata_->epsilon) objdelta = 0.0;

        upscore[candidate] = objdelta;
        upscorereliable[candidate] = true;

        markBranchingVarUpReliableAtNode(col);
        pseudocost.addObservation(col, delta, objdelta);
        analyzeSolution(objdelta, sol);

        if (lp->unscaledPrimalFeasible(status) && integerfeasible) {
          double cutoffbnd = getCutoffBound();
          mipsolver.mipdata_->addIncumbent(
              lp->getLpSolver().getSolution().col_value, solobj,
              inheuristic ? 'H' : 'B');

          if (mipsolver.mipdata_->upper_limit < cutoffbnd)
            lp->setObjectiveLimit(mipsolver.mipdata_->upper_limit);
        }

        if (lp->unscaledDualFeasible(status)) {
          upbound[candidate] = solobj;
          if (solobj > mipsolver.mipdata_->optimality_limit) {
            addBoundExceedingConflict();

            bool pruned = solobj > getCutoffBound();
            if (pruned) mipsolver.mipdata_->debugSolution.nodePruned(localdom);

            localdom.backtrack();
            lp->flushDomain(localdom);

            branchDownwards(col, downval, fracval);
            nodestack[nodestack.size() - 2].opensubtrees = pruned ? 0 : 1;
            nodestack[nodestack.size() - 2].other_child_lb = solobj;
            nodestack[nodestack.size() - 2].skipDepthCount = 1;
            depthoffset -= 1;

            return -1;
          }
        } else if (solobj > getCutoffBound()) {
          addBoundExceedingConflict();
          localdom.propagate();
          bool infeas = localdom.infeasible();
          if (infeas) {
            localdom.backtrack();
            lp->flushDomain(localdom);

            branchDownwards(col, downval, fracval);
            nodestack[nodestack.size() - 2].opensubtrees = 0;
            nodestack[nodestack.size() - 2].skipDepthCount = 1;
            depthoffset -= 1;

            return -1;
          }
        }
      } else if (status == HighsLpRelaxation::Status::kInfeasible) {
        mipsolver.mipdata_->debugSolution.nodePruned(localdom);
        addInfeasibleConflict();
        pseudocost.addCutoffObservation(col, true);
        localdom.backtrack();
        lp->flushDomain(localdom);

        branchDownwards(col, downval, fracval);
        nodestack[nodestack.size() - 2].opensubtrees = 0;
        nodestack[nodestack.size() - 2].skipDepthCount = 1;
        depthoffset -= 1;

        return -1;
      } else {
        // printf("todo2\n");
        // in case of an LP error we set the score of this variable to zero to
        // avoid choosing it as branching candidate if possible
        downscore[candidate] = 0.0;
        upscore[candidate] = 0.0;
        downscorereliable[candidate] = 1;
        upscorereliable[candidate] = 1;
        markBranchingVarUpReliableAtNode(col);
        markBranchingVarDownReliableAtNode(col);
      }

      localdom.backtrack();
      lp->flushDomain(localdom);
    }
  }
}

const HighsSearch::NodeData* HighsSearch::getParentNodeData() const {
  if (nodestack.size() <= 1) return nullptr;

  return &nodestack[nodestack.size() - 2];
}

void HighsSearch::currentNodeToQueue(HighsNodeQueue& nodequeue) {
  auto oldchangedcols = localdom.getChangedCols().size();
  bool prune = nodestack.back().lower_bound > getCutoffBound();
  if (!prune) {
    localdom.propagate();
    localdom.clearChangedCols(oldchangedcols);
    prune = localdom.infeasible();
    if (prune) localdom.conflictAnalysis(mipsolver.mipdata_->conflictPool);
  }
  if (!prune) {
    std::vector<HighsInt> branchPositions;
    auto domchgStack = localdom.getReducedDomainChangeStack(branchPositions);
    double tmpTreeWeight = nodequeue.emplaceNode(
        std::move(domchgStack), std::move(branchPositions),
        std::max(nodestack.back().lower_bound,
                 localdom.getObjectiveLowerBound()),
        nodestack.back().estimate, getCurrentDepth());
    if (countTreeWeight) treeweight += tmpTreeWeight;
  } else {
    mipsolver.mipdata_->debugSolution.nodePruned(localdom);
    if (countTreeWeight) treeweight += std::ldexp(1.0, 1 - getCurrentDepth());
  }
  nodestack.back().opensubtrees = 0;
}

void HighsSearch::openNodesToQueue(HighsNodeQueue& nodequeue) {
  if (nodestack.empty()) return;

  // get the basis of the node highest up in the tree
  std::shared_ptr<const HighsBasis> basis;
  for (NodeData& nodeData : nodestack) {
    if (nodeData.nodeBasis) {
      basis = std::move(nodeData.nodeBasis);
      break;
    }
  }

  if (nodestack.back().opensubtrees == 0) backtrack(false);

  while (!nodestack.empty()) {
    auto oldchangedcols = localdom.getChangedCols().size();
    bool prune = nodestack.back().lower_bound > getCutoffBound();
    if (!prune) {
      localdom.propagate();
      localdom.clearChangedCols(oldchangedcols);
      prune = localdom.infeasible();
      if (prune) localdom.conflictAnalysis(mipsolver.mipdata_->conflictPool);
    }
    if (!prune) {
      std::vector<HighsInt> branchPositions;
      auto domchgStack = localdom.getReducedDomainChangeStack(branchPositions);
      double tmpTreeWeight = nodequeue.emplaceNode(
          std::move(domchgStack), std::move(branchPositions),
          std::max(nodestack.back().lower_bound,
                   localdom.getObjectiveLowerBound()),
          nodestack.back().estimate, getCurrentDepth());
      if (countTreeWeight) treeweight += tmpTreeWeight;
    } else {
      mipsolver.mipdata_->debugSolution.nodePruned(localdom);
      if (countTreeWeight) treeweight += std::ldexp(1.0, 1 - getCurrentDepth());
    }
    nodestack.back().opensubtrees = 0;
    backtrack(false);
  }

  lp->flushDomain(localdom);
  if (basis) {
    if (basis->row_status.size() == lp->numRows())
      lp->setStoredBasis(std::move(basis));
    lp->recoverBasis();
  }
}

void HighsSearch::flushStatistics() {
  mipsolver.mipdata_->num_nodes += nnodes;
  nnodes = 0;

  mipsolver.mipdata_->pruned_treeweight += treeweight;
  treeweight = 0;

  mipsolver.mipdata_->total_lp_iterations += lpiterations;
  lpiterations = 0;

  mipsolver.mipdata_->heuristic_lp_iterations += heurlpiterations;
  heurlpiterations = 0;

  mipsolver.mipdata_->sb_lp_iterations += sblpiterations;
  sblpiterations = 0;
}

int64_t HighsSearch::getHeuristicLpIterations() const {
  return heurlpiterations + mipsolver.mipdata_->heuristic_lp_iterations;
}

int64_t HighsSearch::getTotalLpIterations() const {
  return lpiterations + mipsolver.mipdata_->total_lp_iterations;
}

int64_t HighsSearch::getLocalLpIterations() const { return lpiterations; }

int64_t HighsSearch::getLocalNodes() const { return nnodes; }

int64_t HighsSearch::getStrongBranchingLpIterations() const {
  return sblpiterations + mipsolver.mipdata_->sb_lp_iterations;
}

void HighsSearch::resetLocalDomain() {
  this->lp->resetToGlobalDomain();
  localdom = mipsolver.mipdata_->domain;

#ifndef NDEBUG
  for (HighsInt i = 0; i != mipsolver.numCol(); ++i) {
    assert(lp->getLpSolver().getLp().col_lower_[i] == localdom.col_lower_[i] ||
           mipsolver.variableType(i) == HighsVarType::kContinuous);
    assert(lp->getLpSolver().getLp().col_upper_[i] == localdom.col_upper_[i] ||
           mipsolver.variableType(i) == HighsVarType::kContinuous);
  }
#endif
}

void HighsSearch::installNode(HighsNodeQueue::OpenNode&& node) {
  localdom.setDomainChangeStack(node.domchgstack, node.branchings);
  bool globalSymmetriesValid = true;
  if (mipsolver.mipdata_->globalOrbits) {
    // if global orbits have been computed we check whether they are still valid
    // in this node
    const auto& domchgstack = localdom.getDomainChangeStack();
    const auto& branchpos = localdom.getBranchingPositions();
    for (HighsInt i : localdom.getBranchingPositions()) {
      HighsInt col = domchgstack[i].column;
      if (mipsolver.mipdata_->symmetries.columnPosition[col] == -1) continue;

      if (!mipsolver.mipdata_->domain.isBinary(col) ||
          (domchgstack[i].boundtype == HighsBoundType::kLower &&
           domchgstack[i].boundval == 1.0)) {
        globalSymmetriesValid = false;
        break;
      }
    }
  }
  nodestack.emplace_back(
      node.lower_bound, node.estimate, nullptr,
      globalSymmetriesValid ? mipsolver.mipdata_->globalOrbits : nullptr);
  subrootsol.clear();
  depthoffset = node.depth - 1;
}

HighsSearch::NodeResult HighsSearch::evaluateNode() {
  assert(!nodestack.empty());
  NodeData& currnode = nodestack.back();
  const NodeData* parent = getParentNodeData();

  const auto& domchgstack = localdom.getDomainChangeStack();

  if (!inheuristic &&
      currnode.lower_bound > mipsolver.mipdata_->optimality_limit)
    return NodeResult::kSubOptimal;

  localdom.propagate();

  if (!inheuristic && !localdom.infeasible()) {
    if (mipsolver.mipdata_->symmetries.numPerms > 0 &&
        !currnode.stabilizerOrbits &&
        (parent == nullptr || !parent->stabilizerOrbits ||
         !parent->stabilizerOrbits->orbitCols.empty())) {
      currnode.stabilizerOrbits =
          mipsolver.mipdata_->symmetries.computeStabilizerOrbits(localdom);
    }

    if (currnode.stabilizerOrbits)
      currnode.stabilizerOrbits->orbitalFixing(localdom);
    else
      mipsolver.mipdata_->symmetries.propagateOrbitopes(localdom);
  }
  if (parent != nullptr) {
    int64_t inferences = domchgstack.size() - (currnode.domgchgStackPos + 1);

    pseudocost.addInferenceObservation(
        parent->branchingdecision.column, inferences,
        parent->branchingdecision.boundtype == HighsBoundType::kLower);
  }

  NodeResult result = NodeResult::kOpen;

  if (localdom.infeasible()) {
    result = NodeResult::kDomainInfeasible;
    localdom.clearChangedCols();
    if (parent != nullptr && parent->lp_objective != -kHighsInf &&
        parent->branching_point != parent->branchingdecision.boundval) {
      bool upbranch =
          parent->branchingdecision.boundtype == HighsBoundType::kLower;
      pseudocost.addCutoffObservation(parent->branchingdecision.column,
                                      upbranch);
    }

    localdom.conflictAnalysis(mipsolver.mipdata_->conflictPool);
  } else {
    lp->flushDomain(localdom);
    lp->setObjectiveLimit(mipsolver.mipdata_->upper_limit);

#ifndef NDEBUG
    for (HighsInt i = 0; i != mipsolver.numCol(); ++i) {
      assert(lp->getLpSolver().getLp().col_lower_[i] ==
                 localdom.col_lower_[i] ||
             mipsolver.variableType(i) == HighsVarType::kContinuous);
      assert(lp->getLpSolver().getLp().col_upper_[i] ==
                 localdom.col_upper_[i] ||
             mipsolver.variableType(i) == HighsVarType::kContinuous);
    }
#endif
    int64_t oldnumiters = lp->getNumLpIterations();
    HighsLpRelaxation::Status status = lp->resolveLp(&localdom);
    lpiterations += lp->getNumLpIterations() - oldnumiters;

    currnode.lower_bound =
        std::max(localdom.getObjectiveLowerBound(), currnode.lower_bound);

    if (localdom.infeasible()) {
      result = NodeResult::kDomainInfeasible;
      localdom.clearChangedCols();
      if (parent != nullptr && parent->lp_objective != -kHighsInf &&
          parent->branching_point != parent->branchingdecision.boundval) {
        bool upbranch =
            parent->branchingdecision.boundtype == HighsBoundType::kLower;
        pseudocost.addCutoffObservation(parent->branchingdecision.column,
                                        upbranch);
      }

      localdom.conflictAnalysis(mipsolver.mipdata_->conflictPool);
    } else if (lp->scaledOptimal(status)) {
      lp->storeBasis();
      lp->performAging();

      currnode.nodeBasis = lp->getStoredBasis();
      currnode.estimate = lp->computeBestEstimate(pseudocost);
      currnode.lp_objective = lp->getObjective();

      if (parent != nullptr && parent->lp_objective != -kHighsInf &&
          parent->branching_point != parent->branchingdecision.boundval) {
        double delta =
            parent->branchingdecision.boundval - parent->branching_point;
        double objdelta =
            std::max(0.0, currnode.lp_objective - parent->lp_objective);

        pseudocost.addObservation(parent->branchingdecision.column, delta,
                                  objdelta);
      }

      if (lp->unscaledPrimalFeasible(status)) {
        if (lp->getFractionalIntegers().empty()) {
          double cutoffbnd = getCutoffBound();
          mipsolver.mipdata_->addIncumbent(
              lp->getLpSolver().getSolution().col_value, lp->getObjective(),
              inheuristic ? 'H' : 'T');
          if (mipsolver.mipdata_->upper_limit < cutoffbnd)
            lp->setObjectiveLimit(mipsolver.mipdata_->upper_limit);

          if (lp->unscaledDualFeasible(status)) {
            addBoundExceedingConflict();
            result = NodeResult::kBoundExceeding;
          }
        }
      }

      if (result == NodeResult::kOpen) {
        if (lp->unscaledDualFeasible(status)) {
          currnode.lower_bound =
              std::max(currnode.lp_objective, currnode.lower_bound);

          if (currnode.lower_bound > getCutoffBound()) {
            result = NodeResult::kBoundExceeding;
            addBoundExceedingConflict();
          } else if (mipsolver.mipdata_->upper_limit != kHighsInf) {
            if (!inheuristic) {
              double gap = mipsolver.mipdata_->upper_limit - lp->getObjective();
              lp->computeBasicDegenerateDuals(
                  gap + std::max(10 * mipsolver.mipdata_->feastol,
                                 mipsolver.mipdata_->epsilon * gap),
                  &localdom);
            }
            HighsRedcostFixing::propagateRedCost(mipsolver, localdom, *lp);
            localdom.propagate();
            if (localdom.infeasible()) {
              result = NodeResult::kDomainInfeasible;
              localdom.clearChangedCols();
              if (parent != nullptr && parent->lp_objective != -kHighsInf &&
                  parent->branching_point !=
                      parent->branchingdecision.boundval) {
                bool upbranch = parent->branchingdecision.boundtype ==
                                HighsBoundType::kLower;
                pseudocost.addCutoffObservation(
                    parent->branchingdecision.column, upbranch);
              }

              localdom.conflictAnalysis(mipsolver.mipdata_->conflictPool);
            } else if (!localdom.getChangedCols().empty()) {
              return evaluateNode();
            }
          } else {
            if (!inheuristic) {
              lp->computeBasicDegenerateDuals(kHighsInf, &localdom);
              localdom.propagate();
              if (localdom.infeasible()) {
                result = NodeResult::kDomainInfeasible;
                localdom.clearChangedCols();
                if (parent != nullptr && parent->lp_objective != -kHighsInf &&
                    parent->branching_point !=
                        parent->branchingdecision.boundval) {
                  bool upbranch = parent->branchingdecision.boundtype ==
                                  HighsBoundType::kLower;
                  pseudocost.addCutoffObservation(
                      parent->branchingdecision.column, upbranch);
                }

                localdom.conflictAnalysis(mipsolver.mipdata_->conflictPool);
              } else if (!localdom.getChangedCols().empty()) {
                return evaluateNode();
              }
            }
          }
        } else if (lp->getObjective() > getCutoffBound()) {
          // the LP is not solved to dual feasibilty due to scaling/numerics
          // therefore we compute a conflict constraint as if the LP was bound
          // exceeding and propagate the local domain again. The lp relaxation
          // class will take care to consider the dual multipliers with an
          // increased zero tolerance due to the dual infeasibility when
          // computing the proof conBoundExceedingstraint.
          addBoundExceedingConflict();
          localdom.propagate();
          if (localdom.infeasible()) {
            result = NodeResult::kBoundExceeding;
          }
        }
      }
    } else if (status == HighsLpRelaxation::Status::kInfeasible) {
      if (lp->getLpSolver().getModelStatus() ==
          HighsModelStatus::kObjectiveBound)
        result = NodeResult::kBoundExceeding;
      else
        result = NodeResult::kLpInfeasible;
      addInfeasibleConflict();
      if (parent != nullptr && parent->lp_objective != -kHighsInf &&
          parent->branching_point != parent->branchingdecision.boundval) {
        bool upbranch =
            parent->branchingdecision.boundtype == HighsBoundType::kLower;
        pseudocost.addCutoffObservation(parent->branchingdecision.column,
                                        upbranch);
      }
    }
  }

  if (result != NodeResult::kOpen) {
    mipsolver.mipdata_->debugSolution.nodePruned(localdom);
    treeweight += std::ldexp(1.0, 1 - getCurrentDepth());
    currnode.opensubtrees = 0;
  } else if (!inheuristic) {
    if (currnode.lower_bound > mipsolver.mipdata_->optimality_limit) {
      result = NodeResult::kSubOptimal;
      addBoundExceedingConflict();
    }
  }

  return result;
}

HighsSearch::NodeResult HighsSearch::branch() {
  assert(localdom.getChangedCols().empty());

  assert(nodestack.back().opensubtrees == 2);
  nodestack.back().branchingdecision.column = -1;
  inbranching = true;

  HighsInt minrel = pseudocost.getMinReliable();
  double childLb = getCurrentLowerBound();
  NodeResult result = NodeResult::kOpen;
  while (nodestack.back().opensubtrees == 2 &&
         lp->scaledOptimal(lp->getStatus()) &&
         !lp->getFractionalIntegers().empty()) {
    int64_t sbmaxiters = 0;
    if (minrel > 0) {
      int64_t sbiters = getStrongBranchingLpIterations();
      sbmaxiters =
          100000 + ((getTotalLpIterations() - getHeuristicLpIterations() -
                     getStrongBranchingLpIterations()) >>
                    1);
      if (sbiters > sbmaxiters) {
        pseudocost.setMinReliable(0);
      } else if (sbiters > (sbmaxiters >> 1)) {
        double reductionratio = (sbiters - (sbmaxiters >> 1)) /
                                (double)(sbmaxiters - (sbmaxiters >> 1));

        HighsInt minrelreduced = int(minrel - reductionratio * (minrel - 1));
        pseudocost.setMinReliable(std::min(minrel, minrelreduced));
      }
    }

    double degeneracyFac = lp->computeLPDegneracy(localdom);
    pseudocost.setDegeneracyFactor(degeneracyFac);
    if (degeneracyFac >= 10.0) pseudocost.setMinReliable(0);
    // if (!mipsolver.submip)
    //  printf("selecting branching cand with minrel=%d\n",
    //         pseudocost.getMinReliable());
    double downNodeLb = getCurrentLowerBound();
    double upNodeLb = getCurrentLowerBound();
    HighsInt branchcand =
        selectBranchingCandidate(sbmaxiters, downNodeLb, upNodeLb);
    // if (!mipsolver.submip)
    //   printf("branching cand returned as %d\n", branchcand);
    NodeData& currnode = nodestack.back();
    childLb = currnode.lower_bound;
    if (branchcand != -1) {
      auto branching = lp->getFractionalIntegers()[branchcand];
      currnode.branchingdecision.column = branching.first;
      currnode.branching_point = branching.second;

      HighsInt col = branching.first;

      switch (childselrule) {
        case ChildSelectionRule::kUp:
          currnode.branchingdecision.boundtype = HighsBoundType::kLower;
          currnode.branchingdecision.boundval =
              std::ceil(currnode.branching_point);
          currnode.other_child_lb = downNodeLb;
          childLb = upNodeLb;
          break;
        case ChildSelectionRule::kDown:
          currnode.branchingdecision.boundtype = HighsBoundType::kUpper;
          currnode.branchingdecision.boundval =
              std::floor(currnode.branching_point);
          currnode.other_child_lb = upNodeLb;
          childLb = downNodeLb;
          break;
        case ChildSelectionRule::kRootSol: {
          double downPrio = pseudocost.getAvgInferencesDown(col) +
                            mipsolver.mipdata_->epsilon;
          double upPrio =
              pseudocost.getAvgInferencesUp(col) + mipsolver.mipdata_->epsilon;
          double downVal = std::floor(currnode.branching_point);
          double upVal = std::ceil(currnode.branching_point);
          if (!subrootsol.empty()) {
            double rootsol = subrootsol[col];
            if (rootsol < downVal)
              rootsol = downVal;
            else if (rootsol > upVal)
              rootsol = upVal;

            upPrio *= (1.0 + (currnode.branching_point - rootsol));
            downPrio *= (1.0 + (rootsol - currnode.branching_point));

          } else {
            if (currnode.lp_objective != -kHighsInf)
              subrootsol = lp->getSolution().col_value;
            if (!mipsolver.mipdata_->rootlpsol.empty()) {
              double rootsol = mipsolver.mipdata_->rootlpsol[col];
              if (rootsol < downVal)
                rootsol = downVal;
              else if (rootsol > upVal)
                rootsol = upVal;

              upPrio *= (1.0 + (currnode.branching_point - rootsol));
              downPrio *= (1.0 + (rootsol - currnode.branching_point));
            }
          }
          if (upPrio + mipsolver.mipdata_->epsilon >= downPrio) {
            currnode.branchingdecision.boundtype = HighsBoundType::kLower;
            currnode.branchingdecision.boundval = upVal;
            currnode.other_child_lb = downNodeLb;
            childLb = upNodeLb;
          } else {
            currnode.branchingdecision.boundtype = HighsBoundType::kUpper;
            currnode.branchingdecision.boundval = downVal;
            currnode.other_child_lb = upNodeLb;
            childLb = downNodeLb;
          }
          break;
        }
        case ChildSelectionRule::kObj:
          if (mipsolver.colCost(col) >= 0) {
            currnode.branchingdecision.boundtype = HighsBoundType::kLower;
            currnode.branchingdecision.boundval =
                std::ceil(currnode.branching_point);
            currnode.other_child_lb = downNodeLb;
            childLb = upNodeLb;
          } else {
            currnode.branchingdecision.boundtype = HighsBoundType::kUpper;
            currnode.branchingdecision.boundval =
                std::floor(currnode.branching_point);
            currnode.other_child_lb = upNodeLb;
            childLb = downNodeLb;
          }
          break;
        case ChildSelectionRule::kRandom:
          if (random.bit()) {
            currnode.branchingdecision.boundtype = HighsBoundType::kLower;
            currnode.branchingdecision.boundval =
                std::ceil(currnode.branching_point);
            currnode.other_child_lb = downNodeLb;
            childLb = upNodeLb;
          } else {
            currnode.branchingdecision.boundtype = HighsBoundType::kUpper;
            currnode.branchingdecision.boundval =
                std::floor(currnode.branching_point);
            currnode.other_child_lb = upNodeLb;
            childLb = downNodeLb;
          }
          break;
        case ChildSelectionRule::kBestCost: {
          if (pseudocost.getPseudocostUp(col, currnode.branching_point,
                                         mipsolver.mipdata_->feastol) >
              pseudocost.getPseudocostDown(col, currnode.branching_point,
                                           mipsolver.mipdata_->feastol)) {
            currnode.branchingdecision.boundtype = HighsBoundType::kUpper;
            currnode.branchingdecision.boundval =
                std::floor(currnode.branching_point);
            currnode.other_child_lb = upNodeLb;
            childLb = downNodeLb;
          } else {
            currnode.branchingdecision.boundtype = HighsBoundType::kLower;
            currnode.branchingdecision.boundval =
                std::ceil(currnode.branching_point);
            currnode.other_child_lb = downNodeLb;
            childLb = upNodeLb;
          }
          break;
        }
        case ChildSelectionRule::kWorstCost:
          if (pseudocost.getPseudocostUp(col, currnode.branching_point) >=
              pseudocost.getPseudocostDown(col, currnode.branching_point)) {
            currnode.branchingdecision.boundtype = HighsBoundType::kLower;
            currnode.branchingdecision.boundval =
                std::ceil(currnode.branching_point);
            currnode.other_child_lb = downNodeLb;
            childLb = upNodeLb;
          } else {
            currnode.branchingdecision.boundtype = HighsBoundType::kUpper;
            currnode.branchingdecision.boundval =
                std::floor(currnode.branching_point);
            currnode.other_child_lb = upNodeLb;
            childLb = downNodeLb;
          }
          break;
        case ChildSelectionRule::kDisjunction: {
          int64_t numnodesup;
          int64_t numnodesdown;
          numnodesup = mipsolver.mipdata_->nodequeue.numNodesUp(col);
          numnodesdown = mipsolver.mipdata_->nodequeue.numNodesDown(col);
          if (numnodesup > numnodesdown) {
            currnode.branchingdecision.boundtype = HighsBoundType::kLower;
            currnode.branchingdecision.boundval =
                std::ceil(currnode.branching_point);
            currnode.other_child_lb = downNodeLb;
            childLb = upNodeLb;
          } else if (numnodesdown > numnodesup) {
            currnode.branchingdecision.boundtype = HighsBoundType::kUpper;
            currnode.branchingdecision.boundval =
                std::floor(currnode.branching_point);
            currnode.other_child_lb = upNodeLb;
            childLb = downNodeLb;
          } else {
            if (mipsolver.colCost(col) >= 0) {
              currnode.branchingdecision.boundtype = HighsBoundType::kLower;
              currnode.branchingdecision.boundval =
                  std::ceil(currnode.branching_point);
              currnode.other_child_lb = downNodeLb;
              childLb = upNodeLb;
            } else {
              currnode.branchingdecision.boundtype = HighsBoundType::kUpper;
              currnode.branchingdecision.boundval =
                  std::floor(currnode.branching_point);
              currnode.other_child_lb = upNodeLb;
              childLb = downNodeLb;
            }
          }
          break;
        }
        case ChildSelectionRule::kHybridInferenceCost: {
          double upVal = std::ceil(currnode.branching_point);
          double downVal = std::floor(currnode.branching_point);
          double upScore =
              (1 + pseudocost.getAvgInferencesUp(col)) /
              pseudocost.getPseudocostUp(col, currnode.branching_point,
                                         mipsolver.mipdata_->feastol);
          double downScore =
              (1 + pseudocost.getAvgInferencesDown(col)) /
              pseudocost.getPseudocostDown(col, currnode.branching_point,
                                           mipsolver.mipdata_->feastol);

          if (upScore >= downScore) {
            currnode.branchingdecision.boundtype = HighsBoundType::kLower;
            currnode.branchingdecision.boundval = upVal;
            currnode.other_child_lb = downNodeLb;
            childLb = upNodeLb;
          } else {
            currnode.branchingdecision.boundtype = HighsBoundType::kUpper;
            currnode.branchingdecision.boundval = downVal;
            currnode.other_child_lb = upNodeLb;
            childLb = downNodeLb;
          }
        }
      }
      result = NodeResult::kBranched;
      break;
    }

    assert(!localdom.getChangedCols().empty());
    result = evaluateNode();
    if (result == NodeResult::kSubOptimal) break;
  }
  inbranching = false;
  NodeData& currnode = nodestack.back();
  pseudocost.setMinReliable(minrel);
  pseudocost.setDegeneracyFactor(1.0);

  assert(currnode.opensubtrees == 2 || currnode.opensubtrees == 0);

  if (currnode.opensubtrees != 2 || result == NodeResult::kSubOptimal)
    return result;

  if (currnode.branchingdecision.column == -1) {
    double bestscore = -1.0;
    // solution branching failed, so choose any integer variable to branch
    // on in case we have a different solution status could happen due to a
    // fail in the LP solution process

    for (HighsInt i : mipsolver.mipdata_->integral_cols) {
      if (localdom.col_upper_[i] - localdom.col_lower_[i] < 0.5) continue;

      double fracval;
      if (localdom.col_lower_[i] != -kHighsInf &&
          localdom.col_upper_[i] != kHighsInf)
        fracval = std::floor(0.5 * (localdom.col_lower_[i] +
                                    localdom.col_upper_[i] + 0.5)) +
                  0.5;
      if (localdom.col_lower_[i] != -kHighsInf)
        fracval = localdom.col_lower_[i] + 0.5;
      else if (localdom.col_upper_[i] != kHighsInf)
        fracval = localdom.col_upper_[i] - 0.5;
      else
        fracval = 0.5;

      double score = pseudocost.getScore(i, fracval);
      assert(score >= 0.0);

      if (score > bestscore) {
        bestscore = score;
        if (mipsolver.colCost(i) >= 0) {
          double upval = std::ceil(fracval);
          currnode.branching_point = upval;
          currnode.branchingdecision.boundtype = HighsBoundType::kLower;
          currnode.branchingdecision.column = i;
          currnode.branchingdecision.boundval = upval;
        } else {
          double downval = std::floor(fracval);
          currnode.branching_point = downval;
          currnode.branchingdecision.boundtype = HighsBoundType::kUpper;
          currnode.branchingdecision.column = i;
          currnode.branchingdecision.boundval = downval;
        }
      }
    }
  }

  if (currnode.branchingdecision.column == -1) {
    if (lp->getStatus() == HighsLpRelaxation::Status::kOptimal) {
      // if the LP was solved to optimality and all columns are fixed, then this
      // particular assignment is not feasible or has a worse objective in the
      // original space, otherwise the node would not be open. Hence we prune
      // this particular assignment
      currnode.opensubtrees = 0;
      result = NodeResult::kLpInfeasible;
      return result;
    }
    lp->setIterationLimit();

    // create a fresh LP only with model rows since all integer columns are
    // fixed, the cutting planes are not required and the LP could not be solved
    // so we want to make it as easy as possible
    HighsLpRelaxation lpCopy(mipsolver);
    lpCopy.loadModel();
    lpCopy.getLpSolver().changeColsBounds(0, mipsolver.numCol() - 1,
                                          localdom.col_lower_.data(),
                                          localdom.col_upper_.data());
    // temporarily use the fresh LP for the HighsSearch class
    HighsLpRelaxation* tmpLp = &lpCopy;
    std::swap(tmpLp, lp);

    // reevaluate the node with LP presolve enabled
    lp->getLpSolver().setOptionValue("presolve", "on");
    result = evaluateNode();

    if (result == NodeResult::kOpen) {
      // LP still not solved, reevaluate with primal simplex
      lp->getLpSolver().clearSolver();
      lp->getLpSolver().setOptionValue("simplex_strategy",
                                       kSimplexStrategyPrimal);
      result = evaluateNode();
      lp->getLpSolver().setOptionValue("simplex_strategy",
                                       kSimplexStrategyDual);
      if (result == NodeResult::kOpen) {
        // LP still not solved, reevaluate with IPM instead of simplex
        lp->getLpSolver().clearSolver();
        lp->getLpSolver().setOptionValue("solver", "ipm");
        result = evaluateNode();

        if (result == NodeResult::kOpen) {
          highsLogUser(mipsolver.options_mip_->log_options,
                       HighsLogType::kWarning,
                       "Failed to solve node with all integer columns "
                       "fixed. Declaring node infeasible.\n");
          // LP still not solved, give up and declare as infeasible
          currnode.opensubtrees = 0;
          result = NodeResult::kLpInfeasible;
        }
      }
    }

    // restore old lp relaxation
    std::swap(tmpLp, lp);

    return result;
  }

  // finally open a new node with the branching decision added
  // and remember that we have one open subtree left
  HighsInt domchgPos = localdom.getDomainChangeStack().size();

  bool passStabilizerToChildNode =
      orbitsValidInChildNode(currnode.branchingdecision);
  localdom.changeBound(currnode.branchingdecision);
  currnode.opensubtrees = 1;
  nodestack.emplace_back(
      std::max(childLb, currnode.lower_bound), currnode.estimate,
      currnode.nodeBasis,
      passStabilizerToChildNode ? currnode.stabilizerOrbits : nullptr);
  nodestack.back().domgchgStackPos = domchgPos;

  return NodeResult::kBranched;
}

bool HighsSearch::backtrack(bool recoverBasis) {
  if (nodestack.empty()) return false;
  assert(!nodestack.empty());
  assert(nodestack.back().opensubtrees == 0);
  while (true) {
    while (nodestack.back().opensubtrees == 0) {
      countTreeWeight = true;
      depthoffset += nodestack.back().skipDepthCount;
      if (nodestack.size() == 1) {
        if (recoverBasis && nodestack.back().nodeBasis)
          lp->setStoredBasis(std::move(nodestack.back().nodeBasis));
        nodestack.pop_back();
        localdom.backtrackToGlobal();
        lp->flushDomain(localdom);
        if (recoverBasis) lp->recoverBasis();
        return false;
      }

      nodestack.pop_back();
#ifndef NDEBUG
      HighsDomainChange branchchg =
#endif
          localdom.backtrack();

      if (nodestack.back().opensubtrees != 0) {
        countTreeWeight = nodestack.back().skipDepthCount == 0;
        // repropagate the node, as it may have become infeasible due to
        // conflicts
        HighsInt oldNumDomchgs = localdom.getNumDomainChanges();
        HighsInt oldNumChangedCols = localdom.getChangedCols().size();
        localdom.propagate();
        if (!localdom.infeasible() &&
            oldNumDomchgs != localdom.getNumDomainChanges()) {
          if (nodestack.back().stabilizerOrbits)
            nodestack.back().stabilizerOrbits->orbitalFixing(localdom);
          else
            mipsolver.mipdata_->symmetries.propagateOrbitopes(localdom);
        }
        if (localdom.infeasible()) {
          localdom.clearChangedCols(oldNumChangedCols);
          if (countTreeWeight)
            treeweight += std::ldexp(1.0, -getCurrentDepth());
          nodestack.back().opensubtrees = 0;
        }
      }

      assert(
          (branchchg.boundtype == HighsBoundType::kLower &&
           branchchg.boundval >= nodestack.back().branchingdecision.boundval) ||
          (branchchg.boundtype == HighsBoundType::kUpper &&
           branchchg.boundval <= nodestack.back().branchingdecision.boundval));
      assert(branchchg.boundtype ==
             nodestack.back().branchingdecision.boundtype);
      assert(branchchg.column == nodestack.back().branchingdecision.column);
    }

    NodeData& currnode = nodestack.back();

    assert(currnode.opensubtrees == 1);
    currnode.opensubtrees = 0;
    bool fallbackbranch =
        currnode.branchingdecision.boundval == currnode.branching_point;
    HighsInt domchgPos = localdom.getDomainChangeStack().size();
    if (currnode.branchingdecision.boundtype == HighsBoundType::kLower) {
      currnode.branchingdecision.boundtype = HighsBoundType::kUpper;
      currnode.branchingdecision.boundval =
          std::floor(currnode.branchingdecision.boundval - 0.5);
    } else {
      currnode.branchingdecision.boundtype = HighsBoundType::kLower;
      currnode.branchingdecision.boundval =
          std::ceil(currnode.branchingdecision.boundval + 0.5);
    }

    if (fallbackbranch)
      currnode.branching_point = currnode.branchingdecision.boundval;

    HighsInt numChangedCols = localdom.getChangedCols().size();
    bool passStabilizerToChildNode =
        orbitsValidInChildNode(currnode.branchingdecision);
    localdom.changeBound(currnode.branchingdecision);
    double nodelb = std::max(currnode.lower_bound, currnode.other_child_lb);
    bool prune = nodelb > getCutoffBound() || localdom.infeasible();
    if (!prune) {
      localdom.propagate();
      prune = localdom.infeasible();
      if (prune) localdom.conflictAnalysis(mipsolver.mipdata_->conflictPool);
    }
    if (!prune) {
      mipsolver.mipdata_->symmetries.propagateOrbitopes(localdom);
      prune = localdom.infeasible();
    }
    if (!prune && passStabilizerToChildNode && currnode.stabilizerOrbits) {
      currnode.stabilizerOrbits->orbitalFixing(localdom);
      prune = localdom.infeasible();
    }
    if (prune) {
      localdom.backtrack();
      localdom.clearChangedCols(numChangedCols);
      if (countTreeWeight) treeweight += std::ldexp(1.0, -getCurrentDepth());
      continue;
    }
    nodestack.emplace_back(
        nodelb, currnode.estimate, currnode.nodeBasis,
        passStabilizerToChildNode ? currnode.stabilizerOrbits : nullptr);

    lp->flushDomain(localdom);
    nodestack.back().domgchgStackPos = domchgPos;
    break;
  }

  if (recoverBasis && nodestack.back().nodeBasis) {
    lp->setStoredBasis(nodestack.back().nodeBasis);
    lp->recoverBasis();
  }

  return true;
}

bool HighsSearch::backtrackPlunge(HighsNodeQueue& nodequeue) {
  const std::vector<HighsDomainChange>& domchgstack =
      localdom.getDomainChangeStack();

  if (nodestack.empty()) return false;
  assert(!nodestack.empty());
  assert(nodestack.back().opensubtrees == 0);

  while (true) {
    while (nodestack.back().opensubtrees == 0) {
      countTreeWeight = true;
      depthoffset += nodestack.back().skipDepthCount;

      if (nodestack.size() == 1) {
        if (nodestack.back().nodeBasis)
          lp->setStoredBasis(std::move(nodestack.back().nodeBasis));
        nodestack.pop_back();
        localdom.backtrackToGlobal();
        lp->flushDomain(localdom);
        lp->recoverBasis();
        return false;
      }

      nodestack.pop_back();
#ifndef NDEBUG
      HighsDomainChange branchchg =
#endif
          localdom.backtrack();

      if (nodestack.back().opensubtrees != 0) {
        countTreeWeight = nodestack.back().skipDepthCount == 0;
        // repropagate the node, as it may have become infeasible due to
        // conflicts
        HighsInt oldNumDomchgs = localdom.getNumDomainChanges();
        HighsInt oldNumChangedCols = localdom.getChangedCols().size();
        localdom.propagate();
        if (!localdom.infeasible() &&
            oldNumDomchgs != localdom.getNumDomainChanges()) {
          if (nodestack.back().stabilizerOrbits)
            nodestack.back().stabilizerOrbits->orbitalFixing(localdom);
          else
            mipsolver.mipdata_->symmetries.propagateOrbitopes(localdom);
        }
        if (localdom.infeasible()) {
          localdom.clearChangedCols(oldNumChangedCols);
          if (countTreeWeight)
            treeweight += std::ldexp(1.0, -getCurrentDepth());
          nodestack.back().opensubtrees = 0;
        }
      }

      assert(
          (branchchg.boundtype == HighsBoundType::kLower &&
           branchchg.boundval >= nodestack.back().branchingdecision.boundval) ||
          (branchchg.boundtype == HighsBoundType::kUpper &&
           branchchg.boundval <= nodestack.back().branchingdecision.boundval));
      assert(branchchg.boundtype ==
             nodestack.back().branchingdecision.boundtype);
      assert(branchchg.column == nodestack.back().branchingdecision.column);
    }

    NodeData& currnode = nodestack.back();

    assert(currnode.opensubtrees == 1);
    currnode.opensubtrees = 0;
    bool fallbackbranch =
        currnode.branchingdecision.boundval == currnode.branching_point;
    double nodeScore;
    if (currnode.branchingdecision.boundtype == HighsBoundType::kLower) {
      currnode.branchingdecision.boundtype = HighsBoundType::kUpper;
      currnode.branchingdecision.boundval =
          std::floor(currnode.branchingdecision.boundval - 0.5);
      nodeScore = pseudocost.getScoreDown(
          currnode.branchingdecision.column,
          fallbackbranch ? 0.5 : currnode.branching_point);
    } else {
      currnode.branchingdecision.boundtype = HighsBoundType::kLower;
      currnode.branchingdecision.boundval =
          std::ceil(currnode.branchingdecision.boundval + 0.5);
      nodeScore = pseudocost.getScoreUp(
          currnode.branchingdecision.column,
          fallbackbranch ? 0.5 : currnode.branching_point);
    }

    if (fallbackbranch)
      currnode.branching_point = currnode.branchingdecision.boundval;

    HighsInt domchgPos = domchgstack.size();
    HighsInt numChangedCols = localdom.getChangedCols().size();
    bool passStabilizerToChildNode =
        orbitsValidInChildNode(currnode.branchingdecision);
    localdom.changeBound(currnode.branchingdecision);
    double nodelb = std::max(currnode.lower_bound, currnode.other_child_lb);
    bool prune = nodelb > getCutoffBound() || localdom.infeasible();
    if (!prune) {
      localdom.propagate();
      prune = localdom.infeasible();
      if (prune) localdom.conflictAnalysis(mipsolver.mipdata_->conflictPool);
    }
    if (!prune) {
      mipsolver.mipdata_->symmetries.propagateOrbitopes(localdom);
      prune = localdom.infeasible();
    }
    if (!prune && passStabilizerToChildNode && currnode.stabilizerOrbits) {
      currnode.stabilizerOrbits->orbitalFixing(localdom);
      prune = localdom.infeasible();
    }
    if (prune) {
      localdom.backtrack();
      localdom.clearChangedCols(numChangedCols);
      if (countTreeWeight) treeweight += std::ldexp(1.0, -getCurrentDepth());
      continue;
    }

    nodelb = std::max(nodelb, localdom.getObjectiveLowerBound());
    bool nodeToQueue = nodelb > mipsolver.mipdata_->optimality_limit;
    // we check if switching to the other branch of an anchestor yields a higher
    // additive branch score than staying in this node and if so we postpone the
    // node and put it to the queue to backtrack further.
    if (!nodeToQueue) {
      for (HighsInt i = nodestack.size() - 2; i >= 0; --i) {
        if (nodestack[i].opensubtrees == 0) continue;

        bool fallbackbranch = nodestack[i].branchingdecision.boundval ==
                              nodestack[i].branching_point;
        double branchpoint =
            fallbackbranch ? 0.5 : nodestack[i].branching_point;
        double ancestorScoreActive;
        double ancestorScoreInactive;
        if (nodestack[i].branchingdecision.boundtype ==
            HighsBoundType::kLower) {
          ancestorScoreInactive = pseudocost.getScoreDown(
              nodestack[i].branchingdecision.column, branchpoint);
          ancestorScoreActive = pseudocost.getScoreUp(
              nodestack[i].branchingdecision.column, branchpoint);
        } else {
          ancestorScoreActive = pseudocost.getScoreDown(
              nodestack[i].branchingdecision.column, branchpoint);
          ancestorScoreInactive = pseudocost.getScoreUp(
              nodestack[i].branchingdecision.column, branchpoint);
        }

        // if (!mipsolver.submip)
        //   printf("nodeScore: %g, ancestorScore: %g\n", nodeScore,
        //   ancestorScore);
        nodeToQueue = ancestorScoreInactive - ancestorScoreActive >
                      nodeScore + mipsolver.mipdata_->feastol;
        break;
      }
    }

    if (nodeToQueue) {
      // if (!mipsolver.submip) printf("node goes to queue\n");
      std::vector<HighsInt> branchPositions;
      auto domchgStack = localdom.getReducedDomainChangeStack(branchPositions);
      double tmpTreeWeight = nodequeue.emplaceNode(
          std::move(domchgStack), std::move(branchPositions), nodelb,
          nodestack.back().estimate, getCurrentDepth() + 1);
      if (countTreeWeight) treeweight += tmpTreeWeight;
      localdom.backtrack();
      localdom.clearChangedCols(numChangedCols);
      continue;
    }
    nodestack.emplace_back(
        nodelb, currnode.estimate, currnode.nodeBasis,
        passStabilizerToChildNode ? currnode.stabilizerOrbits : nullptr);

    lp->flushDomain(localdom);
    nodestack.back().domgchgStackPos = domchgPos;
    break;
  }

  if (nodestack.back().nodeBasis) {
    lp->setStoredBasis(nodestack.back().nodeBasis);
    lp->recoverBasis();
  }

  return true;
}

bool HighsSearch::backtrackUntilDepth(HighsInt targetDepth) {
  if (nodestack.empty()) return false;
  assert(!nodestack.empty());
  if (getCurrentDepth() >= targetDepth) nodestack.back().opensubtrees = 0;

  while (nodestack.back().opensubtrees == 0) {
    depthoffset += nodestack.back().skipDepthCount;
    nodestack.pop_back();

#ifndef NDEBUG
    HighsDomainChange branchchg =
#endif
        localdom.backtrack();
    if (nodestack.empty()) {
      lp->flushDomain(localdom);
      return false;
    }
    assert(
        (branchchg.boundtype == HighsBoundType::kLower &&
         branchchg.boundval >= nodestack.back().branchingdecision.boundval) ||
        (branchchg.boundtype == HighsBoundType::kUpper &&
         branchchg.boundval <= nodestack.back().branchingdecision.boundval));
    assert(branchchg.boundtype == nodestack.back().branchingdecision.boundtype);
    assert(branchchg.column == nodestack.back().branchingdecision.column);

    if (getCurrentDepth() >= targetDepth) nodestack.back().opensubtrees = 0;
  }

  NodeData& currnode = nodestack.back();
  assert(currnode.opensubtrees == 1);
  currnode.opensubtrees = 0;
  bool fallbackbranch =
      currnode.branchingdecision.boundval == currnode.branching_point;
  if (currnode.branchingdecision.boundtype == HighsBoundType::kLower) {
    currnode.branchingdecision.boundtype = HighsBoundType::kUpper;
    currnode.branchingdecision.boundval =
        std::floor(currnode.branchingdecision.boundval - 0.5);
  } else {
    currnode.branchingdecision.boundtype = HighsBoundType::kLower;
    currnode.branchingdecision.boundval =
        std::ceil(currnode.branchingdecision.boundval + 0.5);
  }

  if (fallbackbranch)
    currnode.branching_point = currnode.branchingdecision.boundval;

  HighsInt domchgPos = localdom.getDomainChangeStack().size();
  bool passStabilizerToChildNode =
      orbitsValidInChildNode(currnode.branchingdecision);
  localdom.changeBound(currnode.branchingdecision);
  nodestack.emplace_back(
      currnode.lower_bound, currnode.estimate, currnode.nodeBasis,
      passStabilizerToChildNode ? currnode.stabilizerOrbits : nullptr);

  lp->flushDomain(localdom);
  nodestack.back().domgchgStackPos = domchgPos;
  if (nodestack.back().nodeBasis &&
      nodestack.back().nodeBasis->row_status.size() == lp->getLp().num_row_)
    lp->setStoredBasis(nodestack.back().nodeBasis);
  lp->recoverBasis();

  return true;
}

HighsSearch::NodeResult HighsSearch::dive() {
  reliableatnode.clear();

  do {
    ++nnodes;
    NodeResult result = evaluateNode();

    if (mipsolver.mipdata_->checkLimits(nnodes)) return result;

    if (result != NodeResult::kOpen) return result;

    result = branch();
    if (result != NodeResult::kBranched) return result;
  } while (true);
}

void HighsSearch::solveDepthFirst(int64_t maxbacktracks) {
  do {
    if (maxbacktracks == 0) break;

    NodeResult result = dive();
    // if a limit was reached the result might be open
    if (result == NodeResult::kOpen) break;

    --maxbacktracks;

  } while (backtrack());
}
