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
#include "mip/HighsImplications.h"

#include "mip/HighsCliqueTable.h"
#include "mip/HighsMipSolverData.h"
#include "pdqsort/pdqsort.h"

bool HighsImplications::computeImplications(HighsInt col, bool val) {
  HighsDomain& globaldomain = mipsolver.mipdata_->domain;
  HighsCliqueTable& cliquetable = mipsolver.mipdata_->cliquetable;
  globaldomain.propagate();
  if (globaldomain.infeasible() || globaldomain.isFixed(col)) return true;
  const auto& domchgstack = globaldomain.getDomainChangeStack();
  const auto& domchgreason = globaldomain.getDomainChangeReason();
  HighsInt changedend = globaldomain.getChangedCols().size();

  HighsInt stackimplicstart = domchgstack.size() + 1;
  HighsInt numImplications = -stackimplicstart;
  if (val)
    globaldomain.changeBound(HighsBoundType::kLower, col, 1);
  else
    globaldomain.changeBound(HighsBoundType::kUpper, col, 0);

  if (globaldomain.infeasible()) {
    globaldomain.backtrack();
    globaldomain.clearChangedCols(changedend);
    cliquetable.vertexInfeasible(globaldomain, col, val);

    return true;
  }

  globaldomain.propagate();

  if (globaldomain.infeasible()) {
    globaldomain.backtrack();
    globaldomain.clearChangedCols(changedend);

    cliquetable.vertexInfeasible(globaldomain, col, val);

    return true;
  }

  HighsInt stackimplicend = domchgstack.size();
  numImplications += stackimplicend;
  mipsolver.mipdata_->pseudocost.addInferenceObservation(col, numImplications,
                                                         val);

  std::vector<HighsDomainChange> implics;
  implics.reserve(numImplications);

  HighsInt numEntries = mipsolver.mipdata_->cliquetable.getNumEntries();
  HighsInt maxEntries = 100000 + mipsolver.numNonzero();

  for (HighsInt i = stackimplicstart; i < stackimplicend; ++i) {
    if (domchgreason[i].type == HighsDomain::Reason::kCliqueTable &&
        ((domchgreason[i].index >> 1) == col || numEntries >= maxEntries))
      continue;

    implics.push_back(domchgstack[i]);
  }

  globaldomain.backtrack();
  globaldomain.clearChangedCols(changedend);

  // add the implications of binary variables to the clique table
  auto binstart = std::partition(implics.begin(), implics.end(),
                                 [&](const HighsDomainChange& a) {
                                   return !globaldomain.isBinary(a.column);
                                 });

  pdqsort(implics.begin(), binstart);

  HighsCliqueTable::CliqueVar clique[2];
  clique[0] = HighsCliqueTable::CliqueVar(col, val);

  for (auto i = binstart; i != implics.end(); ++i) {
    if (i->boundtype == HighsBoundType::kLower)
      clique[1] = HighsCliqueTable::CliqueVar(i->column, 0);
    else
      clique[1] = HighsCliqueTable::CliqueVar(i->column, 1);

    cliquetable.addClique(mipsolver, clique, 2);
    if (globaldomain.infeasible() || globaldomain.isFixed(col)) return true;
  }

  // store variable bounds derived from implications
  for (auto i = implics.begin(); i != binstart; ++i) {
    if (i->boundtype == HighsBoundType::kLower) {
      if (val == 1) {
        if (globaldomain.col_lower_[i->column] != -kHighsInf)
          addVLB(i->column, col,
                 i->boundval - globaldomain.col_lower_[i->column],
                 globaldomain.col_lower_[i->column]);
      } else
        addVLB(i->column,
               col,  // in case the lower bound is infinite the varbound can
                     // still be tightened as soon as a finite upper bound is
                     // known because the offset is finite
               globaldomain.col_lower_[i->column] - i->boundval, i->boundval);
    } else {
      if (val == 1) {
        if (globaldomain.col_upper_[i->column] != kHighsInf)
          addVUB(i->column, col,
                 i->boundval - globaldomain.col_upper_[i->column],
                 globaldomain.col_upper_[i->column]);
      } else
        addVUB(i->column,
               col,  // in case the upper bound is infinite the varbound can
                     // still be tightened as soon as a finite upper bound is
                     // known because the offset is finite
               globaldomain.col_upper_[i->column] - i->boundval, i->boundval);
    }
  }

  HighsInt loc = 2 * col + val;
  implications[loc].computed = true;
  implics.erase(binstart, implics.end());
  if (!implics.empty()) {
    implications[loc].implics = std::move(implics);
    this->numImplications += implications[loc].implics.size();
  }

  return false;
}

bool HighsImplications::runProbing(HighsInt col, HighsInt& numReductions) {
  HighsDomain& globaldomain = mipsolver.mipdata_->domain;
  if (globaldomain.isBinary(col) && !implicationsCached(col, 1) &&
      !implicationsCached(col, 0) &&
      mipsolver.mipdata_->cliquetable.getSubstitution(col) == nullptr) {
    bool infeasible;

    infeasible = computeImplications(col, 1);
    if (globaldomain.infeasible()) return true;
    if (infeasible) return true;
    if (mipsolver.mipdata_->cliquetable.getSubstitution(col) != nullptr)
      return true;

    infeasible = computeImplications(col, 0);
    if (globaldomain.infeasible()) return true;
    if (infeasible) return true;
    if (mipsolver.mipdata_->cliquetable.getSubstitution(col) != nullptr)
      return true;

    // analyze implications
    const std::vector<HighsDomainChange>& implicsdown =
        getImplications(col, 0, infeasible);
    const std::vector<HighsDomainChange>& implicsup =
        getImplications(col, 1, infeasible);
    HighsInt nimplicsdown = implicsdown.size();
    HighsInt nimplicsup = implicsup.size();
    HighsInt u = 0;
    HighsInt d = 0;

    while (u < nimplicsup && d < nimplicsdown) {
      if (implicsup[u].column < implicsdown[d].column)
        ++u;
      else if (implicsdown[d].column < implicsup[u].column)
        ++d;
      else {
        assert(implicsup[u].column == implicsdown[d].column);
        HighsInt implcol = implicsup[u].column;
        double lbDown = globaldomain.col_lower_[implcol];
        double ubDown = globaldomain.col_upper_[implcol];
        double lbUp = lbDown;
        double ubUp = ubDown;

        do {
          if (implicsdown[d].boundtype == HighsBoundType::kLower)
            lbDown = std::max(lbDown, implicsdown[d].boundval);
          else
            ubDown = std::min(ubDown, implicsdown[d].boundval);
          ++d;
        } while (d < nimplicsdown && implicsdown[d].column == implcol);

        do {
          if (implicsup[u].boundtype == HighsBoundType::kLower)
            lbUp = std::max(lbUp, implicsup[u].boundval);
          else
            ubUp = std::min(ubUp, implicsup[u].boundval);
          ++u;
        } while (u < nimplicsup && implicsup[u].column == implcol);

        if (colsubstituted[implcol] || globaldomain.isFixed(implcol)) continue;

        if (lbDown == ubDown && lbUp == ubUp &&
            std::abs(lbDown - lbUp) > mipsolver.mipdata_->feastol) {
          HighsSubstitution substitution;
          substitution.substcol = implcol;
          substitution.staycol = col;
          substitution.offset = lbDown;
          substitution.scale = lbUp - lbDown;
          substitutions.push_back(substitution);
          colsubstituted[implcol] = true;
          ++numReductions;
        } else {
          double lb = std::min(lbDown, lbUp);
          double ub = std::max(ubDown, ubUp);

          if (lb > globaldomain.col_lower_[implcol]) {
            globaldomain.changeBound(HighsBoundType::kLower, implcol, lb,
                                     HighsDomain::Reason::unspecified());
            ++numReductions;
          }

          if (ub < globaldomain.col_upper_[implcol]) {
            globaldomain.changeBound(HighsBoundType::kUpper, implcol, ub,
                                     HighsDomain::Reason::unspecified());
            ++numReductions;
          }
        }
      }
    }

    return true;
  }

  return false;
}

void HighsImplications::addVUB(HighsInt col, HighsInt vubcol, double vubcoef,
                               double vubconstant) {
  VarBound vub{vubcoef, vubconstant};

  mipsolver.mipdata_->debugSolution.checkVub(col, vubcol, vubcoef, vubconstant);

  double minBound = vub.minValue();
  if (minBound >=
      mipsolver.mipdata_->domain.col_upper_[col] - mipsolver.mipdata_->feastol)
    return;

  auto insertresult = vubs[col].emplace(vubcol, vub);

  if (!insertresult.second) {
    VarBound& currentvub = insertresult.first->second;
    double currentMinBound = currentvub.minValue();
    if (minBound < currentMinBound - mipsolver.mipdata_->feastol) {
      currentvub.coef = vubcoef;
      currentvub.constant = vubconstant;
    }
  }
}

void HighsImplications::addVLB(HighsInt col, HighsInt vlbcol, double vlbcoef,
                               double vlbconstant) {
  VarBound vlb{vlbcoef, vlbconstant};

  mipsolver.mipdata_->debugSolution.checkVlb(col, vlbcol, vlbcoef, vlbconstant);

  double maxBound = vlb.maxValue();
  if (vlb.maxValue() <=
      mipsolver.mipdata_->domain.col_lower_[col] + mipsolver.mipdata_->feastol)
    return;

  auto insertresult = vlbs[col].emplace(vlbcol, vlb);

  if (!insertresult.second) {
    VarBound& currentvlb = insertresult.first->second;

    double currentMaxNound = currentvlb.maxValue();
    if (maxBound > currentMaxNound + mipsolver.mipdata_->feastol) {
      currentvlb.coef = vlbcoef;
      currentvlb.constant = vlbconstant;
    }
  }
}

void HighsImplications::rebuild(HighsInt ncols,
                                const std::vector<HighsInt>& orig2reducedcol,
                                const std::vector<HighsInt>& orig2reducedrow) {
  std::vector<std::map<HighsInt, VarBound>> oldvubs;
  std::vector<std::map<HighsInt, VarBound>> oldvlbs;

  oldvlbs.swap(vlbs);
  oldvubs.swap(vubs);

  colsubstituted.clear();
  colsubstituted.shrink_to_fit();
  implications.clear();
  implications.shrink_to_fit();

  implications.resize(2 * ncols);
  colsubstituted.resize(ncols);
  substitutions.clear();
  vubs.clear();
  vubs.shrink_to_fit();
  vubs.resize(ncols);
  vlbs.clear();
  vlbs.shrink_to_fit();
  vlbs.resize(ncols);
  numImplications = 0;
  HighsInt oldncols = oldvubs.size();

  nextCleanupCall = mipsolver.numNonzero();

  for (HighsInt i = 0; i != oldncols; ++i) {
    HighsInt newi = orig2reducedcol[i];

    if (newi == -1 ||
        !mipsolver.mipdata_->postSolveStack.isColLinearlyTransformable(newi))
      continue;

    for (const auto& oldvub : oldvubs[i]) {
      HighsInt newVubCol = orig2reducedcol[oldvub.first];
      if (newVubCol == -1) continue;

      if (!mipsolver.mipdata_->domain.isBinary(newVubCol) ||
          !mipsolver.mipdata_->postSolveStack.isColLinearlyTransformable(
              newVubCol))
        continue;

      addVUB(newi, newVubCol, oldvub.second.coef, oldvub.second.constant);
    }

    for (const auto& oldvlb : oldvlbs[i]) {
      HighsInt newVlbCol = orig2reducedcol[oldvlb.first];
      if (newVlbCol == -1) continue;

      if (!mipsolver.mipdata_->domain.isBinary(newVlbCol) ||
          !mipsolver.mipdata_->postSolveStack.isColLinearlyTransformable(
              newVlbCol))
        continue;

      addVLB(newi, newVlbCol, oldvlb.second.coef, oldvlb.second.constant);
    }

    // todo also add old implications once implications can be added
    // incrementally for now we discard the old implications as they might be
    // weaker then newly computed ones and adding them would block computation
    // of new implications
  }
}

void HighsImplications::buildFrom(const HighsImplications& init) {
  return;
#if 0
  // todo check if this should be done
  HighsInt numcol = mipsolver.numCol();

  for (HighsInt i = 0; i != numcol; ++i) {
    for (const auto& vub : init.vubs[i]) {
      if (!mipsolver.mipdata_->domain.isBinary(vub.first)) continue;
      addVUB(i, vub.first, vub.second.coef, vub.second.constant);
    }

    for (const auto& vlb : init.vlbs[i]) {
      if (!mipsolver.mipdata_->domain.isBinary(vlb.first)) continue;
      addVLB(i, vlb.first, vlb.second.coef, vlb.second.constant);
    }

    // todo also add old implications once implications can be added
    // incrementally for now we discard the old implications as they might be
    // weaker then newly computed ones and adding them would block computation
    // of new implications
  }
#endif
}

void HighsImplications::separateImpliedBounds(
    const HighsLpRelaxation& lpRelaxation, const std::vector<double>& sol,
    HighsCutPool& cutpool, double feastol) {
  HighsDomain& globaldomain = mipsolver.mipdata_->domain;

  HighsInt inds[2];
  double vals[2];
  double rhs;

  HighsInt numboundchgs = 0;

  // first do probing on all candidates that have not been probed yet
  if (!mipsolver.mipdata_->cliquetable.isFull()) {
    auto oldNumQueries = mipsolver.mipdata_->cliquetable.numNeighborhoodQueries;
    HighsInt oldNumEntries = mipsolver.mipdata_->cliquetable.getNumEntries();

    for (std::pair<HighsInt, double> fracint :
         lpRelaxation.getFractionalIntegers()) {
      HighsInt col = fracint.first;
      if (globaldomain.col_lower_[col] != 0.0 ||
          globaldomain.col_upper_[col] != 1.0 ||
          (implicationsCached(col, 0) && implicationsCached(col, 1)))
        continue;

      if (runProbing(col, numboundchgs)) {
        if (globaldomain.infeasible()) return;
      }

      if (mipsolver.mipdata_->cliquetable.isFull()) break;
    }

    // if (!mipsolver.submip)
    //   printf("numEntries: %d, beforeProbing: %d\n",
    //          mipsolver.mipdata_->cliquetable.getNumEntries(), oldNumEntries);
    HighsInt numNewEntries =
        mipsolver.mipdata_->cliquetable.getNumEntries() - oldNumEntries;

    nextCleanupCall -= std::max(HighsInt{0}, numNewEntries);

    if (nextCleanupCall < 0) {
      HighsInt oldNumEntries = mipsolver.mipdata_->cliquetable.getNumEntries();
      mipsolver.mipdata_->cliquetable.runCliqueMerging(globaldomain);
      // printf("numEntries: %d, beforeMerging: %d\n",
      //        mipsolver.mipdata_->cliquetable.getNumEntries(), oldNumEntries);
      nextCleanupCall =
          std::min(mipsolver.mipdata_->numCliqueEntriesAfterFirstPresolve,
                   mipsolver.mipdata_->cliquetable.getNumEntries());
      // printf("nextCleanupCall: %d\n", nextCleanupCall);
    }

    mipsolver.mipdata_->cliquetable.numNeighborhoodQueries = oldNumQueries;
  }

  for (std::pair<HighsInt, double> fracint :
       lpRelaxation.getFractionalIntegers()) {
    HighsInt col = fracint.first;
    // skip non binary variables
    if (globaldomain.col_lower_[col] != 0.0 ||
        globaldomain.col_upper_[col] != 1.0)
      continue;

    bool infeas;
    if (implicationsCached(col, 1)) {
      const std::vector<HighsDomainChange>& implics =
          getImplications(col, 1, infeas);
      if (globaldomain.infeasible()) return;

      if (infeas) {
        vals[0] = 1.0;
        inds[0] = col;
        cutpool.addCut(mipsolver, inds, vals, 1, 0.0, false, true, false);
        continue;
      }

      HighsInt nimplics = implics.size();
      for (HighsInt i = 0; i < nimplics; ++i) {
        if (implics[i].boundtype == HighsBoundType::kUpper) {
          if (implics[i].boundval + feastol >=
              globaldomain.col_upper_[implics[i].column])
            continue;

          vals[0] = 1.0;
          inds[0] = implics[i].column;
          vals[1] =
              globaldomain.col_upper_[implics[i].column] - implics[i].boundval;
          inds[1] = col;
          rhs = globaldomain.col_upper_[implics[i].column];

        } else {
          if (implics[i].boundval - feastol <=
              globaldomain.col_lower_[implics[i].column])
            continue;

          vals[0] = -1.0;
          inds[0] = implics[i].column;
          vals[1] =
              globaldomain.col_lower_[implics[i].column] - implics[i].boundval;
          inds[1] = col;
          rhs = -globaldomain.col_lower_[implics[i].column];
        }

        double viol = sol[inds[0]] * vals[0] + sol[inds[1]] * vals[1] - rhs;

        if (viol > feastol) {
          // printf("added implied bound cut to pool\n");
          cutpool.addCut(mipsolver, inds, vals, 2, rhs,
                         mipsolver.variableType(implics[i].column) !=
                             HighsVarType::kContinuous,
                         false, false, false);
        }
      }
    }

    if (implicationsCached(col, 0)) {
      const std::vector<HighsDomainChange>& implics =
          getImplications(col, 0, infeas);
      if (globaldomain.infeasible()) return;

      if (infeas) {
        vals[0] = -1.0;
        inds[0] = col;
        cutpool.addCut(mipsolver, inds, vals, 1, -1.0, false, true, false);
        continue;
      }

      HighsInt nimplics = implics.size();
      for (HighsInt i = 0; i < nimplics; ++i) {
        if (implics[i].boundtype == HighsBoundType::kUpper) {
          if (implics[i].boundval + feastol >=
              globaldomain.col_upper_[implics[i].column])
            continue;

          vals[0] = 1.0;
          inds[0] = implics[i].column;
          vals[1] =
              implics[i].boundval - globaldomain.col_upper_[implics[i].column];
          inds[1] = col;
          rhs = implics[i].boundval;
        } else {
          if (implics[i].boundval - feastol <=
              globaldomain.col_lower_[implics[i].column])
            continue;

          vals[0] = -1.0;
          inds[0] = implics[i].column;
          vals[1] =
              globaldomain.col_lower_[implics[i].column] - implics[i].boundval;
          inds[1] = col;
          rhs = -implics[i].boundval;
        }

        double viol = sol[inds[0]] * vals[0] + sol[inds[1]] * vals[1] - rhs;

        if (viol > feastol) {
          // printf("added implied bound cut to pool\n");
          cutpool.addCut(mipsolver, inds, vals, 2, rhs,
                         mipsolver.variableType(implics[i].column) !=
                             HighsVarType::kContinuous,
                         false, false, false);
        }
      }
    }
  }
}

void HighsImplications::cleanupVarbounds(HighsInt col) {
  double ub = mipsolver.mipdata_->domain.col_upper_[col];
  double lb = mipsolver.mipdata_->domain.col_lower_[col];

  if (ub == lb) {
    vlbs[col].clear();
    vubs[col].clear();
    return;
  }

  auto next = vubs[col].begin();
  while (next != vubs[col].end()) {
    auto it = next++;

    mipsolver.mipdata_->debugSolution.checkVub(col, it->first, it->second.coef,
                                               it->second.constant);

    if (it->second.coef > 0) {
      double minub = it->second.constant;
      double maxub = it->second.constant + it->second.coef;
      if (minub >= ub - mipsolver.mipdata_->feastol)
        vubs[col].erase(it);  // variable bound is redundant
      else if (maxub > ub + mipsolver.mipdata_->epsilon) {
        it->second.coef =
            ub - it->second.constant;  // coefficient can be tightened
        mipsolver.mipdata_->debugSolution.checkVub(
            col, it->first, it->second.coef, it->second.constant);
      } else if (maxub < ub - mipsolver.mipdata_->epsilon) {
        mipsolver.mipdata_->domain.changeBound(
            HighsBoundType::kUpper, col, maxub,
            HighsDomain::Reason::unspecified());
        if (mipsolver.mipdata_->domain.infeasible()) return;
      }
    } else {
      HighsCDouble minub = HighsCDouble(it->second.constant) + it->second.coef;
      double maxub = it->second.constant;
      if (minub >= ub - mipsolver.mipdata_->feastol)
        vubs[col].erase(it);  // variable bound is redundant
      else if (maxub > ub + mipsolver.mipdata_->epsilon) {
        // variable bound can be tightened
        it->second.constant = ub;
        it->second.coef = double(minub - ub);
        mipsolver.mipdata_->debugSolution.checkVub(
            col, it->first, it->second.coef, it->second.constant);
      } else if (maxub < ub - mipsolver.mipdata_->epsilon) {
        mipsolver.mipdata_->domain.changeBound(
            HighsBoundType::kUpper, col, maxub,
            HighsDomain::Reason::unspecified());
        if (mipsolver.mipdata_->domain.infeasible()) return;
      }
    }
  }

  next = vlbs[col].begin();
  while (next != vlbs[col].end()) {
    auto it = next++;

    mipsolver.mipdata_->debugSolution.checkVlb(col, it->first, it->second.coef,
                                               it->second.constant);

    if (it->second.coef > 0) {
      HighsCDouble maxlb = HighsCDouble(it->second.constant) + it->second.coef;
      double minlb = it->second.constant;
      if (maxlb <= lb + mipsolver.mipdata_->feastol)
        vlbs[col].erase(it);  // variable bound is redundant
      else if (minlb < lb - mipsolver.mipdata_->epsilon) {
        // variable bound can be tightened
        it->second.constant = lb;
        it->second.coef = double(maxlb - lb);
        mipsolver.mipdata_->debugSolution.checkVlb(
            col, it->first, it->second.coef, it->second.constant);
      } else if (minlb > lb + mipsolver.mipdata_->epsilon) {
        mipsolver.mipdata_->domain.changeBound(
            HighsBoundType::kLower, col, minlb,
            HighsDomain::Reason::unspecified());
        if (mipsolver.mipdata_->domain.infeasible()) return;
      }

    } else {
      double maxlb = it->second.constant;
      double minlb = it->second.constant + it->second.coef;
      if (maxlb <= lb + mipsolver.mipdata_->feastol)
        vlbs[col].erase(it);  // variable bound is redundant
      else if (minlb < lb - mipsolver.mipdata_->epsilon) {
        it->second.coef =
            lb - it->second.constant;  // variable bound can be tightened
        mipsolver.mipdata_->debugSolution.checkVlb(
            col, it->first, it->second.coef, it->second.constant);
      } else if (minlb > lb + mipsolver.mipdata_->epsilon) {
        mipsolver.mipdata_->domain.changeBound(
            HighsBoundType::kLower, col, minlb,
            HighsDomain::Reason::unspecified());
        if (mipsolver.mipdata_->domain.infeasible()) return;
      }
    }
  }
}
