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
#include "mip/HighsSeparation.h"

#include <algorithm>
#include <cassert>
#include <queue>

#include "mip/HighsCliqueTable.h"
#include "mip/HighsDomain.h"
#include "mip/HighsImplications.h"
#include "mip/HighsLpAggregator.h"
#include "mip/HighsLpRelaxation.h"
#include "mip/HighsMipSolverData.h"
#include "mip/HighsModkSeparator.h"
#include "mip/HighsPathSeparator.h"
#include "mip/HighsTableauSeparator.h"
#include "mip/HighsTransformedLp.h"

HighsSeparation::HighsSeparation(const HighsMipSolver& mipsolver) {
  implBoundClock = mipsolver.timer_.clock_def("Implbound sepa", "Ibd");
  cliqueClock = mipsolver.timer_.clock_def("Clique sepa", "Clq");
  separators.emplace_back(new HighsTableauSeparator(mipsolver));
  separators.emplace_back(new HighsPathSeparator(mipsolver));
  separators.emplace_back(new HighsModkSeparator(mipsolver));
}

HighsInt HighsSeparation::separationRound(HighsDomain& propdomain,
                                          HighsLpRelaxation::Status& status) {
  const HighsSolution& sol = lp->getLpSolver().getSolution();

  HighsMipSolverData& mipdata = *lp->getMipSolver().mipdata_;

  auto propagateAndResolve = [&]() {
    if (propdomain.infeasible() || mipdata.domain.infeasible()) {
      status = HighsLpRelaxation::Status::kInfeasible;
      propdomain.clearChangedCols();
      return -1;
    }

    propdomain.propagate();
    if (propdomain.infeasible()) {
      status = HighsLpRelaxation::Status::kInfeasible;
      propdomain.clearChangedCols();
      return -1;
    }

    mipdata.cliquetable.cleanupFixed(mipdata.domain);
    if (mipdata.domain.infeasible()) {
      status = HighsLpRelaxation::Status::kInfeasible;
      propdomain.clearChangedCols();
      return -1;
    }

    int numBoundChgs = (int)propdomain.getChangedCols().size();

    while (!propdomain.getChangedCols().empty()) {
      lp->setObjectiveLimit(mipdata.upper_limit);
      status = lp->resolveLp(&propdomain);
      if (!lp->scaledOptimal(status)) return -1;

      if (&propdomain == &mipdata.domain && lp->unscaledDualFeasible(status)) {
        mipdata.redcostfixing.addRootRedcost(
            mipdata.mipsolver, lp->getSolution().col_dual, lp->getObjective());
        if (mipdata.upper_limit != kHighsInf)
          mipdata.redcostfixing.propagateRootRedcost(mipdata.mipsolver);
      }
    }

    return numBoundChgs;
  };

  lp->getMipSolver().timer_.start(implBoundClock);
  mipdata.implications.separateImpliedBounds(*lp, lp->getSolution().col_value,
                                             mipdata.cutpool, mipdata.feastol);
  lp->getMipSolver().timer_.stop(implBoundClock);

  HighsInt ncuts = 0;
  HighsInt numboundchgs = propagateAndResolve();
  if (numboundchgs == -1)
    return 0;
  else
    ncuts += numboundchgs;

  lp->getMipSolver().timer_.start(cliqueClock);
  mipdata.cliquetable.separateCliques(lp->getMipSolver(), sol.col_value,
                                      mipdata.cutpool, mipdata.feastol);
  lp->getMipSolver().timer_.stop(cliqueClock);

  numboundchgs = propagateAndResolve();
  if (numboundchgs == -1)
    return 0;
  else
    ncuts += numboundchgs;

  HighsTransformedLp transLp(*lp, mipdata.implications);
  if (mipdata.domain.infeasible()) {
    status = HighsLpRelaxation::Status::kInfeasible;
    return 0;
  }
  HighsLpAggregator lpAggregator(*lp);

  for (const std::unique_ptr<HighsSeparator>& separator : separators) {
    separator->run(*lp, lpAggregator, transLp, mipdata.cutpool);
    if (mipdata.domain.infeasible()) {
      status = HighsLpRelaxation::Status::kInfeasible;
      return 0;
    }
  }

  numboundchgs = propagateAndResolve();
  if (numboundchgs == -1)
    return 0;
  else
    ncuts += numboundchgs;

  mipdata.cutpool.separate(sol.col_value, propdomain, cutset, mipdata.feastol);

  if (cutset.numCuts() > 0) {
    ncuts += cutset.numCuts();
    lp->addCuts(cutset);
    status = lp->resolveLp(&propdomain);
    lp->performAging(true);
    if (&propdomain == &mipdata.domain && lp->unscaledDualFeasible(status)) {
      mipdata.redcostfixing.addRootRedcost(
          mipdata.mipsolver, lp->getSolution().col_dual, lp->getObjective());
      if (mipdata.upper_limit != kHighsInf)
        mipdata.redcostfixing.propagateRootRedcost(mipdata.mipsolver);
    }
  }

  return ncuts;
}

void HighsSeparation::separate(HighsDomain& propdomain) {
  HighsLpRelaxation::Status status = lp->getStatus();
  const HighsMipSolver& mipsolver = lp->getMipSolver();

  if (lp->scaledOptimal(status) && !lp->getFractionalIntegers().empty()) {
    // double firstobj = lp->getObjective();
    double firstobj = mipsolver.mipdata_->rootlpsolobj;

    while (lp->getObjective() < mipsolver.mipdata_->optimality_limit) {
      double lastobj = lp->getObjective();

      size_t nlpiters = -lp->getNumLpIterations();
      HighsInt ncuts = separationRound(propdomain, status);
      nlpiters += lp->getNumLpIterations();
      mipsolver.mipdata_->sepa_lp_iterations += nlpiters;
      mipsolver.mipdata_->total_lp_iterations += nlpiters;
      // printf("separated %" HIGHSINT_FORMAT " cuts\n", ncuts);

      // printf(
      //     "separation round %" HIGHSINT_FORMAT " at node %" HIGHSINT_FORMAT "
      //     added %" HIGHSINT_FORMAT " cuts objective changed " "from %g to %g,
      //     first obj is %g\n", nrounds, (HighsInt)nnodes, ncuts, lastobj,
      //     lp->getObjective(), firstobj);
      if (ncuts == 0 || !lp->scaledOptimal(status) ||
          lp->getFractionalIntegers().empty())
        break;

      // if the objective improved considerably we continue
      if ((lp->getObjective() - firstobj) <=
          std::max((lastobj - firstobj), mipsolver.mipdata_->feastol) * 1.01)
        break;
    }

    // printf("done separating\n");
  } else {
    // printf("no separation, just aging. status: %" HIGHSINT_FORMAT "\n",
    //        (HighsInt)status);
    lp->performAging(true);
    mipsolver.mipdata_->cutpool.performAging();
  }
}
