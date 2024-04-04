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
#include "mip/HighsMipSolver.h"

#include "lp_data/HighsLpUtils.h"
#include "lp_data/HighsModelUtils.h"
#include "mip/HighsCliqueTable.h"
#include "mip/HighsCutPool.h"
#include "mip/HighsDomain.h"
#include "mip/HighsImplications.h"
#include "mip/HighsLpRelaxation.h"
#include "mip/HighsMipSolverData.h"
#include "mip/HighsPseudocost.h"
#include "mip/HighsSearch.h"
#include "mip/HighsSeparation.h"
#include "presolve/HPresolve.h"
#include "presolve/HighsPostsolveStack.h"
#include "presolve/PresolveComponent.h"
#include "util/HighsCDouble.h"
#include "util/HighsIntegers.h"

using std::fabs;

HighsMipSolver::HighsMipSolver(const HighsOptions& options, const HighsLp& lp,
                               const HighsSolution& solution, bool submip)
    : options_mip_(&options),
      model_(&lp),
      orig_model_(&lp),
      solution_objective_(kHighsInf),
      submip(submip),
      rootbasis(nullptr),
      pscostinit(nullptr),
      clqtableinit(nullptr),
      implicinit(nullptr) {
  if (solution.value_valid) {
    bound_violation_ = 0;
    row_violation_ = 0;
    integrality_violation_ = 0;

    HighsCDouble obj = orig_model_->offset_;
    assert((HighsInt)solution.col_value.size() == orig_model_->num_col_);
    for (HighsInt i = 0; i != orig_model_->num_col_; ++i) {
      const double value = solution.col_value[i];
      obj += orig_model_->col_cost_[i] * value;

      if (orig_model_->integrality_[i] == HighsVarType::kInteger) {
        double intval = std::floor(value + 0.5);
        integrality_violation_ =
            std::max(fabs(intval - value), integrality_violation_);
      }

      const double lower = orig_model_->col_lower_[i];
      const double upper = orig_model_->col_upper_[i];
      double primal_infeasibility;
      if (value < lower - options_mip_->mip_feasibility_tolerance) {
        primal_infeasibility = lower - value;
      } else if (value > upper + options_mip_->mip_feasibility_tolerance) {
        primal_infeasibility = value - upper;
      } else
        continue;

      bound_violation_ = std::max(bound_violation_, primal_infeasibility);
    }

    for (HighsInt i = 0; i != orig_model_->num_row_; ++i) {
      const double value = solution.row_value[i];
      const double lower = orig_model_->row_lower_[i];
      const double upper = orig_model_->row_upper_[i];

      double primal_infeasibility;
      if (value < lower - options_mip_->mip_feasibility_tolerance) {
        primal_infeasibility = lower - value;
      } else if (value > upper + options_mip_->mip_feasibility_tolerance) {
        primal_infeasibility = value - upper;
      } else
        continue;

      row_violation_ = std::max(row_violation_, primal_infeasibility);
    }

    solution_objective_ = double(obj);
    solution_ = solution.col_value;
  }
}

HighsMipSolver::~HighsMipSolver() = default;

void HighsMipSolver::run() {
  modelstatus_ = HighsModelStatus::kNotset;
  // std::cout << options_mip_->presolve << std::endl;
  timer_.start(timer_.solve_clock);

  mipdata_ = decltype(mipdata_)(new HighsMipSolverData(*this));
  mipdata_->init();
  mipdata_->runPresolve();
  if (modelstatus_ != HighsModelStatus::kNotset) {
    highsLogUser(options_mip_->log_options, HighsLogType::kInfo,
                 "Presolve: %s\n",
                 utilModelStatusToString(modelstatus_).c_str());
    if (modelstatus_ == HighsModelStatus::kOptimal) {
      mipdata_->lower_bound = 0;
      mipdata_->upper_bound = 0;
      mipdata_->transformNewIncumbent(std::vector<double>());
    }
    cleanupSolve();
    return;
  }

  mipdata_->runSetup();
restart:
  if (modelstatus_ == HighsModelStatus::kNotset) {
    mipdata_->evaluateRootNode();
    // age 5 times to remove stored but never violated cuts after root
    // separation
    mipdata_->cutpool.performAging();
    mipdata_->cutpool.performAging();
    mipdata_->cutpool.performAging();
    mipdata_->cutpool.performAging();
    mipdata_->cutpool.performAging();
  }
  if (mipdata_->nodequeue.empty()) {
    cleanupSolve();
    return;
  }

  std::shared_ptr<const HighsBasis> basis;
  HighsSearch search{*this, mipdata_->pseudocost};
  mipdata_->debugSolution.registerDomain(search.getLocalDomain());
  HighsSeparation sepa(*this);

  search.setLpRelaxation(&mipdata_->lp);
  sepa.setLpRelaxation(&mipdata_->lp);

  mipdata_->lower_bound = mipdata_->nodequeue.getBestLowerBound();

  mipdata_->printDisplayLine();
  search.installNode(mipdata_->nodequeue.popBestBoundNode());
  int64_t numStallNodes = 0;
  int64_t lastLbLeave = 0;
  int64_t numQueueLeaves = 0;
  HighsInt numHugeTreeEstim = 0;
  int64_t numNodesLastCheck = mipdata_->num_nodes;
  int64_t nextCheck = mipdata_->num_nodes;
  double treeweightLastCheck = 0.0;
  double upperLimLastCheck = mipdata_->upper_limit;
  double lowerBoundLastCheck = mipdata_->lower_bound;
  while (search.hasNode()) {
    mipdata_->conflictPool.performAging();
    // set iteration limit for each lp solve during the dive to 10 times the
    // average nodes

    HighsInt iterlimit = 10 * std::max(mipdata_->lp.getAvgSolveIters(),
                                       mipdata_->avgrootlpiters);
    iterlimit = std::max({HighsInt{10000}, iterlimit,
                          HighsInt(1.5 * mipdata_->firstrootlpiters)});

    mipdata_->lp.setIterationLimit(iterlimit);

    // perform the dive and put the open nodes to the queue
    size_t plungestart = mipdata_->num_nodes;
    bool limit_reached = false;
    bool considerHeuristics = true;
    while (true) {
      if (considerHeuristics && mipdata_->moreHeuristicsAllowed()) {
        if (search.evaluateNode() == HighsSearch::NodeResult::kSubOptimal)
          break;

        if (search.currentNodePruned()) {
          ++mipdata_->num_leaves;
          search.flushStatistics();
        } else {
          if (mipdata_->incumbent.empty())
            mipdata_->heuristics.randomizedRounding(
                mipdata_->lp.getLpSolver().getSolution().col_value);

          if (mipdata_->incumbent.empty())
            mipdata_->heuristics.RENS(
                mipdata_->lp.getLpSolver().getSolution().col_value);
          else
            mipdata_->heuristics.RINS(
                mipdata_->lp.getLpSolver().getSolution().col_value);

          mipdata_->heuristics.flushStatistics();
        }
      }

      considerHeuristics = false;

      if (mipdata_->domain.infeasible()) break;

      if (!search.currentNodePruned()) {
        if (search.dive() == HighsSearch::NodeResult::kSubOptimal) break;

        ++mipdata_->num_leaves;

        search.flushStatistics();
      }

      if (mipdata_->checkLimits()) {
        limit_reached = true;
        break;
      }

      HighsInt numPlungeNodes = mipdata_->num_nodes - plungestart;
      if (numPlungeNodes >= 100) break;

      if (!search.backtrackPlunge(mipdata_->nodequeue)) break;

      assert(search.hasNode());

      if (mipdata_->conflictPool.getNumConflicts() >
          options_mip_->mip_pool_soft_limit)
        mipdata_->conflictPool.performAging();

      search.flushStatistics();
      mipdata_->printDisplayLine();
      // printf("continue plunging due to good esitmate\n");
    }
    search.openNodesToQueue(mipdata_->nodequeue);
    search.flushStatistics();

    if (limit_reached) {
      mipdata_->lower_bound = std::min(mipdata_->upper_bound,
                                       mipdata_->nodequeue.getBestLowerBound());
      mipdata_->printDisplayLine();
      break;
    }

    // the search datastructure should have no installed node now
    assert(!search.hasNode());

    // propagate the global domain
    mipdata_->domain.propagate();
    mipdata_->pruned_treeweight += mipdata_->nodequeue.pruneInfeasibleNodes(
        mipdata_->domain, mipdata_->feastol);

    // if global propagation detected infeasibility, stop here
    if (mipdata_->domain.infeasible()) {
      mipdata_->nodequeue.clear();
      mipdata_->pruned_treeweight = 1.0;
      mipdata_->lower_bound = std::min(kHighsInf, mipdata_->upper_bound);
      mipdata_->printDisplayLine();
      break;
    }

    mipdata_->lower_bound = std::min(mipdata_->upper_bound,
                                     mipdata_->nodequeue.getBestLowerBound());
    mipdata_->printDisplayLine();
    if (mipdata_->nodequeue.empty()) break;

    // if global propagation found bound changes, we update the local domain
    if (!mipdata_->domain.getChangedCols().empty()) {
      highsLogDev(options_mip_->log_options, HighsLogType::kInfo,
                  "added %" HIGHSINT_FORMAT " global bound changes\n",
                  (HighsInt)mipdata_->domain.getChangedCols().size());
      mipdata_->cliquetable.cleanupFixed(mipdata_->domain);
      for (HighsInt col : mipdata_->domain.getChangedCols())
        mipdata_->implications.cleanupVarbounds(col);

      mipdata_->domain.setDomainChangeStack(std::vector<HighsDomainChange>());
      search.resetLocalDomain();

      mipdata_->domain.clearChangedCols();
      mipdata_->removeFixedIndices();
    }

    if (!submip && mipdata_->num_nodes >= nextCheck) {
      auto nTreeRestarts = mipdata_->numRestarts - mipdata_->numRestartsRoot;
      double currNodeEstim =
          numNodesLastCheck - mipdata_->num_nodes_before_run +
          (mipdata_->num_nodes - numNodesLastCheck) *
              double(1.0 - mipdata_->pruned_treeweight) /
              std::max(
                  double(mipdata_->pruned_treeweight - treeweightLastCheck),
                  mipdata_->epsilon);
      // printf(
      //     "nTreeRestarts: %d, numNodesThisRun: %ld, numNodesLastCheck: %ld,
      //     " "currNodeEstim: %g, " "prunedTreeWeightDelta: %g,
      //     numHugeTreeEstim: %d, numLeavesThisRun:
      //     "
      //     "%ld\n",
      //     nTreeRestarts, mipdata_->num_nodes -
      //     mipdata_->num_nodes_before_run, numNodesLastCheck -
      //     mipdata_->num_nodes_before_run, currNodeEstim, 100.0 *
      //     double(mipdata_->pruned_treeweight - treeweightLastCheck),
      //     numHugeTreeEstim,
      //     mipdata_->num_leaves - mipdata_->num_leaves_before_run);

      bool doRestart = false;

      double activeIntegerRatio =
          1.0 - mipdata_->percentageInactiveIntegers() / 100.0;
      activeIntegerRatio *= activeIntegerRatio;

      if (!doRestart) {
        double gapReduction = 1.0;
        if (mipdata_->upper_limit != kHighsInf) {
          double oldGap = upperLimLastCheck - lowerBoundLastCheck;
          double newGap = mipdata_->upper_limit - mipdata_->lower_bound;
          gapReduction = oldGap / newGap;
        }

        if (gapReduction < 1.0 + (0.05 / activeIntegerRatio) &&
            currNodeEstim >=
                activeIntegerRatio * 20 *
                    (mipdata_->num_nodes - mipdata_->num_nodes_before_run)) {
          nextCheck = mipdata_->num_nodes + 100;
          ++numHugeTreeEstim;
        } else {
          numHugeTreeEstim = 0;
          treeweightLastCheck = double(mipdata_->pruned_treeweight);
          numNodesLastCheck = mipdata_->num_nodes;
          upperLimLastCheck = mipdata_->upper_limit;
          lowerBoundLastCheck = mipdata_->lower_bound;
        }

        int64_t minHugeTreeOffset =
            (mipdata_->num_leaves - mipdata_->num_leaves_before_run) * 1e-3;
        int64_t minHugeTreeEstim = HighsIntegers::nearestInteger(
            activeIntegerRatio * (10 + minHugeTreeOffset) *
            std::pow(1.5, nTreeRestarts));

        doRestart = numHugeTreeEstim >= minHugeTreeEstim;
      } else {
        // count restart due to many fixings within the first 1000 nodes as
        // root restart
        ++mipdata_->numRestartsRoot;
      }

      if (doRestart) {
        highsLogUser(options_mip_->log_options, HighsLogType::kInfo,
                     "\nRestarting search from the root node\n");
        mipdata_->performRestart();
        goto restart;
      }
    }

    // remove the iteration limit when installing a new node
    // mipdata_->lp.setIterationLimit();

    // loop to install the next node for the search
    while (!mipdata_->nodequeue.empty()) {
      // printf("popping node from nodequeue (length = %" HIGHSINT_FORMAT ")\n",
      // (HighsInt)nodequeue.size());
      assert(!search.hasNode());

      if (numQueueLeaves - lastLbLeave >= 10) {
        search.installNode(mipdata_->nodequeue.popBestBoundNode());
        lastLbLeave = numQueueLeaves;
      } else {
        HighsInt bestBoundNodeStackSize =
            mipdata_->nodequeue.getBestBoundDomchgStackSize();
        double bestBoundNodeLb = mipdata_->nodequeue.getBestLowerBound();
        HighsNodeQueue::OpenNode nextNode(mipdata_->nodequeue.popBestNode());
        if (nextNode.lower_bound == bestBoundNodeLb &&
            nextNode.domchgstack.size() == bestBoundNodeStackSize)
          lastLbLeave = numQueueLeaves;
        search.installNode(std::move(nextNode));
      }

      ++numQueueLeaves;

      if (search.getCurrentEstimate() >= mipdata_->upper_limit) {
        ++numStallNodes;
        if (options_mip_->mip_max_stall_nodes != kHighsIInf &&
            numStallNodes >= options_mip_->mip_max_stall_nodes) {
          limit_reached = true;
          modelstatus_ = HighsModelStatus::kIterationLimit;
          break;
        }
      } else
        numStallNodes = 0;

      assert(search.hasNode());

      // we evaluate the node directly here instead of performing a dive
      // because we first want to check if the node is not fathomed due to
      // new global information before we perform separation rounds for the node
      if (search.evaluateNode() == HighsSearch::NodeResult::kSubOptimal)
        search.currentNodeToQueue(mipdata_->nodequeue);

      // if the node was pruned we remove it from the search and install the
      // next node from the queue
      if (search.currentNodePruned()) {
        search.backtrack();
        ++mipdata_->num_leaves;
        ++mipdata_->num_nodes;
        search.flushStatistics();

        mipdata_->domain.propagate();
        mipdata_->pruned_treeweight += mipdata_->nodequeue.pruneInfeasibleNodes(
            mipdata_->domain, mipdata_->feastol);

        if (mipdata_->domain.infeasible()) {
          mipdata_->nodequeue.clear();
          mipdata_->pruned_treeweight = 1.0;
          mipdata_->lower_bound = std::min(kHighsInf, mipdata_->upper_bound);
          break;
        }

        if (mipdata_->checkLimits()) {
          limit_reached = true;
          break;
        }

        mipdata_->lower_bound = std::min(
            mipdata_->upper_bound, mipdata_->nodequeue.getBestLowerBound());

        mipdata_->printDisplayLine();

        if (!mipdata_->domain.getChangedCols().empty()) {
          highsLogDev(options_mip_->log_options, HighsLogType::kInfo,
                      "added %" HIGHSINT_FORMAT " global bound changes\n",
                      (HighsInt)mipdata_->domain.getChangedCols().size());
          mipdata_->cliquetable.cleanupFixed(mipdata_->domain);
          for (HighsInt col : mipdata_->domain.getChangedCols())
            mipdata_->implications.cleanupVarbounds(col);

          mipdata_->domain.setDomainChangeStack(
              std::vector<HighsDomainChange>());
          search.resetLocalDomain();

          mipdata_->domain.clearChangedCols();
          mipdata_->removeFixedIndices();
        }

        continue;
      }

      // the node is still not fathomed, so perform separation
      sepa.separate(search.getLocalDomain());

      if (mipdata_->domain.infeasible()) {
        search.cutoffNode();
        search.openNodesToQueue(mipdata_->nodequeue);
        mipdata_->nodequeue.clear();
        mipdata_->pruned_treeweight = 1.0;
        mipdata_->lower_bound = std::min(kHighsInf, mipdata_->upper_bound);
        break;
      }

      // after separation we store the new basis and proceed with the outer loop
      // to perform a dive from this node
      if (mipdata_->lp.getStatus() != HighsLpRelaxation::Status::kError &&
          mipdata_->lp.getStatus() != HighsLpRelaxation::Status::kNotSet)
        mipdata_->lp.storeBasis();

      basis = mipdata_->lp.getStoredBasis();
      if (!basis || !isBasisConsistent(mipdata_->lp.getLp(), *basis)) {
        HighsBasis b = mipdata_->firstrootbasis;
        b.row_status.resize(mipdata_->lp.numRows(), HighsBasisStatus::kBasic);
        basis = std::make_shared<const HighsBasis>(std::move(b));
        mipdata_->lp.setStoredBasis(basis);
      }

      break;
    }

    if (limit_reached) break;
  }

  cleanupSolve();
}

void HighsMipSolver::cleanupSolve() {
  timer_.start(timer_.postsolve_clock);
  bool havesolution = solution_objective_ != kHighsInf;
  bool feasible;
  if (havesolution)
    feasible =
        bound_violation_ <= options_mip_->mip_feasibility_tolerance &&
        integrality_violation_ <= options_mip_->mip_feasibility_tolerance &&
        row_violation_ <= options_mip_->mip_feasibility_tolerance;
  else
    feasible = false;

  dual_bound_ = mipdata_->lower_bound;
  if (mipdata_->objectiveFunction.isIntegral()) {
    double rounded_lower_bound =
        std::ceil(mipdata_->lower_bound *
                      mipdata_->objectiveFunction.integralScale() -
                  mipdata_->feastol) /
        mipdata_->objectiveFunction.integralScale();
    dual_bound_ = std::max(dual_bound_, rounded_lower_bound);
  }
  dual_bound_ += model_->offset_;
  primal_bound_ = mipdata_->upper_bound + model_->offset_;
  node_count_ = mipdata_->num_nodes;
  dual_bound_ = std::min(dual_bound_, primal_bound_);

  // adjust objective sense in case of maximization problem
  if (orig_model_->sense_ == ObjSense::kMaximize) {
    dual_bound_ = -dual_bound_;
    primal_bound_ = -primal_bound_;
  }

  if (modelstatus_ == HighsModelStatus::kNotset ||
      modelstatus_ == HighsModelStatus::kInfeasible) {
    if (feasible && havesolution)
      modelstatus_ = HighsModelStatus::kOptimal;
    else
      modelstatus_ = HighsModelStatus::kInfeasible;
  }

  timer_.stop(timer_.postsolve_clock);
  timer_.stop(timer_.solve_clock);
  std::string solutionstatus = "-";

  if (havesolution) {
    bool feasible =
        bound_violation_ <= options_mip_->mip_feasibility_tolerance &&
        integrality_violation_ <= options_mip_->mip_feasibility_tolerance &&
        row_violation_ <= options_mip_->mip_feasibility_tolerance;
    solutionstatus = feasible ? "feasible" : "infeasible";
  }

  gap_ = fabs(primal_bound_ - dual_bound_);
  if (primal_bound_ == 0.0)
    gap_ = dual_bound_ == 0.0 ? 0.0 : kHighsInf;
  else if (primal_bound_ != kHighsInf)
    gap_ = fabs(primal_bound_ - dual_bound_) / fabs(primal_bound_);
  else
    gap_ = kHighsInf;

  std::array<char, 128> gapString;

  if (gap_ == kHighsInf)
    std::strcpy(gapString.data(), "inf");
  else {
    double printTol = std::max(std::min(1e-2, 1e-1 * gap_), 1e-6);
    std::array<char, 32> gapValString =
        highsDoubleToString(100.0 * gap_, printTol);
    double gapTol = options_mip_->mip_rel_gap;

    if (options_mip_->mip_abs_gap > options_mip_->mip_feasibility_tolerance) {
      gapTol = primal_bound_ == 0.0
                   ? kHighsInf
                   : std::max(gapTol,
                              options_mip_->mip_abs_gap / fabs(primal_bound_));
    }

    if (gapTol == 0.0)
      std::snprintf(gapString.data(), gapString.size(), "%s%%",
                    gapValString.data());
    else if (gapTol != kHighsInf) {
      printTol = std::max(std::min(1e-2, 1e-1 * gapTol), 1e-6);
      std::array<char, 32> gapTolString =
          highsDoubleToString(100.0 * gapTol, printTol);
      std::snprintf(gapString.data(), gapString.size(),
                    "%s%% (tolerance: %s%%)", gapValString.data(),
                    gapTolString.data());
    } else
      std::snprintf(gapString.data(), gapString.size(), "%s%% (tolerance: inf)",
                    gapValString.data());
  }

  highsLogUser(options_mip_->log_options, HighsLogType::kInfo,
               "\nSolving report\n"
               "  Status            %s\n"
               "  Primal bound      %.12g\n"
               "  Dual bound        %.12g\n"
               "  Gap               %s\n"
               "  Solution status   %s\n",
               utilModelStatusToString(modelstatus_).c_str(), primal_bound_,
               dual_bound_, gapString.data(), solutionstatus.c_str());
  if (solutionstatus != "-")
    highsLogUser(options_mip_->log_options, HighsLogType::kInfo,
                 "                    %.12g (objective)\n"
                 "                    %.12g (bound viol.)\n"
                 "                    %.12g (int. viol.)\n"
                 "                    %.12g (row viol.)\n",
                 solution_objective_, bound_violation_, integrality_violation_,
                 row_violation_);
  highsLogUser(options_mip_->log_options, HighsLogType::kInfo,
               "  Timing            %.2f (total)\n"
               "                    %.2f (presolve)\n"
               "                    %.2f (postsolve)\n"
               "  Nodes             %llu\n"
               "  LP iterations     %llu (total)\n"
               "                    %llu (strong br.)\n"
               "                    %llu (separation)\n"
               "                    %llu (heuristics)\n",
               timer_.read(timer_.solve_clock),
               timer_.read(timer_.presolve_clock),
               timer_.read(timer_.postsolve_clock),
               (long long unsigned)mipdata_->num_nodes,
               (long long unsigned)mipdata_->total_lp_iterations,
               (long long unsigned)mipdata_->sb_lp_iterations,
               (long long unsigned)mipdata_->sepa_lp_iterations,
               (long long unsigned)mipdata_->heuristic_lp_iterations);

  assert(modelstatus_ != HighsModelStatus::kNotset);
}
