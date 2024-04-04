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
#ifndef HIGHS_SEARCH_H_
#define HIGHS_SEARCH_H_

#include <cstdint>
#include <queue>
#include <vector>

#include "mip/HighsConflictPool.h"
#include "mip/HighsDomain.h"
#include "mip/HighsLpRelaxation.h"
#include "mip/HighsMipSolver.h"
#include "mip/HighsNodeQueue.h"
#include "mip/HighsPseudocost.h"
#include "mip/HighsSeparation.h"
#include "presolve/HighsSymmetry.h"
#include "util/HighsHash.h"

class HighsMipSolver;
class HighsImplications;
class HighsCliqueTable;

class HighsSearch {
  HighsMipSolver& mipsolver;
  HighsLpRelaxation* lp;
  HighsDomain localdom;
  HighsPseudocost pseudocost;
  HighsRandom random;
  int64_t nnodes;
  int64_t lpiterations;
  int64_t heurlpiterations;
  int64_t sblpiterations;
  double upper_limit;
  std::vector<HighsInt> inds;
  std::vector<double> vals;
  HighsInt depthoffset;
  bool inbranching;
  bool inheuristic;
  bool countTreeWeight;

 public:
  enum class ChildSelectionRule {
    kUp,
    kDown,
    kRootSol,
    kObj,
    kRandom,
    kBestCost,
    kWorstCost,
    kDisjunction,
    kHybridInferenceCost,
  };

  enum class NodeResult {
    kBoundExceeding,
    kDomainInfeasible,
    kLpInfeasible,
    kBranched,
    kSubOptimal,
    kOpen,
  };

 private:
  ChildSelectionRule childselrule;

  HighsCDouble treeweight;

  struct NodeData {
    double lower_bound;
    double estimate;
    double branching_point;
    // we store the lp objective separately to the lower bound since the lower
    // bound could be above the LP objective when cuts age out or below when the
    // LP is unscaled dual infeasible and it is not set. We still want to use
    // the objective for pseudocost updates and tiebreaking of best bound node
    // selection
    double lp_objective;
    double other_child_lb;
    std::shared_ptr<const HighsBasis> nodeBasis;
    std::shared_ptr<const StabilizerOrbits> stabilizerOrbits;
    HighsDomainChange branchingdecision;
    HighsInt domgchgStackPos;
    uint8_t skipDepthCount;
    uint8_t opensubtrees;

    NodeData(double parentlb = -kHighsInf, double parentestimate = -kHighsInf,
             std::shared_ptr<const HighsBasis> parentBasis = nullptr,
             std::shared_ptr<const StabilizerOrbits> stabilizerOrbits = nullptr)
        : lower_bound(parentlb),
          estimate(parentestimate),
          lp_objective(-kHighsInf),
          other_child_lb(parentlb),
          nodeBasis(std::move(parentBasis)),
          stabilizerOrbits(std::move(stabilizerOrbits)),
          branchingdecision{0.0, -1, HighsBoundType::kLower},
          domgchgStackPos(-1),
          skipDepthCount(0),
          opensubtrees(2) {}
  };

  enum ReliableFlags {
    kUpReliable = 1,
    kDownReliable = 2,
    kReliable = kDownReliable | kUpReliable,
  };

  std::vector<double> subrootsol;
  std::vector<NodeData> nodestack;
  HighsHashTable<HighsInt, int> reliableatnode;

  int branchingVarReliableAtNodeFlags(HighsInt col) const {
    auto it = reliableatnode.find(col);
    if (it == nullptr) return 0;
    return *it;
  }

  bool branchingVarReliableAtNode(HighsInt col) const {
    auto it = reliableatnode.find(col);
    if (it == nullptr || *it != kReliable) return false;

    return true;
  }

  void markBranchingVarUpReliableAtNode(HighsInt col) {
    reliableatnode[col] |= kUpReliable;
  }

  void markBranchingVarDownReliableAtNode(HighsInt col) {
    reliableatnode[col] |= kDownReliable;
  }

  bool orbitsValidInChildNode(const HighsDomainChange& branchChg) const;

 public:
  HighsSearch(HighsMipSolver& mipsolver, const HighsPseudocost& pseudocost);

  void setRINSNeighbourhood(const std::vector<double>& basesol,
                            const std::vector<double>& relaxsol);

  void setRENSNeighbourhood(const std::vector<double>& lpsol);

  double getCutoffBound() const;

  void setLpRelaxation(HighsLpRelaxation* lp) { this->lp = lp; }

  double checkSol(const std::vector<double>& sol, bool& integerfeasible) const;

  void createNewNode();

  void cutoffNode();

  void branchDownwards(HighsInt col, double newub, double branchpoint);

  void branchUpwards(HighsInt col, double newlb, double branchpoint);

  void setMinReliable(HighsInt minreliable);

  void setHeuristic(bool inheuristic) {
    this->inheuristic = inheuristic;
    if (inheuristic) childselrule = ChildSelectionRule::kHybridInferenceCost;
  }

  void addBoundExceedingConflict();

  void resetLocalDomain();

  int64_t getHeuristicLpIterations() const;

  int64_t getTotalLpIterations() const;

  int64_t getLocalLpIterations() const;

  int64_t getLocalNodes() const;

  int64_t getStrongBranchingLpIterations() const;

  bool hasNode() const { return !nodestack.empty(); }

  bool currentNodePruned() const { return nodestack.back().opensubtrees == 0; }

  double getCurrentEstimate() const { return nodestack.back().estimate; }

  double getCurrentLowerBound() const { return nodestack.back().lower_bound; }

  HighsInt getCurrentDepth() const { return nodestack.size() + depthoffset; }

  void openNodesToQueue(HighsNodeQueue& nodequeue);

  void currentNodeToQueue(HighsNodeQueue& nodequeue);

  void flushStatistics();

  void installNode(HighsNodeQueue::OpenNode&& node);

  void addInfeasibleConflict();

  HighsInt selectBranchingCandidate(int64_t maxSbIters, double& downNodeLb,
                                    double& upNodeLb);

  void evalUnreliableBranchCands();

  const NodeData* getParentNodeData() const;

  NodeResult evaluateNode();

  NodeResult branch();

  /// backtrack one level in DFS manner
  bool backtrack(bool recoverBasis = true);

  /// backtrack an unspecified amount of depth level until the next
  /// node that seems worthwhile to continue the plunge. Put unpromising nodes
  /// to the node queue
  bool backtrackPlunge(HighsNodeQueue& nodequeue);

  /// for heuristics. Will discard nodes above targetDepth regardless of their
  /// status
  bool backtrackUntilDepth(HighsInt targetDepth);

  void printDisplayLine(char first, bool header = false);

  NodeResult dive();

  HighsDomain& getLocalDomain() { return localdom; }

  const HighsDomain& getLocalDomain() const { return localdom; }

  HighsPseudocost& getPseudoCost() { return pseudocost; }

  const HighsPseudocost& getPseudoCost() const { return pseudocost; }

  void solveDepthFirst(int64_t maxbacktracks = 1);
};

#endif
