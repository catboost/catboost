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
/**@file mip/HighsDebugSol.h
 * @brief Debug solution for MIP solver
 */

#ifndef HIGHS_DEBUG_SOL_H_
#define HIGHS_DEBUG_SOL_H_

class HighsDomain;
class HighsMipSolver;
class HighsLp;

#include <set>
#include <vector>

#include "mip/HighsCliqueTable.h"
#include "mip/HighsDomain.h"

#ifdef HIGHS_DEBUGSOL

#include <unordered_map>

struct HighsDebugSol {
  const HighsMipSolver* mipsolver;
  double debugSolObjective;
  std::vector<double> debugSolution;
  bool debugSolActive;
  std::unordered_map<const HighsDomain*, std::multiset<HighsDomainChange>>
      conflictingBounds;

  HighsDebugSol(HighsMipSolver& mipsolver);

  void newIncumbentFound();

  void activate();

  void shrink(const std::vector<HighsInt>& newColIndex);

  void registerDomain(const HighsDomain& domain);

  void boundChangeAdded(const HighsDomain& domain,
                        const HighsDomainChange& domchg,
                        bool branching = false);

  void boundChangeRemoved(const HighsDomain& domain,
                          const HighsDomainChange& domchg);

  void resetDomain(const HighsDomain& domain);

  void nodePruned(const HighsDomain& localdomain);

  void checkCut(const HighsInt* Rindex, const double* Rvalue, HighsInt Rlen,
                double rhs);

  void checkRow(const HighsInt* Rindex, const double* Rvalue, HighsInt Rlen,
                double Rlower, double Rupper);

  void checkRowAggregation(const HighsLp& lp, const HighsInt* Rindex,
                           const double* Rvalue, HighsInt Rlen);

  void checkClique(const HighsCliqueTable::CliqueVar* clq, HighsInt clqlen);

  void checkVub(HighsInt col, HighsInt vubcol, double vubcoef,
                double vubconstant) const;

  void checkVlb(HighsInt col, HighsInt vlbcol, double vlbcoef,
                double vlbconstant) const;

  void checkConflictReasonFrontier(
      const std::set<HighsDomain::ConflictSet::LocalDomChg>& reasonSideFrontier,
      const std::vector<HighsDomainChange>& domchgstack) const;

  void checkConflictReconvergenceFrontier(
      const std::set<HighsDomain::ConflictSet::LocalDomChg>&
          reconvergenceFrontier,
      const HighsDomain::ConflictSet::LocalDomChg& reconvDomchgPos,
      const std::vector<HighsDomainChange>& domchgstack) const;
};

#else
struct HighsDebugSol {
  HighsDebugSol(HighsMipSolver& mipsolver) {}

  void newIncumbentFound() const {}

  void activate() const {}

  void shrink(const std::vector<HighsInt>& newColIndex) const {}

  void registerDomain(const HighsDomain& domain) const {}

  void boundChangeAdded(const HighsDomain& domain,
                        const HighsDomainChange& domchg,
                        bool branching = false) const {}

  void boundChangeRemoved(const HighsDomain& domain,
                          const HighsDomainChange& domchg) const {}

  void resetDomain(const HighsDomain& domain) const {}

  void nodePruned(const HighsDomain& localdomain) const {}

  void checkCut(const HighsInt* Rindex, const double* Rvalue, HighsInt Rlen,
                double rhs) const {}

  void checkRow(const HighsInt* Rindex, const double* Rvalue, HighsInt Rlen,
                double Rlower, double Rupper) const {}

  void checkRowAggregation(const HighsLp& lp, const HighsInt* Rindex,
                           const double* Rvalue, HighsInt Rlen) const {}

  void checkClique(const HighsCliqueTable::CliqueVar* clq,
                   HighsInt clqlen) const {}

  void checkVub(HighsInt col, HighsInt vubcol, double vubcoef,
                double vubconstant) const {}

  void checkVlb(HighsInt col, HighsInt vlbcol, double vlbcoef,
                double vlbconstant) const {}

  void checkConflictReasonFrontier(
      const std::set<HighsDomain::ConflictSet::LocalDomChg>& reasonSideFrontier,
      const std::vector<HighsDomainChange>& domchgstack) const {}

  void checkConflictReconvergenceFrontier(
      const std::set<HighsDomain::ConflictSet::LocalDomChg>&
          reconvergenceFrontier,
      const HighsDomain::ConflictSet::LocalDomChg& reconvDomchgPos,
      const std::vector<HighsDomainChange>& domchgstack) const {}
};
#endif

#endif
