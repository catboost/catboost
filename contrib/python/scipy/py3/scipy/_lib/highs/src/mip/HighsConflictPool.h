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
#ifndef HIGHS_CONFLICTPOOL_H_
#define HIGHS_CONFLICTPOOL_H_

#include <set>
#include <vector>

#include "mip/HighsDomain.h"
#include "util/HighsInt.h"

class HighsConflictPool {
 private:
  HighsInt agelim_;
  HighsInt softlimit_;
  std::vector<HighsInt> ageDistribution_;
  std::vector<int16_t> ages_;
  std::vector<unsigned> modification_;

  std::vector<HighsDomainChange> conflictEntries_;
  std::vector<std::pair<HighsInt, HighsInt>> conflictRanges_;

  /// keep an ordered set of free spaces in the row arrays so that they can be
  /// reused efficiently
  std::set<std::pair<HighsInt, HighsInt>> freeSpaces_;

  /// vector of deleted conflicts so that their indices can be reused
  std::vector<HighsInt> deletedConflicts_;

  std::vector<HighsDomain::ConflictPoolPropagation*> propagationDomains;

 public:
  HighsConflictPool(HighsInt agelim, HighsInt softlimit)
      : agelim_(agelim),
        softlimit_(softlimit),
        ageDistribution_(),
        ages_(),
        modification_(),
        conflictEntries_(),
        conflictRanges_(),
        freeSpaces_(),
        deletedConflicts_(),
        propagationDomains() {
    ageDistribution_.resize(agelim_ + 1);
  }

  void addConflictCut(const HighsDomain& domain,
                      const std::set<HighsDomain::ConflictSet::LocalDomChg>&
                          reasonSideFrontier);

  void addReconvergenceCut(
      const HighsDomain& domain,
      const std::set<HighsDomain::ConflictSet::LocalDomChg>&
          reconvergenceFrontier,
      const HighsDomainChange& reconvergenceDomchg);

  void removeConflict(HighsInt conflict);

  void performAging();

  void resetAge(HighsInt conflict) {
    if (ages_[conflict] > 0) {
      ageDistribution_[ages_[conflict]] -= 1;
      ageDistribution_[0] += 1;
      ages_[conflict] = 0;
    }
  }

  void setAgeLimit(HighsInt agelim) {
    agelim_ = agelim;
    ageDistribution_.resize(agelim_ + 1);
  }

  unsigned getModificationCount(HighsInt cut) const {
    return modification_[cut];
  }

  void addPropagationDomain(HighsDomain::ConflictPoolPropagation* domain) {
    propagationDomains.push_back(domain);
  }

  void removePropagationDomain(HighsDomain::ConflictPoolPropagation* domain) {
    for (HighsInt k = propagationDomains.size() - 1; k >= 0; --k) {
      if (propagationDomains[k] == domain) {
        propagationDomains.erase(propagationDomains.begin() + k);
        return;
      }
    }
  }

  const std::vector<HighsDomainChange>& getConflictEntryVector() const {
    return conflictEntries_;
  }

  const std::vector<std::pair<HighsInt, HighsInt>>& getConflictRanges() const {
    return conflictRanges_;
  }

  HighsInt getNumConflicts() const {
    return conflictRanges_.size() - deletedConflicts_.size();
  }
};

#endif
