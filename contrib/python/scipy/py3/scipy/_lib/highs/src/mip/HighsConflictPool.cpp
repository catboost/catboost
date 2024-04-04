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

#include "mip/HighsConflictPool.h"

#include "mip/HighsDomain.h"

void HighsConflictPool::addConflictCut(
    const HighsDomain& domain,
    const std::set<HighsDomain::ConflictSet::LocalDomChg>& reasonSideFrontier) {
  HighsInt conflictIndex;
  HighsInt start;
  HighsInt end;
  HighsInt conflictLen = reasonSideFrontier.size();
  std::set<std::pair<HighsInt, HighsInt>>::iterator it;
  if (freeSpaces_.empty() ||
      (it = freeSpaces_.lower_bound(
           std::make_pair(conflictLen, HighsInt{-1}))) == freeSpaces_.end()) {
    start = conflictEntries_.size();
    end = start + conflictLen;

    conflictEntries_.resize(end);
  } else {
    std::pair<HighsInt, HighsInt> freeslot = *it;
    freeSpaces_.erase(it);

    start = freeslot.second;
    end = start + conflictLen;
    // if the space was not completely occupied, we register the remainder of
    // it again in the priority queue
    if (freeslot.first > conflictLen) {
      freeSpaces_.emplace(freeslot.first - conflictLen, end);
    }
  }

  // register the range of entries for this conflict with a reused or a new
  // index
  if (deletedConflicts_.empty()) {
    conflictIndex = conflictRanges_.size();
    conflictRanges_.emplace_back(start, end);
    ages_.resize(conflictRanges_.size());
    modification_.resize(conflictRanges_.size());
  } else {
    conflictIndex = deletedConflicts_.back();
    deletedConflicts_.pop_back();
    conflictRanges_[conflictIndex].first = start;
    conflictRanges_[conflictIndex].second = end;
  }

  modification_[conflictIndex] += 1;
  ages_[conflictIndex] = 0;
  ageDistribution_[ages_[conflictIndex]] += 1;

  HighsInt i = start;
  const std::vector<HighsDomainChange>& domchgStack_ =
      domain.getDomainChangeStack();
  double feastol = domain.feastol();
  for (const HighsDomain::ConflictSet::LocalDomChg& domchg :
       reasonSideFrontier) {
    assert(i < end);
    assert(domchg.pos >= 0);
    assert(domchg.pos < (HighsInt)domchgStack_.size());
    conflictEntries_[i] = domchg.domchg;
    if (domain.variableType(conflictEntries_[i].column) ==
        HighsVarType::kContinuous) {
      if (conflictEntries_[i].boundtype == HighsBoundType::kLower)
        conflictEntries_[i].boundval += feastol;
      else
        conflictEntries_[i].boundval -= feastol;
    }
    ++i;
  }

  for (HighsDomain::ConflictPoolPropagation* conflictProp : propagationDomains)
    conflictProp->conflictAdded(conflictIndex);
}

void HighsConflictPool::addReconvergenceCut(
    const HighsDomain& domain,
    const std::set<HighsDomain::ConflictSet::LocalDomChg>&
        reconvergenceFrontier,
    const HighsDomainChange& reconvergenceDomchg) {
  HighsInt conflictIndex;
  HighsInt start;
  HighsInt end;
  HighsInt conflictLen = reconvergenceFrontier.size() + 1;
  std::set<std::pair<HighsInt, HighsInt>>::iterator it;
  if (freeSpaces_.empty() ||
      (it = freeSpaces_.lower_bound(
           std::make_pair(conflictLen, HighsInt{-1}))) == freeSpaces_.end()) {
    start = conflictEntries_.size();
    end = start + conflictLen;

    conflictEntries_.resize(end);
  } else {
    std::pair<HighsInt, HighsInt> freeslot = *it;
    freeSpaces_.erase(it);

    start = freeslot.second;
    end = start + conflictLen;
    // if the space was not completely occupied, we register the remainder of
    // it again in the priority queue
    if (freeslot.first > conflictLen) {
      freeSpaces_.emplace(freeslot.first - conflictLen, end);
    }
  }

  // register the range of entries for this conflict with a reused or a new
  // index
  if (deletedConflicts_.empty()) {
    conflictIndex = conflictRanges_.size();
    conflictRanges_.emplace_back(start, end);
    ages_.resize(conflictRanges_.size());
    modification_.resize(conflictRanges_.size());
  } else {
    conflictIndex = deletedConflicts_.back();
    deletedConflicts_.pop_back();
    conflictRanges_[conflictIndex].first = start;
    conflictRanges_[conflictIndex].second = end;
  }

  modification_[conflictIndex] += 1;
  ages_[conflictIndex] = 0;
  ageDistribution_[ages_[conflictIndex]] += 1;

  HighsInt i = start;
  const std::vector<HighsDomainChange>& domchgStack_ =
      domain.getDomainChangeStack();
  assert(i < end);
  conflictEntries_[i++] = domain.flip(reconvergenceDomchg);
  double feastol = domain.feastol();
  for (const HighsDomain::ConflictSet::LocalDomChg& domchg :
       reconvergenceFrontier) {
    assert(i < end);
    assert(domchg.pos >= 0);
    assert(domchg.pos < (HighsInt)domchgStack_.size());
    conflictEntries_[i] = domchg.domchg;
    if (domain.variableType(conflictEntries_[i].column) ==
        HighsVarType::kContinuous) {
      if (conflictEntries_[i].boundtype == HighsBoundType::kLower)
        conflictEntries_[i].boundval += feastol;
      else
        conflictEntries_[i].boundval -= feastol;
    }
    ++i;
  }

  for (HighsDomain::ConflictPoolPropagation* conflictProp : propagationDomains)
    conflictProp->conflictAdded(conflictIndex);
}

void HighsConflictPool::removeConflict(HighsInt conflict) {
  for (HighsDomain::ConflictPoolPropagation* conflictProp : propagationDomains)
    conflictProp->conflictDeleted(conflict);

  if (ages_[conflict] >= 0) {
    ageDistribution_[ages_[conflict]] -= 1;
    ages_[conflict] = -1;
  }

  HighsInt start = conflictRanges_[conflict].first;
  HighsInt end = conflictRanges_[conflict].second;

  // register the space of the deleted row and the index so that it can be
  // reused
  deletedConflicts_.push_back(conflict);
  freeSpaces_.emplace(end - start, start);

  // set the range to -1,-1 to indicate a deleted row
  conflictRanges_[conflict].first = -1;
  conflictRanges_[conflict].second = -1;
  ++modification_[conflict];
}

void HighsConflictPool::performAging() {
  HighsInt conflictMaxIndex = conflictRanges_.size();
  HighsInt agelim = agelim_;
  HighsInt numActiveConflicts = getNumConflicts();
  while (agelim > 5 && numActiveConflicts > softlimit_) {
    numActiveConflicts -= ageDistribution_[agelim];
    --agelim;
  }

  for (HighsInt i = 0; i != conflictMaxIndex; ++i) {
    if (ages_[i] < 0) continue;

    ageDistribution_[ages_[i]] -= 1;
    ages_[i] += 1;

    if (ages_[i] > agelim) {
      ages_[i] = -1;
      removeConflict(i);
    } else
      ageDistribution_[ages_[i]] += 1;
  }
}
