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
/**@file simplex/HSimplexNlaFreeze.cpp
 *
 * @brief Interface to HFactor allowing non-HFactor updates, NLA-only
 * scaling and shifting of NLA analysis below simplex level.
 */
#include <stdio.h>

#include "simplex/HSimplexNla.h"

void SimplexIterate::clear() {
  this->valid_ = false;
  this->basis_.clear();
  this->invert_.clear();
  this->dual_edge_weight_.clear();
}

void FrozenBasis::clear() {
  this->valid_ = false;
  this->prev_ = kNoLink;
  this->next_ = kNoLink;
  this->update_.clear();
  this->basis_.clear();
  this->dual_edge_weight_.clear();
}

bool HSimplexNla::frozenBasisAllDataClear() {
  bool all_clear = this->first_frozen_basis_id_ == kNoLink;
  all_clear = this->last_frozen_basis_id_ == kNoLink && all_clear;
  all_clear = this->frozen_basis_.size() == 0 && all_clear;
  all_clear = !this->update_.valid_ && all_clear;
  return all_clear;
}

void HSimplexNla::frozenBasisClearAllData() {
  this->first_frozen_basis_id_ = kNoLink;
  this->last_frozen_basis_id_ = kNoLink;
  this->frozen_basis_.clear();
  this->update_.clear();
}

void HSimplexNla::frozenBasisClearAllUpdate() {
  for (HighsInt frozen_basis_id = 0;
       frozen_basis_id < this->frozen_basis_.size(); frozen_basis_id++)
    this->frozen_basis_[frozen_basis_id].update_.clear();
  this->update_.clear();
}

bool HSimplexNla::frozenBasisIdValid(const HighsInt frozen_basis_id) const {
  bool valid_id =
      0 <= frozen_basis_id && frozen_basis_id < frozen_basis_.size();
  if (valid_id) valid_id = frozen_basis_[frozen_basis_id].valid_;
  return valid_id;
}

bool HSimplexNla::frozenBasisHasInvert(const HighsInt frozen_basis_id) const {
  // Determine whether there will be an invertible representation to
  // use after unfreezing this basis
  //
  // If there is no last frozen basis, then there can be no invertible
  // representation
  if (this->last_frozen_basis_id_ == kNoLink) return false;
  // Existence of the invertible representation depends on the
  // validity of the current PF updates
  return this->update_.valid_;
}

HighsInt HSimplexNla::freeze(const SimplexBasis& basis,
                             const double col_aq_density) {
  this->frozen_basis_.push_back(FrozenBasis());
  HighsInt this_frozen_basis_id = this->frozen_basis_.size() - 1;
  FrozenBasis& frozen_basis = this->frozen_basis_[this_frozen_basis_id];
  frozen_basis.valid_ = true;
  frozen_basis.prev_ = this->last_frozen_basis_id_;
  frozen_basis.next_ = kNoLink;
  frozen_basis.update_.clear();
  frozen_basis.basis_ = basis;
  if (this->last_frozen_basis_id_ == kNoLink) {
    // There is previously no frozen basis, so record this as the
    // first
    this->first_frozen_basis_id_ = this_frozen_basis_id;
  } else {
    // Update the forward link from the previous last frozen basis
    FrozenBasis& last_frozen_basis =
        this->frozen_basis_[this->last_frozen_basis_id_];
    last_frozen_basis.next_ = this_frozen_basis_id;
    // The PF updates held in simplex NLA now become the updates to
    // apply in order to go from the previous frozen basis to the
    // latest one
    last_frozen_basis.update_ = std::move(update_);
  }
  this->last_frozen_basis_id_ = this_frozen_basis_id;
  // Set up the structure for any PF updates that occur
  this->update_.setup(lp_->num_row_, col_aq_density);
  return this_frozen_basis_id;
}

void HSimplexNla::unfreeze(const HighsInt unfreeze_basis_id,
                           SimplexBasis& basis) {
  assert(frozenBasisIdValid(unfreeze_basis_id));
  FrozenBasis& frozen_basis = this->frozen_basis_[unfreeze_basis_id];
  // Move the frozen basis into the return basis
  basis = std::move(frozen_basis.basis_);
  // This frozen basis, and any linked forward from it, must be
  // cleared. Any link to it must also be cleared.
  HighsInt frozen_basis_id = unfreeze_basis_id;
  HighsInt prev_frozen_basis_id = frozen_basis.prev_;
  if (prev_frozen_basis_id == kNoLink) {
    // There is no previous frozen basis linking to this one, so all
    // frozen basis data can be cleared
    frozenBasisClearAllData();
  } else {
    // The previous frozen basis is now the last and has no link
    // forward
    this->last_frozen_basis_id_ = prev_frozen_basis_id;
    this->frozen_basis_[prev_frozen_basis_id].next_ = kNoLink;
    // Now clear the unfrozen basis any further frozen basis
    for (;;) {
      HighsInt next_frozen_basis_id =
          this->frozen_basis_[frozen_basis_id].next_;
      this->frozen_basis_[frozen_basis_id].clear();
      frozen_basis_id = next_frozen_basis_id;
      if (frozen_basis_id == kNoLink) break;
    }
    // Move any PF updates for the last frozen basis to the PF updates held in
    // simplex NLA
    FrozenBasis& last_frozen_basis =
        this->frozen_basis_[this->last_frozen_basis_id_];
    this->update_ = std::move(last_frozen_basis.update_);
    // Clear the (scalar) data associated with the PF updates of the last frozen
    // basis
    last_frozen_basis.update_.clear();
  }
  // Clear any refactorization information in case we are unfreezing
  // immediately after a factorization for a later basis
  this->factor_.refactor_info_.clear();
}

void HSimplexNla::putInvert() {
  simplex_iterate_.valid_ = true;
  simplex_iterate_.invert_ = factor_.getInvert();
}
void HSimplexNla::getInvert() { factor_.setInvert(simplex_iterate_.invert_); }
