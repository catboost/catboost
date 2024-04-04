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
#include "presolve/HPresolve.h"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <limits>

#include "Highs.h"
#include "io/HighsIO.h"
#include "lp_data/HConst.h"
#include "lp_data/HStruct.h"
#include "lp_data/HighsLpUtils.h"
#include "lp_data/HighsSolution.h"
#include "mip/HighsCliqueTable.h"
#include "mip/HighsImplications.h"
#include "mip/HighsMipSolverData.h"
#include "mip/HighsObjectiveFunction.h"
#include "pdqsort/pdqsort.h"
#include "presolve/HighsPostsolveStack.h"
#include "test/DevKkt.h"
#include "util/HFactor.h"
#include "util/HighsCDouble.h"
#include "util/HighsIntegers.h"
#include "util/HighsLinearSumBounds.h"
#include "util/HighsSplay.h"
#include "util/HighsUtils.h"

#define ENABLE_SPARSIFY_FOR_LP 0

#define HPRESOLVE_CHECKED_CALL(presolveCall)                           \
  do {                                                                 \
    HPresolve::Result __result = presolveCall;                         \
    if (__result != presolve::HPresolve::Result::kOk) return __result; \
  } while (0)

namespace presolve {

#ifndef NDEBUG
void HPresolve::debugPrintRow(HighsPostsolveStack& postsolve_stack,
                              HighsInt row) {
  printf("(row %" HIGHSINT_FORMAT ") %.15g (impl: %.15g) <= ",
         postsolve_stack.getOrigRowIndex(row), model->row_lower_[row],
         impliedRowBounds.getSumLower(row));

  for (const HighsSliceNonzero& nonzero : getSortedRowVector(row)) {
    // for (HighsInt rowiter = rowhead[row]; rowiter != -1; rowiter =
    // ARnext[rowiter]) {
    char colchar =
        model->integrality_[nonzero.index()] == HighsVarType::kInteger ? 'y'
                                                                       : 'x';
    char signchar = nonzero.value() < 0 ? '-' : '+';
    printf("%c%g %c%" HIGHSINT_FORMAT " ", signchar, std::abs(nonzero.value()),
           colchar, postsolve_stack.getOrigColIndex(nonzero.index()));
  }

  printf("<= %.15g (impl: %.15g)\n", model->row_upper_[row],
         impliedRowBounds.getSumUpper(row));
}
#endif

void HPresolve::setInput(HighsLp& model_, const HighsOptions& options_,
                         HighsTimer* timer) {
  model = &model_;
  options = &options_;
  this->timer = timer;

  colLowerSource.resize(model->num_col_, -1);
  colUpperSource.resize(model->num_col_, -1);
  implColLower.resize(model->num_col_, -kHighsInf);
  implColUpper.resize(model->num_col_, kHighsInf);

  rowDualLower.resize(model->num_row_, -kHighsInf);
  rowDualUpper.resize(model->num_row_, kHighsInf);
  implRowDualLower.resize(model->num_row_, -kHighsInf);
  implRowDualUpper.resize(model->num_row_, kHighsInf);
  rowDualUpperSource.resize(model->num_row_, -1);
  rowDualLowerSource.resize(model->num_row_, -1);

  for (HighsInt i = 0; i != model->num_row_; ++i) {
    if (model->row_lower_[i] == -kHighsInf) rowDualUpper[i] = 0;
    if (model->row_upper_[i] == kHighsInf) rowDualLower[i] = 0;
  }

  if (mipsolver == nullptr) {
    primal_feastol = options->primal_feasibility_tolerance;
    model->integrality_.assign(model->num_col_, HighsVarType::kContinuous);
  } else
    primal_feastol = options->mip_feasibility_tolerance;

  if (model_.a_matrix_.isRowwise())
    fromCSR(model->a_matrix_.value_, model->a_matrix_.index_,
            model->a_matrix_.start_);
  else
    fromCSC(model->a_matrix_.value_, model->a_matrix_.index_,
            model->a_matrix_.start_);

  // initialize everything as changed, but do not add all indices
  // since the first thing presolve will do is a scan for easy reductions
  // of each row and column and set the flag of processed columns to false
  // from then on they are added to the vector whenever there are changes
  changedRowFlag.resize(model->num_row_, true);
  rowDeleted.resize(model->num_row_, false);
  changedRowIndices.reserve(model->num_row_);
  changedColFlag.resize(model->num_col_, true);
  colDeleted.resize(model->num_col_, false);
  changedColIndices.reserve(model->num_col_);
  numDeletedCols = 0;
  numDeletedRows = 0;
  reductionLimit = std::numeric_limits<size_t>::max();
}

// for MIP presolve
void HPresolve::setInput(HighsMipSolver& mipsolver) {
  this->mipsolver = &mipsolver;

  probingContingent = 1000;
  probingNumDelCol = 0;
  numProbed = 0;
  numProbes.assign(mipsolver.numCol(), 0);

  if (mipsolver.model_ != &mipsolver.mipdata_->presolvedModel) {
    mipsolver.mipdata_->presolvedModel = *mipsolver.model_;
    mipsolver.model_ = &mipsolver.mipdata_->presolvedModel;
  } else {
    mipsolver.mipdata_->presolvedModel.col_lower_ =
        mipsolver.mipdata_->domain.col_lower_;
    mipsolver.mipdata_->presolvedModel.col_upper_ =
        mipsolver.mipdata_->domain.col_upper_;
  }

  setInput(mipsolver.mipdata_->presolvedModel, *mipsolver.options_mip_,
           &mipsolver.timer_);
}

bool HPresolve::rowCoefficientsIntegral(HighsInt row, double scale) const {
  for (const HighsSliceNonzero& nz : getRowVector(row)) {
    double val = nz.value() * scale;
    if (std::abs(val - std::round(val)) > options->small_matrix_value)
      return false;
  }

  return true;
}

bool HPresolve::isLowerImplied(HighsInt col) const {
  return (model->col_lower_[col] == -kHighsInf ||
          implColLower[col] >= model->col_lower_[col] - primal_feastol);
}

bool HPresolve::isUpperImplied(HighsInt col) const {
  return (model->col_upper_[col] == kHighsInf ||
          implColUpper[col] <= model->col_upper_[col] + primal_feastol);
}

bool HPresolve::isImpliedFree(HighsInt col) const {
  return (model->col_lower_[col] == -kHighsInf ||
          implColLower[col] >= model->col_lower_[col] - primal_feastol) &&
         (model->col_upper_[col] == kHighsInf ||
          implColUpper[col] <= model->col_upper_[col] + primal_feastol);
}

bool HPresolve::isDualImpliedFree(HighsInt row) const {
  return model->row_lower_[row] == model->row_upper_[row] ||
         (model->row_upper_[row] != kHighsInf &&
          implRowDualUpper[row] <= options->dual_feasibility_tolerance) ||
         (model->row_lower_[row] != -kHighsInf &&
          implRowDualLower[row] >= -options->dual_feasibility_tolerance);
}

bool HPresolve::isImpliedIntegral(HighsInt col) {
  bool runDualDetection = true;

  assert(model->integrality_[col] == HighsVarType::kInteger);

  for (const HighsSliceNonzero& nz : getColumnVector(col)) {
    // if not all other columns are integer, skip row and also do not try the
    // dual detection in the second loop as it must hold for all rows
    if (rowsizeInteger[nz.index()] < rowsize[nz.index()]) {
      runDualDetection = false;
      continue;
    }

    double rowLower =
        implRowDualUpper[nz.index()] < -options->dual_feasibility_tolerance
            ? model->row_upper_[nz.index()]
            : model->row_lower_[nz.index()];

    double rowUpper =
        implRowDualLower[nz.index()] > options->dual_feasibility_tolerance
            ? model->row_lower_[nz.index()]
            : model->row_upper_[nz.index()];

    if (rowUpper == rowLower) {
      // if there is an equation the dual detection does not need to be tried
      runDualDetection = false;
      double scale = 1.0 / nz.value();
      if (!rowCoefficientsIntegral(nz.index(), scale)) continue;

      double rhs = model->row_lower_[nz.index()] * scale;

      if (std::abs(rhs - std::round(rhs)) > primal_feastol) {
        // todo infeasible
      }

      return true;
    }
  }

  if (!runDualDetection) return false;

  for (const HighsSliceNonzero& nz : getColumnVector(col)) {
    double scale = 1.0 / nz.value();
    if (!rowCoefficientsIntegral(nz.index(), scale)) return false;
    if (model->row_upper_[nz.index()] != kHighsInf) {
      double rUpper =
          std::abs(nz.value()) *
          std::floor(model->row_upper_[nz.index()] * std::abs(scale) +
                     primal_feastol);
      if (std::abs(model->row_upper_[nz.index()] - rUpper) >
          options->small_matrix_value) {
        model->row_upper_[nz.index()] = rUpper;
        markChangedRow(nz.index());
      }
    } else {
      assert(model->row_lower_[nz.index()] != -kHighsInf);
      double rLower =
          std::abs(nz.value()) *
          std::ceil(model->row_upper_[nz.index()] * std::abs(scale) -
                    primal_feastol);
      if (std::abs(model->row_lower_[nz.index()] - rLower) >
          options->small_matrix_value) {
        model->row_upper_[nz.index()] = rLower;
        markChangedRow(nz.index());
      }
    }
  }

  return true;
}

bool HPresolve::isImpliedInteger(HighsInt col) {
  bool runDualDetection = true;

  assert(model->integrality_[col] == HighsVarType::kContinuous);

  for (const HighsSliceNonzero& nz : getColumnVector(col)) {
    // if not all other columns are integer, skip row and also do not try the
    // dual detection in the second loop as it must hold for all rows
    if (rowsizeInteger[nz.index()] + rowsizeImplInt[nz.index()] <
        rowsize[nz.index()] - 1) {
      runDualDetection = false;
      continue;
    }

    double rowLower =
        implRowDualUpper[nz.index()] < -options->dual_feasibility_tolerance
            ? model->row_upper_[nz.index()]
            : model->row_lower_[nz.index()];

    double rowUpper =
        implRowDualLower[nz.index()] > options->dual_feasibility_tolerance
            ? model->row_lower_[nz.index()]
            : model->row_upper_[nz.index()];

    if (rowUpper == rowLower) {
      // if there is an equation the dual detection does not need to be tried
      runDualDetection = false;
      double scale = 1.0 / nz.value();
      double rhs = model->row_lower_[nz.index()] * scale;

      if (std::abs(rhs - std::round(rhs)) > primal_feastol) {
        continue;
      }

      if (!rowCoefficientsIntegral(nz.index(), scale)) continue;

      return true;
    }
  }

  if (!runDualDetection) return false;

  if ((model->col_lower_[col] != -kHighsInf &&
       std::abs(std::round(model->col_lower_[col]) - model->col_lower_[col]) >
           options->small_matrix_value) ||
      (model->col_upper_[col] != -kHighsInf &&
       std::abs(std::round(model->col_upper_[col]) - model->col_upper_[col]) >
           options->small_matrix_value))
    return false;

  for (const HighsSliceNonzero& nz : getColumnVector(col)) {
    double scale = 1.0 / nz.value();
    if (model->row_upper_[nz.index()] != kHighsInf) {
      double rhs = model->row_upper_[nz.index()];
      if (std::abs(rhs - std::round(rhs)) > primal_feastol) return false;
    }

    if (model->row_lower_[nz.index()] != -kHighsInf) {
      double rhs = model->row_lower_[nz.index()];
      if (std::abs(rhs - std::round(rhs)) > primal_feastol) return false;
    }

    if (!rowCoefficientsIntegral(nz.index(), scale)) return false;
  }

  return true;
}

void HPresolve::link(HighsInt pos) {
  Anext[pos] = colhead[Acol[pos]];
  Aprev[pos] = -1;
  colhead[Acol[pos]] = pos;
  if (Anext[pos] != -1) Aprev[Anext[pos]] = pos;

  ++colsize[Acol[pos]];

  ARleft[pos] = -1;
  ARright[pos] = -1;
  auto get_row_left = [&](HighsInt pos) -> HighsInt& { return ARleft[pos]; };
  auto get_row_right = [&](HighsInt pos) -> HighsInt& { return ARright[pos]; };
  auto get_row_key = [&](HighsInt pos) { return Acol[pos]; };
  highs_splay_link(pos, rowroot[Arow[pos]], get_row_left, get_row_right,
                   get_row_key);

  impliedRowBounds.add(Arow[pos], Acol[pos], Avalue[pos]);
  impliedDualRowBounds.add(Acol[pos], Arow[pos], Avalue[pos]);
  ++rowsize[Arow[pos]];
  if (model->integrality_[Acol[pos]] == HighsVarType::kInteger)
    ++rowsizeInteger[Arow[pos]];
  else if (model->integrality_[Acol[pos]] == HighsVarType::kImplicitInteger)
    ++rowsizeImplInt[Arow[pos]];
}

void HPresolve::unlink(HighsInt pos) {
  HighsInt next = Anext[pos];
  HighsInt prev = Aprev[pos];

  if (next != -1) Aprev[next] = prev;

  if (prev != -1)
    Anext[prev] = next;
  else
    colhead[Acol[pos]] = next;
  --colsize[Acol[pos]];

  if (!colDeleted[Acol[pos]]) {
    if (colsize[Acol[pos]] == 1)
      singletonColumns.push_back(Acol[pos]);
    else
      markChangedCol(Acol[pos]);

    impliedDualRowBounds.remove(Acol[pos], Arow[pos], Avalue[pos]);
    if (colUpperSource[Acol[pos]] == Arow[pos])
      changeImplColUpper(Acol[pos], kHighsInf, -1);

    if (colLowerSource[Acol[pos]] == Arow[pos])
      changeImplColLower(Acol[pos], -kHighsInf, -1);
  }

  auto get_row_left = [&](HighsInt pos) -> HighsInt& { return ARleft[pos]; };
  auto get_row_right = [&](HighsInt pos) -> HighsInt& { return ARright[pos]; };
  auto get_row_key = [&](HighsInt pos) { return Acol[pos]; };
  highs_splay_unlink(pos, rowroot[Arow[pos]], get_row_left, get_row_right,
                     get_row_key);
  --rowsize[Arow[pos]];
  if (model->integrality_[Acol[pos]] == HighsVarType::kInteger)
    --rowsizeInteger[Arow[pos]];
  else if (model->integrality_[Acol[pos]] == HighsVarType::kImplicitInteger)
    --rowsizeImplInt[Arow[pos]];

  if (!rowDeleted[Arow[pos]]) {
    if (rowsize[Arow[pos]] == 1)
      singletonRows.push_back(Arow[pos]);
    else
      markChangedRow(Arow[pos]);
    impliedRowBounds.remove(Arow[pos], Acol[pos], Avalue[pos]);

    if (rowDualUpperSource[Arow[pos]] == Acol[pos])
      changeImplRowDualUpper(Arow[pos], kHighsInf, -1);

    if (rowDualLowerSource[Arow[pos]] == Acol[pos])
      changeImplRowDualLower(Arow[pos], -kHighsInf, -1);
  }

  Avalue[pos] = 0;

  freeslots.push_back(pos);
}

void HPresolve::markChangedRow(HighsInt row) {
  if (!changedRowFlag[row]) {
    changedRowIndices.push_back(row);
    changedRowFlag[row] = true;
  }
}

void HPresolve::markChangedCol(HighsInt col) {
  if (!changedColFlag[col]) {
    changedColIndices.push_back(col);
    changedColFlag[col] = true;
  }
}

double HPresolve::getMaxAbsColVal(HighsInt col) const {
  double maxVal = 0.0;

  for (const auto& nz : getColumnVector(col))
    maxVal = std::max(std::abs(nz.value()), maxVal);

  return maxVal;
}

double HPresolve::getMaxAbsRowVal(HighsInt row) const {
  double maxVal = 0.0;

  for (const auto& nz : getRowVector(row))
    maxVal = std::max(std::abs(nz.value()), maxVal);

  return maxVal;
}

void HPresolve::updateRowDualImpliedBounds(HighsInt row, HighsInt col,
                                           double val) {
  // propagate implied row dual bound bound
  // if the column has an infinite lower bound the reduced cost cannot be
  // positive, i.e. the column corresponds to a <= constraint in the dual with
  // right hand side -cost which becomes a >= constraint with side +cost.
  // Furthermore, we can ignore strictly redundant primal
  // column bounds and treat them as if they are infinite
  double impliedMargin = colsize[col] != 1 ? primal_feastol : -primal_feastol;
  double dualRowLower =
      (model->col_lower_[col] == -kHighsInf) ||
              (implColLower[col] > model->col_lower_[col] + impliedMargin)
          ? model->col_cost_[col]
          : -kHighsInf;

  double dualRowUpper =
      (model->col_upper_[col] == kHighsInf) ||
              (implColUpper[col] < model->col_upper_[col] - impliedMargin)
          ? model->col_cost_[col]
          : kHighsInf;

  if (dualRowUpper != kHighsInf) {
    // get minimal value of other row duals in the column
    double residualMinAct =
        impliedDualRowBounds.getResidualSumLowerOrig(col, row, val);
    if (residualMinAct != -kHighsInf) {
      double impliedBound =
          double((HighsCDouble(dualRowUpper) - residualMinAct) / val);

      if (std::abs(impliedBound) * kHighsTiny <=
          options->dual_feasibility_tolerance) {
        if (val > 0) {
          // only tighten bound if it is tighter by a wide enough margin
          if (impliedBound < implRowDualUpper[row] -
                                 1000 * options->dual_feasibility_tolerance)
            changeImplRowDualUpper(row, impliedBound, col);
        } else {
          if (impliedBound > implRowDualLower[row] + 1000 * primal_feastol)
            changeImplRowDualLower(row, impliedBound, col);
        }
      }
    }
  }

  if (dualRowLower != -kHighsInf) {
    // get maximal value of other columns in the row
    double residualMaxAct =
        impliedDualRowBounds.getResidualSumUpperOrig(col, row, val);
    if (residualMaxAct != kHighsInf) {
      double impliedBound =
          double((HighsCDouble(dualRowLower) - residualMaxAct) / val);

      if (std::abs(impliedBound) * kHighsTiny <=
          options->dual_feasibility_tolerance) {
        if (val > 0) {
          // only tighten bound if it is tighter by a wide enough margin
          if (impliedBound > implRowDualLower[row] + 1000 * primal_feastol)
            changeImplRowDualLower(row, impliedBound, col);
        } else {
          if (impliedBound < implRowDualUpper[row] -
                                 1000 * options->dual_feasibility_tolerance)
            changeImplRowDualUpper(row, impliedBound, col);
        }
      }
    }
  }
}

void HPresolve::updateColImpliedBounds(HighsInt row, HighsInt col, double val) {
  // propagate implied column bound upper bound if row has an upper bound
  double rowUpper = implRowDualLower[row] > options->dual_feasibility_tolerance
                        ? model->row_lower_[row]
                        : model->row_upper_[row];
  double rowLower = implRowDualUpper[row] < -options->dual_feasibility_tolerance
                        ? model->row_upper_[row]
                        : model->row_lower_[row];

  assert(rowLower != kHighsInf);
  assert(rowUpper != -kHighsInf);

  if (rowUpper != kHighsInf) {
    // get minimal value of other columns in the row
    double residualMinAct =
        impliedRowBounds.getResidualSumLowerOrig(row, col, val);
    if (residualMinAct != -kHighsInf) {
      double impliedBound =
          double((HighsCDouble(rowUpper) - residualMinAct) / val);

      if (std::abs(impliedBound) * kHighsTiny <= primal_feastol) {
        if (val > 0) {
          // bound is an upper bound
          // check if we may round the bound due to integrality restrictions
          if (mipsolver != nullptr) {
            if (model->integrality_[col] != HighsVarType::kContinuous) {
              double roundedBound = std::floor(impliedBound + primal_feastol);

              if (roundedBound < model->col_upper_[col])
                changeColUpper(col, roundedBound);
            }

            if (mipsolver->mipdata_->postSolveStack.getOrigRowIndex(row) >=
                mipsolver->orig_model_->num_row_) {
              if (impliedBound < model->col_upper_[col] - 1000 * primal_feastol)
                changeColUpper(col, impliedBound);

              impliedBound = kHighsInf;
            }
          }

          // only tighten bound if it is tighter by a wide enough margin
          if (impliedBound < implColUpper[col] - 1000 * primal_feastol)
            changeImplColUpper(col, impliedBound, row);
        } else {
          // bound is a lower bound
          // check if we may round the bound due to integrality restrictions
          if (mipsolver != nullptr) {
            if (model->integrality_[col] != HighsVarType::kContinuous) {
              double roundedBound = std::ceil(impliedBound - primal_feastol);

              if (roundedBound > model->col_lower_[col])
                changeColLower(col, roundedBound);
            }

            // do not use the implied bound if this a not a model row, since the
            // row can be removed and should not be used, e.g., to identify a
            // column as implied free
            if (mipsolver->mipdata_->postSolveStack.getOrigRowIndex(row) >=
                mipsolver->orig_model_->num_row_) {
              if (impliedBound > model->col_lower_[col] + 1000 * primal_feastol)
                changeColLower(col, impliedBound);

              impliedBound = -kHighsInf;
            }
          }

          // only tighten bound if it is tighter by a wide enough margin
          if (impliedBound > implColLower[col] + 1000 * primal_feastol)
            changeImplColLower(col, impliedBound, row);
        }
      }
    }
  }

  if (rowLower != -kHighsInf) {
    // get maximal value of other columns in the row
    double residualMaxAct =
        impliedRowBounds.getResidualSumUpperOrig(row, col, val);
    if (residualMaxAct != kHighsInf) {
      double impliedBound =
          double((HighsCDouble(rowLower) - residualMaxAct) / val);

      if (std::abs(impliedBound) * kHighsTiny <= primal_feastol) {
        if (val > 0) {
          // bound is a lower bound
          // check if we may round the bound due to integrality restrictions
          if (mipsolver != nullptr) {
            if (model->integrality_[col] != HighsVarType::kContinuous) {
              double roundedBound = std::ceil(impliedBound - primal_feastol);

              // change bounds of integers immediately
              if (roundedBound > model->col_lower_[col])
                changeColLower(col, roundedBound);
            }

            if (mipsolver->mipdata_->postSolveStack.getOrigRowIndex(row) >=
                mipsolver->orig_model_->num_row_) {
              if (impliedBound > model->col_lower_[col] + 1000 * primal_feastol)
                changeColLower(col, impliedBound);

              impliedBound = -kHighsInf;
            }
          }

          // only tighten bound if it is tighter by a wide enough margin
          if (impliedBound > implColLower[col] + 1000 * primal_feastol)
            changeImplColLower(col, impliedBound, row);
        } else {
          // bound is an upper bound
          // check if we may round the bound due to integrality restrictions
          if (mipsolver != nullptr) {
            if (model->integrality_[col] != HighsVarType::kContinuous) {
              double roundedBound = std::floor(impliedBound + primal_feastol);

              // change bounds of integers immediately
              if (roundedBound < model->col_upper_[col])
                changeColUpper(col, roundedBound);
            }

            if (mipsolver->mipdata_->postSolveStack.getOrigRowIndex(row) >=
                mipsolver->orig_model_->num_row_) {
              if (impliedBound < model->col_upper_[col] - 1000 * primal_feastol)
                changeColUpper(col, impliedBound);

              impliedBound = kHighsInf;
            }
          }

          // only tighten bound if it is tighter by a wide enough margin
          if (impliedBound < implColUpper[col] - 1000 * primal_feastol)
            changeImplColUpper(col, impliedBound, row);
        }
      }
    }
  }
}

HighsInt HPresolve::findNonzero(HighsInt row, HighsInt col) {
  if (rowroot[row] == -1) return -1;

  auto get_row_left = [&](HighsInt pos) -> HighsInt& { return ARleft[pos]; };
  auto get_row_right = [&](HighsInt pos) -> HighsInt& { return ARright[pos]; };
  auto get_row_key = [&](HighsInt pos) { return Acol[pos]; };
  rowroot[row] =
      highs_splay(col, rowroot[row], get_row_left, get_row_right, get_row_key);

  if (Acol[rowroot[row]] == col) return rowroot[row];

  return -1;
}

void HPresolve::shrinkProblem(HighsPostsolveStack& postsolve_stack) {
  HighsInt oldNumCol = model->num_col_;
  model->num_col_ = 0;
  std::vector<HighsInt> newColIndex(oldNumCol);
  for (HighsInt i = 0; i != oldNumCol; ++i) {
    if (colDeleted[i])
      newColIndex[i] = -1;
    else {
      newColIndex[i] = model->num_col_++;
      model->col_cost_[newColIndex[i]] = model->col_cost_[i];
      model->col_lower_[newColIndex[i]] = model->col_lower_[i];
      model->col_upper_[newColIndex[i]] = model->col_upper_[i];
      assert(!std::isnan(model->col_lower_[newColIndex[i]]));
      assert(!std::isnan(model->col_upper_[newColIndex[i]]));
      model->integrality_[newColIndex[i]] = model->integrality_[i];
      implColLower[newColIndex[i]] = implColLower[i];
      implColUpper[newColIndex[i]] = implColUpper[i];
      colLowerSource[newColIndex[i]] = colLowerSource[i];
      colUpperSource[newColIndex[i]] = colUpperSource[i];
      colhead[newColIndex[i]] = colhead[i];
      colsize[newColIndex[i]] = colsize[i];
      if ((HighsInt)model->col_names_.size() > 0)
        model->col_names_[newColIndex[i]] = std::move(model->col_names_[i]);
      changedColFlag[newColIndex[i]] = changedColFlag[i];
    }
  }
  colDeleted.assign(model->num_col_, false);
  model->col_cost_.resize(model->num_col_);
  model->col_lower_.resize(model->num_col_);
  model->col_upper_.resize(model->num_col_);
  model->integrality_.resize(model->num_col_);
  implColLower.resize(model->num_col_);
  implColUpper.resize(model->num_col_);
  colLowerSource.resize(model->num_col_);
  colUpperSource.resize(model->num_col_);
  colhead.resize(model->num_col_);
  colsize.resize(model->num_col_);
  if ((HighsInt)model->col_names_.size() > 0)
    model->col_names_.resize(model->num_col_);
  changedColFlag.resize(model->num_col_);
  numDeletedCols = 0;
  HighsInt oldNumRow = model->num_row_;
  model->num_row_ = 0;
  std::vector<HighsInt> newRowIndex(oldNumRow);
  for (HighsInt i = 0; i != oldNumRow; ++i) {
    if (rowDeleted[i])
      newRowIndex[i] = -1;
    else {
      newRowIndex[i] = model->num_row_++;
      model->row_lower_[newRowIndex[i]] = model->row_lower_[i];
      model->row_upper_[newRowIndex[i]] = model->row_upper_[i];
      assert(!std::isnan(model->row_lower_[newRowIndex[i]]));
      assert(!std::isnan(model->row_upper_[newRowIndex[i]]));
      rowDualLower[newRowIndex[i]] = rowDualLower[i];
      rowDualUpper[newRowIndex[i]] = rowDualUpper[i];
      implRowDualLower[newRowIndex[i]] = implRowDualLower[i];
      implRowDualUpper[newRowIndex[i]] = implRowDualUpper[i];
      rowDualLowerSource[newRowIndex[i]] = rowDualLowerSource[i];
      rowDualUpperSource[newRowIndex[i]] = rowDualUpperSource[i];
      rowroot[newRowIndex[i]] = rowroot[i];
      rowsize[newRowIndex[i]] = rowsize[i];
      rowsizeInteger[newRowIndex[i]] = rowsizeInteger[i];
      rowsizeImplInt[newRowIndex[i]] = rowsizeImplInt[i];
      if ((HighsInt)model->row_names_.size() > 0)
        model->row_names_[newRowIndex[i]] = std::move(model->row_names_[i]);
      changedRowFlag[newRowIndex[i]] = changedRowFlag[i];
    }
  }

  for (HighsInt i = 0; i != model->num_col_; ++i) {
    if (colLowerSource[i] != -1)
      colLowerSource[i] = newRowIndex[colLowerSource[i]];
    if (colUpperSource[i] != -1)
      colUpperSource[i] = newRowIndex[colUpperSource[i]];
  }

  for (HighsInt i = 0; i != model->num_row_; ++i) {
    if (rowDualLowerSource[i] != -1)
      rowDualLowerSource[i] = newColIndex[rowDualLowerSource[i]];
    if (rowDualUpperSource[i] != -1)
      rowDualUpperSource[i] = newColIndex[rowDualUpperSource[i]];
  }

  rowDeleted.assign(model->num_row_, false);
  model->row_lower_.resize(model->num_row_);
  model->row_upper_.resize(model->num_row_);
  rowDualLower.resize(model->num_row_);
  rowDualUpper.resize(model->num_row_);
  implRowDualLower.resize(model->num_row_);
  implRowDualUpper.resize(model->num_row_);
  rowDualLowerSource.resize(model->num_row_);
  rowDualUpperSource.resize(model->num_row_);
  rowroot.resize(model->num_row_);
  rowsize.resize(model->num_row_);
  rowsizeInteger.resize(model->num_row_);
  rowsizeImplInt.resize(model->num_row_);
  if ((HighsInt)model->row_names_.size() > 0)
    model->row_names_.resize(model->num_row_);
  changedRowFlag.resize(model->num_row_);

  numDeletedRows = 0;
  postsolve_stack.compressIndexMaps(newRowIndex, newColIndex);
  impliedRowBounds.shrink(newRowIndex, model->num_row_);
  impliedDualRowBounds.shrink(newColIndex, model->num_col_);

  HighsInt numNnz = Avalue.size();
  for (HighsInt i = 0; i != numNnz; ++i) {
    if (Avalue[i] == 0) continue;
    assert(newColIndex[Acol[i]] != -1);
    assert(newRowIndex[Arow[i]] != -1);
    Acol[i] = newColIndex[Acol[i]];
    Arow[i] = newRowIndex[Arow[i]];
  }

  // update index sets
  for (HighsInt& singCol : singletonColumns) singCol = newColIndex[singCol];
  singletonColumns.erase(
      std::remove(singletonColumns.begin(), singletonColumns.end(), -1),
      singletonColumns.end());

  for (HighsInt& chgCol : changedColIndices) chgCol = newColIndex[chgCol];
  changedColIndices.erase(
      std::remove(changedColIndices.begin(), changedColIndices.end(), -1),
      changedColIndices.end());

  for (HighsInt& singRow : singletonRows) singRow = newRowIndex[singRow];
  singletonRows.erase(
      std::remove(singletonRows.begin(), singletonRows.end(), -1),
      singletonRows.end());

  for (HighsInt& chgRow : changedRowIndices) chgRow = newRowIndex[chgRow];
  changedRowIndices.erase(
      std::remove(changedRowIndices.begin(), changedRowIndices.end(), -1),
      changedRowIndices.end());

  for (auto& rowColPair : substitutionOpportunities) {
    rowColPair.first = newRowIndex[rowColPair.first];
    rowColPair.second = newColIndex[rowColPair.second];
  }
  substitutionOpportunities.erase(
      std::remove_if(substitutionOpportunities.begin(),
                     substitutionOpportunities.end(),
                     [&](const std::pair<HighsInt, HighsInt>& p) {
                       return p.first == -1 || p.second == -1;
                     }),
      substitutionOpportunities.end());

  // todo remove equation set and replace with a vector of doubleton eqs
  equations.clear();
  eqiters.assign(model->num_row_, equations.end());
  for (HighsInt i = 0; i != model->num_row_; ++i) {
    if (model->row_lower_[i] == model->row_upper_[i])
      eqiters[i] = equations.emplace(rowsize[i], i).first;
  }

  if (mipsolver != nullptr) {
    mipsolver->mipdata_->rowMatrixSet = false;
    mipsolver->mipdata_->objectiveFunction = HighsObjectiveFunction(*mipsolver);
    mipsolver->mipdata_->domain = HighsDomain(*mipsolver);
    mipsolver->mipdata_->cliquetable.rebuild(model->num_col_, postsolve_stack,
                                             mipsolver->mipdata_->domain,
                                             newColIndex, newRowIndex);
    mipsolver->mipdata_->implications.rebuild(model->num_col_, newColIndex,
                                              newRowIndex);
    mipsolver->mipdata_->cutpool =
        HighsCutPool(mipsolver->model_->num_col_,
                     mipsolver->options_mip_->mip_pool_age_limit,
                     mipsolver->options_mip_->mip_pool_soft_limit);
    mipsolver->mipdata_->conflictPool =
        HighsConflictPool(5 * mipsolver->options_mip_->mip_pool_age_limit,
                          mipsolver->options_mip_->mip_pool_soft_limit);

    for (HighsInt i = 0; i != oldNumCol; ++i)
      if (newColIndex[i] != -1) numProbes[newColIndex[i]] = numProbes[i];

    mipsolver->mipdata_->debugSolution.shrink(newColIndex);
    numProbes.resize(model->num_col_);
    // Need to set the constraint matrix dimensions
    model->setMatrixDimensions();
  }
  // Need to set the constraint matrix dimensions
  model->setMatrixDimensions();
}

HPresolve::Result HPresolve::dominatedColumns(
    HighsPostsolveStack& postsolve_stack) {
  std::vector<std::pair<uint32_t, uint32_t>> signatures(model->num_col_);

  auto isBinary = [&](HighsInt i) {
    return model->integrality_[i] == HighsVarType::kInteger &&
           model->col_lower_[i] == 0.0 && model->col_upper_[i] == 1.0;
  };

  auto addSignature = [&](HighsInt row, HighsInt col, uint32_t rowLowerFinite,
                          uint32_t rowUpperFinite) {
    HighsInt rowHashedPos = (HighsHashHelpers::hash(row) >> 59);
    assert(rowHashedPos < 32);
    signatures[col].first |= rowLowerFinite << rowHashedPos;
    signatures[col].second |= rowUpperFinite << rowHashedPos;
  };

  auto checkDomination = [&](HighsInt scalj, HighsInt j, HighsInt scalk,
                             HighsInt k) {
    // rule out domination from integers to continuous variables
    if (model->integrality_[j] == HighsVarType::kInteger &&
        model->integrality_[k] != HighsVarType::kInteger)
      return false;

    // check the signatures
    uint32_t sjMinus = signatures[j].first;
    uint32_t sjPlus = signatures[j].second;
    if (scalj == -1) std::swap(sjPlus, sjMinus);

    uint32_t skMinus = signatures[k].first;
    uint32_t skPlus = signatures[k].second;
    if (scalk == -1) std::swap(skPlus, skMinus);

    // the set of rows with a negative coefficient must be a superset of the
    // dominated columns set of rows with a negative coefficient
    if ((~sjMinus & skMinus) != 0) return false;

    // the set of rows with a positive coefficient must be a subset of the
    // dominated columns set of rows with a positive coefficient
    if ((sjPlus & ~skPlus) != 0) return false;

    // next check if the columns cost allows for domination
    double cj = scalj * model->col_cost_[j];
    double ck = scalk * model->col_cost_[k];

    // the dominating columns cost must be smaller or equal to the dominated
    // columns cost
    if (cj > ck + options->small_matrix_value) return false;

    // finally check the column vectors
    for (const HighsSliceNonzero& nonz : getColumnVector(j)) {
      HighsInt row = nonz.index();
      double aj = scalj * nonz.value();

      HighsInt akPos = findNonzero(row, k);
      double ak = scalk * (akPos == -1 ? 0.0 : Avalue[akPos]);

      if (model->row_lower_[row] != -kHighsInf &&
          model->row_upper_[row] != kHighsInf) {
        // the row is an equality or ranged row, therefore the coefficients must
        // be parallel, otherwise one of the inequalities given by the row rules
        // out domination
        if (std::abs(aj - ak) > options->small_matrix_value) return false;
        continue;
      }

      // normalize row to a <= constraint
      if (model->row_upper_[row] == kHighsInf) {
        aj = -aj;
        ak = -ak;
      }

      // the coefficient of the dominating column needs to be smaller than or
      // equal to the coefficient of the dominated column
      if (aj > ak + options->small_matrix_value) return false;
    }

    // check row only occuring in the column vector of k
    for (const HighsSliceNonzero& nonz : getColumnVector(k)) {
      HighsInt row = nonz.index();
      double ak = scalk * nonz.value();

      HighsInt ajPos = findNonzero(row, j);
      // only rows in which aj does not occur are left to check
      if (ajPos != -1) continue;
      double aj = 0.0;

      if (model->row_lower_[row] != -kHighsInf &&
          model->row_upper_[row] != kHighsInf) {
        // the row is an equality or ranged row, therefore the coefficients must
        // be parallel, otherwise one of the inequalities given by the row rules
        // out domination
        if (std::abs(aj - ak) > options->small_matrix_value) return false;
        continue;
      }

      // normalize row to a <= constraint
      if (model->row_upper_[row] == kHighsInf) {
        aj = -aj;
        ak = -ak;
      }

      // the coefficient of the dominating column needs to be smaller than or
      // equal to the coefficient of the dominated column
      if (aj > ak + options->small_matrix_value) return false;
    }

    return true;
  };

  HighsInt numNz = Avalue.size();
  for (HighsInt i = 0; i < numNz; ++i) {
    if (Avalue[i] == 0) continue;

    HighsInt row = Arow[i];
    HighsInt col = Acol[i];
    bool rowLowerFinite = model->row_lower_[row] != -kHighsInf;
    bool rowUpperFinite = model->row_upper_[row] != kHighsInf;

    if (Avalue[i] > 0)
      addSignature(row, col, rowLowerFinite, rowUpperFinite);
    else
      addSignature(row, col, rowUpperFinite, rowLowerFinite);
  }

  HighsInt numFixedCols = 0;
  for (HighsInt j = 0; j < model->num_col_; ++j) {
    if (colDeleted[j]) continue;
    bool upperImplied = isUpperImplied(j);
    bool lowerImplied = isLowerImplied(j);
    bool hasPosCliques = false;
    bool hasNegCliques = false;
    bool colIsBinary = isBinary(j);
    if (colIsBinary) {
      hasPosCliques = mipsolver->mipdata_->cliquetable.numCliques(j, 1) > 0;
      hasNegCliques = mipsolver->mipdata_->cliquetable.numCliques(j, 0) > 0;
    } else if (!upperImplied && !lowerImplied)
      continue;

    HighsInt oldNumFixed = numFixedCols;

    HighsInt bestRowPlus = -1;
    HighsInt bestRowPlusLen = kHighsIInf;
    HighsInt bestRowPlusScale = 0;
    double ajBestRowPlus = 0.0;
    HighsInt bestRowMinus = -1;
    HighsInt bestRowMinusLen = kHighsIInf;
    HighsInt bestRowMinusScale = 0;
    double ajBestRowMinus = 0.0;

    double worstCaseLb = -kHighsInf;
    double worstCaseUb = kHighsInf;

    bool checkPosRow = upperImplied || colIsBinary;
    bool checkNegRow = lowerImplied || colIsBinary;
    for (const HighsSliceNonzero& nonz : getColumnVector(j)) {
      HighsInt row = nonz.index();
      HighsInt scale = model->row_upper_[row] != kHighsInf ? 1 : -1;

      if (colIsBinary) {
        if (model->row_upper_[row] != kHighsInf) {
          if (model->col_cost_[j] >= 0.0 && nonz.value() < 0.0) {
            double maxresact =
                impliedRowBounds.getResidualSumUpper(row, j, nonz.value());
            double wcBound =
                (model->row_upper_[row] - maxresact) / nonz.value();
            worstCaseLb = std::max(wcBound, worstCaseLb);
          } else if (model->col_cost_[j] <= 0.0 && nonz.value() > 0.0) {
            double maxresact =
                impliedRowBounds.getResidualSumUpper(row, j, nonz.value());
            double wcBound =
                (model->row_upper_[row] - maxresact) / nonz.value();
            worstCaseUb = std::min(wcBound, worstCaseUb);
          }
        }

        if (model->row_lower_[row] != -kHighsInf) {
          if (model->col_cost_[j] >= 0.0 && nonz.value() > 0.0) {
            double minresact =
                impliedRowBounds.getResidualSumLower(row, j, nonz.value());
            double wcBound =
                (model->row_lower_[row] - minresact) / nonz.value();
            worstCaseLb = std::max(wcBound, worstCaseLb);
          } else if (model->col_cost_[j] <= 0.0 && nonz.value() < 0.0) {
            double minresact =
                impliedRowBounds.getResidualSumLower(row, j, nonz.value());
            double wcBound =
                (model->row_lower_[row] - minresact) / nonz.value();
            worstCaseUb = std::min(wcBound, worstCaseUb);
          }
        }
      }

      double val = scale * nonz.value();
      if (checkPosRow && val > 0.0 && rowsize[row] < bestRowPlusLen) {
        bestRowPlus = row;
        bestRowPlusLen = rowsize[row];
        bestRowPlusScale = scale;
        ajBestRowPlus = val;
      }

      if (checkNegRow && val < 0.0 && rowsize[row] < bestRowMinusLen) {
        bestRowMinus = row;
        bestRowMinusLen = rowsize[row];
        bestRowMinusScale = scale;
        ajBestRowMinus = val;
      }
    }

    if (colIsBinary) {
      if (model->col_cost_[j] >= 0.0 && worstCaseLb <= 1 + primal_feastol) {
        upperImplied = true;
        if (!lowerImplied && bestRowMinus != -1) {
          storeRow(bestRowMinus);

          bool isEqOrRangedRow =
              model->row_lower_[bestRowMinus] != -kHighsInf &&
              model->row_upper_[bestRowMinus] != kHighsInf;

          for (const HighsSliceNonzero& nonz : getStoredRow()) {
            HighsInt k = nonz.index();
            if (k == j || colDeleted[k]) continue;

            double ak = nonz.value() * bestRowMinusScale;

            if (-ajBestRowMinus <= -ak + options->small_matrix_value &&
                (!isEqOrRangedRow ||
                 -ajBestRowMinus >= -ak - options->small_matrix_value) &&
                checkDomination(-1, j, -1, k)) {
              // case (iii)  lb(x_j) = -inf, -x_j > -x_k: set x_k = ub(x_k)
              ++numFixedCols;
              fixColToLower(postsolve_stack, j);
              HPRESOLVE_CHECKED_CALL(removeRowSingletons(postsolve_stack));
              break;
            } else if (-ajBestRowMinus <= ak + options->small_matrix_value &&
                       (!isEqOrRangedRow ||
                        -ajBestRowMinus >= ak - options->small_matrix_value) &&
                       checkDomination(-1, j, 1, k)) {
              // case (iv)  lb(x_j) = -inf, -x_j > x_k: set x_k = lb(x_k)
              ++numFixedCols;
              fixColToLower(postsolve_stack, j);
              HPRESOLVE_CHECKED_CALL(removeRowSingletons(postsolve_stack));
              break;
            }
          }

          if (colDeleted[j]) {
            HPRESOLVE_CHECKED_CALL(removeDoubletonEquations(postsolve_stack));
            continue;
          }
        }
      }

      if (model->col_cost_[j] <= 0.0 && worstCaseUb >= -primal_feastol) {
        lowerImplied = true;
        if (!upperImplied && bestRowPlus != -1) {
          storeRow(bestRowPlus);
          bool isEqOrRangedRow = model->row_lower_[bestRowPlus] != -kHighsInf &&
                                 model->row_upper_[bestRowPlus] != kHighsInf;
          for (const HighsSliceNonzero& nonz : getStoredRow()) {
            HighsInt k = nonz.index();
            if (k == j || colDeleted[k]) continue;

            double ak = nonz.value() * bestRowPlusScale;

            if (ajBestRowPlus <= ak + options->small_matrix_value &&
                (!isEqOrRangedRow ||
                 ajBestRowPlus >= ak - options->small_matrix_value) &&
                checkDomination(1, j, 1, k)) {
              // case (i)  ub(x_j) = inf, x_j > x_k: set x_k = lb(x_k)
              ++numFixedCols;
              fixColToUpper(postsolve_stack, j);
              HPRESOLVE_CHECKED_CALL(removeRowSingletons(postsolve_stack));
              break;
            } else if (ajBestRowPlus <= -ak + options->small_matrix_value &&
                       (!isEqOrRangedRow ||
                        ajBestRowPlus >= -ak - options->small_matrix_value) &&
                       checkDomination(1, j, -1, k)) {
              // case (ii)  ub(x_j) = inf, x_j > -x_k: set x_k = ub(x_k)
              ++numFixedCols;
              fixColToUpper(postsolve_stack, j);
              HPRESOLVE_CHECKED_CALL(removeRowSingletons(postsolve_stack));
              break;
            }
          }

          if (colDeleted[j]) {
            HPRESOLVE_CHECKED_CALL(removeDoubletonEquations(postsolve_stack));
            continue;
          }
        }
      }

      if (!upperImplied && !hasPosCliques) bestRowPlus = -1;

      if (!lowerImplied && !hasNegCliques) bestRowMinus = -1;
    }

    if (bestRowPlus != -1) {
      assert(upperImplied || hasPosCliques);
      storeRow(bestRowPlus);
      bool isEqOrRangedRow = model->row_lower_[bestRowPlus] != -kHighsInf &&
                             model->row_upper_[bestRowPlus] != kHighsInf;
      for (const HighsSliceNonzero& nonz : getStoredRow()) {
        HighsInt k = nonz.index();
        if (k == j || colDeleted[k]) continue;

        double ak = nonz.value() * bestRowPlusScale;

        if (model->col_lower_[k] != -kHighsInf &&
            (upperImplied || mipsolver->mipdata_->cliquetable.haveCommonClique(
                                 HighsCliqueTable::CliqueVar(j, 1),
                                 HighsCliqueTable::CliqueVar(k, 1))) &&
            ajBestRowPlus <= ak + options->small_matrix_value &&
            (!isEqOrRangedRow ||
             ajBestRowPlus >= ak - options->small_matrix_value) &&
            checkDomination(1, j, 1, k)) {
          // case (i)  ub(x_j) = inf, x_j > x_k: set x_k = lb(x_k)
          ++numFixedCols;
          fixColToLower(postsolve_stack, k);
          HPRESOLVE_CHECKED_CALL(removeRowSingletons(postsolve_stack));
        } else if (model->col_upper_[k] != kHighsInf &&
                   (upperImplied ||
                    mipsolver->mipdata_->cliquetable.haveCommonClique(
                        HighsCliqueTable::CliqueVar(j, 1),
                        HighsCliqueTable::CliqueVar(k, 0))) &&
                   ajBestRowPlus <= -ak + options->small_matrix_value &&
                   (!isEqOrRangedRow ||
                    ajBestRowPlus >= -ak - options->small_matrix_value) &&
                   checkDomination(1, j, -1, k)) {
          // case (ii)  ub(x_j) = inf, x_j > -x_k: set x_k = ub(x_k)
          ++numFixedCols;
          fixColToUpper(postsolve_stack, k);
          HPRESOLVE_CHECKED_CALL(removeRowSingletons(postsolve_stack));
        }
      }
    }

    if (bestRowMinus != -1) {
      assert(lowerImplied || hasNegCliques);
      storeRow(bestRowMinus);

      bool isEqOrRangedRow = model->row_lower_[bestRowMinus] != -kHighsInf &&
                             model->row_upper_[bestRowMinus] != kHighsInf;

      for (const HighsSliceNonzero& nonz : getStoredRow()) {
        HighsInt k = nonz.index();
        if (k == j || colDeleted[k]) continue;

        double ak = nonz.value() * bestRowMinusScale;

        if (model->col_upper_[k] != kHighsInf &&
            (lowerImplied || mipsolver->mipdata_->cliquetable.haveCommonClique(
                                 HighsCliqueTable::CliqueVar(j, 0),
                                 HighsCliqueTable::CliqueVar(k, 0))) &&
            -ajBestRowMinus <= -ak + options->small_matrix_value &&
            (!isEqOrRangedRow ||
             -ajBestRowMinus >= -ak - options->small_matrix_value) &&
            checkDomination(-1, j, -1, k)) {
          // case (iii)  lb(x_j) = -inf, -x_j > -x_k: set x_k = ub(x_k)
          ++numFixedCols;
          fixColToUpper(postsolve_stack, k);
          HPRESOLVE_CHECKED_CALL(removeRowSingletons(postsolve_stack));
        } else if (model->col_lower_[k] != -kHighsInf &&
                   (lowerImplied ||
                    mipsolver->mipdata_->cliquetable.haveCommonClique(
                        HighsCliqueTable::CliqueVar(j, 0),
                        HighsCliqueTable::CliqueVar(k, 1))) &&
                   -ajBestRowMinus <= ak + options->small_matrix_value &&
                   (!isEqOrRangedRow ||
                    -ajBestRowMinus >= ak - options->small_matrix_value) &&
                   checkDomination(-1, j, 1, k)) {
          // case (iv)  lb(x_j) = -inf, -x_j > x_k: set x_k = lb(x_k)
          ++numFixedCols;
          fixColToLower(postsolve_stack, k);
          HPRESOLVE_CHECKED_CALL(removeRowSingletons(postsolve_stack));
        }
      }
    }

    if (numFixedCols != oldNumFixed)
      HPRESOLVE_CHECKED_CALL(removeDoubletonEquations(postsolve_stack));
  }

  if (numFixedCols)
    highsLogDev(options->log_options, HighsLogType::kInfo,
                "Fixed %d dominated columns\n", numFixedCols);

  return Result::kOk;
}

HPresolve::Result HPresolve::runProbing(HighsPostsolveStack& postsolve_stack) {
  probingEarlyAbort = false;
  if (numDeletedCols + numDeletedRows != 0) shrinkProblem(postsolve_stack);

  toCSC(model->a_matrix_.value_, model->a_matrix_.index_,
        model->a_matrix_.start_);
  fromCSC(model->a_matrix_.value_, model->a_matrix_.index_,
          model->a_matrix_.start_);

  mipsolver->mipdata_->cliquetable.setMaxEntries(numNonzeros());

  // first tighten all bounds if they have an implied bound that is tighter
  // thatn their column bound before probing this is not done for continuous
  // columns since it may allow stronger dual presolve and more aggregations
  double hugeBound = primal_feastol / kHighsTiny;
  for (HighsInt i = 0; i != model->num_col_; ++i) {
    if (model->col_lower_[i] >= implColLower[i] &&
        model->col_upper_[i] <= implColUpper[i])
      continue;

    if (std::abs(implColLower[i]) <= hugeBound) {
      double newLb = implColLower[i];
      if (newLb > model->col_lower_[i]) changeColLower(i, newLb);
    }

    if (std::abs(implColUpper[i]) <= hugeBound) {
      double newUb = implColUpper[i];
      if (newUb < model->col_upper_[i]) changeColUpper(i, newUb);
    }
  }

  HighsInt oldNumProbed = numProbed;

  mipsolver->mipdata_->setupDomainPropagation();
  HighsDomain& domain = mipsolver->mipdata_->domain;

  domain.propagate();
  if (domain.infeasible()) return Result::kPrimalInfeasible;
  HighsCliqueTable& cliquetable = mipsolver->mipdata_->cliquetable;
  HighsImplications& implications = mipsolver->mipdata_->implications;
  bool firstCall = !mipsolver->mipdata_->cliquesExtracted;
  mipsolver->mipdata_->cliquesExtracted = true;

  // extract cliques that are part of the formulation every time before probing
  // after the first call we only add cliques that directly correspond to set
  // packing constraints so that the clique merging step can extend/delete them
  if (firstCall) {
    cliquetable.extractCliques(*mipsolver);
    if (domain.infeasible()) return Result::kPrimalInfeasible;

    // during presolve we keep the objective upper bound without the current
    // offset so we need to update it

    if (mipsolver->mipdata_->upper_limit != kHighsInf) {
      double tmpLimit = mipsolver->mipdata_->upper_limit;
      mipsolver->mipdata_->upper_limit = tmpLimit - model->offset_;
      cliquetable.extractObjCliques(*mipsolver);
      mipsolver->mipdata_->upper_limit = tmpLimit;

      if (domain.infeasible()) return Result::kPrimalInfeasible;
    }

    domain.propagate();
    if (domain.infeasible()) return Result::kPrimalInfeasible;
  }

  cliquetable.cleanupFixed(domain);
  if (domain.infeasible()) return Result::kPrimalInfeasible;

  // store binary variables in vector with their number of implications on
  // other binaries
  std::vector<std::tuple<int64_t, HighsInt, HighsInt, HighsInt>> binaries;

  if (!mipsolver->mipdata_->cliquetable.isFull()) {
    binaries.reserve(model->num_col_);
    HighsRandom random(options->random_seed);
    for (HighsInt i = 0; i != model->num_col_; ++i) {
      if (domain.isBinary(i)) {
        HighsInt implicsUp = cliquetable.getNumImplications(i, 1);
        HighsInt implicsDown = cliquetable.getNumImplications(i, 0);
        binaries.emplace_back(
            -std::min(int64_t{5000}, int64_t(implicsUp) * implicsDown) /
                (1.0 + numProbes[i]),
            -std::min(HighsInt{100}, implicsUp + implicsDown), random.integer(),
            i);
      }
    }
  }
  if (!binaries.empty()) {
    // sort variables with many implications on other binaries first
    pdqsort(binaries.begin(), binaries.end());

    size_t numChangedCols = 0;
    while (domain.getChangedCols().size() != numChangedCols) {
      if (domain.isFixed(domain.getChangedCols()[numChangedCols++]))
        ++probingNumDelCol;
    }

    HighsInt numCliquesStart = cliquetable.numCliques();
    HighsInt numImplicsStart = implications.getNumImplications();
    HighsInt numDelStart = probingNumDelCol;

    HighsInt numDel = probingNumDelCol - numDelStart +
                      implications.substitutions.size() +
                      cliquetable.getSubstitutions().size();
    int64_t splayContingent =
        cliquetable.numNeighborhoodQueries +
        std::max(mipsolver->submip ? HighsInt{0} : HighsInt{1000000},
                 100 * numNonzeros());
    HighsInt numFail = 0;
    for (const std::tuple<int64_t, HighsInt, HighsInt, HighsInt>& binvar :
         binaries) {
      HighsInt i = std::get<3>(binvar);

      if (cliquetable.getSubstitution(i) != nullptr) continue;

      if (domain.isBinary(i)) {
        // when a large percentage of columns have been deleted, stop this round
        // of probing
        // if (numDel > std::max(model->num_col_ * 0.2, 1000.)) break;
        if (numDel >
            std::max(1000., (model->num_row_ + model->num_col_) * 0.05)) {
          probingEarlyAbort = true;
          break;
        }

        // break in case of too many new implications to not spent ages in
        // probing
        if (cliquetable.isFull() ||
            cliquetable.numCliques() - numCliquesStart >
                std::max(HighsInt{1000000}, 2 * numNonzeros()) ||
            implications.getNumImplications() - numImplicsStart >
                std::max(HighsInt{1000000}, 2 * numNonzeros()))
          break;

        // if (numProbed % 10 == 0)
        //   printf(
        //       "numprobed=%d  numDel=%d  newcliques=%d
        //       numNeighborhoodQueries=%ld  " "splayContingent=%ld\n",
        //       numProbed, numDel, cliquetable.numCliques() - numCliquesStart,
        //       cliquetable.numNeighborhoodQueries, splayContingent);
        if (cliquetable.numNeighborhoodQueries > splayContingent) break;

        if (probingContingent - numProbed < 0) break;

        HighsInt numBoundChgs = 0;
        HighsInt numNewCliques = -cliquetable.numCliques();
        if (!implications.runProbing(i, numBoundChgs)) continue;
        probingContingent += numBoundChgs;
        numNewCliques += cliquetable.numCliques();
        numNewCliques = std::max(numNewCliques, HighsInt{0});
        while (domain.getChangedCols().size() != numChangedCols) {
          if (domain.isFixed(domain.getChangedCols()[numChangedCols++]))
            ++probingNumDelCol;
        }
        HighsInt newNumDel = probingNumDelCol - numDelStart +
                             implications.substitutions.size() +
                             cliquetable.getSubstitutions().size();

        if (newNumDel > numDel) {
          probingContingent += numDel;
          if (!mipsolver->submip) {
            splayContingent += 1000 * (newNumDel + numDelStart);
            splayContingent += 10000 * numNewCliques;
          }
          numDel = newNumDel;
          numFail = 0;
        } else if (mipsolver->submip || numNewCliques == 0) {
          splayContingent -= 1000 * numFail;
          ++numFail;
        } else {
          splayContingent += 10000 * numNewCliques;
          numFail = 0;
        }

        ++numProbed;
        numProbes[i] += 1;

        // printf("nprobed: %" HIGHSINT_FORMAT ", numCliques: %" HIGHSINT_FORMAT
        // "\n", nprobed,
        //       cliquetable.numCliques());
        if (domain.infeasible()) {
          return Result::kPrimalInfeasible;
        }
      }
    }

    cliquetable.cleanupFixed(domain);

    if (!firstCall) cliquetable.extractCliques(*mipsolver, false);
    cliquetable.runCliqueMerging(domain);

    // apply changes from probing

    // first delete redundant clique inequalities
    for (HighsInt delrow : cliquetable.getDeletedRows())
      if (!rowDeleted[delrow]) removeRow(delrow);
    cliquetable.getDeletedRows().clear();

    // add nonzeros from clique lifting before removign fixed variables, since
    // this might lead to stronger constraint sides
    std::vector<std::pair<HighsInt, HighsCliqueTable::CliqueVar>>&
        extensionvars = cliquetable.getCliqueExtensions();
    HighsInt addednnz = extensionvars.size();
    for (std::pair<HighsInt, HighsCliqueTable::CliqueVar> cliqueextension :
         extensionvars) {
      if (rowDeleted[cliqueextension.first]) {
        --addednnz;
        continue;
      }
      double val;
      if (cliqueextension.second.val == 0) {
        model->row_lower_[cliqueextension.first] -= 1;
        model->row_upper_[cliqueextension.first] -= 1;
        val = -1.0;
      } else
        val = 1.0;
      addToMatrix(cliqueextension.first, cliqueextension.second.col, val);
    }
    extensionvars.clear();

    // now remove fixed columns and tighten domains
    for (HighsInt i = 0; i != model->num_col_; ++i) {
      if (colDeleted[i]) continue;
      if (model->col_lower_[i] < domain.col_lower_[i])
        changeColLower(i, domain.col_lower_[i]);
      if (model->col_upper_[i] > domain.col_upper_[i])
        changeColUpper(i, domain.col_upper_[i]);
      if (domain.isFixed(i)) {
        postsolve_stack.removedFixedCol(i, model->col_lower_[i], 0.0,
                                        HighsEmptySlice());
        removeFixedCol(i);
      }
      HPRESOLVE_CHECKED_CALL(checkLimits(postsolve_stack));
    }

    // finally apply substitutions
    HPRESOLVE_CHECKED_CALL(applyConflictGraphSubstitutions(postsolve_stack));

    highsLogDev(options->log_options, HighsLogType::kInfo,
                "%" HIGHSINT_FORMAT " probing evaluations: %" HIGHSINT_FORMAT
                " deleted rows, %" HIGHSINT_FORMAT
                " deleted "
                "columns, %" HIGHSINT_FORMAT " lifted nonzeros\n",
                numProbed - oldNumProbed, numDeletedRows, numDeletedCols,
                addednnz);
  }

  return checkLimits(postsolve_stack);
}

void HPresolve::addToMatrix(HighsInt row, HighsInt col, double val) {
  HighsInt pos = findNonzero(row, col);

  markChangedRow(row);
  markChangedCol(col);

  if (pos == -1) {
    if (freeslots.empty()) {
      pos = Avalue.size();
      Avalue.push_back(val);
      Arow.push_back(row);
      Acol.push_back(col);
      Anext.push_back(-1);
      Aprev.push_back(-1);
      ARleft.push_back(-1);
      ARright.push_back(-1);
    } else {
      pos = freeslots.back();
      freeslots.pop_back();
      Avalue[pos] = val;
      Arow[pos] = row;
      Acol[pos] = col;
      Aprev[pos] = -1;
    }

    link(pos);
  } else {
    double sum = Avalue[pos] + val;
    if (std::abs(sum) <= options->small_matrix_value) {
      unlink(pos);
    } else {
      // remove implied bounds on the row dual that where implied by this
      // columns dual constraint
      if (rowDualUpperSource[row] == col)
        changeImplRowDualUpper(row, kHighsInf, -1);

      if (rowDualLowerSource[row] == col)
        changeImplRowDualLower(row, -kHighsInf, -1);

      // remove implied bounds on the column that where implied by this row
      if (colUpperSource[col] == row) changeImplColUpper(col, kHighsInf, -1);

      if (colLowerSource[col] == row) changeImplColLower(col, -kHighsInf, -1);

      // remove the locks and contribution to implied (dual) row bounds, then
      // add then again
      impliedRowBounds.remove(row, col, Avalue[pos]);
      impliedDualRowBounds.remove(col, row, Avalue[pos]);
      Avalue[pos] = sum;
      // value not zero, add new contributions and locks with opposite sign
      impliedRowBounds.add(row, col, Avalue[pos]);
      impliedDualRowBounds.add(col, row, Avalue[pos]);
    }
  }
}

HighsTripletListSlice HPresolve::getColumnVector(HighsInt col) const {
  return HighsTripletListSlice(Arow.data(), Avalue.data(), Anext.data(),
                               colhead[col]);
}

HighsTripletTreeSlicePreOrder HPresolve::getRowVector(HighsInt row) const {
  return HighsTripletTreeSlicePreOrder(
      Acol.data(), Avalue.data(), ARleft.data(), ARright.data(), rowroot[row]);
}

HighsTripletTreeSliceInOrder HPresolve::getSortedRowVector(HighsInt row) const {
  return HighsTripletTreeSliceInOrder(Acol.data(), Avalue.data(), ARleft.data(),
                                      ARright.data(), rowroot[row]);
}

void HPresolve::markRowDeleted(HighsInt row) {
  assert(!rowDeleted[row]);

  // remove equations from set of equations
  if (model->row_lower_[row] == model->row_upper_[row] &&
      eqiters[row] != equations.end()) {
    equations.erase(eqiters[row]);
    eqiters[row] = equations.end();
  }

  // prevents row from being added to change vector
  changedRowFlag[row] = true;
  rowDeleted[row] = true;
  ++numDeletedRows;
}

void HPresolve::markColDeleted(HighsInt col) {
  assert(!colDeleted[col]);
  // prevents col from being added to change vector
  changedColFlag[col] = true;
  colDeleted[col] = true;
  ++numDeletedCols;
}

void HPresolve::changeColUpper(HighsInt col, double newUpper) {
  if (model->integrality_[col] != HighsVarType::kContinuous) {
    newUpper = std::floor(newUpper + primal_feastol);
    if (newUpper == model->col_upper_[col]) return;
  }

  double oldUpper = model->col_upper_[col];
  model->col_upper_[col] = newUpper;

  for (const HighsSliceNonzero& nonzero : getColumnVector(col)) {
    impliedRowBounds.updatedVarUpper(nonzero.index(), col, nonzero.value(),
                                     oldUpper);
    markChangedRow(nonzero.index());
  }
}

void HPresolve::changeColLower(HighsInt col, double newLower) {
  if (model->integrality_[col] != HighsVarType::kContinuous) {
    newLower = std::ceil(newLower - primal_feastol);
    if (newLower == model->col_lower_[col]) return;
  }

  double oldLower = model->col_lower_[col];
  model->col_lower_[col] = newLower;
  // printf("tightening lower bound of column %" HIGHSINT_FORMAT " from %.15g to
  // %.15g\n", col,
  //        oldLower, newLower);

  for (const HighsSliceNonzero& nonzero : getColumnVector(col)) {
    impliedRowBounds.updatedVarLower(nonzero.index(), col, nonzero.value(),
                                     oldLower);
    markChangedRow(nonzero.index());
  }
}

void HPresolve::changeRowDualUpper(HighsInt row, double newUpper) {
  double oldUpper = rowDualUpper[row];
  rowDualUpper[row] = newUpper;

  // printf("tightening upper bound of column %" HIGHSINT_FORMAT " from %.15g to
  // %.15g\n", col,
  //        oldUpper, newUpper);
  for (const HighsSliceNonzero& nonzero : getRowVector(row)) {
    impliedDualRowBounds.updatedVarUpper(nonzero.index(), row, nonzero.value(),
                                         oldUpper);
    markChangedCol(nonzero.index());
  }
}

void HPresolve::changeRowDualLower(HighsInt row, double newLower) {
  double oldLower = rowDualLower[row];
  rowDualLower[row] = newLower;
  // printf("tightening lower bound of column %" HIGHSINT_FORMAT " from %.15g to
  // %.15g\n", col,
  //        oldLower, newLower);

  for (const HighsSliceNonzero& nonzero : getRowVector(row)) {
    impliedDualRowBounds.updatedVarLower(nonzero.index(), row, nonzero.value(),
                                         oldLower);
    markChangedCol(nonzero.index());
  }
}

void HPresolve::changeImplColUpper(HighsInt col, double newUpper,
                                   HighsInt originRow) {
  double oldImplUpper = implColUpper[col];
  HighsInt oldUpperSource = colUpperSource[col];
  if (oldImplUpper >= model->col_upper_[col] - primal_feastol &&
      newUpper < model->col_upper_[col] - primal_feastol) {
    // the dual constraint can be considered a >= constraint and was free, or a
    // <= constraint before
    markChangedCol(col);
  }
  bool newImpliedFree =
      isLowerImplied(col) &&
      oldImplUpper > model->col_upper_[col] + primal_feastol &&
      newUpper <= model->col_upper_[col] + primal_feastol;

  // remember the source of this lower bound, so that we can correctly identify
  // weak domination
  colUpperSource[col] = originRow;
  implColUpper[col] = newUpper;

  // if the old and the new implied bound not better than the lower bound
  // nothing
  // needs to be updated
  if (!newImpliedFree &&
      std::min(oldImplUpper, newUpper) >= model->col_upper_[col])
    return;

  for (const HighsSliceNonzero& nonzero : getColumnVector(col)) {
    impliedRowBounds.updatedImplVarUpper(nonzero.index(), col, nonzero.value(),
                                         oldImplUpper, oldUpperSource);
    if (newImpliedFree && isDualImpliedFree(nonzero.index()))
      substitutionOpportunities.emplace_back(nonzero.index(), col);

    markChangedRow(nonzero.index());
  }
}

void HPresolve::changeImplColLower(HighsInt col, double newLower,
                                   HighsInt originRow) {
  double oldImplLower = implColLower[col];
  HighsInt oldLowerSource = colLowerSource[col];
  if (oldImplLower <= model->col_lower_[col] + primal_feastol &&
      newLower > model->col_lower_[col] + primal_feastol) {
    // the dual constraint can additionally be considered a <= constraint and
    // was free, or a
    // >= constraint before
    markChangedCol(col);
  }
  bool newImpliedFree =
      isUpperImplied(col) &&
      oldImplLower < model->col_lower_[col] - primal_feastol &&
      newLower >= model->col_lower_[col] - primal_feastol;

  // remember the source of this lower bound, so that we can correctly identify
  // weak domination
  colLowerSource[col] = originRow;
  implColLower[col] = newLower;

  // if the old and the new implied bound not better than the lower bound
  // nothing needs to be updated
  if (!newImpliedFree &&
      std::max(oldImplLower, newLower) <= model->col_lower_[col])
    return;

  for (const HighsSliceNonzero& nonzero : getColumnVector(col)) {
    impliedRowBounds.updatedImplVarLower(nonzero.index(), col, nonzero.value(),
                                         oldImplLower, oldLowerSource);
    if (newImpliedFree && isDualImpliedFree(nonzero.index()))
      substitutionOpportunities.emplace_back(nonzero.index(), col);

    markChangedRow(nonzero.index());
  }
}

void HPresolve::changeImplRowDualUpper(HighsInt row, double newUpper,
                                       HighsInt originCol) {
  double oldImplUpper = implRowDualUpper[row];
  HighsInt oldUpperSource = rowDualUpperSource[row];

  if (oldImplUpper >= -options->dual_feasibility_tolerance &&
      newUpper < -options->dual_feasibility_tolerance)
    markChangedRow(row);

  bool newDualImplied =
      !isDualImpliedFree(row) &&
      oldImplUpper > rowDualUpper[row] + options->dual_feasibility_tolerance &&
      newUpper <= rowDualUpper[row] + options->dual_feasibility_tolerance;

  // remember the source of this lower bound, so that we can correctly identify
  // weakdomination
  rowDualUpperSource[row] = originCol;
  implRowDualUpper[row] = newUpper;

  // nothing needs to be updated
  if (!newDualImplied && std::min(oldImplUpper, newUpper) >= rowDualUpper[row])
    return;

  for (const HighsSliceNonzero& nonzero : getRowVector(row)) {
    impliedDualRowBounds.updatedImplVarUpper(
        nonzero.index(), row, nonzero.value(), oldImplUpper, oldUpperSource);
    markChangedCol(nonzero.index());

    if (newDualImplied && isImpliedFree(nonzero.index()))
      substitutionOpportunities.emplace_back(row, nonzero.index());
  }
}

void HPresolve::changeImplRowDualLower(HighsInt row, double newLower,
                                       HighsInt originCol) {
  double oldImplLower = implRowDualLower[row];
  HighsInt oldLowerSource = rowDualLowerSource[row];

  if (oldImplLower <= options->dual_feasibility_tolerance &&
      newLower > options->dual_feasibility_tolerance)
    markChangedRow(row);

  bool newDualImplied =
      !isDualImpliedFree(row) &&
      oldImplLower < rowDualLower[row] - options->dual_feasibility_tolerance &&
      newLower >= rowDualLower[row] - options->dual_feasibility_tolerance;

  // remember the source of this lower bound, so that we can correctly identify
  // a weakly domination
  rowDualLowerSource[row] = originCol;
  implRowDualLower[row] = newLower;

  // nothing needs to be updated
  if (!newDualImplied && std::max(oldImplLower, newLower) <= rowDualLower[row])
    return;

  for (const HighsSliceNonzero& nonzero : getRowVector(row)) {
    impliedDualRowBounds.updatedImplVarLower(
        nonzero.index(), row, nonzero.value(), oldImplLower, oldLowerSource);
    markChangedCol(nonzero.index());

    if (newDualImplied && isImpliedFree(nonzero.index()))
      substitutionOpportunities.emplace_back(row, nonzero.index());
  }
}

void HPresolve::scaleMIP(HighsPostsolveStack& postsolve_stack) {
  for (HighsInt i = 0; i < model->num_row_; ++i) {
    if (rowDeleted[i] || rowsize[i] < 1 ||
        rowsizeInteger[i] + rowsizeImplInt[i] == rowsize[i])
      continue;

    storeRow(i);

    double maxAbsVal = 0.0;

    HighsInt rowlen = rowpositions.size();

    for (HighsInt j = 0; j < rowlen; ++j) {
      HighsInt nzPos = rowpositions[j];
      if (model->integrality_[Acol[nzPos]] != HighsVarType::kContinuous)
        continue;

      maxAbsVal = std::max(std::abs(Avalue[nzPos]), maxAbsVal);
    }

    assert(maxAbsVal != 0.0);

    double scale = std::exp2(std::round(-std::log2(maxAbsVal)));
    if (scale == 1.0) continue;

    if (model->row_upper_[i] == kHighsInf) scale = -scale;

    scaleStoredRow(i, scale);
  }

  for (HighsInt i = 0; i < model->num_col_; ++i) {
    if (colDeleted[i] || colsize[i] < 1 ||
        model->integrality_[i] != HighsVarType::kContinuous)
      continue;

    double maxAbsVal = 0;

    for (const HighsSliceNonzero& nonz : getColumnVector(i)) {
      maxAbsVal = std::max(std::abs(nonz.value()), maxAbsVal);
    }

    double scale = std::exp2(std::round(-std::log2(maxAbsVal)));
    if (scale == 1.0) continue;

    transformColumn(postsolve_stack, i, scale, 0.0);
  }
}

HPresolve::Result HPresolve::applyConflictGraphSubstitutions(
    HighsPostsolveStack& postsolve_stack) {
  HighsCliqueTable& cliquetable = mipsolver->mipdata_->cliquetable;
  HighsImplications& implications = mipsolver->mipdata_->implications;
  for (const auto& substitution : implications.substitutions) {
    if (colDeleted[substitution.substcol] || colDeleted[substitution.staycol])
      continue;

    ++probingNumDelCol;

    postsolve_stack.doubletonEquation(-1, substitution.substcol,
                                      substitution.staycol, 1.0,
                                      -substitution.scale, substitution.offset,
                                      model->col_lower_[substitution.substcol],
                                      model->col_upper_[substitution.substcol],
                                      0.0, false, false, HighsEmptySlice());
    markColDeleted(substitution.substcol);
    substitute(substitution.substcol, substitution.staycol, substitution.offset,
               substitution.scale);
    HPRESOLVE_CHECKED_CALL(checkLimits(postsolve_stack));
  }

  implications.substitutions.clear();

  for (HighsCliqueTable::Substitution subst : cliquetable.getSubstitutions()) {
    if (colDeleted[subst.substcol] || colDeleted[subst.replace.col]) continue;

    double scale;
    double offset;

    ++probingNumDelCol;

    if (subst.replace.val == 0) {
      scale = -1.0;
      offset = 1.0;
    } else {
      scale = 1.0;
      offset = 0.0;
    }

    postsolve_stack.doubletonEquation(
        -1, subst.substcol, subst.replace.col, 1.0, -scale, offset,
        model->col_lower_[subst.substcol], model->col_upper_[subst.substcol],
        0.0, false, false, HighsEmptySlice());
    markColDeleted(subst.substcol);
    substitute(subst.substcol, subst.replace.col, offset, scale);
    HPRESOLVE_CHECKED_CALL(checkLimits(postsolve_stack));
  }

  cliquetable.getSubstitutions().clear();

  return Result::kOk;
}

void HPresolve::storeRow(HighsInt row) {
  rowpositions.clear();

  auto rowVec = getSortedRowVector(row);
  auto rowVecEnd = rowVec.end();
  for (auto iter = rowVec.begin(); iter != rowVecEnd; ++iter)
    rowpositions.push_back(iter.position());
}

HighsTripletPositionSlice HPresolve::getStoredRow() const {
  return HighsTripletPositionSlice(Acol.data(), Avalue.data(),
                                   rowpositions.data(), rowpositions.size());
}

void HPresolve::fromCSC(const std::vector<double>& Aval,
                        const std::vector<HighsInt>& Aindex,
                        const std::vector<HighsInt>& Astart) {
  Avalue.clear();
  Acol.clear();
  Arow.clear();

  freeslots.clear();
  colhead.assign(model->num_col_, -1);
  rowroot.assign(model->num_row_, -1);
  colsize.assign(model->num_col_, 0);
  rowsize.assign(model->num_row_, 0);
  rowsizeInteger.assign(model->num_row_, 0);
  rowsizeImplInt.assign(model->num_row_, 0);

  impliedRowBounds.setNumSums(0);
  impliedDualRowBounds.setNumSums(0);
  impliedRowBounds.setBoundArrays(
      model->col_lower_.data(), model->col_upper_.data(), implColLower.data(),
      implColUpper.data(), colLowerSource.data(), colUpperSource.data());
  impliedRowBounds.setNumSums(model->num_row_);
  impliedDualRowBounds.setBoundArrays(
      rowDualLower.data(), rowDualUpper.data(), implRowDualLower.data(),
      implRowDualUpper.data(), rowDualLowerSource.data(),
      rowDualUpperSource.data());
  impliedDualRowBounds.setNumSums(model->num_col_);

  HighsInt ncol = Astart.size() - 1;
  assert(ncol == int(colhead.size()));
  HighsInt nnz = Aval.size();

  Avalue = Aval;
  Acol.reserve(nnz);
  Arow.reserve(nnz);

  for (HighsInt i = 0; i != ncol; ++i) {
    HighsInt collen = Astart[i + 1] - Astart[i];
    Acol.insert(Acol.end(), collen, i);
    Arow.insert(Arow.end(), Aindex.begin() + Astart[i],
                Aindex.begin() + Astart[i + 1]);
  }

  Anext.resize(nnz);
  Aprev.resize(nnz);
  ARleft.resize(nnz);
  ARright.resize(nnz);
  for (HighsInt pos = 0; pos != nnz; ++pos) link(pos);

  if (equations.empty()) {
    eqiters.assign(model->num_row_, equations.end());
    for (HighsInt i = 0; i != model->num_row_; ++i) {
      // register equation
      if (model->row_lower_[i] == model->row_upper_[i])
        eqiters[i] = equations.emplace(rowsize[i], i).first;
    }
  }
}

void HPresolve::fromCSR(const std::vector<double>& ARval,
                        const std::vector<HighsInt>& ARindex,
                        const std::vector<HighsInt>& ARstart) {
  Avalue.clear();
  Acol.clear();
  Arow.clear();

  freeslots.clear();
  colhead.assign(model->num_col_, -1);
  rowroot.assign(model->num_row_, -1);
  colsize.assign(model->num_col_, 0);
  rowsize.assign(model->num_row_, 0);
  rowsizeInteger.assign(model->num_row_, 0);
  rowsizeImplInt.assign(model->num_row_, 0);

  impliedRowBounds.setNumSums(0);
  impliedDualRowBounds.setNumSums(0);
  impliedRowBounds.setBoundArrays(
      model->col_lower_.data(), model->col_upper_.data(), implColLower.data(),
      implColUpper.data(), colLowerSource.data(), colUpperSource.data());
  impliedRowBounds.setNumSums(model->num_row_);
  impliedDualRowBounds.setBoundArrays(
      rowDualLower.data(), rowDualUpper.data(), implRowDualLower.data(),
      implRowDualUpper.data(), rowDualLowerSource.data(),
      rowDualUpperSource.data());
  impliedDualRowBounds.setNumSums(model->num_col_);

  HighsInt nrow = ARstart.size() - 1;
  assert(nrow == int(rowroot.size()));
  HighsInt nnz = ARval.size();

  Avalue = ARval;
  Acol.reserve(nnz);
  Arow.reserve(nnz);
  //  entries.reserve(nnz);

  for (HighsInt i = 0; i != nrow; ++i) {
    HighsInt rowlen = ARstart[i + 1] - ARstart[i];
    Arow.insert(Arow.end(), rowlen, i);
    Acol.insert(Acol.end(), ARindex.begin() + ARstart[i],
                ARindex.begin() + ARstart[i + 1]);
  }

  Anext.resize(nnz);
  Aprev.resize(nnz);
  ARleft.resize(nnz);
  ARright.resize(nnz);
  for (HighsInt pos = 0; pos != nnz; ++pos) link(pos);

  if (equations.empty()) {
    eqiters.assign(nrow, equations.end());
    for (HighsInt i = 0; i != nrow; ++i) {
      // register equation
      if (model->row_lower_[i] == model->row_upper_[i])
        eqiters[i] = equations.emplace(rowsize[i], i).first;
    }
  }
}

HighsInt HPresolve::countFillin(HighsInt row) {
  HighsInt fillin = 0;
  for (HighsInt rowiter : rowpositions) {
    if (findNonzero(row, Acol[rowiter]) == -1) fillin += 1;
  }

  return fillin;
}

bool HPresolve::checkFillin(HighsHashTable<HighsInt, HighsInt>& fillinCache,
                            HighsInt row, HighsInt col) {
  // check numerics against markowitz tolerance
  assert(int(rowpositions.size()) == rowsize[row]);

  // check fillin against max fillin
  HighsInt fillin = -(rowsize[row] + colsize[col] - 1);

#if 1
  // first use fillin for rows where it is already computed
  for (HighsInt coliter = colhead[col]; coliter != -1;
       coliter = Anext[coliter]) {
    if (Arow[coliter] == row) continue;

    auto cachedFillin = fillinCache.find(Arow[coliter]);
    if (cachedFillin == nullptr) continue;

    fillin += (*cachedFillin - 1);
    if (fillin > options->presolve_substitution_maxfillin) return false;
  }

  // iterate over rows of substituted column again to count the fillin for the
  // remaining rows
  for (HighsInt coliter = colhead[col]; coliter != -1;
       coliter = Anext[coliter]) {
    assert(Acol[coliter] == col);

    if (Arow[coliter] == row) continue;

    HighsInt& cachedFillin = fillinCache[Arow[coliter]];

    if (cachedFillin != 0) continue;

    HighsInt rowfillin = countFillin(Arow[coliter]);
    cachedFillin = rowfillin + 1;
    fillin += rowfillin;

    if (fillin > options->presolve_substitution_maxfillin) return false;
    // we count a fillin of 1 if the column is not present in the row and
    // a fillin of zero otherwise. the fillin for the substituted column
    // itself was already counted before the loop so we skip that entry.
  }
#else
  for (HighsInt rowiter : rowpositions) {
    if (rowiter == pos) continue;
    for (coliter = colhead[col]; coliter != -1; coliter = Anext[coliter]) {
      assert(Acol[coliter] == col);

      if (rowiter != coliter &&
          findNonzero(Arow[coliter], Acol[rowiter]) == -1) {
        if (fillin == maxfillin) return false;
        fillin += 1;
      }
    }
  }
#endif

  return true;
}

void HPresolve::transformColumn(HighsPostsolveStack& postsolve_stack,
                                HighsInt col, double scale, double constant) {
  if (mipsolver != nullptr) {
    for (std::pair<const HighsInt, HighsImplications::VarBound>& vbd :
         mipsolver->mipdata_->implications.getVLBs(col)) {
      vbd.second.constant -= constant;
      vbd.second.constant /= scale;
      vbd.second.coef /= scale;
    }

    for (std::pair<const HighsInt, HighsImplications::VarBound>& vbd :
         mipsolver->mipdata_->implications.getVUBs(col)) {
      vbd.second.constant -= constant;
      vbd.second.constant /= scale;
      vbd.second.coef /= scale;
    }

    if (scale < 0)
      mipsolver->mipdata_->implications.getVLBs(col).swap(
          mipsolver->mipdata_->implications.getVUBs(col));
  }

  postsolve_stack.linearTransform(col, scale, constant);

  double oldLower = model->col_lower_[col];
  double oldUpper = model->col_upper_[col];
  model->col_upper_[col] -= constant;
  model->col_lower_[col] -= constant;

  for (const HighsSliceNonzero& nonzero : getColumnVector(col)) {
    impliedRowBounds.updatedVarLower(nonzero.index(), col, nonzero.value(),
                                     oldLower);
    impliedRowBounds.updatedVarUpper(nonzero.index(), col, nonzero.value(),
                                     oldUpper);
  }

  double oldImplLower = implColLower[col];
  double oldImplUpper = implColUpper[col];
  implColLower[col] -= constant;
  implColUpper[col] -= constant;

  for (const HighsSliceNonzero& nonzero : getColumnVector(col)) {
    impliedRowBounds.updatedImplVarLower(nonzero.index(), col, nonzero.value(),
                                         oldImplLower, colLowerSource[col]);
    impliedRowBounds.updatedImplVarUpper(nonzero.index(), col, nonzero.value(),
                                         oldImplUpper, colUpperSource[col]);
  }

  // now apply the scaling, which does not change the contributions to the
  // implied row bounds, but requires adjusting the implied bounds of the
  // columns dual constraint
  impliedDualRowBounds.sumScaled(col, scale);

  double boundScale = 1.0 / scale;
  model->col_lower_[col] *= boundScale;
  model->col_upper_[col] *= boundScale;
  implColLower[col] *= boundScale;
  implColUpper[col] *= boundScale;
  if (model->integrality_[col] != HighsVarType::kContinuous) {
    // we rely on the integrality status being already updated to the newly
    // scaled column by the caller, if necessary
    model->col_upper_[col] =
        std::floor(model->col_upper_[col] + primal_feastol);
    model->col_lower_[col] = std::ceil(model->col_lower_[col] - primal_feastol);
  }

  if (scale < 0) {
    std::swap(model->col_lower_[col], model->col_upper_[col]);
    std::swap(implColLower[col], implColUpper[col]);
    std::swap(colLowerSource[col], colUpperSource[col]);
  }

  model->offset_ += model->col_cost_[col] * constant;
  model->col_cost_[col] *= scale;

  for (HighsInt coliter = colhead[col]; coliter != -1;
       coliter = Anext[coliter]) {
    double val = Avalue[coliter];
    Avalue[coliter] *= scale;
    HighsInt row = Arow[coliter];
    double rowConstant = val * constant;
    if (model->row_lower_[row] != -kHighsInf)
      model->row_lower_[row] -= rowConstant;
    if (model->row_upper_[row] != kHighsInf)
      model->row_upper_[row] -= rowConstant;
  }

  markChangedCol(col);
}

void HPresolve::scaleRow(HighsInt row, double scale, bool integral) {
  storeRow(row);

  scaleStoredRow(row, scale, integral);
}

void HPresolve::scaleStoredRow(HighsInt row, double scale, bool integral) {
  HighsInt rowlen = rowpositions.size();

  model->row_upper_[row] *= scale;
  model->row_lower_[row] *= scale;
  implRowDualLower[row] /= scale;
  implRowDualUpper[row] /= scale;

  if (integral) {
    if (model->row_upper_[row] != kHighsInf)
      model->row_upper_[row] = std::round(model->row_upper_[row]);
    if (model->row_lower_[row] != kHighsInf)
      model->row_lower_[row] = std::round(model->row_lower_[row]);
    for (HighsInt j = 0; j < rowlen; ++j) {
      Avalue[rowpositions[j]] *= scale;
      if (std::abs(Avalue[rowpositions[j]]) <= options->small_matrix_value)
        unlink(rowpositions[j]);
    }
  } else
    for (HighsInt j = 0; j < rowlen; ++j) {
      Avalue[rowpositions[j]] *= scale;
      if (std::abs(Avalue[rowpositions[j]]) <= options->small_matrix_value)
        unlink(rowpositions[j]);
    }

  impliedRowBounds.sumScaled(row, scale);
  if (scale < 0) {
    std::swap(rowDualLower[row], rowDualUpper[row]);
    std::swap(implRowDualLower[row], implRowDualUpper[row]);
    std::swap(rowDualLowerSource[row], rowDualUpperSource[row]);
    std::swap(model->row_lower_[row], model->row_upper_[row]);
  }
}

void HPresolve::substitute(HighsInt row, HighsInt col, double rhs) {
  assert(!rowDeleted[row]);
  assert(!colDeleted[col]);
  HighsInt pos = findNonzero(row, col);
  assert(pos != -1);

  assert(Arow[pos] == row);
  assert(Acol[pos] == col);
  double substrowscale = -1.0 / Avalue[pos];
  assert(isImpliedFree(col));

  markRowDeleted(row);
  markColDeleted(col);

  // substitute the column in each row where it occurs
  for (HighsInt coliter = colhead[col]; coliter != -1;) {
    HighsInt colrow = Arow[coliter];
    double colval = Avalue[coliter];

    // walk to the next position before doing any modifications, because
    // the current position will be deleted in the loop below
    assert(Acol[coliter] == col);
    HighsInt colpos = coliter;
    coliter = Anext[coliter];

    // skip the row that is used for substitution
    if (row == colrow) continue;

    assert(findNonzero(colrow, col) != -1);

    // cancels out and bounds of dual row for this column do not need to be
    // updated
    unlink(colpos);

    // printf("\nbefore substitution: ");
    // debugPrintRow(colrow);

    // determine the scale for the substitution row for addition to this row
    double scale = colval * substrowscale;

    // adjust the sides
    if (model->row_lower_[colrow] != -kHighsInf)
      model->row_lower_[colrow] += scale * rhs;

    if (model->row_upper_[colrow] != kHighsInf)
      model->row_upper_[colrow] += scale * rhs;

    for (HighsInt rowiter : rowpositions) {
      assert(Arow[rowiter] == row);

      if (Acol[rowiter] != col)
        addToMatrix(colrow, Acol[rowiter], scale * Avalue[rowiter]);
    }

    // check if this is an equation row and it now has a different size
    if (model->row_lower_[colrow] == model->row_upper_[colrow] &&
        eqiters[colrow] != equations.end() &&
        eqiters[colrow]->first != rowsize[colrow]) {
      // if that is the case reinsert it into the equation set that is ordered
      // by sparsity
      equations.erase(eqiters[colrow]);
      eqiters[colrow] = equations.emplace(rowsize[colrow], colrow).first;
    }
    // printf("after substitution: ");
    // debugPrintRow(colrow);
  }

  assert(colsize[col] == 1);

  // substitute column in the objective function
  if (model->col_cost_[col] != 0.0) {
    HighsCDouble objscale = model->col_cost_[col] * substrowscale;
    model->offset_ = double(model->offset_ - objscale * rhs);
    assert(std::isfinite(model->offset_));
    for (HighsInt rowiter : rowpositions) {
      // printf("changing col cost to %g = %g + %g * %g\n",
      // double(model->col_cost_[Acol[rowiter]] + objscale * Avalue[rowiter]),
      // model->col_cost_[Acol[rowiter]], double(objscale), Avalue[rowiter]);
      model->col_cost_[Acol[rowiter]] =
          double(model->col_cost_[Acol[rowiter]] + objscale * Avalue[rowiter]);
      if (std::abs(model->col_cost_[Acol[rowiter]]) <=
          options->small_matrix_value)
        model->col_cost_[Acol[rowiter]] = 0.0;
    }
    assert(std::abs(model->col_cost_[col]) <=
           std::max(options->dual_feasibility_tolerance,
                    kHighsTiny * std::abs(double(objscale))));
    model->col_cost_[col] = 0.0;
  }

  // finally remove the entries of the row that was used for substitution
  for (HighsInt rowiter : rowpositions) unlink(rowiter);
}

void HPresolve::toCSC(std::vector<double>& Aval, std::vector<HighsInt>& Aindex,
                      std::vector<HighsInt>& Astart) {
  // set up the column starts using the column size array
  HighsInt numcol = colsize.size();
  Astart.resize(numcol + 1);
  HighsInt nnz = 0;
  for (HighsInt i = 0; i != numcol; ++i) {
    Astart[i] = nnz;
    nnz += colsize[i];
  }
  Astart[numcol] = nnz;

  // now setup the entries of the CSC matrix
  // we reuse the colsize array to count down to zero
  // for determining the position of each nonzero
  Aval.resize(nnz);
  Aindex.resize(nnz);
  HighsInt numslots = Avalue.size();
  assert(numslots - int(freeslots.size()) == nnz);
  for (HighsInt i = 0; i != numslots; ++i) {
    if (Avalue[i] == 0.0) continue;
    assert(Acol[i] >= 0 && Acol[i] < model->num_col_);
    HighsInt pos = Astart[Acol[i] + 1] - colsize[Acol[i]];
    --colsize[Acol[i]];
    assert(colsize[Acol[i]] >= 0);
    Aval[pos] = Avalue[i];
    Aindex[pos] = Arow[i];
  }
}

void HPresolve::toCSR(std::vector<double>& ARval,
                      std::vector<HighsInt>& ARindex,
                      std::vector<HighsInt>& ARstart) {
  // set up the row starts using the row size array
  HighsInt numrow = rowsize.size();
  ARstart.resize(numrow + 1);
  HighsInt nnz = 0;
  for (HighsInt i = 0; i != numrow; ++i) {
    ARstart[i] = nnz;
    nnz += rowsize[i];
  }
  ARstart[numrow] = nnz;

  // now setup the entries of the CSC matrix
  // we reuse the colsize array to count down to zero
  // for determining the position of each nonzero
  ARval.resize(nnz);
  ARindex.resize(nnz);
  for (HighsInt i = 0; i != nnz; ++i) {
    if (Avalue[i] == 0.0) continue;
    HighsInt pos = ARstart[Arow[i] + 1] - rowsize[Arow[i]];
    --rowsize[Arow[i]];
    assert(rowsize[Arow[i]] >= 0);
    ARval[pos] = Avalue[i];
    ARindex[pos] = Acol[i];
  }
}

HPresolve::Result HPresolve::doubletonEq(HighsPostsolveStack& postsolve_stack,
                                         HighsInt row) {
  assert(!rowDeleted[row]);
  assert(rowsize[row] == 2);
  assert(model->row_lower_[row] == model->row_upper_[row]);
  // printf("doubleton equation: ");
  // debugPrintRow(row);
  HighsInt nzPos1 = rowroot[row];
  HighsInt nzPos2 = ARright[nzPos1] != -1 ? ARright[nzPos1] : ARleft[nzPos1];

  HighsInt substcol;
  HighsInt staycol;
  double substcoef;
  double staycoef;
  double rhs = model->row_upper_[row];
  if (model->integrality_[Acol[nzPos1]] == HighsVarType::kInteger) {
    if (model->integrality_[Acol[nzPos2]] == HighsVarType::kInteger) {
      // both columns integer. For substitution choose smaller absolute
      // coefficient value, or sparser column if values are equal
      if (std::abs(Avalue[nzPos1]) <
          std::abs(Avalue[nzPos2]) - options->small_matrix_value) {
        substcol = Acol[nzPos1];
        staycol = Acol[nzPos2];

        substcoef = Avalue[nzPos1];
        staycoef = Avalue[nzPos2];
      } else if (std::abs(Avalue[nzPos2]) <
                 std::abs(Avalue[nzPos1]) - options->small_matrix_value) {
        substcol = Acol[nzPos2];
        staycol = Acol[nzPos1];

        substcoef = Avalue[nzPos2];
        staycoef = Avalue[nzPos1];
      } else if (colsize[Acol[nzPos1]] < colsize[Acol[nzPos2]]) {
        substcol = Acol[nzPos1];
        staycol = Acol[nzPos2];

        substcoef = Avalue[nzPos1];
        staycoef = Avalue[nzPos2];
      } else {
        substcol = Acol[nzPos2];
        staycol = Acol[nzPos1];

        substcoef = Avalue[nzPos2];
        staycoef = Avalue[nzPos1];
      }

      // check integrality conditions
      double roundCoef = std::round(staycoef / substcoef) * substcoef;
      if (std::abs(roundCoef - staycoef) > options->small_matrix_value)
        return Result::kOk;
      staycoef = roundCoef;
      double roundRhs = std::round(rhs / substcoef) * substcoef;
      if (std::abs(rhs - roundRhs) > primal_feastol)
        return Result::kPrimalInfeasible;
      rhs = roundRhs;
    } else {
      // one col is integral, substitute the continuous one
      substcol = Acol[nzPos2];
      staycol = Acol[nzPos1];

      substcoef = Avalue[nzPos2];
      staycoef = Avalue[nzPos1];
    }
  } else {
    if (model->integrality_[Acol[nzPos2]] == HighsVarType::kInteger) {
      // one col is integral, substitute the continuous one
      substcol = Acol[nzPos1];
      staycol = Acol[nzPos2];

      substcoef = Avalue[nzPos1];
      staycoef = Avalue[nzPos2];
    } else {
      // both columns continuous the one with a larger absolute coefficient
      // value if the difference is more than factor 2, and otherwise the one
      // with fewer nonzeros if those are equal
      bool colAtPos1Better;
      HighsInt col1Size = colsize[Acol[nzPos1]];
      if (col1Size == 1)
        colAtPos1Better = true;
      else {
        HighsInt col2Size = colsize[Acol[nzPos2]];
        if (col2Size == 1)
          colAtPos1Better = false;
        else {
          double abs1Val = std::fabs(Avalue[nzPos1]);
          double abs2Val = std::fabs(Avalue[nzPos2]);
          if (col1Size != col2Size &&
              std::max(abs1Val, abs2Val) <= 2.0 * std::min(abs1Val, abs2Val))
            colAtPos1Better = col1Size < col2Size;
          else if (abs1Val > abs2Val)
            colAtPos1Better = true;
          else
            colAtPos1Better = false;
        }
      }

      if (colAtPos1Better) {
        substcol = Acol[nzPos1];
        staycol = Acol[nzPos2];

        substcoef = Avalue[nzPos1];
        staycoef = Avalue[nzPos2];
      } else {
        substcol = Acol[nzPos2];
        staycol = Acol[nzPos1];

        substcoef = Avalue[nzPos2];
        staycoef = Avalue[nzPos1];
      }
    }
  }

  double oldStayLower = model->col_lower_[staycol];
  double oldStayUpper = model->col_upper_[staycol];
  double substLower = model->col_lower_[substcol];
  double substUpper = model->col_upper_[substcol];

  double stayImplLower;
  double stayImplUpper;
  if (std::signbit(substcoef) != std::signbit(staycoef)) {
    // coefficients have the opposite sign, therefore the implied lower bound of
    // the stay column is computed from the lower bound of the substituted
    // column:
    // staycol * staycoef + substcol * substcoef = rhs
    // staycol = (rhs - substcol * substcoef) / staycoef
    // staycol >= rhs / staycoef + lower(-substcoef/staycoef * substcol)
    // lower(-substcoef/staycoef * substcol) is (-substcoef/staycoef) *
    // substLower if (-substcoef/staycoef) is positive, i.e. if the coefficients
    // have opposite sign
    stayImplLower =
        substLower == -kHighsInf
            ? -kHighsInf
            : double((HighsCDouble(rhs) - substcoef * substLower) / staycoef);
    stayImplUpper =
        substUpper == kHighsInf
            ? kHighsInf
            : double((HighsCDouble(rhs) - substcoef * substUpper) / staycoef);
  } else {
    stayImplLower =
        substUpper == kHighsInf
            ? -kHighsInf
            : double((HighsCDouble(rhs) - substcoef * substUpper) / staycoef);
    stayImplUpper =
        substLower == -kHighsInf
            ? kHighsInf
            : double((HighsCDouble(rhs) - substcoef * substLower) / staycoef);
  }

  // possibly tighten bounds of the column that stays
  bool lowerTightened = false;
  bool upperTightened = false;
  if (stayImplLower > oldStayLower + primal_feastol) {
    lowerTightened = true;
    changeColLower(staycol, stayImplLower);
  }

  if (stayImplUpper < oldStayUpper - primal_feastol) {
    upperTightened = true;
    changeColUpper(staycol, stayImplUpper);
  }

  postsolve_stack.doubletonEquation(row, substcol, staycol, substcoef, staycoef,
                                    rhs, substLower, substUpper,
                                    model->col_cost_[substcol], lowerTightened,
                                    upperTightened, getColumnVector(substcol));

  // finally modify matrix
  markColDeleted(substcol);
  removeRow(row);
  substitute(substcol, staycol, rhs / substcoef, -staycoef / substcoef);

  // since a column was deleted we might have new row singletons which we
  // immediately remove
  HPRESOLVE_CHECKED_CALL(removeRowSingletons(postsolve_stack));

  return checkLimits(postsolve_stack);
}

HPresolve::Result HPresolve::singletonRow(HighsPostsolveStack& postsolve_stack,
                                          HighsInt row) {
  assert(!rowDeleted[row]);
  assert(rowsize[row] == 1);

  // the tree of nonzeros of this row should just contain the single nonzero
  HighsInt nzPos = rowroot[row];
  assert(nzPos != -1);
  // nonzero should have the row in the row array
  assert(Arow[nzPos] == row);
  // tree with one element should not have children
  assert(ARleft[nzPos] == -1);
  assert(ARright[nzPos] == -1);

  HighsInt col = Acol[nzPos];
  double val = Avalue[nzPos];

  // printf("singleton row\n");
  // debugPrintRow(row);
  // delete row singleton nonzero directly, we have all information that we need
  // in local variables
  markRowDeleted(row);
  unlink(nzPos);

  // check for simple
  if (val > 0) {
    if (model->col_upper_[col] * val <=
            model->row_upper_[row] + primal_feastol &&
        model->col_lower_[col] * val >=
            model->row_lower_[row] - primal_feastol) {
      postsolve_stack.redundantRow(row);
      return checkLimits(postsolve_stack);
    }
  } else {
    if (model->col_lower_[col] * val <=
            model->row_upper_[row] + primal_feastol &&
        model->col_upper_[col] * val >=
            model->row_lower_[row] - primal_feastol) {
      postsolve_stack.redundantRow(row);
      return checkLimits(postsolve_stack);
    }
  }

  // zeros should not be linked in the matrix
  assert(std::fabs(val) > options->small_matrix_value);

  double newColUpper = kHighsInf;
  double newColLower = -kHighsInf;
  if (val > 0) {
    if (model->row_upper_[row] != kHighsInf)
      newColUpper = model->row_upper_[row] / val;
    if (model->row_lower_[row] != -kHighsInf)
      newColLower = model->row_lower_[row] / val;
  } else {
    if (model->row_upper_[row] != kHighsInf)
      newColLower = model->row_upper_[row] / val;
    if (model->row_lower_[row] != -kHighsInf)
      newColUpper = model->row_lower_[row] / val;
  }

  // use either the primal feasibility tolerance for the bound constraint or
  // for the singleton row including scaling, whichever is tighter.
  const double boundTol = primal_feastol / std::max(1.0, std::fabs(val));
  const bool isIntegral = model->integrality_[col] != HighsVarType::kContinuous;

  bool lowerTightened = newColLower > model->col_lower_[col] + boundTol;
  bool upperTightened = newColUpper < model->col_upper_[col] - boundTol;

  double lb, ub;
  if (lowerTightened) {
    if (isIntegral) newColLower = std::ceil(newColLower - boundTol);
    lb = newColLower;
  } else
    lb = model->col_lower_[col];

  if (upperTightened) {
    if (isIntegral) newColUpper = std::floor(newColUpper + boundTol);
    ub = newColUpper;
  } else
    ub = model->col_upper_[col];

  // printf("old bounds [%.15g,%.15g], new bounds [%.15g,%.15g] ... ",
  //        model->col_lower_[col], model->col_upper_[col], lb, ub);
  // check whether the bounds are equal in tolerances
  if (ub <= lb + primal_feastol) {
    // bounds could be infeasible or equal in tolerances, first check infeasible
    if (ub < lb - primal_feastol) return Result::kPrimalInfeasible;

    // bounds are equal in tolerances, if they have a slight infeasibility below
    // those tolerances or they have a slight numerical distance which changes
    // the largest contribution below feasibility tolerance then we can safely
    // set the bound to one of the values. To heuristically get rid of numerical
    // errors we choose the bound that was not tightened, or the midpoint if
    // both where tightened.
    if (ub < lb || (ub > lb && (ub - lb) * std::max(std::fabs(val),
                                                    getMaxAbsColVal(col)) <=
                                   primal_feastol)) {
      if (lowerTightened && upperTightened) {
        ub = 0.5 * (ub + lb);
        lb = ub;
        lowerTightened = lb > model->col_lower_[col];
        upperTightened = ub < model->col_upper_[col];
      } else if (lowerTightened) {
        lb = ub;
        lowerTightened = lb > model->col_lower_[col];
      } else {
        ub = lb;
        upperTightened = ub < model->col_upper_[col];
      }
    }
  }

  // printf("final bounds: [%.15g,%.15g]\n", lb, ub);

  postsolve_stack.singletonRow(row, col, val, lowerTightened, upperTightened);

  // just update bounds (and row activities)
  if (lowerTightened) changeColLower(col, lb);
  // update bounds, or remove as fixed column directly
  if (ub == lb) {
    postsolve_stack.removedFixedCol(col, lb, model->col_cost_[col],
                                    getColumnVector(col));
    removeFixedCol(col);
  } else if (upperTightened)
    changeColUpper(col, ub);

  if (!colDeleted[col] && colsize[col] == 0)
    return emptyCol(postsolve_stack, col);

  return checkLimits(postsolve_stack);
}

HPresolve::Result HPresolve::singletonCol(HighsPostsolveStack& postsolve_stack,
                                          HighsInt col) {
  assert(colsize[col] == 1);
  assert(!colDeleted[col]);
  HighsInt nzPos = colhead[col];
  HighsInt row = Arow[nzPos];
  double colCoef = Avalue[nzPos];

  double colDualUpper =
      -impliedDualRowBounds.getSumLower(col, -model->col_cost_[col]);
  double colDualLower =
      -impliedDualRowBounds.getSumUpper(col, -model->col_cost_[col]);

  // check for dominated column
  if (colDualLower > options->dual_feasibility_tolerance) {
    if (model->col_lower_[col] == -kHighsInf)
      return Result::kDualInfeasible;
    else
      fixColToLower(postsolve_stack, col);
    return checkLimits(postsolve_stack);
  }

  if (colDualUpper < -options->dual_feasibility_tolerance) {
    if (model->col_upper_[col] == kHighsInf)
      return Result::kDualInfeasible;
    else
      fixColToUpper(postsolve_stack, col);
    return checkLimits(postsolve_stack);
  }

  // check for weakly dominated column
  if (colDualUpper <= options->dual_feasibility_tolerance) {
    if (model->col_upper_[col] != kHighsInf)
      fixColToUpper(postsolve_stack, col);
    else if (impliedDualRowBounds.getSumLowerOrig(col) == 0.0) {
      // todo: forcing column, since this implies colDual >= 0 and we
      // already checked that colDual <= 0 and since the cost are 0.0
      // all the rows are at a dual multiplier of zero and we can determine
      // one nonbasic row in postsolve, and make the other rows and the column
      // basic. The columns primal value is computed from the non-basic row
      // which is chosen such that the values of all rows are primal feasible
      // printf("removing forcing column of size %" HIGHSINT_FORMAT "\n",
      // colsize[col]);
      postsolve_stack.forcingColumn(col, getColumnVector(col),
                                    model->col_cost_[col],
                                    model->col_lower_[col], true);
      markColDeleted(col);
      HighsInt coliter = colhead[col];
      while (coliter != -1) {
        HighsInt row = Arow[coliter];
        double rhs = Avalue[coliter] > 0.0 ? model->row_lower_[row]
                                           : model->row_upper_[row];
        coliter = Anext[coliter];

        postsolve_stack.forcingColumnRemovedRow(col, row, rhs,
                                                getRowVector(row));
        removeRow(row);
      }
    }
    return checkLimits(postsolve_stack);
  }
  if (colDualLower >= -options->dual_feasibility_tolerance) {
    if (model->col_lower_[col] != -kHighsInf)
      fixColToLower(postsolve_stack, col);
    else if (impliedDualRowBounds.getSumUpperOrig(col) == 0.0) {
      // forcing column, since this implies colDual <= 0 and we already checked
      // that colDual >= 0
      // printf("removing forcing column of size %" HIGHSINT_FORMAT "\n",
      // colsize[col]);
      postsolve_stack.forcingColumn(col, getColumnVector(col),
                                    model->col_cost_[col],
                                    model->col_upper_[col], false);
      markColDeleted(col);
      HighsInt coliter = colhead[col];
      while (coliter != -1) {
        HighsInt row = Arow[coliter];
        double rhs = Avalue[coliter] > 0.0 ? model->row_upper_[row]
                                           : model->row_lower_[row];
        coliter = Anext[coliter];

        postsolve_stack.forcingColumnRemovedRow(col, row, rhs,
                                                getRowVector(row));
        removeRow(row);
      }
    }
    return checkLimits(postsolve_stack);
  }

  if (mipsolver != nullptr &&
      model->integrality_[col] == HighsVarType::kContinuous &&
      isImpliedInteger(col)) {
    model->integrality_[col] = HighsVarType::kImplicitInteger;
    ++rowsizeImplInt[row];
    double ceilLower = std::ceil(model->col_lower_[col] - primal_feastol);
    double floorUpper = std::floor(model->col_upper_[col] + primal_feastol);

    if (ceilLower > model->col_lower_[col]) changeColLower(col, ceilLower);
    if (floorUpper < model->col_upper_[col]) changeColUpper(col, floorUpper);
  }

  updateColImpliedBounds(row, col, colCoef);

  if (model->integrality_[col] != HighsVarType::kInteger)
    updateRowDualImpliedBounds(row, col, colCoef);

  // now check if column is implied free within an equation and substitute the
  // column if that is the case
  if (isDualImpliedFree(row) && isImpliedFree(col)) {
    if (model->integrality_[col] == HighsVarType::kInteger &&
        !isImpliedIntegral(col))
      return Result::kOk;
    // todo, store which side of an implied free dual variable needs to be used
    // for substitution
    storeRow(row);

    HighsPostsolveStack::RowType rowType = HighsPostsolveStack::RowType::kEq;
    double rhs;
    if (model->row_lower_[row] == model->row_upper_[row]) {
      rhs = model->row_upper_[row];
      rowType = HighsPostsolveStack::RowType::kEq;
    } else if ((model->row_upper_[row] != kHighsInf &&
                implRowDualUpper[row] <= options->dual_feasibility_tolerance)) {
      rhs = model->row_upper_[row];
      rowType = HighsPostsolveStack::RowType::kLeq;
    } else {
      rhs = model->row_lower_[row];
      rowType = HighsPostsolveStack::RowType::kGeq;
    }

    postsolve_stack.freeColSubstitution(row, col, rhs, model->col_cost_[col],
                                        rowType, getStoredRow(),
                                        getColumnVector(col));
    // todo, check integrality of coefficients and allow this
    substitute(row, col, rhs);

    return checkLimits(postsolve_stack);
  }

  // todo: check for zero cost singleton and remove
  return Result::kOk;
}

HPresolve::Result HPresolve::rowPresolve(HighsPostsolveStack& postsolve_stack,
                                         HighsInt row) {
  assert(!rowDeleted[row]);

  // handle special cases directly via a call to the specialized procedure
  switch (rowsize[row]) {
    default:
      break;
    case 0:
      if (model->row_upper_[row] < -primal_feastol ||
          model->row_lower_[row] > primal_feastol)
        // model infeasible
        return Result::kPrimalInfeasible;
      postsolve_stack.redundantRow(row);
      markRowDeleted(row);
      return checkLimits(postsolve_stack);
    case 1:
      return singletonRow(postsolve_stack, row);
  }

  // printf("row presolve: ");
  // debugPrintRow(row);
  double impliedRowUpper = impliedRowBounds.getSumUpper(row);
  double impliedRowLower = impliedRowBounds.getSumLower(row);

  if (impliedRowLower > model->row_upper_[row] + primal_feastol ||
      impliedRowUpper < model->row_lower_[row] - primal_feastol) {
    // model infeasible
    return Result::kPrimalInfeasible;
  }

  if (impliedRowLower >= model->row_lower_[row] - primal_feastol &&
      impliedRowUpper <= model->row_upper_[row] + primal_feastol) {
    // row is redundant
    postsolve_stack.redundantRow(row);
    removeRow(row);
    return checkLimits(postsolve_stack);
  }

  if (model->row_lower_[row] != model->row_upper_[row]) {
    if (implRowDualLower[row] > options->dual_feasibility_tolerance) {
      model->row_upper_[row] = model->row_lower_[row];
      if (mipsolver == nullptr) {
        HighsInt col = rowDualLowerSource[row];
        assert(model->col_cost_[col] != 0.0);
        if (colsize[col] == 1) {
          double colCoef = Avalue[colhead[col]];
          if (model->col_cost_[col] > 0) {
            assert(
                model->col_lower_[col] == -kHighsInf ||
                (model->col_lower_[col] <= implColLower[col] + primal_feastol &&
                 colLowerSource[col] == row));
            if (model->col_lower_[col] > implColLower[col] - primal_feastol)
              changeColLower(col, -kHighsInf);
          } else {
            assert(
                model->col_upper_[col] == kHighsInf ||
                (model->col_upper_[col] >= implColUpper[col] - primal_feastol &&
                 colUpperSource[col] == row));
            if (model->col_upper_[col] < implColUpper[col] + primal_feastol)
              changeColUpper(col, kHighsInf);
          }
        }
      }
    }

    if (implRowDualUpper[row] < -options->dual_feasibility_tolerance) {
      model->row_lower_[row] = model->row_upper_[row];
      if (mipsolver == nullptr) {
        HighsInt col = rowDualUpperSource[row];
        assert(model->col_cost_[col] != 0.0);
        if (colsize[col] == 1) {
          if (model->col_cost_[col] > 0) {
            assert(
                model->col_lower_[col] == -kHighsInf ||
                (model->col_lower_[col] <= implColLower[col] + primal_feastol &&
                 colLowerSource[col] == row));
            if (model->col_lower_[col] > implColLower[col] - primal_feastol)
              changeColLower(col, -kHighsInf);
          } else {
            assert(
                model->col_upper_[col] == kHighsInf ||
                (model->col_upper_[col] >= implColUpper[col] - primal_feastol &&
                 colUpperSource[col] == row));
            if (model->col_upper_[col] < implColUpper[col] + primal_feastol)
              changeColUpper(col, kHighsInf);
          }
        }
      }
    }
  }

  double rowUpper = model->row_upper_[row];
  double rowLower = model->row_lower_[row];

  if (rowsize[row] == 2 && rowLower == rowUpper)
    return doubletonEq(postsolve_stack, row);

  // todo: do additional single row presolve for mip here. It may assume a
  // non-redundant and non-infeasible row when considering variable and implied
  // bounds
  if (rowsizeInteger[row] != 0 || rowsizeImplInt[row] != 0) {
    if (rowLower == rowUpper) {
      // equation
      if (impliedRowLower != -kHighsInf && impliedRowUpper != kHighsInf &&
          std::abs(impliedRowLower + impliedRowUpper - 2 * rowUpper) <=
              options->small_matrix_value) {
        double binCoef = std::abs(impliedRowUpper - rowUpper);
        // simple probing on equation case
        HighsInt binCol = -1;
        storeRow(row);
        for (const HighsSliceNonzero& nonz : getStoredRow()) {
          if (std::abs(std::abs(nonz.value()) - binCoef) <=
                  options->small_matrix_value &&
              model->integrality_[nonz.index()] == HighsVarType::kInteger &&
              std::abs(model->col_upper_[nonz.index()] -
                       model->col_lower_[nonz.index()] - 1.0) <=
                  primal_feastol) {
            // found a binary variable that implies all other variables to be
            // fixed when it sits at one of its bounds therefore we can
            // substitute all other variables in the row
            binCol = nonz.index();
            // store the binary coefficient with its actual sign
            binCoef = nonz.value();
            break;
          }
        }

        if (binCol != -1) {
          // found binary column for substituting all other columns
          // printf("simple probing case on row of size %" HIGHSINT_FORMAT "\n",
          // rowsize[row]);
          for (const HighsSliceNonzero& nonz : getStoredRow()) {
            if (nonz.index() == binCol) continue;

            if (model->col_lower_[nonz.index()] ==
                model->col_upper_[nonz.index()]) {
              postsolve_stack.removedFixedCol(nonz.index(),
                                              model->col_lower_[nonz.index()],
                                              0.0, HighsEmptySlice());
              removeFixedCol(nonz.index());
              continue;
            }

            if (std::signbit(binCoef) == std::signbit(nonz.value())) {
              // binary coefficient is positive:
              // setting the binary to its upper bound
              // increases the minimal activity to be equal to the row upper
              // bound and there for all other variables are fixed to the bound
              // that contributes to the rows minimal activity, i.e. the lower
              // bound for a positive coefficient

              // This case yields the following implications:
              // binCol = ub -> nonzCol = lb
              // binCol = lb -> nonzCol = ub
              // as linear equation:
              // nonzCol = colUb - (colUb - colLb)(binCol - binLb)
              // nonzCol = colUb + binLb * (colUb - colLb) - (colUb - colLb) *
              // binCol
              double scale = model->col_lower_[nonz.index()] -
                             model->col_upper_[nonz.index()];
              double offset = model->col_upper_[nonz.index()] -
                              model->col_lower_[binCol] * scale;
              postsolve_stack.doubletonEquation(
                  -1, nonz.index(), binCol, 1.0, -scale, offset,
                  model->col_lower_[nonz.index()],
                  model->col_upper_[nonz.index()], 0.0, false, false,
                  HighsEmptySlice());
              substitute(nonz.index(), binCol, offset, scale);
            } else {
              // This case yields the following implications:
              // binCol = lb -> nonzCol = lb
              // binCol = ub -> nonzCol = ub
              // as linear equation:
              // nonzCol = colLb + (colUb - colLb)(binCol - binLb)
              // nonzCol =
              //    colLb - binLb*(colUb - colLb) + (colUb - colLb)*binCol
              double scale = model->col_upper_[nonz.index()] -
                             model->col_lower_[nonz.index()];
              double offset = model->col_lower_[nonz.index()] -
                              model->col_lower_[binCol] * scale;
              postsolve_stack.doubletonEquation(
                  -1, nonz.index(), binCol, 1.0, -scale, offset,
                  model->col_lower_[nonz.index()],
                  model->col_upper_[nonz.index()], 0.0, false, false,
                  HighsEmptySlice());
              substitute(nonz.index(), binCol, offset, scale);
            }
          }

          removeRow(row);
          HPRESOLVE_CHECKED_CALL(checkLimits(postsolve_stack));
          return removeRowSingletons(postsolve_stack);
        }
      }

      if (rowsizeInteger[row] + rowsizeImplInt[row] >= rowsize[row] - 1) {
        HighsInt continuousCol = -1;
        double continuousCoef = 0.0;
        std::vector<double> rowCoefsInt;
        rowCoefsInt.reserve(rowsize[row]);
        storeRow(row);

        for (const HighsSliceNonzero& nonz : getStoredRow()) {
          if (model->integrality_[nonz.index()] == HighsVarType::kContinuous) {
            assert(continuousCoef == 0.0);
            continuousCoef = nonz.value();
            continuousCol = nonz.index();
            continue;
          }

          rowCoefsInt.push_back(nonz.value());
        }

        if (continuousCoef != 0.0) {
          rowCoefsInt.push_back(rowUpper);

          double intScale = HighsIntegers::integralScale(
              rowCoefsInt, options->small_matrix_value,
              options->small_matrix_value);

          if (intScale != 0 && intScale <= 1e3) {
            double scale = 1.0 / std::abs(continuousCoef * intScale);
            if (scale != 1.0) {
              // printf(
              //     "transform continuous column x to implicit integer z with x
              //     "
              //     "= %g * z\n",
              //     scale);
              transformColumn(postsolve_stack, continuousCol, scale, 0.0);
              model->integrality_[continuousCol] =
                  HighsVarType::kImplicitInteger;
              for (const HighsSliceNonzero& nonzero :
                   getColumnVector(continuousCol))
                ++rowsizeImplInt[nonzero.index()];
              double ceilLower =
                  std::ceil(model->col_lower_[continuousCol] - primal_feastol);
              double floorUpper =
                  std::floor(model->col_upper_[continuousCol] + primal_feastol);

              if (ceilLower > model->col_lower_[continuousCol])
                changeColLower(continuousCol, ceilLower);
              if (floorUpper < model->col_upper_[continuousCol])
                changeColUpper(continuousCol, floorUpper);

              if (intScale != 1.0) scaleStoredRow(row, intScale, true);
            }
          }
        } else {
          double intScale = HighsIntegers::integralScale(
              rowCoefsInt, options->small_matrix_value,
              options->small_matrix_value);

          if (intScale != 0.0 && intScale <= 1e3) {
            double rhs = rowUpper * intScale;
            if (std::abs(rhs - std::round(rhs)) > primal_feastol)
              return Result::kPrimalInfeasible;

            rhs = std::round(rhs);

            HighsInt rowlen = rowpositions.size();
            HighsInt x1Cand = -1;
            int64_t d = 0;

            for (HighsInt i = 0; i < rowlen; ++i) {
              int64_t newgcd =
                  d == 0 ? int64_t(std::abs(
                               std::round(intScale * Avalue[rowpositions[i]])))
                         : HighsIntegers::gcd(
                               std::abs(std::round(intScale *
                                                   Avalue[rowpositions[i]])),
                               d);
              if (newgcd == 1) {
                // adding this variable would set the gcd to 1, therefore it
                // must be our candidate x1 for substitution. If another
                // candidate already exists no reduction is possible except for
                // scaling the equation
                if (x1Cand != -1) {
                  x1Cand = -1;
                  break;
                }
                x1Cand = i;
              } else {
                d = newgcd;
              }
            }

            if (x1Cand != -1) {
              HighsInt x1Pos = rowpositions[x1Cand];
              HighsInt x1 = Acol[x1Pos];
              double rhs2 = rhs / d;
              if (std::abs(std::round(rhs2) - rhs2) <=
                  mipsolver->mipdata_->epsilon) {
                // the right hand side is integral, so we can substitute
                // x1 = d * z

                // printf(
                //    "substitute integral column x with integral column z with
                //    " "x = %ld * z\n", d);
                transformColumn(postsolve_stack, x1, d, 0.0);
              } else {
                // we can substitute x1 = d * z + b, with b = a1^-1 rhs (mod d)

                // first compute the modular multiplciative inverse of a1^-1
                // (mod d) of a1
                int64_t a1 = std::round(intScale * Avalue[x1Pos]);
                a1 = HighsIntegers::mod(a1, d);
                int64_t a1Inverse = HighsIntegers::modularInverse(a1, d);

                // now compute b = a1^-1 rhs (mod d)
                double b = HighsIntegers::mod(a1Inverse * rhs, (double)d);

                // printf(
                //     "substitute integral column x with integral column z with
                //     " "x = %ld * z + %g\n", d, b);

                // before we substitute, we check whether the resulting variable
                // z is fixed after rounding its new bounds. If that is the case
                // we directly fix x1 instead of first substituting with d * z +
                // b.
                double zLower =
                    std::ceil((model->col_lower_[x1] - b) / d - primal_feastol);
                double zUpper = std::floor((model->col_upper_[x1] - b) / d +
                                           primal_feastol);

                if (zLower == zUpper) {
                  // rounded bounds are equal, so fix x1 to the corresponding
                  // bound
                  double fixVal = zLower * d + b;
                  if (std::abs(model->col_lower_[x1] - fixVal) <=
                      primal_feastol)
                    fixColToLower(postsolve_stack, x1);
                  else
                    fixColToUpper(postsolve_stack, x1);

                  rowpositions.erase(rowpositions.begin() + x1Cand);
                } else {
                  transformColumn(postsolve_stack, x1, d, b);
                }
              }

              intScale /= d;
            }

            if (intScale != 1.0) scaleStoredRow(row, intScale, true);
          }
        }
      }
    } else {
      // inequality or ranged row, first store row posititions
      storeRow(row);

      if (rowsize[row] == rowsizeInteger[row] + rowsizeImplInt[row]) {
        std::vector<double> rowCoefs;
        std::vector<HighsInt> rowIndex;
        rowCoefs.reserve(rowsize[row]);
        rowIndex.reserve(rowsize[row]);

        double deltaDown = model->row_lower_[row] == -kHighsInf
                               ? primal_feastol
                               : options->small_matrix_value;
        double deltaUp = model->row_upper_[row] == kHighsInf
                             ? primal_feastol
                             : options->small_matrix_value;

        for (const HighsSliceNonzero& nonz : getStoredRow()) {
          assert(nonz.value() != 0.0);
          rowCoefs.push_back(nonz.value());
          rowIndex.push_back(nonz.index());
        }

        double intScale =
            HighsIntegers::integralScale(rowCoefs, deltaDown, deltaUp);

        if (intScale != 0.0) {
          HighsInt numRowCoefs = rowCoefs.size();
          if (model->row_lower_[row] == -kHighsInf) {
            // <= inequality
            HighsCDouble rhs = model->row_upper_[row] * intScale;
            bool success = true;
            double minRhsTightening = 0.0;
            double maxVal = 0.0;
            for (HighsInt i = 0; i < numRowCoefs; ++i) {
              double coef = rowCoefs[i];
              HighsCDouble scaleCoef = HighsCDouble(coef) * intScale;
              HighsCDouble intCoef = floor(scaleCoef + 0.5);
              HighsCDouble coefDelta = intCoef - scaleCoef;
              rowCoefs[i] = double(intCoef);
              maxVal = std::max(std::abs(rowCoefs[i]), maxVal);
              if (coefDelta < -options->small_matrix_value) {
                minRhsTightening =
                    std::max(-double(coefDelta), minRhsTightening);
              } else if (coefDelta > options->small_matrix_value) {
                if (model->col_upper_[rowIndex[i]] == kHighsInf) {
                  success = false;
                  break;
                }

                rhs += model->col_upper_[rowIndex[i]] * coefDelta;
              }
            }

            if (success) {
              HighsCDouble roundRhs = floor(rhs + primal_feastol);
              if (rhs - roundRhs >=
                  minRhsTightening - options->small_matrix_value) {
                // scaled and rounded is not weaker than the original constraint
                if (maxVal <= 1000.0 || intScale <= 100.0) {
                  // printf(
                  //     "scaling constraint to integral values with scale %g, "
                  //     "rounded scaled side from %g to %g\n",
                  //     intScale, double(rhs), double(roundRhs));
                  // the scale value is reasonably small, change the row values
                  // to be integral
                  model->row_upper_[row] = double(roundRhs);
                  for (HighsInt i = 0; i < numRowCoefs; ++i) {
                    addToMatrix(row, rowIndex[i],
                                rowCoefs[i] - Avalue[rowpositions[i]]);
                  }
                } else if (rhs - roundRhs < minRhsTightening - primal_feastol) {
                  // printf(
                  //     "tightening right hand side from %g to %g due to "
                  //     "rounding with integral scale %g\n",
                  //     model->row_upper_[row], double(roundRhs / intScale),
                  //     intScale);
                  // scale value is large, so we scale back the altered
                  // constraint the scaled back constraint must be stronger than
                  // the original constraint for this to make sense with is
                  // checked with the condition above
                  model->row_upper_[row] = double(roundRhs / intScale);
                  for (HighsInt i = 0; i < numRowCoefs; ++i) {
                    double delta = double(HighsCDouble(rowCoefs[i]) / intScale -
                                          Avalue[rowpositions[i]]);
                    if (std::abs(delta) > options->small_matrix_value)
                      addToMatrix(row, rowIndex[i], delta);
                  }
                }
              }
            }
          } else if (model->row_upper_[row] == kHighsInf) {
            // >= inequality
            HighsCDouble rhs = model->row_lower_[row] * intScale;
            bool success = true;
            double minRhsTightening = 0.0;
            double maxVal = 0.0;
            for (HighsInt i = 0; i < numRowCoefs; ++i) {
              double coef = rowCoefs[i];
              HighsCDouble scaleCoef = HighsCDouble(coef) * intScale;
              HighsCDouble intCoef = floor(scaleCoef + 0.5);
              HighsCDouble coefDelta = intCoef - scaleCoef;
              rowCoefs[i] = double(intCoef);
              maxVal = std::max(std::abs(rowCoefs[i]), maxVal);
              if (coefDelta < -options->small_matrix_value) {
                if (model->col_upper_[rowIndex[i]] == kHighsInf) {
                  success = false;
                  break;
                }

                rhs += model->col_upper_[rowIndex[i]] * coefDelta;
              } else if (coefDelta > options->small_matrix_value) {
                minRhsTightening =
                    std::max(-double(coefDelta), minRhsTightening);
              }
            }

            if (success) {
              HighsCDouble roundRhs = ceil(rhs - primal_feastol);
              if (rhs - roundRhs <=
                  minRhsTightening + options->small_matrix_value) {
                // scaled and rounded is not weaker than the original constraint
                if (maxVal <= 1000.0 || intScale <= 100.0) {
                  // printf(
                  //     "scaling constraint to integral values with scale %g, "
                  //     "rounded scaled side from %g to %g\n",
                  //     intScale, double(rhs), double(roundRhs));
                  // the scale value is reasonably small, change the row values
                  // to be integral
                  model->row_lower_[row] = double(roundRhs);
                  for (HighsInt i = 0; i < numRowCoefs; ++i)
                    addToMatrix(row, rowIndex[i],
                                rowCoefs[i] - Avalue[rowpositions[i]]);
                } else if (rhs - roundRhs > minRhsTightening + primal_feastol) {
                  // scale value is large, so we scale back the altered
                  // constraint the scaled back constraint must be stronger than
                  // the original constraint for this to make sense with is
                  // checked with the condition above
                  // printf(
                  //     "tightening left hand side from %g to %g due to
                  //     rounding " "with integral scale %g\n",
                  //     model->row_lower_[row], double(roundRhs / intScale),
                  //     intScale);
                  model->row_lower_[row] = double(roundRhs / intScale);
                  for (HighsInt i = 0; i < numRowCoefs; ++i) {
                    double delta = double(HighsCDouble(rowCoefs[i]) / intScale -
                                          Avalue[rowpositions[i]]);
                    if (std::abs(delta) > options->small_matrix_value)
                      addToMatrix(row, rowIndex[i], delta);
                  }
                }
              }
            }
          } else {
            // ranged row or equation, can maybe tighten sides and
            HighsCDouble lhs = model->row_lower_[row] * intScale;
            HighsCDouble rhs = model->row_upper_[row] * intScale;
            bool success = true;
            double minRhsTightening = 0.0;
            double minLhsTightening = 0.0;
            double maxVal = 0.0;
            for (HighsInt i = 0; i < numRowCoefs; ++i) {
              double coef = rowCoefs[i];
              HighsCDouble scaleCoef = HighsCDouble(coef) * intScale;
              HighsCDouble intCoef = floor(scaleCoef + 0.5);
              HighsCDouble coefDelta = intCoef - scaleCoef;
              rowCoefs[i] = double(intCoef);
              maxVal = std::max(std::abs(rowCoefs[i]), maxVal);
              if (coefDelta < -options->small_matrix_value) {
                // for the >= side of the constraint a smaller coefficient is
                // stronger: Therefore we relax the left hand side using the
                // bound constraint, if the bound is infinite, abort
                if (model->col_upper_[rowIndex[i]] == kHighsInf) {
                  success = false;
                  break;
                }

                lhs += model->col_upper_[rowIndex[i]] * coefDelta;
                minRhsTightening =
                    std::max(-double(coefDelta), minRhsTightening);
              } else if (coefDelta > options->small_matrix_value) {
                if (model->col_upper_[rowIndex[i]] == kHighsInf) {
                  success = false;
                  break;
                }

                rhs += model->col_upper_[rowIndex[i]] * coefDelta;

                // the coefficient was relaxed regarding the rows lower bound.
                // Therefore the lower bound should be tightened by at least
                // this amount for the scaled constraint to dominate the
                // unscaled constraint be rounded by at least this value
                minLhsTightening =
                    std::max(double(coefDelta), minLhsTightening);
              }
            }

            if (success) {
              HighsCDouble roundLhs = ceil(lhs - primal_feastol);
              HighsCDouble roundRhs = floor(rhs + primal_feastol);

              // rounded row proves infeasibility regardless of coefficient
              // values
              if (roundRhs - roundLhs < -0.5) return Result::kPrimalInfeasible;

              if (roundLhs >= intScale * model->row_lower_[row] +
                                  minLhsTightening -
                                  options->small_matrix_value &&
                  roundRhs <= intScale * model->row_upper_[row] -
                                  minRhsTightening +
                                  options->small_matrix_value) {
                // scaled row with adjusted coefficients and sides is not weaker
                // than the original row
                if (maxVal <= 1000.0 || intScale <= 100.0) {
                  // printf(
                  //     "scaling constraint to integral values with scale %g, "
                  //     "rounded scaled sides from %g to %g and %g to %g\n",
                  //     intScale, double(rhs), double(roundRhs), double(lhs),
                  //     double(roundLhs));
                  // the scale value is reasonably small, change the row values
                  // to be integral
                  model->row_lower_[row] = double(roundLhs);
                  model->row_upper_[row] = double(roundRhs);
                  for (HighsInt i = 0; i < numRowCoefs; ++i)
                    addToMatrix(row, rowIndex[i],
                                rowCoefs[i] - Avalue[rowpositions[i]]);
                } else {
                  // scale value is large, just tighten the sides
                  roundLhs /= intScale;
                  roundRhs /= intScale;
                  if (roundRhs < model->row_upper_[row] - primal_feastol)
                    model->row_upper_[row] = double(roundRhs);
                  if (roundLhs > model->row_lower_[row] + primal_feastol)
                    model->row_lower_[row] = double(roundLhs);
                }
              }
            }
          }

          impliedRowUpper = impliedRowBounds.getSumUpper(row);
          impliedRowLower = impliedRowBounds.getSumLower(row);
        }
      }

      if (model->row_lower_[row] == -kHighsInf &&
          impliedRowUpper != kHighsInf) {
        HighsInt numTightened = 0;
        double maxCoefValue = impliedRowUpper - model->row_upper_[row];
        HighsCDouble rhs = model->row_upper_[row];
        for (const HighsSliceNonzero& nonz : getStoredRow()) {
          if (model->integrality_[nonz.index()] == HighsVarType::kContinuous)
            continue;

          if (nonz.value() > maxCoefValue + primal_feastol) {
            // <= contraint, we decrease the coefficient value and the right
            // hand side
            double delta = maxCoefValue - nonz.value();
            addToMatrix(row, nonz.index(), delta);
            rhs += delta * model->col_upper_[nonz.index()];
            ++numTightened;
          } else if (nonz.value() < -maxCoefValue - primal_feastol) {
            double delta = -maxCoefValue - nonz.value();
            addToMatrix(row, nonz.index(), delta);
            rhs += delta * model->col_lower_[nonz.index()];
            ++numTightened;
          }
        }

        model->row_upper_[row] = double(rhs);
      }

      if (model->row_upper_[row] == kHighsInf &&
          impliedRowLower != -kHighsInf) {
        HighsInt numTightened = 0;
        double maxCoefValue = model->row_lower_[row] - impliedRowLower;
        HighsCDouble rhs = model->row_lower_[row];
        for (const HighsSliceNonzero& nonz : getStoredRow()) {
          if (model->integrality_[nonz.index()] == HighsVarType::kContinuous)
            continue;

          if (nonz.value() > maxCoefValue + primal_feastol) {
            double delta = maxCoefValue - nonz.value();
            addToMatrix(row, nonz.index(), delta);
            rhs += delta * model->col_lower_[nonz.index()];
            ++numTightened;
          } else if (nonz.value() < -maxCoefValue - primal_feastol) {
            double delta = -maxCoefValue - nonz.value();
            addToMatrix(row, nonz.index(), delta);
            rhs += delta * model->col_upper_[nonz.index()];
            ++numTightened;
          }
        }

        model->row_lower_[row] = double(rhs);
      }
    }
  }

  impliedRowUpper = impliedRowBounds.getSumUpperOrig(row);
  impliedRowLower = impliedRowBounds.getSumLowerOrig(row);

  // printf("implied bounds without tightenings: [%g,%g]\n", baseiRLower,
  //        baseiRUpper);

  if (impliedRowUpper <=  // check for forcing row on the row lower bound
      model->row_lower_[row] + primal_feastol) {
    // the row upper bound that is implied by the column bounds is equal to
    // the row lower bound there for we can fix all columns at their bound
    // as this is the only feasible assignment for this row and then find a
    // suitable dual multiplier in postsolve. First we store the row on the
    // postsolve stack (forcingRow() call) afterwards we store each column
    // fixing on the postsolve stack. As the postsolve goes over the stack
    // in reverse, it will first restore the column primal and dual values
    // as the dual values are required to find the proper dual multiplier for
    // the row and the column that we put in the basis.
    storeRow(row);
    auto rowVector = getStoredRow();

    HighsInt nfixings = 0;
    for (const HighsSliceNonzero& nonzero : rowVector) {
      if (nonzero.value() > 0) {
        if (model->col_upper_[nonzero.index()] <= implColUpper[nonzero.index()])
          ++nfixings;
      } else {
        if (model->col_lower_[nonzero.index()] >= implColLower[nonzero.index()])
          ++nfixings;
      }
    }

    if (nfixings == rowsize[row]) {
      postsolve_stack.forcingRow(row, rowVector, model->row_lower_[row],
                                 HighsPostsolveStack::RowType::kGeq);
      // already mark the row as deleted, since otherwise it would be registered
      // as changed/singleton in the process of fixing and removing the
      // contained columns

      markRowDeleted(row);
      for (const HighsSliceNonzero& nonzero : rowVector) {
        if (nonzero.value() > 0) {
          // the upper bound of the column is as tight as the implied upper
          // bound or comes from this row, which means it is not used in the
          // rows implied bounds. Therefore we can fix the variable at its
          // upper bound.
          postsolve_stack.fixedColAtUpper(nonzero.index(),
                                          model->col_upper_[nonzero.index()],
                                          model->col_cost_[nonzero.index()],
                                          getColumnVector(nonzero.index()));
          if (model->col_lower_[nonzero.index()] <
              model->col_upper_[nonzero.index()])
            changeColLower(nonzero.index(), model->col_upper_[nonzero.index()]);
          removeFixedCol(nonzero.index());
        } else {
          postsolve_stack.fixedColAtLower(nonzero.index(),
                                          model->col_lower_[nonzero.index()],
                                          model->col_cost_[nonzero.index()],
                                          getColumnVector(nonzero.index()));

          if (model->col_upper_[nonzero.index()] >
              model->col_lower_[nonzero.index()])
            changeColUpper(nonzero.index(), model->col_lower_[nonzero.index()]);
          removeFixedCol(nonzero.index());
        }
      }
      // now the row might be empty, but not necessarily because the implied
      // column bounds might be implied by other rows in which case we cannot
      // fix the column
      postsolve_stack.redundantRow(row);

      HPRESOLVE_CHECKED_CALL(removeRowSingletons(postsolve_stack));
      return checkLimits(postsolve_stack);
    }
    // if there are any new row singletons, also remove them immediately
  } else if (impliedRowLower >= model->row_upper_[row] - primal_feastol) {
    // forcing row in the other direction
    storeRow(row);
    auto rowVector = getStoredRow();

    HighsInt nfixings = 0;
    for (const HighsSliceNonzero& nonzero : rowVector) {
      if (nonzero.value() < 0) {
        if (model->col_upper_[nonzero.index()] <= implColUpper[nonzero.index()])
          ++nfixings;
      } else {
        if (model->col_lower_[nonzero.index()] >= implColLower[nonzero.index()])
          ++nfixings;
      }
    }
    if (nfixings == rowsize[row]) {
      postsolve_stack.forcingRow(row, rowVector, model->row_upper_[row],
                                 HighsPostsolveStack::RowType::kLeq);
      markRowDeleted(row);
      for (const HighsSliceNonzero& nonzero : rowVector) {
        if (nonzero.value() < 0) {
          postsolve_stack.fixedColAtUpper(nonzero.index(),
                                          model->col_upper_[nonzero.index()],
                                          model->col_cost_[nonzero.index()],
                                          getColumnVector(nonzero.index()));
          if (model->col_lower_[nonzero.index()] <
              model->col_upper_[nonzero.index()])
            changeColLower(nonzero.index(), model->col_upper_[nonzero.index()]);

          removeFixedCol(nonzero.index());
        } else {
          postsolve_stack.fixedColAtLower(nonzero.index(),
                                          model->col_lower_[nonzero.index()],
                                          model->col_cost_[nonzero.index()],
                                          getColumnVector(nonzero.index()));
          if (model->col_upper_[nonzero.index()] >
              model->col_lower_[nonzero.index()])
            changeColUpper(nonzero.index(), model->col_lower_[nonzero.index()]);

          removeFixedCol(nonzero.index());
        }
      }

      postsolve_stack.redundantRow(row);

      HPRESOLVE_CHECKED_CALL(removeRowSingletons(postsolve_stack));
      return checkLimits(postsolve_stack);
    }
  }

  bool hasRowUpper =
      model->row_upper_[row] != kHighsInf ||
      implRowDualUpper[row] < -options->dual_feasibility_tolerance;
  bool hasRowLower =
      model->row_lower_[row] != kHighsInf ||
      implRowDualLower[row] > options->dual_feasibility_tolerance;

  if ((hasRowUpper && impliedRowBounds.getNumInfSumLowerOrig(row) <= 1) ||
      (hasRowLower && impliedRowBounds.getNumInfSumUpperOrig(row) <= 1)) {
    for (const HighsSliceNonzero& nonzero : getRowVector(row))
      updateColImpliedBounds(row, nonzero.index(), nonzero.value());
  }

  return checkLimits(postsolve_stack);
}

HPresolve::Result HPresolve::emptyCol(HighsPostsolveStack& postsolve_stack,
                                      HighsInt col) {
  if ((model->col_cost_[col] > 0 && model->col_lower_[col] == -kHighsInf) ||
      (model->col_cost_[col] < 0 && model->col_upper_[col] == kHighsInf)) {
    if (std::abs(model->col_cost_[col]) <= options->dual_feasibility_tolerance)
      model->col_cost_[col] = 0;
    else
      return Result::kDualInfeasible;
  }

  if (model->col_cost_[col] > 0)
    fixColToLower(postsolve_stack, col);
  else if (model->col_cost_[col] < 0 ||
           std::abs(model->col_upper_[col]) < std::abs(model->col_lower_[col]))
    fixColToUpper(postsolve_stack, col);
  else if (model->col_lower_[col] != -kHighsInf)
    fixColToLower(postsolve_stack, col);
  else
    fixColToZero(postsolve_stack, col);

  return checkLimits(postsolve_stack);
}

HPresolve::Result HPresolve::colPresolve(HighsPostsolveStack& postsolve_stack,
                                         HighsInt col) {
  assert(!colDeleted[col]);

  double boundDiff = model->col_upper_[col] - model->col_lower_[col];
  if (boundDiff <= primal_feastol) {
    if (boundDiff <= options->small_matrix_value ||
        getMaxAbsColVal(col) * boundDiff <= primal_feastol) {
      if (boundDiff < -primal_feastol) return Result::kPrimalInfeasible;
      postsolve_stack.removedFixedCol(col, model->col_lower_[col],
                                      model->col_cost_[col],
                                      getColumnVector(col));
      removeFixedCol(col);
      return checkLimits(postsolve_stack);
    }
  }

  switch (colsize[col]) {
    case 0:
      return emptyCol(postsolve_stack, col);
    case 1:
      return singletonCol(postsolve_stack, col);
    default:
      break;
  }

  double colDualUpper =
      -impliedDualRowBounds.getSumLower(col, -model->col_cost_[col]);
  double colDualLower =
      -impliedDualRowBounds.getSumUpper(col, -model->col_cost_[col]);

  // check for dominated column
  if (colDualLower > options->dual_feasibility_tolerance) {
    if (model->col_lower_[col] == -kHighsInf)
      return Result::kDualInfeasible;
    else {
      fixColToLower(postsolve_stack, col);
      HPRESOLVE_CHECKED_CALL(removeRowSingletons(postsolve_stack));
    }
    return checkLimits(postsolve_stack);
  }

  if (colDualUpper < -options->dual_feasibility_tolerance) {
    if (model->col_upper_[col] == kHighsInf)
      return Result::kDualInfeasible;
    else {
      fixColToUpper(postsolve_stack, col);
      HPRESOLVE_CHECKED_CALL(removeRowSingletons(postsolve_stack));
    }
    return checkLimits(postsolve_stack);
  }

  // check for weakly dominated column
  if (colDualUpper <= options->dual_feasibility_tolerance) {
    if (model->col_upper_[col] != kHighsInf) {
      fixColToUpper(postsolve_stack, col);
      HPRESOLVE_CHECKED_CALL(removeRowSingletons(postsolve_stack));
      return checkLimits(postsolve_stack);
    } else if (impliedDualRowBounds.getSumLowerOrig(col) == 0.0) {
      postsolve_stack.forcingColumn(col, getColumnVector(col),
                                    model->col_cost_[col],
                                    model->col_lower_[col], true);
      markColDeleted(col);
      HighsInt coliter = colhead[col];
      while (coliter != -1) {
        HighsInt row = Arow[coliter];
        double rhs = Avalue[coliter] > 0.0 ? model->row_lower_[row]
                                           : model->row_upper_[row];
        coliter = Anext[coliter];
        postsolve_stack.forcingColumnRemovedRow(col, row, rhs,
                                                getRowVector(row));
        removeRow(row);
      }
    }
  } else if (colDualLower >= -options->dual_feasibility_tolerance) {
    // symmetric case for fixing to the lower bound
    if (model->col_lower_[col] != -kHighsInf) {
      fixColToLower(postsolve_stack, col);
      HPRESOLVE_CHECKED_CALL(removeRowSingletons(postsolve_stack));
      return checkLimits(postsolve_stack);
    } else if (impliedDualRowBounds.getSumUpperOrig(col) == 0.0) {
      postsolve_stack.forcingColumn(col, getColumnVector(col),
                                    model->col_cost_[col],
                                    model->col_upper_[col], false);
      markColDeleted(col);
      HighsInt coliter = colhead[col];
      while (coliter != -1) {
        HighsInt row = Arow[coliter];
        double rhs = Avalue[coliter] > 0.0 ? model->row_upper_[row]
                                           : model->row_lower_[row];
        coliter = Anext[coliter];
        postsolve_stack.forcingColumnRemovedRow(col, row, rhs,
                                                getRowVector(row));
        removeRow(row);
      }
    }
  }

  // column is not (weakly) dominated

  // the associated dual constraint has an upper bound if there is an infinite
  // or redundant column lower bound as then the reduced cost of the column must
  // not be positive i.e. <= 0
  bool dualConsHasUpper = isUpperImplied(col);
  bool dualConsHasLower = isLowerImplied(col);

  // integer columns cannot be used to tighten bounds on dual multipliers
  if (mipsolver != nullptr) {
    if (dualConsHasLower && colLowerSource[col] != -1 &&
        impliedDualRowBounds.getNumInfSumUpperOrig(col) == 1 &&
        model->col_cost_[col] >= 0) {
      HighsInt row = colLowerSource[col];

      if (model->row_lower_[row] == -kHighsInf ||
          model->row_upper_[row] == kHighsInf) {
        HighsInt nzPos = findNonzero(row, col);

        if (model->integrality_[col] != HighsVarType::kInteger ||
            (rowsizeInteger[row] == rowsize[row] &&
             rowCoefficientsIntegral(row, 1.0 / Avalue[nzPos]))) {
          if (Avalue[nzPos] > 0)
            changeImplRowDualLower(row, 0.0, col);
          else
            changeImplRowDualUpper(row, 0.0, col);
        }
      }
    }

    if (dualConsHasUpper && colUpperSource[col] != -1 &&
        impliedDualRowBounds.getNumInfSumLowerOrig(col) == 1 &&
        model->col_cost_[col] <= 0) {
      HighsInt row = colUpperSource[col];

      if (model->row_lower_[row] == -kHighsInf ||
          model->row_upper_[row] == kHighsInf) {
        HighsInt nzPos = findNonzero(row, col);

        if (model->integrality_[col] != HighsVarType::kInteger ||
            (rowsizeInteger[row] == rowsize[row] &&
             rowCoefficientsIntegral(row, 1.0 / Avalue[nzPos]))) {
          if (Avalue[nzPos] > 0)
            changeImplRowDualUpper(row, 0.0, col);
          else
            changeImplRowDualLower(row, 0.0, col);
        }
      }
    }

    if (model->integrality_[col] == HighsVarType::kContinuous &&
        isImpliedInteger(col)) {
      model->integrality_[col] = HighsVarType::kImplicitInteger;
      for (const HighsSliceNonzero& nonzero : getColumnVector(col))
        ++rowsizeImplInt[nonzero.index()];
      double ceilLower = std::ceil(model->col_lower_[col] - primal_feastol);
      double floorUpper = std::floor(model->col_upper_[col] + primal_feastol);

      if (ceilLower > model->col_lower_[col]) changeColLower(col, ceilLower);
      if (floorUpper < model->col_upper_[col]) changeColUpper(col, floorUpper);
    }

    // shift integral variables to have a lower bound of zero
    if (model->integrality_[col] != HighsVarType::kContinuous &&
        model->col_lower_[col] != 0.0 &&
        (model->col_lower_[col] != -kHighsInf ||
         model->col_upper_[col] != kHighsInf) &&
        model->col_upper_[col] - model->col_lower_[col] > 0.5) {
      // substitute with the bound that is smaller in magnitude and only
      // suibstitute if bound is not large for an integer
      if (std::abs(model->col_upper_[col]) > std::abs(model->col_lower_[col])) {
        if (std::abs(model->col_lower_[col]) < 1000.5)
          transformColumn(postsolve_stack, col, 1.0, model->col_lower_[col]);
      } else {
        if (std::abs(model->col_upper_[col]) < 1000.5)
          transformColumn(postsolve_stack, col, -1.0, model->col_upper_[col]);
      }
    }

    if (model->integrality_[col] == HighsVarType::kInteger) return Result::kOk;
  }

  // now check if we can expect to tighten at least one bound
  if ((dualConsHasLower && impliedDualRowBounds.getNumInfSumUpper(col) <= 1) ||
      (dualConsHasUpper && impliedDualRowBounds.getNumInfSumLower(col) <= 1)) {
    for (const HighsSliceNonzero& nonzero : getColumnVector(col))
      updateRowDualImpliedBounds(nonzero.index(), col, nonzero.value());
  }

  return Result::kOk;
}

HPresolve::Result HPresolve::initialRowAndColPresolve(
    HighsPostsolveStack& postsolve_stack) {
  // do a full scan over the rows as the singleton arrays and the changed row
  // arrays are not initialized, also unset changedRowFlag so that the row will
  // be added to the changed row vector when it is changed after it was
  // processed
  for (HighsInt row = 0; row != model->num_row_; ++row) {
    if (rowDeleted[row]) continue;
    HPRESOLVE_CHECKED_CALL(rowPresolve(postsolve_stack, row));
    changedRowFlag[row] = false;
  }

  // same for the columns
  for (HighsInt col = 0; col != model->num_col_; ++col) {
    if (colDeleted[col]) continue;
    if (model->integrality_[col] != HighsVarType::kContinuous) {
      double ceilLower = std::ceil(model->col_lower_[col] - primal_feastol);
      double floorUpper = std::floor(model->col_upper_[col] + primal_feastol);

      if (ceilLower > model->col_lower_[col]) changeColLower(col, ceilLower);
      if (floorUpper < model->col_upper_[col]) changeColUpper(col, floorUpper);
    }
    HPRESOLVE_CHECKED_CALL(colPresolve(postsolve_stack, col));
    changedColFlag[col] = false;
  }

  return checkLimits(postsolve_stack);
}

HPresolve::Result HPresolve::fastPresolveLoop(
    HighsPostsolveStack& postsolve_stack) {
  do {
    storeCurrentProblemSize();

    HPRESOLVE_CHECKED_CALL(removeRowSingletons(postsolve_stack));

    HPRESOLVE_CHECKED_CALL(presolveChangedRows(postsolve_stack));

    HPRESOLVE_CHECKED_CALL(removeDoubletonEquations(postsolve_stack));

    HPRESOLVE_CHECKED_CALL(presolveColSingletons(postsolve_stack));

    HPRESOLVE_CHECKED_CALL(presolveChangedCols(postsolve_stack));

  } while (problemSizeReduction() > 0.01);

  return Result::kOk;
}

HPresolve::Result HPresolve::presolve(HighsPostsolveStack& postsolve_stack) {
  // for the inner most loop we take the order roughly from the old presolve
  // but we nest the rounds with a new outer loop which layers the newer
  // presolvers
  //    fast presolve loop
  //        - empty, forcing and dominated rows and row singletons immediately
  //        after each forcing row
  //        - doubleton equations and row singletons immediately after each
  //        successful substitution
  //        - col singletons (can this introduce row singletons? If yes then
  //        immediately remove)
  //        - empty, dominated and weakly dominated columns
  //        - row singletons
  //        - if( !has enough changes ) stop
  // main loop
  //    - fast presolve loop
  //    - parallel rows and columns
  //    - if (changes found) fast presolve loop
  //    - aggregator // add limit that catches many subsitutions but stops when
  //    many failures, do not run exhaustively as now
  //    - if (changes found) start main loop from beginning
  //    - primal and dual matrix sparsification
  //    - if (changes found) fast presolve loop
  //    - stop
  //

  // convert model to minimization problem
  if (model->sense_ == ObjSense::kMaximize) {
    for (HighsInt i = 0; i != model->num_col_; ++i)
      model->col_cost_[i] = -model->col_cost_[i];

    model->offset_ = -model->offset_;
    assert(std::isfinite(model->offset_));
    model->sense_ = ObjSense::kMinimize;
  }

  if (options->presolve != "off") {
    if (mipsolver) mipsolver->mipdata_->cliquetable.setPresolveFlag(true);
    if (!mipsolver || mipsolver->mipdata_->numRestarts == 0)
      highsLogUser(options->log_options, HighsLogType::kInfo,
                   "Presolving model\n");

    auto report = [&]() {
      if (!mipsolver || mipsolver->mipdata_->numRestarts == 0) {
        HighsInt numCol = model->num_col_ - numDeletedCols;
        HighsInt numRow = model->num_row_ - numDeletedRows;
        HighsInt numNonz = Avalue.size() - freeslots.size();
        highsLogUser(options->log_options, HighsLogType::kInfo,
                     "%" HIGHSINT_FORMAT " rows, %" HIGHSINT_FORMAT
                     " cols, %" HIGHSINT_FORMAT " nonzeros\n",
                     numRow, numCol, numNonz);
      }
    };

    HPRESOLVE_CHECKED_CALL(initialRowAndColPresolve(postsolve_stack));

    HighsInt numParallelRowColCalls = 0;
#if ENABLE_SPARSIFY_FOR_LP
    bool trySparsify = true;  // mipsolver != nullptr;
#else
    bool trySparsify =
        mipsolver != nullptr || !options->lp_presolve_requires_basis_postsolve;
#endif
    bool tryProbing = mipsolver != nullptr;
    HighsInt numCliquesBeforeProbing = -1;
    bool domcolAfterProbingCalled = false;
    bool dependentEquationsCalled = mipsolver != nullptr;
    HighsInt lastPrintSize = kHighsIInf;
    while (true) {
      HighsInt currSize =
          model->num_col_ - numDeletedCols + model->num_row_ - numDeletedRows;
      if (currSize < 0.85 * lastPrintSize) {
        lastPrintSize = currSize;
        report();
      }

      HPRESOLVE_CHECKED_CALL(fastPresolveLoop(postsolve_stack));

      storeCurrentProblemSize();

      // when presolving after a restart the clique table and implication
      // structure may contain substitutions which we apply directly before
      // running the aggregator as they might loose validity otherwise
      if (mipsolver != nullptr) {
        HPRESOLVE_CHECKED_CALL(
            applyConflictGraphSubstitutions(postsolve_stack));
      }

      HPRESOLVE_CHECKED_CALL(aggregator(postsolve_stack));

      if (problemSizeReduction() > 0.05) continue;

      if (trySparsify) {
        HighsInt numNz = numNonzeros();
        HPRESOLVE_CHECKED_CALL(sparsify(postsolve_stack));
        double nzReduction = 100.0 * (1.0 - (numNonzeros() / (double)numNz));

        if (nzReduction > 0) {
          highsLogDev(options->log_options, HighsLogType::kInfo,
                      "Sparsify removed %.1f%% of nonzeros\n", nzReduction);

          fastPresolveLoop(postsolve_stack);
        }
        trySparsify = false;
      }

      if (numParallelRowColCalls < 5) {
        if (shrinkProblemEnabled && (numDeletedCols >= 0.5 * model->num_col_ ||
                                     numDeletedRows >= 0.5 * model->num_row_)) {
          shrinkProblem(postsolve_stack);

          toCSC(model->a_matrix_.value_, model->a_matrix_.index_,
                model->a_matrix_.start_);
          fromCSC(model->a_matrix_.value_, model->a_matrix_.index_,
                  model->a_matrix_.start_);
        }
        storeCurrentProblemSize();
        HPRESOLVE_CHECKED_CALL(detectParallelRowsAndCols(postsolve_stack));
        ++numParallelRowColCalls;
        if (problemSizeReduction() > 0.05) continue;
      }

      HPRESOLVE_CHECKED_CALL(fastPresolveLoop(postsolve_stack));

      if (mipsolver != nullptr) {
        HighsInt numStrenghtened = strengthenInequalities();
        if (numStrenghtened > 0)
          highsLogDev(options->log_options, HighsLogType::kInfo,
                      "Strengthened %" HIGHSINT_FORMAT " coefficients\n",
                      numStrenghtened);
      }

      HPRESOLVE_CHECKED_CALL(fastPresolveLoop(postsolve_stack));

      if (mipsolver != nullptr && numCliquesBeforeProbing == -1) {
        numCliquesBeforeProbing = mipsolver->mipdata_->cliquetable.numCliques();
        storeCurrentProblemSize();
        HPRESOLVE_CHECKED_CALL(dominatedColumns(postsolve_stack));
        if (problemSizeReduction() > 0.0)
          HPRESOLVE_CHECKED_CALL(fastPresolveLoop(postsolve_stack));
        if (problemSizeReduction() > 0.05) continue;
      }

      if (tryProbing) {
        detectImpliedIntegers();
        storeCurrentProblemSize();
        HPRESOLVE_CHECKED_CALL(runProbing(postsolve_stack));
        tryProbing = probingContingent > numProbed &&
                     (problemSizeReduction() > 1.0 || probingEarlyAbort);
        trySparsify = true;
        if (problemSizeReduction() > 0.05 || tryProbing) continue;
        HPRESOLVE_CHECKED_CALL(fastPresolveLoop(postsolve_stack));
      }

      if (!dependentEquationsCalled) {
        if (shrinkProblemEnabled && (numDeletedCols >= 0.5 * model->num_col_ ||
                                     numDeletedRows >= 0.5 * model->num_row_)) {
          shrinkProblem(postsolve_stack);

          toCSC(model->a_matrix_.value_, model->a_matrix_.index_,
                model->a_matrix_.start_);
          fromCSC(model->a_matrix_.value_, model->a_matrix_.index_,
                  model->a_matrix_.start_);
        }
        storeCurrentProblemSize();
        HPRESOLVE_CHECKED_CALL(removeDependentEquations(postsolve_stack));
        HPRESOLVE_CHECKED_CALL(removeDependentFreeCols(postsolve_stack));
        dependentEquationsCalled = true;
        if (problemSizeReduction() > 0.05) continue;
      }

      if (mipsolver != nullptr &&
          mipsolver->mipdata_->cliquetable.numCliques() >
              numCliquesBeforeProbing &&
          !domcolAfterProbingCalled) {
        domcolAfterProbingCalled = true;
        storeCurrentProblemSize();
        HPRESOLVE_CHECKED_CALL(dominatedColumns(postsolve_stack));
        if (problemSizeReduction() > 0.0)
          HPRESOLVE_CHECKED_CALL(fastPresolveLoop(postsolve_stack));
        if (problemSizeReduction() > 0.05) continue;
      }

      break;
    }

    report();
  } else {
    highsLogUser(options->log_options, HighsLogType::kInfo,
                 "\nPresolve is switched off\n");
  }

  if (mipsolver != nullptr) scaleMIP(postsolve_stack);

  //  if (options->log_options.log_dev_level)
  //    reportReductions(options->log_options);

  return Result::kOk;
}

HPresolve::Result HPresolve::checkLimits(HighsPostsolveStack& postsolve_stack) {
  // todo: check timelimit
  size_t numreductions = postsolve_stack.numReductions();

  if (timer != nullptr && (numreductions & 1023u) == 0) {
    if (timer->readRunHighsClock() >= options->time_limit)
      return Result::kStopped;
  }

  return numreductions >= reductionLimit ? Result::kStopped : Result::kOk;
}

void HPresolve::storeCurrentProblemSize() {
  oldNumCol = model->num_col_ - numDeletedCols;
  oldNumRow = model->num_row_ - numDeletedRows;
}

double HPresolve::problemSizeReduction() {
  double colReduction = 100.0 *
                        double(oldNumCol - (model->num_col_ - numDeletedCols)) /
                        oldNumCol;
  double rowReduction = 100.0 *
                        double(oldNumRow - (model->num_row_ - numDeletedRows)) /
                        oldNumRow;

  return std::max(rowReduction, colReduction);
}

HighsModelStatus HPresolve::run(HighsPostsolveStack& postsolve_stack) {
  shrinkProblemEnabled = true;
  switch (presolve(postsolve_stack)) {
    case Result::kStopped:
    case Result::kOk:
      break;
    case Result::kPrimalInfeasible:
      return HighsModelStatus::kInfeasible;
    case Result::kDualInfeasible:
      return HighsModelStatus::kUnboundedOrInfeasible;
  }

  shrinkProblem(postsolve_stack);

  if (mipsolver != nullptr) {
    mipsolver->mipdata_->cliquetable.setPresolveFlag(false);
    mipsolver->mipdata_->cliquetable.setMaxEntries(numNonzeros());
    mipsolver->mipdata_->domain.addCutpool(mipsolver->mipdata_->cutpool);
    mipsolver->mipdata_->domain.addConflictPool(
        mipsolver->mipdata_->conflictPool);

    if (mipsolver->mipdata_->numRestarts != 0) {
      std::vector<HighsInt> cutinds;
      std::vector<double> cutvals;
      cutinds.reserve(model->num_col_);
      cutvals.reserve(model->num_col_);
      HighsInt numcuts = 0;
      for (HighsInt i = model->num_row_ - 1; i >= 0; --i) {
        // check if we already reached the original rows
        if (postsolve_stack.getOrigRowIndex(i) <
            mipsolver->orig_model_->num_row_)
          break;

        // row is a cut, remove it from matrix but add to cutpool
        ++numcuts;
        storeRow(i);
        cutinds.clear();
        cutvals.clear();
        for (HighsInt j : rowpositions) {
          cutinds.push_back(Acol[j]);
          cutvals.push_back(Avalue[j]);
        }

        mipsolver->mipdata_->cutpool.addCut(
            *mipsolver, cutinds.data(), cutvals.data(), cutinds.size(),
            model->row_upper_[i],
            rowsizeInteger[i] + rowsizeImplInt[i] == rowsize[i] &&
                rowCoefficientsIntegral(i, 1.0),
            true, false, false);

        markRowDeleted(i);
        for (HighsInt j : rowpositions) unlink(j);
      }

      model->num_row_ -= numcuts;
      model->row_lower_.resize(model->num_row_);
      model->row_upper_.resize(model->num_row_);
      model->row_names_.resize(model->num_row_);
    }
  }

  toCSC(model->a_matrix_.value_, model->a_matrix_.index_,
        model->a_matrix_.start_);

  if (model->num_col_ == 0) {
    if (mipsolver) {
      if (model->offset_ > mipsolver->mipdata_->upper_limit)
        return HighsModelStatus::kInfeasible;

      mipsolver->mipdata_->lower_bound = 0;
    } else {
      assert(model->num_row_ == 0);
      if (model->num_row_ != 0) return HighsModelStatus::kNotset;
    }
    return HighsModelStatus::kOptimal;
  }

  if (!mipsolver && options->use_implied_bounds_from_presolve)
    setRelaxedImpliedBounds();

  return HighsModelStatus::kNotset;
}

void HPresolve::computeIntermediateMatrix(std::vector<HighsInt>& flagRow,
                                          std::vector<HighsInt>& flagCol,
                                          size_t& numreductions) {
  shrinkProblemEnabled = false;
  HighsPostsolveStack stack;
  stack.initializeIndexMaps(flagRow.size(), flagCol.size());
  setReductionLimit(numreductions);
  presolve(stack);
  numreductions = stack.numReductions();

  toCSC(model->a_matrix_.value_, model->a_matrix_.index_,
        model->a_matrix_.start_);

  for (HighsInt i = 0; i != model->num_row_; ++i)
    flagRow[i] = 1 - rowDeleted[i];
  for (HighsInt i = 0; i != model->num_col_; ++i)
    flagCol[i] = 1 - colDeleted[i];
}

HPresolve::Result HPresolve::removeDependentEquations(
    HighsPostsolveStack& postsolve_stack) {
  if (equations.empty()) return Result::kOk;

  HighsSparseMatrix matrix;
  matrix.num_col_ = equations.size();
  // printf("got %d equations, checking for dependent equations\n",
  //        (int)matrix.num_col_);
  matrix.num_row_ = model->num_col_ + 1;
  matrix.start_.resize(matrix.num_col_ + 1);
  matrix.start_[0] = 0;
  const HighsInt maxCapacity = numNonzeros() + matrix.num_col_;
  matrix.value_.reserve(maxCapacity);
  matrix.index_.reserve(maxCapacity);

  std::vector<HighsInt> eqSet(matrix.num_col_);
  HighsInt i = 0;
  for (const std::pair<HighsInt, HighsInt>& p : equations) {
    HighsInt eq = p.second;
    eqSet[i++] = eq;

    // add entries of equation
    for (const HighsSliceNonzero& nonz : getRowVector(eq)) {
      matrix.value_.push_back(nonz.value());
      matrix.index_.push_back(nonz.index());
    }

    // add entry for artifical rhs column
    if (model->row_lower_[eq] != 0.0) {
      matrix.value_.push_back(model->row_lower_[eq]);
      matrix.index_.push_back(model->num_col_);
    }

    matrix.start_[i] = matrix.value_.size();
  }
  std::vector<HighsInt> colSet(matrix.num_col_);
  std::iota(colSet.begin(), colSet.end(), 0);
  HFactor factor;
  factor.setup(matrix, colSet);
  HighsInt rank_deficiency = factor.build();
  HighsInt num_removed_row = 0;
  HighsInt num_removed_nz = 0;
  HighsInt num_fictitious_rows_skipped = 0;
  for (HighsInt k = 0; k < rank_deficiency; k++) {
    if (factor.var_with_no_pivot[k] >= 0) {
      HighsInt redundant_row = eqSet[factor.var_with_no_pivot[k]];
      num_removed_row++;
      num_removed_nz += rowsize[redundant_row];
      postsolve_stack.redundantRow(redundant_row);
      removeRow(redundant_row);
    } else {
      num_fictitious_rows_skipped++;
    }
  }

  highsLogDev(
      options->log_options, HighsLogType::kInfo,
      "HPresolve::removeDependentEquations Removed %d rows and %d nonzeros",
      (int)num_removed_row, (int)num_removed_nz);
  if (num_fictitious_rows_skipped)
    highsLogDev(options->log_options, HighsLogType::kInfo,
                ", avoiding %d fictitious rows",
                (int)num_fictitious_rows_skipped);
  highsLogDev(options->log_options, HighsLogType::kInfo, "\n");

  return Result::kOk;
}

HPresolve::Result HPresolve::removeDependentFreeCols(
    HighsPostsolveStack& postsolve_stack) {
  return Result::kOk;

  // todo the postsolve step does not work properly
  std::vector<HighsInt> freeCols;
  freeCols.reserve(model->num_col_);

  for (HighsInt i = 0; i < model->num_col_; ++i) {
    if (colDeleted[i]) continue;
    if (model->col_lower_[i] == -kHighsInf && model->col_upper_[i] == kHighsInf)
      freeCols.push_back(i);
  }

  if (freeCols.empty()) return Result::kOk;

  HighsSparseMatrix matrix;
  matrix.num_col_ = freeCols.size();
  highsLogDev(options->log_options, HighsLogType::kInfo,
              "HPresolve::removeDependentFreeCols Got %d free cols, checking "
              "for dependent free cols\n",
              (int)matrix.num_col_);
  matrix.num_row_ = model->num_row_ + 1;
  matrix.start_.resize(matrix.num_col_ + 1);
  matrix.start_[0] = 0;
  const HighsInt maxCapacity = numNonzeros() + matrix.num_col_;
  matrix.value_.reserve(maxCapacity);
  matrix.index_.reserve(maxCapacity);

  for (HighsInt i = 0; i < matrix.num_col_; ++i) {
    HighsInt col = freeCols[i];
    // add entries of free column
    for (const HighsSliceNonzero& nonz : getColumnVector(col)) {
      matrix.value_.push_back(nonz.value());
      matrix.index_.push_back(nonz.index());
    }

    // add entry for artifical cost row
    if (model->col_cost_[col] != 0.0) {
      matrix.value_.push_back(model->col_cost_[col]);
      matrix.index_.push_back(model->num_row_);
    }

    matrix.start_[i + 1] = matrix.value_.size();
  }
  std::vector<HighsInt> colSet(matrix.num_col_);
  std::iota(colSet.begin(), colSet.end(), 0);
  HFactor factor;
  factor.setup(matrix, colSet);
  HighsInt rank_deficiency = factor.build();
  HighsInt num_removed_row = 0;
  HighsInt num_removed_nz = 0;
  HighsInt num_fictitious_cols_skipped = 0;
  for (HighsInt k = 0; k < rank_deficiency; k++) {
    if (factor.var_with_no_pivot[k] >= 0) {
      HighsInt redundant_col = freeCols[factor.var_with_no_pivot[k]];
      num_removed_nz += colsize[redundant_col];
      fixColToZero(postsolve_stack, redundant_col);
    } else {
      num_fictitious_cols_skipped++;
    }
  }
  highsLogDev(
      options->log_options, HighsLogType::kInfo,
      "HPresolve::removeDependentFreeCols Removed %d rows and %d nonzeros",
      (int)num_removed_row, (int)num_removed_nz);
  if (num_fictitious_cols_skipped)
    highsLogDev(options->log_options, HighsLogType::kInfo,
                ", avoiding %d fictitious rows",
                (int)num_fictitious_cols_skipped);
  highsLogDev(options->log_options, HighsLogType::kInfo, "\n");

  return Result::kOk;
}

HPresolve::Result HPresolve::aggregator(HighsPostsolveStack& postsolve_stack) {
  HighsInt numsubst = 0;
  HighsInt numsubstint = 0;
  substitutionOpportunities.erase(
      std::remove_if(substitutionOpportunities.begin(),
                     substitutionOpportunities.end(),
                     [&](const std::pair<HighsInt, HighsInt>& p) {
                       HighsInt row = p.first;
                       HighsInt col = p.second;
                       return rowDeleted[row] || colDeleted[col] ||
                              !isImpliedFree(col) || !isDualImpliedFree(row);
                     }),
      substitutionOpportunities.end());

  pdqsort(
      substitutionOpportunities.begin(), substitutionOpportunities.end(),
      [&](const std::pair<HighsInt, HighsInt>& nz1,
          const std::pair<HighsInt, HighsInt>& nz2) {
        HighsInt minLen1 = std::min(rowsize[nz1.first], colsize[nz1.second]);
        HighsInt minLen2 = std::min(rowsize[nz2.first], colsize[nz2.second]);
        if (minLen1 == 2 && minLen2 != 2) return true;
        if (minLen2 == 2 && minLen1 != 2) return false;

        int64_t sizeProd1 = int64_t(rowsize[nz1.first]) * colsize[nz1.second];
        int64_t sizeProd2 = int64_t(rowsize[nz2.first]) * colsize[nz2.second];
        if (sizeProd1 < sizeProd2) return true;
        if (sizeProd2 < sizeProd1) return false;
        if (minLen1 < minLen2) return true;
        if (minLen2 < minLen1) return false;

        return std::make_tuple(HighsHashHelpers::hash(std::make_pair(
                                   uint32_t(nz1.first), uint32_t(nz1.second))),
                               nz1.first, nz1.second) <
               std::make_tuple(HighsHashHelpers::hash(std::make_pair(
                                   uint32_t(nz2.first), uint32_t(nz2.second))),
                               nz2.first, nz2.second);
      });

  HighsInt nfail = 0;
  for (size_t i = 0; i < substitutionOpportunities.size(); ++i) {
    HighsInt row = substitutionOpportunities[i].first;
    HighsInt col = substitutionOpportunities[i].second;

    if (rowDeleted[row] || colDeleted[col] || !isImpliedFree(col) ||
        !isDualImpliedFree(row)) {
      substitutionOpportunities[i].first = -1;
      continue;
    }

    HighsInt nzPos = findNonzero(row, col);
    if (nzPos == -1) {
      substitutionOpportunities[i].first = -1;
      continue;
    }
    if (model->integrality_[col] == HighsVarType::kInteger) {
      bool impliedIntegral =
          (rowsizeInteger[row] == rowsize[row] &&
           rowCoefficientsIntegral(row, 1.0 / Avalue[nzPos])) ||
          isImpliedIntegral(col);
      if (!impliedIntegral) continue;
    }

    // in the case where the row has length two or the column has length two
    // we always do the substitution since the fillin can never be problematic
    if (rowsize[row] == 2 || colsize[col] == 2) {
      double rhs;
      HighsPostsolveStack::RowType rowType;
      if (model->row_lower_[row] == model->row_upper_[row]) {
        rowType = HighsPostsolveStack::RowType::kEq;
        rhs = model->row_upper_[row];
      } else if ((model->row_upper_[row] != kHighsInf &&
                  implRowDualUpper[row] <=
                      options->dual_feasibility_tolerance)) {
        rowType = HighsPostsolveStack::RowType::kLeq;
        rhs = model->row_upper_[row];
        changeRowDualUpper(row, kHighsInf);
      } else {
        rowType = HighsPostsolveStack::RowType::kGeq;
        rhs = model->row_lower_[row];
        changeRowDualLower(row, -kHighsInf);
      }

      ++numsubst;
      if (model->integrality_[col] == HighsVarType::kInteger) ++numsubstint;
      storeRow(row);

      postsolve_stack.freeColSubstitution(row, col, rhs, model->col_cost_[col],
                                          rowType, getStoredRow(),
                                          getColumnVector(col));
      substitutionOpportunities[i].first = -1;

      substitute(row, col, rhs);
      HPRESOLVE_CHECKED_CALL(removeRowSingletons(postsolve_stack));
      HPRESOLVE_CHECKED_CALL(checkLimits(postsolve_stack));
      continue;
    }

    double maxVal = rowsize[row] < colsize[col] ? getMaxAbsRowVal(row)
                                                : getMaxAbsColVal(col);
    if (std::fabs(Avalue[nzPos]) < maxVal * options->presolve_pivot_threshold) {
      maxVal = rowsize[row] < colsize[col] ? getMaxAbsColVal(col)
                                           : getMaxAbsRowVal(row);
      if (std::fabs(Avalue[nzPos]) <
          maxVal * options->presolve_pivot_threshold) {
        substitutionOpportunities[i].first = -1;
        continue;
      }
    }

    storeRow(row);
    HighsInt fillin = -(rowsize[row] + colsize[col] - 1);
    for (const auto& nz : getColumnVector(col)) {
      if (nz.index() == row) continue;
      fillin += countFillin(nz.index());

      if (fillin > options->presolve_substitution_maxfillin) break;
    }

    if (fillin > options->presolve_substitution_maxfillin) {
      ++nfail;
      // if the fill in is too much for multiple tries, then we stop
      // as this indicates that the rows/columns are becoming too dense
      // for substitutions
      if (nfail == 3) break;
      continue;
    }

    nfail = 0;
    ++numsubst;
    if (model->integrality_[col] == HighsVarType::kInteger) ++numsubstint;
    double rhs;
    HighsPostsolveStack::RowType rowType;
    if (model->row_lower_[row] == model->row_upper_[row]) {
      rowType = HighsPostsolveStack::RowType::kEq;
      rhs = model->row_upper_[row];
    } else if ((model->row_upper_[row] != kHighsInf &&
                implRowDualUpper[row] <= options->dual_feasibility_tolerance)) {
      rowType = HighsPostsolveStack::RowType::kLeq;
      rhs = model->row_upper_[row];
      changeRowDualUpper(row, kHighsInf);
    } else {
      rowType = HighsPostsolveStack::RowType::kGeq;
      rhs = model->row_lower_[row];
      changeRowDualLower(row, -kHighsInf);
    }

    postsolve_stack.freeColSubstitution(row, col, rhs, model->col_cost_[col],
                                        rowType, getStoredRow(),
                                        getColumnVector(col));
    substitutionOpportunities[i].first = -1;
    substitute(row, col, rhs);
    HPRESOLVE_CHECKED_CALL(removeRowSingletons(postsolve_stack));
    HPRESOLVE_CHECKED_CALL(checkLimits(postsolve_stack));
  }

  substitutionOpportunities.erase(
      std::remove_if(
          substitutionOpportunities.begin(), substitutionOpportunities.end(),
          [](const std::pair<HighsInt, HighsInt>& p) { return p.first == -1; }),
      substitutionOpportunities.end());

  return Result::kOk;
}

void HPresolve::substitute(HighsInt substcol, HighsInt staycol, double offset,
                           double scale) {
  // substitute the column in each row where it occurs
  for (HighsInt coliter = colhead[substcol]; coliter != -1;) {
    HighsInt colrow = Arow[coliter];
    double colval = Avalue[coliter];
    // walk to the next position before doing any modifications, because
    // the current position will be deleted in the loop below
    assert(Acol[coliter] == substcol);
    HighsInt colpos = coliter;
    coliter = Anext[coliter];
    assert(!rowDeleted[colrow]);
    unlink(colpos);

    // adjust the sides
    if (model->row_lower_[colrow] != -kHighsInf)
      model->row_lower_[colrow] -= colval * offset;

    if (model->row_upper_[colrow] != kHighsInf)
      model->row_upper_[colrow] -= colval * offset;

    addToMatrix(colrow, staycol, scale * colval);
    // printf("after substitution: ");
    // debugPrintRow(colrow);

    // check if this is an equation row and it now has a different size
    if (model->row_lower_[colrow] == model->row_upper_[colrow] &&
        eqiters[colrow] != equations.end() &&
        eqiters[colrow]->first != rowsize[colrow]) {
      // if that is the case reinsert it into the equation set that is ordered
      // by sparsity
      equations.erase(eqiters[colrow]);
      eqiters[colrow] = equations.emplace(rowsize[colrow], colrow).first;
    }
  }

  // substitute column in the objective function
  if (model->col_cost_[substcol] != 0.0) {
    model->offset_ += model->col_cost_[substcol] * offset;
    assert(std::isfinite(model->offset_));

    model->col_cost_[staycol] += scale * model->col_cost_[substcol];

    if (std::abs(model->col_cost_[staycol]) <= options->small_matrix_value)
      model->col_cost_[staycol] = 0.0;
    model->col_cost_[substcol] = 0.0;
  }
}

void HPresolve::fixColToLower(HighsPostsolveStack& postsolve_stack,
                              HighsInt col) {
  double fixval = model->col_lower_[col];
  assert(fixval != -kHighsInf);

  // printf("fixing column %" HIGHSINT_FORMAT " to %.15g\n", col, fixval);

  // mark the column as deleted first so that it is not registered as singleton
  // column upon removing its nonzeros
  postsolve_stack.fixedColAtLower(col, fixval, model->col_cost_[col],
                                  getColumnVector(col));
  markColDeleted(col);

  for (HighsInt coliter = colhead[col]; coliter != -1;) {
    HighsInt colrow = Arow[coliter];
    double colval = Avalue[coliter];
    assert(Acol[coliter] == col);

    HighsInt colpos = coliter;
    coliter = Anext[coliter];

    if (model->row_lower_[colrow] != -kHighsInf)
      model->row_lower_[colrow] -= colval * fixval;

    if (model->row_upper_[colrow] != kHighsInf)
      model->row_upper_[colrow] -= colval * fixval;

    unlink(colpos);

    if (model->row_lower_[colrow] == model->row_upper_[colrow] &&
        eqiters[colrow] != equations.end() &&
        eqiters[colrow]->first != rowsize[colrow]) {
      // if that is the case reinsert it into the equation set that is ordered
      // by sparsity
      equations.erase(eqiters[colrow]);
      eqiters[colrow] = equations.emplace(rowsize[colrow], colrow).first;
    }
  }

  model->offset_ += model->col_cost_[col] * fixval;
  assert(std::isfinite(model->offset_));
  model->col_cost_[col] = 0;
}

void HPresolve::fixColToUpper(HighsPostsolveStack& postsolve_stack,
                              HighsInt col) {
  double fixval = model->col_upper_[col];
  assert(fixval != kHighsInf);
  // printf("fixing column %" HIGHSINT_FORMAT " to %.15g\n", col, fixval);

  // mark the column as deleted first so that it is not registered as singleton
  // column upon removing its nonzeros
  postsolve_stack.fixedColAtUpper(col, fixval, model->col_cost_[col],
                                  getColumnVector(col));
  markColDeleted(col);

  for (HighsInt coliter = colhead[col]; coliter != -1;) {
    HighsInt colrow = Arow[coliter];
    double colval = Avalue[coliter];
    assert(Acol[coliter] == col);

    HighsInt colpos = coliter;
    coliter = Anext[coliter];

    if (model->row_lower_[colrow] != -kHighsInf)
      model->row_lower_[colrow] -= colval * fixval;

    if (model->row_upper_[colrow] != kHighsInf)
      model->row_upper_[colrow] -= colval * fixval;

    unlink(colpos);

    if (model->row_lower_[colrow] == model->row_upper_[colrow] &&
        eqiters[colrow] != equations.end() &&
        eqiters[colrow]->first != rowsize[colrow]) {
      // if that is the case reinsert it into the equation set that is ordered
      // by sparsity
      equations.erase(eqiters[colrow]);
      eqiters[colrow] = equations.emplace(rowsize[colrow], colrow).first;
    }
  }

  model->offset_ += model->col_cost_[col] * fixval;
  assert(std::isfinite(model->offset_));
  model->col_cost_[col] = 0;
}

void HPresolve::fixColToZero(HighsPostsolveStack& postsolve_stack,
                             HighsInt col) {
  postsolve_stack.fixedColAtZero(col, model->col_cost_[col],
                                 getColumnVector(col));
  // mark the column as deleted first so that it is not registered as singleton
  // column upon removing its nonzeros
  markColDeleted(col);

  for (HighsInt coliter = colhead[col]; coliter != -1;) {
    HighsInt colrow = Arow[coliter];
    assert(Acol[coliter] == col);

    HighsInt colpos = coliter;
    coliter = Anext[coliter];

    unlink(colpos);

    if (model->row_lower_[colrow] == model->row_upper_[colrow] &&
        eqiters[colrow] != equations.end() &&
        eqiters[colrow]->first != rowsize[colrow]) {
      // if that is the case reinsert it into the equation set that is ordered
      // by sparsity
      equations.erase(eqiters[colrow]);
      eqiters[colrow] = equations.emplace(rowsize[colrow], colrow).first;
    }
  }

  model->col_cost_[col] = 0;
}

void HPresolve::removeRow(HighsInt row) {
  assert(row < int(rowroot.size()));
  assert(row >= 0);
  // first mark the row as logically deleted, so that it is not register as
  // singleton row upon removing its nonzeros
  markRowDeleted(row);
  storeRow(row);
  for (HighsInt rowiter : rowpositions) {
    assert(Arow[rowiter] == row);
    unlink(rowiter);
  }
}

void HPresolve::removeFixedCol(HighsInt col) {
  double fixval = model->col_lower_[col];

  markColDeleted(col);

  for (HighsInt coliter = colhead[col]; coliter != -1;) {
    HighsInt colrow = Arow[coliter];
    double colval = Avalue[coliter];
    assert(Acol[coliter] == col);

    HighsInt colpos = coliter;
    coliter = Anext[coliter];

    if (model->row_lower_[colrow] != -kHighsInf)
      model->row_lower_[colrow] -= colval * fixval;

    if (model->row_upper_[colrow] != kHighsInf)
      model->row_upper_[colrow] -= colval * fixval;

    unlink(colpos);

    if (model->row_lower_[colrow] == model->row_upper_[colrow] &&
        eqiters[colrow] != equations.end() &&
        eqiters[colrow]->first != rowsize[colrow]) {
      // if that is the case reinsert it into the equation set that is ordered
      // by sparsity
      equations.erase(eqiters[colrow]);
      eqiters[colrow] = equations.emplace(rowsize[colrow], colrow).first;
    }
  }

  model->offset_ += model->col_cost_[col] * fixval;
  assert(std::isfinite(model->offset_));
  model->col_cost_[col] = 0;
}

HPresolve::Result HPresolve::removeRowSingletons(
    HighsPostsolveStack& postsolve_stack) {
  for (size_t i = 0; i != singletonRows.size(); ++i) {
    HighsInt row = singletonRows[i];
    if (rowDeleted[row] || rowsize[row] > 1) continue;
    // row presolve will delegate to rowSingleton() if the row size is 1
    // if the singleton row has become empty it will also remove the row
    HPRESOLVE_CHECKED_CALL(rowPresolve(postsolve_stack, row));
  }

  singletonRows.clear();

  return Result::kOk;
}

HPresolve::Result HPresolve::presolveColSingletons(
    HighsPostsolveStack& postsolve_stack) {
  for (size_t i = 0; i != singletonColumns.size(); ++i) {
    HighsInt col = singletonColumns[i];
    if (colDeleted[col]) continue;
    HPRESOLVE_CHECKED_CALL(colPresolve(postsolve_stack, col));
  }
  singletonColumns.erase(
      std::remove_if(
          singletonColumns.begin(), singletonColumns.end(),
          [&](HighsInt col) { return colDeleted[col] || colsize[col] > 1; }),
      singletonColumns.end());

  return Result::kOk;
}

HPresolve::Result HPresolve::presolveChangedRows(
    HighsPostsolveStack& postsolve_stack) {
  std::vector<HighsInt> changedRows;
  changedRows.reserve(model->num_row_ - numDeletedRows);
  changedRows.swap(changedRowIndices);
  for (HighsInt row : changedRows) {
    if (rowDeleted[row]) continue;
    HPRESOLVE_CHECKED_CALL(rowPresolve(postsolve_stack, row));
    changedRowFlag[row] = rowDeleted[row];
  }

  return Result::kOk;
}

HPresolve::Result HPresolve::presolveChangedCols(
    HighsPostsolveStack& postsolve_stack) {
  std::vector<HighsInt> changedCols;
  changedCols.reserve(model->num_col_ - numDeletedCols);
  changedCols.swap(changedColIndices);
  for (HighsInt col : changedCols) {
    if (colDeleted[col]) continue;
    HPRESOLVE_CHECKED_CALL(colPresolve(postsolve_stack, col));
    changedColFlag[col] = colDeleted[col];
  }

  return Result::kOk;
}

HPresolve::Result HPresolve::removeDoubletonEquations(
    HighsPostsolveStack& postsolve_stack) {
  auto eq = equations.begin();
  while (eq != equations.end()) {
    HighsInt eqrow = eq->second;
    assert(!rowDeleted[eqrow]);
    assert(eq->first == rowsize[eqrow]);
    assert(model->row_lower_[eqrow] == model->row_upper_[eqrow]);
    if (rowsize[eqrow] > 2) return Result::kOk;
    HPRESOLVE_CHECKED_CALL(rowPresolve(postsolve_stack, eqrow));
    if (rowDeleted[eqrow])
      eq = equations.begin();
    else
      ++eq;
  }

  return Result::kOk;
}

HighsInt HPresolve::strengthenInequalities() {
  std::vector<int8_t> complementation;
  std::vector<double> reducedcost;
  std::vector<double> upper;
  std::vector<HighsInt> indices;
  std::vector<HighsInt> positions;
  std::vector<HighsInt> stack;
  std::vector<double> coefs;
  std::vector<HighsInt> cover;

  HighsInt numstrenghtened = 0;

  for (HighsInt row = 0; row != model->num_row_; ++row) {
    if (rowsize[row] <= 1) continue;
    if (model->row_lower_[row] != -kHighsInf &&
        model->row_upper_[row] != kHighsInf)
      continue;

    // do not run on very dense rows as this could get expensive
    if (rowsize[row] >
        std::max(HighsInt{1000},
                 HighsInt(0.05 * (model->num_col_ - numDeletedCols))))
      continue;

    // printf("strengthening knapsack of %" HIGHSINT_FORMAT " vars\n",
    // rowsize[row]);

    HighsCDouble maxviolation;
    HighsCDouble continuouscontribution = 0.0;
    double scale;

    if (model->row_lower_[row] != -kHighsInf) {
      maxviolation = model->row_lower_[row];
      scale = -1.0;
    } else {
      maxviolation = -model->row_upper_[row];
      scale = 1.0;
    }

    complementation.clear();
    reducedcost.clear();
    upper.clear();
    indices.clear();
    positions.clear();
    complementation.reserve(rowsize[row]);
    reducedcost.reserve(rowsize[row]);
    upper.reserve(rowsize[row]);
    indices.reserve(rowsize[row]);
    stack.reserve(rowsize[row]);
    stack.push_back(rowroot[row]);

    bool skiprow = false;

    while (!stack.empty()) {
      HighsInt pos = stack.back();
      stack.pop_back();

      if (ARright[pos] != -1) stack.push_back(ARright[pos]);
      if (ARleft[pos] != -1) stack.push_back(ARleft[pos]);

      int8_t comp;
      double weight;
      double ub;
      weight = Avalue[pos] * scale;
      HighsInt col = Acol[pos];
      ub = model->col_upper_[col] - model->col_lower_[col];

      if (ub == kHighsInf) {
        skiprow = true;
        break;
      }

      if (weight > 0) {
        if (model->col_upper_[col] == kHighsInf) {
          skiprow = true;
          break;
        }

        comp = 1;
        maxviolation += model->col_upper_[col] * weight;
      } else {
        if (model->col_lower_[col] == -kHighsInf) {
          skiprow = true;
          break;
        }
        comp = -1;
        maxviolation += model->col_lower_[col] * weight;
        weight = -weight;
      }

      if (ub <= primal_feastol || weight <= primal_feastol) continue;

      if (model->integrality_[col] == HighsVarType::kContinuous) {
        continuouscontribution += weight * ub;
        continue;
      }

      indices.push_back(reducedcost.size());
      positions.push_back(pos);
      reducedcost.push_back(weight);
      complementation.push_back(comp);
      upper.push_back(ub);
    }

    if (skiprow) {
      stack.clear();
      continue;
    }

    const double smallVal =
        std::max(100 * primal_feastol, primal_feastol * double(maxviolation));
    while (true) {
      if (maxviolation - continuouscontribution <= smallVal || indices.empty())
        break;

      pdqsort(indices.begin(), indices.end(), [&](HighsInt i1, HighsInt i2) {
        return std::make_pair(reducedcost[i1], i1) >
               std::make_pair(reducedcost[i2], i2);
      });

      HighsCDouble lambda = maxviolation - continuouscontribution;

      cover.clear();
      cover.reserve(indices.size());

      for (HighsInt i = indices.size() - 1; i >= 0; --i) {
        double delta = upper[indices[i]] * reducedcost[indices[i]];

        if (upper[indices[i]] <= 1000.0 && reducedcost[indices[i]] > smallVal &&
            lambda - delta <= smallVal)
          cover.push_back(indices[i]);
        else
          lambda -= delta;
      }

      if (cover.empty() || lambda <= smallVal) break;

      HighsInt alpos = *std::min_element(
          cover.begin(), cover.end(), [&](HighsInt i1, HighsInt i2) {
            if (reducedcost[i1] <= 1e-3 || reducedcost[i2] <= 1e-3)
              return reducedcost[i1] > reducedcost[i2];
            return reducedcost[i1] < reducedcost[i2];
          });

      HighsInt coverend = cover.size();

      double al = reducedcost[alpos];
      coefs.resize(coverend);
      double coverrhs =
          std::max(std::ceil(double(lambda / al - primal_feastol)), 1.0);
      HighsCDouble slackupper = -coverrhs;

      double step = kHighsInf;
      for (HighsInt i = 0; i != coverend; ++i) {
        coefs[i] =
            std::ceil(std::min(reducedcost[cover[i]], double(lambda)) / al -
                      options->small_matrix_value);
        slackupper += upper[cover[i]] * coefs[i];
        step = std::min(step, reducedcost[cover[i]] / coefs[i]);
      }
      step = std::min(step, double(maxviolation / coverrhs));
      maxviolation -= step * coverrhs;

      HighsInt slackind = reducedcost.size();
      reducedcost.push_back(step);
      upper.push_back(double(slackupper));

      for (HighsInt i = 0; i != coverend; ++i)
        reducedcost[cover[i]] -= step * coefs[i];

      indices.erase(std::remove_if(indices.begin(), indices.end(),
                                   [&](HighsInt i) {
                                     return reducedcost[i] <= primal_feastol;
                                   }),
                    indices.end());
      indices.push_back(slackind);
    }

    double threshold = double(maxviolation + primal_feastol);

    indices.erase(std::remove_if(indices.begin(), indices.end(),
                                 [&](HighsInt i) {
                                   return i >= (HighsInt)positions.size() ||
                                          std::abs(reducedcost[i]) <= threshold;
                                 }),
                  indices.end());
    if (indices.empty()) continue;

    if (scale == -1.0) {
      HighsCDouble lhs = model->row_lower_[row];
      for (HighsInt i : indices) {
        double coefdelta = double(reducedcost[i] - maxviolation);
        HighsInt pos = positions[i];

        if (complementation[i] == -1) {
          lhs -= coefdelta * model->col_lower_[Acol[pos]];
          addToMatrix(row, Acol[pos], -coefdelta);
        } else {
          lhs += coefdelta * model->col_upper_[Acol[pos]];
          addToMatrix(row, Acol[pos], coefdelta);
        }
      }

      model->row_lower_[row] = double(lhs);
    } else {
      HighsCDouble rhs = model->row_upper_[row];
      for (HighsInt i : indices) {
        double coefdelta = double(reducedcost[i] - maxviolation);
        HighsInt pos = positions[i];

        if (complementation[i] == -1) {
          rhs += coefdelta * model->col_lower_[Acol[pos]];
          addToMatrix(row, Acol[pos], coefdelta);
        } else {
          rhs -= coefdelta * model->col_upper_[Acol[pos]];
          addToMatrix(row, Acol[pos], -coefdelta);
        }
      }

      model->row_upper_[row] = double(rhs);
    }

    numstrenghtened += indices.size();
  }

  return numstrenghtened;
}

HighsInt HPresolve::detectImpliedIntegers() {
  HighsInt numImplInt = 0;

  for (HighsInt col = 0; col != model->num_col_; ++col) {
    if (colDeleted[col]) continue;
    if (model->integrality_[col] == HighsVarType::kContinuous &&
        isImpliedInteger(col)) {
      ++numImplInt;
      model->integrality_[col] = HighsVarType::kImplicitInteger;

      for (const HighsSliceNonzero& nonzero : getColumnVector(col))
        ++rowsizeImplInt[nonzero.index()];

      double ceilLower = std::ceil(model->col_lower_[col] - primal_feastol);
      double floorUpper = std::floor(model->col_upper_[col] + primal_feastol);

      if (ceilLower > model->col_lower_[col]) changeColLower(col, ceilLower);
      if (floorUpper < model->col_upper_[col]) changeColUpper(col, floorUpper);
    }
  }

  return numImplInt;
}

HPresolve::Result HPresolve::detectParallelRowsAndCols(
    HighsPostsolveStack& postsolve_stack) {
  std::vector<std::uint64_t> rowHashes;
  std::vector<std::uint64_t> colHashes;
  std::vector<std::pair<double, HighsInt>> rowMax(rowsize.size());
  std::vector<std::pair<double, HighsInt>> colMax(colsize.size());

  HighsHashTable<HighsInt, HighsInt> numRowSingletons;

  HighsInt nnz = Avalue.size();
  rowHashes.assign(rowsize.begin(), rowsize.end());
  colHashes.assign(colsize.begin(), colsize.end());

  // Step 1: Determine scales for rows and columns and remove column singletons
  // from the intial row hashes which are initialized with the row sizes
  for (HighsInt i = 0; i != nnz; ++i) {
    if (Avalue[i] == 0.0) continue;
    assert(!colDeleted[Acol[i]]);
    if (colsize[Acol[i]] == 1) {
      colMax[Acol[i]].first = Avalue[i];
      --rowHashes[Arow[i]];
      numRowSingletons[Arow[i]] += 1;
      continue;
    }
    double absVal = std::abs(Avalue[i]);
    double absRowMax = std::abs(rowMax[Arow[i]].first);

    // among the largest values which are equal in tolerance
    // we use the nonzero with the smalles row/column index for the column/row
    // scale so that we ensure that duplicate rows/columns are scaled to have
    // the same sign
    if (absVal >= absRowMax - options->small_matrix_value) {
      // we are greater or equal with tolerances, check if we are either
      // strictly larger or equal with a smaller index and remember the signed
      // nonzero if one of those things is the case
      if (absVal > absRowMax + options->small_matrix_value ||
          Acol[i] < rowMax[Arow[i]].second) {
        rowMax[Arow[i]].first = Avalue[i];
        rowMax[Arow[i]].second = Acol[i];
      }
    }

    double absColMax = std::abs(colMax[Acol[i]].first);
    if (absVal >= absColMax - options->small_matrix_value) {
      if (absVal > absColMax + options->small_matrix_value ||
          Arow[i] < colMax[Acol[i]].second) {
        colMax[Acol[i]].first = Avalue[i];
        colMax[Acol[i]].second = Arow[i];
      }
    }
  }

  // Step 2: Compute hash values for rows and columns excluding singleton
  // columns
  for (HighsInt i = 0; i != nnz; ++i) {
    if (Avalue[i] == 0.0) continue;
    assert(!rowDeleted[Arow[i]] && !colDeleted[Acol[i]]);
    if (colsize[Acol[i]] == 1) {
      colHashes[Acol[i]] = Arow[i];
    } else {
      HighsHashHelpers::sparse_combine(rowHashes[Arow[i]], Acol[i],
                                       HighsHashHelpers::double_hash_code(
                                           Avalue[i] / rowMax[Arow[i]].first));
      HighsHashHelpers::sparse_combine(colHashes[Acol[i]], Arow[i],
                                       HighsHashHelpers::double_hash_code(
                                           Avalue[i] / colMax[Acol[i]].first));
    }
  }

  // Step 3: Loop over the rows and columns and put them into buckets using the
  // computed hash values. Whenever a bucket already contains a row/column,
  // check if we can apply a (nearly) parallel row reduction or a
  // parallel/dominated column reduction.
  HighsInt numRowBuckets = 0;
  HighsInt numColBuckets = 0;

  std::unordered_multimap<std::uint64_t, HighsInt> buckets;

  for (HighsInt i = 0; i != model->num_col_; ++i) {
    if (colDeleted[i]) continue;
    if (colsize[i] == 0) {
      HPRESOLVE_CHECKED_CALL(colPresolve(postsolve_stack, i));
      continue;
    }
    auto it = buckets.find(colHashes[i]);
    decltype(it) last = it;

    HighsInt delCol = -1;
    HighsInt parallelColCandidate = -2;

    if (it == buckets.end()) ++numColBuckets;
    while (it != buckets.end() && it->first == colHashes[i]) {
      parallelColCandidate = it->second;
      last = it++;

      // we want to check if the columns are parallel, first rule out
      // hash collisions with different size columns
      if (colsize[i] != colsize[parallelColCandidate]) continue;
      // The columns have the same length. Next we determine whether domination
      // is possible in one of the directions, and if it is we designate the
      // dominating column as column 2. The first thing we check is whether the
      // the objective value of one of the (scaled) columns is strictly better
      // then the objective value of the other column which rules out domination
      // in one direction.

      HighsInt col = -1;
      HighsInt duplicateCol = -1;
      double colScale;

      // helpers for checking dominance between parallel columns which is
      // possible for different cases of the variable types: if col can be
      // increased infinitely in which case duplicateCol can be fixed to its
      // lower bound. duplicateCol can be decreased infinitely in which case col
      // can be fixed to its upper bound. for both cases we exploit that the
      // column that remains unfixed can always compensate for the fixed column.
      // This only holds if the compensating column can compensate exactly for
      // feasible value of the fixed column. In the continuous case this
      // trivially holds. In the case where both variables are integer and the
      // scale is +- 1 this also holds trivially. If the scale is > 1 and both
      // variables are integer, this only holds in one direction. We can apply
      // the reduction due to the following reasoning: Applying the scale to
      // col, means we change its meaning and it is not an integer variable
      // anymore, but a variable that moves on multiples of 1/scale. As we have
      // taken care that the scale is >=1 and integral for two integer
      // variables, the scaled column can always exactly compensate for the
      // other column as it can move by 1/k with k being integer. Hence every
      // kth allowed value is integral and no integral value is skipped. If the
      // compensating column is integral
      bool checkColImplBounds = true;
      bool checkDuplicateColImplBounds = true;
      auto colUpperInf = [&]() {
        if (!checkColImplBounds) return false;
        if (mipsolver == nullptr) {
          // for LP we check strict reduncancy of the bounds as otherwise dual
          // postsolve might fail when the bound is used in the optimal solution
          return colScale > 0 ? model->col_upper_[col] == kHighsInf ||
                                    implColUpper[col] <
                                        model->col_upper_[col] - primal_feastol
                              : model->col_lower_[col] == -kHighsInf ||
                                    implColLower[col] >
                                        model->col_lower_[col] + primal_feastol;
        } else {
          // for MIP we do not need dual postsolve so the reduction is valid if
          // the bound is weakly redundant
          return colScale > 0 ? model->col_upper_[col] == kHighsInf ||
                                    implColUpper[col] <=
                                        model->col_upper_[col] + primal_feastol
                              : model->col_lower_[col] == -kHighsInf ||
                                    implColLower[col] >=
                                        model->col_lower_[col] - primal_feastol;
        }
      };

      auto colLowerInf = [&]() {
        if (!checkColImplBounds) return false;
        if (mipsolver == nullptr) {
          return colScale > 0 ? model->col_lower_[col] == -kHighsInf ||
                                    implColLower[col] >
                                        model->col_lower_[col] + primal_feastol
                              : model->col_upper_[col] == kHighsInf ||
                                    implColUpper[col] <
                                        model->col_upper_[col] - primal_feastol;
        } else {
          return colScale > 0 ? model->col_lower_[col] == -kHighsInf ||
                                    implColLower[col] >=
                                        model->col_lower_[col] - primal_feastol
                              : model->col_upper_[col] == kHighsInf ||
                                    implColUpper[col] <=
                                        model->col_upper_[col] + primal_feastol;
        }
      };

      auto duplicateColUpperInf = [&]() {
        if (!checkDuplicateColImplBounds) return false;
        if (mipsolver == nullptr) {
          return model->col_upper_[duplicateCol] == kHighsInf ||
                 implColUpper[duplicateCol] <
                     model->col_upper_[duplicateCol] - primal_feastol;
        } else {
          return model->col_upper_[duplicateCol] == kHighsInf ||
                 implColUpper[duplicateCol] <=
                     model->col_upper_[duplicateCol] + primal_feastol;
        }
      };

      auto duplicateColLowerInf = [&]() {
        if (!checkDuplicateColImplBounds) return false;
        if (mipsolver == nullptr) {
          return model->col_lower_[duplicateCol] == -kHighsInf ||
                 implColLower[duplicateCol] >
                     model->col_lower_[duplicateCol] + primal_feastol;
        } else {
          return model->col_lower_[duplicateCol] == -kHighsInf ||
                 implColLower[duplicateCol] >=
                     model->col_lower_[duplicateCol] - primal_feastol;
        }
      };

      // Now check the if the variable types rule out domination in one
      // direction and already skip the column if that rules out domination in
      // both directions due to the previous check on the objective.
      if (model->integrality_[i] == HighsVarType::kInteger &&
          model->integrality_[parallelColCandidate] == HighsVarType::kInteger) {
        // both variables are integral, hence the scale must be integral
        // therefore first choose the smaller colMax value for col2, then check
        // integrality of colMax[col1] / colMax[col2].
        if (std::abs(colMax[i].first) <
            std::abs(colMax[parallelColCandidate].first)) {
          col = i;
          duplicateCol = parallelColCandidate;
        } else {
          col = parallelColCandidate;
          duplicateCol = i;
        }

        double scaleCand = colMax[duplicateCol].first / colMax[col].first;
        colScale = std::round(scaleCand);
        assert(std::abs(colScale) >= 1.0);
        if (std::abs(colScale - scaleCand) > options->small_matrix_value)
          continue;

        // if the scale is larger than 1, duplicate column cannot compensate for
        // all values of scaled col due to integrality as the scaled column
        // moves on a grid of 1/scale.
        if (colScale != 1.0) checkDuplicateColImplBounds = false;
      } else if (model->integrality_[i] == HighsVarType::kInteger) {
        col = i;
        duplicateCol = parallelColCandidate;
        colScale = colMax[duplicateCol].first / colMax[col].first;

        // as col is integral and dulicateCol is not col cannot compensate for
        // duplicate col
        checkColImplBounds = false;
      } else {
        col = parallelColCandidate;
        duplicateCol = i;
        colScale = colMax[duplicateCol].first / colMax[col].first;

        // as col might be integral and dulicateCol is not integral. In that
        // case col cannot compensate for duplicate col
        checkColImplBounds =
            model->integrality_[parallelColCandidate] != HighsVarType::kInteger;
      }

      double objDiff = double(model->col_cost_[col] * HighsCDouble(colScale) -
                              model->col_cost_[duplicateCol]);
      // if (std::abs(objDiff) > options->small_matrix_value) continue;
      constexpr HighsInt kMergeParallelCols = 0;
      constexpr HighsInt kDominanceColToUpper = 1;
      constexpr HighsInt kDominanceColToLower = 2;
      constexpr HighsInt kDominanceDuplicateColToUpper = 3;
      constexpr HighsInt kDominanceDuplicateColToLower = 4;

      HighsInt reductionCase = kMergeParallelCols;
      // now do the case distinctions for dominated columns
      // the cases are a lot simpler due to the helper functions
      // for checking the infinite bounds which automatically
      // incorporate the check for the variable types that allow domination.
      if (objDiff < -options->dual_feasibility_tolerance) {
        // scaled col is better than duplicate col
        if (colUpperInf() && model->col_lower_[duplicateCol] != kHighsInf)
          reductionCase = kDominanceDuplicateColToLower;
        else if (duplicateColLowerInf() &&
                 (colScale < 0 || model->col_upper_[col] != kHighsInf) &&
                 (colScale > 0 || model->col_lower_[col] != -kHighsInf))
          reductionCase =
              colScale > 0 ? kDominanceColToUpper : kDominanceColToLower;
        else
          continue;
      } else if (objDiff > options->dual_feasibility_tolerance) {
        // duplicate col is better than scaled col
        if (colLowerInf() && model->col_upper_[duplicateCol] != kHighsInf)
          reductionCase = kDominanceDuplicateColToUpper;
        else if (duplicateColUpperInf() &&
                 (colScale < 0 || model->col_lower_[col] != -kHighsInf) &&
                 (colScale > 0 || model->col_upper_[col] != kHighsInf))
          reductionCase =
              colScale > 0 ? kDominanceColToLower : kDominanceColToUpper;
        else
          continue;
      } else {
        if (colUpperInf() && model->col_lower_[duplicateCol] != -kHighsInf)
          reductionCase = kDominanceDuplicateColToLower;
        else if (colLowerInf() && model->col_upper_[duplicateCol] != kHighsInf)
          reductionCase = kDominanceDuplicateColToUpper;
        else if (duplicateColUpperInf() &&
                 (colScale < 0 || model->col_lower_[col] != -kHighsInf) &&
                 (colScale > 0 || model->col_upper_[col] != kHighsInf))
          reductionCase =
              colScale > 0 ? kDominanceColToLower : kDominanceColToUpper;
        else if (duplicateColLowerInf() &&
                 (colScale < 0 || model->col_upper_[col] != kHighsInf) &&
                 (colScale > 0 || model->col_lower_[col] != -kHighsInf))
          reductionCase =
              colScale > 0 ? kDominanceColToUpper : kDominanceColToLower;
      }
      double mergeLower = 0;
      double mergeUpper = 0;
      if (reductionCase == kMergeParallelCols) {
        if (colScale > 0) {
          mergeLower = model->col_lower_[col] +
                       colScale * model->col_lower_[duplicateCol];
          mergeUpper = model->col_upper_[col] +
                       colScale * model->col_upper_[duplicateCol];
        } else {
          mergeLower = model->col_lower_[col] +
                       colScale * model->col_upper_[duplicateCol];
          mergeUpper = model->col_upper_[col] +
                       colScale * model->col_lower_[duplicateCol];
        }
        if (model->integrality_[col] == HighsVarType::kInteger) {
          // the only possible reduction if the column parallelism check
          // succeeds is to merge the two columns into one. If one column is
          // integral this means we have restrictions on integers and need to
          // check additional conditions to allow the merging of two integer
          // columns, or a continuous column and an integer.
          if (model->integrality_[duplicateCol] != HighsVarType::kInteger) {
            // only one column is integral which cannot be duplicateCol due to
            // the way we assign the columns above
            if (std::abs(colScale * (model->col_upper_[duplicateCol] -
                                     model->col_lower_[duplicateCol])) <
                1.0 - primal_feastol)
              continue;
          } else if (colScale > 1.0) {
            // round bounds to exact integer values to make sure they are not
            // wrongly truncated in conversions happening below
            mergeLower = std::round(mergeLower);
            mergeUpper = std::round(mergeUpper);

            // this should not happen, since this would allow domination and
            // would have been caught by the cases above
            assert(mergeLower != -kHighsInf);
            assert(mergeUpper != kHighsInf);

            HighsInt kMax = mergeUpper;
            bool representable = true;
            for (HighsInt k = mergeLower; k <= kMax; ++k) {
              // we loop over the domain of the merged variable to check whether
              // there exists a value for col and duplicateCol so that both are
              // within their bounds. since the merged column y is defined as y
              // = col + colScale * duplicateCol, we know that the value of col
              // can be computed as col = y - colScale * duplicateCol. Hence we
              // loop over the domain of col2 until we verify that a suitable
              // value of column 1 exists to yield the desired value for y.
              double mergeVal = mergeLower + k;
              HighsInt k2Max = model->col_upper_[duplicateCol];
              assert(k2Max == model->col_upper_[duplicateCol]);
              representable = false;
              for (HighsInt k2 = model->col_lower_[duplicateCol]; k2 <= k2Max;
                   ++k2) {
                double colVal = mergeVal - colScale * k2;
                if (colVal >= model->col_lower_[col] - primal_feastol &&
                    colVal <= model->col_upper_[col] + primal_feastol) {
                  representable = true;
                  break;
                }
              }

              if (!representable) break;
            }

            if (!representable) continue;
          }
        }
      }

      bool parallel = true;
      // now check whether the coefficients are actually parallel
      for (const HighsSliceNonzero& colNz : getColumnVector(col)) {
        HighsInt duplicateColRowPos = findNonzero(colNz.index(), duplicateCol);
        if (duplicateColRowPos == -1) {
          parallel = false;
          break;
        }

        double difference = std::abs(
            double(Avalue[duplicateColRowPos] - colScale * colNz.value()));
        if (difference > options->small_matrix_value) {
          parallel = false;
          break;
        }
      }

      if (!parallel) continue;

      switch (reductionCase) {
        case kDominanceDuplicateColToLower:
          delCol = duplicateCol;
          if (colsize[duplicateCol] == 1) {
            HighsInt row = Arow[colhead[duplicateCol]];
            numRowSingletons[row] -= 1;
          }
          fixColToLower(postsolve_stack, duplicateCol);
          break;
        case kDominanceDuplicateColToUpper:
          delCol = duplicateCol;
          if (colsize[duplicateCol] == 1) {
            HighsInt row = Arow[colhead[duplicateCol]];
            numRowSingletons[row] -= 1;
          }
          fixColToUpper(postsolve_stack, duplicateCol);
          break;
        case kDominanceColToLower:
          delCol = col;
          if (colsize[col] == 1) {
            HighsInt row = Arow[colhead[col]];
            numRowSingletons[row] -= 1;
          }
          fixColToLower(postsolve_stack, col);
          break;
        case kDominanceColToUpper:
          delCol = col;
          if (colsize[col] == 1) {
            HighsInt row = Arow[colhead[col]];
            numRowSingletons[row] -= 1;
          }
          fixColToUpper(postsolve_stack, col);
          break;
        case kMergeParallelCols:
          postsolve_stack.duplicateColumn(
              colScale, model->col_lower_[col], model->col_upper_[col],
              model->col_lower_[duplicateCol], model->col_upper_[duplicateCol],
              col, duplicateCol,
              model->integrality_[col] == HighsVarType::kInteger,
              model->integrality_[duplicateCol] == HighsVarType::kInteger);
          HighsInt rowsizeIntReduction = 0;
          if (model->integrality_[duplicateCol] != HighsVarType::kInteger &&
              model->integrality_[col] == HighsVarType::kInteger) {
            rowsizeIntReduction = 1;
            model->integrality_[col] = HighsVarType::kContinuous;
          }
          markChangedCol(col);
          if (colsize[duplicateCol] == 1) {
            HighsInt row = Arow[colhead[duplicateCol]];
            numRowSingletons[row] -= 1;
          }

          // by updating the bounds properly, the unlink calls will update the
          // implied row upper bounds to the correct values. For finite bounds
          // simply setting the bounds of duplicate col to zero suffices. For
          // infinite bounds we need to make sure the counters for the number of
          // infinite bounds that contribute to the implied row bounds are
          // updated correctly and that all finite contributions are removed.
          if (colScale > 0) {
            if (mergeUpper == kHighsInf && model->col_upper_[col] != kHighsInf)
              model->col_upper_[duplicateCol] =
                  model->col_upper_[col] / colScale;
            else
              model->col_upper_[duplicateCol] = 0;

            if (mergeLower == -kHighsInf &&
                model->col_lower_[col] != -kHighsInf)
              // make sure that upon removal of the duplicate column the finite
              // contribution of col's lower bound is removed and the infinite
              // contribution of duplicateCol is retained
              model->col_lower_[duplicateCol] =
                  model->col_lower_[col] / colScale;
            else
              model->col_lower_[duplicateCol] = 0;
          } else {
            if (mergeUpper == kHighsInf && model->col_upper_[col] != kHighsInf)
              model->col_lower_[duplicateCol] =
                  model->col_upper_[col] / colScale;
            else
              model->col_lower_[duplicateCol] = 0;

            if (mergeLower == -kHighsInf &&
                model->col_lower_[col] != -kHighsInf)
              // make sure that upon removal of the duplicate column the finite
              // contribution of col's lower bound is removed and the infinite
              // contribution of duplicateCol is retained
              model->col_upper_[duplicateCol] =
                  model->col_lower_[col] / colScale;
            else
              model->col_upper_[duplicateCol] = 0;
          }

          model->col_lower_[col] = mergeLower;
          model->col_upper_[col] = mergeUpper;

          // mark duplicate column as deleted
          markColDeleted(duplicateCol);
          // remove all nonzeros of duplicateCol
          for (HighsInt coliter = colhead[duplicateCol]; coliter != -1;) {
            assert(Acol[coliter] == duplicateCol);

            HighsInt colpos = coliter;
            HighsInt colrow = Arow[coliter];
            // if an an integer column was merged into a continuous one make
            // sure to update the integral rowsize
            if (rowsizeIntReduction)
              rowsizeInteger[colrow] -= rowsizeIntReduction;
            coliter = Anext[coliter];

            unlink(colpos);

            if (model->row_lower_[colrow] == model->row_upper_[colrow] &&
                eqiters[colrow] != equations.end() &&
                eqiters[colrow]->first != rowsize[colrow]) {
              // if that is the case reinsert it into the equation set that is
              // ordered by sparsity
              equations.erase(eqiters[colrow]);
              eqiters[colrow] =
                  equations.emplace(rowsize[colrow], colrow).first;
            }
          }
          // set cost to zero
          model->col_cost_[duplicateCol] = 0;
          delCol = duplicateCol;

          // remove implied bounds, since they might in general not be valid
          // anymore
          if (colLowerSource[col] != -1)
            changeImplColLower(col, -kHighsInf, -1);

          if (colUpperSource[col] != -1) changeImplColUpper(col, kHighsInf, -1);

          // if an implicit integer and an integer column where merge, check if
          // merged continuous column is implicit integer after merge
          if (rowsizeIntReduction &&
              model->integrality_[duplicateCol] ==
                  HighsVarType::kImplicitInteger &&
              isImpliedInteger(col)) {
            model->integrality_[col] = HighsVarType::kImplicitInteger;
            for (const HighsSliceNonzero& nonz : getColumnVector(col))
              ++rowsizeImplInt[nonz.index()];
          }

          break;
      }

      break;
    }

    if (delCol != -1) {
      if (delCol != i) buckets.erase(last);

      // we could have new row singletons since a column was removed. Remove
      // those rows immediately
      HPRESOLVE_CHECKED_CALL(checkLimits(postsolve_stack));
      HPRESOLVE_CHECKED_CALL(removeRowSingletons(postsolve_stack));
    } else {
      buckets.emplace_hint(last, colHashes[i], i);
    }
  }

  buckets.clear();

  for (HighsInt i = 0; i != model->num_row_; ++i) {
    if (rowDeleted[i]) continue;
    if (rowsize[i] <= 1 ||
        (rowsize[i] == 2 && model->row_lower_[i] == model->row_upper_[i])) {
      HPRESOLVE_CHECKED_CALL(rowPresolve(postsolve_stack, i));
      ++numRowBuckets;
      continue;
    }
    auto it = buckets.find(rowHashes[i]);
    decltype(it) last = it;

    const HighsInt* numSingletonPtr = numRowSingletons.find(i);
    HighsInt numSingleton = numSingletonPtr ? *numSingletonPtr : 0;

#if !ENABLE_SPARSIFY_FOR_LP
    if (mipsolver == nullptr && options->lp_presolve_requires_basis_postsolve &&
        numSingleton != 0)
      continue;
#endif
    HighsInt delRow = -1;
    if (it == buckets.end())
      ++numRowBuckets;
    else
      storeRow(i);
    while (it != buckets.end() && it->first == rowHashes[i]) {
      HighsInt parallelRowCand = it->second;
      last = it++;

      numSingletonPtr = numRowSingletons.find(parallelRowCand);
      const HighsInt numSingletonCandidate =
          numSingletonPtr ? *numSingletonPtr : 0;
#if !ENABLE_SPARSIFY_FOR_LP
      if (mipsolver == nullptr &&
          options->lp_presolve_requires_basis_postsolve &&
          numSingletonCandidate != 0)
        continue;
#endif
      if (rowsize[i] - numSingleton !=
          rowsize[parallelRowCand] - numSingletonCandidate)
        continue;

      if (numSingletonCandidate > 1 || numSingleton > 1) {
        // we only handle the case where the rows have at most one extra
        // singleton except when one row has no extra singleton and is an
        // equation. In that case we sparsify the other row by adding the
        // equation and can subsequently solve it as an individual component as
        // it is a row which only contains singletons
        if ((numSingleton != 0 ||
             model->row_lower_[i] != model->row_upper_[i]) &&
            (numSingletonCandidate != 0 ||
             model->row_lower_[parallelRowCand] !=
                 model->row_upper_[parallelRowCand]))
          continue;
      } else if (numSingletonCandidate != numSingleton) {
        // if only one of the two constraints has an extra singleton,
        // we require at least one of the constraints to be an equation
        // if that is the case we can add that equation to the other row
        // and will make it into either a row singleton or a doubleton equation
        // which is removed afterwards
        if (model->row_lower_[i] != model->row_upper_[i] &&
            model->row_lower_[parallelRowCand] !=
                model->row_upper_[parallelRowCand])
          continue;
      }

      double rowScale = rowMax[parallelRowCand].first / rowMax[i].first;
      // check parallel case
      bool parallel = true;
      for (const HighsSliceNonzero& rowNz : getStoredRow()) {
        if (colsize[rowNz.index()] == 1)  // skip singletons
          continue;
        HighsInt nzPos = findNonzero(parallelRowCand, rowNz.index());
        if (nzPos == -1) {
          parallel = false;
          break;
        }

        if (std::abs(double(Avalue[nzPos] -
                            HighsCDouble(rowScale) * rowNz.value())) >
            options->small_matrix_value) {
          parallel = false;
          break;
        }
      }
      if (!parallel) continue;

      if (numSingleton == 0 && numSingletonCandidate == 0) {
        bool rowLowerTightened = false;
        bool rowUpperTightened = false;
        double newUpper;
        double newLower;
        if (rowScale > 0) {
          newUpper = model->row_upper_[i] * rowScale;
          newLower = model->row_lower_[i] * rowScale;
        } else {
          newLower = model->row_upper_[i] * rowScale;
          newUpper = model->row_lower_[i] * rowScale;
        }

        if (newUpper < model->row_upper_[parallelRowCand]) {
          if (newUpper < model->row_lower_[parallelRowCand] - primal_feastol)
            return Result::kPrimalInfeasible;

          if (newUpper <= model->row_lower_[parallelRowCand] + primal_feastol)
            newUpper = model->row_lower_[parallelRowCand];

          if (newUpper < model->row_upper_[parallelRowCand]) {
            rowUpperTightened = true;
            if (rowScale > 0) {
              double tmp = rowDualLower[i] / rowScale;
              rowDualLower[i] = rowDualLower[parallelRowCand] * rowScale;
              rowDualLower[parallelRowCand] = tmp;
            } else {
              double tmp = rowDualUpper[i] / rowScale;
              rowDualUpper[i] = rowDualLower[parallelRowCand] * rowScale;
              rowDualLower[parallelRowCand] = tmp;
            }

            model->row_upper_[parallelRowCand] = newUpper;
          }
        }

        if (newLower > model->row_lower_[parallelRowCand]) {
          if (newLower > model->row_upper_[parallelRowCand] + primal_feastol)
            return Result::kPrimalInfeasible;

          if (newLower >= model->row_upper_[parallelRowCand] - primal_feastol)
            newLower = model->row_upper_[parallelRowCand];

          if (newLower > model->row_lower_[parallelRowCand]) {
            // the rows lower bound is tightened
            // instead of updating the activities of dual constraints, we
            // can simply swap the bounds on the row duals. If the old
            // lower bound on the row dual was finite, the new row dual
            // lower bound is infinite as the new row lower bound must be
            // a finite value. This infinite contribution, was, however,
            // already counted from the parallel row. Therefore by
            // swapping the bounds unlinking the other row will not
            // decrease the infinity counter, but simply remove a bound
            // with zero contribution. For a negative scale we need to
            // swap with the negated upper bound of the row dual of row i.
            rowLowerTightened = true;
            if (rowScale > 0) {
              double tmp = rowDualUpper[i] / rowScale;
              rowDualUpper[i] = rowDualUpper[parallelRowCand] * rowScale;
              rowDualUpper[parallelRowCand] = tmp;
            } else {
              double tmp = rowDualLower[i] / rowScale;
              rowDualLower[i] = rowDualUpper[parallelRowCand] * rowScale;
              rowDualUpper[parallelRowCand] = tmp;
            }

            model->row_lower_[parallelRowCand] = newLower;
          }
        }
        if (rowDualLowerSource[parallelRowCand] != -1)
          changeImplRowDualLower(parallelRowCand, -kHighsInf, -1);
        if (rowDualUpperSource[parallelRowCand] != -1)
          changeImplRowDualUpper(parallelRowCand, kHighsInf, -1);

        postsolve_stack.duplicateRow(parallelRowCand, rowUpperTightened,
                                     rowLowerTightened, i, rowScale);
        delRow = i;
        markRowDeleted(i);
        for (HighsInt rowiter : rowpositions) unlink(rowiter);
        break;
      } else if (model->row_lower_[i] == model->row_upper_[i]) {
        // row i is equation and parallel (except for singletons)
        // add to the row parallelRowCand
        // printf(
        //    "nearly parallel case with %" HIGHSINT_FORMAT " singletons in eq
        //    row and %" HIGHSINT_FORMAT " " "singletons in other row(eq=%"
        //    HIGHSINT_FORMAT ")\n", numSingleton, numSingletonCandidate,
        //    model->row_lower_[parallelRowCand] ==
        //        model->row_upper_[parallelRowCand]);
        postsolve_stack.equalityRowAddition(parallelRowCand, i, -rowScale,
                                            getStoredRow());
        for (const HighsSliceNonzero& rowNz : getStoredRow()) {
          HighsInt pos = findNonzero(parallelRowCand, rowNz.index());
          if (pos != -1)
            unlink(pos);  // all common nonzeros are cancelled, as the rows are
                          // parallel
          else            // might introduce a singleton
            addToMatrix(parallelRowCand, rowNz.index(),
                        -rowScale * rowNz.value());
        }

        if (model->row_upper_[parallelRowCand] != kHighsInf)
          model->row_upper_[parallelRowCand] =
              double(model->row_upper_[parallelRowCand] -
                     HighsCDouble(rowScale) * model->row_upper_[i]);
        if (model->row_lower_[parallelRowCand] != -kHighsInf)
          model->row_lower_[parallelRowCand] =
              double(model->row_lower_[parallelRowCand] -
                     HighsCDouble(rowScale) * model->row_upper_[i]);

        // parallelRowCand is now a singleton row, doubleton equation, or a row
        // that contains only singletons and we let the normal row presolve
        // handle the cases
        HPRESOLVE_CHECKED_CALL(rowPresolve(postsolve_stack, parallelRowCand));
        delRow = parallelRowCand;
      } else if (model->row_lower_[parallelRowCand] ==
                 model->row_upper_[parallelRowCand]) {
        // printf(
        //    "nearly parallel case with %" HIGHSINT_FORMAT " singletons in eq
        //    row and %" HIGHSINT_FORMAT " " "singletons in other inequality
        //    row\n", numSingletonCandidate, numSingleton);
        // the row parallelRowCand is an equation; add it to the other row
        double scale = -rowMax[i].first / rowMax[parallelRowCand].first;
        postsolve_stack.equalityRowAddition(i, parallelRowCand, scale,
                                            getRowVector(parallelRowCand));
        for (const HighsSliceNonzero& rowNz : getRowVector(parallelRowCand)) {
          HighsInt pos = findNonzero(i, rowNz.index());
          if (pos != -1)
            unlink(pos);  // all common nonzeros are cancelled, as the rows are
                          // parallel
          else            // might introduce a singleton
            addToMatrix(i, rowNz.index(), scale * rowNz.value());
        }

        if (model->row_upper_[i] != kHighsInf)
          model->row_upper_[i] =
              double(model->row_upper_[i] +
                     HighsCDouble(scale) * model->row_upper_[parallelRowCand]);
        if (model->row_lower_[i] != -kHighsInf)
          model->row_lower_[i] =
              double(model->row_lower_[i] +
                     HighsCDouble(scale) * model->row_upper_[parallelRowCand]);

        HPRESOLVE_CHECKED_CALL(rowPresolve(postsolve_stack, i));
        delRow = i;
      } else {
        assert(numSingleton == 1);
        assert(numSingletonCandidate == 1);

        double rowUpper;
        double rowLower;
        if (rowScale > 0) {
          rowUpper = model->row_upper_[i] * rowScale;
          rowLower = model->row_lower_[i] * rowScale;
        } else {
          rowLower = model->row_upper_[i] * rowScale;
          rowUpper = model->row_lower_[i] * rowScale;
        }
        // todo: two inequalities with one singleton. check whether the rows can
        // be converted to equations by introducing a shared slack variable
        // which is the case if the singletons have similar properties
        // (objective sign, bounds, scaled coefficient) and the scaled right
        // hand sides match. Then the case reduces to adding one equation to the
        // other and substituting one of the singletons due to the resulting
        // doubleton equation.
        //        printf("todo, two inequalities with one additional
        //        singleton\n");
        (void)rowLower;
        (void)rowUpper;
      }
    }

    if (delRow != -1) {
      if (delRow != i) buckets.erase(last);

      HPRESOLVE_CHECKED_CALL(checkLimits(postsolve_stack));
    } else
      buckets.emplace_hint(last, rowHashes[i], i);
  }

  return Result::kOk;
}

void HPresolve::setRelaxedImpliedBounds() {
  double hugeBound = primal_feastol / kHighsTiny;
  for (HighsInt i = 0; i != model->num_col_; ++i) {
    if (model->col_lower_[i] >= implColLower[i] &&
        model->col_upper_[i] <= implColUpper[i])
      continue;

    if (std::abs(implColLower[i]) <= hugeBound) {
      // if the bound is derived from a small nonzero value
      // then we want to increase the margin so that we make sure
      // the row it was derived from is violated if the column sits
      // at this relaxed bound in the final solution.
      HighsInt nzPos = findNonzero(colLowerSource[i], i);

      double boundRelax = std::max(1000.0, std::abs(implColLower[i])) *
                          primal_feastol /
                          std::min(1.0, std::abs(Avalue[nzPos]));

      double newLb = implColLower[i] - boundRelax;
      if (newLb > model->col_lower_[i] + boundRelax)
        model->col_lower_[i] = newLb;
    }

    if (std::abs(implColUpper[i]) <= hugeBound) {
      HighsInt nzPos = findNonzero(colUpperSource[i], i);

      double boundRelax = std::max(1000.0, std::abs(implColUpper[i])) *
                          primal_feastol /
                          std::min(1.0, std::abs(Avalue[nzPos]));

      double newUb = implColUpper[i] + boundRelax;
      if (newUb < model->col_upper_[i] - boundRelax)
        model->col_upper_[i] = newUb;
    }
  }
}

void HPresolve::debug(const HighsLp& lp, const HighsOptions& options) {
  HighsSolution reducedsol;
  HighsBasis reducedbasis;

  HighsSolution sol;
  HighsBasis basis;

  HighsLp model = lp;
  model.integrality_.assign(lp.num_col_, HighsVarType::kContinuous);

  HighsPostsolveStack postsolve_stack;
  postsolve_stack.initializeIndexMaps(lp.num_row_, lp.num_col_);
  {
    HPresolve presolve;
    presolve.setInput(model, options);
    // presolve.setReductionLimit(1622017);
    if (presolve.run(postsolve_stack) != HighsModelStatus::kNotset) return;
    Highs highs;
    highs.passModel(model);
    highs.passOptions(options);
    highs.setOptionValue("presolve", "off");
    highs.run();
    if (highs.getModelStatus() != HighsModelStatus::kOptimal) return;
    reducedsol = highs.getSolution();
    reducedbasis = highs.getBasis();
  }
  model = lp;
  sol = reducedsol;
  basis = reducedbasis;
  postsolve_stack.undo(options, sol, basis);
  refineBasis(lp, sol, basis);
  calculateRowValues(model, sol);
#if 0
  Highs highs;
  highs.passModel(model);
  highs.passOptions(options);
  highs.setSolution(sol);
  basis.debug_origin_name = "HPresolve::debug";
  highs.setBasis(basis);
  highs.run();
  return;
#endif
  std::vector<HighsInt> flagCol(lp.num_col_, 1);
  std::vector<HighsInt> flagRow(lp.num_row_, 1);
  std::vector<HighsInt> Aend;
  std::vector<HighsInt> ARstart;
  std::vector<HighsInt> ARindex;
  std::vector<double> ARvalue;
  dev_kkt_check::KktInfo kktinfo = dev_kkt_check::initInfo();
  Aend.assign(model.a_matrix_.start_.begin() + 1, model.a_matrix_.start_.end());
  highsSparseTranspose(model.num_row_, model.num_col_, model.a_matrix_.start_,
                       model.a_matrix_.index_, model.a_matrix_.value_, ARstart,
                       ARindex, ARvalue);
  dev_kkt_check::State state(
      model.num_col_, model.num_row_, model.a_matrix_.start_, Aend,
      model.a_matrix_.index_, model.a_matrix_.value_, ARstart, ARindex, ARvalue,
      model.col_cost_, model.col_lower_, model.col_upper_, model.row_lower_,
      model.row_upper_, flagCol, flagRow, sol.col_value, sol.col_dual,
      sol.row_value, sol.row_dual, basis.col_status, basis.row_status);
  bool checkResult = dev_kkt_check::checkKkt(state, kktinfo);
  if (checkResult && kktinfo.pass_bfs) {
    printf("kkt check of postsolved solution and basis passed\n");
    return;
  }
  size_t good = postsolve_stack.numReductions();
  size_t bad = 0;
  size_t reductionLim = (good + bad) / 2;

  // good = 1734357, bad = 1734289;
  // good = 1050606, bad = 1050605;
  // good = 1811527, bad = 1811526;
  // reductionLim = bad;
  do {
    model = lp;
    model.integrality_.assign(lp.num_col_, HighsVarType::kContinuous);

    {
      HPresolve presolve;
      presolve.setInput(model, options);
      presolve.computeIntermediateMatrix(flagRow, flagCol, reductionLim);
    }
#if 1
    model = lp;
    model.integrality_.assign(lp.num_col_, HighsVarType::kContinuous);
    HPresolve presolve;
    presolve.setInput(model, options);
    HighsPostsolveStack tmp;
    tmp.initializeIndexMaps(model.num_row_, model.num_col_);
    presolve.setReductionLimit(reductionLim);
    presolve.run(tmp);

    sol = reducedsol;
    basis = reducedbasis;
    postsolve_stack.undoUntil(options, flagRow, flagCol, sol, basis,
                              tmp.numReductions());

    HighsBasis temp_basis;
    HighsSolution temp_sol;
    temp_basis.col_status.resize(model.num_col_);
    temp_sol.col_dual.resize(model.num_col_);
    temp_sol.col_value.resize(model.num_col_);
    for (HighsInt i = 0; i != model.num_col_; ++i) {
      temp_sol.col_dual[i] = sol.col_dual[tmp.getOrigColIndex(i)];
      temp_sol.col_value[i] = sol.col_value[tmp.getOrigColIndex(i)];
      temp_basis.col_status[i] = basis.col_status[tmp.getOrigColIndex(i)];
    }

    temp_basis.row_status.resize(model.num_row_);
    temp_sol.row_dual.resize(model.num_row_);
    for (HighsInt i = 0; i != model.num_row_; ++i) {
      temp_sol.row_dual[i] = sol.row_dual[tmp.getOrigRowIndex(i)];
      temp_basis.row_status[i] = basis.row_status[tmp.getOrigRowIndex(i)];
    }
    temp_sol.row_value.resize(model.num_row_);
    calculateRowValues(model, sol);
    temp_basis.valid = true;
    refineBasis(model, temp_sol, temp_basis);
    Highs highs;
    highs.passOptions(options);
    highs.passModel(model);
    temp_basis.debug_origin_name = "HPresolve::debug";
    highs.setBasis(temp_basis);
    // highs.writeModel("model.mps");
    // highs.writeBasis("bad.bas");
    highs.run();
    printf("simplex iterations with postsolved basis: %" HIGHSINT_FORMAT "\n",
           highs.getInfo().simplex_iteration_count);
    checkResult = highs.getInfo().simplex_iteration_count == 0;
#else

    if (reductionLim == good) break;

    Aend.assign(model.a_matrix_.start_.begin() + 1,
                model.a_matrix_.start_.end());
    highsSparseTranspose(model.num_row_, model.num_col_, model.a_matrix_.start_,
                         model.a_matrix_.index_, model.a_matrix_.value_,
                         ARstart, ARindex, ARvalue);
    sol = reducedsol;
    basis = reducedbasis;
    postsolve_stack.undoUntil(options, flagRow, flagCol, sol, basis,
                              reductionLim);

    calculateRowValues(model, sol);
    kktinfo = dev_kkt_check::initInfo();
    checkResult = dev_kkt_check::checkKkt(state, kktinfo);
    checkResult = checkResult && kktinfo.pass_bfs;
#endif
    if (bad == good - 1) break;

    if (checkResult) {
      good = reductionLim;
    } else {
      bad = reductionLim;
    }
    reductionLim = (bad + good) / 2;
    printf("binary search ongoing: good=%zu, bad=%zu\n", good, bad);
  } while (true);

  printf("binary search finished: good=%zu, bad=%zu\n", good, bad);
  assert(false);
}

HPresolve::Result HPresolve::sparsify(HighsPostsolveStack& postsolve_stack) {
  std::vector<HighsPostsolveStack::Nonzero> sparsifyRows;
  HPRESOLVE_CHECKED_CALL(removeRowSingletons(postsolve_stack));
  HPRESOLVE_CHECKED_CALL(removeDoubletonEquations(postsolve_stack));
  std::vector<HighsInt> tmpEquations;
  tmpEquations.reserve(equations.size());

  const double minNonzeroVal = std::sqrt(primal_feastol);

  for (const auto& eq : equations) tmpEquations.emplace_back(eq.second);
  for (HighsInt eqrow : tmpEquations) {
    if (rowDeleted[eqrow]) continue;

    assert(!rowDeleted[eqrow]);
    assert(model->row_lower_[eqrow] == model->row_upper_[eqrow]);

    storeRow(eqrow);

    HighsInt secondSparsestColumn = -1;
    HighsInt sparsestCol = Acol[rowpositions[0]];
    HighsInt sparsestColLen = kHighsIInf;
    for (size_t i = 1; i < rowpositions.size(); ++i) {
      HighsInt col = Acol[rowpositions[i]];
      if (colsize[col] < sparsestColLen) {
        sparsestColLen = colsize[col];
        secondSparsestColumn = sparsestCol;
        sparsestCol = col;
      }
    }

    if (colsize[secondSparsestColumn] < colsize[sparsestCol])
      std::swap(sparsestCol, secondSparsestColumn);

    assert(sparsestCol != -1 && secondSparsestColumn != -1);

    std::map<double, HighsInt> possibleScales;
    sparsifyRows.clear();

    for (const HighsSliceNonzero& colNz : getColumnVector(sparsestCol)) {
      HighsInt candRow = colNz.index();
      if (candRow == eqrow) continue;

      possibleScales.clear();

      HighsInt misses = 0;
      // allow no fillin if a completely continuous row is used to cancel a row
      // that has integers as there are instances where this leads to a huge
      // deterioration of cut performance
      HighsInt maxMisses = 1;
      if (rowsizeInteger[eqrow] == 0 && rowsizeInteger[candRow] != 0)
        --maxMisses;
      for (const HighsSliceNonzero& nonzero : getStoredRow()) {
        double candRowVal;
        if (nonzero.index() == sparsestCol) {
          candRowVal = colNz.value();
        } else {
          HighsInt nzPos = findNonzero(candRow, nonzero.index());
          if (nzPos == -1) {
            if (model->integrality_[nonzero.index()] ==
                    HighsVarType::kInteger &&
                model->col_upper_[nonzero.index()] -
                        model->col_lower_[nonzero.index()] >
                    1.5) {
              // do not allow fillin of general integers
              misses = 2;
              break;
            }
            ++misses;
            if (misses > maxMisses) break;
            continue;
          }
          candRowVal = Avalue[nzPos];
        }

        double scale = -candRowVal / nonzero.value();
        if (std::abs(scale) > 1e3) continue;

        double scaleTolerance = minNonzeroVal / std::abs(nonzero.value());
        auto it = possibleScales.lower_bound(scale - scaleTolerance);
        if (it != possibleScales.end() &&
            std::abs(it->first - scale) <= scaleTolerance) {
          // there already is a scale that is very close and could produces
          // a matrix value for this nonzero that is below the allowed
          // threshold. Therefore we check if the matrix value is small enough
          // for this nonzero to be deleted, in which case the number of
          // deleted nonzeros for the other scale is increased. If it is not
          // small enough we do not use this scale or the other one because
          // such small matrix values may lead to numerical troubles.

          // scale is already marked to be numerically bad
          if (it->second == -1) continue;

          if (std::abs(it->first * nonzero.value() + candRowVal) <=
              options->small_matrix_value)
            it->second += 1;
          else
            it->second = -1;
        } else
          possibleScales.emplace(scale, 1);
      }

      if (misses > maxMisses || possibleScales.empty()) continue;

      HighsInt numCancel = 0;
      double scale = 0.0;

      for (const auto& s : possibleScales) {
        if (s.second <= misses) continue;

        if (s.second > numCancel ||
            (s.second == numCancel && std::abs(s.first) < std::abs(scale))) {
          scale = s.first;
          numCancel = s.second;
        }
      }

      assert(scale != 0.0 || numCancel == 0);

      // cancels at least one nonzero if the scale cancels more than there is
      // fillin
      if (numCancel > misses) sparsifyRows.emplace_back(candRow, scale);
    }

    if (model->integrality_[sparsestCol] != HighsVarType::kInteger ||
        (model->col_upper_[sparsestCol] - model->col_lower_[sparsestCol]) <
            1.5) {
      // now check for rows which do not contain the sparsest column but all
      // other columns by scanning the second sparsest column
      for (const HighsSliceNonzero& colNz :
           getColumnVector(secondSparsestColumn)) {
        HighsInt candRow = colNz.index();
        if (candRow == eqrow) continue;

        if (rowsizeInteger[eqrow] == 0 && rowsizeInteger[candRow] != 0)
          continue;

        HighsInt sparsestColPos = findNonzero(candRow, sparsestCol);

        // if the row has a nonzero for the sparsest column we have already
        // checked it
        if (sparsestColPos != -1) continue;

        possibleScales.clear();
        bool skip = false;
        for (const HighsSliceNonzero& nonzero : getStoredRow()) {
          double candRowVal;
          if (nonzero.index() == secondSparsestColumn) {
            candRowVal = colNz.value();
          } else {
            HighsInt nzPos = findNonzero(candRow, nonzero.index());
            if (nzPos == -1) {
              // we already have a miss for the sparsest column, so with another
              // one we want to skip the row
              skip = true;
              break;
            }

            candRowVal = Avalue[nzPos];
          }

          double scale = -candRowVal / nonzero.value();
          if (std::abs(scale) > 1e3) continue;

          double scaleTolerance = minNonzeroVal / std::abs(nonzero.value());
          auto it = possibleScales.lower_bound(scale - scaleTolerance);
          if (it != possibleScales.end() &&
              std::abs(it->first - scale) <= scaleTolerance) {
            // there already is a scale that is very close and could produces
            // a matrix value for this nonzero that is below the allowed
            // threshold. Therefore we check if the matrix value is small enough
            // for this nonzero to be deleted, in which case the number of
            // deleted nonzeros for the other scale is increased. If it is not
            // small enough we do not use this scale or the other one because
            // such small matrix values may lead to numerical troubles.

            // scale is already marked to be numerically bad
            if (it->second == -1) continue;

            if (std::abs(it->first * nonzero.value() + candRowVal) <=
                options->small_matrix_value) {
              it->second += 1;
            } else {
              // mark scale to be numerically bad
              it->second = -1;
              continue;
            }
          } else
            possibleScales.emplace(scale, 1);
        }

        if (skip || possibleScales.empty()) continue;

        HighsInt numCancel = 0;
        double scale = 0.0;

        for (const auto& s : possibleScales) {
          if (s.second <= 1) continue;
          if (s.second > numCancel ||
              (s.second == numCancel && std::abs(s.first) < std::abs(scale))) {
            scale = s.first;
            numCancel = s.second;
          }
        }

        assert(scale != 0.0 || numCancel == 0);

        // cancels at least one nonzero if the scale cancels more than there is
        // fillin
        if (numCancel > 1) sparsifyRows.emplace_back(candRow, scale);
      }
    }

    if (sparsifyRows.empty()) continue;

    postsolve_stack.equalityRowAdditions(eqrow, getStoredRow(), sparsifyRows);
    double rhs = model->row_lower_[eqrow];
    for (const auto& sparsifyRow : sparsifyRows) {
      HighsInt row = sparsifyRow.index;
      double scale = sparsifyRow.value;

      if (model->row_lower_[row] != -kHighsInf)
        model->row_lower_[row] += scale * rhs;

      if (model->row_upper_[row] != kHighsInf)
        model->row_upper_[row] += scale * rhs;

      for (HighsInt pos : rowpositions)
        addToMatrix(row, Acol[pos], scale * Avalue[pos]);

      if (model->row_lower_[row] == model->row_upper_[row] &&
          eqiters[row] != equations.end() &&
          eqiters[row]->first != rowsize[row]) {
        // if that is the case reinsert it into the equation set that is ordered
        // by sparsity
        equations.erase(eqiters[row]);
        eqiters[row] = equations.emplace(rowsize[row], row).first;
      }
    }

    HPRESOLVE_CHECKED_CALL(checkLimits(postsolve_stack));
    HPRESOLVE_CHECKED_CALL(removeRowSingletons(postsolve_stack));
    HPRESOLVE_CHECKED_CALL(removeDoubletonEquations(postsolve_stack));
  }

  return Result::kOk;
}

void HPresolve::reportReductions(const HighsLogOptions& log_options) {
  highsLogUser(log_options, HighsLogType::kInfo, "\nReporting presolve\n");
}

}  // namespace presolve
