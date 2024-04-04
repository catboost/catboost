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
#include "presolve/HighsPostsolveStack.h"

#include <numeric>

#include "lp_data/HConst.h"
#include "lp_data/HighsOptions.h"
#include "util/HighsCDouble.h"

namespace presolve {

void HighsPostsolveStack::initializeIndexMaps(HighsInt numRow,
                                              HighsInt numCol) {
  origNumRow = numRow;
  origNumCol = numCol;

  origRowIndex.resize(numRow);
  std::iota(origRowIndex.begin(), origRowIndex.end(), 0);

  origColIndex.resize(numCol);
  std::iota(origColIndex.begin(), origColIndex.end(), 0);

  linearlyTransformable.resize(numCol, true);
}

void HighsPostsolveStack::compressIndexMaps(
    const std::vector<HighsInt>& newRowIndex,
    const std::vector<HighsInt>& newColIndex) {
  // loop over rows, decrease row counter for deleted rows (marked with -1),
  // store original index at new index position otherwise
  HighsInt numRow = origRowIndex.size();
  for (size_t i = 0; i != newRowIndex.size(); ++i) {
    if (newRowIndex[i] == -1)
      --numRow;
    else
      origRowIndex[newRowIndex[i]] = origRowIndex[i];
  }
  // resize original index array to new size
  origRowIndex.resize(numRow);

  // now compress the column array
  HighsInt numCol = origColIndex.size();
  for (size_t i = 0; i != newColIndex.size(); ++i) {
    if (newColIndex[i] == -1)
      --numCol;
    else
      origColIndex[newColIndex[i]] = origColIndex[i];
  }
  origColIndex.resize(numCol);
}

void HighsPostsolveStack::LinearTransform::undo(const HighsOptions& options,
                                                HighsSolution& solution) const {
  solution.col_value[col] *= scale;
  solution.col_value[col] += constant;

  if (solution.dual_valid) solution.col_dual[col] /= scale;
}

void HighsPostsolveStack::LinearTransform::transformToPresolvedSpace(
    std::vector<double>& primalSol) const {
  primalSol[col] -= constant;
  primalSol[col] /= scale;
}

void HighsPostsolveStack::FreeColSubstitution::undo(
    const HighsOptions& options, const std::vector<Nonzero>& rowValues,
    const std::vector<Nonzero>& colValues, HighsSolution& solution,
    HighsBasis& basis) {
  double colCoef = 0;
  // compute primal values
  HighsCDouble rowValue = 0;
  for (const auto& rowVal : rowValues) {
    if (rowVal.index == col)
      colCoef = rowVal.value;
    else
      rowValue += rowVal.value * solution.col_value[rowVal.index];
  }

  assert(colCoef != 0);
  solution.row_value[row] =
      double(rowValue + colCoef * solution.col_value[col]);
  solution.col_value[col] = double((rhs - rowValue) / colCoef);

  // if no dual values requested, return here
  if (!solution.dual_valid) return;

  // compute the row dual value such that reduced cost of basic column is 0
  solution.row_dual[row] = 0;
  HighsCDouble dualval = colCost;
  for (const auto& colVal : colValues)
    dualval -= colVal.value * solution.row_dual[colVal.index];

  solution.col_dual[col] = 0;
  solution.row_dual[row] = double(dualval / colCoef);

  // set basis status if necessary
  if (!basis.valid) return;

  basis.col_status[col] = HighsBasisStatus::kBasic;
  if (rowType == RowType::kEq)
    basis.row_status[row] = solution.row_dual[row] < 0
                                ? HighsBasisStatus::kUpper
                                : HighsBasisStatus::kLower;
  else if (rowType == RowType::kGeq)
    basis.row_status[row] = HighsBasisStatus::kLower;
  else
    basis.row_status[row] = HighsBasisStatus::kUpper;
}

void HighsPostsolveStack::DoubletonEquation::undo(
    const HighsOptions& options, const std::vector<Nonzero>& colValues,
    HighsSolution& solution, HighsBasis& basis) const {
  // retrieve the row and column index, the row side and the two
  // coefficients then compute the primal values
  solution.col_value[colSubst] =
      double((rhs - HighsCDouble(coef) * solution.col_value[col]) / coefSubst);

  // can only do primal postsolve, stop here
  if (row == -1 || !solution.dual_valid) return;

  HighsBasisStatus colStatus;

  if (basis.valid) {
    if (solution.col_dual[col] > options.dual_feasibility_tolerance)
      basis.col_status[col] = HighsBasisStatus::kLower;
    else if (solution.col_dual[col] < -options.dual_feasibility_tolerance)
      basis.col_status[col] = HighsBasisStatus::kUpper;

    colStatus = basis.col_status[col];
  } else {
    if (solution.col_dual[col] > options.dual_feasibility_tolerance)
      colStatus = HighsBasisStatus::kLower;
    else if (solution.col_dual[col] < -options.dual_feasibility_tolerance)
      colStatus = HighsBasisStatus::kUpper;
    else
      colStatus = HighsBasisStatus::kBasic;
  }

  // compute the current dual values of the row and the substituted column
  // before deciding on which column becomes basic
  // for each entry in a row i of the substituted column we added the doubleton
  // equation row with scale -a_i/substCoef. Therefore the dual multiplier of
  // this row i implicitly increases the dual multiplier of this doubleton
  // equation row with that scale.
  HighsCDouble rowDual = 0.0;
  solution.row_dual[row] = 0;
  for (const auto& colVal : colValues)
    rowDual -= colVal.value * solution.row_dual[colVal.index];
  rowDual /= coefSubst;
  solution.row_dual[row] = double(rowDual);

  // the equation was also added to the objective, so the current values need to
  // be changed
  solution.col_dual[colSubst] = substCost;
  solution.col_dual[col] += substCost * coef / coefSubst;

  if ((upperTightened && colStatus == HighsBasisStatus::kUpper) ||
      (lowerTightened && colStatus == HighsBasisStatus::kLower)) {
    // column must get zero reduced cost as the current bound cannot be used
    // so alter the dual multiplier of the row to make the dual multiplier of
    // column zero
    double rowDualDelta = solution.col_dual[col] / coef;
    solution.row_dual[row] = double(rowDual + rowDualDelta);
    solution.col_dual[col] = 0.0;
    solution.col_dual[colSubst] = double(
        HighsCDouble(solution.col_dual[colSubst]) - rowDualDelta * coefSubst);

    if (basis.valid) {
      if ((std::signbit(coef) == std::signbit(coefSubst) &&
           basis.col_status[col] == HighsBasisStatus::kUpper) ||
          (std::signbit(coef) != std::signbit(coefSubst) &&
           basis.col_status[col] == HighsBasisStatus::kLower))
        basis.col_status[colSubst] = HighsBasisStatus::kLower;
      else
        basis.col_status[colSubst] = HighsBasisStatus::kUpper;
      basis.col_status[col] = HighsBasisStatus::kBasic;
    }
  } else {
    // otherwise make the reduced cost of the subsituted column zero and make
    // that column basic
    double rowDualDelta = solution.col_dual[colSubst] / coefSubst;
    solution.row_dual[row] = double(rowDual + rowDualDelta);
    solution.col_dual[colSubst] = 0.0;
    solution.col_dual[col] =
        double(HighsCDouble(solution.col_dual[col]) - rowDualDelta * coef);
    if (basis.valid) basis.col_status[colSubst] = HighsBasisStatus::kBasic;
  }

  if (!basis.valid) return;

  if (solution.row_dual[row] < 0)
    basis.row_status[row] = HighsBasisStatus::kLower;
  else
    basis.row_status[row] = HighsBasisStatus::kUpper;
}

void HighsPostsolveStack::EqualityRowAddition::undo(
    const HighsOptions& options, const std::vector<Nonzero>& eqRowValues,
    HighsSolution& solution, HighsBasis& basis) const {
  // nothing more to do if the row is zero in the dual solution or there is
  // no dual solution
  if (!solution.dual_valid || solution.row_dual[row] == 0.0) return;

  // the dual multiplier of the row implicitly increases the dual multiplier
  // of the equation with the scale the equation was added with
  solution.row_dual[addedEqRow] =
      double(HighsCDouble(eqRowScale) * solution.row_dual[row] +
             solution.row_dual[addedEqRow]);

  assert(!basis.valid);
}

void HighsPostsolveStack::EqualityRowAdditions::undo(
    const HighsOptions& options, const std::vector<Nonzero>& eqRowValues,
    const std::vector<Nonzero>& targetRows, HighsSolution& solution,
    HighsBasis& basis) const {
  // nothing more to do if the row is zero in the dual solution or there is
  // no dual solution
  if (!solution.dual_valid) return;

  // the dual multiplier of the rows where the eq row was added implicitly
  // increases the dual multiplier of the equation with the scale that was used
  // for adding the equation
  HighsCDouble eqRowDual = solution.row_dual[addedEqRow];
  for (const auto& targetRow : targetRows)
    eqRowDual +=
        HighsCDouble(targetRow.value) * solution.row_dual[targetRow.index];

  solution.row_dual[addedEqRow] = double(eqRowDual);

  assert(!basis.valid);
}

void HighsPostsolveStack::ForcingColumn::undo(
    const HighsOptions& options, const std::vector<Nonzero>& colValues,
    HighsSolution& solution, HighsBasis& basis) const {
  HighsInt nonbasicRow = -1;
  HighsBasisStatus nonbasicRowStatus = HighsBasisStatus::kNonbasic;
  double colValFromNonbasicRow = colBound;

  if (atInfiniteUpper) {
    // choose largest value as then all rows are feasible
    for (const auto& colVal : colValues) {
      double colValFromRow = solution.row_value[colVal.index] / colVal.value;
      if (colValFromRow > colValFromNonbasicRow) {
        nonbasicRow = colVal.index;
        colValFromNonbasicRow = colValFromRow;
        nonbasicRowStatus = colVal.value > 0 ? HighsBasisStatus::kLower
                                             : HighsBasisStatus::kUpper;
      }
    }
  } else {
    // choose smallest value, as then all rows are feasible
    for (const auto& colVal : colValues) {
      double colValFromRow = solution.row_value[colVal.index] / colVal.value;
      if (colValFromRow < colValFromNonbasicRow) {
        nonbasicRow = colVal.index;
        colValFromNonbasicRow = colValFromRow;
        nonbasicRowStatus = colVal.value > 0 ? HighsBasisStatus::kUpper
                                             : HighsBasisStatus::kLower;
      }
    }
  }

  solution.col_value[col] = colValFromNonbasicRow;

  if (!solution.dual_valid) return;

  solution.col_dual[col] = 0.0;

  if (!basis.valid) return;
  if (nonbasicRow == -1) {
    basis.col_status[col] =
        atInfiniteUpper ? HighsBasisStatus::kLower : HighsBasisStatus::kUpper;
  } else {
    basis.col_status[col] = HighsBasisStatus::kBasic;
    basis.row_status[nonbasicRow] = nonbasicRowStatus;
  }
}

void HighsPostsolveStack::ForcingColumnRemovedRow::undo(
    const HighsOptions& options, const std::vector<Nonzero>& rowValues,
    HighsSolution& solution, HighsBasis& basis) const {
  // we use the row value as storage for the scaled value implied on the column
  // dual
  HighsCDouble val = rhs;
  for (const auto& rowVal : rowValues)
    val -= rowVal.value * solution.col_value[rowVal.index];

  solution.row_value[row] = double(val);

  if (solution.dual_valid) solution.row_dual[row] = 0.0;
  if (basis.valid) basis.row_status[row] = HighsBasisStatus::kBasic;
}

void HighsPostsolveStack::SingletonRow::undo(const HighsOptions& options,
                                             HighsSolution& solution,
                                             HighsBasis& basis) const {
  // nothing to do if the rows dual value is zero in the dual solution or
  // there is no dual solution
  if (!solution.dual_valid) return;

  HighsBasisStatus colStatus;

  if (basis.valid) {
    if (solution.col_dual[col] > options.dual_feasibility_tolerance)
      basis.col_status[col] = HighsBasisStatus::kLower;
    else if (solution.col_dual[col] < -options.dual_feasibility_tolerance)
      basis.col_status[col] = HighsBasisStatus::kUpper;

    colStatus = basis.col_status[col];
  } else {
    if (solution.col_dual[col] > options.dual_feasibility_tolerance)
      colStatus = HighsBasisStatus::kLower;
    else if (solution.col_dual[col] < -options.dual_feasibility_tolerance)
      colStatus = HighsBasisStatus::kUpper;
    else
      colStatus = HighsBasisStatus::kBasic;
  }

  if ((!colLowerTightened || colStatus != HighsBasisStatus::kLower) &&
      (!colUpperTightened || colStatus != HighsBasisStatus::kUpper)) {
    // the tightened bound is not used in the basic solution
    // hence we simply make the row basic and give it a dual multiplier of 0
    if (basis.valid) basis.row_status[row] = HighsBasisStatus::kBasic;
    solution.row_dual[row] = 0;
    return;
  }

  // choose the row dual value such that the columns reduced cost becomes
  // zero
  solution.row_dual[row] = solution.col_dual[col] / coef;
  solution.col_dual[col] = 0;

  if (!basis.valid) return;

  switch (colStatus) {
    case HighsBasisStatus::kLower:
      assert(colLowerTightened);
      if (coef > 0)
        // tightened lower bound comes from row lower bound
        basis.row_status[row] = HighsBasisStatus::kLower;
      else
        // tightened lower bound comes from row upper bound
        basis.row_status[row] = HighsBasisStatus::kUpper;

      break;
    case HighsBasisStatus::kUpper:
      if (coef > 0)
        // tightened upper bound comes from row lower bound
        basis.row_status[row] = HighsBasisStatus::kUpper;
      else
        // tightened lower bound comes from row upper bound
        basis.row_status[row] = HighsBasisStatus::kLower;
      break;
    default:
      assert(false);
  }

  // column becomes basic
  basis.col_status[col] = HighsBasisStatus::kBasic;
}

// column fixed to lower or upper bound
void HighsPostsolveStack::FixedCol::undo(const HighsOptions& options,
                                         const std::vector<Nonzero>& colValues,
                                         HighsSolution& solution,
                                         HighsBasis& basis) const {
  // set solution value
  solution.col_value[col] = fixValue;

  if (!solution.dual_valid) return;

  // compute reduced cost

  HighsCDouble reducedCost = colCost;
  for (const auto& colVal : colValues) {
    assert((HighsInt)solution.row_dual.size() > colVal.index);
    reducedCost -= colVal.value * solution.row_dual[colVal.index];
  }

  solution.col_dual[col] = double(reducedCost);

  // set basis status
  if (basis.valid) {
    basis.col_status[col] = fixType;
    if (basis.col_status[col] == HighsBasisStatus::kNonbasic)
      basis.col_status[col] = solution.col_dual[col] >= 0
                                  ? HighsBasisStatus::kLower
                                  : HighsBasisStatus::kUpper;
  }
}

void HighsPostsolveStack::RedundantRow::undo(const HighsOptions& options,
                                             HighsSolution& solution,
                                             HighsBasis& basis) const {
  // set row dual to zero if dual solution requested
  if (!solution.dual_valid) return;

  solution.row_dual[row] = 0.0;

  if (basis.valid) basis.row_status[row] = HighsBasisStatus::kBasic;
}

void HighsPostsolveStack::ForcingRow::undo(
    const HighsOptions& options, const std::vector<Nonzero>& rowValues,
    HighsSolution& solution, HighsBasis& basis) const {
  if (!solution.dual_valid) return;

  // compute the row dual multiplier and determine the new basic column
  HighsInt basicCol = -1;
  double dualDelta = 0;
  if (rowType == RowType::kLeq) {
    for (const auto& rowVal : rowValues) {
      double colDual =
          solution.col_dual[rowVal.index] - rowVal.value * dualDelta;
      if (colDual * rowVal.value < 0) {
        // column is dual infeasible, decrease the row dual such that its
        // reduced cost become zero and remember this column as the new basic
        // column for this row
        dualDelta = solution.col_dual[rowVal.index] / rowVal.value;
        basicCol = rowVal.index;
      }
    }
  } else {
    for (const auto& rowVal : rowValues) {
      double colDual =
          solution.col_dual[rowVal.index] - rowVal.value * dualDelta;
      if (colDual * rowVal.value > 0) {
        // column is dual infeasible, decrease the row dual such that its
        // reduced cost become zero and remember this column as the new basic
        // column for this row
        dualDelta = solution.col_dual[rowVal.index] / rowVal.value;
        basicCol = rowVal.index;
      }
    }
  }

  if (basicCol != -1) {
    solution.row_dual[row] = solution.row_dual[row] + dualDelta;
    for (const auto& rowVal : rowValues) {
      solution.col_dual[rowVal.index] =
          double(solution.col_dual[rowVal.index] -
                 HighsCDouble(dualDelta) * rowVal.value);
    }
    solution.col_dual[basicCol] = 0;

    if (basis.valid) {
      basis.row_status[row] =
          (rowType == RowType::kGeq ? HighsBasisStatus::kLower
                                    : HighsBasisStatus::kUpper);

      basis.col_status[basicCol] = HighsBasisStatus::kBasic;
    }
  }
}

void HighsPostsolveStack::DuplicateRow::undo(const HighsOptions& options,
                                             HighsSolution& solution,
                                             HighsBasis& basis) const {
  if (!solution.dual_valid) return;
  if (!rowUpperTightened && !rowLowerTightened) {
    // simple case of row2 being redundant, in which case it just gets a
    // dual multiplier of 0 and is made basic
    solution.row_dual[duplicateRow] = 0.0;
    if (basis.valid) basis.row_status[duplicateRow] = HighsBasisStatus::kBasic;
    return;
  }

  HighsBasisStatus rowStatus;

  if (basis.valid) {
    if (solution.row_dual[row] < -options.dual_feasibility_tolerance)
      basis.row_status[row] = HighsBasisStatus::kUpper;
    else if (solution.row_dual[row] > options.dual_feasibility_tolerance)
      basis.row_status[row] = HighsBasisStatus::kLower;

    rowStatus = basis.row_status[row];
  } else {
    if (solution.row_dual[row] < -options.dual_feasibility_tolerance)
      rowStatus = HighsBasisStatus::kUpper;
    else if (solution.row_dual[row] > options.dual_feasibility_tolerance)
      rowStatus = HighsBasisStatus::kLower;
    else
      rowStatus = HighsBasisStatus::kBasic;
  }

  // at least one bound of the row was tightened by using the bound of the
  // scaled parallel row, hence we might need to make the parallel row
  // nonbasic and the row basic

  switch (rowStatus) {
    case HighsBasisStatus::kBasic:
      // if row is basic the parallel row is also basic
      solution.row_dual[duplicateRow] = 0.0;
      if (basis.valid)
        basis.row_status[duplicateRow] = HighsBasisStatus::kBasic;
      break;
    case HighsBasisStatus::kUpper:
      // if row sits on its upper bound, and the row upper bound was
      // tightened using the parallel row we make the row basic and
      // transfer its dual value to the parallel row with the proper scale
      if (rowUpperTightened) {
        solution.row_dual[duplicateRow] =
            solution.row_dual[row] / duplicateRowScale;
        solution.row_dual[row] = 0.0;
        if (basis.valid) {
          basis.row_status[row] = HighsBasisStatus::kBasic;
          if (duplicateRowScale > 0)
            basis.row_status[duplicateRow] = HighsBasisStatus::kUpper;
          else
            basis.row_status[duplicateRow] = HighsBasisStatus::kLower;
        }
      } else {
        solution.row_dual[duplicateRow] = 0.0;
        if (basis.valid)
          basis.row_status[duplicateRow] = HighsBasisStatus::kBasic;
      }
      break;
    case HighsBasisStatus::kLower:
      if (rowLowerTightened) {
        solution.row_dual[duplicateRow] =
            solution.row_dual[row] / duplicateRowScale;
        solution.row_dual[row] = 0.0;
        if (basis.valid) {
          basis.row_status[row] = HighsBasisStatus::kBasic;
          if (duplicateRowScale > 0)
            basis.row_status[duplicateRow] = HighsBasisStatus::kUpper;
          else
            basis.row_status[duplicateRow] = HighsBasisStatus::kLower;
        }
      } else {
        solution.row_dual[duplicateRow] = 0.0;
        if (basis.valid)
          basis.row_status[duplicateRow] = HighsBasisStatus::kBasic;
      }
      break;
    default:
      assert(false);
  }
}

void HighsPostsolveStack::DuplicateColumn::undo(const HighsOptions& options,
                                                HighsSolution& solution,
                                                HighsBasis& basis) const {
  // the column dual of the duplicate column is easily computed by scaling
  // since col * colScale yields the coefficient values and cost of the
  // duplicate column.
  if (solution.dual_valid)
    solution.col_dual[duplicateCol] = solution.col_dual[col] * colScale;

  if (basis.valid) {
    // do postsolve using basis status if a basis is available:
    // if the merged column is nonbasic, we can just set both columns
    // to the corresponding basis status and value
    switch (basis.col_status[col]) {
      case HighsBasisStatus::kLower: {
        solution.col_value[col] = colLower;
        if (colScale > 0) {
          basis.col_status[duplicateCol] = HighsBasisStatus::kLower;
          solution.col_value[duplicateCol] = duplicateColLower;
        } else {
          basis.col_status[duplicateCol] = HighsBasisStatus::kUpper;
          solution.col_value[duplicateCol] = duplicateColUpper;
        }
        // nothing else to do
        return;
      }
      case HighsBasisStatus::kUpper: {
        solution.col_value[col] = colUpper;
        if (colScale > 0) {
          basis.col_status[duplicateCol] = HighsBasisStatus::kUpper;
          solution.col_value[duplicateCol] = duplicateColUpper;
        } else {
          basis.col_status[duplicateCol] = HighsBasisStatus::kLower;
          solution.col_value[duplicateCol] = duplicateColLower;
        }
        // nothing else to do
        return;
      }
      case HighsBasisStatus::kZero: {
        solution.col_value[col] = 0.0;
        basis.col_status[duplicateCol] = HighsBasisStatus::kZero;
        solution.col_value[duplicateCol] = 0.0;
        // nothing else to do
        return;
      }
      case HighsBasisStatus::kBasic:
      case HighsBasisStatus::kNonbasic:;
    }

    assert(basis.col_status[col] == HighsBasisStatus::kBasic);
  }

  // either no basis for postsolve, or column status is basic. One of
  // the two columns must become nonbasic. In case of integrality it is
  // simpler to choose col, since it has a coefficient of +1 in the equation y
  // = col + colScale * duplicateCol where the merged column is y and is
  // currently using the index of col. The duplicateCol can have a positive or
  // negative coefficient. So for postsolve, we first start out with col
  // sitting at the lower bound and compute the corresponding value for the
  // duplicate column as (y - colLower)/colScale. Then the following things
  // might happen:
  // - case 1: the value computed for duplicateCol is within the bounds
  // - case 1.1: duplicateCol is continuous -> accept value, make col nonbasic
  // at lower and duplicateCol basic
  // - case 1.2: duplicateCol is integer -> accept value if integer feasible,
  // otherwise round down and compute value of col as
  // col = y - colScale * duplicateCol
  // - case 2: the value for duplicateCol violates the column bounds: make it
  // sit at the bound that is violated
  //           and compute the value of col as col = y - colScale *
  //           duplicateCol for basis postsolve col is basic and duplicateCol
  //           nonbasic at lower/upper depending on which bound is violated.

  double mergeVal = solution.col_value[col];

  if (colLower != -kHighsInf)
    solution.col_value[col] = colLower;
  else
    solution.col_value[col] = std::min(0.0, colUpper);
  solution.col_value[duplicateCol] =
      double((HighsCDouble(mergeVal) - solution.col_value[col]) / colScale);

  bool recomputeCol = false;

  if (solution.col_value[duplicateCol] > duplicateColUpper) {
    solution.col_value[duplicateCol] = duplicateColUpper;
    recomputeCol = true;
    if (basis.valid) basis.col_status[duplicateCol] = HighsBasisStatus::kUpper;
  } else if (solution.col_value[duplicateCol] < duplicateColLower) {
    solution.col_value[duplicateCol] = duplicateColLower;
    recomputeCol = true;
    if (basis.valid) basis.col_status[duplicateCol] = HighsBasisStatus::kLower;
  } else if (duplicateColIntegral) {
    double roundVal = std::round(solution.col_value[duplicateCol]);
    if (std::abs(roundVal - solution.col_value[duplicateCol]) >
        options.mip_feasibility_tolerance) {
      solution.col_value[duplicateCol] =
          std::floor(solution.col_value[duplicateCol]);
      recomputeCol = true;
    }
  }

  if (recomputeCol) {
    solution.col_value[col] =
        mergeVal - colScale * solution.col_value[duplicateCol];
    if (!duplicateColIntegral && colIntegral) {
      // if column is integral and duplicateCol is not we need to make sure
      // we split the values into an integral one for col
      solution.col_value[col] = std::ceil(solution.col_value[col] -
                                          options.mip_feasibility_tolerance);
      solution.col_value[duplicateCol] =
          double((HighsCDouble(mergeVal) - solution.col_value[col]) / colScale);
    }
  } else {
    // setting col to its lower bound yielded a feasible value for
    // duplicateCol
    if (basis.valid) {
      basis.col_status[duplicateCol] = basis.col_status[col];
      basis.col_status[col] = HighsBasisStatus::kLower;
    }
  }
}

void HighsPostsolveStack::DuplicateColumn::transformToPresolvedSpace(
    std::vector<double>& primalSol) const {
  primalSol[col] = primalSol[col] + colScale * primalSol[duplicateCol];
}

}  // namespace presolve
