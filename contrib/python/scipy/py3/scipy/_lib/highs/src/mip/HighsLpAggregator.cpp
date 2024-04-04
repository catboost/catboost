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

#include "mip/HighsLpAggregator.h"

#include "mip/HighsLpRelaxation.h"

HighsLpAggregator::HighsLpAggregator(const HighsLpRelaxation& lprelaxation)
    : lprelaxation(lprelaxation) {
  vectorsum.setDimension(lprelaxation.getLp().num_row_ +
                         lprelaxation.getLp().num_col_);
}

void HighsLpAggregator::addRow(HighsInt row, double weight) {
  HighsInt len;
  const double* vals;
  const HighsInt* inds;
  lprelaxation.getRow(row, len, inds, vals);

  for (HighsInt i = 0; i != len; ++i) vectorsum.add(inds[i], weight * vals[i]);

  vectorsum.add(lprelaxation.getLp().num_col_ + row, -weight);
}

void HighsLpAggregator::getCurrentAggregation(std::vector<HighsInt>& inds,
                                              std::vector<double>& vals,
                                              bool negate) {
  const double droptol =
      lprelaxation.getMipSolver().options_mip_->small_matrix_value;
  const HighsInt numCol = lprelaxation.numCols();
  vectorsum.cleanup([droptol, numCol](HighsInt col, double val) {
    // only drop values for columns and not for slack variables of rows as the
    // former are the only ones which might be subject to numerical error. The
    // values for row slack variables are exact copies of the weights used
    // (assuming that separators only add each row once). Also all the
    // separators will only allow rows that do have meaningful coefficient
    // contributions with the used weight
    return col < numCol && std::fabs(val) <= droptol;
  });

  inds = vectorsum.getNonzeros();
  HighsInt len = inds.size();
  vals.resize(len);

  if (negate)
    for (HighsInt i = 0; i != len; ++i) vals[i] = -vectorsum.getValue(inds[i]);
  else
    for (HighsInt i = 0; i != len; ++i) vals[i] = vectorsum.getValue(inds[i]);
}

void HighsLpAggregator::clear() { vectorsum.clear(); }
