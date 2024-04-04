#include "basis.hpp"

#include <cassert>
#include <memory>

Basis::Basis(Runtime& rt, std::vector<HighsInt> active,
             std::vector<BasisStatus> lower, std::vector<HighsInt> inactive)
    : runtime(rt),
      buffer_column_aq(rt.instance.num_var),
      buffer_row_ep(rt.instance.num_var) {
  buffer_vec2hvec.setup(rt.instance.num_var);
  for (HighsInt i = 0; i < active.size(); i++) {
    activeconstraintidx.push_back(active[i]);
    basisstatus[activeconstraintidx[i]] = lower[i];
  }
  for (HighsInt i : inactive) {
    nonactiveconstraintsidx.push_back(i);
  }

  Atran = rt.instance.A.t();

  col_aq.setup(rt.instance.num_var);
  row_ep.setup(rt.instance.num_var);

  build();
}

void Basis::build() {
  updatessinceinvert = 0;

  baseindex.resize(activeconstraintidx.size() + nonactiveconstraintsidx.size());
  constraintindexinbasisfactor.clear();

  basisfactor = HFactor();

  constraintindexinbasisfactor.assign(Atran.num_row + Atran.num_col, -1);
  assert(nonactiveconstraintsidx.size() + activeconstraintidx.size() ==
         Atran.num_row);

  HighsInt counter = 0;
  for (HighsInt i : nonactiveconstraintsidx) {
    baseindex[counter++] = i;
  }
  for (HighsInt i : activeconstraintidx) {
    baseindex[counter++] = i;
  }

  const bool empty_matrix = (int)Atran.index.size() == 0;
  if (empty_matrix) {
    // The index/value vectors have size zero if the matrix has no
    // columns. However, in the Windows build, referring to index 0 of a
    // vector of size zero causes a failure, so resize to 1 to prevent
    // this.
    assert(Atran.num_col == 0);
    Atran.index.resize(1);
    Atran.value.resize(1);
  }
  basisfactor.setup(Atran.num_col, Atran.num_row, (HighsInt*)&Atran.start[0],
                    (HighsInt*)&Atran.index[0], (const double*)&Atran.value[0],
                    (HighsInt*)&baseindex[0]);
  basisfactor.build();

  for (size_t i = 0;
       i < activeconstraintidx.size() + nonactiveconstraintsidx.size(); i++) {
    constraintindexinbasisfactor[baseindex[i]] = i;
  }
}

void Basis::rebuild() {
  updatessinceinvert = 0;
  constraintindexinbasisfactor.clear();

  constraintindexinbasisfactor.assign(Atran.num_row + Atran.num_col, -1);
  assert(nonactiveconstraintsidx.size() + activeconstraintidx.size() ==
         Atran.num_row);

  basisfactor.build();

  for (size_t i = 0;
       i < activeconstraintidx.size() + nonactiveconstraintsidx.size(); i++) {
    constraintindexinbasisfactor[baseindex[i]] = i;
  }
}

void Basis::report() {
  printf("basis: ");
  for (HighsInt a_ : activeconstraintidx) {
    printf("%" HIGHSINT_FORMAT " ", a_);
  }
  printf(" - ");
  for (HighsInt n_ : nonactiveconstraintsidx) {
    printf("%" HIGHSINT_FORMAT " ", n_);
  }
  printf("\n");
}

// move that constraint into V section basis (will correspond to Nullspace
// from now on)
void Basis::deactivate(HighsInt conid) {
  // printf("deact %" HIGHSINT_FORMAT "\n", conid);
  assert(contains(activeconstraintidx, conid));
  basisstatus.erase(conid);
  remove(activeconstraintidx, conid);
  nonactiveconstraintsidx.push_back(conid);
}

QpSolverStatus Basis::activate(const Settings& settings, HighsInt conid, BasisStatus atlower,
                               HighsInt nonactivetoremove, Pricing* pricing) {
  // printf("activ %" HIGHSINT_FORMAT "\n", conid);
  if (!contains(activeconstraintidx, (HighsInt)conid)) {
    basisstatus[conid] = atlower;
    activeconstraintidx.push_back(conid);
  } else {
    printf("Degeneracy? constraint %" HIGHSINT_FORMAT " already in basis\n",
           conid);
    return QpSolverStatus::DEGENERATE;
  }

  // printf("drop %d\n", nonactivetoremove);
  // remove non-active row from basis
  HighsInt rowtoremove = constraintindexinbasisfactor[nonactivetoremove];

  baseindex[rowtoremove] = conid;
  remove(nonactiveconstraintsidx, nonactivetoremove);
  updatebasis(settings, conid, nonactivetoremove, pricing);

  if (updatessinceinvert != 0) {
    constraintindexinbasisfactor[nonactivetoremove] = -1;
    constraintindexinbasisfactor[conid] = rowtoremove;
  }
  return QpSolverStatus::OK;
}

void Basis::updatebasis(const Settings& settings, HighsInt newactivecon, HighsInt droppedcon,
                        Pricing* pricing) {
  if (newactivecon == droppedcon) {
    return;
  }

  HighsInt hint = 99999;

  HighsInt droppedcon_rowindex = constraintindexinbasisfactor[droppedcon];
  if (buffered_p != droppedcon) {
    row_ep.clear();
    row_ep.packFlag = true;
    row_ep.index[0] = droppedcon_rowindex;
    row_ep.array[droppedcon_rowindex] = 1.0;
    row_ep.count = 1;
    basisfactor.btranCall(row_ep, 1.0);
  }

  pricing->update_weights(hvec2vec(col_aq), hvec2vec(row_ep), droppedcon,
                          newactivecon);
  HighsInt row_out = droppedcon_rowindex;

  basisfactor.update(&col_aq, &row_ep, &row_out, &hint);

  updatessinceinvert++;
  if (updatessinceinvert >= settings.reinvertfrequency || hint != 99999) {
    rebuild();
  }
  // since basis changed, buffered values are no longer valid
  buffered_p = -1;
  buffered_q = -1;
}

Vector& Basis::btran(const Vector& rhs, Vector& target, bool buffer,
                     HighsInt p) {
  HVector rhs_hvec = vec2hvec(rhs);
  basisfactor.btranCall(rhs_hvec, 1.0);
  if (buffer) {
    row_ep.copy(&rhs_hvec);
    for (HighsInt i = 0; i < rhs_hvec.packCount; i++) {
      row_ep.packIndex[i] = rhs_hvec.packIndex[i];
      row_ep.packValue[i] = rhs_hvec.packValue[i];
    }
    row_ep.packCount = rhs_hvec.packCount;
    row_ep.packFlag = rhs_hvec.packFlag;
    buffered_q = p;
  }
  return hvec2vec(rhs_hvec, target);
}

Vector Basis::btran(const Vector& rhs, bool buffer, HighsInt p) {
  HVector rhs_hvec = vec2hvec(rhs);
  basisfactor.btranCall(rhs_hvec, 1.0);
  if (buffer) {
    row_ep.copy(&rhs_hvec);
    for (HighsInt i = 0; i < rhs_hvec.packCount; i++) {
      row_ep.packIndex[i] = rhs_hvec.packIndex[i];
      row_ep.packValue[i] = rhs_hvec.packValue[i];
    }
    row_ep.packCount = rhs_hvec.packCount;
    row_ep.packFlag = rhs_hvec.packFlag;
    buffered_q = p;
  }
  return hvec2vec(rhs_hvec);
}

Vector& Basis::ftran(const Vector& rhs, Vector& target, bool buffer,
                     HighsInt q) {
  HVector rhs_hvec = vec2hvec(rhs);
  basisfactor.ftranCall(rhs_hvec, 1.0);
  if (buffer) {
    col_aq.copy(&rhs_hvec);
    for (HighsInt i = 0; i < rhs_hvec.packCount; i++) {
      col_aq.packIndex[i] = rhs_hvec.packIndex[i];
      col_aq.packValue[i] = rhs_hvec.packValue[i];
    }
    col_aq.packCount = rhs_hvec.packCount;
    col_aq.packFlag = rhs_hvec.packFlag;
    buffered_q = q;
  }

  return hvec2vec(rhs_hvec, target);
}

Vector Basis::ftran(const Vector& rhs, bool buffer, HighsInt q) {
  HVector rhs_hvec = vec2hvec(rhs);
  basisfactor.ftranCall(rhs_hvec, 1.0);
  if (buffer) {
    col_aq.copy(&rhs_hvec);
    for (HighsInt i = 0; i < rhs_hvec.packCount; i++) {
      col_aq.packIndex[i] = rhs_hvec.packIndex[i];
      col_aq.packValue[i] = rhs_hvec.packValue[i];
    }
    col_aq.packCount = rhs_hvec.packCount;
    col_aq.packFlag = rhs_hvec.packFlag;
    buffered_q = q;
  }
  return hvec2vec(rhs_hvec);
}

Vector Basis::recomputex(const Instance& inst) {
  assert(activeconstraintidx.size() == inst.num_var);
  Vector rhs(inst.num_var);

  for (HighsInt i = 0; i < inst.num_var; i++) {
    HighsInt con = activeconstraintidx[i];
    if (constraintindexinbasisfactor[con] == -1) {
      printf("error\n");
    }
    if (basisstatus[con] == BasisStatus::ActiveAtLower) {
      if (con < inst.num_con) {
        rhs.value[constraintindexinbasisfactor[con]] = inst.con_lo[con];
      } else {
        rhs.value[constraintindexinbasisfactor[con]] =
            inst.var_lo[con - inst.num_con];
      }
    } else {
      if (con < inst.num_con) {
        rhs.value[constraintindexinbasisfactor[con]] = inst.con_up[con];
        // rhs.value[i] = inst.con_up[con];
      } else {
        rhs.value[constraintindexinbasisfactor[con]] =
            inst.var_up[con - inst.num_con];
        // rhs.value[i] = inst.var_up[con - inst.num_con];
      }
    }

    rhs.index[i] = i;
    rhs.num_nz++;
  }
  HVector rhs_hvec = vec2hvec(rhs);
  basisfactor.btranCall(rhs_hvec, 1.0);
  return hvec2vec(rhs_hvec);
}

Vector& Basis::Ztprod(const Vector& rhs, Vector& target, bool buffer,
                      HighsInt q) {
  Vector res_ = ftran(rhs, buffer, q);

  target.reset();
  for (HighsInt i = 0; i < nonactiveconstraintsidx.size(); i++) {
    HighsInt nonactive = nonactiveconstraintsidx[i];
    HighsInt idx = constraintindexinbasisfactor[nonactive];
    target.index[i] = i;
    target.value[i] = res_.value[idx];
  }
  target.resparsify();
  return target;
}

Vector& Basis::Zprod(const Vector& rhs, Vector& target) {
  Vector temp(target.dim);
  for (HighsInt i = 0; i < rhs.num_nz; i++) {
    HighsInt nz = rhs.index[i];
    HighsInt nonactive = nonactiveconstraintsidx[nz];
    HighsInt idx = constraintindexinbasisfactor[nonactive];
    temp.index[i] = idx;
    temp.value[idx] = rhs.value[nz];
  }
  temp.num_nz = rhs.num_nz;
  return btran(temp, target);
}

// void Basis::write(std::string filename) {
//    FILE* file = fopen(filename.c_str(), "w");

//    fprintf(file, "%lu %lu\n", activeconstraintidx.size(),
//    nonactiveconstraintsidx.size()); for (HighsInt i=0;
//    i<activeconstraintidx.size(); i++) {
//       fprintf(file, "%" HIGHSINT_FORMAT " %" HIGHSINT_FORMAT "\n",
//       activeconstraintidx[i], (HighsInt)rowstatus[i]);
//    }
//    for (HighsInt i=0; i<nonactiveconstraintsidx.size(); i++) {
//       fprintf(file, "%" HIGHSINT_FORMAT " %" HIGHSINT_FORMAT "\n",
//       nonactiveconstraintsidx[i], (HighsInt)rowstatus[i]);
//    }
//    // TODO

//    fclose(file);
// }
