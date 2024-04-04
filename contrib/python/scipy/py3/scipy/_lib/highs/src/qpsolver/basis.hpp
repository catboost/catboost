#pragma once
#ifndef __SRC_LIB_BASIS_HPP__
#define __SRC_LIB_BASIS_HPP__

#include <map>
#include <vector>

#include "instance.hpp"
#include "pricing.hpp"
#include "qpconst.hpp"
#include "runtime.hpp"
#include "snippets.hpp"
#include "util/HFactor.h"
#include "util/HVector.h"
#include "util/HVectorBase.h"

enum class BasisStatus {
  Default,
  ActiveAtLower = 1,
  ActiveAtUpper,
  ActiveAtZero,
  Inactive
};

class Basis {
  HVector buffer_vec2hvec;

  HVector& vec2hvec(const Vector& vec) {
    buffer_vec2hvec.clear();
    for (HighsInt i = 0; i < vec.num_nz; i++) {
      buffer_vec2hvec.index[i] = vec.index[i];
      buffer_vec2hvec.array[vec.index[i]] = vec.value[vec.index[i]];
    }
    buffer_vec2hvec.count = vec.num_nz;
    buffer_vec2hvec.packFlag = true;
    return buffer_vec2hvec;
  }

  Vector& hvec2vec(const HVector& hvec, Vector& target) {
    target.reset();
    for (HighsInt i = 0; i < hvec.count; i++) {
      target.index[i] = hvec.index[i];
      target.value[target.index[i]] = hvec.array[hvec.index[i]];
    }
    // for (HighsInt i = 0; i < hvec.size; i++) {
    //   target.index[i] = hvec.index[i];
    //   target.value[i] = hvec.array[i];
    // }
    target.num_nz = hvec.count;
    return target;
  }

  Vector hvec2vec(const HVector& hvec) {
    Vector vec(hvec.size);

    return hvec2vec(hvec, vec);
  }

  Runtime& runtime;
  HFactor basisfactor;
  HighsInt updatessinceinvert = 0;

  MatrixBase Atran;

  // indices of active constraints in basis
  std::vector<HighsInt> activeconstraintidx;

  // ids of constraints that are in the basis but not active
  // I need to extract those columns to get Z
  std::vector<HighsInt> nonactiveconstraintsidx;

  // ids of constraints that are in the basis
  std::vector<HighsInt> baseindex;

  std::map<int, BasisStatus> basisstatus;

  // index i: -1 if constraint not in basis, [0, num_var] if
  // constraint in basis (active or not)
  std::vector<HighsInt> constraintindexinbasisfactor;

  void build();
  void rebuild();

  // buffer to avoid recreating vectors
  Vector buffer_column_aq;
  Vector buffer_row_ep;

  // buffers to prevent multiple btran/ftran
  HighsInt buffered_q = -1;
  HighsInt buffered_p = -1;
  HVector row_ep;
  HVector col_aq;

 public:
  Basis(Runtime& rt, std::vector<HighsInt> active,
        std::vector<BasisStatus> atlower, std::vector<HighsInt> inactive);

  HighsInt getnupdatessinceinvert() { return updatessinceinvert; }

  HighsInt getnumactive() const { return activeconstraintidx.size(); };

  HighsInt getnuminactive() const { return nonactiveconstraintsidx.size(); };

  const std::vector<HighsInt>& getactive() const {
    return activeconstraintidx;
  };

  const std::vector<HighsInt>& getinactive() const {
    return nonactiveconstraintsidx;
  };

  const std::vector<HighsInt>& getindexinfactor() const {
    return constraintindexinbasisfactor;
  };

  BasisStatus getstatus(HighsInt conid) { return basisstatus[conid]; };

  void report();

  // move that constraint into V section basis (will correspond to
  // Nullspace from now on)
  void deactivate(HighsInt conid);

  QpSolverStatus activate(const Settings& settings, HighsInt conid, BasisStatus atlower,
                          HighsInt nonactivetoremove, Pricing* pricing);

  void updatebasis(const Settings& settings, HighsInt newactivecon, HighsInt droppedcon,
                   Pricing* pricing);

  Vector btran(const Vector& rhs, bool buffer = false, HighsInt p = -1);

  Vector ftran(const Vector& rhs, bool buffer = false, HighsInt q = -1);

  Vector& btran(const Vector& rhs, Vector& target, bool buffer = false,
                HighsInt p = -1);

  Vector& ftran(const Vector& rhs, Vector& target, bool buffer = false,
                HighsInt q = -1);

  Vector recomputex(const Instance& inst);

  void write(std::string filename);

  Vector& Ztprod(const Vector& rhs, Vector& target, bool buffer = false,
                 HighsInt q = -1);

  Vector& Zprod(const Vector& rhs, Vector& target);
};

#endif
