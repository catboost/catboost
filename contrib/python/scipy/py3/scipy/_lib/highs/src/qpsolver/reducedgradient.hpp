#ifndef __SRC_LIB_REDUCEDGRADIENT_HPP__
#define __SRC_LIB_REDUCEDGRADIENT_HPP__

#include "basis.hpp"
#include "runtime.hpp"
#include "vector.hpp"

class ReducedGradient {
  Vector rg;
  bool uptodate = false;
  Gradient& gradient;
  Basis& basis;

  void recompute() {
    rg.dim = basis.getinactive().size();
    basis.Ztprod(gradient.getGradient(), rg);
    uptodate = true;
  }

 public:
  ReducedGradient(Runtime& rt, Basis& bas, Gradient& grad)
      : rg(rt.instance.num_var), gradient(grad), basis(bas) {}

  Vector& get() {
    if (!uptodate) {
      recompute();
    }
    return rg;
  }

  void reduce(const Vector& buffer_d, const HighsInt maxabsd) {
    if (!uptodate) {
      return;
    }
    // Vector r(rg.dim-1);
    // for (HighsInt col=0; col<nrr.maxabsd; col++) {
    //    r.index[col] = col;
    //    r.value[col] = -nrr.d[col] / nrr.d[nrr.maxabsd];
    // }
    // for (HighsInt col=nrr.maxabsd+1; col<rg.dim; col++) {
    //    r.index[col-1] = col-1;
    //    r.value[col-1] = -nrr.d[col] / nrr.d[nrr.maxabsd];
    // }
    // r.num_nz = rg.dim-1;

    for (HighsInt i = 0; i < buffer_d.num_nz; i++) {
      HighsInt idx = buffer_d.index[i];
      if (idx == maxabsd) {
        continue;
      }
      rg.value[idx] -=
          rg.value[maxabsd] * buffer_d.value[idx] / buffer_d.value[maxabsd];
    }

    rg.resparsify();

    uptodate = true;
  }

  void expand(const Vector& yp) {
    if (!uptodate) {
      return;
    }

    double newval = yp * gradient.getGradient();
    rg.value.push_back(newval);
    rg.index.push_back(0);
    rg.index[rg.num_nz++] = rg.dim++;

    uptodate = true;
  }

  void update(double alpha, bool minor) {
    if (!uptodate) {
      return;
    }
    if (minor) {
      for (HighsInt i = 0; i < rg.num_nz; i++) {
        rg.value[rg.index[i]] *= (1.0 - alpha);
      }
      uptodate = true;
    } else {
      uptodate = false;
    }
  }
};

#endif
