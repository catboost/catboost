#ifndef __SRC_LIB_GRADIENT_HPP__
#define __SRC_LIB_GRADIENT_HPP__

#include "runtime.hpp"
#include "vector.hpp"

class Gradient {
  Runtime& runtime;

  Vector gradient;
  bool uptodate;
  HighsInt numupdates = 0;

  void recompute() {
    runtime.instance.Q.vec_mat(runtime.primal, gradient);
    gradient += runtime.instance.c;
    uptodate = true;
    numupdates = 0;
  }

 public:
  Gradient(Runtime& rt)
      : runtime(rt), gradient(Vector(rt.instance.num_var)), uptodate(false) {}

  Vector& getGradient() {
    if (!uptodate ||
        numupdates >= runtime.settings.gradientrecomputefrequency) {
      recompute();
    }
    return gradient;
  }

  void update(Vector& buffer_Qp, double stepsize) {
    gradient.saxpy(stepsize, buffer_Qp);
    numupdates++;
  }
};

#endif
