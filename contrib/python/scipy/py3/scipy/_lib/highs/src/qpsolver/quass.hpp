#ifndef __SRC_LIB_QUASS_HPP__
#define __SRC_LIB_QUASS_HPP__

#include "basis.hpp"
#include "eventhandler.hpp"
#include "factor.hpp"
#include "instance.hpp"
#include "runtime.hpp"

struct Quass {
  Quass(Runtime& rt);

  void solve(const Vector& x0, const Vector& ra, Basis& b0);

  void solve();

 private:
  Runtime& runtime;

  void loginformation(Runtime& rt, Basis& basis, CholeskyFactor& factor);
};

#endif
