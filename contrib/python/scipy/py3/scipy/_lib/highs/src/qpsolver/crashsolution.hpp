#ifndef __SRC_LIB_CRASHSOLUTION_HPP__
#define __SRC_LIB_CRASHSOLUTION_HPP__

#include <cstdlib>
#include "runtime.hpp"

bool isfreevar(Runtime& runtime, HighsInt idx) {
  return runtime.instance.var_lo[idx] == -std::numeric_limits<double>::infinity() && runtime.instance.var_up[idx] == std::numeric_limits<double>::infinity();
}

struct CrashSolution {
  std::vector<HighsInt> active;
  std::vector<HighsInt> inactive;
  std::vector<BasisStatus> rowstatus;
  Vector primal;
  Vector rowact;

  CrashSolution(HighsInt num_var, HighsInt num_row)
      : primal(Vector(num_var)), rowact(Vector(num_row)) {}
};

#endif
