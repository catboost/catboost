#ifndef __SRC_LIB_RESULT_HPP__
#define __SRC_LIB_RESULT_HPP__

#include "vector.hpp"

enum class ProblemStatus { INDETERMINED, OPTIMAL, UNBOUNDED, INFEASIBLE };

struct Result {
  ProblemStatus status;
  Vector primalsolution;
  Vector dualsolution;

  Result(HighsInt num_var, HighsInt num_con)
      : primalsolution(num_var), dualsolution(num_con) {}
};

#endif
