#ifndef __SRC_LIB_RATIOTEST_HPP__
#define __SRC_LIB_RATIOTEST_HPP__

#include <limits>

#include "runtime.hpp"

struct RatiotestResult {
  double alpha;
  HighsInt limitingconstraint;
  bool nowactiveatlower;
};

RatiotestResult ratiotest(Runtime& runtime, const Vector& p,
                          const Vector& rowmove, double alphastart);

#endif
