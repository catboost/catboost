#ifndef __SRC_LIB_PRICING_DANTZIGPRICING_HPP__
#define __SRC_LIB_PRICING_DANTZIGPRICING_HPP__

#include "basis.hpp"
#include "pricing.hpp"
#include "reducedcosts.hpp"
#include "runtime.hpp"

// 51561, 78965838.346823, 559, 213.280772, 0.000812, 801

class DantzigPricing : public Pricing {
 private:
  Runtime& runtime;
  Basis& basis;
  ReducedCosts& redcosts;

  HighsInt chooseconstrainttodrop(const Vector& lambda) {
    auto activeconstraintidx = basis.getactive();
    auto constraintindexinbasisfactor = basis.getindexinfactor();

    HighsInt minidx = -1;
    double maxabslambda = 0.0;
    for (HighsInt i = 0; i < activeconstraintidx.size(); i++) {
      HighsInt indexinbasis =
          constraintindexinbasisfactor[activeconstraintidx[i]];
      if (indexinbasis == -1) {
        printf("error\n");
      }
      assert(indexinbasis != -1);

      if (basis.getstatus(activeconstraintidx[i]) ==
              BasisStatus::ActiveAtLower &&
          -lambda.value[indexinbasis] > maxabslambda) {
        minidx = activeconstraintidx[i];
        maxabslambda = -lambda.value[indexinbasis];
      } else if (basis.getstatus(activeconstraintidx[i]) ==
                     BasisStatus::ActiveAtUpper &&
                 lambda.value[indexinbasis] > maxabslambda) {
        minidx = activeconstraintidx[i];
        maxabslambda = lambda.value[indexinbasis];
      } else {
        // TODO
      }
    }

    if (maxabslambda <= runtime.settings.lambda_zero_threshold) {
      // printf("maxabslambda %lf\n", log(maxabslambda));
      return -1;
    }
    return minidx;
  }

 public:
  DantzigPricing(Runtime& rt, Basis& bas, ReducedCosts& rc)
      : runtime(rt), basis(bas), redcosts(rc){};

  HighsInt price(const Vector& x, const Vector& gradient) {
    // Vector lambda = basis.ftran(gradient);
    HighsInt minidx = chooseconstrainttodrop(redcosts.getReducedCosts());
    return minidx;
  }

  void update_weights(const Vector& aq, const Vector& ep, HighsInt p,
                      HighsInt q) {
    // does nothing
  }
};

#endif
