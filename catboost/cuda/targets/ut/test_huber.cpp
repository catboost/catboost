#include "loss_functions_gpu_helper.h"

Y_UNIT_TEST_SUITE(HuberMetricOnGpuTest) {

    Y_UNIT_TEST(HuberTest) {
        for (ui64 seed : {0, 42, 100}) {
            for (double delta : {0.01, 0.5, 1.0, 1.5}) {
                TestLossFunctionImpl(seed,
                                     delta,
                                     "delta",
                                     ELossFunction::Huber,
                                     THuberError(delta, /*isExpApprox=*/false));
            }
        }
    }
}
