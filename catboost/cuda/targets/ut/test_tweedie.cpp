#include "loss_functions_gpu_helper.h"

Y_UNIT_TEST_SUITE(TweedieMetricOnGpuTest) {

    Y_UNIT_TEST(TweedieTest) {
        for (ui64 seed : {0, 42, 100}) {
            for (double variance_power : {1.1, 1.5, 1.9}) {
                TestLossFunctionImpl(seed,
                                     variance_power,
                                     "variance_power",
                                     ELossFunction::Tweedie,
                                     TTweedieError(variance_power, /*isExpApprox=*/false));
            }
        }
    }
}
