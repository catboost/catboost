#include <library/cpp/testing/unittest/registar.h>
#include <library/cpp/json/json_reader.h>
#include <catboost/private/libs/options/option.h>
#include <catboost/private/libs/options/json_helper.h>
#include <catboost/private/libs/options/enums.h>
#include <catboost/private/libs/options/enum_helpers.h>
#include <catboost/private/libs/options/system_options.h>
#include <catboost/private/libs/options/catboost_options.h>

#include <util/generic/serialized_enum.h>

Y_UNIT_TEST_SUITE(TestAllLosses) {
    Y_UNIT_TEST(TestAllLossesDescribed) {
        for (auto loss : GetEnumAllValues<ELossFunction>()) {
            (void) IsRegressionMetric(loss);
        }
    }
}
