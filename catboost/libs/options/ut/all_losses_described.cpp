#include <library/unittest/registar.h>
#include <library/json/json_reader.h>
#include <catboost/libs/options/option.h>
#include <catboost/libs/options/json_helper.h>
#include <catboost/libs/options/enums.h>
#include <catboost/libs/options/enum_helpers.h>
#include <catboost/libs/options/system_options.h>
#include <catboost/libs/options/catboost_options.h>

#include <util/generic/serialized_enum.h>

Y_UNIT_TEST_SUITE(TestAllLosses) {
    Y_UNIT_TEST(TestAllLossesDescribed) {
        for (auto loss : GetEnumAllValues<ELossFunction>()) {
            (void) IsRegressionMetric(loss);
        }
    }
}
