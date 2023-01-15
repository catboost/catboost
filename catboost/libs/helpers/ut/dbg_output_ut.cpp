#include <catboost/libs/helpers/dbg_output.h>

#include <util/generic/array_ref.h>
#include <util/generic/string.h>
#include <util/generic/vector.h>
#include <util/string/builder.h>

#include <library/cpp/testing/unittest/registar.h>


Y_UNIT_TEST_SUITE(TDbgOutput) {
    Y_UNIT_TEST(DbgDumpWithIndices) {
        TVector<int> v = {1, 2, 3};

        UNIT_ASSERT_VALUES_EQUAL(
            TStringBuilder() << NCB::DbgDumpWithIndices<int>(v),
            "[0:1, 1:2, 2:3]"
        );

        UNIT_ASSERT_VALUES_EQUAL(
            TStringBuilder() << NCB::DbgDumpWithIndices<int>(v, true),
            "[\n\t0:1\n\t1:2\n\t2:3\n]\n"
        );

    }
}
