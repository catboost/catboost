#include "md5.h"

#include <library/cpp/testing/unittest/registar.h>

Y_UNIT_TEST_SUITE(TMD5MediumTest) {
    Y_UNIT_TEST(TestOverflow) {
        if (sizeof(size_t) > sizeof(unsigned int)) {
            const size_t maxUi32 = (size_t)Max<unsigned int>();
            TArrayHolder<char> buf(new char[maxUi32]);

            memset(buf.Get(), 0, maxUi32);

            MD5 r;
            for (int i = 0; i < 5; ++i) {
                r.Update(buf.Get(), maxUi32);
            }

            char rs[33];
            TString s(r.End(rs));
            s.to_lower();

            UNIT_ASSERT_VALUES_EQUAL(s, "34a5a7ed4f0221310084e21a1e599659");
        }
    }
}
