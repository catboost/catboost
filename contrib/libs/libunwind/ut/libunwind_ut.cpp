#include <library/cpp/testing/unittest/registar.h>

#include <util/system/compiler.h>

#include <exception>
#include <vector>

Y_NO_INLINE void Except(int arg, ...) {
    (void)arg;
    throw std::exception();
}

Y_UNIT_TEST_SUITE(LibunwindSuite) {
    static void Y_NO_INLINE DoTestVarargs() {
        std::vector<int> v;
        v.push_back(0);
        Except(0x11, 0x22, 0x33, 0x44, 0xAA, 0xBB, 0xCC, 0xDD);
    }

    Y_UNIT_TEST(TestVarargs) {
        try {
            DoTestVarargs();
        } catch (const std::exception& e) {
            return;
        }

        UNIT_FAIL("Should not be here");
    }
}
