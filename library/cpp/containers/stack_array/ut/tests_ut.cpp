#include <library/cpp/containers/stack_array/stack_array.h>
#include <library/cpp/testing/unittest/registar.h>

Y_UNIT_TEST_SUITE(TestStackArray) {
    using namespace NStackArray;

    static inline void* FillWithTrash(void* d, size_t l) {
        memset(d, 0xCC, l);

        return d;
    }

#define ALLOC(type, len) FillWithTrash(alloca(sizeof(type) * len), sizeof(type) * len), len

    Y_UNIT_TEST(Test1) {
        TStackArray<ui32> s(ALLOC(ui32, 10));

        UNIT_ASSERT_VALUES_EQUAL(s.size(), 10);

        for (size_t i = 0; i < s.size(); ++i) {
            UNIT_ASSERT_VALUES_EQUAL(s[i], 0xCCCCCCCC);
        }

        for (auto&& x : s) {
            UNIT_ASSERT_VALUES_EQUAL(x, 0xCCCCCCCC);
        }

        for (size_t i = 0; i < s.size(); ++i) {
            s[i] = i;
        }

        size_t ss = 0;

        for (auto&& x : s) {
            ss += x;
        }

        UNIT_ASSERT_VALUES_EQUAL(ss, 45);
    }

    static int N1 = 0;

    struct TX1 {
        inline TX1() {
            ++N1;
        }

        inline ~TX1() {
            --N1;
        }
    };

    Y_UNIT_TEST(Test2) {
        {
            TStackArray<TX1> s(ALLOC(TX1, 10));

            UNIT_ASSERT_VALUES_EQUAL(N1, 10);
        }

        UNIT_ASSERT_VALUES_EQUAL(N1, 0);
    }

    static int N2 = 0;
    static int N3 = 0;

    struct TX2 {
        inline TX2() {
            if (N2 >= 5) {
                ythrow yexception() << "ups";
            }

            ++N3;
            ++N2;
        }

        inline ~TX2() {
            --N2;
        }
    };

    Y_UNIT_TEST(Test3) {
        bool haveException = false;

        try {
            TStackArray<TX2> s(ALLOC_ON_STACK(TX2, 10));
        } catch (...) {
            haveException = true;
        }

        UNIT_ASSERT(haveException);
        UNIT_ASSERT_VALUES_EQUAL(N2, 0);
        UNIT_ASSERT_VALUES_EQUAL(N3, 5);
    }
}
