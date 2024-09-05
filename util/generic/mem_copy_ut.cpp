#include "mem_copy.h"

#include <library/cpp/testing/unittest/registar.h>

namespace {
    class TAssignBCalled: public yexception {
    };

    struct TB {
        inline TB& operator=(const TB&) {
            throw TAssignBCalled();

            return *this;
        }
    };

    struct TC: public TB {
    };
} // namespace

Y_DECLARE_PODTYPE(TB);

Y_UNIT_TEST_SUITE(TestMemCopy) {
    Y_UNIT_TEST(Test1) {
        char buf[] = "123";
        char buf1[sizeof(buf)];

        UNIT_ASSERT_EQUAL(MemCopy(buf1, buf, sizeof(buf)), buf1);

        for (size_t i = 0; i < sizeof(buf); ++i) {
            UNIT_ASSERT_VALUES_EQUAL(buf[i], buf1[i]);
        }
    }

    static int x = 0;

    struct TA {
        inline TA() {
            X = ++x;
        }

        int X;
    };

    Y_UNIT_TEST(Test2) {
        x = 0;

        TA a1[5];
        TA a2[5];

        UNIT_ASSERT_VALUES_EQUAL(a1[0].X, 1);
        UNIT_ASSERT_VALUES_EQUAL(a2[0].X, 6);

        MemCopy(a2, a1, 5);

        for (size_t i = 0; i < 5; ++i) {
            UNIT_ASSERT_VALUES_EQUAL(a1[i].X, a2[i].X);
        }
    }

    Y_UNIT_TEST(Test3) {
        TB b1[5];
        TB b2[5];

        MemCopy(b2, b1, 5);
    }

    Y_UNIT_TEST(Test4) {
        TC c1[5];
        TC c2[5];

        UNIT_ASSERT_EXCEPTION(MemCopy(c2, c1, 5), TAssignBCalled);
    }

    template <class T>
    inline void FillX(T* b, T* e) {
        int tmp = 0;

        while (b != e) {
            (b++)->X = ++tmp;
        }
    }

    Y_UNIT_TEST(Test5) {
        struct TD {
            int X;
        };

        TD orig[50];

        for (ssize_t i = -15; i < 15; ++i) {
            TD* b = orig + 20;
            TD* e = b + 10;

            FillX(b, e);

            TD* to = b + i;

            MemMove(to, b, e - b - 1);

            for (size_t j = 0; j < (e - b) - (size_t)1; ++j) {
                UNIT_ASSERT_VALUES_EQUAL(to[j].X, j + 1);
            }
        }
    }

    Y_UNIT_TEST(TestEmpty) {
        char* tmp = nullptr;

        UNIT_ASSERT(MemCopy(tmp, tmp, 0) == nullptr);
        UNIT_ASSERT(MemMove(tmp, tmp, 0) == nullptr);
    }
} // Y_UNIT_TEST_SUITE(TestMemCopy)
