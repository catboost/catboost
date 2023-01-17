#include "atomic.h"

#include <library/cpp/testing/unittest/registar.h>

#include <util/generic/ylimits.h>

template <typename TAtomic>
class TAtomicTest
    : public TTestBase {
    UNIT_TEST_SUITE(TAtomicTest);
    UNIT_TEST(TestAtomicInc1)
    UNIT_TEST(TestAtomicInc2)
    UNIT_TEST(TestAtomicGetAndInc)
    UNIT_TEST(TestAtomicDec)
    UNIT_TEST(TestAtomicGetAndDec)
    UNIT_TEST(TestAtomicAdd)
    UNIT_TEST(TestAtomicGetAndAdd)
    UNIT_TEST(TestAtomicSub)
    UNIT_TEST(TestAtomicGetAndSub)
    UNIT_TEST(TestAtomicSwap)
    UNIT_TEST(TestAtomicOr)
    UNIT_TEST(TestAtomicAnd)
    UNIT_TEST(TestAtomicXor)
    UNIT_TEST(TestCAS)
    UNIT_TEST(TestGetAndCAS)
    UNIT_TEST(TestLockUnlock)
    UNIT_TEST_SUITE_END();

private:
    inline void TestLockUnlock() {
        TAtomic v = 0;

        UNIT_ASSERT(AtomicTryLock(&v));
        UNIT_ASSERT(!AtomicTryLock(&v));
        UNIT_ASSERT_VALUES_EQUAL(v, 1);
        AtomicUnlock(&v);
        UNIT_ASSERT_VALUES_EQUAL(v, 0);
    }

    inline void TestCAS() {
        TAtomic v = 0;

        UNIT_ASSERT(AtomicCas(&v, 1, 0));
        UNIT_ASSERT(!AtomicCas(&v, 1, 0));
        UNIT_ASSERT_VALUES_EQUAL(v, 1);
        UNIT_ASSERT(AtomicCas(&v, 0, 1));
        UNIT_ASSERT_VALUES_EQUAL(v, 0);
        UNIT_ASSERT(AtomicCas(&v, Max<intptr_t>(), 0));
        UNIT_ASSERT_VALUES_EQUAL(v, Max<intptr_t>());
    }

    inline void TestGetAndCAS() {
        TAtomic v = 0;

        UNIT_ASSERT_VALUES_EQUAL(AtomicGetAndCas(&v, 1, 0), 0);
        UNIT_ASSERT_VALUES_EQUAL(AtomicGetAndCas(&v, 2, 0), 1);
        UNIT_ASSERT_VALUES_EQUAL(v, 1);
        UNIT_ASSERT_VALUES_EQUAL(AtomicGetAndCas(&v, 0, 1), 1);
        UNIT_ASSERT_VALUES_EQUAL(v, 0);
        UNIT_ASSERT_VALUES_EQUAL(AtomicGetAndCas(&v, Max<intptr_t>(), 0), 0);
        UNIT_ASSERT_VALUES_EQUAL(v, Max<intptr_t>());
    }

    inline void TestAtomicInc1() {
        TAtomic v = 0;

        UNIT_ASSERT(AtomicAdd(v, 1));
        UNIT_ASSERT_VALUES_EQUAL(v, 1);
        UNIT_ASSERT(AtomicAdd(v, 10));
        UNIT_ASSERT_VALUES_EQUAL(v, 11);
    }

    inline void TestAtomicInc2() {
        TAtomic v = 0;

        UNIT_ASSERT(AtomicIncrement(v));
        UNIT_ASSERT_VALUES_EQUAL(v, 1);
        UNIT_ASSERT(AtomicIncrement(v));
        UNIT_ASSERT_VALUES_EQUAL(v, 2);
    }

    inline void TestAtomicGetAndInc() {
        TAtomic v = 0;

        UNIT_ASSERT_EQUAL(AtomicGetAndIncrement(v), 0);
        UNIT_ASSERT_VALUES_EQUAL(v, 1);
        UNIT_ASSERT_EQUAL(AtomicGetAndIncrement(v), 1);
        UNIT_ASSERT_VALUES_EQUAL(v, 2);
    }

    inline void TestAtomicDec() {
        TAtomic v = 2;

        UNIT_ASSERT(AtomicDecrement(v));
        UNIT_ASSERT_VALUES_EQUAL(v, 1);
        UNIT_ASSERT(!AtomicDecrement(v));
        UNIT_ASSERT_VALUES_EQUAL(v, 0);
    }

    inline void TestAtomicGetAndDec() {
        TAtomic v = 2;

        UNIT_ASSERT_VALUES_EQUAL(AtomicGetAndDecrement(v), 2);
        UNIT_ASSERT_VALUES_EQUAL(v, 1);
        UNIT_ASSERT_VALUES_EQUAL(AtomicGetAndDecrement(v), 1);
        UNIT_ASSERT_VALUES_EQUAL(v, 0);
    }

    inline void TestAtomicAdd() {
        TAtomic v = 0;

        UNIT_ASSERT_VALUES_EQUAL(AtomicAdd(v, 1), 1);
        UNIT_ASSERT_VALUES_EQUAL(AtomicAdd(v, 2), 3);
        UNIT_ASSERT_VALUES_EQUAL(AtomicAdd(v, -4), -1);
        UNIT_ASSERT_VALUES_EQUAL(v, -1);
    }

    inline void TestAtomicGetAndAdd() {
        TAtomic v = 0;

        UNIT_ASSERT_VALUES_EQUAL(AtomicGetAndAdd(v, 1), 0);
        UNIT_ASSERT_VALUES_EQUAL(AtomicGetAndAdd(v, 2), 1);
        UNIT_ASSERT_VALUES_EQUAL(AtomicGetAndAdd(v, -4), 3);
        UNIT_ASSERT_VALUES_EQUAL(v, -1);
    }

    inline void TestAtomicSub() {
        TAtomic v = 4;

        UNIT_ASSERT_VALUES_EQUAL(AtomicSub(v, 1), 3);
        UNIT_ASSERT_VALUES_EQUAL(AtomicSub(v, 2), 1);
        UNIT_ASSERT_VALUES_EQUAL(AtomicSub(v, 3), -2);
        UNIT_ASSERT_VALUES_EQUAL(v, -2);
    }

    inline void TestAtomicGetAndSub() {
        TAtomic v = 4;

        UNIT_ASSERT_VALUES_EQUAL(AtomicGetAndSub(v, 1), 4);
        UNIT_ASSERT_VALUES_EQUAL(AtomicGetAndSub(v, 2), 3);
        UNIT_ASSERT_VALUES_EQUAL(AtomicGetAndSub(v, 3), 1);
        UNIT_ASSERT_VALUES_EQUAL(v, -2);
    }

    inline void TestAtomicSwap() {
        TAtomic v = 0;

        UNIT_ASSERT_VALUES_EQUAL(AtomicSwap(&v, 3), 0);
        UNIT_ASSERT_VALUES_EQUAL(AtomicSwap(&v, 5), 3);
        UNIT_ASSERT_VALUES_EQUAL(AtomicSwap(&v, -7), 5);
        UNIT_ASSERT_VALUES_EQUAL(AtomicSwap(&v, Max<intptr_t>()), -7);
        UNIT_ASSERT_VALUES_EQUAL(v, Max<intptr_t>());
    }

    inline void TestAtomicOr() {
        TAtomic v = 0xf0;

        UNIT_ASSERT_VALUES_EQUAL(AtomicOr(v, 0x0f), 0xff);
        UNIT_ASSERT_VALUES_EQUAL(v, 0xff);
    }

    inline void TestAtomicAnd() {
        TAtomic v = 0xff;

        UNIT_ASSERT_VALUES_EQUAL(AtomicAnd(v, 0xf0), 0xf0);
        UNIT_ASSERT_VALUES_EQUAL(v, 0xf0);
    }

    inline void TestAtomicXor() {
        TAtomic v = 0x00;

        UNIT_ASSERT_VALUES_EQUAL(AtomicXor(v, 0xff), 0xff);
        UNIT_ASSERT_VALUES_EQUAL(AtomicXor(v, 0xff), 0x00);
    }

    inline void TestAtomicPtr() {
        int* p;
        AtomicSet(p, nullptr);

        UNIT_ASSERT_VALUES_EQUAL(AtomicGet(p), 0);

        int i;
        AtomicSet(p, &i);

        UNIT_ASSERT_VALUES_EQUAL(AtomicGet(p), &i);
        UNIT_ASSERT_VALUES_EQUAL(AtomicSwap(&p, nullptr), &i);
        UNIT_ASSERT(AtomicCas(&p, &i, nullptr));
    }
};

UNIT_TEST_SUITE_REGISTRATION(TAtomicTest<TAtomic>);

#ifndef _MSC_VER
// chooses type *other than* T1
template <typename T1, typename T2, typename T3>
struct TChooser {
    using TdType = T2;
};

template <typename T1, typename T2>
struct TChooser<T1, T1, T2> {
    using TdType = T2;
};

template <typename T1>
struct TChooser<T1, T1, T1> {};

    #if defined(__IOS__) && defined(_32_)
using TAltAtomic = int;
    #else
using TAltAtomic = volatile TChooser<TAtomicBase, long, long long>::TdType;
    #endif

class TTTest: public TAtomicTest<TAltAtomic> {
public:
    TString Name() const noexcept override {
        return "TAtomicTest<TAltAtomic>";
    }

    static TString StaticName() noexcept {
        return "TAtomicTest<TAltAtomic>";
    }
};

UNIT_TEST_SUITE_REGISTRATION(TTTest);

#endif
