#include "guard.h"
#include "rwlock.h"

#include <library/cpp/testing/unittest/registar.h>

#include <util/thread/pool.h>

struct TTestGuard: public TTestBase {
    UNIT_TEST_SUITE(TTestGuard);
    UNIT_TEST(TestGuard)
    UNIT_TEST(TestTryGuard)
    UNIT_TEST(TestMove)
    UNIT_TEST(TestSync)
    UNIT_TEST(TestUnguard)
    UNIT_TEST(TestTryReadGuard)
    UNIT_TEST(TestWithLock)
    UNIT_TEST(TestWithLockScope);
    UNIT_TEST_SUITE_END();

    struct TGuardChecker {
        TGuardChecker()
            : guarded(false)
        {
        }

        void Acquire() {
            guarded = true;
        }
        void Release() {
            guarded = false;
        }
        bool TryAcquire() {
            if (guarded) {
                return false;
            } else {
                guarded = true;
                return true;
            }
        }

        bool guarded;
    };

    void TestUnguard() {
        TGuardChecker m;

        {
            auto guard = Guard(m);

            UNIT_ASSERT(m.guarded);

            {
                auto unguard = Unguard(guard);

                UNIT_ASSERT(!m.guarded);
            }

            UNIT_ASSERT(m.guarded);
        }

        {
            auto guard = Guard(m);

            UNIT_ASSERT(m.guarded);

            {
                auto unguard = Unguard(m);

                UNIT_ASSERT(!m.guarded);
            }

            UNIT_ASSERT(m.guarded);
        }
    }

    void TestMove() {
        TGuardChecker m;
        size_t n = 0;

        {
            auto guard = Guard(m);

            UNIT_ASSERT(m.guarded);
            ++n;
        }

        UNIT_ASSERT(!m.guarded);
        UNIT_ASSERT_VALUES_EQUAL(n, 1);
    }

    void TestSync() {
        TGuardChecker m;
        size_t n = 0;

        with_lock (m) {
            UNIT_ASSERT(m.guarded);
            ++n;
        }

        UNIT_ASSERT(!m.guarded);
        UNIT_ASSERT_VALUES_EQUAL(n, 1);
    }

    void TestGuard() {
        TGuardChecker checker;

        UNIT_ASSERT(!checker.guarded);
        {
            TGuard<TGuardChecker> guard(checker);
            UNIT_ASSERT(checker.guarded);
        }
        UNIT_ASSERT(!checker.guarded);
    }

    void TestTryGuard() {
        TGuardChecker checker;

        UNIT_ASSERT(!checker.guarded);
        {
            TTryGuard<TGuardChecker> guard(checker);
            UNIT_ASSERT(checker.guarded);
            UNIT_ASSERT(guard.WasAcquired());
            {
                TTryGuard<TGuardChecker> guard2(checker);
                UNIT_ASSERT(checker.guarded);
                UNIT_ASSERT(!guard2.WasAcquired());
            }
            UNIT_ASSERT(checker.guarded);
        }
        UNIT_ASSERT(!checker.guarded);
    }

    void TestTryReadGuard() {
        TRWMutex mutex;
        {
            TTryReadGuard tryGuard(mutex);
            UNIT_ASSERT(tryGuard.WasAcquired());
            TReadGuard readGuard(mutex);
            TTryReadGuard anotherTryGuard(mutex);
            UNIT_ASSERT(tryGuard.WasAcquired());
        }
        {
            TReadGuard readGuard(mutex);
            TTryReadGuard tryGuard(mutex);
            UNIT_ASSERT(tryGuard.WasAcquired());
        }
        {
            TWriteGuard writeGuard(mutex);
            TTryReadGuard tryGuard(mutex);
            UNIT_ASSERT(!tryGuard.WasAcquired());
        }
        TTryReadGuard tryGuard(mutex);
        UNIT_ASSERT(tryGuard.WasAcquired());
    }

    int WithLockIncrement(TGuardChecker& m, int n) {
        with_lock (m) {
            UNIT_ASSERT(m.guarded);
            return n + 1;
        }
    }

    void TestWithLock() {
        TGuardChecker m;
        int n = 42;
        n = WithLockIncrement(m, n);
        UNIT_ASSERT(!m.guarded);
        UNIT_ASSERT_EQUAL(n, 43);
    }

    void TestWithLockScope() {
        auto Guard = [](auto) { UNIT_FAIL("Non global Guard used"); return 0; };
        TGuardChecker m;
        with_lock (m) {
            Y_UNUSED(Guard);
        }
    }
};

UNIT_TEST_SUITE_REGISTRATION(TTestGuard)
