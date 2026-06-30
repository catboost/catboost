#include "flock.h"
#include "file_lock.h"
#include "guard.h"
#include "tempfile.h"

#include <library/cpp/testing/unittest/registar.h>

Y_UNIT_TEST_SUITE(TFileLockTest) {
    Y_UNIT_TEST(TestFlock) {
        TTempFileHandle tmp("./file");

        UNIT_ASSERT_EQUAL(Flock(tmp.GetHandle(), LOCK_EX), 0);
        UNIT_ASSERT_EQUAL(Flock(tmp.GetHandle(), LOCK_UN), 0);
    }

    Y_UNIT_TEST(TestFileLocker) {
        TTempFileHandle tmp("./file.locker");
        TFileLock fileLockExclusive1("./file.locker");
        TFileLock fileLockExclusive2("./file.locker");
        TFileLock fileLockShared1("./file.locker", EFileLockType::Shared);
        TFileLock fileLockShared2("./file.locker", EFileLockType::Shared);
        TFileLock fileLockShared3("./file.locker", EFileLockType::Shared);
        {
            TGuard<TFileLock> guard(fileLockExclusive1);
        }
        {
            TTryGuard<TFileLock> tryGuard(fileLockExclusive1);
            UNIT_ASSERT(tryGuard.WasAcquired());
        }
        {
            TGuard<TFileLock> guard1(fileLockExclusive1);
            TTryGuard<TFileLock> guard2(fileLockExclusive2);
            UNIT_ASSERT(!guard2.WasAcquired());
        }
        {
            TGuard<TFileLock> guard1(fileLockShared1);
            TTryGuard<TFileLock> guard2(fileLockShared2);
            TTryGuard<TFileLock> guard3(fileLockShared3);
            UNIT_ASSERT(guard2.WasAcquired());
            UNIT_ASSERT(guard3.WasAcquired());
        }
        {
            TGuard<TFileLock> guard1(fileLockExclusive1);
            TTryGuard<TFileLock> guard2(fileLockShared1);
            TTryGuard<TFileLock> guard3(fileLockShared2);
            UNIT_ASSERT(!guard2.WasAcquired());
            UNIT_ASSERT(!guard3.WasAcquired());
        }
        {
            TGuard<TFileLock> guard1(fileLockShared1);
            TTryGuard<TFileLock> guard2(fileLockExclusive1);
            TTryGuard<TFileLock> guard3(fileLockShared2);
            UNIT_ASSERT(!guard2.WasAcquired());
            UNIT_ASSERT(guard3.WasAcquired());
        }
    }
} // Y_UNIT_TEST_SUITE(TFileLockTest)
