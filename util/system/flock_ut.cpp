#include "flock.h"
#include "file_lock.h"
#include "guard.h"
#include "tempfile.h"

#include <library/unittest/registar.h>

Y_UNIT_TEST_SUITE(TFileLockTest) {
    Y_UNIT_TEST(TestFlock) {
        TTempFileHandle tmp("./file");

        UNIT_ASSERT_EQUAL(Flock(tmp.GetHandle(), LOCK_EX), 0);
        UNIT_ASSERT_EQUAL(Flock(tmp.GetHandle(), LOCK_UN), 0);
    }

    Y_UNIT_TEST(TestFileLocker) {
        TTempFileHandle tmp("./file.locker");
        TFileLock fileLock("./file.locker");
        // create lock
        {
            // acquire it
            TGuard<TFileLock> guard(fileLock);
        }
        {
            TTryGuard<TFileLock> tryGuard(fileLock);
            UNIT_ASSERT(tryGuard.WasAcquired());
        }
    }
}
