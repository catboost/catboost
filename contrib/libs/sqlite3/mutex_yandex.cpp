#include <util/system/mutex.h>

extern "C" {
#include "sqlite3.h"
}

struct sqlite3_mutex : public TMutex {
};

const size_t StaticMutexesCount = 6;

TArrayHolder<sqlite3_mutex> StaticMutexes;

/*
** Initialize and deinitialize the mutex subsystem.
*/
static int YandexMutexInit() {
    if (!StaticMutexes) {
        StaticMutexes.Reset(new sqlite3_mutex[StaticMutexesCount]);
    }
    return SQLITE_OK;
}

static int YandexMutexEnd() {
    StaticMutexes.Destroy();
    return SQLITE_OK;
}

/*
** The sqlite3_mutex_alloc() routine allocates a new
** mutex and returns a pointer to it.  If it returns NULL
** that means that a mutex could not be allocated.  SQLite
** will unwind its stack and return an error.  The argument
** to sqlite3_mutex_alloc() is one of these integer constants:
**
** <ul>
** <li>  SQLITE_MUTEX_FAST
** <li>  SQLITE_MUTEX_RECURSIVE
** <li>  SQLITE_MUTEX_STATIC_MASTER
** <li>  SQLITE_MUTEX_STATIC_MEM
** <li>  SQLITE_MUTEX_STATIC_MEM2
** <li>  SQLITE_MUTEX_STATIC_PRNG
** <li>  SQLITE_MUTEX_STATIC_LRU
** <li>  SQLITE_MUTEX_STATIC_LRU2
** </ul>
**
** The first two constants cause sqlite3_mutex_alloc() to create
** a new mutex.  The new mutex is recursive when SQLITE_MUTEX_RECURSIVE
** is used but not necessarily so when SQLITE_MUTEX_FAST is used.
** The mutex implementation does not need to make a distinction
** between SQLITE_MUTEX_RECURSIVE and SQLITE_MUTEX_FAST if it does
** not want to.  But SQLite will only request a recursive mutex in
** cases where it really needs one.  If a faster non-recursive mutex
** implementation is available on the host platform, the mutex subsystem
** might return such a mutex in response to SQLITE_MUTEX_FAST.
**
** The other allowed parameters to sqlite3_mutex_alloc() each return
** a pointer to a static preexisting mutex.  Six static mutexes are
** used by the current version of SQLite.  Future versions of SQLite
** may add additional static mutexes.  Static mutexes are for internal
** use by SQLite only.  Applications that use SQLite mutexes should
** use only the dynamic mutexes returned by SQLITE_MUTEX_FAST or
** SQLITE_MUTEX_RECURSIVE.
**
** Note that if one of the dynamic mutex parameters (SQLITE_MUTEX_FAST
** or SQLITE_MUTEX_RECURSIVE) is used then sqlite3_mutex_alloc()
** returns a different mutex on every call.  But for the static 
** mutex types, the same mutex is returned on every call that has

** the same type number.
*/
static sqlite3_mutex* YandexMutexAlloc(int type) {
    switch (type) {
    case SQLITE_MUTEX_RECURSIVE:
    case SQLITE_MUTEX_FAST:
        return new sqlite3_mutex();
    default:
        assert(!!StaticMutexes);
        assert(type - 2 >= 0);
        assert(type - 2 < StaticMutexesCount);
        return &StaticMutexes[type - 2];
  }
}


/*
** This routine deallocates a previously
** allocated mutex.  SQLite is careful to deallocate every
** mutex that it allocates.
*/
static void YandexMutexFree(sqlite3_mutex* p) {
    if (p < StaticMutexes.Get() || p >= StaticMutexes.Get() + StaticMutexesCount) {
        delete p;
    }
}

/*
** The sqlite3_mutex_enter() and sqlite3_mutex_try() routines attempt
** to enter a mutex.  If another thread is already within the mutex,
** sqlite3_mutex_enter() will block and sqlite3_mutex_try() will return
** SQLITE_BUSY.  The sqlite3_mutex_try() interface returns SQLITE_OK
** upon successful entry.  Mutexes created using SQLITE_MUTEX_RECURSIVE can
** be entered multiple times by the same thread.  In such cases the,
** mutex must be exited an equal number of times before another thread
** can enter.  If the same thread tries to enter any other kind of mutex
** more than once, the behavior is undefined.
*/
static void YandexMutexEnter(sqlite3_mutex* p) {
    p->Acquire();
}

static int YandexMutexTry(sqlite3_mutex* p) {
    if (p->TryAcquire()) {
        return SQLITE_OK;
    }
    return SQLITE_BUSY;
}

/*
** The sqlite3_mutex_leave() routine exits a mutex that was
** previously entered by the same thread.  The behavior
** is undefined if the mutex is not currently entered or
** is not currently allocated.  SQLite will never do either.
*/
static void YandexMutexLeave(sqlite3_mutex* p) {
    p->Release();
}

extern "C" {

sqlite3_mutex_methods* sqlite3DefaultMutex() {
    static sqlite3_mutex_methods MutexMethods = {
        YandexMutexInit,
        YandexMutexEnd,
        YandexMutexAlloc,
        YandexMutexFree,
        YandexMutexEnter,
        YandexMutexTry,
        YandexMutexLeave,
        0,
        0
    };

    return &MutexMethods;
}

}
