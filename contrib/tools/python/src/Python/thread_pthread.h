
/* Posix threads interface */

#include <stdlib.h>
#include <string.h>
#if defined(__APPLE__) || defined(HAVE_PTHREAD_DESTRUCTOR)
#define destructor xxdestructor
#endif
#include <pthread.h>
#if defined(__APPLE__) || defined(HAVE_PTHREAD_DESTRUCTOR)
#undef destructor
#endif
#include <signal.h>

/* The POSIX spec requires that use of pthread_attr_setstacksize
   be conditional on _POSIX_THREAD_ATTR_STACKSIZE being defined. */
#ifdef _POSIX_THREAD_ATTR_STACKSIZE
#ifndef THREAD_STACK_SIZE
#define THREAD_STACK_SIZE       0       /* use default stack size */
#endif

#if (defined(__APPLE__) || defined(__FreeBSD__)) && defined(THREAD_STACK_SIZE) && THREAD_STACK_SIZE == 0
   /* The default stack size for new threads on OSX is small enough that
    * we'll get hard crashes instead of 'maximum recursion depth exceeded'
    * exceptions.
    *
    * The default stack size below is the minimal stack size where a
    * simple recursive function doesn't cause a hard crash.
    */
#undef  THREAD_STACK_SIZE
#define THREAD_STACK_SIZE       0x400000
#endif
/* for safety, ensure a viable minimum stacksize */
#define THREAD_STACK_MIN        0x8000  /* 32kB */
#else  /* !_POSIX_THREAD_ATTR_STACKSIZE */
#ifdef THREAD_STACK_SIZE
#error "THREAD_STACK_SIZE defined but _POSIX_THREAD_ATTR_STACKSIZE undefined"
#endif
#endif

/* The POSIX spec says that implementations supporting the sem_*
   family of functions must indicate this by defining
   _POSIX_SEMAPHORES. */
#ifdef _POSIX_SEMAPHORES
/* On FreeBSD 4.x, _POSIX_SEMAPHORES is defined empty, so
   we need to add 0 to make it work there as well. */
#if (_POSIX_SEMAPHORES+0) == -1
#define HAVE_BROKEN_POSIX_SEMAPHORES
#else
#include <semaphore.h>
#include <errno.h>
#endif
#endif

/* Before FreeBSD 5.4, system scope threads was very limited resource
   in default setting.  So the process scope is preferred to get
   enough number of threads to work. */
#ifdef __FreeBSD__
#include <osreldate.h>
#if __FreeBSD_version >= 500000 && __FreeBSD_version < 504101
#undef PTHREAD_SYSTEM_SCHED_SUPPORTED
#endif
#endif

#if !defined(pthread_attr_default)
#  define pthread_attr_default ((pthread_attr_t *)NULL)
#endif
#if !defined(pthread_mutexattr_default)
#  define pthread_mutexattr_default ((pthread_mutexattr_t *)NULL)
#endif
#if !defined(pthread_condattr_default)
#  define pthread_condattr_default ((pthread_condattr_t *)NULL)
#endif


/* Whether or not to use semaphores directly rather than emulating them with
 * mutexes and condition variables:
 */
#if defined(_POSIX_SEMAPHORES) && !defined(HAVE_BROKEN_POSIX_SEMAPHORES)
#  define USE_SEMAPHORES
#else
#  undef USE_SEMAPHORES
#endif


/* On platforms that don't use standard POSIX threads pthread_sigmask()
 * isn't present.  DEC threads uses sigprocmask() instead as do most
 * other UNIX International compliant systems that don't have the full
 * pthread implementation.
 */
#if defined(HAVE_PTHREAD_SIGMASK) && !defined(HAVE_BROKEN_PTHREAD_SIGMASK)
#  define SET_THREAD_SIGMASK pthread_sigmask
#else
#  define SET_THREAD_SIGMASK sigprocmask
#endif


/* A pthread mutex isn't sufficient to model the Python lock type
 * because, according to Draft 5 of the docs (P1003.4a/D5), both of the
 * following are undefined:
 *  -> a thread tries to lock a mutex it already has locked
 *  -> a thread tries to unlock a mutex locked by a different thread
 * pthread mutexes are designed for serializing threads over short pieces
 * of code anyway, so wouldn't be an appropriate implementation of
 * Python's locks regardless.
 *
 * The pthread_lock struct implements a Python lock as a "locked?" bit
 * and a <condition, mutex> pair.  In general, if the bit can be acquired
 * instantly, it is, else the pair is used to block the thread until the
 * bit is cleared.     9 May 1994 tim@ksr.com
 */

typedef struct {
    char             locked; /* 0=unlocked, 1=locked */
    /* a <cond, mutex> pair to handle an acquire of a locked lock */
    pthread_cond_t   lock_released;
    pthread_mutex_t  mut;
} pthread_lock;

#define CHECK_STATUS(name)  if (status != 0) { perror(name); error = 1; }

/*
 * Initialization.
 */

#ifdef _HAVE_BSDI
static
void _noop(void)
{
}

static void
PyThread__init_thread(void)
{
    /* DO AN INIT BY STARTING THE THREAD */
    static int dummy = 0;
    pthread_t thread1;
    pthread_create(&thread1, NULL, (void *) _noop, &dummy);
    pthread_join(thread1, NULL);
}

#else /* !_HAVE_BSDI */

static void
PyThread__init_thread(void)
{
#if defined(_AIX) && defined(__GNUC__)
    extern void pthread_init(void);
    pthread_init();
#endif
}

#endif /* !_HAVE_BSDI */

/*
 * Thread support.
 */

/* bpo-33015: pythread_callback struct and pythread_wrapper() cast
   "void func(void *)" to "void* func(void *)": always return NULL.

   PyThread_start_new_thread() uses "void func(void *)" type, whereas
   pthread_create() requires a void* return value. */
typedef struct {
    void (*func) (void *);
    void *arg;
} pythread_callback;

static void *
pythread_wrapper(void *arg)
{
    /* copy func and func_arg and free the temporary structure */
    pythread_callback *callback = arg;
    void (*func)(void *) = callback->func;
    void *func_arg = callback->arg;
    free(arg);

    func(func_arg);
    return NULL;
}

long
PyThread_start_new_thread(void (*func)(void *), void *arg)
{
    pthread_t th;
    int status;
#if defined(THREAD_STACK_SIZE) || defined(PTHREAD_SYSTEM_SCHED_SUPPORTED)
    pthread_attr_t attrs;
#endif
#if defined(THREAD_STACK_SIZE)
    size_t      tss;
#endif

    dprintf(("PyThread_start_new_thread called\n"));
    if (!initialized)
        PyThread_init_thread();

#if defined(THREAD_STACK_SIZE) || defined(PTHREAD_SYSTEM_SCHED_SUPPORTED)
    if (pthread_attr_init(&attrs) != 0)
        return -1;
#endif
#if defined(THREAD_STACK_SIZE)
    tss = (_pythread_stacksize != 0) ? _pythread_stacksize
                                     : THREAD_STACK_SIZE;
    if (tss != 0) {
        if (pthread_attr_setstacksize(&attrs, tss) != 0) {
            pthread_attr_destroy(&attrs);
            return -1;
        }
    }
#endif
#if defined(PTHREAD_SYSTEM_SCHED_SUPPORTED)
    pthread_attr_setscope(&attrs, PTHREAD_SCOPE_SYSTEM);
#endif

    pythread_callback *callback = malloc(sizeof(pythread_callback));

    if (callback == NULL) {
        return -1;
    }

    callback->func = func;
    callback->arg = arg;

    status = pthread_create(&th,
#if defined(THREAD_STACK_SIZE) || defined(PTHREAD_SYSTEM_SCHED_SUPPORTED)
                             &attrs,
#else
                             (pthread_attr_t*)NULL,
#endif
                             pythread_wrapper, callback);

#if defined(THREAD_STACK_SIZE) || defined(PTHREAD_SYSTEM_SCHED_SUPPORTED)
    pthread_attr_destroy(&attrs);
#endif

    if (status != 0) {
        free(callback);
        return -1;
    }

    pthread_detach(th);

#if SIZEOF_PTHREAD_T <= SIZEOF_LONG
    return (long) th;
#else
    return (long) *(long *) &th;
#endif
}

/* XXX This implementation is considered (to quote Tim Peters) "inherently
   hosed" because:
     - It does not guarantee the promise that a non-zero integer is returned.
     - The cast to long is inherently unsafe.
     - It is not clear that the 'volatile' (for AIX?) and ugly casting in the
       latter return statement (for Alpha OSF/1) are any longer necessary.
*/
long
PyThread_get_thread_ident(void)
{
    volatile pthread_t threadid;
    if (!initialized)
        PyThread_init_thread();
    /* Jump through some hoops for Alpha OSF/1 */
    threadid = pthread_self();
#if SIZEOF_PTHREAD_T <= SIZEOF_LONG
    return (long) threadid;
#else
    return (long) *(long *) &threadid;
#endif
}

void
PyThread_exit_thread(void)
{
    dprintf(("PyThread_exit_thread called\n"));
    if (!initialized) {
        exit(0);
    }
}

#ifdef USE_SEMAPHORES

/*
 * Lock support.
 */

PyThread_type_lock
PyThread_allocate_lock(void)
{
    sem_t *lock;
    int status, error = 0;

    dprintf(("PyThread_allocate_lock called\n"));
    if (!initialized)
        PyThread_init_thread();

    lock = (sem_t *)malloc(sizeof(sem_t));

    if (lock) {
        status = sem_init(lock,0,1);
        CHECK_STATUS("sem_init");

        if (error) {
            free((void *)lock);
            lock = NULL;
        }
    }

    dprintf(("PyThread_allocate_lock() -> %p\n", lock));
    return (PyThread_type_lock)lock;
}

void
PyThread_free_lock(PyThread_type_lock lock)
{
    sem_t *thelock = (sem_t *)lock;
    int status, error = 0;

    (void) error; /* silence unused-but-set-variable warning */
    dprintf(("PyThread_free_lock(%p) called\n", lock));

    if (!thelock)
        return;

    status = sem_destroy(thelock);
    CHECK_STATUS("sem_destroy");

    free((void *)thelock);
}

/*
 * As of February 2002, Cygwin thread implementations mistakenly report error
 * codes in the return value of the sem_ calls (like the pthread_ functions).
 * Correct implementations return -1 and put the code in errno. This supports
 * either.
 */
static int
fix_status(int status)
{
    return (status == -1) ? errno : status;
}

int
PyThread_acquire_lock(PyThread_type_lock lock, int waitflag)
{
    int success;
    sem_t *thelock = (sem_t *)lock;
    int status, error = 0;

    (void) error; /* silence unused-but-set-variable warning */
    dprintf(("PyThread_acquire_lock(%p, %d) called\n", lock, waitflag));

    do {
        if (waitflag)
            status = fix_status(sem_wait(thelock));
        else
            status = fix_status(sem_trywait(thelock));
    } while (status == EINTR); /* Retry if interrupted by a signal */

    if (waitflag) {
        CHECK_STATUS("sem_wait");
    } else if (status != EAGAIN) {
        CHECK_STATUS("sem_trywait");
    }

    success = (status == 0) ? 1 : 0;

    dprintf(("PyThread_acquire_lock(%p, %d) -> %d\n", lock, waitflag, success));
    return success;
}

void
PyThread_release_lock(PyThread_type_lock lock)
{
    sem_t *thelock = (sem_t *)lock;
    int status, error = 0;

    (void) error; /* silence unused-but-set-variable warning */
    dprintf(("PyThread_release_lock(%p) called\n", lock));

    status = sem_post(thelock);
    CHECK_STATUS("sem_post");
}

#else /* USE_SEMAPHORES */

/*
 * Lock support.
 */
PyThread_type_lock
PyThread_allocate_lock(void)
{
    pthread_lock *lock;
    int status, error = 0;

    dprintf(("PyThread_allocate_lock called\n"));
    if (!initialized)
        PyThread_init_thread();

    lock = (pthread_lock *) malloc(sizeof(pthread_lock));
    if (lock) {
        memset((void *)lock, '\0', sizeof(pthread_lock));
        lock->locked = 0;

        status = pthread_mutex_init(&lock->mut,
                                    pthread_mutexattr_default);
        CHECK_STATUS("pthread_mutex_init");

        status = pthread_cond_init(&lock->lock_released,
                                   pthread_condattr_default);
        CHECK_STATUS("pthread_cond_init");

        if (error) {
            free((void *)lock);
            lock = 0;
        }
    }

    dprintf(("PyThread_allocate_lock() -> %p\n", lock));
    return (PyThread_type_lock) lock;
}

void
PyThread_free_lock(PyThread_type_lock lock)
{
    pthread_lock *thelock = (pthread_lock *)lock;
    int status, error = 0;

    (void) error; /* silence unused-but-set-variable warning */
    dprintf(("PyThread_free_lock(%p) called\n", lock));

    status = pthread_mutex_destroy( &thelock->mut );
    CHECK_STATUS("pthread_mutex_destroy");

    status = pthread_cond_destroy( &thelock->lock_released );
    CHECK_STATUS("pthread_cond_destroy");

    free((void *)thelock);
}

int
PyThread_acquire_lock(PyThread_type_lock lock, int waitflag)
{
    int success = 0;
    pthread_lock *thelock = (pthread_lock *)lock;
    int status, error = 0;

    dprintf(("PyThread_acquire_lock(%p, %d) called\n", lock, waitflag));

    if (waitflag) {
        status = pthread_mutex_lock( &thelock->mut );
        CHECK_STATUS("pthread_mutex_lock[1]");
    }
    else {
        status = pthread_mutex_trylock( &thelock->mut );
        if (status != EBUSY)
            CHECK_STATUS("pthread_mutex_trylock[1]");
    }
    if (status == 0) {
        success = thelock->locked == 0;

        if ( !success && waitflag ) {
            /* continue trying until we get the lock */

            /* mut must be locked by me -- part of the condition
             * protocol */
            while ( thelock->locked ) {
                status = pthread_cond_wait(&thelock->lock_released,
                                           &thelock->mut);
                CHECK_STATUS("pthread_cond_wait");
            }
            success = 1;
        }

        if (success) thelock->locked = 1;
        status = pthread_mutex_unlock( &thelock->mut );
        CHECK_STATUS("pthread_mutex_unlock[1]");
    }

    if (error) success = 0;
    dprintf(("PyThread_acquire_lock(%p, %d) -> %d\n", lock, waitflag, success));
    return success;
}

void
PyThread_release_lock(PyThread_type_lock lock)
{
    pthread_lock *thelock = (pthread_lock *)lock;
    int status, error = 0;

    (void) error; /* silence unused-but-set-variable warning */
    dprintf(("PyThread_release_lock(%p) called\n", lock));

    status = pthread_mutex_lock( &thelock->mut );
    CHECK_STATUS("pthread_mutex_lock[3]");

    thelock->locked = 0;

    status = pthread_mutex_unlock( &thelock->mut );
    CHECK_STATUS("pthread_mutex_unlock[3]");

    /* wake up someone (anyone, if any) waiting on the lock */
    status = pthread_cond_signal( &thelock->lock_released );
    CHECK_STATUS("pthread_cond_signal");
}

#endif /* USE_SEMAPHORES */

/* set the thread stack size.
 * Return 0 if size is valid, -1 if size is invalid,
 * -2 if setting stack size is not supported.
 */
static int
_pythread_pthread_set_stacksize(size_t size)
{
#if defined(THREAD_STACK_SIZE)
    pthread_attr_t attrs;
    size_t tss_min;
    int rc = 0;
#endif

    /* set to default */
    if (size == 0) {
        _pythread_stacksize = 0;
        return 0;
    }

#if defined(THREAD_STACK_SIZE)
#if defined(PTHREAD_STACK_MIN)
    tss_min = PTHREAD_STACK_MIN > THREAD_STACK_MIN ? PTHREAD_STACK_MIN
                                                   : THREAD_STACK_MIN;
#else
    tss_min = THREAD_STACK_MIN;
#endif
    if (size >= tss_min) {
        /* validate stack size by setting thread attribute */
        if (pthread_attr_init(&attrs) == 0) {
            rc = pthread_attr_setstacksize(&attrs, size);
            pthread_attr_destroy(&attrs);
            if (rc == 0) {
                _pythread_stacksize = size;
                return 0;
            }
        }
    }
    return -1;
#else
    return -2;
#endif
}

#define THREAD_SET_STACKSIZE(x) _pythread_pthread_set_stacksize(x)
