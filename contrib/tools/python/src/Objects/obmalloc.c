#include "Python.h"

#ifdef _asan_enabled_
  #define ATTRIBUTE_NO_ADDRESS_SAFETY_ANALYSIS \
        __attribute__((no_address_safety_analysis)) \
        __attribute__ ((noinline))
  #undef WITH_PYMALLOC
#else
  #define ATTRIBUTE_NO_ADDRESS_SAFETY_ANALYSIS
#endif

#ifdef _tsan_enabled_
  #define ATTRIBUTE_NO_THREAD_ANALYSIS \
        __attribute__((no_sanitize_thread))
#else
  #define ATTRIBUTE_NO_THREAD_ANALYSIS
#endif

#ifdef _msan_enabled_
  #define ATTRIBUTE_NO_MEMORY_ANALYSIS \
        __attribute__((no_sanitize_memory))
#else
  #define ATTRIBUTE_NO_MEMORY_ANALYSIS
#endif

#define ATTRIBUTE_SUPRESS_SANITZER \
    ATTRIBUTE_NO_ADDRESS_SAFETY_ANALYSIS \
    ATTRIBUTE_NO_THREAD_ANALYSIS \
    ATTRIBUTE_NO_MEMORY_ANALYSIS

/* Python's malloc wrappers (see pymem.h) */

#ifdef PYMALLOC_DEBUG   /* WITH_PYMALLOC && PYMALLOC_DEBUG */
/* Forward declaration */
static void* _PyMem_DebugMallocCtx(void *ctx, size_t size);
static void _PyMem_DebugFreeCtx(void *ctx, void *p);
static void* _PyMem_DebugReallocCtx(void *ctx, void *ptr, size_t size);

static void _PyMem_DebugCheckAddress(char api_id, const void *p);
#endif

#ifdef WITH_PYMALLOC

#ifdef MS_WINDOWS
#  include <windows.h>
#elif defined(HAVE_MMAP)
#  include <sys/mman.h>
#  ifdef MAP_ANONYMOUS
#    define ARENAS_USE_MMAP
#  endif
#endif

/* Forward declaration */
static void* _PyObject_Malloc(void *ctx, size_t size);
static void _PyObject_Free(void *ctx, void *p);
static void* _PyObject_Realloc(void *ctx, void *ptr, size_t size);
#endif


static void *
_PyMem_RawMalloc(void *ctx, size_t size)
{
    /* PyMem_Malloc(0) means malloc(1). Some systems would return NULL
       for malloc(0), which would be treated as an error. Some platforms would
       return a pointer with no memory behind it, which would break pymalloc.
       To solve these problems, allocate an extra byte. */
    if (size == 0)
        size = 1;
    return malloc(size);
}

static void *
_PyMem_RawRealloc(void *ctx, void *ptr, size_t size)
{
    if (size == 0)
        size = 1;
    return realloc(ptr, size);
}

static void
_PyMem_RawFree(void *ctx, void *ptr)
{
    free(ptr);
}

#define PYRAW_FUNCS _PyMem_RawMalloc, _PyMem_RawRealloc, _PyMem_RawFree
#ifdef WITH_PYMALLOC
#define PYOBJECT_FUNCS _PyObject_Malloc, _PyObject_Realloc, _PyObject_Free
#else
#define PYOBJECT_FUNCS PYRAW_FUNCS
#endif

#ifdef PYMALLOC_DEBUG
typedef struct {
    /* We tag each block with an API ID in order to tag API violations */
    char api_id;
    PyMemAllocator alloc;
} debug_alloc_api_t;
static struct {
    debug_alloc_api_t raw;
    debug_alloc_api_t mem;
    debug_alloc_api_t obj;
} _PyMem_Debug = {
    {'r', {NULL, PYRAW_FUNCS}},
    {'m', {NULL, PYRAW_FUNCS}},
    {'o', {NULL, PYOBJECT_FUNCS}}
    };

#define PYDEBUG_FUNCS _PyMem_DebugMallocCtx, _PyMem_DebugReallocCtx, _PyMem_DebugFreeCtx
#endif

static PyMemAllocator _PyMem_Raw = {
#ifdef PYMALLOC_DEBUG
    &_PyMem_Debug.raw, PYDEBUG_FUNCS
#else
    NULL, PYRAW_FUNCS
#endif
    };

static PyMemAllocator _PyMem = {
#ifdef PYMALLOC_DEBUG
    &_PyMem_Debug.mem, PYDEBUG_FUNCS
#else
    NULL, PYRAW_FUNCS
#endif
    };

static PyMemAllocator _PyObject = {
#ifdef PYMALLOC_DEBUG
    &_PyMem_Debug.obj, PYDEBUG_FUNCS
#else
    NULL, PYOBJECT_FUNCS
#endif
    };

#undef PYRAW_FUNCS
#undef PYOBJECT_FUNCS
#undef PYDEBUG_FUNCS

void
PyMem_SetupDebugHooks(void)
{
#ifdef PYMALLOC_DEBUG
    PyMemAllocator alloc;

    alloc.malloc = _PyMem_DebugMallocCtx;
    alloc.realloc = _PyMem_DebugReallocCtx;
    alloc.free = _PyMem_DebugFreeCtx;

    if (_PyMem_Raw.malloc != _PyMem_DebugMallocCtx) {
        alloc.ctx = &_PyMem_Debug.raw;
        PyMem_GetAllocator(PYMEM_DOMAIN_RAW, &_PyMem_Debug.raw.alloc);
        PyMem_SetAllocator(PYMEM_DOMAIN_RAW, &alloc);
    }

    if (_PyMem.malloc != _PyMem_DebugMallocCtx) {
        alloc.ctx = &_PyMem_Debug.mem;
        PyMem_GetAllocator(PYMEM_DOMAIN_MEM, &_PyMem_Debug.mem.alloc);
        PyMem_SetAllocator(PYMEM_DOMAIN_MEM, &alloc);
    }

    if (_PyObject.malloc != _PyMem_DebugMallocCtx) {
        alloc.ctx = &_PyMem_Debug.obj;
        PyMem_GetAllocator(PYMEM_DOMAIN_OBJ, &_PyMem_Debug.obj.alloc);
        PyMem_SetAllocator(PYMEM_DOMAIN_OBJ, &alloc);
    }
#endif
}

void
PyMem_GetAllocator(PyMemAllocatorDomain domain, PyMemAllocator *allocator)
{
    switch(domain)
    {
    case PYMEM_DOMAIN_RAW: *allocator = _PyMem_Raw; break;
    case PYMEM_DOMAIN_MEM: *allocator = _PyMem; break;
    case PYMEM_DOMAIN_OBJ: *allocator = _PyObject; break;
    default:
        /* unknown domain */
        allocator->ctx = NULL;
        allocator->malloc = NULL;
        allocator->realloc = NULL;
        allocator->free = NULL;
    }
}

void
PyMem_SetAllocator(PyMemAllocatorDomain domain, PyMemAllocator *allocator)
{
    switch(domain)
    {
    case PYMEM_DOMAIN_RAW: _PyMem_Raw = *allocator; break;
    case PYMEM_DOMAIN_MEM: _PyMem = *allocator; break;
    case PYMEM_DOMAIN_OBJ: _PyObject = *allocator; break;
    /* ignore unknown domain */
    }

}

void *
PyMem_RawMalloc(size_t size)
{
    /*
     * Limit ourselves to PY_SSIZE_T_MAX bytes to prevent security holes.
     * Most python internals blindly use a signed Py_ssize_t to track
     * things without checking for overflows or negatives.
     * As size_t is unsigned, checking for size < 0 is not required.
     */
    if (size > (size_t)PY_SSIZE_T_MAX)
        return NULL;

    return _PyMem_Raw.malloc(_PyMem_Raw.ctx, size);
}

void*
PyMem_RawRealloc(void *ptr, size_t new_size)
{
    /* see PyMem_RawMalloc() */
    if (new_size > (size_t)PY_SSIZE_T_MAX)
        return NULL;
    return _PyMem_Raw.realloc(_PyMem_Raw.ctx, ptr, new_size);
}

void PyMem_RawFree(void *ptr)
{
    _PyMem_Raw.free(_PyMem_Raw.ctx, ptr);
}

void *
PyMem_Malloc(size_t size)
{
    /* see PyMem_RawMalloc() */
    if (size > (size_t)PY_SSIZE_T_MAX)
        return NULL;
    return _PyMem.malloc(_PyMem.ctx, size);
}

void *
PyMem_Realloc(void *ptr, size_t new_size)
{
    /* see PyMem_RawMalloc() */
    if (new_size > (size_t)PY_SSIZE_T_MAX)
        return NULL;
    return _PyMem.realloc(_PyMem.ctx, ptr, new_size);
}

void
PyMem_Free(void *ptr)
{
    _PyMem.free(_PyMem.ctx, ptr);
}

char *
_PyMem_RawStrdup(const char *str)
{
    size_t size;
    char *copy;

    size = strlen(str) + 1;
    copy = PyMem_RawMalloc(size);
    if (copy == NULL)
        return NULL;
    memcpy(copy, str, size);
    return copy;
}

char *
_PyMem_Strdup(const char *str)
{
    size_t size;
    char *copy;

    size = strlen(str) + 1;
    copy = PyMem_Malloc(size);
    if (copy == NULL)
        return NULL;
    memcpy(copy, str, size);
    return copy;
}

void *
PyObject_Malloc(size_t size)
{
    /* see PyMem_RawMalloc() */
    if (size > (size_t)PY_SSIZE_T_MAX)
        return NULL;
    return _PyObject.malloc(_PyObject.ctx, size);
}

void *
PyObject_Realloc(void *ptr, size_t new_size)
{
    /* see PyMem_RawMalloc() */
    if (new_size > (size_t)PY_SSIZE_T_MAX)
        return NULL;
    return _PyObject.realloc(_PyObject.ctx, ptr, new_size);
}

void
PyObject_Free(void *ptr)
{
    _PyObject.free(_PyObject.ctx, ptr);
}


#ifdef WITH_PYMALLOC

#ifdef HAVE_MMAP
 #include <sys/mman.h>
 #ifdef MAP_ANONYMOUS
  #define ARENAS_USE_MMAP
 #endif
#endif

#ifdef WITH_VALGRIND
#include <valgrind/valgrind.h>

/* If we're using GCC, use __builtin_expect() to reduce overhead of
   the valgrind checks */
#if defined(__GNUC__) && (__GNUC__ > 2) && defined(__OPTIMIZE__)
#  define UNLIKELY(value) __builtin_expect((value), 0)
#else
#  define UNLIKELY(value) (value)
#endif

/* -1 indicates that we haven't checked that we're running on valgrind yet. */
static int running_on_valgrind = -1;
#endif

/* An object allocator for Python.

   Here is an introduction to the layers of the Python memory architecture,
   showing where the object allocator is actually used (layer +2), It is
   called for every object allocation and deallocation (PyObject_New/Del),
   unless the object-specific allocators implement a proprietary allocation
   scheme (ex.: ints use a simple free list). This is also the place where
   the cyclic garbage collector operates selectively on container objects.


    Object-specific allocators
    _____   ______   ______       ________
   [ int ] [ dict ] [ list ] ... [ string ]       Python core         |
+3 | <----- Object-specific memory -----> | <-- Non-object memory --> |
    _______________________________       |                           |
   [   Python's object allocator   ]      |                           |
+2 | ####### Object memory ####### | <------ Internal buffers ------> |
    ______________________________________________________________    |
   [          Python's raw memory allocator (PyMem_ API)          ]   |
+1 | <----- Python memory (under PyMem manager's control) ------> |   |
    __________________________________________________________________
   [    Underlying general-purpose allocator (ex: C library malloc)   ]
 0 | <------ Virtual memory allocated for the python process -------> |

   =========================================================================
    _______________________________________________________________________
   [                OS-specific Virtual Memory Manager (VMM)               ]
-1 | <--- Kernel dynamic storage allocation & management (page-based) ---> |
    __________________________________   __________________________________
   [                                  ] [                                  ]
-2 | <-- Physical memory: ROM/RAM --> | | <-- Secondary storage (swap) --> |

*/
/*==========================================================================*/

/* A fast, special-purpose memory allocator for small blocks, to be used
   on top of a general-purpose malloc -- heavily based on previous art. */

/* Vladimir Marangozov -- August 2000 */

/*
 * "Memory management is where the rubber meets the road -- if we do the wrong
 * thing at any level, the results will not be good. And if we don't make the
 * levels work well together, we are in serious trouble." (1)
 *
 * (1) Paul R. Wilson, Mark S. Johnstone, Michael Neely, and David Boles,
 *    "Dynamic Storage Allocation: A Survey and Critical Review",
 *    in Proc. 1995 Int'l. Workshop on Memory Management, September 1995.
 */

/* #undef WITH_MEMORY_LIMITS */         /* disable mem limit checks  */

/*==========================================================================*/

/*
 * Allocation strategy abstract:
 *
 * For small requests, the allocator sub-allocates <Big> blocks of memory.
 * Requests greater than SMALL_REQUEST_THRESHOLD bytes are routed to the
 * system's allocator.
 *
 * Small requests are grouped in size classes spaced 8 bytes apart, due
 * to the required valid alignment of the returned address. Requests of
 * a particular size are serviced from memory pools of 4K (one VMM page).
 * Pools are fragmented on demand and contain free lists of blocks of one
 * particular size class. In other words, there is a fixed-size allocator
 * for each size class. Free pools are shared by the different allocators
 * thus minimizing the space reserved for a particular size class.
 *
 * This allocation strategy is a variant of what is known as "simple
 * segregated storage based on array of free lists". The main drawback of
 * simple segregated storage is that we might end up with lot of reserved
 * memory for the different free lists, which degenerate in time. To avoid
 * this, we partition each free list in pools and we share dynamically the
 * reserved space between all free lists. This technique is quite efficient
 * for memory intensive programs which allocate mainly small-sized blocks.
 *
 * For small requests we have the following table:
 *
 * Request in bytes     Size of allocated block      Size class idx
 * ----------------------------------------------------------------
 *        1-8                     8                       0
 *        9-16                   16                       1
 *       17-24                   24                       2
 *       25-32                   32                       3
 *       33-40                   40                       4
 *       41-48                   48                       5
 *       49-56                   56                       6
 *       57-64                   64                       7
 *       65-72                   72                       8
 *        ...                   ...                     ...
 *      497-504                 504                      62
 *      505-512                 512                      63
 *
 *      0, SMALL_REQUEST_THRESHOLD + 1 and up: routed to the underlying
 *      allocator.
 */

/*==========================================================================*/

/*
 * -- Main tunable settings section --
 */

/*
 * Alignment of addresses returned to the user. 8-bytes alignment works
 * on most current architectures (with 32-bit or 64-bit address busses).
 * The alignment value is also used for grouping small requests in size
 * classes spaced ALIGNMENT bytes apart.
 *
 * You shouldn't change this unless you know what you are doing.
 */
#if SIZEOF_VOID_P > 4
#define ALIGNMENT              16               /* must be 2^N */
#define ALIGNMENT_SHIFT         4
#else
#define ALIGNMENT               8               /* must be 2^N */
#define ALIGNMENT_SHIFT         3
#endif
#define ALIGNMENT_MASK          (ALIGNMENT - 1)

/* Return the number of bytes in size class I, as a uint. */
#define INDEX2SIZE(I) (((uint)(I) + 1) << ALIGNMENT_SHIFT)

/*
 * Max size threshold below which malloc requests are considered to be
 * small enough in order to use preallocated memory pools. You can tune
 * this value according to your application behaviour and memory needs.
 *
 * The following invariants must hold:
 *      1) ALIGNMENT <= SMALL_REQUEST_THRESHOLD <= 512
 *      2) SMALL_REQUEST_THRESHOLD is evenly divisible by ALIGNMENT
 *
 * Note: a size threshold of 512 guarantees that newly created dictionaries
 * will be allocated from preallocated memory pools on 64-bit.
 *
 * Although not required, for better performance and space efficiency,
 * it is recommended that SMALL_REQUEST_THRESHOLD is set to a power of 2.
 */
#define SMALL_REQUEST_THRESHOLD 512
#define NB_SMALL_SIZE_CLASSES   (SMALL_REQUEST_THRESHOLD / ALIGNMENT)

/*
 * The system's VMM page size can be obtained on most unices with a
 * getpagesize() call or deduced from various header files. To make
 * things simpler, we assume that it is 4K, which is OK for most systems.
 * It is probably better if this is the native page size, but it doesn't
 * have to be.  In theory, if SYSTEM_PAGE_SIZE is larger than the native page
 * size, then `POOL_ADDR(p)->arenaindex' could rarely cause a segmentation
 * violation fault.  4K is apparently OK for all the platforms that python
 * currently targets.
 */
#define SYSTEM_PAGE_SIZE        (4 * 1024)
#define SYSTEM_PAGE_SIZE_MASK   (SYSTEM_PAGE_SIZE - 1)

/*
 * Maximum amount of memory managed by the allocator for small requests.
 */
#ifdef WITH_MEMORY_LIMITS
#ifndef SMALL_MEMORY_LIMIT
#define SMALL_MEMORY_LIMIT      (64 * 1024 * 1024)      /* 64 MB -- more? */
#endif
#endif

/*
 * The allocator sub-allocates <Big> blocks of memory (called arenas) aligned
 * on a page boundary. This is a reserved virtual address space for the
 * current process (obtained through a malloc()/mmap() call). In no way this
 * means that the memory arenas will be used entirely. A malloc(<Big>) is
 * usually an address range reservation for <Big> bytes, unless all pages within
 * this space are referenced subsequently. So malloc'ing big blocks and not
 * using them does not mean "wasting memory". It's an addressable range
 * wastage...
 *
 * Arenas are allocated with mmap() on systems supporting anonymous memory
 * mappings to reduce heap fragmentation.
 */
#define ARENA_SIZE              (256 << 10)     /* 256KiB */

#ifdef WITH_MEMORY_LIMITS
#define MAX_ARENAS              (SMALL_MEMORY_LIMIT / ARENA_SIZE)
#endif

/*
 * Size of the pools used for small blocks. Should be a power of 2,
 * between 1K and SYSTEM_PAGE_SIZE, that is: 1k, 2k, 4k.
 */
#define POOL_SIZE               SYSTEM_PAGE_SIZE        /* must be 2^N */
#define POOL_SIZE_MASK          SYSTEM_PAGE_SIZE_MASK

/*
 * -- End of tunable settings section --
 */

/*==========================================================================*/

/*
 * Locking
 *
 * To reduce lock contention, it would probably be better to refine the
 * crude function locking with per size class locking. I'm not positive
 * however, whether it's worth switching to such locking policy because
 * of the performance penalty it might introduce.
 *
 * The following macros describe the simplest (should also be the fastest)
 * lock object on a particular platform and the init/fini/lock/unlock
 * operations on it. The locks defined here are not expected to be recursive
 * because it is assumed that they will always be called in the order:
 * INIT, [LOCK, UNLOCK]*, FINI.
 */

/*
 * Python's threads are serialized, so object malloc locking is disabled.
 */
#define SIMPLELOCK_DECL(lock)   /* simple lock declaration              */
#define SIMPLELOCK_INIT(lock)   /* allocate (if needed) and initialize  */
#define SIMPLELOCK_FINI(lock)   /* free/destroy an existing lock        */
#define SIMPLELOCK_LOCK(lock)   /* acquire released lock */
#define SIMPLELOCK_UNLOCK(lock) /* release acquired lock */

/*
 * Basic types
 * I don't care if these are defined in <sys/types.h> or elsewhere. Axiom.
 */
#undef  uchar
#define uchar   unsigned char   /* assuming == 8 bits  */

#undef  uint
#define uint    unsigned int    /* assuming >= 16 bits */

#undef  ulong
#define ulong   unsigned long   /* assuming >= 32 bits */

#undef uptr
#define uptr    Py_uintptr_t

/* When you say memory, my mind reasons in terms of (pointers to) blocks */
typedef uchar block;

/* Pool for small blocks. */
struct pool_header {
    union { block *_padding;
            uint count; } ref;          /* number of allocated blocks    */
    block *freeblock;                   /* pool's free list head         */
    struct pool_header *nextpool;       /* next pool of this size class  */
    struct pool_header *prevpool;       /* previous pool       ""        */
    uint arenaindex;                    /* index into arenas of base adr */
    uint szidx;                         /* block size class index        */
    uint nextoffset;                    /* bytes to virgin block         */
    uint maxnextoffset;                 /* largest valid nextoffset      */
};

typedef struct pool_header *poolp;

/* Record keeping for arenas. */
struct arena_object {
    /* The address of the arena, as returned by malloc.  Note that 0
     * will never be returned by a successful malloc, and is used
     * here to mark an arena_object that doesn't correspond to an
     * allocated arena.
     */
    uptr address;

    /* Pool-aligned pointer to the next pool to be carved off. */
    block* pool_address;

    /* The number of available pools in the arena:  free pools + never-
     * allocated pools.
     */
    uint nfreepools;

    /* The total number of pools in the arena, whether or not available. */
    uint ntotalpools;

    /* Singly-linked list of available pools. */
    struct pool_header* freepools;

    /* Whenever this arena_object is not associated with an allocated
     * arena, the nextarena member is used to link all unassociated
     * arena_objects in the singly-linked `unused_arena_objects` list.
     * The prevarena member is unused in this case.
     *
     * When this arena_object is associated with an allocated arena
     * with at least one available pool, both members are used in the
     * doubly-linked `usable_arenas` list, which is maintained in
     * increasing order of `nfreepools` values.
     *
     * Else this arena_object is associated with an allocated arena
     * all of whose pools are in use.  `nextarena` and `prevarena`
     * are both meaningless in this case.
     */
    struct arena_object* nextarena;
    struct arena_object* prevarena;
};

#undef  ROUNDUP
#define ROUNDUP(x)              (((x) + ALIGNMENT_MASK) & ~ALIGNMENT_MASK)
#define POOL_OVERHEAD           ROUNDUP(sizeof(struct pool_header))

#define DUMMY_SIZE_IDX          0xffff  /* size class of newly cached pools */

/* Round pointer P down to the closest pool-aligned address <= P, as a poolp */
#define POOL_ADDR(P) ((poolp)((uptr)(P) & ~(uptr)POOL_SIZE_MASK))

/* Return total number of blocks in pool of size index I, as a uint. */
#define NUMBLOCKS(I) ((uint)(POOL_SIZE - POOL_OVERHEAD) / INDEX2SIZE(I))

/*==========================================================================*/

/*
 * This malloc lock
 */
SIMPLELOCK_DECL(_malloc_lock)
#define LOCK()          SIMPLELOCK_LOCK(_malloc_lock)
#define UNLOCK()        SIMPLELOCK_UNLOCK(_malloc_lock)
#define LOCK_INIT()     SIMPLELOCK_INIT(_malloc_lock)
#define LOCK_FINI()     SIMPLELOCK_FINI(_malloc_lock)

/*
 * Pool table -- headed, circular, doubly-linked lists of partially used pools.

This is involved.  For an index i, usedpools[i+i] is the header for a list of
all partially used pools holding small blocks with "size class idx" i. So
usedpools[0] corresponds to blocks of size 8, usedpools[2] to blocks of size
16, and so on:  index 2*i <-> blocks of size (i+1)<<ALIGNMENT_SHIFT.

Pools are carved off an arena's highwater mark (an arena_object's pool_address
member) as needed.  Once carved off, a pool is in one of three states forever
after:

used == partially used, neither empty nor full
    At least one block in the pool is currently allocated, and at least one
    block in the pool is not currently allocated (note this implies a pool
    has room for at least two blocks).
    This is a pool's initial state, as a pool is created only when malloc
    needs space.
    The pool holds blocks of a fixed size, and is in the circular list headed
    at usedpools[i] (see above).  It's linked to the other used pools of the
    same size class via the pool_header's nextpool and prevpool members.
    If all but one block is currently allocated, a malloc can cause a
    transition to the full state.  If all but one block is not currently
    allocated, a free can cause a transition to the empty state.

full == all the pool's blocks are currently allocated
    On transition to full, a pool is unlinked from its usedpools[] list.
    It's not linked to from anything then anymore, and its nextpool and
    prevpool members are meaningless until it transitions back to used.
    A free of a block in a full pool puts the pool back in the used state.
    Then it's linked in at the front of the appropriate usedpools[] list, so
    that the next allocation for its size class will reuse the freed block.

empty == all the pool's blocks are currently available for allocation
    On transition to empty, a pool is unlinked from its usedpools[] list,
    and linked to the front of its arena_object's singly-linked freepools list,
    via its nextpool member.  The prevpool member has no meaning in this case.
    Empty pools have no inherent size class:  the next time a malloc finds
    an empty list in usedpools[], it takes the first pool off of freepools.
    If the size class needed happens to be the same as the size class the pool
    last had, some pool initialization can be skipped.


Block Management

Blocks within pools are again carved out as needed.  pool->freeblock points to
the start of a singly-linked list of free blocks within the pool.  When a
block is freed, it's inserted at the front of its pool's freeblock list.  Note
that the available blocks in a pool are *not* linked all together when a pool
is initialized.  Instead only "the first two" (lowest addresses) blocks are
set up, returning the first such block, and setting pool->freeblock to a
one-block list holding the second such block.  This is consistent with that
pymalloc strives at all levels (arena, pool, and block) never to touch a piece
of memory until it's actually needed.

So long as a pool is in the used state, we're certain there *is* a block
available for allocating, and pool->freeblock is not NULL.  If pool->freeblock
points to the end of the free list before we've carved the entire pool into
blocks, that means we simply haven't yet gotten to one of the higher-address
blocks.  The offset from the pool_header to the start of "the next" virgin
block is stored in the pool_header nextoffset member, and the largest value
of nextoffset that makes sense is stored in the maxnextoffset member when a
pool is initialized.  All the blocks in a pool have been passed out at least
once when and only when nextoffset > maxnextoffset.


Major obscurity:  While the usedpools vector is declared to have poolp
entries, it doesn't really.  It really contains two pointers per (conceptual)
poolp entry, the nextpool and prevpool members of a pool_header.  The
excruciating initialization code below fools C so that

    usedpool[i+i]

"acts like" a genuine poolp, but only so long as you only reference its
nextpool and prevpool members.  The "- 2*sizeof(block *)" gibberish is
compensating for that a pool_header's nextpool and prevpool members
immediately follow a pool_header's first two members:

    union { block *_padding;
            uint count; } ref;
    block *freeblock;

each of which consume sizeof(block *) bytes.  So what usedpools[i+i] really
contains is a fudged-up pointer p such that *if* C believes it's a poolp
pointer, then p->nextpool and p->prevpool are both p (meaning that the headed
circular list is empty).

It's unclear why the usedpools setup is so convoluted.  It could be to
minimize the amount of cache required to hold this heavily-referenced table
(which only *needs* the two interpool pointer members of a pool_header). OTOH,
referencing code has to remember to "double the index" and doing so isn't
free, usedpools[0] isn't a strictly legal pointer, and we're crucially relying
on that C doesn't insert any padding anywhere in a pool_header at or before
the prevpool member.
**************************************************************************** */

#define PTA(x)  ((poolp )((uchar *)&(usedpools[2*(x)]) - 2*sizeof(block *)))
#define PT(x)   PTA(x), PTA(x)

static poolp usedpools[2 * ((NB_SMALL_SIZE_CLASSES + 7) / 8) * 8] = {
    PT(0), PT(1), PT(2), PT(3), PT(4), PT(5), PT(6), PT(7)
#if NB_SMALL_SIZE_CLASSES > 8
    , PT(8), PT(9), PT(10), PT(11), PT(12), PT(13), PT(14), PT(15)
#if NB_SMALL_SIZE_CLASSES > 16
    , PT(16), PT(17), PT(18), PT(19), PT(20), PT(21), PT(22), PT(23)
#if NB_SMALL_SIZE_CLASSES > 24
    , PT(24), PT(25), PT(26), PT(27), PT(28), PT(29), PT(30), PT(31)
#if NB_SMALL_SIZE_CLASSES > 32
    , PT(32), PT(33), PT(34), PT(35), PT(36), PT(37), PT(38), PT(39)
#if NB_SMALL_SIZE_CLASSES > 40
    , PT(40), PT(41), PT(42), PT(43), PT(44), PT(45), PT(46), PT(47)
#if NB_SMALL_SIZE_CLASSES > 48
    , PT(48), PT(49), PT(50), PT(51), PT(52), PT(53), PT(54), PT(55)
#if NB_SMALL_SIZE_CLASSES > 56
    , PT(56), PT(57), PT(58), PT(59), PT(60), PT(61), PT(62), PT(63)
#if NB_SMALL_SIZE_CLASSES > 64
#error "NB_SMALL_SIZE_CLASSES should be less than 64"
#endif /* NB_SMALL_SIZE_CLASSES > 64 */
#endif /* NB_SMALL_SIZE_CLASSES > 56 */
#endif /* NB_SMALL_SIZE_CLASSES > 48 */
#endif /* NB_SMALL_SIZE_CLASSES > 40 */
#endif /* NB_SMALL_SIZE_CLASSES > 32 */
#endif /* NB_SMALL_SIZE_CLASSES > 24 */
#endif /* NB_SMALL_SIZE_CLASSES > 16 */
#endif /* NB_SMALL_SIZE_CLASSES >  8 */
};

/*==========================================================================
Arena management.

`arenas` is a vector of arena_objects.  It contains maxarenas entries, some of
which may not be currently used (== they're arena_objects that aren't
currently associated with an allocated arena).  Note that arenas proper are
separately malloc'ed.

Prior to Python 2.5, arenas were never free()'ed.  Starting with Python 2.5,
we do try to free() arenas, and use some mild heuristic strategies to increase
the likelihood that arenas eventually can be freed.

unused_arena_objects

    This is a singly-linked list of the arena_objects that are currently not
    being used (no arena is associated with them).  Objects are taken off the
    head of the list in new_arena(), and are pushed on the head of the list in
    PyObject_Free() when the arena is empty.  Key invariant:  an arena_object
    is on this list if and only if its .address member is 0.

usable_arenas

    This is a doubly-linked list of the arena_objects associated with arenas
    that have pools available.  These pools are either waiting to be reused,
    or have not been used before.  The list is sorted to have the most-
    allocated arenas first (ascending order based on the nfreepools member).
    This means that the next allocation will come from a heavily used arena,
    which gives the nearly empty arenas a chance to be returned to the system.
    In my unscientific tests this dramatically improved the number of arenas
    that could be freed.

Note that an arena_object associated with an arena all of whose pools are
currently in use isn't on either list.
*/

/* Array of objects used to track chunks of memory (arenas). */
static struct arena_object* arenas = NULL;
/* Number of slots currently allocated in the `arenas` vector. */
static uint maxarenas = 0;

/* The head of the singly-linked, NULL-terminated list of available
 * arena_objects.
 */
static struct arena_object* unused_arena_objects = NULL;

/* The head of the doubly-linked, NULL-terminated at each end, list of
 * arena_objects associated with arenas that have pools available.
 */
static struct arena_object* usable_arenas = NULL;

/* How many arena_objects do we initially allocate?
 * 16 = can allocate 16 arenas = 16 * ARENA_SIZE = 4MB before growing the
 * `arenas` vector.
 */
#define INITIAL_ARENA_OBJECTS 16

/* Number of arenas allocated that haven't been free()'d. */
static size_t narenas_currently_allocated = 0;

#ifdef PYMALLOC_DEBUG
/* Total number of times malloc() called to allocate an arena. */
static size_t ntimes_arena_allocated = 0;
/* High water mark (max value ever seen) for narenas_currently_allocated. */
static size_t narenas_highwater = 0;
#endif

/* Allocate a new arena.  If we run out of memory, return NULL.  Else
 * allocate a new arena, and return the address of an arena_object
 * describing the new arena.  It's expected that the caller will set
 * `usable_arenas` to the return value.
 */
static struct arena_object*
new_arena(void)
{
    struct arena_object* arenaobj;
    uint excess;        /* number of bytes above pool alignment */
    void *address;
    int err;

#ifdef PYMALLOC_DEBUG
    if (Py_GETENV("PYTHONMALLOCSTATS"))
        _PyObject_DebugMallocStats();
#endif
    if (unused_arena_objects == NULL) {
        uint i;
        uint numarenas;
        size_t nbytes;

        /* Double the number of arena objects on each allocation.
         * Note that it's possible for `numarenas` to overflow.
         */
        numarenas = maxarenas ? maxarenas << 1 : INITIAL_ARENA_OBJECTS;
        if (numarenas <= maxarenas)
            return NULL;                /* overflow */
#if SIZEOF_SIZE_T <= SIZEOF_INT
        if (numarenas > PY_SIZE_MAX / sizeof(*arenas))
            return NULL;                /* overflow */
#endif
        nbytes = numarenas * sizeof(*arenas);
        arenaobj = (struct arena_object *)PyMem_Realloc(arenas, nbytes);
        if (arenaobj == NULL)
            return NULL;
        arenas = arenaobj;

        /* We might need to fix pointers that were copied.  However,
         * new_arena only gets called when all the pages in the
         * previous arenas are full.  Thus, there are *no* pointers
         * into the old array. Thus, we don't have to worry about
         * invalid pointers.  Just to be sure, some asserts:
         */
        assert(usable_arenas == NULL);
        assert(unused_arena_objects == NULL);

        /* Put the new arenas on the unused_arena_objects list. */
        for (i = maxarenas; i < numarenas; ++i) {
            arenas[i].address = 0;              /* mark as unassociated */
            arenas[i].nextarena = i < numarenas - 1 ?
                                   &arenas[i+1] : NULL;
        }

        /* Update globals. */
        unused_arena_objects = &arenas[maxarenas];
        maxarenas = numarenas;
    }

    /* Take the next available arena object off the head of the list. */
    assert(unused_arena_objects != NULL);
    arenaobj = unused_arena_objects;
    unused_arena_objects = arenaobj->nextarena;
    assert(arenaobj->address == 0);
#ifdef ARENAS_USE_MMAP
    address = mmap(NULL, ARENA_SIZE, PROT_READ|PROT_WRITE,
                   MAP_PRIVATE|MAP_ANONYMOUS, -1, 0);
    err = (address == MAP_FAILED);
#else
    address = malloc(ARENA_SIZE);
    err = (address == 0);
#endif
    if (err) {
        /* The allocation failed: return NULL after putting the
         * arenaobj back.
         */
        arenaobj->nextarena = unused_arena_objects;
        unused_arena_objects = arenaobj;
        return NULL;
    }
    arenaobj->address = (uptr)address;

    ++narenas_currently_allocated;
#ifdef PYMALLOC_DEBUG
    ++ntimes_arena_allocated;
    if (narenas_currently_allocated > narenas_highwater)
        narenas_highwater = narenas_currently_allocated;
#endif
    arenaobj->freepools = NULL;
    /* pool_address <- first pool-aligned address in the arena
       nfreepools <- number of whole pools that fit after alignment */
    arenaobj->pool_address = (block*)arenaobj->address;
    arenaobj->nfreepools = ARENA_SIZE / POOL_SIZE;
    assert(POOL_SIZE * arenaobj->nfreepools == ARENA_SIZE);
    excess = (uint)(arenaobj->address & POOL_SIZE_MASK);
    if (excess != 0) {
        --arenaobj->nfreepools;
        arenaobj->pool_address += POOL_SIZE - excess;
    }
    arenaobj->ntotalpools = arenaobj->nfreepools;

    return arenaobj;
}

/*
Py_ADDRESS_IN_RANGE(P, POOL)

Return true if and only if P is an address that was allocated by pymalloc.
POOL must be the pool address associated with P, i.e., POOL = POOL_ADDR(P)
(the caller is asked to compute this because the macro expands POOL more than
once, and for efficiency it's best for the caller to assign POOL_ADDR(P) to a
variable and pass the latter to the macro; because Py_ADDRESS_IN_RANGE is
called on every alloc/realloc/free, micro-efficiency is important here).

Tricky:  Let B be the arena base address associated with the pool, B =
arenas[(POOL)->arenaindex].address.  Then P belongs to the arena if and only if

    B <= P < B + ARENA_SIZE

Subtracting B throughout, this is true iff

    0 <= P-B < ARENA_SIZE

By using unsigned arithmetic, the "0 <=" half of the test can be skipped.

Obscure:  A PyMem "free memory" function can call the pymalloc free or realloc
before the first arena has been allocated.  `arenas` is still NULL in that
case.  We're relying on that maxarenas is also 0 in that case, so that
(POOL)->arenaindex < maxarenas  must be false, saving us from trying to index
into a NULL arenas.

Details:  given P and POOL, the arena_object corresponding to P is AO =
arenas[(POOL)->arenaindex].  Suppose obmalloc controls P.  Then (barring wild
stores, etc), POOL is the correct address of P's pool, AO.address is the
correct base address of the pool's arena, and P must be within ARENA_SIZE of
AO.address.  In addition, AO.address is not 0 (no arena can start at address 0
(NULL)).  Therefore Py_ADDRESS_IN_RANGE correctly reports that obmalloc
controls P.

Now suppose obmalloc does not control P (e.g., P was obtained via a direct
call to the system malloc() or realloc()).  (POOL)->arenaindex may be anything
in this case -- it may even be uninitialized trash.  If the trash arenaindex
is >= maxarenas, the macro correctly concludes at once that obmalloc doesn't
control P.

Else arenaindex is < maxarena, and AO is read up.  If AO corresponds to an
allocated arena, obmalloc controls all the memory in slice AO.address :
AO.address+ARENA_SIZE.  By case assumption, P is not controlled by obmalloc,
so P doesn't lie in that slice, so the macro correctly reports that P is not
controlled by obmalloc.

Finally, if P is not controlled by obmalloc and AO corresponds to an unused
arena_object (one not currently associated with an allocated arena),
AO.address is 0, and the second test in the macro reduces to:

    P < ARENA_SIZE

If P >= ARENA_SIZE (extremely likely), the macro again correctly concludes
that P is not controlled by obmalloc.  However, if P < ARENA_SIZE, this part
of the test still passes, and the third clause (AO.address != 0) is necessary
to get the correct result:  AO.address is 0 in this case, so the macro
correctly reports that P is not controlled by obmalloc (despite that P lies in
slice AO.address : AO.address + ARENA_SIZE).

Note:  The third (AO.address != 0) clause was added in Python 2.5.  Before
2.5, arenas were never free()'ed, and an arenaindex < maxarena always
corresponded to a currently-allocated arena, so the "P is not controlled by
obmalloc, AO corresponds to an unused arena_object, and P < ARENA_SIZE" case
was impossible.

Note that the logic is excruciating, and reading up possibly uninitialized
memory when P is not controlled by obmalloc (to get at (POOL)->arenaindex)
creates problems for some memory debuggers.  The overwhelming advantage is
that this test determines whether an arbitrary address is controlled by
obmalloc in a small constant time, independent of the number of arenas
obmalloc controls.  Since this test is needed at every entry point, it's
extremely desirable that it be this fast.

Since Py_ADDRESS_IN_RANGE may be reading from memory which was not allocated
by Python, it is important that (POOL)->arenaindex is read only once, as
another thread may be concurrently modifying the value without holding the
GIL.  To accomplish this, the arenaindex_temp variable is used to store
(POOL)->arenaindex for the duration of the Py_ADDRESS_IN_RANGE macro's
execution.  The caller of the macro is responsible for declaring this
variable.
*/
#define Py_ADDRESS_IN_RANGE(P, POOL)                    \
    ((arenaindex_temp = (POOL)->arenaindex) < maxarenas &&              \
     (uptr)(P) - arenas[arenaindex_temp].address < (uptr)ARENA_SIZE && \
     arenas[arenaindex_temp].address != 0)


/* This is only useful when running memory debuggers such as
 * Purify or Valgrind.  Uncomment to use.
 *
#define Py_USING_MEMORY_DEBUGGER
 */

#ifdef Py_USING_MEMORY_DEBUGGER

/* Py_ADDRESS_IN_RANGE may access uninitialized memory by design
 * This leads to thousands of spurious warnings when using
 * Purify or Valgrind.  By making a function, we can easily
 * suppress the uninitialized memory reads in this one function.
 * So we won't ignore real errors elsewhere.
 *
 * Disable the macro and use a function.
 */

#undef Py_ADDRESS_IN_RANGE

#if defined(__GNUC__) && ((__GNUC__ == 3) && (__GNUC_MINOR__ >= 1) || \
                          (__GNUC__ >= 4))
#define Py_NO_INLINE __attribute__((__noinline__))
#else
#define Py_NO_INLINE
#endif

/* Don't make static, to try to ensure this isn't inlined. */
int Py_ADDRESS_IN_RANGE(void *P, poolp pool) Py_NO_INLINE;
#undef Py_NO_INLINE
#endif

/*==========================================================================*/

/* malloc.  Note that nbytes==0 tries to return a non-NULL pointer, distinct
 * from all other currently live pointers.  This may not be possible.
 */

/*
 * The basic blocks are ordered by decreasing execution frequency,
 * which minimizes the number of jumps in the most common cases,
 * improves branching prediction and instruction scheduling (small
 * block allocations typically result in a couple of instructions).
 * Unless the optimizer reorders everything, being too smart...
 */

static void *
_PyObject_Malloc(void *ctx, size_t nbytes)
{
    block *bp;
    poolp pool;
    poolp next;
    uint size;

#ifdef WITH_VALGRIND
    if (UNLIKELY(running_on_valgrind == -1))
        running_on_valgrind = RUNNING_ON_VALGRIND;
    if (UNLIKELY(running_on_valgrind))
        goto redirect;
#endif

    /*
     * This implicitly redirects malloc(0).
     */
    if ((nbytes - 1) < SMALL_REQUEST_THRESHOLD) {
        LOCK();
        /*
         * Most frequent paths first
         */
        size = (uint)(nbytes - 1) >> ALIGNMENT_SHIFT;
        pool = usedpools[size + size];
        if (pool != pool->nextpool) {
            /*
             * There is a used pool for this size class.
             * Pick up the head block of its free list.
             */
            ++pool->ref.count;
            bp = pool->freeblock;
            assert(bp != NULL);
            if ((pool->freeblock = *(block **)bp) != NULL) {
                UNLOCK();
                return (void *)bp;
            }
            /*
             * Reached the end of the free list, try to extend it.
             */
            if (pool->nextoffset <= pool->maxnextoffset) {
                /* There is room for another block. */
                pool->freeblock = (block*)pool +
                                  pool->nextoffset;
                pool->nextoffset += INDEX2SIZE(size);
                *(block **)(pool->freeblock) = NULL;
                UNLOCK();
                return (void *)bp;
            }
            /* Pool is full, unlink from used pools. */
            next = pool->nextpool;
            pool = pool->prevpool;
            next->prevpool = pool;
            pool->nextpool = next;
            UNLOCK();
            return (void *)bp;
        }

        /* There isn't a pool of the right size class immediately
         * available:  use a free pool.
         */
        if (usable_arenas == NULL) {
            /* No arena has a free pool:  allocate a new arena. */
#ifdef WITH_MEMORY_LIMITS
            if (narenas_currently_allocated >= MAX_ARENAS) {
                UNLOCK();
                goto redirect;
            }
#endif
            usable_arenas = new_arena();
            if (usable_arenas == NULL) {
                UNLOCK();
                goto redirect;
            }
            usable_arenas->nextarena =
                usable_arenas->prevarena = NULL;
        }
        assert(usable_arenas->address != 0);

        /* Try to get a cached free pool. */
        pool = usable_arenas->freepools;
        if (pool != NULL) {
            /* Unlink from cached pools. */
            usable_arenas->freepools = pool->nextpool;

            /* This arena already had the smallest nfreepools
             * value, so decreasing nfreepools doesn't change
             * that, and we don't need to rearrange the
             * usable_arenas list.  However, if the arena has
             * become wholly allocated, we need to remove its
             * arena_object from usable_arenas.
             */
            --usable_arenas->nfreepools;
            if (usable_arenas->nfreepools == 0) {
                /* Wholly allocated:  remove. */
                assert(usable_arenas->freepools == NULL);
                assert(usable_arenas->nextarena == NULL ||
                       usable_arenas->nextarena->prevarena ==
                       usable_arenas);

                usable_arenas = usable_arenas->nextarena;
                if (usable_arenas != NULL) {
                    usable_arenas->prevarena = NULL;
                    assert(usable_arenas->address != 0);
                }
            }
            else {
                /* nfreepools > 0:  it must be that freepools
                 * isn't NULL, or that we haven't yet carved
                 * off all the arena's pools for the first
                 * time.
                 */
                assert(usable_arenas->freepools != NULL ||
                       usable_arenas->pool_address <=
                       (block*)usable_arenas->address +
                           ARENA_SIZE - POOL_SIZE);
            }
        init_pool:
            /* Frontlink to used pools. */
            next = usedpools[size + size]; /* == prev */
            pool->nextpool = next;
            pool->prevpool = next;
            next->nextpool = pool;
            next->prevpool = pool;
            pool->ref.count = 1;
            if (pool->szidx == size) {
                /* Luckily, this pool last contained blocks
                 * of the same size class, so its header
                 * and free list are already initialized.
                 */
                bp = pool->freeblock;
                pool->freeblock = *(block **)bp;
                UNLOCK();
                return (void *)bp;
            }
            /*
             * Initialize the pool header, set up the free list to
             * contain just the second block, and return the first
             * block.
             */
            pool->szidx = size;
            size = INDEX2SIZE(size);
            bp = (block *)pool + POOL_OVERHEAD;
            pool->nextoffset = POOL_OVERHEAD + (size << 1);
            pool->maxnextoffset = POOL_SIZE - size;
            pool->freeblock = bp + size;
            *(block **)(pool->freeblock) = NULL;
            UNLOCK();
            return (void *)bp;
        }

        /* Carve off a new pool. */
        assert(usable_arenas->nfreepools > 0);
        assert(usable_arenas->freepools == NULL);
        pool = (poolp)usable_arenas->pool_address;
        assert((block*)pool <= (block*)usable_arenas->address +
                               ARENA_SIZE - POOL_SIZE);
        pool->arenaindex = usable_arenas - arenas;
        assert(&arenas[pool->arenaindex] == usable_arenas);
        pool->szidx = DUMMY_SIZE_IDX;
        usable_arenas->pool_address += POOL_SIZE;
        --usable_arenas->nfreepools;

        if (usable_arenas->nfreepools == 0) {
            assert(usable_arenas->nextarena == NULL ||
                   usable_arenas->nextarena->prevarena ==
                   usable_arenas);
            /* Unlink the arena:  it is completely allocated. */
            usable_arenas = usable_arenas->nextarena;
            if (usable_arenas != NULL) {
                usable_arenas->prevarena = NULL;
                assert(usable_arenas->address != 0);
            }
        }

        goto init_pool;
    }

    /* The small block allocator ends here. */

redirect:
    /* Redirect the original request to the underlying (libc) allocator.
     * We jump here on bigger requests, on error in the code above (as a
     * last chance to serve the request) or when the max memory limit
     * has been reached.
     */
    return PyMem_Malloc(nbytes);
}

/* free */

static void
_PyObject_Free(void *ctx, void *p)
{
    poolp pool;
    block *lastfree;
    poolp next, prev;
    uint size;
#ifndef Py_USING_MEMORY_DEBUGGER
    uint arenaindex_temp;
#endif

    if (p == NULL)      /* free(NULL) has no effect */
        return;

#ifdef WITH_VALGRIND
    if (UNLIKELY(running_on_valgrind > 0))
        goto redirect;
#endif

    pool = POOL_ADDR(p);
    if (Py_ADDRESS_IN_RANGE(p, pool)) {
        /* We allocated this address. */
        LOCK();
        /* Link p to the start of the pool's freeblock list.  Since
         * the pool had at least the p block outstanding, the pool
         * wasn't empty (so it's already in a usedpools[] list, or
         * was full and is in no list -- it's not in the freeblocks
         * list in any case).
         */
        assert(pool->ref.count > 0);            /* else it was empty */
        *(block **)p = lastfree = pool->freeblock;
        pool->freeblock = (block *)p;
        if (lastfree) {
            struct arena_object* ao;
            uint nf;  /* ao->nfreepools */

            /* freeblock wasn't NULL, so the pool wasn't full,
             * and the pool is in a usedpools[] list.
             */
            if (--pool->ref.count != 0) {
                /* pool isn't empty:  leave it in usedpools */
                UNLOCK();
                return;
            }
            /* Pool is now empty:  unlink from usedpools, and
             * link to the front of freepools.  This ensures that
             * previously freed pools will be allocated later
             * (being not referenced, they are perhaps paged out).
             */
            next = pool->nextpool;
            prev = pool->prevpool;
            next->prevpool = prev;
            prev->nextpool = next;

            /* Link the pool to freepools.  This is a singly-linked
             * list, and pool->prevpool isn't used there.
             */
            ao = &arenas[pool->arenaindex];
            pool->nextpool = ao->freepools;
            ao->freepools = pool;
            nf = ++ao->nfreepools;

            /* All the rest is arena management.  We just freed
             * a pool, and there are 4 cases for arena mgmt:
             * 1. If all the pools are free, return the arena to
             *    the system free().
             * 2. If this is the only free pool in the arena,
             *    add the arena back to the `usable_arenas` list.
             * 3. If the "next" arena has a smaller count of free
             *    pools, we have to "slide this arena right" to
             *    restore that usable_arenas is sorted in order of
             *    nfreepools.
             * 4. Else there's nothing more to do.
             */
            if (nf == ao->ntotalpools) {
                /* Case 1.  First unlink ao from usable_arenas.
                 */
                assert(ao->prevarena == NULL ||
                       ao->prevarena->address != 0);
                assert(ao ->nextarena == NULL ||
                       ao->nextarena->address != 0);

                /* Fix the pointer in the prevarena, or the
                 * usable_arenas pointer.
                 */
                if (ao->prevarena == NULL) {
                    usable_arenas = ao->nextarena;
                    assert(usable_arenas == NULL ||
                           usable_arenas->address != 0);
                }
                else {
                    assert(ao->prevarena->nextarena == ao);
                    ao->prevarena->nextarena =
                        ao->nextarena;
                }
                /* Fix the pointer in the nextarena. */
                if (ao->nextarena != NULL) {
                    assert(ao->nextarena->prevarena == ao);
                    ao->nextarena->prevarena =
                        ao->prevarena;
                }
                /* Record that this arena_object slot is
                 * available to be reused.
                 */
                ao->nextarena = unused_arena_objects;
                unused_arena_objects = ao;

                /* Free the entire arena. */
#ifdef ARENAS_USE_MMAP
                munmap((void *)ao->address, ARENA_SIZE);
#else
                free((void *)ao->address);
#endif
                ao->address = 0;                        /* mark unassociated */
                --narenas_currently_allocated;

                UNLOCK();
                return;
            }
            if (nf == 1) {
                /* Case 2.  Put ao at the head of
                 * usable_arenas.  Note that because
                 * ao->nfreepools was 0 before, ao isn't
                 * currently on the usable_arenas list.
                 */
                ao->nextarena = usable_arenas;
                ao->prevarena = NULL;
                if (usable_arenas)
                    usable_arenas->prevarena = ao;
                usable_arenas = ao;
                assert(usable_arenas->address != 0);

                UNLOCK();
                return;
            }
            /* If this arena is now out of order, we need to keep
             * the list sorted.  The list is kept sorted so that
             * the "most full" arenas are used first, which allows
             * the nearly empty arenas to be completely freed.  In
             * a few un-scientific tests, it seems like this
             * approach allowed a lot more memory to be freed.
             */
            if (ao->nextarena == NULL ||
                         nf <= ao->nextarena->nfreepools) {
                /* Case 4.  Nothing to do. */
                UNLOCK();
                return;
            }
            /* Case 3:  We have to move the arena towards the end
             * of the list, because it has more free pools than
             * the arena to its right.
             * First unlink ao from usable_arenas.
             */
            if (ao->prevarena != NULL) {
                /* ao isn't at the head of the list */
                assert(ao->prevarena->nextarena == ao);
                ao->prevarena->nextarena = ao->nextarena;
            }
            else {
                /* ao is at the head of the list */
                assert(usable_arenas == ao);
                usable_arenas = ao->nextarena;
            }
            ao->nextarena->prevarena = ao->prevarena;

            /* Locate the new insertion point by iterating over
             * the list, using our nextarena pointer.
             */
            while (ao->nextarena != NULL &&
                            nf > ao->nextarena->nfreepools) {
                ao->prevarena = ao->nextarena;
                ao->nextarena = ao->nextarena->nextarena;
            }

            /* Insert ao at this point. */
            assert(ao->nextarena == NULL ||
                ao->prevarena == ao->nextarena->prevarena);
            assert(ao->prevarena->nextarena == ao->nextarena);

            ao->prevarena->nextarena = ao;
            if (ao->nextarena != NULL)
                ao->nextarena->prevarena = ao;

            /* Verify that the swaps worked. */
            assert(ao->nextarena == NULL ||
                      nf <= ao->nextarena->nfreepools);
            assert(ao->prevarena == NULL ||
                      nf > ao->prevarena->nfreepools);
            assert(ao->nextarena == NULL ||
                ao->nextarena->prevarena == ao);
            assert((usable_arenas == ao &&
                ao->prevarena == NULL) ||
                ao->prevarena->nextarena == ao);

            UNLOCK();
            return;
        }
        /* Pool was full, so doesn't currently live in any list:
         * link it to the front of the appropriate usedpools[] list.
         * This mimics LRU pool usage for new allocations and
         * targets optimal filling when several pools contain
         * blocks of the same size class.
         */
        --pool->ref.count;
        assert(pool->ref.count > 0);            /* else the pool is empty */
        size = pool->szidx;
        next = usedpools[size + size];
        prev = next->prevpool;
        /* insert pool before next:   prev <-> pool <-> next */
        pool->nextpool = next;
        pool->prevpool = prev;
        next->prevpool = pool;
        prev->nextpool = pool;
        UNLOCK();
        return;
    }

#ifdef WITH_VALGRIND
redirect:
#endif
    /* We didn't allocate this address. */
    PyMem_Free(p);
}

/* realloc.  If p is NULL, this acts like malloc(nbytes).  Else if nbytes==0,
 * then as the Python docs promise, we do not treat this like free(p), and
 * return a non-NULL result.
 */

static void *
_PyObject_Realloc(void *ctx, void *p, size_t nbytes)
{
    void *bp;
    poolp pool;
    size_t size;
#ifndef Py_USING_MEMORY_DEBUGGER
    uint arenaindex_temp;
#endif

    if (p == NULL)
        return _PyObject_Malloc(ctx, nbytes);

#ifdef WITH_VALGRIND
    /* Treat running_on_valgrind == -1 the same as 0 */
    if (UNLIKELY(running_on_valgrind > 0))
        goto redirect;
#endif

    pool = POOL_ADDR(p);
    if (Py_ADDRESS_IN_RANGE(p, pool)) {
        /* We're in charge of this block */
        size = INDEX2SIZE(pool->szidx);
        if (nbytes <= size) {
            /* The block is staying the same or shrinking.  If
             * it's shrinking, there's a tradeoff:  it costs
             * cycles to copy the block to a smaller size class,
             * but it wastes memory not to copy it.  The
             * compromise here is to copy on shrink only if at
             * least 25% of size can be shaved off.
             */
            if (4 * nbytes > 3 * size) {
                /* It's the same,
                 * or shrinking and new/old > 3/4.
                 */
                return p;
            }
            size = nbytes;
        }
        bp = _PyObject_Malloc(ctx, nbytes);
        if (bp != NULL) {
            memcpy(bp, p, size);
            _PyObject_Free(ctx, p);
        }
        return bp;
    }
#ifdef WITH_VALGRIND
 redirect:
#endif
    /* We're not managing this block.  If nbytes <=
     * SMALL_REQUEST_THRESHOLD, it's tempting to try to take over this
     * block.  However, if we do, we need to copy the valid data from
     * the C-managed block to one of our blocks, and there's no portable
     * way to know how much of the memory space starting at p is valid.
     * As bug 1185883 pointed out the hard way, it's possible that the
     * C-managed block is "at the end" of allocated VM space, so that
     * a memory fault can occur if we try to copy nbytes bytes starting
     * at p.  Instead we punt:  let C continue to manage this block.
     */
    if (nbytes)
        return PyMem_Realloc(p, nbytes);
    /* C doesn't define the result of realloc(p, 0) (it may or may not
     * return NULL then), but Python's docs promise that nbytes==0 never
     * returns NULL.  We don't pass 0 to realloc(), to avoid that endcase
     * to begin with.  Even then, we can't be sure that realloc() won't
     * return NULL.
     */
    bp = PyMem_Realloc(p, 1);
    return bp ? bp : p;
}

#endif /* WITH_PYMALLOC */

#ifdef PYMALLOC_DEBUG
/*==========================================================================*/
/* A x-platform debugging allocator.  This doesn't manage memory directly,
 * it wraps a real allocator, adding extra debugging info to the memory blocks.
 */

/* Special bytes broadcast into debug memory blocks at appropriate times.
 * Strings of these are unlikely to be valid addresses, floats, ints or
 * 7-bit ASCII.
 */
#undef CLEANBYTE
#undef DEADBYTE
#undef FORBIDDENBYTE
#define CLEANBYTE      0xCB    /* clean (newly allocated) memory */
#define DEADBYTE       0xDB    /* dead (newly freed) memory */
#define FORBIDDENBYTE  0xFB    /* untouchable bytes at each end of a block */

static size_t serialno = 0;     /* incremented on each debug {m,re}alloc */

/* serialno is always incremented via calling this routine.  The point is
 * to supply a single place to set a breakpoint.
 */
static void
bumpserialno(void)
{
    ++serialno;
}

#define SST SIZEOF_SIZE_T

/* Read sizeof(size_t) bytes at p as a big-endian size_t. */
static size_t
read_size_t(const void *p)
{
    const uchar *q = (const uchar *)p;
    size_t result = *q++;
    int i;

    for (i = SST; --i > 0; ++q)
        result = (result << 8) | *q;
    return result;
}

/* Write n as a big-endian size_t, MSB at address p, LSB at
 * p + sizeof(size_t) - 1.
 */
static void
write_size_t(void *p, size_t n)
{
    uchar *q = (uchar *)p + SST - 1;
    int i;

    for (i = SST; --i >= 0; --q) {
        *q = (uchar)(n & 0xff);
        n >>= 8;
    }
}

#ifdef Py_DEBUG
/* Is target in the list?  The list is traversed via the nextpool pointers.
 * The list may be NULL-terminated, or circular.  Return 1 if target is in
 * list, else 0.
 */
static int
pool_is_in_list(const poolp target, poolp list)
{
    poolp origlist = list;
    assert(target != NULL);
    if (list == NULL)
        return 0;
    do {
        if (target == list)
            return 1;
        list = list->nextpool;
    } while (list != NULL && list != origlist);
    return 0;
}

#else
#define pool_is_in_list(X, Y) 1

#endif  /* Py_DEBUG */

static void *
_PyMem_Malloc(size_t nbytes)
{
    if (nbytes > (size_t)PY_SSIZE_T_MAX) {
        return NULL;
    }
    if (nbytes == 0) {
        nbytes = 1;
    }
    return malloc(nbytes);
}

static void *
_PyMem_Realloc(void *p, size_t nbytes)
{
    if (nbytes > (size_t)PY_SSIZE_T_MAX) {
        return NULL;
    }
    if (nbytes == 0) {
        nbytes = 1;
    }
    return realloc(p, nbytes);
}


static void
_PyMem_Free(void *p)
{
    free(p);
}


/* Let S = sizeof(size_t).  The debug malloc asks for 4*S extra bytes and
   fills them with useful stuff, here calling the underlying malloc's result p:

p[0: S]
    Number of bytes originally asked for.  This is a size_t, big-endian (easier
    to read in a memory dump).
p[S: 2*S]
    Copies of FORBIDDENBYTE.  Used to catch under- writes and reads.
p[2*S: 2*S+n]
    The requested memory, filled with copies of CLEANBYTE.
    Used to catch reference to uninitialized memory.
    &p[2*S] is returned.  Note that this is 8-byte aligned if pymalloc
    handled the request itself.
p[2*S+n: 2*S+n+S]
    Copies of FORBIDDENBYTE.  Used to catch over- writes and reads.
p[2*S+n+S: 2*S+n+2*S]
    A serial number, incremented by 1 on each call to _PyMem_DebugMalloc
    and _PyMem_DebugRealloc.
    This is a big-endian size_t.
    If "bad memory" is detected later, the serial number gives an
    excellent way to set a breakpoint on the next run, to capture the
    instant at which this block was passed out.
*/

static void *
_PyMem_DebugMallocCtx(void *ctx, size_t nbytes)
{
    debug_alloc_api_t *api = (debug_alloc_api_t *)ctx;
    uchar *p;           /* base address of malloc'ed block */
    uchar *tail;        /* p + 2*SST + nbytes == pointer to tail pad bytes */
    size_t total;       /* nbytes + 4*SST */

    bumpserialno();
    total = nbytes + 4*SST;
    if (total < nbytes)
        /* overflow:  can't represent total as a size_t */
        return NULL;

    p = (uchar *)api->alloc.malloc(api->alloc.ctx, total);
    if (p == NULL)
        return NULL;

    /* at p, write size (SST bytes), api (1 byte), pad (SST-1 bytes) */
    write_size_t(p, nbytes);
    p[SST] = (uchar)api->api_id;
    memset(p + SST + 1, FORBIDDENBYTE, SST-1);

    if (nbytes > 0)
        memset(p + 2*SST, CLEANBYTE, nbytes);

    /* at tail, write pad (SST bytes) and serialno (SST bytes) */
    tail = p + 2*SST + nbytes;
    memset(tail, FORBIDDENBYTE, SST);
    write_size_t(tail + SST, serialno);

    return p + 2*SST;
}

/* The debug free first checks the 2*SST bytes on each end for sanity (in
   particular, that the FORBIDDENBYTEs with the api ID are still intact).
   Then fills the original bytes with DEADBYTE.
   Then calls the underlying free.
*/
static void
_PyMem_DebugFreeCtx(void *ctx, void *p)
{
    debug_alloc_api_t *api = (debug_alloc_api_t *)ctx;
    uchar *q = (uchar *)p - 2*SST;  /* address returned from malloc */
    size_t nbytes;

    if (p == NULL)
        return;
    _PyMem_DebugCheckAddress(api->api_id, p);
    nbytes = read_size_t(q);
    nbytes += 4*SST;
    if (nbytes > 0)
        memset(q, DEADBYTE, nbytes);
    api->alloc.free(api->alloc.ctx, q);
}

static void *
_PyMem_DebugReallocCtx(void *ctx, void *p, size_t nbytes)
{
    debug_alloc_api_t *api = (debug_alloc_api_t *)ctx;
    uchar *q = (uchar *)p, *oldq;
    uchar *tail;
    size_t total;       /* nbytes + 4*SST */
    size_t original_nbytes;
    int i;

    if (p == NULL)
        return _PyMem_DebugMallocCtx(ctx, nbytes);

    _PyMem_DebugCheckAddress(api->api_id, p);
    bumpserialno();
    original_nbytes = read_size_t(q - 2*SST);
    total = nbytes + 4*SST;
    if (total < nbytes)
        /* overflow:  can't represent total as a size_t */
        return NULL;

    /* Resize and add decorations. We may get a new pointer here, in which
     * case we didn't get the chance to mark the old memory with DEADBYTE,
     * but we live with that.
     */
    oldq = q;
    q = (uchar *)api->alloc.realloc(api->alloc.ctx, q - 2*SST, total);
    if (q == NULL) {
        if (nbytes <= original_nbytes) {
            /* bpo-31626: the memset() above expects that realloc never fails
               on shrinking a memory block. */
            Py_FatalError("Shrinking reallocation failed");
        }
        return NULL;
    }

    if (q == oldq && nbytes <= original_nbytes) {
        /* shrinking:  mark old extra memory dead */
        memset(q + nbytes, DEADBYTE, original_nbytes - nbytes);
    }

    write_size_t(q, nbytes);
    assert(q[SST] == (uchar)api->api_id);
    for (i = 1; i < SST; ++i)
        assert(q[SST + i] == FORBIDDENBYTE);
    q += 2*SST;

    tail = q + nbytes;
    memset(tail, FORBIDDENBYTE, SST);
    write_size_t(tail + SST, serialno);

    if (nbytes > original_nbytes) {
        /* growing:  mark new extra memory clean */
        memset(q + original_nbytes, CLEANBYTE,
               nbytes - original_nbytes);
    }

    return q;
}

/* Check the forbidden bytes on both ends of the memory allocated for p.
 * If anything is wrong, print info to stderr via _PyObject_DebugDumpAddress,
 * and call Py_FatalError to kill the program.
 * The API id, is also checked.
 */
static void
_PyMem_DebugCheckAddress(char api, const void *p)
{
    const uchar *q = (const uchar *)p;
    char msgbuf[64];
    char *msg;
    size_t nbytes;
    const uchar *tail;
    int i;
    char id;

    if (p == NULL) {
        msg = "didn't expect a NULL pointer";
        goto error;
    }

    /* Check the API id */
    id = (char)q[-SST];
    if (id != api) {
        msg = msgbuf;
        snprintf(msg, sizeof(msgbuf), "bad ID: Allocated using API '%c', verified using API '%c'", id, api);
        msgbuf[sizeof(msgbuf)-1] = 0;
        goto error;
    }

    /* Check the stuff at the start of p first:  if there's underwrite
     * corruption, the number-of-bytes field may be nuts, and checking
     * the tail could lead to a segfault then.
     */
    for (i = SST-1; i >= 1; --i) {
        if (*(q-i) != FORBIDDENBYTE) {
            msg = "bad leading pad byte";
            goto error;
        }
    }

    nbytes = read_size_t(q - 2*SST);
    tail = q + nbytes;
    for (i = 0; i < SST; ++i) {
        if (tail[i] != FORBIDDENBYTE) {
            msg = "bad trailing pad byte";
            goto error;
        }
    }

    return;

error:
    _PyObject_DebugDumpAddress(p);
    Py_FatalError(msg);
}

/* Display info to stderr about the memory block at p. */
void
_PyObject_DebugDumpAddress(const void *p)
{
    const uchar *q = (const uchar *)p;
    const uchar *tail;
    size_t nbytes, serial;
    int i;
    int ok;
    char id;

    fprintf(stderr, "Debug memory block at address p=%p:", p);
    if (p == NULL) {
        fprintf(stderr, "\n");
        return;
    }
    id = (char)q[-SST];
    fprintf(stderr, " API '%c'\n", id);

    nbytes = read_size_t(q - 2*SST);
    fprintf(stderr, "    %" PY_FORMAT_SIZE_T "u bytes originally "
                    "requested\n", nbytes);

    /* In case this is nuts, check the leading pad bytes first. */
    fprintf(stderr, "    The %d pad bytes at p-%d are ", SST-1, SST-1);
    ok = 1;
    for (i = 1; i <= SST-1; ++i) {
        if (*(q-i) != FORBIDDENBYTE) {
            ok = 0;
            break;
        }
    }
    if (ok)
        fputs("FORBIDDENBYTE, as expected.\n", stderr);
    else {
        fprintf(stderr, "not all FORBIDDENBYTE (0x%02x):\n",
            FORBIDDENBYTE);
        for (i = SST-1; i >= 1; --i) {
            const uchar byte = *(q-i);
            fprintf(stderr, "        at p-%d: 0x%02x", i, byte);
            if (byte != FORBIDDENBYTE)
                fputs(" *** OUCH", stderr);
            fputc('\n', stderr);
        }

        fputs("    Because memory is corrupted at the start, the "
              "count of bytes requested\n"
              "       may be bogus, and checking the trailing pad "
              "bytes may segfault.\n", stderr);
    }

    tail = q + nbytes;
    fprintf(stderr, "    The %d pad bytes at tail=%p are ", SST, tail);
    ok = 1;
    for (i = 0; i < SST; ++i) {
        if (tail[i] != FORBIDDENBYTE) {
            ok = 0;
            break;
        }
    }
    if (ok)
        fputs("FORBIDDENBYTE, as expected.\n", stderr);
    else {
        fprintf(stderr, "not all FORBIDDENBYTE (0x%02x):\n",
                FORBIDDENBYTE);
        for (i = 0; i < SST; ++i) {
            const uchar byte = tail[i];
            fprintf(stderr, "        at tail+%d: 0x%02x",
                    i, byte);
            if (byte != FORBIDDENBYTE)
                fputs(" *** OUCH", stderr);
            fputc('\n', stderr);
        }
    }

    serial = read_size_t(tail + SST);
    fprintf(stderr, "    The block was made by call #%" PY_FORMAT_SIZE_T
                    "u to debug malloc/realloc.\n", serial);

    if (nbytes > 0) {
        i = 0;
        fputs("    Data at p:", stderr);
        /* print up to 8 bytes at the start */
        while (q < tail && i < 8) {
            fprintf(stderr, " %02x", *q);
            ++i;
            ++q;
        }
        /* and up to 8 at the end */
        if (q < tail) {
            if (tail - q > 8) {
                fputs(" ...", stderr);
                q = tail - 8;
            }
            while (q < tail) {
                fprintf(stderr, " %02x", *q);
                ++q;
            }
        }
        fputc('\n', stderr);
    }
}

static size_t
printone(const char* msg, size_t value)
{
    int i, k;
    char buf[100];
    size_t origvalue = value;

    fputs(msg, stderr);
    for (i = (int)strlen(msg); i < 35; ++i)
        fputc(' ', stderr);
    fputc('=', stderr);

    /* Write the value with commas. */
    i = 22;
    buf[i--] = '\0';
    buf[i--] = '\n';
    k = 3;
    do {
        size_t nextvalue = value / 10;
        unsigned int digit = (unsigned int)(value - nextvalue * 10);
        value = nextvalue;
        buf[i--] = (char)(digit + '0');
        --k;
        if (k == 0 && value && i >= 0) {
            k = 3;
            buf[i--] = ',';
        }
    } while (value && i >= 0);

    while (i >= 0)
        buf[i--] = ' ';
    fputs(buf, stderr);

    return origvalue;
}

/* Print summary info to stderr about the state of pymalloc's structures.
 * In Py_DEBUG mode, also perform some expensive internal consistency
 * checks.
 */
void
_PyObject_DebugMallocStats(void)
{
    uint i;
    const uint numclasses = SMALL_REQUEST_THRESHOLD >> ALIGNMENT_SHIFT;
    /* # of pools, allocated blocks, and free blocks per class index */
    size_t numpools[SMALL_REQUEST_THRESHOLD >> ALIGNMENT_SHIFT];
    size_t numblocks[SMALL_REQUEST_THRESHOLD >> ALIGNMENT_SHIFT];
    size_t numfreeblocks[SMALL_REQUEST_THRESHOLD >> ALIGNMENT_SHIFT];
    /* total # of allocated bytes in used and full pools */
    size_t allocated_bytes = 0;
    /* total # of available bytes in used pools */
    size_t available_bytes = 0;
    /* # of free pools + pools not yet carved out of current arena */
    uint numfreepools = 0;
    /* # of bytes for arena alignment padding */
    size_t arena_alignment = 0;
    /* # of bytes in used and full pools used for pool_headers */
    size_t pool_header_bytes = 0;
    /* # of bytes in used and full pools wasted due to quantization,
     * i.e. the necessarily leftover space at the ends of used and
     * full pools.
     */
    size_t quantization = 0;
    /* # of arenas actually allocated. */
    size_t narenas = 0;
    /* running total -- should equal narenas * ARENA_SIZE */
    size_t total;
    char buf[128];

    fprintf(stderr, "Small block threshold = %d, in %u size classes.\n",
            SMALL_REQUEST_THRESHOLD, numclasses);

    for (i = 0; i < numclasses; ++i)
        numpools[i] = numblocks[i] = numfreeblocks[i] = 0;

    /* Because full pools aren't linked to from anything, it's easiest
     * to march over all the arenas.  If we're lucky, most of the memory
     * will be living in full pools -- would be a shame to miss them.
     */
    for (i = 0; i < maxarenas; ++i) {
        uint j;
        uptr base = arenas[i].address;

        /* Skip arenas which are not allocated. */
        if (arenas[i].address == (uptr)NULL)
            continue;
        narenas += 1;

        numfreepools += arenas[i].nfreepools;

        /* round up to pool alignment */
        if (base & (uptr)POOL_SIZE_MASK) {
            arena_alignment += POOL_SIZE;
            base &= ~(uptr)POOL_SIZE_MASK;
            base += POOL_SIZE;
        }

        /* visit every pool in the arena */
        assert(base <= (uptr) arenas[i].pool_address);
        for (j = 0;
                    base < (uptr) arenas[i].pool_address;
                    ++j, base += POOL_SIZE) {
            poolp p = (poolp)base;
            const uint sz = p->szidx;
            uint freeblocks;

            if (p->ref.count == 0) {
                /* currently unused */
                assert(pool_is_in_list(p, arenas[i].freepools));
                continue;
            }
            ++numpools[sz];
            numblocks[sz] += p->ref.count;
            freeblocks = NUMBLOCKS(sz) - p->ref.count;
            numfreeblocks[sz] += freeblocks;
#ifdef Py_DEBUG
            if (freeblocks > 0)
                assert(pool_is_in_list(p, usedpools[sz + sz]));
#endif
        }
    }
    assert(narenas == narenas_currently_allocated);

    fputc('\n', stderr);
    fputs("class   size   num pools   blocks in use  avail blocks\n"
          "-----   ----   ---------   -------------  ------------\n",
          stderr);

    for (i = 0; i < numclasses; ++i) {
        size_t p = numpools[i];
        size_t b = numblocks[i];
        size_t f = numfreeblocks[i];
        uint size = INDEX2SIZE(i);
        if (p == 0) {
            assert(b == 0 && f == 0);
            continue;
        }
        fprintf(stderr, "%5u %6u "
                        "%11" PY_FORMAT_SIZE_T "u "
                        "%15" PY_FORMAT_SIZE_T "u "
                        "%13" PY_FORMAT_SIZE_T "u\n",
                i, size, p, b, f);
        allocated_bytes += b * size;
        available_bytes += f * size;
        pool_header_bytes += p * POOL_OVERHEAD;
        quantization += p * ((POOL_SIZE - POOL_OVERHEAD) % size);
    }
    fputc('\n', stderr);
    (void)printone("# times object malloc called", serialno);

    (void)printone("# arenas allocated total", ntimes_arena_allocated);
    (void)printone("# arenas reclaimed", ntimes_arena_allocated - narenas);
    (void)printone("# arenas highwater mark", narenas_highwater);
    (void)printone("# arenas allocated current", narenas);

    PyOS_snprintf(buf, sizeof(buf),
        "%" PY_FORMAT_SIZE_T "u arenas * %d bytes/arena",
        narenas, ARENA_SIZE);
    (void)printone(buf, narenas * ARENA_SIZE);

    fputc('\n', stderr);

    total = printone("# bytes in allocated blocks", allocated_bytes);
    total += printone("# bytes in available blocks", available_bytes);

    PyOS_snprintf(buf, sizeof(buf),
        "%u unused pools * %d bytes", numfreepools, POOL_SIZE);
    total += printone(buf, (size_t)numfreepools * POOL_SIZE);

    total += printone("# bytes lost to pool headers", pool_header_bytes);
    total += printone("# bytes lost to quantization", quantization);
    total += printone("# bytes lost to arena alignment", arena_alignment);
    (void)printone("Total", total);
}

#endif  /* PYMALLOC_DEBUG */

#ifdef Py_USING_MEMORY_DEBUGGER
/* Make this function last so gcc won't inline it since the definition is
 * after the reference.
 */
int
Py_ADDRESS_IN_RANGE(void *P, poolp pool)
{
    uint arenaindex_temp = pool->arenaindex;

    return arenaindex_temp < maxarenas &&
           (uptr)P - arenas[arenaindex_temp].address < (uptr)ARENA_SIZE &&
           arenas[arenaindex_temp].address != 0;
}
#endif


#if defined(WITH_PYMALLOC) && defined(PYMALLOC_DEBUG)
/* Dummy functions only present to keep the same ABI with the vanilla Python
   compiled in debug mode: they are not used in practice. See issue:
   https://github.com/vstinner/pytracemalloc/issues/1 */

void* _PyMem_DebugMalloc(size_t nbytes)
{ return PyMem_RawMalloc(nbytes); }

void* _PyMem_DebugRealloc(void *p, size_t nbytes)
{ return PyMem_RawRealloc(p, nbytes); }

void _PyObject_DebugFree(void *p)
{ return PyObject_Free(p); }

void* _PyObject_DebugMalloc(size_t nbytes)
{ return PyObject_Malloc(nbytes); }

void* _PyObject_DebugRealloc(void *p, size_t nbytes)
{ return PyObject_Realloc(p, nbytes); }

void _PyMem_DebugFree(void *p)
{ PyMem_RawFree(p); }

void _PyObject_DebugCheckAddress(const void *p)
{}

void * _PyObject_DebugMallocApi(char api, size_t nbytes)
{ return PyObject_Malloc(nbytes); }

void * _PyObject_DebugReallocApi(char api, void *p, size_t nbytes)
{ return PyObject_Realloc(p, nbytes); }

void _PyObject_DebugFreeApi(char api, void *p)
{ return PyObject_Free(p); }

void _PyObject_DebugCheckAddressApi(char api, const void *p)
{}
#endif
