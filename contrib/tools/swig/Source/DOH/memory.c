/* ----------------------------------------------------------------------------- 
 * This file is part of SWIG, which is licensed as a whole under version 3 
 * (or any later version) of the GNU General Public License. Some additional
 * terms also apply to certain portions of SWIG. The full details of the SWIG
 * license and copyrights can be found in the LICENSE and COPYRIGHT files
 * included with the SWIG source code as distributed by the SWIG developers
 * and at https://www.swig.org/legal.html.
 *
 * memory.c
 *
 *     This file implements all of DOH's memory management including allocation
 *     of objects and checking of objects.
 * ----------------------------------------------------------------------------- */

#include "dohint.h"

#include <stdio.h>
#include <stdlib.h>

#ifndef DOH_POOL_SIZE
#define DOH_POOL_SIZE         4194304
#endif

/* Checks stale DOH object use - will use a lot more memory as pool memory is not re-used. */
/*
#define DOH_DEBUG_MEMORY_POOLS
*/

static int PoolSize = DOH_POOL_SIZE;

DOH *DohNone = 0;		/* The DOH None object */

typedef struct pool {
  DohBase *ptr;			/* Start of pool */
  int len;			/* Length of pool */
  int blen;			/* Byte length of pool */
  int current;			/* Current position for next allocation */
  char *pbeg;			/* Beg of pool */
  char *pend;			/* End of pool */
  struct pool *next;		/* Next pool */
} Pool;

static DohBase *FreeList = 0;	/* List of free objects */
static Pool *Pools = 0;
static int pools_initialized = 0;

/* ----------------------------------------------------------------------
 * CreatePool() - Create a new memory pool 
 * ---------------------------------------------------------------------- */

static void CreatePool(void) {
  Pool *p = 0;
  p = (Pool *) DohMalloc(sizeof(Pool));
  p->ptr = (DohBase *) DohCalloc(PoolSize, sizeof(DohBase));
  p->len = PoolSize;
  p->blen = PoolSize * sizeof(DohBase);
  p->current = 0;
  p->pbeg = ((char *) p->ptr);
  p->pend = p->pbeg + p->blen;
  p->next = Pools;
  Pools = p;
}

/* ----------------------------------------------------------------------
 * InitPools() - Initialize the memory allocator
 * ---------------------------------------------------------------------- */

static void InitPools(void) {
  if (pools_initialized)
    return;
  CreatePool();			/* Create initial pool */
  pools_initialized = 1;
  DohNone = NewVoid(0, 0);	/* Create the None object */
  DohIntern(DohNone);
}

/* ----------------------------------------------------------------------
 * DohCheck()
 *
 * Returns 1 if an arbitrary pointer is a DOH object.
 * ---------------------------------------------------------------------- */

int DohCheck(const DOH *ptr) {
  Pool *p = Pools;
  char *cptr = (char *) ptr;
  while (p) {
    if ((cptr >= p->pbeg) && (cptr < p->pend)) {
#ifdef DOH_DEBUG_MEMORY_POOLS
      DohBase *b = (DohBase *) ptr;
      int DOH_object_already_deleted = b->type == 0;
      assert(!DOH_object_already_deleted);
#endif
      return 1;
    }
    /*
       pptr = (char *) p->ptr;
       if ((cptr >= pptr) && (cptr < (pptr+(p->current*sizeof(DohBase))))) return 1; */
    p = p->next;
  }
  return 0;
}

/* -----------------------------------------------------------------------------
 * DohIntern()
 * ----------------------------------------------------------------------------- */

void DohIntern(DOH *obj) {
  DohBase *b = (DohBase *) obj;
  b->flag_intern = 1;
}

/* ----------------------------------------------------------------------
 * DohObjMalloc()
 *
 * Allocate memory for a new object.
 * ---------------------------------------------------------------------- */

DOH *DohObjMalloc(DohObjInfo *type, void *data) {
  DohBase *obj;
  if (!pools_initialized)
    InitPools();
#ifndef DOH_DEBUG_MEMORY_POOLS
  if (FreeList) {
    obj = FreeList;
    FreeList = (DohBase *) obj->data;
  } else {
#endif
    while (Pools->current == Pools->len) {
      CreatePool();
    }
    obj = Pools->ptr + Pools->current;
    ++Pools->current;
#ifndef DOH_DEBUG_MEMORY_POOLS
  }
#endif
  obj->type = type;
  obj->data = data;
  obj->meta = 0;
  obj->refcount = 1;
  obj->flag_intern = 0;
  obj->flag_marked = 0;
  obj->flag_user = 0;
  obj->flag_usermark = 0;
  return (DOH *) obj;
}

/* ----------------------------------------------------------------------
 * DohObjFree() - Free a DOH object
 * ---------------------------------------------------------------------- */

void DohObjFree(DOH *ptr) {
  DohBase *b, *meta;
  b = (DohBase *) ptr;
  if (b->flag_intern)
    return;
  meta = (DohBase *) b->meta;
  b->data = (void *) FreeList;
  b->meta = 0;
  b->type = 0;
  b->refcount = 0;
  FreeList = b;
  if (meta) {
    Delete(meta);
  }
}

/* ----------------------------------------------------------------------
 * DohMemoryDebug()
 *
 * Display memory usage statistics
 * ---------------------------------------------------------------------- */

void DohMemoryDebug(void) {
  extern DohObjInfo DohStringType;
  extern DohObjInfo DohListType;
  extern DohObjInfo DohHashType;

  Pool *p;
  int totsize = 0;
  int totused = 0;
  int totfree = 0;

  int numstring = 0;
  int numlist = 0;
  int numhash = 0;

  printf("Memory statistics:\n\n");
  printf("Pools:\n");

  p = Pools;
  while (p) {
    /* Calculate number of used, free items */
    int i;
    int nused = 0, nfree = 0;
    for (i = 0; i < p->len; i++) {
      if (p->ptr[i].refcount <= 0)
	nfree++;
      else {
	nused++;
	if (p->ptr[i].type == &DohStringType)
	  numstring++;
	else if (p->ptr[i].type == &DohListType)
	  numlist++;
	else if (p->ptr[i].type == &DohHashType)
	  numhash++;
      }
    }
    printf("    Pool %8p: size = %10d. used = %10d. free = %10d\n", (void *) p, p->len, nused, nfree);
    totsize += p->len;
    totused += nused;
    totfree += nfree;
    p = p->next;
  }
  printf("\n    Total:          size = %10d, used = %10d, free = %10d\n", totsize, totused, totfree);

  printf("\nObject types\n");
  printf("    Strings   : %d\n", numstring);
  printf("    Lists     : %d\n", numlist);
  printf("    Hashes    : %d\n", numhash);

#if 0
  p = Pools;
  while (p) {
    int i;
    for (i = 0; i < p->len; i++) {
      if (p->ptr[i].refcount > 0) {
	if (p->ptr[i].type == &DohStringType) {
	  Printf(stdout, "%s\n", p->ptr + i);
	}
      }
    }
    p = p->next;
  }
#endif

}

/* Function to call instead of exit(). */
static void (*doh_exit_handler)(int) = NULL;

void DohSetExitHandler(void (*new_handler)(int)) {
  doh_exit_handler = new_handler;
}

void DohExit(int status) {
  if (doh_exit_handler) {
    void (*handler)(int) = doh_exit_handler;
    /* Unset the handler to avoid infinite loops if it tries to do something
     * which calls DohExit() (e.g. calling Malloc() and that failing).
     */
    doh_exit_handler = NULL;
    handler(status);
  }
  doh_internal_exit(status);
}

static void allocation_failed(size_t n, size_t size) {
  /* Report and exit as directly as possible to try to avoid further issues due
   * to lack of memory. */
  if (n == 1) {
#if defined __STDC_VERSION__ && __STDC_VERSION__-0 >= 19901L
    fprintf(stderr, "Failed to allocate %zu bytes\n", size);
#else
    fprintf(stderr, "Failed to allocate %lu bytes\n", (unsigned long)size);
#endif
  } else {
#if defined __STDC_VERSION__ && __STDC_VERSION__-0 >= 19901L
    fprintf(stderr, "Failed to allocate %zu*%zu bytes\n", n, size);
#else
    fprintf(stderr, "Failed to allocate %lu*%lu bytes\n", (unsigned long)n, (unsigned long)size);
#endif
  }
  DohExit(EXIT_FAILURE);
}

void *DohMalloc(size_t size) {
  void *p = doh_internal_malloc(size);
  if (!p) allocation_failed(1, size);
  return p;
}

void *DohRealloc(void *ptr, size_t size) {
  void *p = doh_internal_realloc(ptr, size);
  if (!p) allocation_failed(1, size);
  return p;
}

void *DohCalloc(size_t n, size_t size) {
  void *p = doh_internal_calloc(n, size);
  if (!p) allocation_failed(n, size);
  return p;
}
