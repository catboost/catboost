#ifndef ATOMIC__H
#define ATOMIC__H

typedef volatile intptr_t atomic_t;

#ifdef __cplusplus
  #define EXTERN_C extern "C"
#else
  #define EXTERN_C
#endif

EXTERN_C void acquire_lock(atomic_t *lock);
EXTERN_C void release_lock(atomic_t *lock);

#endif
