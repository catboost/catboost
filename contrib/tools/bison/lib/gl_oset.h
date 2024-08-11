/* Abstract ordered set data type.
   Copyright (C) 2006-2007, 2009-2020 Free Software Foundation, Inc.
   Written by Bruno Haible <bruno@clisp.org>, 2006.

   This program is free software: you can redistribute it and/or modify
   it under the terms of the GNU General Public License as published by
   the Free Software Foundation; either version 3 of the License, or
   (at your option) any later version.

   This program is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
   GNU General Public License for more details.

   You should have received a copy of the GNU General Public License
   along with this program.  If not, see <https://www.gnu.org/licenses/>.  */

#ifndef _GL_OSET_H
#define _GL_OSET_H

#include <stdbool.h>
#include <stddef.h>

#ifndef _GL_INLINE_HEADER_BEGIN
 #error "Please include config.h first."
#endif
_GL_INLINE_HEADER_BEGIN
#ifndef GL_OSET_INLINE
# define GL_OSET_INLINE _GL_INLINE
#endif

#ifdef __cplusplus
extern "C" {
#endif


/* gl_oset is an abstract ordered set data type.  It can contain any number
   of objects ('void *' or 'const void *' pointers) in the order of a given
   comparator function.  Duplicates (in the sense of the comparator) are
   forbidden.

   There are several implementations of this ordered set datatype, optimized
   for different operations or for memory.  You can start using the simplest
   ordered set implementation, GL_ARRAY_OSET, and switch to a different
   implementation later, when you realize which operations are performed
   the most frequently.  The API of the different implementations is exactly
   the same; when switching to a different implementation, you only have to
   change the gl_oset_create call.

   The implementations are:
     GL_ARRAY_OSET        a growable array
     GL_AVLTREE_OSET      a binary tree (AVL tree)
     GL_RBTREE_OSET       a binary tree (red-black tree)

   The memory consumption is asymptotically the same: O(1) for every object
   in the set.  When looking more closely at the average memory consumed
   for an object, GL_ARRAY_OSET is the most compact representation, and
   GL_AVLTREE_OSET, GL_RBTREE_OSET need more memory.

   The guaranteed average performance of the operations is, for a set of
   n elements:

   Operation                  ARRAY     TREE

   gl_oset_size                O(1)     O(1)
   gl_oset_add                 O(n)   O(log n)
   gl_oset_remove              O(n)   O(log n)
   gl_oset_update              O(n)   O(log n)
   gl_oset_search            O(log n) O(log n)
   gl_oset_search_atleast    O(log n) O(log n)
   gl_oset_iterator            O(1)   O(log n)
   gl_oset_iterator_atleast  O(log n) O(log n)
   gl_oset_iterator_next       O(1)   O(log n)
 */

/* -------------------------- gl_oset_t Data Type -------------------------- */

/* Type of function used to compare two elements.  Same as for qsort().
   NULL denotes pointer comparison.  */
typedef int (*gl_setelement_compar_fn) (const void *elt1, const void *elt2);

#ifndef _GL_SETELEMENT_DISPOSE_FN_DEFINED
/* Type of function used to dispose an element once it's removed from a set.
   NULL denotes a no-op.  */
typedef void (*gl_setelement_dispose_fn) (const void *elt);
# define _GL_SETELEMENT_DISPOSE_FN_DEFINED 1
#endif

/* Type of function used to compare an element with a threshold.
   Returns true if the element is greater or equal than the threshold.  */
typedef bool (*gl_setelement_threshold_fn) (const void *elt, const void *threshold);

struct gl_oset_impl;
/* Type representing an entire ordered set.  */
typedef struct gl_oset_impl * gl_oset_t;

struct gl_oset_implementation;
/* Type representing a ordered set datatype implementation.  */
typedef const struct gl_oset_implementation * gl_oset_implementation_t;

#if 0 /* Unless otherwise specified, these are defined inline below.  */

/* Creates an empty set.
   IMPLEMENTATION is one of GL_ARRAY_OSET, GL_AVLTREE_OSET, GL_RBTREE_OSET.
   COMPAR_FN is an element comparison function or NULL.
   DISPOSE_FN is an element disposal function or NULL.  */
/* declared in gl_xoset.h */
extern gl_oset_t gl_oset_create_empty (gl_oset_implementation_t implementation,
                                       gl_setelement_compar_fn compar_fn,
                                       gl_setelement_dispose_fn dispose_fn);
/* Likewise.  Returns NULL upon out-of-memory.  */
extern gl_oset_t gl_oset_nx_create_empty (gl_oset_implementation_t implementation,
                                          gl_setelement_compar_fn compar_fn,
                                          gl_setelement_dispose_fn dispose_fn);

/* Returns the current number of elements in an ordered set.  */
extern size_t gl_oset_size (gl_oset_t set);

/* Searches whether an element is already in the ordered set.
   Returns true if found, or false if not present in the set.  */
extern bool gl_oset_search (gl_oset_t set, const void *elt);

/* Searches the least element in the ordered set that compares greater or equal
   to the given THRESHOLD.  The representation of the THRESHOLD is defined
   by the THRESHOLD_FN.
   Returns true and stores the found element in *ELTP if found, otherwise returns
   false.  */
extern bool gl_oset_search_atleast (gl_oset_t set,
                                    gl_setelement_threshold_fn threshold_fn,
                                    const void *threshold,
                                    const void **eltp);

/* Adds an element to an ordered set.
   Returns true if it was not already in the set and added, false otherwise.  */
/* declared in gl_xoset.h */
extern bool gl_oset_add (gl_oset_t set, const void *elt);
/* Likewise.  Returns -1 upon out-of-memory.  */
extern int gl_oset_nx_add (gl_oset_t set, const void *elt)
  _GL_ATTRIBUTE_NODISCARD;

/* Removes an element from an ordered set.
   Returns true if it was found and removed.  */
extern bool gl_oset_remove (gl_oset_t set, const void *elt);

/* Invokes ACTION (ELT, ACTION_DATA) and updates the given ordered set if,
   during this invocation, the attributes/properties of the element ELT change
   in a way that influences the comparison function.
   Warning: During the invocation of ACTION, the ordered set is inconsistent
   and must not be accessed!
   Returns 1 if the position of the element in the ordered set has changed as
   a consequence, 0 if the element stayed at the same position, or -1 if it
   collided with another element and was therefore removed.  */
extern int gl_oset_update (gl_oset_t set, const void *elt,
                           void (*action) (const void *elt, void *action_data),
                           void *action_data);

/* Frees an entire ordered set.
   (But this call does not free the elements of the set.  It only invokes
   the DISPOSE_FN on each of the elements of the set.)  */
extern void gl_oset_free (gl_oset_t set);

#endif /* End of inline and gl_xoset.h-defined functions.  */

/* --------------------- gl_oset_iterator_t Data Type --------------------- */

/* Functions for iterating through an ordered set.  */

/* Type of an iterator that traverses an ordered set.
   This is a fixed-size struct, so that creation of an iterator doesn't need
   memory allocation on the heap.  */
typedef struct
{
  /* For fast dispatch of gl_oset_iterator_next.  */
  const struct gl_oset_implementation *vtable;
  /* For detecting whether the last returned element was removed.  */
  gl_oset_t set;
  size_t count;
  /* Other, implementation-private fields.  */
  void *p; void *q;
  size_t i; size_t j;
} gl_oset_iterator_t;

#if 0 /* These are defined inline below.  */

/* Creates an iterator traversing an ordered set.
   The set's contents must not be modified while the iterator is in use,
   except for removing the last returned element.  */
extern gl_oset_iterator_t gl_oset_iterator (gl_oset_t set);

/* Creates an iterator traversing the tail of an ordered set, that comprises
   the elements that compare greater or equal to the given THRESHOLD.  The
   representation of the THRESHOLD is defined by the THRESHOLD_FN.  */
extern gl_oset_iterator_t gl_oset_iterator_atleast (gl_oset_t set,
                                                    gl_setelement_threshold_fn threshold_fn,
                                                    const void *threshold);

/* If there is a next element, stores the next element in *ELTP, advances the
   iterator and returns true.  Otherwise, returns false.  */
extern bool gl_oset_iterator_next (gl_oset_iterator_t *iterator,
                                   const void **eltp);

/* Frees an iterator.  */
extern void gl_oset_iterator_free (gl_oset_iterator_t *iterator);

#endif /* End of inline functions.  */

/* ------------------------ Implementation Details ------------------------ */

struct gl_oset_implementation
{
  /* gl_oset_t functions.  */
  gl_oset_t (*nx_create_empty) (gl_oset_implementation_t implementation,
                                gl_setelement_compar_fn compar_fn,
                                gl_setelement_dispose_fn dispose_fn);
  size_t (*size) (gl_oset_t set);
  bool (*search) (gl_oset_t set, const void *elt);
  bool (*search_atleast) (gl_oset_t set,
                          gl_setelement_threshold_fn threshold_fn,
                          const void *threshold, const void **eltp);
  int (*nx_add) (gl_oset_t set, const void *elt);
  bool (*remove_elt) (gl_oset_t set, const void *elt);
  int (*update) (gl_oset_t set, const void *elt,
                 void (*action) (const void * /*elt*/, void * /*action_data*/),
                 void *action_data);
  void (*oset_free) (gl_oset_t set);
  /* gl_oset_iterator_t functions.  */
  gl_oset_iterator_t (*iterator) (gl_oset_t set);
  gl_oset_iterator_t (*iterator_atleast) (gl_oset_t set,
                                          gl_setelement_threshold_fn threshold_fn,
                                          const void *threshold);
  bool (*iterator_next) (gl_oset_iterator_t *iterator, const void **eltp);
  void (*iterator_free) (gl_oset_iterator_t *iterator);
};

struct gl_oset_impl_base
{
  const struct gl_oset_implementation *vtable;
  gl_setelement_compar_fn compar_fn;
  gl_setelement_dispose_fn dispose_fn;
};

/* Define all functions of this file as accesses to the
   struct gl_oset_implementation.  */

GL_OSET_INLINE gl_oset_t
gl_oset_nx_create_empty (gl_oset_implementation_t implementation,
                         gl_setelement_compar_fn compar_fn,
                         gl_setelement_dispose_fn dispose_fn)
{
  return implementation->nx_create_empty (implementation, compar_fn,
                                          dispose_fn);
}

GL_OSET_INLINE size_t
gl_oset_size (gl_oset_t set)
{
  return ((const struct gl_oset_impl_base *) set)->vtable->size (set);
}

GL_OSET_INLINE bool
gl_oset_search (gl_oset_t set, const void *elt)
{
  return ((const struct gl_oset_impl_base *) set)->vtable->search (set, elt);
}

GL_OSET_INLINE bool
gl_oset_search_atleast (gl_oset_t set,
                        gl_setelement_threshold_fn threshold_fn,
                        const void *threshold, const void **eltp)
{
  return ((const struct gl_oset_impl_base *) set)->vtable
         ->search_atleast (set, threshold_fn, threshold, eltp);
}

GL_OSET_INLINE _GL_ATTRIBUTE_NODISCARD int
gl_oset_nx_add (gl_oset_t set, const void *elt)
{
  return ((const struct gl_oset_impl_base *) set)->vtable->nx_add (set, elt);
}

GL_OSET_INLINE bool
gl_oset_remove (gl_oset_t set, const void *elt)
{
  return ((const struct gl_oset_impl_base *) set)->vtable
         ->remove_elt (set, elt);
}

GL_OSET_INLINE int
gl_oset_update (gl_oset_t set, const void *elt,
                void (*action) (const void * /*elt*/, void * /*action_data*/),
                void *action_data)
{
  return ((const struct gl_oset_impl_base *) set)->vtable
         ->update (set, elt, action, action_data);
}

GL_OSET_INLINE void
gl_oset_free (gl_oset_t set)
{
  ((const struct gl_oset_impl_base *) set)->vtable->oset_free (set);
}

GL_OSET_INLINE gl_oset_iterator_t
gl_oset_iterator (gl_oset_t set)
{
  return ((const struct gl_oset_impl_base *) set)->vtable->iterator (set);
}

GL_OSET_INLINE gl_oset_iterator_t
gl_oset_iterator_atleast (gl_oset_t set,
                          gl_setelement_threshold_fn threshold_fn,
                          const void *threshold)
{
  return ((const struct gl_oset_impl_base *) set)->vtable
         ->iterator_atleast (set, threshold_fn, threshold);
}

GL_OSET_INLINE bool
gl_oset_iterator_next (gl_oset_iterator_t *iterator, const void **eltp)
{
  return iterator->vtable->iterator_next (iterator, eltp);
}

GL_OSET_INLINE void
gl_oset_iterator_free (gl_oset_iterator_t *iterator)
{
  iterator->vtable->iterator_free (iterator);
}

#ifdef __cplusplus
}
#endif

_GL_INLINE_HEADER_END

#endif /* _GL_OSET_H */
