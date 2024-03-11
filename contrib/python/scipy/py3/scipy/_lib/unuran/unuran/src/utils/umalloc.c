/*****************************************************************************
 *                                                                           *
 *          UNURAN -- Universal Non-Uniform Random number generator          *
 *                                                                           *
 *****************************************************************************
 *                                                                           *
 *   FILE: umalloc.c                                                         *
 *                                                                           *
 *   allocate memory                                                         *
 *                                                                           *
 *****************************************************************************
 *                                                                           *
 *   Copyright (c) 2000-2022 Wolfgang Hoermann and Josef Leydold             *
 *   Department of Statistics and Mathematics, WU Wien, Austria              *
 *   SPDX-License-Identifier: BSD-3-Clause                                   *
 *                                                                           *

 *                                                                           *
 *****************************************************************************/

/*---------------------------------------------------------------------------*/

#include <unur_source.h>

#ifdef R_UNURAN
#error #include <R_ext/Error.h>
#endif

/*---------------------------------------------------------------------------*/

void*
_unur_xmalloc(size_t size)
     /*----------------------------------------------------------------------*/
     /* allocate memory                                                      */
     /*                                                                      */
     /* parameters:                                                          */
     /*   size ... size of allocated block                                   */
     /*                                                                      */
     /* error:                                                               */
     /*   abort program                                                      */
     /*----------------------------------------------------------------------*/
{
  register void *ptr;

  /* allocate memory */
  ptr = malloc( size );

  /* successful ? */
  if (ptr == NULL) {
    _unur_error(NULL,UNUR_ERR_MALLOC,"");
#ifdef R_UNURAN
    error("memory exhausted");
#else
    exit (EXIT_FAILURE);
#endif
  }

  return ptr;

} /* end of _unur_xmalloc() */

/*---------------------------------------------------------------------------*/

void*
_unur_xrealloc(void *ptr, size_t size)
     /*----------------------------------------------------------------------*/
     /* reallocate memory                                                    */
     /*                                                                      */
     /* parameters:                                                          */
     /*   ptr  ... address of memory block previously allocated by malloc.   */
     /*   size ... size of reallocated block                                 */
     /*                                                                      */
     /* error:                                                               */
     /*   abort program                                                      */
     /*----------------------------------------------------------------------*/
{
  register void *new_ptr;

  /* reallocate memory */
  
  new_ptr = realloc( ptr, size );

  /* successful ? */
  if (new_ptr == NULL) {
    _unur_error(NULL,UNUR_ERR_MALLOC,"");
#ifdef R_UNURAN
    error("memory exhausted");
#else
    exit (EXIT_FAILURE);
#endif
  }

  return new_ptr;

} /* end of _unur_xrealloc() */

/*---------------------------------------------------------------------------*/
