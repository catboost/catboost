/*****************************************************************************
 *                                                                           *
 *          UNURAN -- Universal Non-Uniform Random number generator          *
 *                                                                           *
 *****************************************************************************
 *                                                                           *
 *   FILE: slist_struct.h                                                    *
 *                                                                           *
 *   PURPOSE:                                                                *
 *         declares structures for simple list                               *
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
/* Store simple list.                                                        */

struct unur_slist {
  void **ptr;        /* pointer to array with pointers to list elements      */
  int n_ptr;         /* number of list elements                              */

#ifdef UNUR_COOKIES
  unsigned cookie;   /* magic cookie                                         */
#endif
};

/*---------------------------------------------------------------------------*/
