/*****************************************************************************
 *                                                                           *
 *          UNURAN -- Universal Non-Uniform Random number generator          *
 *                                                                           *
 *****************************************************************************
 *                                                                           *
 *   FILE: slist.h                                                           *
 *                                                                           *
 *   PURPOSE:                                                                *
 *         defines function prototypes for simple list                       *
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
#ifndef SLIST_H_SEEN
#define SLIST_H_SEEN
/*---------------------------------------------------------------------------*/
/* Not part of manual!                                                       */
/*---------------------------------------------------------------------------*/
/*                                                                           */
/* A simple list can be used to store an arbitrary numbers of pointers       */
/* to allocated memory in a list.                                            */
/*                                                                           */
/* IMPORTANT: These elements must be allocated via (c|m|re)alloc()!!         */
/*                                                                           */
/*---------------------------------------------------------------------------*/

struct unur_slist *_unur_slist_new( void );
/*---------------------------------------------------------------------------*/
/* Make new simple list.                                                     */
/*---------------------------------------------------------------------------*/

int _unur_slist_append( struct unur_slist *slist, void *element );
/*---------------------------------------------------------------------------*/
/* Append pointer to element to simple list.                                 */
/*---------------------------------------------------------------------------*/

int _unur_slist_length( const struct unur_slist *slist );
/*---------------------------------------------------------------------------*/
/* Get length if list (number of list entries).                              */
/*---------------------------------------------------------------------------*/

void *_unur_slist_get( const struct unur_slist *slist, int n );
/*---------------------------------------------------------------------------*/
/* Get pointer to n-th element.                                              */
/*---------------------------------------------------------------------------*/

void *_unur_slist_replace( struct unur_slist *slist, int n, void *element );
/*---------------------------------------------------------------------------*/
/* Replace (existing) pointer to n-th element by 'element'.                  */
/*---------------------------------------------------------------------------*/

void _unur_slist_free( struct unur_slist *slist );
/*---------------------------------------------------------------------------*/
/* Free all elements and list in simple list.                                */
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
#endif  /* SLIST_H_SEEN */
/*---------------------------------------------------------------------------*/
