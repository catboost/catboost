/*****************************************************************************
 *                                                                           *
 *          UNURAN -- Universal Non-Uniform Random number generator          *
 *                                                                           *
 *****************************************************************************
 *                                                                           *
 *   FILE: functparser_source.h                                              *
 *                                                                           *
 *   PURPOSE:                                                                *
 *         declares prototypes for function parser                           *
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
#ifndef FUNCTPARSER_SOURCE_H_SEEN
#define FUNCTPARSER_SOURCE_H_SEEN

/*---------------------------------------------------------------------------*/
/* Function prototypes for function string parser                            */
/*---------------------------------------------------------------------------*/

struct ftreenode *_unur_fstr2tree ( const char *functstring );
/*---------------------------------------------------------------------------*/
/* Compute funtion tree from string.                                         */
/*---------------------------------------------------------------------------*/

struct ftreenode *_unur_fstr2tree_DefFunct ( const char *functstring );
/*---------------------------------------------------------------------------*/
/* Compute funtion tree from string.                                         */
/* (Same as _unur_fstr2tree() but string must start with "f(x)=".            */
/*---------------------------------------------------------------------------*/

double _unur_fstr_eval_tree ( const struct ftreenode *functtree_root, double x );
/*---------------------------------------------------------------------------*/
/* Evalutes function given by a function tree at x.                          */
/*---------------------------------------------------------------------------*/

struct ftreenode *_unur_fstr_dup_tree (const struct ftreenode *functtree_root);
/*---------------------------------------------------------------------------*/
/* Duplicate function tree rooted at root.                                   */
/*---------------------------------------------------------------------------*/

void _unur_fstr_free ( struct ftreenode *functtree_root );
/*---------------------------------------------------------------------------*/
/* Destroys function tree and frees memory.                                  */
/*---------------------------------------------------------------------------*/

char *_unur_fstr_tree2string ( const struct ftreenode *functtree_root,
			       const char *variable, const char *function, int spaces );
/*---------------------------------------------------------------------------*/
/* Produce string from function tree.                                        */
/* It returns a pointer to the resulting string. This should be freed when   */
/* it is not used any more.                                                  */
/*---------------------------------------------------------------------------*/

struct ftreenode *_unur_fstr_make_derivative ( const struct ftreenode *functtree_root );
/*---------------------------------------------------------------------------*/
/* Make function tree for derivate of given function (tree).                 */ 
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
#endif   /* FUNCTPARSER_SOURCE_H_SEEN */
/*---------------------------------------------------------------------------*/

