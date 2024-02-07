/*****************************************************************************
 *                                                                           *
 *          UNURAN -- Universal Non-Uniform Random number generator          *
 *                                                                           *
 *****************************************************************************
 *                                                                           *
 *   FILE: functparser_struct.h                                              *
 *                                                                           *
 *   PURPOSE:                                                                *
 *         declares structures for function parser                           *
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
/* Structure for function tree                                               */

struct ftreenode { 
  char            *symbol;      /* name of token                             */
  int             token;        /* location of token in list of symbols      */
  int             type;         /* type of token (e.g. S_ADD_OP)             */
  double          val;          /* value of constant or (and)
				   value of node during evalution of tree    */
  struct ftreenode *left;       /* pointer to left branch/leave of node      */
  struct ftreenode *right;      /* pointer to right branch/leave of node     */

#ifdef UNUR_COOKIES
  unsigned cookie;              /* magic cookie                              */
#endif
}; 


/*---------------------------------------------------------------------------*/

