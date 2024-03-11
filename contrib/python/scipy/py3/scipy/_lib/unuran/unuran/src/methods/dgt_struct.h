/*****************************************************************************
 *                                                                           *
 *          UNURAN -- Universal Non-Uniform Random number generator          *
 *                                                                           *
 *****************************************************************************
 *                                                                           *
 *   FILE: dgt_struct.h                                                      *
 *                                                                           *
 *   PURPOSE:                                                                *
 *         declares structures for method DGT                                *
 *         (Discrete Guide Table (indexed search))                           *
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
/* Information for constructing the generator                                */

struct unur_dgt_par { 
  double  guide_factor; /* relative length of guide table.  (DEFAULT = 1)    */
                        /*   length of guide table = guide_factor * len      */
};

/*---------------------------------------------------------------------------*/
/* The generator object                                                      */

struct unur_dgt_gen { 
  double  sum;          /* sum of all probabilities = cumpv[len-1]           */
  double *cumpv;        /* pointer to the vector of cumulated probabilities  */
  int    *guide_table;  /* pointer to guide table                            */
  int     guide_size;   /* length of guide table                             */
  double  guide_factor; /* relative length of guide table.  (DEFAULT = 1)    */
                        /*   length of guide table = guide_factor * len      */
};

/*---------------------------------------------------------------------------*/
