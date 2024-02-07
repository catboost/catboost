/*****************************************************************************
 *                                                                           *
 *          UNURAN -- Universal Non-Uniform Random number generator          *
 *                                                                           *
 *****************************************************************************
 *                                                                           *
 *   FILE: ninv_struct.h                                                     *
 *                                                                           *
 *   PURPOSE:                                                                *
 *         declares structures for method NINV                               *
 *         (Numerical INVersion of cumulative distribution function)         *
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

struct unur_ninv_par { 
  int     max_iter;          /* maximal number of iterations                 */
  double  x_resolution;      /* maximal tolerated relative x-error           */
  double  u_resolution;      /* maximal tolerated (absolute) u-error         */
  double  s[2];              /* interval boundaries at start (left/right)    */
  int     table_on;          /* if TRUE a table for starting points is used  */
  int     table_size;        /* size of table                                */
};

/*---------------------------------------------------------------------------*/
/* The generator object                                                      */

struct unur_ninv_gen { 
  int     max_iter;          /* maximal number of iterations                 */
  double  x_resolution;      /* maximal tolerated relative x-error           */
  double  u_resolution;      /* maximal tolerated (absolute) u-error         */
  double *table;             /* table with possible starting values for NINV */
  double *f_table;	     /* function values of points stored in table    */
  int     table_on;          /* if TRUE a table for starting points is used  */
  int     table_size;        /* size of table                                */
  double  Umin, Umax;        /* bounds for iid random variable in respect to
                                the given (truncated) domain of the distr.   */
  double  CDFmin, CDFmax;    /* CDF-bounds of domain                         */
  double  s[2];              /* interval boundaries at start (left/right) ...*/
  double  CDFs[2];           /* ... and their CDF-values                     */
};

/*---------------------------------------------------------------------------*/
























