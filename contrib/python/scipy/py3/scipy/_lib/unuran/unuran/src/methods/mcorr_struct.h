/*****************************************************************************
 *                                                                           *
 *          UNURAN -- Universal Non-Uniform Random number generator          *
 *                                                                           *
 *****************************************************************************
 *                                                                           *
 *   FILE: mcorr_struct.h                                                    *
 *                                                                           *
 *   PURPOSE:                                                                *
 *         declares structures for method MCORR                              *
 *         (Matrix -- CORRelation matrix)                                    *
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

struct unur_mcorr_par { 
  int    dim;            /* dimension (number of rows and columns) of matrix */
  const double *eigenvalues;   /* optional eigenvalues of correlation matrix */
};

/*---------------------------------------------------------------------------*/
/* The generator object                                                      */

struct unur_mcorr_gen { 
  int    dim;            /* dimension (number of rows and columns) of matrix */
  double *H;             /* working array                                    */
  double *M;             /* working array                                    */
  double *eigenvalues;   /* optional eigenvalues of the correlation matrix   */
};

/*---------------------------------------------------------------------------*/

