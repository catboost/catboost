/*****************************************************************************
 *                                                                           *
 *          UNURAN -- Universal Non-Uniform Random number generator          *
 *                                                                           *
 *****************************************************************************
 *                                                                           *
 *   FILE: matr.h                                                            *
 *                                                                           *
 *   PURPOSE:                                                                *
 *         function prototypes for manipulating distribution objects of      *
 *         type  MATR  (matrix distribution)                                 *
 *                                                                           *
 *   USAGE:                                                                  *
 *         only included in unuran.h                                         *
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

/* 
   =NODEX   MATR   MATRix distributions

   =UP Distribution_objects [45]

   =DESCRIPTION
      Distributions for random matrices. Notice that UNU.RAN uses
      arrays of @code{double}s to handle matrices. The rows of
      the matrix are stored consecutively.

   =END
*/

/*---------------------------------------------------------------------------*/

/* 
   Routines for handling matrix distributions (MATR).
*/

/* =ROUTINES */

UNUR_DISTR *unur_distr_matr_new( int n_rows, int n_cols );
/* 
   Create a new (empty) object for a matrix distribution. @var{n_rows}
   and @var{n_cols} are the respective numbers of rows and columns of
   the random matrix (i.e. its dimensions). It is also possible to 
   have only one number or rows and/or columns.
   Notice, however, that this is treated as a distribution of random 
   matrices with only one row or column or component and not as a
   distribution of vectors or real numbers. For the latter 
   unur_distr_cont_new() or unur_distr_cvec_new() should be
   used to create an object for a univariate distribution and a
   multivariate (vector) distribution, respectively.
*/

/* ==DOC
   @subsubheading Essential parameters
*/

int unur_distr_matr_get_dim( const UNUR_DISTR *distribution, int *n_rows, int *n_cols );
/* 
   Get number of rows and columns of random matrix (its dimension).
   It returns the total number of components. If successfull
   @code{UNUR_SUCCESS} is returned.
*/

/* =END */


/*---------------------------------------------------------------------------*/

/* not implemented: */
/* DOC
   @subsubheading Derived parameters

   The following paramters @strong{must} be set whenever one of the
   essential parameters has been set or changed (and the parameter is
   required for the chosen method).
*/

/*---------------------------------------------------------------------------*/
