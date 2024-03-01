/*****************************************************************************
 *                                                                           *
 *          UNURAN -- Universal Non-Uniform Random number generator          *
 *                                                                           *
 *****************************************************************************
 *                                                                           *
 *   FILE: mcorr.h                                                           *
 *                                                                           *
 *   PURPOSE:                                                                *
 *         function prototypes for method MCORR                              *
 *         (Matrix - CORRelation matrix)                                     *
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

/* 
   =METHOD  MCORR   Random CORRelation matrix

   =UP  Methods_for_MATR

   =REQUIRED  Distribution object for random correlation matrix

   =OPTIONAL 

   =SPEED Set-up: fast,
          Sampling: depends on dimension

   =REINIT supported

   =REF  [DLa86: Sect.6.1; p.605] [MOa84]

   =DESCRIPTION
      MCORR generates a random correlation matrix
      (Pearson's correlation).
      Two methods are used:

      @enumerate 
      @item 
      When a random correlation matrix having given eigenvalues is
      sought, the method of Marsaglia and Olkin [MOa84] is used. 
      In this case, the correlation matrix @unurmath{R}
      is given as @unurmath{R=PDP'} where @unurmath{D} is a diagonal
      matrix containing the eigenvalues and @unurmath{P} is a random
      orthonormal matrix. In higher dimensions, the rounding-errors
      introduced in the previous matrix multiplications could lead
      to a non-symmetric correlation matrix. Therefore the symmetric 
      correlation matrix is computed as @unurmath{R=(PDP'+P'DP)/2}.

      @item 
      A matrix @unurmath{H} is generated where all rows are
      independent random vectors of unit length uniformly on a sphere.
      Then @unurmath{HH'} is a correlation matrix (and vice versa if
      @unurmath{HH'} is a correlation matrix then the rows of
      @unurmath{H} are random vectors on a sphere).

      @end enumerate

      Notice that due to round-off errors the generated matrices might
      not be positive definite in extremely rare cases
      (especially when the given eigenvalues are amost 0).

      There are many other possibilites (distributions) of sampling
      the random rows from a sphere. The chosen methods are simple but
      does not result in a uniform distriubution of the random
      correlation matrices.

      It only works with distribution objects of random correlation
      matrices (@pxref{correlation,,Random Correlation Matrix}).

   =HOWTOUSE
      Create a distibution object for random correlation matrices by a
      @code{unur_distr_correlation} call
      (@pxref{correlation,,Random Correlation Matrix}).
      
      When a correlation matrix with given eigenvalues should be
      generated, these eigenvalues can be set by a
      unur_mcorr_set_eigenvalues() call.
      
      Otherwise, a faster algorithm is used that generates
      correlation matrices with random eigenstructure.

      Notice that due to round-off errors,
      there is a (small) chance that the resulting matrix is
      not positive definite for a Cholesky decomposition algorithm,
      especially when the dimension of the distribution is high.

      It is possible to change the given eigenvalues using
      unur_mcorr_chg_eigenvalues() and run unur_reinit() to 
      reinitialize the generator object. 

   =END
*/

/*---------------------------------------------------------------------------*/
/* Routines for user interface                                               */

/* =ROUTINES */

UNUR_PAR *unur_mcorr_new( const UNUR_DISTR *distribution );
/* 
   Get default parameters for generator.
*/

int unur_mcorr_set_eigenvalues( UNUR_PAR *par, const double *eigenvalues );
/*
   Sets the (optional) eigenvalues of the correlation matrix.
   If set, then the Marsaglia and Olkin algorithm will be used 
   to generate random correlation matrices with given eigenvalues.

   Important: the given eigenvalues of the correlation matrix must be 
   strictly positive and sum to the dimension of the matrix.
   If non-positive eigenvalues are attempted, no eigenvalues are set
   and an error code is returned.
   In case, that their sum is different from the dimension, an implicit
   scaling to give the correct sum is performed. 
*/

int unur_mcorr_chg_eigenvalues( UNUR_GEN *gen, const double *eigenvalues );
/*
   Change the eigenvalues of the correlation matrix.
   One must run unur_reinit() to reinitialize the generator 
   object then.
*/

/* =END */
/*---------------------------------------------------------------------------*/


