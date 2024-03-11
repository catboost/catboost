/*****************************************************************************
 *                                                                           *
 *          UNURAN -- Universal Non-Uniform Random number generator          *
 *                                                                           *
 *****************************************************************************
 *                                                                           *
 *   FILE: mvstd.h                                                           *
 *                                                                           *
 *   PURPOSE:                                                                *
 *         function prototypes for method MVSTD                              *
 *         (wrapper for special generators for                               *
 *         MultiVariate continuous STandarD distributions)                   *
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
   =METHOD  MVSTD   MultiVariate continuous STandarD distributions

   =UP  Methods_for_CVEC

   =REQUIRED standard distribution from UNU.RAN library
      (@pxref{Stddist,,Standard distributions}).

   =SPEED depends on distribution and generator

   =REINIT supported

   =DESCRIPTION
      MVSTD is a wrapper for special generators for multivariate
      continuous standard distributions. It only works for
      distributions in the UNU.RAN library of standard distributions
      (@pxref{Stddist,,Standard distributions}).
      If a distribution object is provided that is build from scratch,
      or if no special generator for the given standard distribution is
      provided, the NULL pointer is returned.

   =HOWTOUSE
      Create a distribution object for a standard distribution
      from the UNU.RAN library (@pxref{Stddist,,Standard distributions}).
      
      Sampling from truncated distributions (which can be constructed by 
      changing the default domain of a distribution by means of
      unur_distr_cvec_set_domain_rect() call) is not possible.
   
      It is possible to change the parameters and the domain of the chosen 
      distribution and run unur_reinit() to reinitialize the generator object.

   =END
*/

/*---------------------------------------------------------------------------*/
/* Routines for user interface                                               */

/* =ROUTINES */

UNUR_PAR *unur_mvstd_new( const UNUR_DISTR *distribution );
/* 
   Get default parameters for new generator. It requires a distribution object 
   for a multivariate continuous distribution from the 
   UNU.RAN library of standard distributions 
   (@pxref{Stddist,,Standard distributions}).
   Using a truncated distribution is not possible.
*/

/* =END */
/*---------------------------------------------------------------------------*/
