/*****************************************************************************
 *                                                                           *
 *          UNURAN -- Universal Non-Uniform Random number generator          *
 *                                                                           *
 *****************************************************************************
 *                                                                           *
 *   FILE: dau.h                                                             *
 *                                                                           *
 *   PURPOSE:                                                                *
 *         function prototypes for method DAU                                *
 *         ((Discrete) Alias-Urn)                                            *
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
   =METHOD  DAU  (Discrete) Alias-Urn method

   =UP  Methods_for_DISCR

   =REQUIRED probability vector (PV)

   =SPEED Set-up: slow (linear with the vector-length), Sampling: very fast

   =REINIT supported

   =REF  [WAa77] [HLD04: Sect.3.2]

   =DESCRIPTION
      DAU samples from distributions with arbitrary but finite
      probability vectors (PV) of length @i{N}.
      The algorithmus is based on an ingeneous method by A.J. Walker
      and requires a table of size (at least) @i{N}.
      It needs one random numbers and only one comparison for each
      generated random variate. The setup time for constructing the
      tables is @i{O(N)}.
      
      By default the probability vector is indexed starting at
      @code{0}. However this can be changed in the distribution object by
      a unur_distr_discr_set_domain() call.

      The method also works when no probability vector but a PMF is
      given. However then additionally a bounded (not too large) domain
      must be given or the sum over the PMF (see
      unur_distr_discr_make_pv() for details).

   =HOWTOUSE
      Create an object for a discrete distribution either by setting a
      probability vector or a PMF. The performance can be slightly
      influenced by setting the size of the used table which can be
      changed by unur_dau_set_urnfactor().

      It is possible to change the parameters and the domain of the chosen 
      distribution and run unur_reinit() to reinitialize the generator object.
   =END
*/

/*---------------------------------------------------------------------------*/
/* Routines for user interface                                               */

/* =ROUTINES */

UNUR_PAR *unur_dau_new( const UNUR_DISTR *distribution );
/* 
   Get default parameters for generator.
*/

/*...........................................................................*/

int unur_dau_set_urnfactor( UNUR_PAR *parameters, double factor );
/* 
   Set size of urn table relative to length of the probability
   vector. It must not be less than 1. Larger tables result in
   (slightly) faster generation times but require a more expensive
   setup. However sizes larger than 2 are not recommended.

   Default is @code{1}.
*/

/* =END */

/*---------------------------------------------------------------------------*/










