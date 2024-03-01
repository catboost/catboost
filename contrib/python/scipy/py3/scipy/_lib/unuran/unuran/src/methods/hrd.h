/*****************************************************************************
 *                                                                           *
 *          UNURAN -- Universal Non-Uniform Random number generator          *
 *                                                                           *
 *****************************************************************************
 *                                                                           *
 *   FILE: hrd.h                                                             *
 *                                                                           *
 *   PURPOSE:                                                                *
 *         function prototypes for method HRD                                *
 *         (Hazard Rate Decreasing)                                          *
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
   =METHOD  HRD   Hazard Rate Decreasing

   =UP  Methods_for_CONT

   =REQUIRED decreasing (non-increasing) hazard rate 

   =SPEED Set-up: fast, Sampling: slow

   =REF  [HLD04: Sect.9.1.5, Alg.9.5]

   =REINIT supported

   =DESCRIPTION
      Generates random variate with given non-increasing hazard rate.
      It is necessary that the distribution object contains this
      hazard rate. Decreasing hazard rate implies that the
      corresponding PDF of the distribution has heavier tails than the
      exponential distribution (which has constant hazard rate).

   =HOWTOUSE
      HRD requires a hazard function for a continuous distribution
      with non-increasing hazard rate. There are no parameters for
      this method.

      It is important to note that the domain of the distribution can
      be set via a unur_distr_cont_set_domain() call. However, only
      the left hand boundary is used. For computational reasons the
      right hand boundary is always reset to @code{UNUR_INFINITY}.
      If no domain is given by the user then the left hand boundary is
      set to @code{0}.
      
      For distributions which do not have decreasing hazard rates but
      are bounded from above use method HRB
      (@pxref{HRB,,Hazard Rate Bounded}).
      For distributions with increasing hazard rate method HRI 
      (@pxref{HRI,,Hazard Rate Increasing}) is required.

      It is possible to change the parameters and the domain of the chosen 
      distribution and run unur_reinit() to reinitialize the generator object.

   =END
*/

/*---------------------------------------------------------------------------*/
/* Routines for user interface                                               */

/* =ROUTINES */

UNUR_PAR *unur_hrd_new( const UNUR_DISTR *distribution );
/* 
   Get default parameters for generator.
*/

/*...........................................................................*/

int unur_hrd_set_verify( UNUR_PAR *parameters, int verify );
/* */

int unur_hrd_chg_verify( UNUR_GEN *generator, int verify );
/* 
   Turn verifying of algorithm while sampling on/off.
   If the hazard rate is not bounded by the given bound, then
   @code{unur_errno} is set to @code{UNUR_ERR_GEN_CONDITION}. 

   Default is FALSE.
*/

/* =END */
/*---------------------------------------------------------------------------*/
