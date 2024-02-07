/*****************************************************************************
 *                                                                           *
 *          UNURAN -- Universal Non-Uniform Random number generator          *
 *                                                                           *
 *****************************************************************************
 *                                                                           *
 *   FILE: hri.h                                                             *
 *                                                                           *
 *   PURPOSE:                                                                *
 *         function prototypes for method HRI                                *
 *         (Hazard Rate Increasing)                                          *
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
   =METHOD  HRI   Hazard Rate Increasing

   =UP  Methods_for_CONT

   =REQUIRED increasing (non-decreasing) hazard rate 

   =SPEED Set-up: fast, Sampling: slow

   =REINIT supported

   =REF  [HLD04: Sect.9.1.6, Alg.9.6]

   =DESCRIPTION
      Generates random variate with given non-increasing hazard rate. 
      It is necessary that the distribution object contains this hazard rate.
      Increasing hazard rate implies that the corresponding PDF of the
      distribution has heavier tails than the exponential distribution
      (which has constant hazard rate). 

      The method uses a decomposition of the hazard rate into a main
      part which is constant for all @i{x} beyond some point @i{p0}
      and a remaining part. From both of these parts points are
      sampled using the thinning method and the minimum of both is
      returned. Sampling from the first part is easier as we have a
      constant dominating hazard rate. Thus @i{p0} should be large. On
      the other hand, if @i{p0} is large than the thinning algorithm
      needs many iteration. Thus the performance of the the algorithm
      deponds on the choice of @i{p0}. We found that values close to
      the expectation of the generated distribution result in good
      performance.

   =HOWTOUSE
      HRI requires a hazard function for a continuous distribution
      with non-decreasing hazard rate. 
      The parameter @i{p0} should be set to a value close to the
      expectation of the required distribution using
      unur_hri_set_p0(). If performance is crucial one may try other
      values as well.

      It is important to note that the domain of the distribution can
      be set via a unur_distr_cont_set_domain() call. However, only
      the left hand boundary is used. For computational reasons the
      right hand boundary is always reset to @code{UNUR_INFINITY}.
      If no domain is given by the user then the left hand boundary is
      set to @code{0}.
      
      For distributions with decreasing hazard rate method HRD
      (@pxref{HRI,,Hazard Rate Decreasing}) is required.
      For distributions which do not have increasing or decreasing
      hazard rates but are bounded from above use method HRB
      (@pxref{HRB,,Hazard Rate Bounded}).

      It is possible to change the parameters and the domain of the chosen 
      distribution and run unur_reinit() to reinitialize the generator object.


      Notice, that the upper bound given by the unur_hrb_set_upperbound() call
      cannot be changed and must be valid for the changed distribution.
      Notice that the parameter @i{p0} which has been set by a unur_hri_set_p0()
      call cannot be changed and must be valid for the changed distribution.

   =END
*/

/*---------------------------------------------------------------------------*/
/* Routines for user interface                                               */

/* =ROUTINES */

UNUR_PAR *unur_hri_new( const UNUR_DISTR *distribution );
/* 
   Get default parameters for generator.
*/

/*...........................................................................*/

int unur_hri_set_p0( UNUR_PAR *parameters, double p0 );
/* 
   Set design point for algorithm. It is used to split the domain of the 
   distribution. Values for @var{p0} close to the expectation of the
   distribution results in a relatively good performance of the algorithm.
   It is important that the hazard rate at this point must be greater
   than @code{0} and less than @code{UNUR_INFINITY}.
   
   Default: left boundary of domain + @code{1.}
*/

int unur_hri_set_verify( UNUR_PAR *parameters, int verify );
/* */

int unur_hri_chg_verify( UNUR_GEN *generator, int verify );
/* 
   Turn verifying of algorithm while sampling on/off.
   If the hazard rate is not bounded by the given bound, then
   @code{unur_errno} is set to @code{UNUR_ERR_GEN_CONDITION}. 

   Default is FALSE.
*/

/* =END */
/*---------------------------------------------------------------------------*/
