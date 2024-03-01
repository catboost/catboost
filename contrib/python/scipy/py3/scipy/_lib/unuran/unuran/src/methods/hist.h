/*****************************************************************************
 *                                                                           *
 *          UNURAN -- Universal Non-Uniform Random number generator          *
 *                                                                           *
 *****************************************************************************
 *                                                                           *
 *   FILE: hist.h                                                            *
 *                                                                           *
 *   PURPOSE:                                                                *
 *         function prototypes for method HIST                               *
 *         (HISTogram of empirical distribution)                             *
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
   =METHOD  HIST   HISTogramm of empirical distribution

   =UP  Methods_for_CEMP

   =REQUIRED histogram 

   =SPEED Set-up: moderate,
          Sampling: fast

   =REINIT not implemented

   =DESCRIPTION
      Method HIST generates random variates from an empirical distribution
      that is given as histogram. Sampling is done using the inversion
      method. 

      If observed (raw) data are provided we recommend method EMPK
      (@pxref{EMPK,,EMPirical distribution with Kernel smoothing})
      instead of compting a histogram as this reduces information.

   =HOWTOUSE
      Method HIST uses empirical distributions that are given as a
      histgram. There are no optional parameters.

   =END
*/

/*---------------------------------------------------------------------------*/
/* Routines for user interface                                               */

/* =ROUTINES */

UNUR_PAR *unur_hist_new( const UNUR_DISTR *distribution );
/* 
   Get default parameters for generator.
*/

/*...........................................................................*/

/* =END */

/*---------------------------------------------------------------------------*/

