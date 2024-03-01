/*****************************************************************************
 *                                                                           *
 *          UNURAN -- Universal Non-Uniform Random number generator          *
 *                                                                           *
 *****************************************************************************
 *                                                                           *
 *   FILE: unur_source.h                                                     *
 *                                                                           *
 *   macros for calling and resetting uniform random number generators       *
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
#ifndef URNG_SOURCE_H_SEEN
#define URNG_SOURCE_H_SEEN
/*---------------------------------------------------------------------------*/
#ifdef UNUR_URNG_UNURAN
/*---------------------------------------------------------------------------*/

/* function call to uniform RNG */
#define _unur_call_urng(urng)    ((urng)->sampleunif((urng)->state))

/*---------------------------------------------------------------------------*/
#else
/*---------------------------------------------------------------------------*/
#error
#error UNUR_URNG changed!
#error
#error Define _unur_call_urng(urng) and _unur_call_reset(urng) in
#error file 'urng_source'
#error
/*---------------------------------------------------------------------------*/
/* The UNU.RAN API to uniform random number generator should be flexible     */
/* enough to use any source of uniform random number generators.             */
/*                                                                           */
/* Please, report any problem with this interface to this interface to the   */
/* UNU.RAN development team.                                                 */
/*                                                                           */
/* When the UNU.RAN interface to uniform random number generator must be     */
/* changed (not recommended), proceed as following:                          */
/*                                                                           */
/* (1) Change the typedef for 'UNUR_URNG' in file 'src/unur_typedefs.h'.     */
/* (2) Comment out the #definition of 'UNUR_URNG_UNURAN' to remove all       */
/*     code from the UNU.RAN API.                                            */
/* (3) Comment out the above error directives.                               */
/* (4) Define  _unur_call_urng(urng) and _unur_call_reset(urng).             */
/*     Here is a template:                                                   */
/*                                                                           */
/*  -  function call to uniform RNG:                                         */
/*       #define _unur_call_urng(urng)  (my_uniform_generator(urng))         */
/*                                                                           */
/*  -  reset uniform RNG:                                                    */
/*       #define _unur_call_reset(urng) (my_reset_unif(urng),UNUR_SUCCESS)   */
/*     if no such reset call exists:                                         */
/*       #define _unur_call_reset(urng) (UNUR_FAILURE)                       */
/*                                                                           */
/* (5) Define defaults 'UNUR_URNG_DEFAULT' and 'UNUR_URNG_AUX_DEFAULT'       */
/*     in file 'src/unuran_config.h'.                                        */
/*                                                                           */
/* Notice that some test routines (like counting URNGs) and the test suite   */
/* do not work for a non-UNU.RAN interface.                                  */
/*---------------------------------------------------------------------------*/
#endif  /* UNUR_URNG_UNURAN */
/*---------------------------------------------------------------------------*/
#endif  /* #ifndef URNG_SOURCE_H_SEEN */
/*---------------------------------------------------------------------------*/

