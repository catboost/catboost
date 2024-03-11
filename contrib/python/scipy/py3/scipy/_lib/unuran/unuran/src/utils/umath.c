/*****************************************************************************
 *                                                                           *
 *          UNURAN -- Universal Non-Uniform Random number generator          *
 *                                                                           *
 *****************************************************************************
 *                                                                           *
 *   FILE: umath.c                                                           *
 *                                                                           *
 *   miscelleanous mathematical routines                                     *
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

#include <unur_source.h>

/*---------------------------------------------------------------------------*/

/* If the macro INFINITY is not already defined we store infinity in the     */
/* global variable INFINITY.                                                 */
#ifndef INFINITY

#  if defined(HAVE_DECL_HUGE_VAL)

      const double INFINITY = HUGE_VAL;

#  elif defined(HAVE_DIVIDE_BY_ZERO)

      const double INFINITY = 1.0 / 0.0;

#  elif defined(HAVE_DECL_DBL_MAX)

      const double INFINITY = DBL_MAX;

#  else

#     error
#     error +--------------------------------------------+
#     error ! Sorry, Cannot define INFINITY correctly!.. !
#     error ! Please contact <unuran@statmath.wu.ac.at>. !
#     error +--------------------------------------------+
#     error

#  endif

#endif

/*---------------------------------------------------------------------------*/

#define ARCMEAN_HARMONIC   (1.e3)   /* use harmonic mean when abs larger than this value */
#define ARCMEAN_ARITHMETIC (1.e-6)  /* use harmonic mean when abs larger than this value */

double
_unur_arcmean( double x0, double x1 )
     /*----------------------------------------------------------------------*/
     /* compute "arctan mean" of two numbers.                                */
     /*                                                                      */
     /* parameters:                                                          */
     /*   x0, x1 ... two numbers                                             */
     /*                                                                      */
     /* return:                                                              */
     /*   mean                                                               */
     /*                                                                      */
     /* comment:                                                             */
     /*   "arctan mean" = tan(0.5*(arctan(x0)+arctan(x1)))                   */
     /*                                                                      */
     /*   a combination of arithmetical mean (for x0 and x1 close to 0)      */
     /*   and the harmonic mean (for |x0| and |x1| large).                   */
     /*----------------------------------------------------------------------*/
{
  double a0,a1;
  double r;

  /* we need x0 < x1 */
  if (x0>x1) {double tmp = x0; x0=x1; x1=tmp;}

  if (x1 < -ARCMEAN_HARMONIC || x0 > ARCMEAN_HARMONIC)
    /* use harmonic mean */
    return (2./(1./x0 + 1./x1));

  a0 = (x0<=-UNUR_INFINITY) ? -M_PI/2. : atan(x0);
  a1 = (x1>= UNUR_INFINITY) ?  M_PI/2. : atan(x1);

  if (fabs(a0-a1) < ARCMEAN_ARITHMETIC)
    /* use arithmetic mean */
    r = 0.5*x0 + 0.5*x1;

  else
    /* use "arc mean" */
    r = tan((a0 + a1)/2.);

  return r;
} /* end of _unur_arcmean() */

/*---------------------------------------------------------------------------*/




