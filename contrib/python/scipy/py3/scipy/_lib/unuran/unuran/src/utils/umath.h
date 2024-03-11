/*****************************************************************************
 *                                                                           *
 *          UNURAN -- Universal Non-Uniform Random number generator          *
 *                                                                           *
 *****************************************************************************
 *                                                                           *
 *   FILE: umath.h                                                           *
 *                                                                           *
 *   PURPOSE:                                                                *
 *         defines macros and function prototypes for miscelleanous          *
 *         mathematical routines                                             *
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
#ifndef MATH_H_SEEN
#define MATH_H_SEEN
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/

/* 
   =NODE Math Mathematics

   =UP Misc [0]

   =DESCRIPTION
      The following macros have been defined

      @ftable @code
      @item UNUR_INFINITY
      indicates infinity for floating point numbers (of type @code{double}).
      Internally @code{HUGE_VAL} is used.

      @item INT_MAX
      @itemx INT_MIN
      indicate infinity and minus infinity, resp., for integers
      (defined by ISO C standard).

      @item  TRUE
      @itemx FALSE
      bolean expression for return values of @code{set} functions.
      @end ftable

   =END
*/

/*---------------------------------------------------------------------------*/
/* Define INFINITY                                                           */
/* (we use the largest possible value to indicate infinity)                  */
#include <math.h>

#ifndef INFINITY
/* use a global variable to store infinity */
/* (definition in umath.c)                 */
extern const double INFINITY;
#endif

#define UNUR_INFINITY  (INFINITY)

/*---------------------------------------------------------------------------*/
/* True and false                                                            */

#ifndef TRUE
#define TRUE   (1)
#endif

#ifndef FALSE
#define FALSE  (0)
#endif

/*---------------------------------------------------------------------------*/
#endif  /* MATH_H_SEEN */
/*---------------------------------------------------------------------------*/






