/*****************************************************************************
 *                                                                           *
 *          UNURAN -- Universal Non-Uniform Random number generator          *
 *                                                                           *
 *****************************************************************************
 *                                                                           *
 *   FILE: srou_struct.h                                                     *
 *                                                                           *
 *   PURPOSE:                                                                *
 *         declares structures for method SROU                               *
 *         (Simple universal generator, Ratio-Of-Uniforms method)            *
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
/* Information for constructing the generator                                */

struct unur_srou_par { 
  double  r;                 /* parameter for power transformation           */
  double  Fmode;             /* cdf at mode                                  */
  double  um;                /* square root of pdf at mode                   */
};

/*---------------------------------------------------------------------------*/
/* The generator object                                                      */

struct unur_srou_gen { 
  double  um;                /* height of rectangle: square root of f(mode)  */
  double  vl, vr;            /* left and right boundary of rectangle         */
  double  xl, xr;            /* ratios vl/um and vr/um                       */
  double  Fmode;             /* cdf at mode                                  */

  /* parameters for generalized SROU                                         */
  double  r;                 /* parameter for power transformation           */
  double  p;                 /* construction point for bounding curve        */
  double  a, b;              /* parameters for bounding curve                */
  double  log_ab;            /* parameter for bounding curve                 */
};

/*---------------------------------------------------------------------------*/
