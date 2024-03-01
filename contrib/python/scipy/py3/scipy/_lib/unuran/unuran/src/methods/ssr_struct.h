/*****************************************************************************
 *                                                                           *
 *          UNURAN -- Universal Non-Uniform Random number generator          *
 *                                                                           *
 *****************************************************************************
 *                                                                           *
 *   FILE: ssr_struct.h                                                      *
 *                                                                           *
 *   PURPOSE:                                                                *
 *         declares structures for method SSR                                *
 *         (Simple Setup, Rejection with universal bounds)                   *
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

struct unur_ssr_par { 
  double  Fmode;             /* cdf at mode                                  */
  double  fm;                /* pdf at mode                                  */
  double  um;                /* sqrt of pdf at mode                          */
};

/*---------------------------------------------------------------------------*/
/* The generator object                                                      */

struct unur_ssr_gen { 
  double  fm;                /* pdf at mode                                  */
  double  um;                /* sqrt of pdf at mode                          */
  double  vl, vr;            /* parameters for hat function                  */
  double  xl, xr;            /* partition points of hat                      */
  double  al, ar;            /* areas below hat in first and secont part     */
  double  A;                 /* area below hat                               */
  double  Aleft, Ain;        /* areas below hat in left tails and inside domain of pdf */
  double  Fmode;             /* cdf at mode                                  */
};

/*---------------------------------------------------------------------------*/
