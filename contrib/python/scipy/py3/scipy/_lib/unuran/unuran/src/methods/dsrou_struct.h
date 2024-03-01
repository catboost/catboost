/*****************************************************************************
 *                                                                           *
 *          UNURAN -- Universal Non-Uniform Random number generator          *
 *                                                                           *
 *****************************************************************************
 *                                                                           *
 *   FILE: dsrou_struct.h                                                    *
 *                                                                           *
 *   PURPOSE:                                                                *
 *         declares structures for method DSROU                              *
 *         (Discrete Simple universal generator, Ratio-Of-Uniforms method)   *
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

struct unur_dsrou_par { 
  double  Fmode;             /* cdf at mode                                  */
};

/*---------------------------------------------------------------------------*/
/* The generator object                                                      */

struct unur_dsrou_gen { 
  double  ul, ur;            /* height of rectangles                         */
  double  al, ar;            /* area of rectangles                           */
  double  Fmode;             /* cdf at mode                                  */
};

/*---------------------------------------------------------------------------*/
