/*****************************************************************************
 *                                                                           *
 *          UNURAN -- Universal Non-Uniform Random number generator          *
 *                                                                           *
 *****************************************************************************
 *                                                                           *
 *   FILE: nrou_struct.h                                                     *
 *                                                                           *
 *   PURPOSE:                                                                *
 *         declares structures for method NROU                               *
 *         (Naive Ratio-Of-Uniforms method)                                  *
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

struct unur_nrou_par { 
  double  umin, umax;        /* u boundary for bounding rectangle            */
  double  vmax;              /* v boundary for bounding rectangle            */
  double  center;            /* center of distribution                       */
  double  r;		     /* r-parameter                                  */
};

/*---------------------------------------------------------------------------*/
/* The generator object                                                      */

struct unur_nrou_gen { 
  double  umin, umax;        /* u boundary for bounding rectangle            */
  double  vmax;              /* v boundary for bounding rectangle            */
  double  center;            /* center of distribution                       */
  double  r;		     /* r-parameter                                  */
};

/*---------------------------------------------------------------------------*/
