/*****************************************************************************
 *                                                                           *
 *          UNURAN -- Universal Non-Uniform Random number generator          *
 *                                                                           *
 *****************************************************************************
 *                                                                           *
 *   FILE: vnrou_struct.h                                                    *
 *                                                                           *
 *   PURPOSE:                                                                *
 *         declares structures for method VNROU                              *
 *         (Vector Naive Ratio Of Uniforms)                                  *
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

struct unur_vnrou_par { 
  double r;		    /* r-parameter of the vnrou method 	             */
  double *umin, *umax;      /* boundary rectangle u-coordinates              */
  double vmax;              /* boundary rectangle v-coordinate               */
};

/*---------------------------------------------------------------------------*/
/* The generator object                                                      */

struct unur_vnrou_gen { 
  int    dim;               /* dimension of distribution                     */
  double r;		    /* r-parameter of the vnrou method 	             */
  double *umin, *umax;      /* boundary rectangle u-coordinates              */
  double vmax;              /* boundary rectangle v-coordinate               */
  const double *center;     /* center of distribution                        */  
};

/*---------------------------------------------------------------------------*/

