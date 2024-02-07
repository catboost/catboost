/*****************************************************************************
 *                                                                           *
 *          UNURAN -- Universal Non-Uniform Random number generator          *
 *                                                                           *
 *****************************************************************************
 *                                                                           *
 *   FILE: dstd_struct.h                                                     *
 *                                                                           *
 *   PURPOSE:                                                                *
 *         declares structures for method DSTD                               *
 *         (wrapper for special generators for                               *
 *         Discrete STanDard distributions)                                  *
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

struct unur_dstd_par { 
  int dummy;              /* no special parameters                           */
};

/*---------------------------------------------------------------------------*/
/* The generator object                                                      */

struct unur_dstd_gen { 
  double *gen_param;      /* parameters for the generator                    */
  int     n_gen_param;    /* number of parameters for the generator          */

  int    *gen_iparam;     /* integer parameters for generator                */
  int     n_gen_iparam;   /* number of integer parameters for the generator  */

  double  Umin;           /* cdf at left boundary of domain                  */
  double  Umax;           /* cdf at right boundary of domain                 */
  int  is_inversion;      /* indicate whether method is inversion method     */     
  const char *sample_routine_name; /* name of sampling routine               */
};

/*---------------------------------------------------------------------------*/
