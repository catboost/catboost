/*****************************************************************************
 *                                                                           *
 *          UNURAN -- Universal Non-Uniform Random number generator          *
 *                                                                           *
 *****************************************************************************
 *                                                                           *
 *   FILE: vempk_struct.h                                                    *
 *                                                                           *
 *   PURPOSE:                                                                *
 *         declares structures for method VEMPK                              *
 *         ((Vector) EMPirical distribution with Kernel smoothing)           *
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

struct unur_vempk_par {
  /* the observed sample is stored in the distribution object */
  double  smoothing;   /* determines how "smooth" the estimated density will be */
};

/*---------------------------------------------------------------------------*/
/* The generator object                                                      */

struct unur_vempk_gen {
  double *observ;      /* pointer to the array of the observations           */
  int     n_observ;    /* number of observations                             */
  int     dim;         /* dimension of distribution                          */

  UNUR_GEN *kerngen;   /* random variate generator for kernel                */

  double  smoothing;   /* determines how "smooth" the estimated density will be */

  double  hopt;        /* for bandwidth selection                            */
  double  hact;        /* actually used value for bandwith                   */
  double  corfac;      /* correction for variance correction                 */
  double *xbar;        /* mean vector of sample, for variance correction     */
};

/*---------------------------------------------------------------------------*/

