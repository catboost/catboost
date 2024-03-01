/*****************************************************************************
 *                                                                           *
 *          UNURAN -- Universal Non-Uniform Random number generator          *
 *                                                                           *
 *****************************************************************************
 *                                                                           *
 *   FILE: gibbs_struct.h                                                    *
 *                                                                           *
 *   PURPOSE:                                                                *
 *         declares structures for method GIBBS                              *
 *         (Markov Chain - GIBBS sampler)                                    *
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

struct unur_gibbs_par { 
  int thinning;             /* thinning factor for generated chain           */
  int burnin;               /* length of burn-in for chain                   */
  double  c_T;              /* parameter c for transformation T_c            */
  const double *x0;         /* starting point of chain                       */
};

/*---------------------------------------------------------------------------*/
/* The generator object                                                      */

struct unur_gibbs_gen {
  int dim;                  /* dimension of distribution                     */
  int thinning;             /* thinning factor for generated chain           */
  double  c_T;              /* parameter c for transformation T_c            */

  double *state;            /* state of chain / current point                */

  struct unur_distr *distr_condi; /* conditional distribution                */

  int    coord;             /* current coordinate used for GIBBS chain       */
  double *direction;        /* working array for random direction            */

  int burnin;               /* length of burn-in for chain                   */
  double *x0;               /* starting point of chain                       */
};

/*---------------------------------------------------------------------------*/

