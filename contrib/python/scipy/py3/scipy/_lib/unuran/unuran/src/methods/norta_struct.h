/*****************************************************************************
 *                                                                           *
 *          UNURAN -- Universal Non-Uniform Random number generator          *
 *                                                                           *
 *****************************************************************************
 *                                                                           *
 *   FILE: norta_struct.h                                                    *
 *                                                                           *
 *   PURPOSE:                                                                *
 *         declares structures for method NORTA                              *
 *         (NORmal To Anything)                                              *
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

struct unur_norta_par { 
  int dummy;
};

/*---------------------------------------------------------------------------*/
/* The generator object                                                      */

struct unur_norta_gen { 
  int    dim;                          /* dimension of distribution          */
  double *copula;                      /* pointer to intermediate copula     */
  struct unur_distr *normaldistr;      /* standard normal distribution       */
  struct unur_gen **marginalgen_list;  /* list of generators for marginal distributions */

  /* Remark: We use gen->gen_aux to store the pointer to the                 */
  /*         multinormal generator.                                          */
  /*         It is accessed via the macro 'MNORMAL'.                         */
};

/*---------------------------------------------------------------------------*/

