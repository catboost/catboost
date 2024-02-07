/*****************************************************************************
 *                                                                           *
 *          UNURAN -- Universal Non-Uniform Random number generator          *
 *                                                                           *
 *****************************************************************************
 *                                                                           *
 *   FILE: empl_struct.h                                                     *
 *                                                                           *
 *   PURPOSE:                                                                *
 *         declares structures for method EMPL                               *
 *         (EMPirical distribution with Linear interpolation)                *
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

struct unur_empl_par {
  /* the observed sample is stored in the distribution object */
  int dummy;
};

/*---------------------------------------------------------------------------*/
/* The generator object                                                      */

struct unur_empl_gen {
  double *observ;      /* pointer to the array of the observations           */
  int     n_observ;    /* number of observations                             */
};

/*---------------------------------------------------------------------------*/


