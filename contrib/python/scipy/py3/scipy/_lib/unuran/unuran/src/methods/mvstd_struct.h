/*****************************************************************************
 *                                                                           *
 *          UNURAN -- Universal Non-Uniform Random number generator          *
 *                                                                           *
 *****************************************************************************
 *                                                                           *
 *   FILE: mvstd_struct.h                                                     *
 *                                                                           *
 *   PURPOSE:                                                                *
 *         declares structures for method MVSTD                              *
 *         (wrapper for special generators for                               *
 *         MultiVariate continuous STandarD distributions)                   *
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

struct unur_mvstd_par { 
  int dummy;              /* no special parameters                           */
};

/*---------------------------------------------------------------------------*/
/* The generator object                                                      */

struct unur_mvstd_gen { 
  const char *sample_routine_name; /* name of sampling routine               */
  /* Currently, there is no need to store constants and parameter for        */
  /* special generators. So it is not implemented yet.                       */
  /* (When added, do not forget to initialize, copy and free corresponding   */
  /* pointers!)                                                              */
};

/*---------------------------------------------------------------------------*/
