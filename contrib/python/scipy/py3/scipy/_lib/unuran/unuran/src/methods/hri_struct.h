/*****************************************************************************
 *                                                                           *
 *          UNURAN -- Universal Non-Uniform Random number generator          *
 *                                                                           *
 *****************************************************************************
 *                                                                           *
 *   FILE: hri_struct.h                                                      *
 *                                                                           *
 *   PURPOSE:                                                                *
 *         declares structures for method HRI                                *
 *         (Hazard Rate Increasing)                                          *
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

struct unur_hri_par { 
  double p0;                          /* design (splitting) point            */
};

/*---------------------------------------------------------------------------*/
/* The generator object                                                      */

struct unur_hri_gen { 
  double p0;                          /* design (splitting) point            */
  double left_border;                 /* left border of domain               */
  double hrp0;                        /* hazard rate at p0                   */
};

/*---------------------------------------------------------------------------*/
























