/*****************************************************************************
 *                                                                           *
 *          UNURAN -- Universal Non-Uniform Random number generator          *
 *                                                                           *
 *****************************************************************************
 *                                                                           *
 *   FILE: hist_struct.h                                                     *
 *                                                                           *
 *   PURPOSE:                                                                *
 *         declares structures for method HIST                               *
 *         (HISTogram of empirical distribution)                             *
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

struct unur_hist_par {
  /* the histogram is stored in the distribution object */
  int dummy;
};

/*---------------------------------------------------------------------------*/
/* The generator object                                                      */

struct unur_hist_gen {
  int     n_hist;       /* number of bins in histogram                       */
  double *prob;         /* probabilities for bins                            */
  double *bins;         /* location of bins (when different width)           */
  double  hmin, hmax;   /* lower and upper bound for histogram               */
  double  hwidth;       /* width of bins (when equal width)                  */
  double  sum;          /* sum of all probabilities = cumpv[len-1]           */
  double *cumpv;        /* pointer to the vector of cumulated probabilities  */
  int    *guide_table;  /* pointer to guide table                            */
};

/*---------------------------------------------------------------------------*/


