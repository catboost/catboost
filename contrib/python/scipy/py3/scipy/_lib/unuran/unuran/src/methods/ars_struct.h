/*****************************************************************************
 *                                                                           *
 *          UNURAN -- Universal Non-Uniform Random number generator          *
 *                                                                           *
 *****************************************************************************
 *                                                                           *
 *   FILE: ars_struct.h                                                      *
 *                                                                           *
 *   PURPOSE:                                                                *
 *         declares structures for method ARS                                *
 *         (Adaptive Rejection Sampling - Gilks & Wild)                      *
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

struct unur_ars_par { 

  const double *starting_cpoints; /* pointer to array of starting points     */
  int n_starting_cpoints;         /* number of construction points at start  */
  const double *percentiles; /* percentiles of hat for c. points of new hat  */
  int n_percentiles;         /* number of percentiles                        */
  int retry_ncpoints;        /* number of cpoints for second trial of reinit */

  int max_ivs;               /* maximum number of intervals                  */
  int max_iter;              /* maximum number of iterations                 */
};

/*---------------------------------------------------------------------------*/
/* store data for segments                                                   */

struct unur_ars_interval {

  double  x;              /* (left hand side) construction point (cp)        */
  double  logfx;          /* value of logPDF at cp                           */
  double  dlogfx;         /* derivative of logPDF at cp                      */
  double  sq;             /* slope of transformed squeeze in interval        */

  double  Acum;           /* cumulated area of intervals                     */
  double  logAhat;        /* log of area below hat                           */
  double  Ahatr_fract;    /* fraction of area below hat on r.h.s.            */

  struct unur_ars_interval *next; /* pointer to next interval in list        */

#ifdef DEBUG_STORE_IP 
  double  ip;             /* intersection point between two tangents         */
#endif
#ifdef UNUR_COOKIES
  unsigned cookie;        /* magic cookie                                    */
#endif
};

/*---------------------------------------------------------------------------*/
/* The generator object                                                      */

struct unur_ars_gen { 

  double  Atotal;               /* area below hat                            */
  double  logAmax;              /* log of maximum area in intervals          */

  struct unur_ars_interval *iv; /* pointer to linked list of intervals       */
  int     n_ivs;                /* number of intervals                       */
  int     max_ivs;              /* maximum number of intervals               */
  int     max_iter;             /* maximum number of iterations              */

  double *starting_cpoints;     /* pointer to array of starting points       */
  int     n_starting_cpoints;   /* number of construction points at start    */

  double *percentiles;       /* percentiles of hat for c. points of new hat  */
  int n_percentiles;         /* number of percentiles                        */
  int retry_ncpoints;        /* number of cpoints for second trial of reinit */
};

/*---------------------------------------------------------------------------*/
