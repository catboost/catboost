/*****************************************************************************
 *                                                                           *
 *          UNURAN -- Universal Non-Uniform Random number generator          *
 *                                                                           *
 *****************************************************************************
 *                                                                           *
 *   FILE: unur_struct.h                                                     *
 *                                                                           *
 *   PURPOSE:                                                                *
 *         declares structures for distribution, parameter, and generator    *
 *         objects.                                                          *
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
#ifndef UNUR_STRUCT_H_SEEN
#define UNUR_STRUCT_H_SEEN
/*---------------------------------------------------------------------------*/

/*****************************************************************************/
/**  Basic header files                                                     **/
/*****************************************************************************/

/*****************************************************************************/
/**  UNU.RAN objects                                                        **/
/*****************************************************************************/

struct unur_distr;    /* distribution object      */
struct unur_par;      /* parameters for generator */
struct unur_gen;      /* generator object         */

/*****************************************************************************/
/**  Generic functions                                                      **/
/*****************************************************************************/

/*---------------------------------------------------------------------------*/
/* Generic functions                                                          */

typedef double UNUR_FUNCT_GENERIC  (double  x, void *params);
typedef double UNUR_FUNCT_VGENERIC (double *x, void *params);

/* for univariate functions with optional parameter array */
struct unur_funct_generic {
  UNUR_FUNCT_GENERIC *f;
  void *params;
};

/* for multivariate functions with optional parameter array */
struct unur_funct_vgeneric {
  UNUR_FUNCT_VGENERIC *f;
  void *params;
};

/*****************************************************************************/
/**  Auxiliary tools                                                        **/
/*****************************************************************************/

#include <utils/slist_struct.h>
#include <utils/string_struct.h>

/*****************************************************************************/
/**  Declaration for parser                                                 **/
/*****************************************************************************/

#include <parser/functparser_struct.h>

/*****************************************************************************/
/**  URNG (uniform random number generator) objects                         **/
/*****************************************************************************/

#include <urng/urng_struct.h>

/*****************************************************************************/
/**  Distribution objects                                                   **/
/*****************************************************************************/

#include <distr/distr_struct.h>

/*****************************************************************************/
/**  Parameter and generators objects                                       **/
/*****************************************************************************/

#include <methods/x_gen_struct.h>

/*---------------------------------------------------------------------------*/
#endif  /* UNUR_STRUCT_H_SEEN */
/*---------------------------------------------------------------------------*/
