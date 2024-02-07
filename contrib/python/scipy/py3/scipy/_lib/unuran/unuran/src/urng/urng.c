/*****************************************************************************
 *                                                                           *
 *          UNURAN -- Universal Non-Uniform Random number generator          *
 *                                                                           *
 *****************************************************************************
 *                                                                           *
 *   FILE: urng.c                                                            *
 *                                                                           *
 *   Unified interface for UNURAN uniform random number generators.          *
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
#include <unur_source.h>
#include "urng.h"
/*---------------------------------------------------------------------------*/

/*****************************************************************************/
/**                                                                         **/
/**  Sample from and reset an URNG object                                   **/
/**                                                                         **/
/*****************************************************************************/

double
unur_urng_sample (UNUR_URNG *urng)
     /*----------------------------------------------------------------------*/
     /* Sample from URNG object.                                             */
     /*                                                                      */
     /* parameters:                                                          */
     /*   urng   ... pointer to URNG object                                  */
     /*                                                                      */
     /* return:                                                              */
     /*   uniform random number                                              */
     /*                                                                      */
     /* error:                                                               */
     /*   return UNUR_INFINITY                                               */
     /*----------------------------------------------------------------------*/
{
  /* check argument */
  if (urng == NULL) 
    /* use default generator */
    urng = unur_get_default_urng();

  return _unur_call_urng(urng);
} /* end of unur_urng_sample() */ 

/*---------------------------------------------------------------------------*/

double
unur_sample_urng (UNUR_GEN *gen)
     /*----------------------------------------------------------------------*/
     /* Sample from underlying URNG object.                                  */
     /*                                                                      */
     /* parameters:                                                          */
     /*   gen ... pointer to generator object                                */
     /*                                                                      */
     /* return:                                                              */
     /*   uniform random number                                              */
     /*                                                                      */
     /* error:                                                               */
     /*   return UNUR_INFINITY                                               */
     /*----------------------------------------------------------------------*/
{
  struct unur_urng *urng = (gen) ? gen->urng : unur_get_default_urng(); 
  return _unur_call_urng(urng);
} /* end of unur_sample_urng() */

/*---------------------------------------------------------------------------*/

int
unur_urng_sample_array (UNUR_URNG *urng, double *X, int dim)
     /*----------------------------------------------------------------------*/
     /* Sample from URNG object and fill array X.                            */
     /* if URNG has a "natural" dimension s then only the first min(s,dim)   */
     /* entries are filled.                                                  */
     /*                                                                      */
     /* parameters:                                                          */
     /*   urng   ... pointer to URNG object                                  */
     /*   X      ... pointer to array of (at least) length dim               */
     /*   dim    ... (maximal) number of entries in X to be set              */
     /*                                                                      */
     /* return:                                                              */
     /*   number of uniform random numbers filled into array                 */
     /*                                                                      */
     /* error:                                                               */
     /*   return 0                                                           */
     /*----------------------------------------------------------------------*/
{
  /* check argument */
  if (urng == NULL) 
    /* use default generator */
    urng = unur_get_default_urng();

  if (urng->samplearray) {
    return (urng->samplearray(urng->state,X,dim));
  }
  else {
    int i;
    for (i=0; i<dim; i++) 
      X[i] = _unur_call_urng(urng);
    return dim;
  }
} /* end of unur_urng_sample_array() */ 

/*---------------------------------------------------------------------------*/

int
unur_urng_reset (UNUR_URNG *urng)
     /*----------------------------------------------------------------------*/
     /* Reset URNG object.                                                   */
     /*                                                                      */
     /* parameters:                                                          */
     /*   urng   ... pointer to URNG object                                  */
     /*                                                                      */
     /* return:                                                              */
     /*   UNUR_SUCCESS ... on success                                        */
     /*   error code   ... on error                                          */
     /*----------------------------------------------------------------------*/
{
  /* check argument */
  if (urng == NULL) 
    /* use default generator */
    urng = unur_get_default_urng();

#ifdef UNUR_URNG_UNURAN

  COOKIE_CHECK(urng,CK_URNG,UNUR_ERR_COOKIE);

  /* first we look for the reset function */
  if (urng->reset != NULL) {
    urng->reset (urng->state);
    return UNUR_SUCCESS;
  }

  /* if no reset function exists, we look for an initial seed */
  if (urng->setseed != NULL && urng->seed != ULONG_MAX) {
    unur_urng_seed(urng,urng->seed);
    return UNUR_SUCCESS;
  }

  /* neither works */
  _unur_error("URNG",UNUR_ERR_URNG_MISS,"reset");
  return UNUR_ERR_URNG_MISS;

#else

  return _unur_call_reset(urng);

#endif

} /* end of unur_urng_reset() */ 

/*---------------------------------------------------------------------------*/
