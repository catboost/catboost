/*****************************************************************************
 *                                                                           *
 *          UNURAN -- Universal Non-Uniform Random number generator          *
 *                                                                           *
 *****************************************************************************
 *                                                                           *
 *   FILE: urng_default.c                                                    *
 *                                                                           *
 *   routines to set, change and get the pointers to the                     *
 *   UNURAN default uniform random number generators.                        *
 *                                                                           *
 *****************************************************************************
 *                                                                           *
 *   Copyright (c) 2000-2022 Wolfgang Hoermann and Josef Leydold             *
 *   Department of Statistics and Mathematics, WU Wien, Austria              *
 *   SPDX-License-Identifier: BSD-3-Clause                                   *
 *                                                                           *

 *                                                                           *
 *****************************************************************************
 *                                                                           *
 *                       Modified for use in SciPy                           *
 *                                                                           *
 *****************************************************************************/

/*---------------------------------------------------------------------------*/

#include <unur_source.h>
#include "urng.h"

/*---------------------------------------------------------------------------*/
/* pointer to default uniform random number generator */

static UNUR_URNG *urng_default = NULL;
static UNUR_URNG *urng_aux_default = NULL;

/*---------------------------------------------------------------------------*/

/*****************************************************************************/
/**                                                                         **/
/**  Main uniform random number generator                                   **/
/**                                                                         **/
/*****************************************************************************/

UNUR_URNG *
unur_get_default_urng( void )
     /*----------------------------------------------------------------------*/
     /* return default uniform random number generator                       */
     /* (initialize generator if necessary)                                  */
     /*                                                                      */
     /* parameters: none                                                     */
     /*                                                                      */
     /* return:                                                              */
     /*   pointer to default generator                                       */
     /*                                                                      */
     /* error:                                                               */
     /*   return NULL                                                        */
     /*----------------------------------------------------------------------*/
{
  /* default generator already running ? */
  if( urng_default == NULL ) {
    _unur_error("URNG",UNUR_ERR_NULL,"Default URNG not set. EXIT !!!");
    /* we cannot recover from this error */
    exit(EXIT_FAILURE);
  }

  /* return default generator */
  return (urng_default);
} /* end of unur_get_default_urng() */

/*---------------------------------------------------------------------------*/

UNUR_URNG *
unur_set_default_urng( UNUR_URNG *urng_new )
     /*----------------------------------------------------------------------*/
     /* set default uniform random number generator and return old one       */
     /*                                                                      */
     /* parameters: pointer to new default uniform random number generator   */
     /*                                                                      */
     /* return:                                                              */
     /*   pointer to old  uniform random number generator                    */
     /*----------------------------------------------------------------------*/
{
  UNUR_URNG *urng_old = urng_default;

  /* NULL pointer not allowed */
  _unur_check_NULL("URNG", urng_new, urng_default);

  urng_default = urng_new;     /* set urng */

  /* return old default generator */
  return (urng_old);
} /* end of unur_set_default_urng() */


/*****************************************************************************/
/**                                                                         **/
/**  Auxiliary uniform random number generator                              **/
/**                                                                         **/
/*****************************************************************************/

UNUR_URNG *
unur_get_default_urng_aux( void )
     /*----------------------------------------------------------------------*/
     /* return default auxilliary uniform random number generator            */
     /* (initialize generator if necessary)                                  */
     /*                                                                      */
     /* parameters: none                                                     */
     /*                                                                      */
     /* return:                                                              */
     /*   pointer to default auxilliary uniform generator                    */
     /*                                                                      */
     /* error:                                                               */
     /*   return NULL                                                        */
     /*----------------------------------------------------------------------*/
{
  /* default generator already running ? */
  if( urng_aux_default == NULL ) {
    _unur_error("URNG",UNUR_ERR_NULL,"Default auxilliary URNG not set. EXIT !!!");
    /* we cannot recover from this error */
    exit(EXIT_FAILURE);
  }

  /* return default generator */
  return (urng_aux_default);
} /* end of unur_get_default_urng_aux() */

/*---------------------------------------------------------------------------*/

UNUR_URNG *
unur_set_default_urng_aux( UNUR_URNG *urng_aux_new )
     /*----------------------------------------------------------------------*/
     /* set default auxilliary uniform RNG and return old one.               */
     /*                                                                      */
     /* parameters: pointer to new default auxilliary uniform RNG            */
     /*                                                                      */
     /* return:                                                              */
     /*   pointer to old auxilliary uniform RNG                              */
     /*----------------------------------------------------------------------*/
{
  UNUR_URNG *urng_aux_old = urng_aux_default;

  /* NULL pointer not allowed */
  _unur_check_NULL("URNG", urng_aux_new, urng_aux_default);

  urng_aux_default = urng_aux_new;     /* set auxilliary urng */

  /* return old default generator */
  return (urng_aux_old);
} /* end of unur_set_default_urng_aux() */

/*---------------------------------------------------------------------------*/