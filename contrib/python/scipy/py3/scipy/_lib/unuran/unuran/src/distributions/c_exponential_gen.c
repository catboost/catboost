/*****************************************************************************
 *                                                                           *
 *          UNURAN -- Universal Non-Uniform Random number generator          *
 *                                                                           *
 *****************************************************************************
 *                                                                           *
 *   FILE:      c_exponential_gen.c                                          *
 *                                                                           *
 *   Special generators for Exponential distribution                         *
 *                                                                           *
 *****************************************************************************
 *                                                                           *
 *   Copyright (c) 2000-2022 Wolfgang Hoermann and Josef Leydold             *
 *   Department of Statistics and Mathematics, WU Wien, Austria              *
 *   SPDX-License-Identifier: BSD-3-Clause                                   *
 *                                                                           *

 *                                                                           *
 *****************************************************************************/

/*****************************************************************************
 * We use the marginal generation time for inversion from the                *
 * exponential distribution as time unit for timings.                        *
 * Thus we do not use the inverse CDF but have the particular method         *
 * implemented.                                                              *
 *****************************************************************************/

/*---------------------------------------------------------------------------*/

#include <unur_source.h>
#include <methods/cstd.h>
#include <methods/cstd_struct.h>
#include "unur_distributions_source.h"

/*---------------------------------------------------------------------------*/
/* Prototypes for special generators                                         */

/*---------------------------------------------------------------------------*/
/* abbreviations */

#define PAR       ((struct unur_cstd_par*)par->datap) /* data for parameter object */
#define GEN       ((struct unur_cstd_gen*)gen->datap) /* data for generator object */
#define DISTR     gen->distr->data.cont /* data for distribution in generator object */

#define uniform()  _unur_call_urng(gen->urng) /* call for uniform prng       */

#define sigma (DISTR.params[0])     /* scale */
#define theta (DISTR.params[1])     /* location */

/*---------------------------------------------------------------------------*/

/*****************************************************************************/
/**                                                                         **/
/**  Inititialize                                                           **/
/**                                                                         **/
/*****************************************************************************/

/*---------------------------------------------------------------------------*/

int 
_unur_stdgen_exponential_init( struct unur_par *par, struct unur_gen *gen )
     /*----------------------------------------------------------------------*/
     /* initialize special generator for exponential distribution            */
     /* if gen == NULL then only check existance of variant.                 */
     /*                                                                      */
     /* parameters:                                                          */
     /*   par ... pointer to parameter for building generator object         */
     /*   gen ... pointer to generator object                                */
     /*                                                                      */
     /* return:                                                              */
     /*   UNUR_SUCCESS ... on success                                        */
     /*   error code   ... on error                                          */
     /*----------------------------------------------------------------------*/
{
  /* one of par and gen must not be the NULL pointer */
  switch ((par) ? par->variant : gen->variant) {

  case 0:  /* DEFAULT */
  case UNUR_STDGEN_INVERSION:   /* inversion method */
    if (gen) GEN->is_inversion = TRUE;
    _unur_cstd_set_sampling_routine(gen, _unur_stdgen_sample_exponential_inv);
    /** Remark: We use sampling from exponential distribution as timing unit **/
    /* _unur_cstd_set_sampling_routine(gen, _unur_cstd_sample_inv); */
    return UNUR_SUCCESS;

  default: /* no such generator */
    if (gen) _unur_warning(gen->genid,UNUR_ERR_SHOULD_NOT_HAPPEN,"");
    return UNUR_FAILURE;
  }
  
} /* end of _unur_stdgen_exponential_init() */

/*---------------------------------------------------------------------------*/

/*****************************************************************************/
/**                                                                         **/
/**  Special generators                                                     **/
/**                                                                         **/
/*****************************************************************************/

/*---------------------------------------------------------------------------*/

double _unur_stdgen_sample_exponential_inv( struct unur_gen *gen )
     /* Inversion method                                                     */
{
  /* -X- generator code -X- */
  double U,X;

  /* check arguments */
  CHECK_NULL(gen,UNUR_INFINITY);
  COOKIE_CHECK(gen,CK_CSTD_GEN,UNUR_INFINITY);

  /* sample from uniform random number generator */
  U = GEN->Umin + uniform() * (GEN->Umax-GEN->Umin);

  /* transform to random variate */
  X = - log( 1. - U );

  /* -X- end of generator code -X- */

  return ((DISTR.n_params==0) ? X : theta + sigma * X);

} /* end of _unur_stdgen_sample_exponential_inv() */

/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
#undef sigma
#undef theta
/*---------------------------------------------------------------------------*/


