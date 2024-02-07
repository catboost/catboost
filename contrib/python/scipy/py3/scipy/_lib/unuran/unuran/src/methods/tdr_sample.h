/*****************************************************************************
 *                                                                           *
 *          UNURAN -- Universal Non-Uniform Random number generator          *
 *                                                                           *
 *****************************************************************************
 *                                                                           *
 *   FILE:      tdr_sample.c                                                 *
 *                                                                           *
 *   TYPE:      continuous univariate random variate                         *
 *   METHOD:    transformed density rejection                                *
 *                                                                           *
 *   DESCRIPTION:                                                            *
 *      Given PDF of a T-concave distribution                                *
 *      produce a value x consistent with its density                        *
 *                                                                           *
 *****************************************************************************
 *                                                                           *
 *   Copyright (c) 2000-2022 Wolfgang Hoermann and Josef Leydold             *
 *   Department of Statistics and Mathematics, WU Wien, Austria              *
 *   SPDX-License-Identifier: BSD-3-Clause                                   *
 *                                                                           *

 *                                                                           *
 *****************************************************************************/

/*****************************************************************************/
/**  Sampling routines                                                      **/
/*****************************************************************************/

#include "tdr_gw_sample.h"
#include "tdr_ia_sample.h"
#include "tdr_ps_sample.h"

/*---------------------------------------------------------------------------*/

double
unur_tdr_eval_invcdfhat( const struct unur_gen *gen, double u,
			 double *hx, double *fx, double *sqx )
     /*----------------------------------------------------------------------*/
     /* evaluate inverse CDF of hat at u                                     */
     /*                                                                      */
     /* parameters:                                                          */
     /*   gen ... pointer to generator object                                */
     /*   u   ... argument for inverse CDF (0<=u<=1)                         */
     /*   hx  ... pointer for storing hat at sampled X                       */
     /*   fx  ... pointer for storing squeeze at sampled X                   */
     /*   sqx ... pointer for storing density at sampled X                   */
     /*                                                                      */
     /* return:                                                              */
     /*   inverse hat CDF.                                                   */
     /*                                                                      */
     /*   values of hat, density, squeeze (computation is suppressed if      */
     /*   corresponding pointer is NULL).                                    */
     /*----------------------------------------------------------------------*/
{ 
  /* check arguments */
  _unur_check_NULL( GENTYPE, gen, UNUR_INFINITY );
  if ( gen->method != UNUR_METH_TDR ) {
    _unur_error(gen->genid,UNUR_ERR_GEN_INVALID,"");
    return UNUR_INFINITY; 
  }
  COOKIE_CHECK(gen,CK_TDR_GEN,UNUR_INFINITY);

  if (GEN->iv == NULL) {
    _unur_error(gen->genid,UNUR_ERR_GEN_DATA,"empty generator object");
    return UNUR_INFINITY;
  } 

  if ( u<0. || u>1.) {
    _unur_warning(gen->genid,UNUR_ERR_DOMAIN,"argument u not in [0,1]");
  }

  /* validate argument */
  if (u<=0.) return DISTR.domain[0];
  if (u>=1.) return DISTR.domain[1];

  /* compute inverse CDF */
  /* sampling routines */
  switch (gen->variant & TDR_VARMASK_VARIANT) {
  case TDR_VARIANT_GW:    /* original variant (Gilks&Wild) */
    return _unur_tdr_gw_eval_invcdfhat(gen,u,hx,fx,sqx,NULL,NULL);
  case TDR_VARIANT_IA:    /* immediate acceptance */
    /* this does not make to much sense, since IA is not
       a pure rejection method. Nevertheless, we treat
       it in the same way as variant PS.                 */
  case TDR_VARIANT_PS:    /* proportional squeeze */
    return _unur_tdr_ps_eval_invcdfhat(gen,u,hx,fx,sqx,NULL);
  default:
    _unur_error(GENTYPE,UNUR_ERR_SHOULD_NOT_HAPPEN,"");
    return UNUR_INFINITY;
  }

} /* end of unur_tdr_eval_invcdfhat() */

/*---------------------------------------------------------------------------*/
