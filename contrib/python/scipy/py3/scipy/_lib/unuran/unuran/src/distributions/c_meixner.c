/*****************************************************************************
 *                                                                           *
 *          UNURAN -- Universal Non-Uniform Random number generator          *
 *                                                                           *
 *****************************************************************************
 *                                                                           *
 *   FILE:      c_meixner.c                                                  *
 *                                                                           *
 *   REFERENCES:                                                             *
 *                                                                           *
 *****************************************************************************
 *                                                                           *
 *  distr: Meixner distribution                                              *
 *                                                                           *
 *  pdf:   f(x) = exp(beta*(x-mu)/alpha) * |Gamma(delta+ i*(x-mu)/alpha)|^2  *
 *                                                                           *
 *  domain:   infinity < x < infinity                                        *
 *                                                                           *
 *  constant: (2*cos(beta/2))^(2*delta) / (2*alpha*pi*Gamma(2*delta))        *
 *                                                                           *
 *  parameters: 4                                                            *
 *     0 : alpha > 0 ... scale                                               *
 *     1 : beta      ... shape (asymmetry) in [-pi,pi]                       *
 *     2 : delta >0  ... shape                                               *
 *     3 : mu        ... location                                            *
 *                                                                           *
 *****************************************************************************
 *                                                                           *
 *   Copyright (c) 2011-2012 Wolfgang Hoermann and Josef Leydold             *
 *   Institute for Statistics and Mathematics, WU Wien, Austria              *
 *                                                                           *

 *                                                                           *
 *****************************************************************************/

/*---------------------------------------------------------------------------*/

#include <unur_source.h>
#include <distr/distr_source.h>
#include <distr/cont.h>
#include "unur_distributions.h"
#include "unur_distributions_source.h"
#include "unur_stddistr.h"

/*---------------------------------------------------------------------------*/

static const char distr_name[] = "meixner";

/* parameters */
#define alpha   params[0]    /* scale */
#define beta    params[1]    /* shape (asymmetry) */
#define delta   params[2]    /* shape */
#define mu      params[3]    /* location */

#define DISTR distr->data.cont
#define LOGNORMCONSTANT (distr->data.cont.norm_constant)

/* function prototypes                                                       */
static double _unur_pdf_meixner( double x, const UNUR_DISTR *distr );
static double _unur_logpdf_meixner( double x, const UNUR_DISTR *distr );
/* static double _unur_dpdf_meixner( double x, const UNUR_DISTR *distr ); */
/* static double _unur_cdf_meixner( double x, const UNUR_DISTR *distr ); */

static int _unur_upd_center_meixner( UNUR_DISTR *distr );
static double _unur_lognormconstant_meixner( const double *params, int n_params );
static int _unur_set_params_meixner( UNUR_DISTR *distr, const double *params, int n_params );

/*---------------------------------------------------------------------------*/

double
_unur_pdf_meixner(double x, const UNUR_DISTR *distr)
{
  /* Original implementation by Kemal Dingic */
  /*  f(x) = exp(beta*(x-mu)/alpha) * |Gamma(delta+ i*(x-mu)/alpha)|^2 */

  return exp(_unur_logpdf_meixner(x,distr));
} /* end of _unur_pdf_meixner() */

/*---------------------------------------------------------------------------*/

double
_unur_logpdf_meixner(double x, const UNUR_DISTR *distr)
{
  const double *params = DISTR.params;
  double res;           /* result of computation */
  double y;             /* auxiliary variables   */
  
  y = (x-mu) / alpha;
  res = LOGNORMCONSTANT + beta*y + 2*_unur_SF_Relcgamma(delta, y);

  return res;
} /* end of _unur_logpdf_meixner() */

/*---------------------------------------------------------------------------*/

int
_unur_upd_center_meixner( UNUR_DISTR *distr )
{
  const double *params = DISTR.params;

  /* we simply use parameter 'mu' */
  DISTR.center = mu;

  /* an alternative approach would be the mean of the distribution:          */
  /* DISTR.center = mu + alpha*delta*tan(beta/2);                            */

  /* center must be in domain */
  if (DISTR.center < DISTR.domain[0])
    DISTR.center = DISTR.domain[0];
  else if (DISTR.center > DISTR.domain[1])
    DISTR.center = DISTR.domain[1];

  return UNUR_SUCCESS;
} /* end of _unur_upd_center_meixner() */

/*---------------------------------------------------------------------------*/

double
_unur_lognormconstant_meixner(const double *params, int n_params ATTRIBUTE__UNUSED)
{
  /*
    (2*cos(beta/2))^(2*delta) / (2*alpha*pi*Gamma(2*delta))
  */

  return ( 2.*delta*log(2.*cos(beta/2.))
	   - (log(2.*alpha*M_PI) + _unur_SF_ln_gamma(2.*delta)));
} /* end of _unur_normconstant_meixner() */

/*---------------------------------------------------------------------------*/

int
_unur_set_params_meixner( UNUR_DISTR *distr, const double *params, int n_params )
{
  /* check number of parameters for distribution */
  if (n_params < 4) {
    _unur_error(distr_name,UNUR_ERR_DISTR_NPARAMS,"too few"); return UNUR_ERR_DISTR_NPARAMS; }
  if (n_params > 4) {
    _unur_warning(distr_name,UNUR_ERR_DISTR_NPARAMS,"too many");
    n_params = 4; }
  CHECK_NULL(params,UNUR_ERR_NULL);

  /* check parameter omega */
  if (alpha <= 0. || delta <= 0.) {
    _unur_error(distr_name,UNUR_ERR_DISTR_DOMAIN,"alpha or delta <= 0");
    return UNUR_ERR_DISTR_DOMAIN;
  }

  if (fabs(beta) >= M_PI) {
    _unur_error(distr_name,UNUR_ERR_DISTR_DOMAIN,"beta not in (-PI,PI)");
    return UNUR_ERR_DISTR_DOMAIN;
  }

  /* copy parameters for standard form */
  DISTR.alpha = alpha;
  DISTR.beta = beta;
  DISTR.delta = delta;
  DISTR.mu = mu;

  /* default parameters: none */

  /* store number of parameters */
  DISTR.n_params = n_params;

  /* set (standard) domain */
  if (distr->set & UNUR_DISTR_SET_STDDOMAIN) {
    DISTR.domain[0] = -UNUR_INFINITY;   /* left boundary  */
    DISTR.domain[1] = UNUR_INFINITY;    /* right boundary */
  }

  return UNUR_SUCCESS;
} /* end of _unur_set_params_meixner() */

/*---------------------------------------------------------------------------*/

struct unur_distr *
unur_distr_meixner( const double *params, int n_params)
{
  register struct unur_distr *distr;

  /* get new (empty) distribution object */
  distr = unur_distr_cont_new();

  /* set distribution id */
  distr->id = UNUR_DISTR_MEIXNER;

  /* name of distribution */
  distr->name = distr_name;
             
  /* how to get special generators */
  /* DISTR.init = _unur_stdgen_meixner_init; */
   
  /* functions */
  DISTR.pdf     = _unur_pdf_meixner;     /* pointer to PDF                  */
  DISTR.logpdf  = _unur_logpdf_meixner;  /* pointer to logPDF               */
 
  /* indicate which parameters are set */
  distr->set = ( UNUR_DISTR_SET_DOMAIN |
		 UNUR_DISTR_SET_STDDOMAIN |
		 UNUR_DISTR_SET_CENTER |
		 UNUR_DISTR_SET_PDFAREA );
                
  /* set parameters for distribution */
  if (_unur_set_params_meixner(distr,params,n_params)!=UNUR_SUCCESS) {
    free(distr);
    return NULL;
  }

  /* normalization constant */
  LOGNORMCONSTANT = _unur_lognormconstant_meixner(DISTR.params,DISTR.n_params);

  /* we need the center of the distribution */
  if (_unur_upd_center_meixner(distr)!=UNUR_SUCCESS) {
    free(distr);
    return NULL;
  }

  /* mode and area below p.d.f. */
  /* DISTR.mode = ? */
  DISTR.area = 1;

  /* function for setting parameters and updating domain */
  DISTR.set_params = _unur_set_params_meixner;

  /* function for updating derived parameters */
  /* DISTR.upd_mode  = _unur_upd_mode_meixner; /\* funct for computing mode *\/ */
  /* DISTR.upd_area  = _unur_upd_area_meixner; /\* funct for computing area *\/ */

  /* return pointer to object */
  return distr;

} /* end of unur_distr_meixner() */

/*---------------------------------------------------------------------------*/
#undef alpha
#undef beta
#undef delta
#undef mu
#undef DISTR
/*---------------------------------------------------------------------------*/
