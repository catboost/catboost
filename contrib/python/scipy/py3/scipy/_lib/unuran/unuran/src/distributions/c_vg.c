/*****************************************************************************
 *                                                                           *
 *          UNURAN -- Universal Non-Uniform Random number generator          *
 *                                                                           *
 *****************************************************************************
 *                                                                           *
 *   FILE:      c_vg.c                                                       *
 *                                                                           *
 *   REFERENCES:                                                             *
 *                                                                           *
 *****************************************************************************
 *                                                                           *
 *  distr: Variance gamma distribution                                       *
 *                                                                           *
 *  pdf:   f(x) = |x-mu|^(lambda-1/2) * exp(beta*(x-mu))                     *
 *                    * K_{lambda-1/2}(alpha*|x-mu|)}                        *
 *                                                                           *
 *  domain:   infinity < x < infinity                                        *
 *                                                                           *
 *  constant: (alpha^2 - beta^2)^lambda                                      *
 *                / (sqrt(pi) * (2*alpha)^(lambda-1/2) * Gamma(lambda))      *
 *                                                                           *
 *             [K_theta(.) ... modified Bessel function of second kind]      *
 *                                                                           *
 *  parameters: 4                                                            *
 *     0 : lambda > 0    ... shape                                           *
 *     1 : alpha >|beta| ... shape                                           *
 *     2 : beta          ... shape (asymmetry)                               *
 *     3 : mu            ... location                                        *
 *                                                                           *
 *****************************************************************************
 *                                                                           *
 *   We use the Rmath library for computing the Bessel function K_n.         *
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

static const char distr_name[] = "vg";

/* parameters */
#define lambda  params[0]    /* shape */
#define alpha   params[1]    /* shape */
#define beta    params[2]    /* shape (asymmetry) */
#define mu      params[3]    /* location */

#define DISTR distr->data.cont
#define LOGNORMCONSTANT (distr->data.cont.norm_constant)

/* function prototypes                                                       */
#ifdef _unur_SF_bessel_k
static double _unur_pdf_vg( double x, const UNUR_DISTR *distr );
static double _unur_logpdf_vg( double x, const UNUR_DISTR *distr );
/* static double _unur_dpdf_vg( double x, const UNUR_DISTR *distr ); */
/* static double _unur_cdf_vg( double x, const UNUR_DISTR *distr ); */
#endif

static int _unur_upd_center_vg( UNUR_DISTR *distr );
static double _unur_lognormconstant_vg( const double *params, int n_params );
static int _unur_set_params_vg( UNUR_DISTR *distr, const double *params, int n_params );

/*---------------------------------------------------------------------------*/

#ifdef _unur_SF_bessel_k
double
_unur_pdf_vg(double x, const UNUR_DISTR *distr)
{
  /* Original implementation by Kemal Dingic */
  /* f(x) = |x-mu|^(lambda-1/2) * exp(beta*(x-mu)) * K_{lambda-1/2}(alpha*|x-mu|)} */

  return exp(_unur_logpdf_vg(x,distr));
} /* end of _unur_pdf_vg() */

/*---------------------------------------------------------------------------*/

double
_unur_logpdf_vg(double x, const UNUR_DISTR *distr)
{
  /* Original implementation by Kemal Dingic */
  /* f(x) = |x-mu|^(lambda-1/2) * exp(beta*(x-mu)) * K_{lambda-1/2}(alpha*|x-mu|) */

  const double *params = DISTR.params;
  double nu = lambda - 0.5;   /* order of modified Bessel function K()       */
  double res;                 /* result of computation                       */
  double y, absy;             /* auxiliary variables                         */

  y = x - mu;
  absy = fabs(y);

  /* Using nu and y we find:
   *   f(x) = |y|^nu * exp(beta*y) * K_nu(alpha*|y|)
   * and
   *   log(f(x)) = nu*log(|y|)+ beta*y + log(K_nu(alpha*|y|)
   */

  do {
    if (absy>0) {
      /* first simply compute Bessel K_nu and compute logarithm. */
      double besk;

      /* we currently use two algorithms based on our experiences
       * with the functions in the Rmath library.
       * (Maybe we could change this when we link against
       * other libraries.)
       */
      if (nu < 100) 
	/* the "standard implementation" using log(bessel_k) */
	besk = _unur_SF_ln_bessel_k(alpha*absy, nu);
      else
	/* an algorithm for large nu */
	besk = _unur_SF_bessel_k_nuasympt(alpha*absy, nu, TRUE, FALSE);

      /* there can be numerical problems with the Bessel function K_nu. */
      if (_unur_isfinite(besk) && besk < MAXLOG - 20.0) {
	/* o.k. */
	res = LOGNORMCONSTANT + besk + log(absy)*nu + beta*y;
	break;
      }
    }

    /* Case: numerical problems with Bessel function K_nu. */

    if (absy < 1.0) {
      /* Case: Bessel function K_nu overflows for small values of y.
       * The following code is inspired by gsl_sf_bessel_lnKnu_e() from
       * the GSL (GNU Scientific Library).
       */

      res = LOGNORMCONSTANT + beta*y;
      res += -M_LN2 + _unur_SF_ln_gamma(nu) + nu*log(2./alpha);
      
      if (nu > 1.0) {
	double xi = 0.25*(alpha*absy)*(alpha*absy);
	double sum = 1.0 - xi/(nu-1.0);
	if(nu > 2.0) sum += (xi/(nu-1.0)) * (xi/(nu-2.0));
	res += log(sum);
      }
    }

    else {
      /* Case: Bessel function K_nu underflows for very large values of y
       * and we get NaN. 
       * However, then the PDF of the Variance Gamma distribution is 0.
       */
      res = -UNUR_INFINITY;
    } 

  } while(0);

  /*
    Remark: a few references

    NIST Digital Library of Mathematical Functions

    http://dlmf.nist.gov/10.27.E3
    K_{-nu}(z) = K_nu(z)

    http://dlmf.nist.gov/10.32.E10
    K_nu(z) = 0.5*(0.5*z)^nu * \int_0^\infty exp(-t-z^2/(4*t)) * t^(-nu-1) dt

    This implies
    K_nu(z) = K_{-nu}(z)
     = 0.5*(0.5*z)^(-nu) * \int_0^\infty exp(-t-z^2/(4*t)) * t^(nu-1) dt
    <=  0.5*(0.5*z)^(-nu) * \int_0^\infty exp(-t) * t^(nu-1) dt
     =  0.5*(0.5*z)^(-nu) * Gamma(z)

    http://dlmf.nist.gov/10.30.E2
    K_nu(z) ~ 0.5*Gamma(nu)*(0.5*z)^(-nu) for z->0
  */

  return res;
} /* end of _unur_logpdf_vg() */
#endif

/*---------------------------------------------------------------------------*/

int
_unur_upd_center_vg( UNUR_DISTR *distr )
{
  const double *params = DISTR.params;

  /* we use the mean of the distribution: */
  double gam = sqrt(alpha*alpha-beta*beta);
  DISTR.center = mu + 2*beta*lambda / (gam*gam);

  /* there is some change of overflow. then we simply use 'mu' */
  if (!_unur_isfinite(DISTR.center))
    DISTR.center = mu;

  /* center must be in domain */
  if (DISTR.center < DISTR.domain[0])
    DISTR.center = DISTR.domain[0];
  else if (DISTR.center > DISTR.domain[1])
    DISTR.center = DISTR.domain[1];

  return UNUR_SUCCESS;
} /* end of _unur_upd_center_vg() */

/*---------------------------------------------------------------------------*/

double
_unur_lognormconstant_vg(const double *params, int n_params ATTRIBUTE__UNUSED)
{
  /*
    (alpha^2 - beta^2)^lambda 
    / (sqrt(pi) * (2*alpha)^(lambda-1/2) * Gamma(lambda))
  */

  return (lambda*log(alpha*alpha - beta*beta) - 0.5*M_LNPI 
	  - (lambda-0.5)*log(2*alpha) - _unur_SF_ln_gamma(lambda));
} /* end of _unur_normconstant_vg() */

/*---------------------------------------------------------------------------*/

int
_unur_set_params_vg( UNUR_DISTR *distr, const double *params, int n_params )
{
  /* check number of parameters for distribution */
  if (n_params < 4) {
    _unur_error(distr_name,UNUR_ERR_DISTR_NPARAMS,"too few"); return UNUR_ERR_DISTR_NPARAMS; }
  if (n_params > 4) {
    _unur_warning(distr_name,UNUR_ERR_DISTR_NPARAMS,"too many");
    n_params = 4; }
  CHECK_NULL(params,UNUR_ERR_NULL);

  /* check parameter omega */
  if (lambda <= 0.) {
    _unur_error(distr_name,UNUR_ERR_DISTR_DOMAIN,"lambda <= 0");
    return UNUR_ERR_DISTR_DOMAIN;
  }

  if (alpha <= fabs(beta)) {
    _unur_error(distr_name,UNUR_ERR_DISTR_DOMAIN,"alpha <= |beta|");
    return UNUR_ERR_DISTR_DOMAIN;
  }

  /* copy parameters for standard form */
  DISTR.lambda = lambda;
  DISTR.alpha = alpha;
  DISTR.beta = beta;
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
} /* end of _unur_set_params_vg() */

/*---------------------------------------------------------------------------*/

struct unur_distr *
unur_distr_vg( const double *params, int n_params)
{
  register struct unur_distr *distr;

  /* get new (empty) distribution object */
  distr = unur_distr_cont_new();

  /* set distribution id */
  distr->id = UNUR_DISTR_VG;

  /* name of distribution */
  distr->name = distr_name;
             
  /* how to get special generators */
  /* DISTR.init = _unur_stdgen_vg_init; */

  /* functions */
#ifdef _unur_SF_bessel_k
  DISTR.pdf     = _unur_pdf_vg;     /* pointer to PDF                  */
  DISTR.logpdf  = _unur_logpdf_vg;  /* pointer to log-PDF              */
#endif

  /* indicate which parameters are set */
  distr->set = ( UNUR_DISTR_SET_DOMAIN |
		 UNUR_DISTR_SET_STDDOMAIN |
		 UNUR_DISTR_SET_CENTER |
		 UNUR_DISTR_SET_PDFAREA );
                
  /* set parameters for distribution */
  if (_unur_set_params_vg(distr,params,n_params)!=UNUR_SUCCESS) {
    free(distr);
    return NULL;
  }

  /* normalization constant */
  LOGNORMCONSTANT = _unur_lognormconstant_vg(DISTR.params,DISTR.n_params);

  /* we need the center of the distribution */
  if (_unur_upd_center_vg(distr)!=UNUR_SUCCESS) {
    free(distr);
    return NULL;
  }

  /* mode and area below p.d.f. */
  /* DISTR.mode = ? */
  DISTR.area = 1;

  /* function for setting parameters and updating domain */
  DISTR.set_params = _unur_set_params_vg;

  /* function for updating derived parameters */
  /* DISTR.upd_mode  = _unur_upd_mode_vg; /\* funct for computing mode *\/ */
  /* DISTR.upd_area  = _unur_upd_area_vg; /\* funct for computing area *\/ */

  /* return pointer to object */
  return distr;

} /* end of unur_distr_vg() */

/*---------------------------------------------------------------------------*/
#undef mu
#undef alpha
#undef beta
#undef lambda
#undef DISTR
/*---------------------------------------------------------------------------*/
