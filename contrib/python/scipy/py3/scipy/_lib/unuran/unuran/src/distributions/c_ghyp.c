/*****************************************************************************
 *                                                                           *
 *          UNURAN -- Universal Non-Uniform Random number generator          *
 *                                                                           *
 *****************************************************************************
 *                                                                           *
 *   FILE:      c_ghyp.c                                                     *
 *                                                                           *
 *   REFERENCES:                                                             *
 *                                                                           *
 *****************************************************************************
 *                                                                           *
 *  distr: Generalized hyperbolic distribution                               *
 *                                                                           *
 *  pdf:   f(x) = (delta^2+(x-mu)^2)^(1/2*(lambda-1/2)) * exp(beta*(x-mu))   *
 *                  * K_{lambda-1/2}(alpha*sqrt(delta^2+(x-mu)^2))           * 
 *                                                                           *
 *  domain:   infinity < x < infinity                                        *
 *                                                                           *
 *  constant: ( (gamma/delta)^lambda )                                       *
 *            / ( sqrt(2*pi) * alpha^(lambda-1/2) * K_{lambda}(delta*gamma) )
 *                                                                           *
 *             [gamma = sqrt(alpha^2 - beta^2)                        ]      *
 *             [K_theta(.) ... modified Bessel function of second kind]      *
 *                                                                           *
 *  parameters: 4                                                            *
 *     0 : lambda        ... shape                                           *
 *     1 : alpha >|beta| ... shape                                           *
 *     2 : beta          ... shape (asymmetry)                               *
 *     3 : delta > 0     ... scale                                           *
 *     4 : mu            ... location                                        *
 *                                                                           *
 *****************************************************************************
 *                                                                           *
 *   We use the Rmath library for computing the Bessel function K_n.         *
 *                                                                           *
 *****************************************************************************
 *                                                                           *
 *   Copyright (c) 2009-2012 Wolfgang Hoermann and Josef Leydold             *
 *   Department of Statistics and Mathematics, WU Wien, Austria              *
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

static const char distr_name[] = "ghyp";

/* parameters */
#define lambda  params[0]    /* shape */
#define alpha   params[1]    /* shape */
#define beta    params[2]    /* shape (asymmetry) */
#define delta   params[3]    /* scale */
#define mu      params[4]    /* location */

#define DISTR distr->data.cont
#define LOGNORMCONSTANT (distr->data.cont.norm_constant)

/* function prototypes                                                       */
#ifdef _unur_SF_bessel_k
static double _unur_pdf_ghyp( double x, const UNUR_DISTR *distr );
static double _unur_logpdf_ghyp( double x, const UNUR_DISTR *distr );
static double _unur_normconstant_ghyp( const double *params, int n_params );
/* static double _unur_dpdf_ghyp( double x, const UNUR_DISTR *distr ); */
/* static double _unur_cdf_ghyp( double x, const UNUR_DISTR *distr ); */
#endif

static int _unur_upd_center_ghyp( UNUR_DISTR *distr );
static int _unur_set_params_ghyp( UNUR_DISTR *distr, const double *params, int n_params );

/*---------------------------------------------------------------------------*/

#ifdef _unur_SF_bessel_k
double
_unur_pdf_ghyp(double x, const UNUR_DISTR *distr)
{ 
  /* f(x) = (delta^2+(x-mu)^2)^(1/2*(lambda-1/2)) * exp(beta*(x-mu))    */
  /*          * K_{lambda-1/2}(alpha*sqrt(delta^2+(x-mu)^2))            */

  return exp(_unur_logpdf_ghyp(x,distr));
} /* end of _unur_pdf_ghyp() */

/*---------------------------------------------------------------------------*/

double
_unur_logpdf_ghyp(double x, const UNUR_DISTR *distr)
{
  /* f(x) = (delta^2+(x-mu)^2)^(1/2*(lambda-1/2)) * exp(beta*(x-mu))    */
  /*          * K_{lambda-1/2}(alpha*sqrt(delta^2+(x-mu)^2))            */

  register const double *params = DISTR.params;
  double res = 0.;            /* result of computation                 */
  double nu = lambda - 0.5;   /* order of modified Bessel function K() */
  double y;                   /* auxiliary variable                    */

  y = sqrt(delta*delta + (x-mu)*(x-mu));

  /* Using nu and y we find:
   *   f(x) = y^nu * exp(beta*(x-mu) * K_nu(alpha*y)
   * and
   *   log(f(x)) =  nu*log(y) + beta*(x-mu) + log(K_nu(alpha*y)
   */

  /* see also c_vg.c */

  do {

    if (y>0.) {
      /* first simply compute Bessel K_nu and compute logarithm. */
      double besk;

      /* we currently use two algorithms based on our experiences
       * with the functions in the Rmath library.
       * (Maybe we could change this when we link against
       * other libraries.)
       */
      if (nu < 100)
        /* the "standard implementation" using log(bessel_k) */
        besk = _unur_SF_ln_bessel_k(alpha*y, nu);
      else
        /* an algorithm for large nu */
        besk = _unur_SF_bessel_k_nuasympt(alpha*y, nu, TRUE, FALSE);

      /* there can be numerical problems with the Bessel function K_nu. */
      if (_unur_isfinite(besk) && besk < MAXLOG - 20.0) {
        /* o.k. */
        res = LOGNORMCONSTANT + besk + nu*log(y) + beta*(x-mu);
        break;
      }
    }

    /* Case: numerical problems with Bessel function K_nu. */

    if (y < 1.0) {
      /* Case: Bessel function K_nu overflows for small values of y.
       * The following code is inspired by gsl_sf_bessel_lnKnu_e() from
       * the GSL (GNU Scientific Library).
       */

      res = LOGNORMCONSTANT + beta*(x-mu);
      res += -M_LN2 + _unur_SF_ln_gamma(nu) + nu*log(2./alpha);

      if (nu > 1.0) {
        double xi = 0.25*(alpha*y)*(alpha*y);
        double sum = 1.0 - xi/(nu-1.0);
        if(nu > 2.0) sum += (xi/(nu-1.0)) * (xi/(nu-2.0));
        res += log(sum);
      }
    }

    else {
      /* Case: Bessel function K_nu underflows for very large values of y
       * and we get NaN.
       * However, then the PDF of the Generalized Hyperbolic distribution is 0.
       */
      res = -UNUR_INFINITY;
    }
  } while(0);

  /* see also c_vg.c */

  return res;

} /* end of _unur_logpdf_ghyp() */

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

#endif

/*---------------------------------------------------------------------------*/

int
_unur_upd_center_ghyp( UNUR_DISTR *distr )
{
  register const double *params = DISTR.params;

  /* we simply use parameter 'mu' */
  DISTR.center = mu;

  /* an alternative approach would be the mean of the distribution:          */
  /* double gamma = sqrt(alpha*alpha-beta*beta);                             */
  /* DISTR.center = ( mu                                                     */
  /* 		   + ( delta*beta * bessel_k( delta*gamma, lambda+1, 1) )    */
  /* 		   / ( gamma * bessel_k( delta*gamma, lambda, 1) ) );        */

  /* center must be in domain */
  if (DISTR.center < DISTR.domain[0]) 
    DISTR.center = DISTR.domain[0];
  else if (DISTR.center > DISTR.domain[1]) 
    DISTR.center = DISTR.domain[1];

  return UNUR_SUCCESS;
} /* end of _unur_upd_center_ghyp() */

/*---------------------------------------------------------------------------*/

#ifdef _unur_SF_bessel_k
double
_unur_normconstant_ghyp(const double *params ATTRIBUTE__UNUSED, int n_params ATTRIBUTE__UNUSED)
{ 
  double gamm = sqrt(alpha*alpha-beta*beta);

  /* ( (gamm/delta)^lambda ) / ( sqrt(2*pi) * alpha^(lambda-1/2) * K_{lambda}(delta*gamm) ) */

  double logconst =  -0.5*(M_LNPI+M_LN2) + lambda * log(gamm/delta);
  logconst -= (lambda-0.5) * log(alpha);

  if (lambda < 50)
    /* threshold value 50 is selected by experiments */
    logconst -= _unur_SF_ln_bessel_k( delta*gamm, lambda );
  else
    logconst -= _unur_SF_bessel_k_nuasympt( delta*gamm, lambda, TRUE, FALSE );

  return logconst;
} /* end of _unur_normconstant_ghyp() */
#endif

/*---------------------------------------------------------------------------*/

int
_unur_set_params_ghyp( UNUR_DISTR *distr, const double *params, int n_params )
{
  /* check number of parameters for distribution */
  if (n_params < 5) {
    _unur_error(distr_name,UNUR_ERR_DISTR_NPARAMS,"too few"); return UNUR_ERR_DISTR_NPARAMS; }
  if (n_params > 5) {
    _unur_warning(distr_name,UNUR_ERR_DISTR_NPARAMS,"too many");
    n_params = 5; }
  CHECK_NULL(params,UNUR_ERR_NULL);

  /* check parameter omega */
  if (delta <= 0.) {
    _unur_error(distr_name,UNUR_ERR_DISTR_DOMAIN,"delta <= 0");
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
} /* end of _unur_set_params_ghyp() */

/*---------------------------------------------------------------------------*/

struct unur_distr *
unur_distr_ghyp( const double *params, int n_params)
{
  register struct unur_distr *distr;

  /* get new (empty) distribution object */
  distr = unur_distr_cont_new();

  /* set distribution id */
  distr->id = UNUR_DISTR_GHYP;

  /* name of distribution */
  distr->name = distr_name;
             
  /* how to get special generators */
  /* DISTR.init = _unur_stdgen_ghyp_init; */
   
  /* functions */
#ifdef _unur_SF_bessel_k
  DISTR.pdf     = _unur_pdf_ghyp;     /* pointer to PDF                  */
  DISTR.logpdf  = _unur_logpdf_ghyp;  /* pointer to log-PDF              */
#endif

  /* indicate which parameters are set */
#ifdef _unur_SF_bessel_k
  distr->set = ( UNUR_DISTR_SET_DOMAIN |
		 UNUR_DISTR_SET_STDDOMAIN |
		 UNUR_DISTR_SET_CENTER |
		 UNUR_DISTR_SET_PDFAREA );
#else
  distr->set = ( UNUR_DISTR_SET_DOMAIN | UNUR_DISTR_SET_STDDOMAIN );
#endif
                
  /* set parameters for distribution */
  if (_unur_set_params_ghyp(distr,params,n_params)!=UNUR_SUCCESS) {
    free(distr);
    return NULL;
  }

  /* normalization constant */
#ifdef _unur_SF_bessel_k
  LOGNORMCONSTANT = _unur_normconstant_ghyp(DISTR.params,DISTR.n_params);
#else
  LOGNORMCONSTANT = 0.;
#endif

  /* we need the center of the distribution */
  if (_unur_upd_center_ghyp(distr)!=UNUR_SUCCESS) {
    free(distr);
    return NULL;
  }

  /* mode and area below p.d.f. */
  /* DISTR.mode = ? */
#ifdef _unur_SF_bessel_k
  DISTR.area = 1;
#endif

  /* function for setting parameters and updating domain */
  DISTR.set_params = _unur_set_params_ghyp;

  /* function for updating derived parameters */
  /* DISTR.upd_mode  = _unur_upd_mode_ghyp; /\* funct for computing mode *\/ */
  /* DISTR.upd_area  = _unur_upd_area_ghyp; funct for computing area */

  /* return pointer to object */
  return distr;

} /* end of unur_distr_ghyp() */

/*---------------------------------------------------------------------------*/
#undef mu
#undef alpha
#undef beta
#undef delta
#undef lambda
#undef DISTR
/*---------------------------------------------------------------------------*/
