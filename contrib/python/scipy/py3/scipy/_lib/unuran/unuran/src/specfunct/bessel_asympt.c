/*---------------------------------------------------------------------------*/
/*                                                                           */
/* Asymptotic expnasion of Bessel functions for large orders nu.             */
/*                                                                           */
/*---------------------------------------------------------------------------*/

#include <unur_source.h>

/*---------------------------------------------------------------------------*/

double
_unur_bessel_k_nuasympt (double x, double nu, int islog, int expon_scaled)
/*---------------------------------------------------------------------------*/
/* Asymptotic expansion of Bessel K_nu(x) function                           */
/* when BOTH  nu and x  are large.                                           */
/*                                                                           */
/* parameters:                                                               */
/*   x            ... argument for K_nu()                                    */
/*   nu           ... order or Bessel function                               */
/*   islog        ... return logarithm of result TRUE and result when FALSE  */
/*   expon_scaled ... return exp(-x)*K_nu(x) when TRUE and K_nu(x) when FALSE*/
/*                                                                           */
/*---------------------------------------------------------------------------*/
/*                                                                           */
/* references:                                                               */
/* ##  Abramowitz & Stegun , p.378, __ 9.7.8. __                             */
/*                                                                           */
/* ## K_nu(nu * z) ~ sqrt(pi/(2*nu)) * exp(-nu*eta)/(1+z^2)^(1/4)            */
/* ##                   * {1 - u_1(t)/nu + u_2(t)/nu^2 - ... }               */
/*                                                                           */
/* ## where   t = 1 / sqrt(1 + z^2),                                         */
/* ##       eta = sqrt(1 + z^2) + log(z / (1 + sqrt(1+z^2)))                 */
/* ##                                                                        */
/* ## and u_k(t)  from  p.366  __ 9.3.9 __                                   */
/*                                                                           */
/* ## u0(t) = 1                                                              */
/* ## u1(t) = (3*t - 5*t^3)/24                                               */
/* ## u2(t) = (81*t^2 - 462*t^4 + 385*t^6)/1152                              */
/* ## ...                                                                    */
/*                                                                           */
/* ## with recursion  9.3.10    for  k = 0, 1, .... :                        */
/* ##                                                                        */
/* ## u_{k+1}(t) = t^2/2 * (1 - t^2) * u'_k(t) +                             */
/* ##            1/8  \int_0^t (1 - 5*s^2)* u_k(s) ds                        */
/*---------------------------------------------------------------------------*/
/*                                                                           */
/* Original implementation in R code (R package "Bessel" v. 0.5-3) by        */
/*   Martin Maechler, Date: 23 Nov 2009, 13:39                               */
/*                                                                           */
/* Translated into C code by Kemal Dingic, Oct. 2011.                        */
/*                                                                           */
/* Modified by Josef Leydold on Tue Nov  1 13:22:09 CET 2011                 */
/*                                                                           */
/*---------------------------------------------------------------------------*/
{
  double z;                   /* rescaled argument for K_nu() */
  double sz, t, t2, eta;      /* auxiliary variables */
  double d, u1t,u2t,u3t,u4t;  /* (auxiliary) results for Debye polynomials */
  double res;                 /* value of log(K_nu(x)) [= result] */
  
  /* rescale: we comute K_nu(z * nu) */
  z = x / nu;

  /* auxiliary variables */
  sz = hypot(1,z);   /* = sqrt(1+z^2) */
  t = 1. / sz;
  t2 = t*t;

  eta = (expon_scaled) ? (1./(z + sz)) : sz;
  eta += log(z) - log1p(sz);                  /* = log(z/(1+sz)) */

  /* evaluate Debye polynomials u_j(t) */
  u1t = (t * (3. - 5.*t2))/24.;
  u2t = t2 * (81. + t2*(-462. + t2 * 385.))/1152.;
  u3t = t*t2 * (30375. + t2 * (-369603. + t2 * (765765. - t2 * 425425.)))/414720.;
  u4t = t2*t2 * (4465125. 
		 + t2 * (-94121676.
			 + t2 * (349922430. 
				 + t2 * (-446185740. 
					 + t2 * 185910725.)))) / 39813120.;
  d = (-u1t + (u2t + (-u3t + u4t/nu)/nu)/nu)/nu;

  /* log(K_nu(x)) */
  res = log(1.+d) - nu*eta - 0.5*(log(2.*nu*sz) - M_LNPI);

  return (islog ? res : exp(res));
} /* end of _unur_bessel_k_nuasympt() */

/*---------------------------------------------------------------------------*/
