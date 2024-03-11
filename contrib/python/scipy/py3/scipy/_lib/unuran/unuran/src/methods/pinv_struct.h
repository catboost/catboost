/*****************************************************************************
 *                                                                           *
 *          UNURAN -- Universal Non-Uniform Random number generator          *
 *                                                                           *
 *****************************************************************************
 *                                                                           *
 *   FILE: pinv_struct.h                                                     *
 *                                                                           *
 *   PURPOSE:                                                                *
 *         declares structures for method PINV                               *
 *         (Polynomial interpolation based INVersion of CDF)                 *
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

struct unur_pinv_par { 
  int order;               /* order of interpolating polynomial              */
  int smooth;              /* smoothness parameter                           */
  double u_resolution;     /* maximal error in u                             */
  double bleft;            /* left boundary of the computational domain      */
  double bright;           /* right boundary of the computational domain     */
  int sleft;               /* whether to search for left boundary point      */
  int sright;              /* whether to search for right boundary point     */
  int max_ivs;             /* maximum number of subintervals                 */
};

/*---------------------------------------------------------------------------*/
/* store information about splines                                           */

struct unur_pinv_interval {
  double *ui;  /* points for constructing Newton interpolation */
  double *zi;  /* values of inverse CDF at these U values */
  double xi;   /* left point of interval */
  double cdfi; /* CDF at left point of interval */

#ifdef UNUR_COOKIES
  unsigned cookie;         /* magic cookie                                    */
#endif
};

/*---------------------------------------------------------------------------*/
/* The generator object                                                      */

struct unur_pinv_gen { 
  int order;               /* order of interpolating polynomial              */

  int    *guide;           /* pointer to guide table                         */ 
  int     guide_size;      /* size of guide table                            */

  double  Umax;            /* upper bound for uniform random variable U 
			      [ Umin = 0. ]                                  */

  double  u_resolution;    /* maximal error in u                             */
  int smooth;              /* smoothness parameter                           */
  double  bleft;           /* left border of the computational domain        */
  double  bright;          /* right border of the computational domain       */

  struct unur_pinv_interval *iv; /* list of intervals                        */
  int n_ivs;               /* number of subintervals                         */
  int max_ivs;             /* maximum number of subintervals                 */


  double  bleft_par;       /* border of the computational domain as ...      */
  double  bright_par;      /* ... given by user                              */

  double  dleft;           /* left and right boundary of domain / support    */
  double  dright;          /* of distribution                                */

  int sleft;               /* whether to search for left boundary point      */
  int sright;              /* whether to search for right boundary point     */

  double area;             /* approximate area below PDF                     */ 
  /* double logPDFconstant;  rescaling constant for logPDF: Disabled!        */

  struct unur_lobatto_table *aCDF; /* polynomial approximation of CDF        */
};

/*---------------------------------------------------------------------------*/
