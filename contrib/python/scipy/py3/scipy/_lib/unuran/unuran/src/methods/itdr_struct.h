/*****************************************************************************
 *                                                                           *
 *          UNURAN -- Universal Non-Uniform Random number generator          *
 *                                                                           *
 *****************************************************************************
 *                                                                           *
 *   FILE: itdr_struct.h                                                     *
 *                                                                           *
 *   PURPOSE:                                                                *
 *         declares structures for method ITDR                               *
 *         (Inverse Transformed Density Rejection)                           *
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

struct unur_itdr_par { 
  double xi;                 /* intersection point lc(x)=ilc(x)              */
  double cp, ct;             /* c-value for pole and tail region, resp.      */
};

/*---------------------------------------------------------------------------*/
/* The generator object                                                      */

struct unur_itdr_gen { 
  double bx;                 /* splitting point between pole and tail region */
  double Atot;               /* total area below hat                         */     
  double Ap, Ac, At;         /* areas in upper pole, center, and tail region */     
  double cp, xp;             /* c-value and design point for pole region     */
  double alphap, betap;      /* parameters for hat in pole region            */
  double by;                 /* hat of pole region at bx                     */
  double sy;                 /* PDF(bx) = squeeze for central region         */
  double ct, xt;             /* c-value and design point for tail region     */
  double Tfxt, dTfxt;        /* parameters for hat in tail region            */
  double pole;               /* location of pole                             */
  double bd_right;           /* right boundary of shifted domain             */
  double sign;               /* region: +1 ... (-oo,0], -1 ... [0,oo)        */
  double xi;                 /* intersection point lc(x)=ilc(x)              */
};

/*---------------------------------------------------------------------------*/
