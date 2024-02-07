/*****************************************************************************
 *                                                                           *
 *          UNURAN -- Universal Non-Uniform Random number generator          *
 *                                                                           *
 *****************************************************************************
 *                                                                           *
 *   FILE: hitro_struct.h                                                    *
 *                                                                           *
 *   PURPOSE:                                                                *
 *         declares structures for method HITRO                              *
 *         (Markov Chain - HIT-and-run sampler with Ratio-Of-uniforms)       *
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

struct unur_hitro_par { 
  double r;                  /* r-parameter of HITRO method                  */
  int thinning;              /* thinning factor for generated chain          */
  int burnin;                /* length of burn-in for chain                  */
  double adaptive_mult;      /* multiplier for adaptive rectangle            */
  double vmax;               /* bounding rectangle v-coordinate              */
  const double *umin, *umax; /* bounding rectangle u-coordinates             */
  const double *x0;          /* starting point of chain                      */
};

/*---------------------------------------------------------------------------*/
/* The generator object                                                      */

struct unur_hitro_gen {
  int dim;                   /* dimension of distribution                    */
  int thinning;              /* thinning factor for generated chain          */
  double r;                  /* r-parameter of HITRO method                  */

  double *state;             /* state of chain / current point.
				the state is a point in the acceptance region
				of the RoU-method in vu-hyperplane.
				the coordinates are stored in the following order:
				state = {v, u_1, u_2, ... u_n}               */

  int    coord;              /* current coordinate used for HITRO chain      */
  double *direction;         /* working array for random direction           */

  double *vu;                /* working point in RoU scale                   */
  double *vumin, *vumax;     /* vertices of bounding rectangles (vu-coordinates) 
				(see 'state' variable for location of v and 
				u-coordinates.)                              */
  
  double *x;                 /* working point in the original scale          */
  const double *center;      /* center of distribution                       */

  double adaptive_mult;      /* multiplier for adaptive rectangles           */
  int burnin;                /* length of burn-in for chain                  */
  double *x0;                /* starting point of chain                      */
  double fx0;                /* PDF at starting point of chain               */
};

/*---------------------------------------------------------------------------*/
