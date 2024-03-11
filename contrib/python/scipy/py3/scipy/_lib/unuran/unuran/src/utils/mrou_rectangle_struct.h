/*****************************************************************************
 *                                                                           *
 *          UNURAN -- Universal Non-Uniform Random number generator          *
 *                                                                           *
 *****************************************************************************
 *                                                                           *
 *   FILE:      rou_rectangle_source.c                                       *
 *                                                                           *
 *                                                                           *
 *   DESCRIPTION:                                                            *
 *      Structure needed for the bounding rectangle calculations used in     *
 *      the multivariate RoU-methods.                                        *
 *                                                                           *
 *****************************************************************************
 *                                                                           *
 *   Copyright (c) 2000-2022 Wolfgang Hoermann and Josef Leydold             *
 *   Department of Statistics and Mathematics, WU Wien, Austria              *
 *   SPDX-License-Identifier: BSD-3-Clause                                   *
 *                                                                           *

 *                                                                           *
 *****************************************************************************/

struct MROU_RECTANGLE {
  UNUR_DISTR *distr;        /* distribution object                           */
  int    dim;               /* dimension of distribution                     */
  double r;	            /* r-parameter of the mrou method 	             */
  int bounding_rectangle;   /* flag to calculate bounding rectangle / strip  */
  double *umin, *umax;      /* boundary rectangle u-coordinates              */
  double vmax;              /* boundary rectangle v-coordinate               */
  const double *center;     /* center of distribution                        */
  int aux_dim;              /* parameter used in auxiliary functions         */
  const char *genid;        /* generator id */
};

/*---------------------------------------------------------------------------*/
