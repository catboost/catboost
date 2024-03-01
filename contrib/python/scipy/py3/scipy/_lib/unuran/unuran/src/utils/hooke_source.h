/*****************************************************************************
 *                                                                           *
 *          UNURAN -- Universal Non-Uniform Random number generator          *
 *                                                                           *
 *****************************************************************************
 *                                                                           *
 *   FILE: hooke_source.h                                                    *
 *                                                                           *
 *   PURPOSE:                                                                *
 *         defines function prototypes for finding extremum of multivariate  *
 *         function                                                          *
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
/* Function prototypes                                                       */

/* Algorithm by Hooke and Jeeves                                             */
/* (direct search minimization algorithm)                                    */
int _unur_hooke( struct unur_funct_vgeneric faux, 
           int dim, double *startpt, double *endpt, 
           double rho, double epsilon, long itermax);

/*---------------------------------------------------------------------------*/
