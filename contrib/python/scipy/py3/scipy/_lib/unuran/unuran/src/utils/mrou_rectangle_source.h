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
 *      Declarations for the bounding rectangle calculations used in         *
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

/*---------------------------------------------------------------------------*/
/* create MROU rectangle object.                                             */
/*---------------------------------------------------------------------------*/
struct MROU_RECTANGLE *_unur_mrou_rectangle_new( void );

/*---------------------------------------------------------------------------*/
/* compute (minimal) bounding hyper-rectangle.                               */
/*---------------------------------------------------------------------------*/
int _unur_mrou_rectangle_compute( struct MROU_RECTANGLE *rr );

/*---------------------------------------------------------------------------*/
