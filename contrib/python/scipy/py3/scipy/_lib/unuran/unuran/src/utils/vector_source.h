/*****************************************************************************
 *                                                                           *
 *          UNURAN -- Universal Non-Uniform Random number generator          *
 *                                                                           *
 *****************************************************************************
 *                                                                           *
 *   FILE: vector_source.h                                                   *
 *                                                                           *
 *   Routines for computations with vectors (arrays).                        *
 *                                                                           *
 *****************************************************************************
 *                                                                           *
 *   Copyright (c) 2000-2022 Wolfgang Hoermann and Josef Leydold             *
 *   Department of Statistics and Mathematics, WU Wien, Austria              *
 *   SPDX-License-Identifier: BSD-3-Clause                                   *
 *                                                                           *

 *                                                                           *
 *****************************************************************************/
				                                                                                    
/*--------------------------------------------------------------------------*/

double *_unur_vector_new(int dim);
/* allocate memory for new vector and initialize to 0 */

void _unur_vector_free(double *v);
/* free allocated memory used by vector */

double _unur_vector_norm(int dim, double *v);
/* calculation of euclidean (L2) norm of vector */

double _unur_vector_scalar_product(int dim, double *v1, double *v2);
/* calculation of scalar product */

void _unur_vector_normalize(int dim, double *v);
/* normalize a vector to have unit norm */

/*---------------------------------------------------------------------------*/
