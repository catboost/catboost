/*****************************************************************************
 *                                                                           *
 *          UNURAN -- Universal Non-Uniform Random number generator          *
 *                                                                           *
 *****************************************************************************
 *                                                                           *
 *   FILE: lobatto_struct.h                                                  *
 *                                                                           *
 *   PURPOSE:                                                                *
 *         declares structures for Gauss-Lobatto integration                 *
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
/* 'Lobatto object':                                                         */
/*   store integrand, boundaries and integrals of subintervals computed      */
/*   during adaptive Gauss-Lobatto integration.                              */

struct unur_lobatto_nodes {
  double x;   /* right boundary of subinterval */
  double u;   /* integral of PDF over subinterval */
}; 

struct unur_lobatto_table {
  struct unur_lobatto_nodes *values; /* boundaries and integral values       */
  int n_values;              /* number of stored integral values (nodes)     */
  int cur_iv;                /* position of first entry whose x value is
				larger than left boundary of current interval*/
  int size;                  /* size of table                                */
  
  UNUR_LOBATTO_FUNCT *funct; /* pointer to integrand                         */
  struct unur_gen *gen;      /* pointer to generator object                  */
  double tol;                /* tolerated ABSOLUTE integration error         */
  UNUR_LOBATTO_ERROR *uerror; /* function for estimating error               */
  double bleft;              /* left boundary of computational domain        */
  double bright;             /* right boundary of computational domain       */
  double integral;           /* integral over whole domain                   */
}; 


/*---------------------------------------------------------------------------*/
