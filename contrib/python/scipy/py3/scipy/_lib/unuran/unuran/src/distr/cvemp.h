/*****************************************************************************
 *                                                                           *
 *          UNURAN -- Universal Non-Uniform Random number generator          *
 *                                                                           *
 *****************************************************************************
 *                                                                           *
 *   FILE: cvemp.h                                                           *
 *                                                                           *
 *   PURPOSE:                                                                *
 *         function prototypes for manipulating distribution objects of      *
 *         type  CVEMP  (continuous empirical multivariate distribution)     *
 *                                                                           *
 *   USAGE:                                                                  *
 *         only included in unuran.h                                         *
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

/* 
   =NODEX   CVEMP   Continuous empirical multivariate distributions

   =UP Distribution_objects [40]

   =DESCRIPTION
      Empirical multivariate distributions are just lists of vectors
      (with the same dimension). 
      Thus there are only calls to insert these data.
      How these data are used to sample from the empirical distribution
      depends from the chosen generation method.  

   =END
*/

/*---------------------------------------------------------------------------*/

/* 
   Routines for handling empirical multivariate continuous distributions (VCEMP).
*/

/* =ROUTINES */

UNUR_DISTR *unur_distr_cvemp_new( int dim ); 
/* 
   Create a new (empty) object for an empirical multivariate
   continuous distribution. @var{dim} is the number of components of
   the random vector (i.e. its dimension). It must be at least 2;
   otherwise unur_distr_cemp_new() should be used to create an object
   for an empirical univariate distribution.
*/

/* ==DOC
   @subsubheading Essential parameters
*/

int unur_distr_cvemp_set_data( UNUR_DISTR *distribution, const double *sample, int n_sample );
/* 
   Set observed sample for empirical @var{distribution}.
   @var{sample} is an array of doubles of size 
   @code{dim} x @var{n_sample}, where
   @code{dim} is the dimension of the @var{distribution} returned by
   unur_distr_get_dim(). 
   The data points must be stored consecutively in @var{sample}, i.e.,
   data points (x1, y1), (x2, y2), @dots{} are given as an array
   @{x1, y1, x2, y2, @dots{}@}.
*/

int unur_distr_cvemp_read_data( UNUR_DISTR *distribution, const char *filename );
/* 
   Read data from file @file{filename}.
   It reads the first @code{dim} numbers from each line, where
   @code{dim} is the dimension of the @var{distribution} returned by
   unur_distr_get_dim(). 
   Numbers are parsed by means of the C standard routine @command{strtod}.
   Lines that do not start with @code{+}, @code{-}, @code{.}, or a
   digit are ignored. (Beware of lines starting with a blank!)

   In case of an error (file cannot be opened, too few entries in a
   line, invalid string for double in line) no data are copied into
   the distribution object and an error code is returned.
*/

int unur_distr_cvemp_get_data( const UNUR_DISTR *distribution, const double **sample );
/* 
   Get number of samples and set pointer @var{sample} to array of
   observations. If no sample has been given, an error code 
   is returned and @var{sample} is set to NULL.
   If successful @var{sample} points to an array of length
   @code{dim} x @code{n_sample}, where
   @code{dim} is the dimension of the distribution returned by
   unur_distr_get_dim() and @code{n_sample} the return value of the
   function.

   @emph{Important:} Do @strong{not} modify the array @var{sample}.
*/

/* =END */

/*---------------------------------------------------------------------------*/
