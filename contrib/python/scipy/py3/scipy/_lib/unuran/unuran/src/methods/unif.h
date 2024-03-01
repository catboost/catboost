/*****************************************************************************
 *                                                                           *
 *          UNURAN -- Universal Non-Uniform Random number generator          *
 *                                                                           *
 *****************************************************************************
 *                                                                           *
 *   FILE: unif.h                                                            *
 *                                                                           *
 *   PURPOSE:                                                                *
 *         function prototypes for method UNIF                               *
 *         (passes UNIForm random numbers through UNU.RAN framework;         *
 *         for testing only)                                                 *
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

/*
   =METHOD  UNIF  wrapper for UNIForm random number generator

   =UP  Methods_for_UNID

   =REINIT supported

   =DESCRIPTION
      UNIF is a simple wrapper that makes it possible to use a uniform
      random number generator as a UNU.RAN generator. There are no
      parameters for this method.

   =HOWTOUSE
      Create a generator object with NULL as argument. The created generator
      object returns raw random numbers from the underlying uniform 
      random number generator.

   =END
*/

/*---------------------------------------------------------------------------*/
/* Routines for user interface                                               */

/* =ROUTINES */

UNUR_PAR *unur_unif_new( const UNUR_DISTR *dummy );
/* 
   Get default parameters for generator.                                     
   UNIF does not need a distribution object. @var{dummy} is not used and
   can (should) be set to NULL. It is used to keep the API consistent.
*/

/* =END */

/*---------------------------------------------------------------------------*/












