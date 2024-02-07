/*****************************************************************************
 *                                                                           *
 *          UNURAN -- Universal Non-Uniform Random number generator          *
 *                                                                           *
 *****************************************************************************
 *                                                                           *
 *   FILE: parser_source.h                                                   *
 *                                                                           *
 *   PURPOSE:                                                                *
 *         function prototypes for parser                                    *
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

char *_unur_parser_prepare_string( const char *str );
/*---------------------------------------------------------------------------*/
/* Prepare string for processing:                                            */
/*   Make a working copy of the str, remove all white spaces and convert to  */
/*   lower case letters.                                                     */
/* The string returned by this call should be freed when it is not required  */
/* any more.                                                                 */
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/





