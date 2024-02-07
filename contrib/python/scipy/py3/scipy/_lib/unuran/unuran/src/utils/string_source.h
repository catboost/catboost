/*****************************************************************************
 *                                                                           *
 *          UNURAN -- Universal Non-Uniform Random number generator          *
 *                                                                           *
 *****************************************************************************
 *                                                                           *
 *   FILE: string_source.h                                                   *
 *                                                                           *
 *   PURPOSE:                                                                *
 *         defines macros and function prototypes for strings                *
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

/* Make new string                                                           */
struct unur_string * _unur_string_new ( void );

/* Append to string                                                          */
/* Important: The generated string must not be longer than 1023 characters!  */
int _unur_string_append ( struct unur_string *string, const char *format, ... )
     ATTRIBUTE__FORMAT(2,3);

/* Append text to string                                                     */
int _unur_string_appendtext ( struct unur_string *string, const char *text );

/* Destroy string                                                            */
void _unur_string_free ( struct unur_string *string );

/* Clear string (set length of string to 0)                                  */
void _unur_string_clear ( struct unur_string *string );

/*---------------------------------------------------------------------------*/
