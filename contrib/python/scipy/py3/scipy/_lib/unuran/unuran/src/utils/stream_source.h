/*****************************************************************************
 *                                                                           *
 *          UNURAN -- Universal Non-Uniform Random number generator          *
 *                                                                           *
 *****************************************************************************
 *                                                                           *
 *   FILE: stream_source.h                                                   *
 *                                                                           *
 *   PURPOSE:                                                                *
 *         defines macros and function prototypes for input/output streams   *
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
void _unur_log_printf (const char *genid, const char *filename, int line, const char *format, ...)
  ATTRIBUTE__FORMAT(4,5);
void _unur_log_debug (const char *format, ...)
  ATTRIBUTE__FORMAT(1,2);

/*---------------------------------------------------------------------------*/

/* Read data from file into double array.                                    */
int _unur_read_data (const char *file, int no_of_entries, double **array);

/*---------------------------------------------------------------------------*/
