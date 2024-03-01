/*****************************************************************************
 *                                                                           *
 *          UNURAN -- Universal Non-Uniform Random number generator          *
 *                                                                           *
 *****************************************************************************
 *                                                                           *
 *   FILE: string_struct.h                                                   *
 *                                                                           *
 *   PURPOSE:                                                                *
 *         declares structures for strings                                   *
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
/* String                                                                    */

struct unur_string {
  char *text;           /* pointer to string text                            */
  int   length;         /* length of string                                  */
  int   allocated;      /* length allocated memory block                     */
};

/*---------------------------------------------------------------------------*/
