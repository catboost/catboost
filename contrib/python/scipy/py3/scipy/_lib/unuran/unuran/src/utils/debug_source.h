/*****************************************************************************
 *                                                                           *
 *          UNURAN -- Universal Non-Uniform Random number generator          *
 *                                                                           *
 *****************************************************************************
 *                                                                           *
 *   FILE: debug_source.h                                                    *
 *                                                                           *
 *   PURPOSE:                                                                *
 *         defines macros and function prototypes for debugging routines.    *
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
/* set generator id                                                          */

char *_unur_make_genid( const char *gentype );
#define _unur_set_genid(gentype) _unur_make_genid(gentype)
#define _unur_free_genid(gen)    do {if (gen->genid) free((gen)->genid);} while(0)

/*---------------------------------------------------------------------------*/
/* default debugging flag for generator                                      */

extern unsigned _unur_default_debugflag;     /* default debugging flags      */

/*---------------------------------------------------------------------------*/
/* warnings and error messages                                               */

/* an abbreviation */
#define _unur_print_if_default(par,flag)   if(!((par)->set & (flag))) fprintf(LOG,"  [default]")

/*---------------------------------------------------------------------------*/
/* Check for NULL pointer                                                    */

#ifdef UNUR_ENABLE_CHECKNULL

#define CHECK_NULL(ptr,rval)             \
  if (!(ptr)) {                          \
    _unur_error(NULL,UNUR_ERR_NULL,"");  \
    return rval;                         \
  }

#else               /* do not check (be carefull) */

#define CHECK_NULL(ptr,rval)  do {} while(0)

#endif

/* the second macro cannot be switched off by a compiler switch */
#define _unur_check_NULL(gid,ptr,rval)    \
  if (!(ptr)) {                           \
    _unur_error((gid),UNUR_ERR_NULL,"");  \
    return rval;                          \
  }

/* Some preprocessor (e.g. in MacOS X) fail when 'rval' is an empty string.  */
/* The following should fix this problem.                                    */

#define RETURN_VOID ;

/*---------------------------------------------------------------------------*/
