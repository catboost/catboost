/*****************************************************************************
 *                                                                           *
 *          UNURAN -- Universal Non-Uniform Random number generator          *
 *                                                                           *
 *****************************************************************************
 *                                                                           *
 *   FILE: debug.c                                                           *
 *                                                                           *
 *   debugging routines                                                      *
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

#include <unur_source.h>

/*---------------------------------------------------------------------------*/

/*****************************************************************************/
/**                                                                         **/
/**  Set debuging flags                                                     **/
/**                                                                         **/
/*****************************************************************************/

/*---------------------------------------------------------------------------*/

/* global variable for default debugging flags */
unsigned _unur_default_debugflag = UNUR_DEBUGFLAG_DEFAULT;

/*---------------------------------------------------------------------------*/

int
unur_set_debug( struct unur_par *par ATTRIBUTE__UNUSED,
		unsigned debug ATTRIBUTE__UNUSED )
     /*----------------------------------------------------------------------*/
     /* set debugging flag for generator                                     */
     /*                                                                      */
     /* parameters:                                                          */
     /*   par   ... pointer to parameter for building generator object       */
     /*   debug ... debugging flag                                           */
     /*----------------------------------------------------------------------*/
{
#ifdef UNUR_ENABLE_LOGGING
  _unur_check_NULL( NULL,par,UNUR_ERR_NULL );
  par->debug = debug;
  return UNUR_SUCCESS;
#else
  _unur_warning("DEBUG",UNUR_ERR_COMPILE,"debugging not enabled");
  return UNUR_ERR_COMPILE;
#endif

} /* end of unur_set_debug() */
  
/*---------------------------------------------------------------------------*/

int
unur_chg_debug( struct unur_gen *gen ATTRIBUTE__UNUSED,
		unsigned debug ATTRIBUTE__UNUSED )
     /*----------------------------------------------------------------------*/
     /* change debugging flag for generator                                  */
     /*                                                                      */
     /* parameters:                                                          */
     /*   gen   ... pointer to generator object                              */
     /*   debug ... debugging flag                                           */
     /*----------------------------------------------------------------------*/
{
#ifdef UNUR_ENABLE_LOGGING
  CHECK_NULL( gen, UNUR_ERR_NULL );
  gen->debug = debug;
  return UNUR_SUCCESS;
#else
  _unur_warning("DEBUG",UNUR_ERR_COMPILE,"debugging not enabled");
  return UNUR_ERR_COMPILE;
#endif

} /* end of unur_chg_debug() */
  
/*---------------------------------------------------------------------------*/

int
unur_set_default_debug( unsigned debug )
     /*----------------------------------------------------------------------*/
     /* set default debugging flag for generator                             */
     /*                                                                      */
     /* parameters:                                                          */
     /*   par   ... pointer to parameter for building generator object       */
     /*   debug ... debugging flag                                           */
     /*----------------------------------------------------------------------*/
{
  _unur_default_debugflag = debug;
  return UNUR_SUCCESS;
} /* end of unur_set_default_debug() */
  
/*---------------------------------------------------------------------------*/

char * 
_unur_make_genid( const char *gentype )
     /*----------------------------------------------------------------------*/
     /* make a new generator identifier                                      */
     /*                                                                      */
     /* parameters:                                                          */
     /*   gentype ... type of generator                                      */
     /*                                                                      */
     /* return:                                                              */
     /*   pointer generator id (char string)                                 */
     /*----------------------------------------------------------------------*/
{
  static int count = 0;   /* counter for identifiers */
  char *genid;
  size_t len;

  /* allocate memory for identifier */
  len = strlen(gentype);
  genid = _unur_xmalloc(sizeof(char)*(len+5));

  /* make new identifier */
  ++count; count %= 1000;      
  /* 1000 different generators should be enough */

#if HAVE_DECL_SNPRINTF
  /* this is a GNU extension */
  snprintf(genid, len+5, "%s.%03d", gentype, count);
#else
  sprintf(genid, "%s.%03d", gentype, count);
#endif

  return genid;

} /* end of _unur_make_genid() */

/*---------------------------------------------------------------------------*/

