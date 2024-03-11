/*****************************************************************************
 *                                                                           *
 *          UNURAN -- Universal Non-Uniform Random number generator          *
 *                                                                           *
 *****************************************************************************
 *                                                                           *
 *   FILE: error_source.h                                                    *
 *                                                                           *
 *   PURPOSE:                                                                *
 *         defines macros and function prototypes error messages             *
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
#ifndef UNUR_ERROR_SOURCE_H_SEEN
#define UNUR_ERROR_SOURCE_H_SEEN
/*---------------------------------------------------------------------------*/

void _unur_error_x( const char *objid, const char *file, int line, 
		    const char *errortype, int errorcode, const char *reason );
/*---------------------------------------------------------------------------*/
/* set unur_errno and call error handler.                                    */
/*---------------------------------------------------------------------------*/

/* void _unur_error_formated( const char *objid, const char *file, int line,  */
/* 			   const char *errortype, int errorcode, const char *format, ... ); */
/*---------------------------------------------------------------------------*/
/* like _unur_error_x but used a template for more sophisticated messages.   */
/*---------------------------------------------------------------------------*/

void _unur_error_handler_default( const char *objid, const char *file, int line, 
				  const char *errortype, int errorcode, const char *reason );
/*---------------------------------------------------------------------------*/
/* unuran default error handler                                              */
/*---------------------------------------------------------------------------*/

void _unur_error_handler_off( const char *objid, const char *file, int line, 
			      const char *errortype, int errorcode, const char *reason );
/*---------------------------------------------------------------------------*/
/* disabled error handler (suppress all warnings and error messages          */
/*---------------------------------------------------------------------------*/

#ifdef UNUR_COOKIES
void _unur_error_cookies( const char *file, int line, unsigned observed, unsigned expected );
/*---------------------------------------------------------------------------*/
/* report invalid cookie                                                     */
/*---------------------------------------------------------------------------*/
#endif

#define _unur_error(genid,errorcode,reason) \
   do { \
      _unur_error_x((genid),__FILE__,__LINE__,"error",(errorcode),(reason)); \
   } while (0)
/*---------------------------------------------------------------------------*/
/* call error handler in case of a (fatal) error.                            */
/*---------------------------------------------------------------------------*/

#define _unur_warning(genid,errorcode,reason) \
   do { \
      _unur_error_x((genid),__FILE__,__LINE__,"warning",(errorcode),(reason)); \
   } while (0)
/*---------------------------------------------------------------------------*/
/* call error handler in case of a warning.                                  */
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
#endif  /* UNUR_ERROR_SOURCE_H_SEEN */
/*---------------------------------------------------------------------------*/
