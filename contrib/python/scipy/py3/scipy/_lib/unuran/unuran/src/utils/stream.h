/*****************************************************************************
 *                                                                           *
 *          UNURAN -- Universal Non-Uniform Random number generator          *
 *                                                                           *
 *****************************************************************************
 *                                                                           *
 *   FILE: stream.h                                                          *
 *                                                                           *
 *   PURPOSE:                                                                *
 *         routines for output streams and reading data                      *
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
   =NODE  Output_streams Output streams

   =UP Error_Debug [10] 

   =DESCRIPTION
      @cindex Error handlers
      @cindex Output streams

      UNU.RAN uses a logfile for writing all error messages, warnings,
      and debugging information onto an output stream. This stream can
      be set at runtime by the unur_set_stream() call. 
      If no such stream is given by the user a default stream is used
      by the library: all messages are written into the file  
      @file{unuran.log} in the current working directory. The name of
      this logfile is defined by the macro @code{UNUR_LOG_FILE} in 
      @file{unuran_config.h}. 
      (If UNU.RAN fails to open this file for writing, @file{stderr}
      is used instead.)

      To destinguish between messages for different objects each of
      these has its own identifier which is composed by the name of
      the distribution obejct and generator type, resp., followed by a
      dot and three digits. 
      (If there are more than 999 generators then the identifiers are
      not unique.)

      @emph{Remark:} Writting debugging information must be switched
      on at compile time using the configure flag
      @code{--enable-logging}, see @ref{Debug,,Debugging}.

   =END      
*/

/* =ROUTINES */

/*---------------------------------------------------------------------------*/
/* manipulate output stream                                                  */

FILE *unur_set_stream( FILE *new_stream );
/*
   This function sets a new file handler for the output stream,
   @var{new_stream}, for the UNU.RAN library routines. The previous
   handler is returned (so that you can restore it later). 
   Note that the pointer to a user defined file handler is stored in a
   static variable, so there can be only one output stream handler per
   program. This function should be not be used in multi-threaded
   programs except to set up a program-wide error handler from a
   master thread.

   The NULL pointer is not allowed. 
   (If you want to disable logging of debugging information use 
   unur_set_default_debug(UNUR_DEBUG_OFF) instead.
   If you want to disable error messages at all use
   unur_set_error_handler_off().)
*/

FILE *unur_get_stream( void );
/*
  Get the file handle for the current output stream. It can be used to
  allow applications to write additional information into the logfile.
*/

/* =END */

/*---------------------------------------------------------------------------*/
