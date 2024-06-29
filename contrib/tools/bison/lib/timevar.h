/* Timing variables for measuring application performance.

   Copyright (C) 2000, 2002, 2004, 2009-2015, 2018-2019 Free Software
   Foundation, Inc.

   Contributed by Alex Samuel <samuel@codesourcery.com>

   This program is free software: you can redistribute it and/or modify
   it under the terms of the GNU General Public License as published by
   the Free Software Foundation, either version 3 of the License, or
   (at your option) any later version.

   This program is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
   GNU General Public License for more details.

   You should have received a copy of the GNU General Public License
   along with this program.  If not, see <http://www.gnu.org/licenses/>.  */

#ifndef _TIMEVAR_H
# define _TIMEVAR_H 1

# include <stdio.h>

# include "xtime.h"

# ifdef  __cplusplus
extern "C" {
# endif

/* Timing variables are used to measure elapsed time in various
   portions of the application.  Each measures elapsed user, system, and
   wall-clock time, as appropriate to and supported by the host
   system.

   Timing variables are defined using the DEFTIMEVAR macro in
   timevar.def.  Each has an enumeral identifier, used when referring
   to the timing variable in code, and a character string name.

   Timing variables can be used in two ways:

     - On the timing stack, using timevar_push and timevar_pop.
       Timing variables may be pushed onto the stack; elapsed time is
       attributed to the topmost timing variable on the stack.  When
       another variable is pushed on, the previous topmost variable is
       'paused' until the pushed variable is popped back off.

     - As a standalone timer, using timevar_start and timevar_stop.
       All time elapsed between the two calls is attributed to the
       variable.
*/

/* This structure stores the various varieties of time that can be
   measured.  Times are stored in seconds.  The time may be an
   absolute time or a time difference; in the former case, the time
   base is undefined, except that the difference between two times
   produces a valid time difference.  */

struct timevar_time_def
{
  /* User time in this process.  */
  xtime_t user;

  /* System time (if applicable for this host platform) in this
     process.  */
  xtime_t sys;

  /* Wall clock time.  */
  xtime_t wall;
};

/* An enumeration of timing variable identifiers.  Constructed from
   the contents of timevar.def.  */

#define DEFTIMEVAR(identifier__, name__) \
    identifier__,
typedef enum
{
#include "timevar.def"
  TIMEVAR_LAST
}
timevar_id_t;
#undef DEFTIMEVAR

/* Initialize timing variables.  */

void timevar_init (void);

/* Push TIMEVAR onto the timing stack.  No further elapsed time is
   attributed to the previous topmost timing variable on the stack;
   subsequent elapsed time is attributed to TIMEVAR, until it is
   popped or another element is pushed on top.

   TIMEVAR cannot be running as a standalone timer.  */

void timevar_push (timevar_id_t timevar);

/* Pop the topmost timing variable element off the timing stack.  The
   popped variable must be TIMEVAR.  Elapsed time since the that
   element was pushed on, or since it was last exposed on top of the
   stack when the element above it was popped off, is credited to that
   timing variable.  */

void timevar_pop (timevar_id_t timevar);

/* Start timing TIMEVAR independently of the timing stack.  Elapsed
   time until timevar_stop is called for the same timing variable is
   attributed to TIMEVAR.  */

void timevar_start (timevar_id_t timevar);

/* Stop timing TIMEVAR.  Time elapsed since timevar_start was called
   is attributed to it.  */

void timevar_stop (timevar_id_t timevar);

/* Fill the elapsed time for TIMEVAR into ELAPSED.  Returns
   update-to-date information even if TIMEVAR is currently running.  */

void timevar_get (timevar_id_t timevar, struct timevar_time_def *elapsed);

/* Summarize timing variables to FP.  The timing variable TV_TOTAL has
   a special meaning -- it's considered to be the total elapsed time,
   for normalizing the others, and is displayed last.  */

void timevar_print (FILE *fp);

/* Set to to nonzero to enable timing variables.  All the timevar
   functions make an early exit if timevar is disabled.  */

extern int timevar_enabled;

# ifdef  __cplusplus
}
# endif

#endif /* ! _TIMEVAR_H */
