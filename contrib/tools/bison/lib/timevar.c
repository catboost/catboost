/* Timing variables for measuring compiler performance.

   Copyright (C) 2000, 2002, 2004, 2006, 2009-2015, 2018-2019 Free Software
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

#include <config.h>

/* Specification.  */
#include "timevar.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "gethrxtime.h"
#include "gettext.h"
#define _(msgid) gettext (msgid)
#include "xalloc.h"

/* See timevar.h for an explanation of timing variables.  */

int timevar_enabled = 0;

/* A timing variable.  */

struct timevar_def
{
  /* Elapsed time for this variable.  */
  struct timevar_time_def elapsed;

  /* If this variable is timed independently of the timing stack,
     using timevar_start, this contains the start time.  */
  struct timevar_time_def start_time;

  /* The name of this timing variable.  */
  const char *name;

  /* Non-zero if this timing variable is running as a standalone
     timer.  */
  unsigned standalone : 1;

  /* Non-zero if this timing variable was ever started or pushed onto
     the timing stack.  */
  unsigned used : 1;
};

/* An element on the timing stack.  Elapsed time is attributed to the
   topmost timing variable on the stack.  */

struct timevar_stack_def
{
  /* The timing variable at this stack level.  */
  struct timevar_def *timevar;

  /* The next lower timing variable context in the stack.  */
  struct timevar_stack_def *next;
};

/* Declared timing variables.  Constructed from the contents of
   timevar.def.  */
static struct timevar_def timevars[TIMEVAR_LAST];

/* The top of the timing stack.  */
static struct timevar_stack_def *stack;

/* A list of unused (i.e. allocated and subsequently popped)
   timevar_stack_def instances.  */
static struct timevar_stack_def *unused_stack_instances;

/* The time at which the topmost element on the timing stack was
   pushed.  Time elapsed since then is attributed to the topmost
   element.  */
static struct timevar_time_def start_time;

/* Fill the current times into TIME.  */

static void
set_to_current_time (struct timevar_time_def *now)
{
  now->user = 0;
  now->sys  = 0;
  now->wall = 0;

  if (!timevar_enabled)
    return;
  /*
  struct rusage self;
  getrusage (RUSAGE_SELF, &self);
  struct rusage chld;
  getrusage (RUSAGE_CHILDREN, &chld);

  now->user =
    xtime_make (self.ru_utime.tv_sec + chld.ru_utime.tv_sec,
                (self.ru_utime.tv_usec + chld.ru_utime.tv_usec) * 1000);

  now->sys =
    xtime_make (self.ru_stime.tv_sec + chld.ru_stime.tv_sec,
                (self.ru_stime.tv_usec + chld.ru_stime.tv_usec) * 1000);
  */
  now->wall = gethrxtime();
}

/* Return the current time.  */

static struct timevar_time_def
get_current_time (void)
{
  struct timevar_time_def now;
  set_to_current_time (&now);
  return now;
}

/* Add the difference between STOP and START to TIMER.  */

static void
timevar_accumulate (struct timevar_time_def *timer,
                    const struct timevar_time_def *start,
                    const struct timevar_time_def *stop)
{
  timer->user += stop->user - start->user;
  timer->sys += stop->sys - start->sys;
  timer->wall += stop->wall - start->wall;
}

void
timevar_init ()
{
  if (!timevar_enabled)
    return;

  /* Zero all elapsed times.  */
  memset ((void *) timevars, 0, sizeof (timevars));

  /* Initialize the names of timing variables.  */
#define DEFTIMEVAR(identifier__, name__) \
  timevars[identifier__].name = name__;
#include "timevar.def"
#undef DEFTIMEVAR
}

void
timevar_push (timevar_id_t timevar)
{
  if (!timevar_enabled)
    return;

  struct timevar_def *tv = &timevars[timevar];

  /* Mark this timing variable as used.  */
  tv->used = 1;

  /* Can't push a standalone timer.  */
  if (tv->standalone)
    abort ();

  /* What time is it?  */
  struct timevar_time_def const now = get_current_time ();

  /* If the stack isn't empty, attribute the current elapsed time to
     the old topmost element.  */
  if (stack)
    timevar_accumulate (&stack->timevar->elapsed, &start_time, &now);

  /* Reset the start time; from now on, time is attributed to
     TIMEVAR.  */
  start_time = now;

  /* See if we have a previously-allocated stack instance.  If so,
     take it off the list.  If not, malloc a new one.  */
  struct timevar_stack_def *context = NULL;
  if (unused_stack_instances != NULL)
    {
      context = unused_stack_instances;
      unused_stack_instances = unused_stack_instances->next;
    }
  else
    context = (struct timevar_stack_def *)
      xmalloc (sizeof (struct timevar_stack_def));

  /* Fill it in and put it on the stack.  */
  context->timevar = tv;
  context->next = stack;
  stack = context;
}

void
timevar_pop (timevar_id_t timevar)
{
  if (!timevar_enabled)
    return;

  if (&timevars[timevar] != stack->timevar)
    abort ();

  /* What time is it?  */
  struct timevar_time_def const now = get_current_time ();

  /* Attribute the elapsed time to the element we're popping.  */
  struct timevar_stack_def *popped = stack;
  timevar_accumulate (&popped->timevar->elapsed, &start_time, &now);

  /* Reset the start time; from now on, time is attributed to the
     element just exposed on the stack.  */
  start_time = now;

  /* Take the item off the stack.  */
  stack = stack->next;

  /* Don't delete the stack element; instead, add it to the list of
     unused elements for later use.  */
  popped->next = unused_stack_instances;
  unused_stack_instances = popped;
}

void
timevar_start (timevar_id_t timevar)
{
  if (!timevar_enabled)
    return;

  struct timevar_def *tv = &timevars[timevar];

  /* Mark this timing variable as used.  */
  tv->used = 1;

  /* Don't allow the same timing variable to be started more than
     once.  */
  if (tv->standalone)
    abort ();
  tv->standalone = 1;

  set_to_current_time (&tv->start_time);
}

void
timevar_stop (timevar_id_t timevar)
{
  if (!timevar_enabled)
    return;

  struct timevar_def *tv = &timevars[timevar];

  /* TIMEVAR must have been started via timevar_start.  */
  if (!tv->standalone)
    abort ();

  struct timevar_time_def const now = get_current_time ();
  timevar_accumulate (&tv->elapsed, &tv->start_time, &now);
}

void
timevar_get (timevar_id_t timevar,
             struct timevar_time_def *elapsed)
{
  struct timevar_def *tv = &timevars[timevar];
  *elapsed = tv->elapsed;

  /* Is TIMEVAR currently running as a standalone timer?  */
  if (tv->standalone)
    {
      struct timevar_time_def const now = get_current_time ();
      timevar_accumulate (elapsed, &tv->start_time, &now);
    }
  /* Or is TIMEVAR at the top of the timer stack?  */
  else if (stack->timevar == tv)
    {
      struct timevar_time_def const now = get_current_time ();
      timevar_accumulate (elapsed, &start_time, &now);
    }
}

void
timevar_print (FILE *fp)
{
  if (!timevar_enabled)
    return;

  /* Update timing information in case we're calling this from GDB.  */

  if (fp == 0)
    fp = stderr;

  /* What time is it?  */
  struct timevar_time_def const now = get_current_time ();

  /* If the stack isn't empty, attribute the current elapsed time to
     the old topmost element.  */
  if (stack)
    timevar_accumulate (&stack->timevar->elapsed, &start_time, &now);

  /* Reset the start time; from now on, time is attributed to
     TIMEVAR.  */
  start_time = now;

  struct timevar_time_def const* total = &timevars[tv_total].elapsed;

  fprintf (fp, "%-22s\n",
           _("Execution times (seconds)"));
  fprintf (fp, " %-22s   %-13s %-13s %-16s\n",
           "", _("CPU user"), _("CPU system"), _("wall clock"));
  for (unsigned /* timevar_id_t */ id = 0; id < (unsigned) TIMEVAR_LAST; ++id)
    {
      /* Don't print the total execution time here; that goes at the
         end.  */
      if ((timevar_id_t) id == tv_total)
        continue;

      /* Don't print timing variables that were never used.  */
      struct timevar_def *tv = &timevars[(timevar_id_t) id];
      if (!tv->used)
        continue;

      /* Percentages.  */
      const int usr = total->user ? tv->elapsed.user * 100 / total->user : 0;
      const int sys = total->sys ? tv->elapsed.sys * 100 / total->sys : 0;
      const int wall = total->wall ? tv->elapsed.wall * 100 / total->wall : 0;

      /* Ignore insignificant lines.  */
      if (!usr && !sys && !wall)
        continue;

      fprintf (fp, " %-22s", tv->name);
      fprintf (fp, "%8.3f (%2d%%)", tv->elapsed.user * 1e-9, usr);
      fprintf (fp, "%8.3f (%2d%%)", tv->elapsed.sys * 1e-9, sys);
      fprintf (fp, "%11.6f (%2d%%)\n", tv->elapsed.wall * 1e-9, wall);
    }

  /* Print total time.  */
  fprintf (fp, " %-22s", timevars[tv_total].name);
  fprintf (fp, "%8.3f      ", total->user * 1e-9);
  fprintf (fp, "%8.3f      ", total->sys * 1e-9);
  fprintf (fp, "%11.6f\n", total->wall * 1e-9);
}
