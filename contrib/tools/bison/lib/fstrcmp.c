/* Functions to make fuzzy comparisons between strings
   Copyright (C) 1988-1989, 1992-1993, 1995, 2001-2003, 2006, 2008-2020 Free
   Software Foundation, Inc.

   This program is free software: you can redistribute it and/or modify
   it under the terms of the GNU General Public License as published by
   the Free Software Foundation; either version 3 of the License, or
   (at your option) any later version.

   This program is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
   GNU General Public License for more details.

   You should have received a copy of the GNU General Public License
   along with this program.  If not, see <https://www.gnu.org/licenses/>.  */


#include <config.h>

/* Specification.  */
#include "fstrcmp.h"

#include <string.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <limits.h>

#include "glthread/lock.h"
#include "glthread/tls.h"
#include "minmax.h"
#include "xalloc.h"


#define ELEMENT char
#define EQUAL(x,y) ((x) == (y))
#define OFFSET ptrdiff_t
#define EXTRA_CONTEXT_FIELDS \
  /* The number of edits beyond which the computation can be aborted. */ \
  ptrdiff_t edit_count_limit; \
  /* The number of edits (= number of elements inserted, plus the number of \
     elements deleted), temporarily minus edit_count_limit. */ \
  ptrdiff_t edit_count;
#define NOTE_DELETE(ctxt, xoff) ctxt->edit_count++
#define NOTE_INSERT(ctxt, yoff) ctxt->edit_count++
#define NOTE_ORDERED false
#define EARLY_ABORT(ctxt) ctxt->edit_count > 0
/* We don't need USE_HEURISTIC, since it is unlikely in typical uses of
   fstrcmp().  */
#include "diffseq.h"


/* Because fstrcmp is typically called multiple times, attempt to minimize
   the number of memory allocations performed.  Thus, let a call reuse the
   memory already allocated by the previous call, if it is sufficient.
   To make it multithread-safe, without need for a lock that protects the
   already allocated memory, store the allocated memory per thread.  Free
   it only when the thread exits.  */

static gl_tls_key_t buffer_key; /* TLS key for a 'ptrdiff_t *' */
static gl_tls_key_t bufmax_key; /* TLS key for a 'uintptr_t' */

static void
keys_init (void)
{
  gl_tls_key_init (buffer_key, free);
  gl_tls_key_init (bufmax_key, NULL);
  /* The per-thread initial values are NULL and 0, respectively.  */
}

/* Ensure that keys_init is called once only.  */
gl_once_define(static, keys_init_once)

void
fstrcmp_free_resources (void)
{
  ptrdiff_t *buffer;

  gl_once (keys_init_once, keys_init);
  buffer = gl_tls_get (buffer_key);
  if (buffer != NULL)
    {
      gl_tls_set (buffer_key, NULL);
      gl_tls_set (bufmax_key, (void *) (uintptr_t) 0);
      free (buffer);
    }
}


/* In the code below, branch probabilities were measured by Ralf Wildenhues,
   by running "msgmerge LL.po coreutils.pot" with msgmerge 0.18 for many
   values of LL.  The probability indicates that the condition evaluates
   to true; whether that leads to a branch or a non-branch in the code,
   depends on the compiler's reordering of basic blocks.  */


double
fstrcmp_bounded (const char *string1, const char *string2, double lower_bound)
{
  struct context ctxt;
  size_t xvec_length = strlen (string1);
  size_t yvec_length = strlen (string2);
  size_t length_sum = xvec_length + yvec_length;
  ptrdiff_t i;

  ptrdiff_t fdiag_len;
  ptrdiff_t *buffer;
  uintptr_t bufmax;

  /* short-circuit obvious comparisons */
  if (xvec_length == 0 || yvec_length == 0) /* Prob: 1% */
    return length_sum == 0;

  if (! (xvec_length <= length_sum
         && length_sum <= MIN (UINTPTR_MAX, PTRDIFF_MAX) - 3))
    xalloc_die ();

  if (lower_bound > 0)
    {
      /* Compute a quick upper bound.
         Each edit is an insertion or deletion of an element, hence modifies
         the length of the sequence by at most 1.
         Therefore, when starting from a sequence X and ending at a sequence Y,
         with N edits,  | yvec_length - xvec_length | <= N.  (Proof by
         induction over N.)
         So, at the end, we will have
           edit_count >= | xvec_length - yvec_length |.
         and hence
           result
             = (xvec_length + yvec_length - edit_count)
               / (xvec_length + yvec_length)
             <= (xvec_length + yvec_length - | yvec_length - xvec_length |)
                / (xvec_length + yvec_length)
             = 2 * min (xvec_length, yvec_length) / (xvec_length + yvec_length).
       */
      ptrdiff_t length_min = MIN (xvec_length, yvec_length);
      volatile double upper_bound = 2.0 * length_min / length_sum;

      if (upper_bound < lower_bound) /* Prob: 74% */
        /* Return an arbitrary value < LOWER_BOUND.  */
        return 0.0;

#if CHAR_BIT <= 8
      /* When X and Y are both small, avoid the overhead of setting up an
         array of size 256.  */
      if (length_sum >= 20) /* Prob: 99% */
        {
          /* Compute a less quick upper bound.
             Each edit is an insertion or deletion of a character, hence
             modifies the occurrence count of a character by 1 and leaves the
             other occurrence counts unchanged.
             Therefore, when starting from a sequence X and ending at a
             sequence Y, and denoting the occurrence count of C in X with
             OCC (X, C), with N edits,
               sum_C | OCC (X, C) - OCC (Y, C) | <= N.
             (Proof by induction over N.)
             So, at the end, we will have
               edit_count >= sum_C | OCC (X, C) - OCC (Y, C) |,
             and hence
               result
                 = (xvec_length + yvec_length - edit_count)
                   / (xvec_length + yvec_length)
                 <= (xvec_length + yvec_length - sum_C | OCC(X,C) - OCC(Y,C) |)
                    / (xvec_length + yvec_length).
           */
          ptrdiff_t occ_diff[UCHAR_MAX + 1]; /* array C -> OCC(X,C) - OCC(Y,C) */
          ptrdiff_t sum;
          double dsum;

          /* Determine the occurrence counts in X.  */
          memset (occ_diff, 0, sizeof (occ_diff));
          for (i = xvec_length - 1; i >= 0; i--)
            occ_diff[(unsigned char) string1[i]]++;
          /* Subtract the occurrence counts in Y.  */
          for (i = yvec_length - 1; i >= 0; i--)
            occ_diff[(unsigned char) string2[i]]--;
          /* Sum up the absolute values.  */
          sum = 0;
          for (i = 0; i <= UCHAR_MAX; i++)
            {
              ptrdiff_t d = occ_diff[i];
              sum += (d >= 0 ? d : -d);
            }

          dsum = sum;
          upper_bound = 1.0 - dsum / length_sum;

          if (upper_bound < lower_bound) /* Prob: 66% */
            /* Return an arbitrary value < LOWER_BOUND.  */
            return 0.0;
        }
#endif
    }

  /* set the info for each string.  */
  ctxt.xvec = string1;
  ctxt.yvec = string2;

  /* Set TOO_EXPENSIVE to be approximate square root of input size,
     bounded below by 4096.  */
  ctxt.too_expensive = 1;
  for (i = xvec_length + yvec_length; i != 0; i >>= 2)
    ctxt.too_expensive <<= 1;
  if (ctxt.too_expensive < 4096)
    ctxt.too_expensive = 4096;

  /* Allocate memory for fdiag and bdiag from a thread-local pool.  */
  fdiag_len = length_sum + 3;
  gl_once (keys_init_once, keys_init);
  buffer = gl_tls_get (buffer_key);
  bufmax = (uintptr_t) gl_tls_get (bufmax_key);
  if (fdiag_len > bufmax)
    {
      /* Need more memory.  */
      bufmax = 2 * bufmax;
      if (fdiag_len > bufmax)
        bufmax = fdiag_len;
      /* Calling xrealloc would be a waste: buffer's contents does not need
         to be preserved.  */
      free (buffer);
      buffer = xnmalloc (bufmax, 2 * sizeof *buffer);
      gl_tls_set (buffer_key, buffer);
      gl_tls_set (bufmax_key, (void *) (uintptr_t) bufmax);
    }
  ctxt.fdiag = buffer + yvec_length + 1;
  ctxt.bdiag = ctxt.fdiag + fdiag_len;

  /* The edit_count is only ever increased.  The computation can be aborted
     when
       (xvec_length + yvec_length - edit_count) / (xvec_length + yvec_length)
       < lower_bound,
     or equivalently
       edit_count > (xvec_length + yvec_length) * (1 - lower_bound)
     or equivalently
       edit_count > floor((xvec_length + yvec_length) * (1 - lower_bound)).
     We need to add an epsilon inside the floor(...) argument, to neutralize
     rounding errors.  */
  ctxt.edit_count_limit =
    (lower_bound < 1.0
     ? (ptrdiff_t) (length_sum * (1.0 - lower_bound + 0.000001))
     : 0);

  /* Now do the main comparison algorithm */
  ctxt.edit_count = - ctxt.edit_count_limit;
  if (compareseq (0, xvec_length, 0, yvec_length, 0, &ctxt)) /* Prob: 98% */
    /* The edit_count passed the limit.  Hence the result would be
       < lower_bound.  We can return any value < lower_bound instead.  */
    return 0.0;
  ctxt.edit_count += ctxt.edit_count_limit;

  /* The result is
        ((number of chars in common) / (average length of the strings)).
     The numerator is
        = xvec_length - (number of calls to NOTE_DELETE)
        = yvec_length - (number of calls to NOTE_INSERT)
        = 1/2 * (xvec_length + yvec_length - (number of edits)).
     This is admittedly biased towards finding that the strings are
     similar, however it does produce meaningful results.  */
  return ((double) (xvec_length + yvec_length - ctxt.edit_count)
          / (xvec_length + yvec_length));
}
