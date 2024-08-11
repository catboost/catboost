/* Analyze differences between two vectors.

   Copyright (C) 1988-1989, 1992-1995, 2001-2004, 2006-2020 Free Software
   Foundation, Inc.

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


/* The basic idea is to consider two vectors as similar if, when
   transforming the first vector into the second vector through a
   sequence of edits (inserts and deletes of one element each),
   this sequence is short - or equivalently, if the ordered list
   of elements that are untouched by these edits is long.  For a
   good introduction to the subject, read about the "Levenshtein
   distance" in Wikipedia.

   The basic algorithm is described in:
   "An O(ND) Difference Algorithm and its Variations", Eugene W. Myers,
   Algorithmica Vol. 1, 1986, pp. 251-266,
   <https://doi.org/10.1007/BF01840446>.
   See especially section 4.2, which describes the variation used below.

   The basic algorithm was independently discovered as described in:
   "Algorithms for Approximate String Matching", Esko Ukkonen,
   Information and Control Vol. 64, 1985, pp. 100-118,
   <https://doi.org/10.1016/S0019-9958(85)80046-2>.

   Unless the 'find_minimal' flag is set, this code uses the TOO_EXPENSIVE
   heuristic, by Paul Eggert, to limit the cost to O(N**1.5 log N)
   at the price of producing suboptimal output for large inputs with
   many differences.  */

/* Before including this file, you need to define:
     ELEMENT                 The element type of the vectors being compared.
     EQUAL                   A two-argument macro that tests two elements for
                             equality.
     OFFSET                  A signed integer type sufficient to hold the
                             difference between two indices.  Usually
                             something like ptrdiff_t.
     EXTRA_CONTEXT_FIELDS    Declarations of fields for 'struct context'.
     NOTE_DELETE(ctxt, xoff) Record the removal of the object xvec[xoff].
     NOTE_INSERT(ctxt, yoff) Record the insertion of the object yvec[yoff].
     NOTE_ORDERED            (Optional) A boolean expression saying that
                             NOTE_DELETE and NOTE_INSERT calls must be
                             issued in offset order.
     EARLY_ABORT(ctxt)       (Optional) A boolean expression that triggers an
                             early abort of the computation.
     USE_HEURISTIC           (Optional) Define if you want to support the
                             heuristic for large vectors.

   It is also possible to use this file with abstract arrays.  In this case,
   xvec and yvec are not represented in memory.  They only exist conceptually.
   In this case, the list of defines above is amended as follows:
     ELEMENT                 Undefined.
     EQUAL                   Undefined.
     XVECREF_YVECREF_EQUAL(ctxt, xoff, yoff)
                             A three-argument macro: References xvec[xoff] and
                             yvec[yoff] and tests these elements for equality.

   Before including this file, you also need to include:
     #include <limits.h>
     #include <stdbool.h>
     #include "minmax.h"
 */

/* Maximum value of type OFFSET.  */
#define OFFSET_MAX \
  ((((OFFSET)1 << (sizeof (OFFSET) * CHAR_BIT - 2)) - 1) * 2 + 1)

/* Default to no early abort.  */
#ifndef EARLY_ABORT
# define EARLY_ABORT(ctxt) false
#endif

#ifndef NOTE_ORDERED
# define NOTE_ORDERED false
#endif

/* Use this to suppress gcc's "...may be used before initialized" warnings.
   Beware: The Code argument must not contain commas.  */
#ifndef IF_LINT
# if defined GCC_LINT || defined lint
#  define IF_LINT(Code) Code
# else
#  define IF_LINT(Code) /* empty */
# endif
#endif

/*
 * Context of comparison operation.
 */
struct context
{
  #ifdef ELEMENT
  /* Vectors being compared.  */
  ELEMENT const *xvec;
  ELEMENT const *yvec;
  #endif

  /* Extra fields.  */
  EXTRA_CONTEXT_FIELDS

  /* Vector, indexed by diagonal, containing 1 + the X coordinate of the point
     furthest along the given diagonal in the forward search of the edit
     matrix.  */
  OFFSET *fdiag;

  /* Vector, indexed by diagonal, containing the X coordinate of the point
     furthest along the given diagonal in the backward search of the edit
     matrix.  */
  OFFSET *bdiag;

  #ifdef USE_HEURISTIC
  /* This corresponds to the diff --speed-large-files flag.  With this
     heuristic, for vectors with a constant small density of changes,
     the algorithm is linear in the vector size.  */
  bool heuristic;
  #endif

  /* Edit scripts longer than this are too expensive to compute.  */
  OFFSET too_expensive;

  /* Snakes bigger than this are considered "big".  */
  #define SNAKE_LIMIT 20
};

struct partition
{
  /* Midpoints of this partition.  */
  OFFSET xmid;
  OFFSET ymid;

  /* True if low half will be analyzed minimally.  */
  bool lo_minimal;

  /* Likewise for high half.  */
  bool hi_minimal;
};


/* Find the midpoint of the shortest edit script for a specified portion
   of the two vectors.

   Scan from the beginnings of the vectors, and simultaneously from the ends,
   doing a breadth-first search through the space of edit-sequence.
   When the two searches meet, we have found the midpoint of the shortest
   edit sequence.

   If FIND_MINIMAL is true, find the minimal edit script regardless of
   expense.  Otherwise, if the search is too expensive, use heuristics to
   stop the search and report a suboptimal answer.

   Set PART->(xmid,ymid) to the midpoint (XMID,YMID).  The diagonal number
   XMID - YMID equals the number of inserted elements minus the number
   of deleted elements (counting only elements before the midpoint).

   Set PART->lo_minimal to true iff the minimal edit script for the
   left half of the partition is known; similarly for PART->hi_minimal.

   This function assumes that the first elements of the specified portions
   of the two vectors do not match, and likewise that the last elements do not
   match.  The caller must trim matching elements from the beginning and end
   of the portions it is going to specify.

   If we return the "wrong" partitions, the worst this can do is cause
   suboptimal diff output.  It cannot cause incorrect diff output.  */

static void
diag (OFFSET xoff, OFFSET xlim, OFFSET yoff, OFFSET ylim, bool find_minimal,
      struct partition *part, struct context *ctxt)
{
  OFFSET *const fd = ctxt->fdiag;       /* Give the compiler a chance. */
  OFFSET *const bd = ctxt->bdiag;       /* Additional help for the compiler. */
#ifdef ELEMENT
  ELEMENT const *const xv = ctxt->xvec; /* Still more help for the compiler. */
  ELEMENT const *const yv = ctxt->yvec; /* And more and more . . . */
  #define XREF_YREF_EQUAL(x,y)  EQUAL (xv[x], yv[y])
#else
  #define XREF_YREF_EQUAL(x,y)  XVECREF_YVECREF_EQUAL (ctxt, x, y)
#endif
  const OFFSET dmin = xoff - ylim;      /* Minimum valid diagonal. */
  const OFFSET dmax = xlim - yoff;      /* Maximum valid diagonal. */
  const OFFSET fmid = xoff - yoff;      /* Center diagonal of top-down search. */
  const OFFSET bmid = xlim - ylim;      /* Center diagonal of bottom-up search. */
  OFFSET fmin = fmid;
  OFFSET fmax = fmid;           /* Limits of top-down search. */
  OFFSET bmin = bmid;
  OFFSET bmax = bmid;           /* Limits of bottom-up search. */
  OFFSET c;                     /* Cost. */
  bool odd = (fmid - bmid) & 1; /* True if southeast corner is on an odd
                                   diagonal with respect to the northwest. */

  fd[fmid] = xoff;
  bd[bmid] = xlim;

  for (c = 1;; ++c)
    {
      OFFSET d;                 /* Active diagonal. */
      bool big_snake = false;

      /* Extend the top-down search by an edit step in each diagonal. */
      if (fmin > dmin)
        fd[--fmin - 1] = -1;
      else
        ++fmin;
      if (fmax < dmax)
        fd[++fmax + 1] = -1;
      else
        --fmax;
      for (d = fmax; d >= fmin; d -= 2)
        {
          OFFSET x;
          OFFSET y;
          OFFSET tlo = fd[d - 1];
          OFFSET thi = fd[d + 1];
          OFFSET x0 = tlo < thi ? thi : tlo + 1;

          for (x = x0, y = x0 - d;
               x < xlim && y < ylim && XREF_YREF_EQUAL (x, y);
               x++, y++)
            continue;
          if (x - x0 > SNAKE_LIMIT)
            big_snake = true;
          fd[d] = x;
          if (odd && bmin <= d && d <= bmax && bd[d] <= x)
            {
              part->xmid = x;
              part->ymid = y;
              part->lo_minimal = part->hi_minimal = true;
              return;
            }
        }

      /* Similarly extend the bottom-up search.  */
      if (bmin > dmin)
        bd[--bmin - 1] = OFFSET_MAX;
      else
        ++bmin;
      if (bmax < dmax)
        bd[++bmax + 1] = OFFSET_MAX;
      else
        --bmax;
      for (d = bmax; d >= bmin; d -= 2)
        {
          OFFSET x;
          OFFSET y;
          OFFSET tlo = bd[d - 1];
          OFFSET thi = bd[d + 1];
          OFFSET x0 = tlo < thi ? tlo : thi - 1;

          for (x = x0, y = x0 - d;
               xoff < x && yoff < y && XREF_YREF_EQUAL (x - 1, y - 1);
               x--, y--)
            continue;
          if (x0 - x > SNAKE_LIMIT)
            big_snake = true;
          bd[d] = x;
          if (!odd && fmin <= d && d <= fmax && x <= fd[d])
            {
              part->xmid = x;
              part->ymid = y;
              part->lo_minimal = part->hi_minimal = true;
              return;
            }
        }

      if (find_minimal)
        continue;

#ifdef USE_HEURISTIC
      bool heuristic = ctxt->heuristic;
#else
      bool heuristic = false;
#endif

      /* Heuristic: check occasionally for a diagonal that has made lots
         of progress compared with the edit distance.  If we have any
         such, find the one that has made the most progress and return it
         as if it had succeeded.

         With this heuristic, for vectors with a constant small density
         of changes, the algorithm is linear in the vector size.  */

      if (200 < c && big_snake && heuristic)
        {
          {
            OFFSET best = 0;

            for (d = fmax; d >= fmin; d -= 2)
              {
                OFFSET dd = d - fmid;
                OFFSET x = fd[d];
                OFFSET y = x - d;
                OFFSET v = (x - xoff) * 2 - dd;

                if (v > 12 * (c + (dd < 0 ? -dd : dd)))
                  {
                    if (v > best
                        && xoff + SNAKE_LIMIT <= x && x < xlim
                        && yoff + SNAKE_LIMIT <= y && y < ylim)
                      {
                        /* We have a good enough best diagonal; now insist
                           that it end with a significant snake.  */
                        int k;

                        for (k = 1; XREF_YREF_EQUAL (x - k, y - k); k++)
                          if (k == SNAKE_LIMIT)
                            {
                              best = v;
                              part->xmid = x;
                              part->ymid = y;
                              break;
                            }
                      }
                  }
              }
            if (best > 0)
              {
                part->lo_minimal = true;
                part->hi_minimal = false;
                return;
              }
          }

          {
            OFFSET best = 0;

            for (d = bmax; d >= bmin; d -= 2)
              {
                OFFSET dd = d - bmid;
                OFFSET x = bd[d];
                OFFSET y = x - d;
                OFFSET v = (xlim - x) * 2 + dd;

                if (v > 12 * (c + (dd < 0 ? -dd : dd)))
                  {
                    if (v > best
                        && xoff < x && x <= xlim - SNAKE_LIMIT
                        && yoff < y && y <= ylim - SNAKE_LIMIT)
                      {
                        /* We have a good enough best diagonal; now insist
                           that it end with a significant snake.  */
                        int k;

                        for (k = 0; XREF_YREF_EQUAL (x + k, y + k); k++)
                          if (k == SNAKE_LIMIT - 1)
                            {
                              best = v;
                              part->xmid = x;
                              part->ymid = y;
                              break;
                            }
                      }
                  }
              }
            if (best > 0)
              {
                part->lo_minimal = false;
                part->hi_minimal = true;
                return;
              }
          }
        }

      /* Heuristic: if we've gone well beyond the call of duty, give up
         and report halfway between our best results so far.  */
      if (c >= ctxt->too_expensive)
        {
          OFFSET fxybest;
          OFFSET fxbest IF_LINT (= 0);
          OFFSET bxybest;
          OFFSET bxbest IF_LINT (= 0);

          /* Find forward diagonal that maximizes X + Y.  */
          fxybest = -1;
          for (d = fmax; d >= fmin; d -= 2)
            {
              OFFSET x = MIN (fd[d], xlim);
              OFFSET y = x - d;
              if (ylim < y)
                {
                  x = ylim + d;
                  y = ylim;
                }
              if (fxybest < x + y)
                {
                  fxybest = x + y;
                  fxbest = x;
                }
            }

          /* Find backward diagonal that minimizes X + Y.  */
          bxybest = OFFSET_MAX;
          for (d = bmax; d >= bmin; d -= 2)
            {
              OFFSET x = MAX (xoff, bd[d]);
              OFFSET y = x - d;
              if (y < yoff)
                {
                  x = yoff + d;
                  y = yoff;
                }
              if (x + y < bxybest)
                {
                  bxybest = x + y;
                  bxbest = x;
                }
            }

          /* Use the better of the two diagonals.  */
          if ((xlim + ylim) - bxybest < fxybest - (xoff + yoff))
            {
              part->xmid = fxbest;
              part->ymid = fxybest - fxbest;
              part->lo_minimal = true;
              part->hi_minimal = false;
            }
          else
            {
              part->xmid = bxbest;
              part->ymid = bxybest - bxbest;
              part->lo_minimal = false;
              part->hi_minimal = true;
            }
          return;
        }
    }
  #undef XREF_YREF_EQUAL
}


/* Compare in detail contiguous subsequences of the two vectors
   which are known, as a whole, to match each other.

   The subsequence of vector 0 is [XOFF, XLIM) and likewise for vector 1.

   Note that XLIM, YLIM are exclusive bounds.  All indices into the vectors
   are origin-0.

   If FIND_MINIMAL, find a minimal difference no matter how
   expensive it is.

   The results are recorded by invoking NOTE_DELETE and NOTE_INSERT.

   Return false if terminated normally, or true if terminated through early
   abort.  */

static bool
compareseq (OFFSET xoff, OFFSET xlim, OFFSET yoff, OFFSET ylim,
            bool find_minimal, struct context *ctxt)
{
#ifdef ELEMENT
  ELEMENT const *xv = ctxt->xvec; /* Help the compiler.  */
  ELEMENT const *yv = ctxt->yvec;
  #define XREF_YREF_EQUAL(x,y)  EQUAL (xv[x], yv[y])
#else
  #define XREF_YREF_EQUAL(x,y)  XVECREF_YVECREF_EQUAL (ctxt, x, y)
#endif

  while (true)
    {
      /* Slide down the bottom initial diagonal.  */
      while (xoff < xlim && yoff < ylim && XREF_YREF_EQUAL (xoff, yoff))
        {
          xoff++;
          yoff++;
        }

      /* Slide up the top initial diagonal. */
      while (xoff < xlim && yoff < ylim && XREF_YREF_EQUAL (xlim - 1, ylim - 1))
        {
          xlim--;
          ylim--;
        }

      /* Handle simple cases. */
      if (xoff == xlim)
        {
          while (yoff < ylim)
            {
              NOTE_INSERT (ctxt, yoff);
              if (EARLY_ABORT (ctxt))
                return true;
              yoff++;
            }
          break;
        }
      if (yoff == ylim)
        {
          while (xoff < xlim)
            {
              NOTE_DELETE (ctxt, xoff);
              if (EARLY_ABORT (ctxt))
                return true;
              xoff++;
            }
          break;
        }

      struct partition part;

      /* Find a point of correspondence in the middle of the vectors.  */
      diag (xoff, xlim, yoff, ylim, find_minimal, &part, ctxt);

      /* Use the partitions to split this problem into subproblems.  */
      OFFSET xoff1, xlim1, yoff1, ylim1, xoff2, xlim2, yoff2, ylim2;
      bool find_minimal1, find_minimal2;
      if (!NOTE_ORDERED
          && ((xlim + ylim) - (part.xmid + part.ymid)
              < (part.xmid + part.ymid) - (xoff + yoff)))
        {
          /* The second problem is smaller and the caller doesn't
             care about order, so do the second problem first to
             lessen recursion.  */
          xoff1 = part.xmid; xlim1 = xlim;
          yoff1 = part.ymid; ylim1 = ylim;
          find_minimal1 = part.hi_minimal;

          xoff2 = xoff; xlim2 = part.xmid;
          yoff2 = yoff; ylim2 = part.ymid;
          find_minimal2 = part.lo_minimal;
        }
      else
        {
          xoff1 = xoff; xlim1 = part.xmid;
          yoff1 = yoff; ylim1 = part.ymid;
          find_minimal1 = part.lo_minimal;

          xoff2 = part.xmid; xlim2 = xlim;
          yoff2 = part.ymid; ylim2 = ylim;
          find_minimal2 = part.hi_minimal;
        }

      /* Recurse to do one subproblem.  */
      bool early = compareseq (xoff1, xlim1, yoff1, ylim1, find_minimal1, ctxt);
      if (early)
        return early;

      /* Iterate to do the other subproblem.  */
      xoff = xoff2; xlim = xlim2;
      yoff = yoff2; ylim = ylim2;
      find_minimal = find_minimal2;
    }

  return false;
  #undef XREF_YREF_EQUAL
}

#undef ELEMENT
#undef EQUAL
#undef OFFSET
#undef EXTRA_CONTEXT_FIELDS
#undef NOTE_DELETE
#undef NOTE_INSERT
#undef EARLY_ABORT
#undef USE_HEURISTIC
#undef XVECREF_YVECREF_EQUAL
#undef OFFSET_MAX
