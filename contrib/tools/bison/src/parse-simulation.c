/* Parser simulator for unifying counterexample search

   Copyright (C) 2020-2021 Free Software Foundation, Inc.

   This file is part of Bison, the GNU Compiler Compiler.

   This program is free software: you can redistribute it and/or modify
   it under the terms of the GNU General Public License as published by
   the Free Software Foundation, either version 3 of the License, or
   (at your option) any later version.

   This program is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
   GNU General Public License for more details.

   You should have received a copy of the GNU General Public License
   along with this program.  If not, see <https://www.gnu.org/licenses/>.  */

#include <config.h>

#include "parse-simulation.h"

#include <gl_linked_list.h>
#include <gl_xlist.h>
#include <stdlib.h>

#include "lssi.h"
#include "nullable.h"

struct parse_state
{
  // Path of state-items the parser has traversed.
  struct si_chunk
  {
    // Elements newly added in this chunk.
    state_item_list contents;
    // Properties of the linked list this chunk represents.
    const state_item *head_elt;
    const state_item *tail_elt;
    size_t total_size;
  } state_items;
  // List of derivations of the symbols.
  struct deriv_chunk
  {
    derivation_list contents;
    const derivation *head_elt;
    const derivation *tail_elt;
    size_t total_size;
  } derivs;
  struct parse_state *parent;
  int reference_count;
  // Incremented during productions, decremented during reductions.
  int depth;
  // Whether the contents of the chunks should be prepended or
  // appended to the list the chunks represent.
  bool prepend;
  // Causes chunk contents to be freed when the reference count is
  // one. Used when only the chunk metadata will be needed.
  bool free_contents_early;
};


static void
ps_si_prepend (parse_state *ps, const state_item *si)
{
  struct si_chunk *sic = &ps->state_items;
  gl_list_add_first (sic->contents, si);
  sic->head_elt = si;
  ++sic->total_size;
  if (!sic->tail_elt)
    sic->tail_elt = si;
}

static void
ps_si_append (parse_state *ps, const state_item *si)
{
  struct si_chunk *sic = &ps->state_items;
  gl_list_add_last (sic->contents, si);
  sic->tail_elt = si;
  ++sic->total_size;
  if (!sic->head_elt)
    sic->head_elt = si;
}

static void
ps_derivs_prepend (parse_state *ps, derivation *d)
{
  struct deriv_chunk *dc = &ps->derivs;
  derivation_list_prepend (dc->contents, d);
  dc->head_elt = d;
  ++dc->total_size;
  if (!dc->tail_elt)
    dc->tail_elt = d;
}

static void
ps_derivs_append (parse_state *ps, derivation *d)
{
  struct deriv_chunk *dc = &ps->derivs;
  derivation_list_append (dc->contents, d);
  dc->tail_elt = d;
  ++dc->total_size;
  if (!dc->head_elt)
    dc->head_elt = d;
}

static int allocs = 0;
static int frees = 0;

static parse_state *
empty_parse_state (void)
{
  parse_state *res = xcalloc (1, sizeof *res);
  res->state_items.contents
    = gl_list_create_empty (GL_LINKED_LIST, NULL, NULL, NULL, true);
  res->derivs.contents = derivation_list_new ();
  ++allocs;
  return res;
}

parse_state *
new_parse_state (const state_item *si)
{
  parse_state *res = empty_parse_state ();
  ps_si_append (res, si);
  ps_derivs_append (res, derivation_dot ());
  return res;
}

static parse_state *
copy_parse_state (bool prepend, parse_state *parent)
{
  parse_state *res = xmalloc (sizeof *res);
  *res = *parent;
  res->state_items.contents
    = gl_list_create_empty (GL_LINKED_LIST, NULL, NULL, NULL, true);
  res->derivs.contents = derivation_list_new ();
  res->parent = parent;
  res->prepend = prepend;
  res->reference_count = 0;
  res->free_contents_early = false;
  parse_state_retain (parent);
  ++allocs;
  return res;
}

bool
parse_state_derivation_completed (const parse_state *ps)
{
  return ps->derivs.total_size == 1;
}

derivation *
parse_state_derivation (const parse_state *ps)
{
  return (derivation *) ps->derivs.head_elt;
}

const state_item *
parse_state_head (const parse_state *ps)
{
  return ps->state_items.head_elt;
}

const state_item *
parse_state_tail (const parse_state *ps)
{
  return ps->state_items.tail_elt;
}

int
parse_state_length (const parse_state *ps)
{
  return ps->state_items.total_size;
}

int
parse_state_depth (const parse_state *ps)
{
  return ps->depth;
}

void
parse_state_retain (parse_state *ps)
{
  ++ps->reference_count;
}

void
parse_state_free_contents_early (parse_state *ps)
{
  ps->free_contents_early = true;
}

void
free_parse_state (parse_state *original_ps)
{
  bool free_contents = true;
  parse_state *parent_ps = NULL;
  for (parse_state *ps = original_ps; ps && free_contents; ps = parent_ps)
    {
      --ps->reference_count;
      free_contents = (ps->reference_count == 1 && ps->free_contents_early)
        || (ps->reference_count == 0 && !ps->free_contents_early);
      // need to keep the parse state around for visited hash set,
      // but its contents and parent can be freed
      if (free_contents)
        {
          if (ps->state_items.contents)
            gl_list_free (ps->state_items.contents);
          if (ps->derivs.contents)
            derivation_list_free (ps->derivs.contents);
        }
      parent_ps = ps->parent;
      if (ps->reference_count <= 0)
        {
          free (ps);
          ++frees;
        }
    }
}

size_t
parse_state_hasher (const parse_state *ps, size_t max)
{
  const struct si_chunk *sis = &ps->state_items;
  return ((state_item *) sis->head_elt - state_items +
          (state_item *) sis->tail_elt - state_items + sis->total_size) % max;
}

bool
parse_state_comparator (const parse_state *ps1, const parse_state *ps2)
{
  const struct si_chunk *sis1 = &ps1->state_items;
  const struct si_chunk *sis2 = &ps2->state_items;
  return sis1->head_elt == sis2->head_elt
    && sis1->tail_elt == sis2->tail_elt
    && sis1->total_size == sis2->total_size;
}


void
parse_state_completed_steps (const parse_state *ps, int *shifts, int *productions)
{
  // traverse to the root parse_state,
  // which will have a list of all completed productions.
  const parse_state *root_ps = ps;
  while (root_ps->parent)
    root_ps = root_ps->parent;

  state_item_list sis = root_ps->state_items.contents;
  int count = 0;

  state_item *last = NULL;
  state_item *next = NULL;
  for (gl_list_iterator_t it = gl_list_iterator (sis);
       state_item_list_next (&it, &next);
       )
    {
      if (last && last->state == next->state)
        ++count;
      last = next;
    }
  *productions = count;
  *shifts = root_ps->state_items.total_size - count;
}

typedef void (*chunk_append_fn) (gl_list_t, const void *);

// A version of gl_list_add_last which has the chunk_append_fn
// signature.
static void
list_add_last (gl_list_t list, const void *elt)
{
  gl_list_add_last (list, elt);
}

// takes an array of n gl_lists and flattens them into two list
// based off of the index split
static void
list_flatten_and_split (gl_list_t *list, gl_list_t *rets, int split, int n,
                        chunk_append_fn append_fn)
{
  int ret_index = 0;
  int ret_array = 0;
  for (int i = 0; i < n; ++i)
    {
      const void *p = NULL;
      gl_list_iterator_t it = gl_list_iterator (list[i]);
      while (gl_list_iterator_next (&it, &p, NULL))
        if (p)
          {
            gl_list_t l = (gl_list_t) p;
            const void *si = NULL;
            gl_list_iterator_t it2 = gl_list_iterator (l);
            while (gl_list_iterator_next (&it2, &si, NULL))
              {
                if (ret_index++ == split)
                  ++ret_array;
                if (rets[ret_array])
                  append_fn (rets[ret_array], si);
              }
            gl_list_iterator_free (&it2);
          }
      gl_list_iterator_free (&it);
    }
}

static parse_state_list
parse_state_list_new (void)
{
  return gl_list_create_empty (GL_LINKED_LIST, NULL, NULL,
                               (gl_listelement_dispose_fn)free_parse_state,
                               true);
}

static void
parse_state_list_append (parse_state_list pl, parse_state *ps)
{
  parse_state_retain (ps);
  gl_list_add_last (pl, ps);
}

// Emulates a reduction on a parse state by popping some amount of
// derivations and state_items off of the parse_state and returning
// the result in ret. Returns the derivation of what's popped.
static derivation_list
parser_pop (parse_state *ps, int deriv_index,
            int si_index, parse_state *ret)
{
  // prepend sis, append sis, prepend derivs, append derivs
  gl_list_t chunks[4];
  for (int i = 0; i < 4; ++i)
    chunks[i] = gl_list_create_empty (GL_LINKED_LIST, NULL, NULL, NULL, true);
  for (parse_state *pn = ps; pn != NULL; pn = pn->parent)
    if (pn->prepend)
      {
        gl_list_add_last (chunks[0], pn->state_items.contents);
        gl_list_add_last (chunks[2], pn->derivs.contents);
      }
    else
      {
        gl_list_add_first (chunks[1], pn->state_items.contents);
        gl_list_add_first (chunks[3], pn->derivs.contents);
      }
  derivation_list popped_derivs = derivation_list_new ();
  gl_list_t ret_chunks[4] = { ret->state_items.contents, NULL,
    ret->derivs.contents, popped_derivs
  };
  list_flatten_and_split (chunks, ret_chunks, si_index, 2,
                          list_add_last);
  list_flatten_and_split (chunks + 2, ret_chunks + 2, deriv_index, 2,
                          (chunk_append_fn)derivation_list_append);
  size_t s_size = gl_list_size (ret->state_items.contents);
  ret->state_items.total_size = s_size;
  if (s_size > 0)
    {
      ret->state_items.tail_elt = gl_list_get_at (ret->state_items.contents,
                                                  s_size - 1);
      ret->state_items.head_elt =
        gl_list_get_at (ret->state_items.contents, 0);
    }
  else
    {
      ret->state_items.tail_elt = NULL;
      ret->state_items.head_elt = NULL;
    }
  size_t d_size = gl_list_size (ret->derivs.contents);
  ret->derivs.total_size = d_size;
  if (d_size > 0)
    {
      ret->derivs.tail_elt = gl_list_get_at (ret->derivs.contents,
                                             d_size - 1);
      ret->derivs.head_elt = gl_list_get_at (ret->derivs.contents, 0);
    }
  else
    {
      ret->derivs.tail_elt = NULL;
      ret->derivs.head_elt = NULL;
    }
  for (int i = 0; i < 4; ++i)
    gl_list_free (chunks[i]);
  return popped_derivs;
}

void
parse_state_lists (parse_state *ps, state_item_list *sitems,
                   derivation_list *derivs)
{
  parse_state *temp = empty_parse_state ();
  size_t si_size = ps->state_items.total_size;
  size_t deriv_size = ps->derivs.total_size;
  derivation_list dl = parser_pop (ps, si_size, deriv_size, temp);
  *sitems = temp->state_items.contents;
  *derivs = temp->derivs.contents;
  // prevent the return lists from being freed
  temp->state_items.contents = NULL;
  temp->derivs.contents = NULL;
  free_parse_state (temp);
  derivation_list_free (dl);
}

/**
 * Compute the parse states that result from taking a transition on
 * nullable symbols whenever possible from the given state_item.
 */
static void
nullable_closure (parse_state *ps, state_item *si, parse_state_list state_list)
{
  parse_state *current_ps = ps;
  state_item_number prev_sin = si - state_items;
  for (state_item_number sin = si->trans; sin != -1;
       prev_sin = sin, sin = state_items[sin].trans)
    {
      state_item *psi = &state_items[prev_sin];
      symbol_number sp = item_number_as_symbol_number (*psi->item);
      if (ISTOKEN (sp) || !nullable[sp - ntokens])
        break;

      state_item *nsi = &state_items[sin];
      current_ps = copy_parse_state (false, current_ps);
      ps_si_append (current_ps, nsi);
      ps_derivs_append (current_ps, derivation_new (sp, derivation_list_new ()));
      parse_state_list_append (state_list, current_ps);
    }
}

parse_state_list
simulate_transition (parse_state *ps)
{
  const state_item *si = ps->state_items.tail_elt;
  symbol_number sym = item_number_as_symbol_number (*si->item);
  // Transition on the same next symbol, taking nullable
  // symbols into account.
  parse_state_list result = parse_state_list_new ();
  state_item_number si_next = si->trans;
  // Check for disabled transition, shouldn't happen as any
  // state_items that lead to these should be disabled.
  if (si_next < 0)
    return result;
  parse_state *next_ps = copy_parse_state (false, ps);
  ps_si_append (next_ps, &state_items[si_next]);
  ps_derivs_append (next_ps, derivation_new_leaf (sym));
  parse_state_list_append (result, next_ps);

  nullable_closure (next_ps, &state_items[si_next], result);
  return result;
}

/**
 * Determine if the given symbols are equal or their first sets
 * intersect.
 */
static bool
compatible (symbol_number sym1, symbol_number sym2)
{
  if (sym1 == sym2)
    return true;
  if (ISTOKEN (sym1) && ISVAR (sym2))
    return bitset_test (FIRSTS (sym2), sym1);
  else if (ISVAR (sym1) && ISTOKEN (sym2))
    return bitset_test (FIRSTS (sym1), sym2);
  else if (ISVAR (sym1) && ISVAR (sym2))
    return !bitset_disjoint_p (FIRSTS (sym1), FIRSTS (sym2));
  else
    return false;
}

parse_state_list
simulate_production (parse_state *ps, symbol_number compat_sym)
{
  parse_state_list result = parse_state_list_new ();
  const state_item *si = parse_state_tail (ps);
  if (si->prods)
    {
      bitset_iterator biter;
      state_item_number sin;
      BITSET_FOR_EACH (biter, si->prods, sin, 0)
        {
          // Take production step only if lhs is not nullable and
          // if first rhs symbol is compatible with compat_sym
          state_item *next = &state_items[sin];
          item_number *itm1 = next->item;
          if (!compatible (*itm1, compat_sym) || !production_allowed (si, next))
            continue;
          parse_state *next_ps = copy_parse_state (false, ps);
          ps_si_append (next_ps, next);
          parse_state_list_append (result, next_ps);
          if (next_ps->depth >= 0)
            ++next_ps->depth;
          nullable_closure (next_ps, next, result);
        }
    }
  return result;
}

// simulates a reduction on the given parse state, conflict_item is the
// item associated with ps's conflict. symbol_set is a lookahead set this
// reduction must be compatible with
parse_state_list
simulate_reduction (parse_state *ps, int rule_len, bitset symbol_set)
{
  parse_state_list result = parse_state_list_new ();

  int s_size = ps->state_items.total_size;
  int d_size = ps->derivs.total_size;
  if (ps->depth >= 0)
    d_size--;                   // account for dot
  parse_state *new_root = empty_parse_state ();
  derivation_list popped_derivs =
    parser_pop (ps, d_size - rule_len,
                s_size - rule_len - 1, new_root);

  // update derivation
  state_item *si = (state_item *) ps->state_items.tail_elt;
  const rule *r = item_rule (si->item);
  symbol_number lhs = r->lhs->number;
  derivation *deriv = derivation_new (lhs, popped_derivs);
  --new_root->depth;
  ps_derivs_append (new_root, deriv);

  if (s_size != rule_len + 1)
    {
      state_item *tail = (state_item *) new_root->state_items.tail_elt;
      ps_si_append (new_root, &state_items[tail->trans]);
      parse_state_list_append (result, new_root);
    }
  else
    {
      // The head state_item is a production item, so we need to prepend
      // with possible source state-items.
      const state_item *head = ps->state_items.head_elt;
      state_item_list prev = lssi_reverse_production (head, symbol_set);
      // TODO: better understand what causes this case.
      if (gl_list_size (prev) == 0)
        {
          // new_root needs to have an RC of 1 to be freed correctly here.
          parse_state_retain (new_root);
          free_parse_state (new_root);
        }
      else
        {
          state_item *psis = NULL;
          for (gl_list_iterator_t it = gl_list_iterator (prev);
               state_item_list_next (&it, &psis);
               )
            {
              // Prepend the result from the reverse production.
              parse_state *copy = copy_parse_state (true, new_root);
              ps_si_prepend (copy, psis);

              // Append the left hand side to the end of the parser state
              copy = copy_parse_state (false, copy);
              struct si_chunk *sis = &copy->state_items;
              const state_item *tail = sis->tail_elt;
              ps_si_append (copy, &state_items[tail->trans]);
              parse_state_list_append (result, copy);
              nullable_closure (copy, (state_item *) sis->tail_elt, result);
            }
        }
      gl_list_free (prev);
    }
  return result;
}

parse_state_list
parser_prepend (parse_state *ps)
{
  parse_state_list res = parse_state_list_new ();
  const state_item *head = ps->state_items.head_elt;
  symbol_number prepend_sym =
    item_number_as_symbol_number (*(head->item - 1));
  bitset_iterator biter;
  state_item_number sin;
  BITSET_FOR_EACH (biter, head->revs, sin, 0)
  {
    parse_state *copy = copy_parse_state (true, ps);
    ps_si_prepend (copy, &state_items[sin]);
    if (SI_TRANSITION (head))
      ps_derivs_prepend (copy, derivation_new_leaf (prepend_sym));
    parse_state_list_append (res, copy);
  }
  return res;
}

void
print_parse_state (parse_state *ps)
{
  FILE *out = stderr;
  fprintf (out, "(size %zu depth %d rc %d)\n",
          ps->state_items.total_size, ps->depth, ps->reference_count);
  state_item_print (ps->state_items.head_elt, out, "");
  state_item_print (ps->state_items.tail_elt, out, "");
  if (ps->derivs.total_size > 0)
    derivation_print (ps->derivs.head_elt, out, "");
  putc ('\n', out);
}
