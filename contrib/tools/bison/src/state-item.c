/* Counterexample Generation Search Nodes

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

#include "state-item.h"

#include <assert.h>
#include <gl_linked_list.h>
#include <gl_xlist.h>
#include <stdlib.h>
#include <time.h>

#include "closure.h"
#include "getargs.h"
#include "nullable.h"

size_t nstate_items;
state_item_number *state_item_map;
state_item *state_items;

// Hash functions for index -> bitset hash maps.
typedef struct
{
  int key;
  bitset l;
} hash_pair;

static size_t
hash_pair_hasher (const hash_pair *sl, size_t max)
{
  return sl->key % max;
}

static bool
hash_pair_comparator (const hash_pair *l, const hash_pair *r)
{
  return l->key == r->key;
}

static void
hash_pair_free (hash_pair *hp)
{
  bitset_free (hp->l);
  free (hp);
}

static Hash_table *
hash_pair_table_create (int size)
{
  return hash_xinitialize (size,
                           NULL,
                           (Hash_hasher) hash_pair_hasher,
                           (Hash_comparator) hash_pair_comparator,
                           (Hash_data_freer) hash_pair_free);
}

static bitset
hash_pair_lookup (Hash_table *tab, int key)
{
  hash_pair probe;
  probe.key = key;
  hash_pair *hp = hash_lookup (tab, &probe);
  return hp ? hp->l : NULL;
}

static void
hash_pair_insert (Hash_table *tab, int key, bitset val)
{
  hash_pair *hp = xmalloc (sizeof *hp);
  hp->key = key;
  hp->l = val;
  hash_pair *res = hash_xinsert (tab, hp);
  // This must be the first insertion.
  (void) res;
  assert (res == hp);
}

/* A state_item from a state's id and the offset of the item within
   the state. */
state_item *
state_item_lookup (state_number s, state_item_number off)
{
  return &state_items[state_item_index_lookup (s, off)];
}

static inline void
state_item_set (state_item_number sidx, const state *s, item_number off)
{
  state_items[sidx].state = s;
  state_items[sidx].item = &ritem[off];
  state_items[sidx].lookahead = NULL;
  state_items[sidx].trans = -1;
  state_items[sidx].prods = NULL;
  state_items[sidx].revs = bitset_create (nstate_items, BITSET_SPARSE);
}

/**
 * Initialize state_items set
 */
static void
init_state_items (void)
{
  nstate_items = 0;
  bitsetv production_items = bitsetv_create (nstates, nritems, BITSET_SPARSE);
  for (int i = 0; i < nstates; ++i)
    {
      const state *s = states[i];
      nstate_items += s->nitems;
      closure (s->items, s->nitems);
      for (size_t j = 0; j < nitemset; ++j)
        if (0 < itemset[j]
            && item_number_is_rule_number (ritem[itemset[j] - 1]))
          {
            bitset_set (production_items[i], itemset[j]);
            ++nstate_items;
          }
    }
  state_item_map = xnmalloc (nstates + 1, sizeof (state_item_number));
  state_items = xnmalloc (nstate_items, sizeof (state_item));
  state_item_number sidx = 0;
  for (int i = 0; i < nstates; ++i)
    {
      state_item_map[i] = sidx;
      int rule_search_idx = 0;
      const state *s = states[i];
      const reductions *red = s->reductions;
      for (int j = 0; j < s->nitems; ++j)
        {
          state_item_set (sidx, s, s->items[j]);
          state_item *si = &state_items[sidx];
          const rule *r = item_rule (si->item);
          if (rule_search_idx < red->num && red->rules[rule_search_idx] < r)
            ++rule_search_idx;
          if (rule_search_idx < red->num && r == red->rules[rule_search_idx])
            {
              bitsetv lookahead = red->lookaheads;
              if (lookahead)
                si->lookahead = lookahead[rule_search_idx];
            }
          ++sidx;
        }
      bitset_iterator biter;
      item_number off;
      BITSET_FOR_EACH (biter, production_items[i], off, 0)
        {
          state_item_set (sidx, s, off);
          if (item_number_is_rule_number (ritem[off]))
            {
              bitsetv lookahead = red->lookaheads;
              if (lookahead)
                state_items[sidx].lookahead = lookahead[rule_search_idx];
              ++rule_search_idx;
            }
          ++sidx;
        }

    }
  state_item_map[nstates] = nstate_items;
  bitsetv_free (production_items);
}

static size_t
state_sym_hasher (const void *st, size_t max)
{
  return ((state *) st)->accessing_symbol % max;
}

static bool
state_sym_comparator (const void *s1, const void *s2)
{
  return ((state *) s1)->accessing_symbol == ((state *) s2)->accessing_symbol;
}

static state *
state_sym_lookup (symbol_number sym, Hash_table *h)
{
  state probe;
  probe.accessing_symbol = sym;
  return hash_lookup (h, &probe);
}

static void
init_trans (void)
{
  for (state_number i = 0; i < nstates; ++i)
    {
      // Generate a hash set that maps from accepting symbols to the states
      // this state transitions to.
      state *s = states[i];
      transitions *t = s->transitions;
      Hash_table *transition_set
        = hash_xinitialize (t->num, NULL, (Hash_hasher) state_sym_hasher,
                            (Hash_comparator) state_sym_comparator, NULL);
      for (int j = 0; j < t->num; ++j)
        if (!TRANSITION_IS_DISABLED (t, j))
          hash_xinsert (transition_set, t->states[j]);
      for (state_item_number j = state_item_map[i]; j < state_item_map[i + 1]; ++j)
        {
          item_number *item = state_items[j].item;
          if (item_number_is_rule_number (*item))
            continue;
          state *dst = state_sym_lookup (*item, transition_set);
          if (!dst)
            continue;
          // find the item in the destination state that corresponds
          // to the transition of item
          for (int k = 0; k < dst->nitems; ++k)
            if (item + 1 == ritem + dst->items[k])
              {
                state_item_number dstSI =
                  state_item_index_lookup (dst->number, k);

                state_items[j].trans = dstSI;
                bitset_set (state_items[dstSI].revs, j);
                break;
            }
        }
      hash_free (transition_set);
    }
}

static void
init_prods (void)
{
  for (int i = 0; i < nstates; ++i)
    {
      state *s = states[i];
      // closure_map is a hash map from nonterminals to a set
      // of the items that produce those nonterminals
      Hash_table *closure_map = hash_pair_table_create (nsyms - ntokens);

      // Add the nitems of state to skip to the production portion
      // of that state's state_items
      for (state_item_number j = state_item_map[i] + s->nitems;
           j < state_item_map[i + 1]; ++j)
        {
          state_item *src = &state_items[j];
          item_number *item = src->item;
          symbol_number lhs = item_rule (item)->lhs->number;
          bitset itms = hash_pair_lookup (closure_map, lhs);
          if (!itms)
            {
              itms = bitset_create (nstate_items, BITSET_SPARSE);
              hash_pair_insert (closure_map, lhs, itms);
            }
          bitset_set (itms, j);
        }
      // For each item with a dot followed by a nonterminal,
      // try to create a production edge.
      for (state_item_number j = state_item_map[i]; j < state_item_map[i + 1]; ++j)
        {
          state_item *src = &state_items[j];
          item_number item = *(src->item);
          // Skip reduce items and items with terminals after the dot
          if (item_number_is_rule_number (item) || ISTOKEN (item))
            continue;
          symbol_number sym = item_number_as_symbol_number (item);
          bitset lb = hash_pair_lookup (closure_map, sym);
          if (lb)
            {
              bitset copy = bitset_create (nstate_items, BITSET_SPARSE);
              bitset_copy (copy, lb);
              // update prods.
              state_items[j].prods = copy;

              // update revs.
              bitset_iterator biter;
              state_item_number prod;
              BITSET_FOR_EACH (biter, copy, prod, 0)
                bitset_set (state_items[prod].revs, j);
            }
        }
      hash_free (closure_map);
    }
}

/* Since lookaheads are only generated for reductions, we need to
   propagate lookahead sets backwards as the searches require each
   state_item to have a lookahead. */
static inline void
gen_lookaheads (void)
{
  for (state_item_number i = 0; i < nstate_items; ++i)
    {
      state_item *si = &state_items[i];
      if (item_number_is_symbol_number (*(si->item)) || !si->lookahead)
        continue;

      bitset lookahead = si->lookahead;
      state_item_list queue =
        gl_list_create (GL_LINKED_LIST, NULL, NULL, NULL, true, 1,
                        (const void **) &si);

      // For each reduction item, traverse through all state_items
      // accessible through reverse transition steps, and set their
      // lookaheads to the reduction items lookahead
      while (gl_list_size (queue) > 0)
        {
          state_item *prev = (state_item *) gl_list_get_at (queue, 0);
          gl_list_remove_at (queue, 0);
          prev->lookahead = lookahead;
          if (SI_TRANSITION (prev))
            {
              bitset rsi = state_items[prev - state_items].revs;
              bitset_iterator biter;
              state_item_number sin;
              BITSET_FOR_EACH (biter, rsi, sin, 0)
                gl_list_add_first (queue, &state_items[sin]);
            }
        }
      gl_list_free (queue);
    }
}

bitsetv firsts = NULL;

static void
init_firsts (void)
{
  firsts = bitsetv_create (nnterms, nsyms, BITSET_FIXED);
  for (rule_number i = 0; i < nrules; ++i)
    {
      rule *r = &rules[i];
      item_number *n = r->rhs;
      // Iterate through nullable nonterminals to try to find a terminal.
      while (item_number_is_symbol_number (*n) && ISVAR (*n)
             && nullable[*n - ntokens])
        ++n;
      if (item_number_is_rule_number (*n) || ISVAR (*n))
        continue;

      symbol_number lhs = r->lhs->number;
      bitset_set (FIRSTS (lhs), *n);
    }
  bool change = true;
  while (change)
    {
      change = false;
      for (rule_number i = 0; i < nrules; ++i)
        {
          rule *r = &rules[i];
          symbol_number lhs = r->lhs->number;
          bitset f_lhs = FIRSTS (lhs);
          for (item_number *n = r->rhs;
               item_number_is_symbol_number (*n) && ISVAR (*n);
               ++n)
            {
              bitset f = FIRSTS (*n);
              if (!bitset_subset_p (f_lhs, f))
                {
                  change = true;
                  bitset_union (f_lhs, f_lhs, f);
                }
              if (!nullable[*n - ntokens])
                break;
            }
        }
    }
}

static inline void
disable_state_item (state_item *si)
{
  si->trans = -2;
  bitset_free (si->revs);
  if (si->prods)
    bitset_free (si->prods);
}

/* Disable all state_item paths that lead to/from SI and nowhere
   else.  */
static void
prune_state_item (const state_item *si)
{
  state_item_list queue =
    gl_list_create (GL_LINKED_LIST, NULL, NULL, NULL, true, 1,
                    (const void **) &si);

  while (gl_list_size (queue) > 0)
    {
      state_item *dsi = (state_item *) gl_list_get_at (queue, 0);
      gl_list_remove_at (queue, 0);
      if (SI_DISABLED (dsi - state_items))
        continue;

      if (dsi->trans >= 0 && !SI_DISABLED (dsi->trans))
      {
        const state_item *trans = &state_items[dsi->trans];
        bitset_reset (trans->revs, dsi - state_items);
        if (bitset_empty_p (trans->revs))
          gl_list_add_last (queue, trans);
      }

      if (dsi->prods)
        {
          bitset_iterator biter;
          state_item_number sin;
          BITSET_FOR_EACH (biter, dsi->prods, sin, 0)
            {
              if (SI_DISABLED (sin))
                continue;
              const state_item *prod = &state_items[sin];
              bitset_reset (prod->revs, dsi - state_items);
              if (bitset_empty_p (prod->revs))
                gl_list_add_last (queue, prod);
            }
        }

      bitset_iterator biter;
      state_item_number sin;
      BITSET_FOR_EACH (biter, dsi->revs, sin, 0)
        {
          if (SI_DISABLED (sin))
            continue;
          state_item *rev = &state_items[sin];
          if (&state_items[rev->trans] == dsi)
            gl_list_add_last (queue, rev);
          else if (rev->prods)
            {
              bitset_reset (rev->prods, dsi - state_items);
              if (bitset_empty_p (rev->prods))
                gl_list_add_last (queue, rev);
            }
          else
            gl_list_add_last (queue, rev);
        }
      disable_state_item (dsi);
    }
  gl_list_free (queue);
}

/* To make searches more efficient, prune away paths that are caused
   by disabled transitions.  */
static void
prune_disabled_paths (void)
{
  for (int i = nstate_items - 1; i >= 0; --i)
    {
      state_item *si = &state_items[i];
      if (si->trans == -1 && item_number_is_symbol_number (*si->item))
        prune_state_item (si);
    }
}

void
state_item_print (const state_item *si, FILE *out, const char *prefix)
{
  fputs (prefix, out);
  item_print (si->item, NULL, out);
  putc ('\n', out);
}

/**
 * Report the state_item graph
 */
static void
state_items_report (FILE *out)
{
  fprintf (out, "# state items: %zu\n", nstate_items);
  for (state_number i = 0; i < nstates; ++i)
    {
      fprintf (out, "State %d:\n", i);
      for (state_item_number j = state_item_map[i]; j < state_item_map[i + 1]; ++j)
        {
          const state_item *si = &state_items[j];
          item_print (si->item, NULL, out);
          if (SI_DISABLED (j))
            fputs ("  DISABLED\n", out);
          else
            {
              putc ('\n', out);
              if (si->trans >= 0)
                {
                  fputs ("    -> ", out);
                  state_item_print (&state_items[si->trans], out, "");
                }

              bitset sets[2] = { si->prods, si->revs };
              const char *txt[2] = { "    => ", "    <- " };
              for (int seti = 0; seti < 2; ++seti)
                {
                  bitset b = sets[seti];
                  if (b)
                    {
                      bitset_iterator biter;
                      state_item_number sin;
                      BITSET_FOR_EACH (biter, b, sin, 0)
                        {
                          fputs (txt[seti], out);
                          state_item_print (&state_items[sin], out, "");
                        }
                    }
                }
            }
          putc ('\n', out);
        }
    }
  fprintf (out, "FIRSTS\n");
  for (symbol_number i = ntokens; i < nsyms; ++i)
    {
      fprintf (out, "  %s firsts\n", symbols[i]->tag);
      bitset_iterator iter;
      symbol_number j;
      BITSET_FOR_EACH (iter, FIRSTS (i), j, 0)
        fprintf (out, "    %s\n", symbols[j]->tag);
    }
  fputs ("\n\n", out);
}

void
state_items_init (void)
{
  time_t start = time (NULL);
  init_state_items ();
  init_trans ();
  init_prods ();
  gen_lookaheads ();
  init_firsts ();
  prune_disabled_paths ();
  if (trace_flag & trace_cex)
    {
      fprintf (stderr, "init: %f\n", difftime (time (NULL), start));
      state_items_report (stderr);
    }
}

void
state_items_free (void)
{
  for (state_item_number i = 0; i < nstate_items; ++i)
    if (!SI_DISABLED (i))
      {
        state_item *si = &state_items[i];
        if (si->prods)
          bitset_free (si->prods);
        bitset_free (si->revs);
      }
  free (state_items);
  bitsetv_free (firsts);
}

/**
 * Determine, using precedence and associativity, whether the next
 * production is allowed from the current production.
 */
bool
production_allowed (const state_item *si, const state_item *next)
{
  sym_content *s1 = item_rule (si->item)->lhs;
  sym_content *s2 = item_rule (next->item)->lhs;
  int prec1 = s1->prec;
  int prec2 = s2->prec;
  if (prec1 >= 0 && prec2 >= 0)
    {
      // Do not expand if lower precedence.
      if (prec1 > prec2)
        return false;
      // Do not expand if same precedence, but left-associative.
      if (prec1 == prec2 && s1->assoc == left_assoc)
        return false;
    }
  return true;
}
