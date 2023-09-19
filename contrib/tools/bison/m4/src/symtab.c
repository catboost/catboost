/* GNU m4 -- A simple macro processor

   Copyright (C) 1989-1994, 2003, 2006-2013 Free Software Foundation,
   Inc.

   This file is part of GNU M4.

   GNU M4 is free software: you can redistribute it and/or modify
   it under the terms of the GNU General Public License as published by
   the Free Software Foundation, either version 3 of the License, or
   (at your option) any later version.

   GNU M4 is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
   GNU General Public License for more details.

   You should have received a copy of the GNU General Public License
   along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/

/* This file handles all the low level work around the symbol table.  The
   symbol table is a simple chained hash table.  Each symbol is described
   by a struct symbol, which is placed in the hash table based upon the
   symbol name.  Symbols that hash to the same entry in the table are
   kept on a list, sorted by name.  As a special case, to facilitate the
   "pushdef" and "popdef" builtins, a symbol can be several times in the
   symbol table, one for each definition.  Since the name is the same,
   all the entries for the symbol will be on the same list, and will
   also, because the list is sorted, be adjacent.  All the entries for a
   name are simply ordered on the list by age.  The current definition
   will then always be the first found.  */

#include "m4.h"
#include <limits.h>

#ifdef DEBUG_SYM
/* When evaluating hash table performance, this profiling code shows
   how many collisions were encountered.  */

struct profile
{
  int entry; /* Number of times lookup_symbol called with this mode.  */
  int comparisons; /* Number of times strcmp was called.  */
  int misses; /* Number of times strcmp did not return 0.  */
  long long bytes; /* Number of bytes compared.  */
};

static struct profile profiles[5];
static symbol_lookup current_mode;

/* On exit, show a profile of symbol table performance.  */
static void
show_profile (void)
{
  int i;
  for (i = 0; i < 5; i++)
    {
      xfprintf(stderr, "m4: lookup mode %d called %d times, %d compares, "
               "%d misses, %lld bytes\n",
               i, profiles[i].entry, profiles[i].comparisons,
               profiles[i].misses, profiles[i].bytes);
    }
}

/* Like strcmp (S1, S2), but also track profiling statistics.  */
static int
profile_strcmp (const char *s1, const char *s2)
{
  int i = 1;
  int result;
  while (*s1 && *s1 == *s2)
    {
      s1++;
      s2++;
      i++;
    }
  result = (unsigned char) *s1 - (unsigned char) *s2;
  profiles[current_mode].comparisons++;
  if (result != 0)
    profiles[current_mode].misses++;
  profiles[current_mode].bytes += i;
  return result;
}

# define strcmp profile_strcmp
#endif /* DEBUG_SYM */


/*------------------------------------------------------------------.
| Initialise the symbol table, by allocating the necessary storage, |
| and zeroing all the entries.                                      |
`------------------------------------------------------------------*/

/* Pointer to symbol table.  */
symbol **symtab;

void
symtab_init (void)
{
  size_t i;
  symbol **s;

  s = symtab = (symbol **) xnmalloc (hash_table_size, sizeof (symbol *));

  for (i = 0; i < hash_table_size; i++)
    s[i] = NULL;

#ifdef DEBUG_SYM
  {
    int e = atexit(show_profile);
    if (e != 0)
      M4ERROR ((warning_status, 0,
                "INTERNAL ERROR: unable to show symtab profile"));
  }
#endif /* DEBUG_SYM */
}

/*--------------------------------------------------.
| Return a hashvalue for a string, from GNU-emacs.  |
`--------------------------------------------------*/

static size_t M4_GNUC_PURE
hash (const char *s)
{
  size_t val = 0;

  const char *ptr = s;
  char ch;

  while ((ch = *ptr++) != '\0')
    val = (val << 7) + (val >> (sizeof (val) * CHAR_BIT - 7)) + ch;
  return val;
}

/*--------------------------------------------.
| Free all storage associated with a symbol.  |
`--------------------------------------------*/

void
free_symbol (symbol *sym)
{
  if (SYMBOL_PENDING_EXPANSIONS (sym) > 0)
    SYMBOL_DELETED (sym) = true;
  else
    {
      free (SYMBOL_NAME (sym));
      if (SYMBOL_TYPE (sym) == TOKEN_TEXT)
        free (SYMBOL_TEXT (sym));
      free (sym);
    }
}

/*-------------------------------------------------------------------.
| Search in, and manipulation of the symbol table, are all done by   |
| lookup_symbol ().  It basically hashes NAME to a list in the       |
| symbol table, and searches this list for the first occurrence of a |
| symbol with the name.                                              |
|                                                                    |
| The MODE parameter determines what lookup_symbol () will do.  It   |
| can either just do a lookup, do a lookup and insert if not         |
| present, do an insertion even if the name is already in the list,  |
| delete the first occurrence of the name on the list, or delete all |
| occurrences of the name on the list.                               |
`-------------------------------------------------------------------*/

symbol *
lookup_symbol (const char *name, symbol_lookup mode)
{
  size_t h;
  int cmp = 1;
  symbol *sym, *prev;
  symbol **spp;

#if DEBUG_SYM
  current_mode = mode;
  profiles[mode].entry++;
#endif /* DEBUG_SYM */

  h = hash (name);
  sym = symtab[h % hash_table_size];

  for (prev = NULL; sym != NULL; prev = sym, sym = sym->next)
    {
      cmp = strcmp (SYMBOL_NAME (sym), name);
      if (cmp >= 0)
        break;
    }

  /* If just searching, return status of search.  */

  if (mode == SYMBOL_LOOKUP)
    return cmp == 0 ? sym : NULL;

  /* Symbol not found.  */

  spp = (prev != NULL) ?  &prev->next : &symtab[h % hash_table_size];

  switch (mode)
    {

    case SYMBOL_INSERT:

      /* If the name was found in the table, check whether it is still in
         use by a pending expansion.  If so, replace the table element with
         a new one; if not, just return the symbol.  If not found, just
         insert the name, and return the new symbol.  */

      if (cmp == 0 && sym != NULL)
        {
          if (SYMBOL_PENDING_EXPANSIONS (sym) > 0)
            {
              symbol *old = sym;
              SYMBOL_DELETED (old) = true;

              sym = (symbol *) xmalloc (sizeof (symbol));
              SYMBOL_TYPE (sym) = TOKEN_VOID;
              SYMBOL_TRACED (sym) = SYMBOL_TRACED (old);
              SYMBOL_NAME (sym) = xstrdup (name);
              SYMBOL_SHADOWED (sym) = false;
              SYMBOL_MACRO_ARGS (sym) = false;
              SYMBOL_BLIND_NO_ARGS (sym) = false;
              SYMBOL_DELETED (sym) = false;
              SYMBOL_PENDING_EXPANSIONS (sym) = 0;

              SYMBOL_NEXT (sym) = SYMBOL_NEXT (old);
              SYMBOL_NEXT (old) = NULL;
              (*spp) = sym;
            }
          return sym;
        }
      /* Fall through.  */

    case SYMBOL_PUSHDEF:

      /* Insert a name in the symbol table.  If there is already a symbol
         with the name, insert this in front of it, and mark the old
         symbol as "shadowed".  */

      sym = (symbol *) xmalloc (sizeof (symbol));
      SYMBOL_TYPE (sym) = TOKEN_VOID;
      SYMBOL_TRACED (sym) = false;
      SYMBOL_NAME (sym) = xstrdup (name);
      SYMBOL_SHADOWED (sym) = false;
      SYMBOL_MACRO_ARGS (sym) = false;
      SYMBOL_BLIND_NO_ARGS (sym) = false;
      SYMBOL_DELETED (sym) = false;
      SYMBOL_PENDING_EXPANSIONS (sym) = 0;

      SYMBOL_NEXT (sym) = *spp;
      (*spp) = sym;

      if (mode == SYMBOL_PUSHDEF && cmp == 0)
        {
          SYMBOL_SHADOWED (SYMBOL_NEXT (sym)) = true;
          SYMBOL_TRACED (sym) = SYMBOL_TRACED (SYMBOL_NEXT (sym));
        }
      return sym;

    case SYMBOL_DELETE:
    case SYMBOL_POPDEF:

      /* Delete occurrences of symbols with NAME.  SYMBOL_DELETE kills
         all definitions, SYMBOL_POPDEF kills only the first.
         However, if the last instance of a symbol is marked for
         tracing, reinsert a placeholder in the table.  And if the
         definition is still in use, let the caller free the memory
         after it is done with the symbol.  */

      if (cmp != 0 || sym == NULL)
        return NULL;
      {
        bool traced = false;
        if (SYMBOL_NEXT (sym) != NULL
            && SYMBOL_SHADOWED (SYMBOL_NEXT (sym))
            && mode == SYMBOL_POPDEF)
          {
            SYMBOL_SHADOWED (SYMBOL_NEXT (sym)) = false;
            SYMBOL_TRACED (SYMBOL_NEXT (sym)) = SYMBOL_TRACED (sym);
          }
        else
          traced = SYMBOL_TRACED (sym);
        do
          {
            *spp = SYMBOL_NEXT (sym);
            free_symbol (sym);
            sym = *spp;
          }
        while (*spp != NULL && SYMBOL_SHADOWED (*spp)
               && mode == SYMBOL_DELETE);
        if (traced)
          {
            sym = (symbol *) xmalloc (sizeof (symbol));
            SYMBOL_TYPE (sym) = TOKEN_VOID;
            SYMBOL_TRACED (sym) = true;
            SYMBOL_NAME (sym) = xstrdup (name);
            SYMBOL_SHADOWED (sym) = false;
            SYMBOL_MACRO_ARGS (sym) = false;
            SYMBOL_BLIND_NO_ARGS (sym) = false;
            SYMBOL_DELETED (sym) = false;
            SYMBOL_PENDING_EXPANSIONS (sym) = 0;

            SYMBOL_NEXT (sym) = *spp;
            (*spp) = sym;
          }
      }
      return NULL;

    case SYMBOL_LOOKUP:
    default:
      M4ERROR ((warning_status, 0,
                "INTERNAL ERROR: invalid mode to symbol_lookup ()"));
      abort ();
    }
}

/*-----------------------------------------------------------------.
| The following function is used for the cases where we want to do |
| something to each and every symbol in the table.  The function   |
| hack_all_symbols () traverses the symbol table, and calls a      |
| specified function FUNC for each symbol in the table.  FUNC is   |
| called with a pointer to the symbol, and the DATA argument.      |
|                                                                  |
| FUNC may safely call lookup_symbol with mode SYMBOL_POPDEF or    |
| SYMBOL_LOOKUP, but any other mode can break the iteration.       |
`-----------------------------------------------------------------*/

void
hack_all_symbols (hack_symbol *func, void *data)
{
  size_t h;
  symbol *sym;
  symbol *next;

  for (h = 0; h < hash_table_size; h++)
    {
      /* We allow func to call SYMBOL_POPDEF, which can invalidate
         sym, so we must grab the next element to traverse before
         calling func.  */
      for (sym = symtab[h]; sym != NULL; sym = next)
        {
          next = SYMBOL_NEXT (sym);
          func (sym, data);
        }
    }
}

#ifdef DEBUG_SYM

static void symtab_print_list (int i);

static void M4_GNUC_UNUSED
symtab_debug (void)
{
  token_data td;
  const char *text;
  symbol *s;
  int delete;
  static int i;

  while (next_token (&td, NULL) == TOKEN_WORD)
    {
      text = TOKEN_DATA_TEXT (&td);
      if (*text == '_')
        {
          delete = 1;
          text++;
        }
      else
        delete = 0;

      s = lookup_symbol (text, SYMBOL_LOOKUP);

      if (s == NULL)
        xprintf ("Name `%s' is unknown\n", text);

      lookup_symbol (text, delete ? SYMBOL_DELETE : SYMBOL_INSERT);
    }
  symtab_print_list (i++);
}

static void
symtab_print_list (int i)
{
  symbol *sym;
  size_t h;

  xprintf ("Symbol dump #%d:\n", i);
  for (h = 0; h < hash_table_size; h++)
    for (sym = symtab[h]; sym != NULL; sym = sym->next)
      xprintf ("\tname %s, bucket %lu, addr %p, next %p, "
               "flags%s%s%s, pending %d\n",
               SYMBOL_NAME (sym),
               (unsigned long int) h, sym, SYMBOL_NEXT (sym),
               SYMBOL_TRACED (sym) ? " traced" : "",
               SYMBOL_SHADOWED (sym) ? " shadowed" : "",
               SYMBOL_DELETED (sym) ? " deleted" : "",
               SYMBOL_PENDING_EXPANSIONS (sym));
}

#endif /* DEBUG_SYM */
