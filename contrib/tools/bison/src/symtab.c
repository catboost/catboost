/* Symbol table manager for Bison.

   Copyright (C) 1984, 1989, 2000-2002, 2004-2015, 2018-2020 Free
   Software Foundation, Inc.

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
   along with this program.  If not, see <http://www.gnu.org/licenses/>.  */

#include <config.h>
#include "symtab.h"

#include "system.h"

#include <assure.h>
#include <fstrcmp.h>
#include <hash.h>
#include <quote.h>

#include "complain.h"
#include "getargs.h"
#include "gram.h"
#include "intprops.h"

static struct hash_table *symbol_table = NULL;
static struct hash_table *semantic_type_table = NULL;

/*----------------------------------------------------------------.
| Symbols sorted by tag.  Allocated by table_sort, after which no |
| more symbols should be created.                                 |
`----------------------------------------------------------------*/

static symbol **symbols_sorted = NULL;
static semantic_type **semantic_types_sorted = NULL;


/*------------------------.
| Distinguished symbols.  |
`------------------------*/

symbol *errtoken = NULL;
symbol *undeftoken = NULL;
symbol *endtoken = NULL;
symbol *accept = NULL;
symbol *startsymbol = NULL;
location startsymbol_loc;

/* Precedence relation graph. */
static symgraph **prec_nodes;

/* Store which associativity is used.  */
static bool *used_assoc = NULL;

bool tag_seen = false;


/*--------------------------.
| Create a new sym_content. |
`--------------------------*/

static sym_content *
sym_content_new (symbol *s)
{
  sym_content *res = xmalloc (sizeof *res);

  res->symbol = s;

  res->type_name = NULL;
  res->type_loc = empty_loc;
  for (int i = 0; i < CODE_PROPS_SIZE; ++i)
    code_props_none_init (&res->props[i]);

  res->number = NUMBER_UNDEFINED;
  res->prec_loc = empty_loc;
  res->prec = 0;
  res->assoc = undef_assoc;
  res->user_token_number = USER_NUMBER_UNDEFINED;

  res->class = unknown_sym;
  res->status = undeclared;

  return res;
}

/*---------------------------------.
| Create a new symbol, named TAG.  |
`---------------------------------*/

static symbol *
symbol_new (uniqstr tag, location loc)
{
  symbol *res = xmalloc (sizeof *res);
  uniqstr_assert (tag);

  /* If the tag is not a string (starts with a double quote), check
     that it is valid for Yacc. */
  if (tag[0] != '\"' && tag[0] != '\'' && strchr (tag, '-'))
    complain (&loc, Wyacc,
              _("POSIX Yacc forbids dashes in symbol names: %s"), tag);

  res->tag = tag;
  res->location = loc;
  res->location_of_lhs = false;
  res->alias = NULL;
  res->content = sym_content_new (res);
  res->is_alias = false;

  if (nsyms == SYMBOL_NUMBER_MAXIMUM)
    complain (NULL, fatal, _("too many symbols in input grammar (limit is %d)"),
              SYMBOL_NUMBER_MAXIMUM);
  nsyms++;
  return res;
}

/*--------------------.
| Free a sym_content. |
`--------------------*/

static void
sym_content_free (sym_content *sym)
{
  free (sym);
}


/*---------------------------------------------------------.
| Free a symbol and its associated content if appropriate. |
`---------------------------------------------------------*/

static void
symbol_free (void *ptr)
{
  symbol *sym = (symbol *)ptr;
  if (!sym->is_alias)
    sym_content_free (sym->content);
  free (sym);

}

/* If needed, swap first and second so that first has the earliest
   location (according to location_cmp).

   Many symbol features (e.g., user token numbers) are not assigned
   during the parsing, but in a second step, via a traversal of the
   symbol table sorted on tag.

   However, error messages make more sense if we keep the first
   declaration first.
*/

static void
symbols_sort (symbol **first, symbol **second)
{
  if (0 < location_cmp ((*first)->location, (*second)->location))
    {
      symbol* tmp = *first;
      *first = *second;
      *second = tmp;
    }
}

/* Likewise, for locations.  */

static void
locations_sort (location *first, location *second)
{
  if (0 < location_cmp (*first, *second))
    {
      location tmp = *first;
      *first = *second;
      *second = tmp;
    }
}

char const *
code_props_type_string (code_props_type kind)
{
  switch (kind)
    {
    case destructor:
      return "%destructor";
    case printer:
      return "%printer";
    }
  abort ();
}


/*----------------------------------------.
| Create a new semantic type, named TAG.  |
`----------------------------------------*/

static semantic_type *
semantic_type_new (uniqstr tag, const location *loc)
{
  semantic_type *res = xmalloc (sizeof *res);

  uniqstr_assert (tag);
  res->tag = tag;
  res->location = loc ? *loc : empty_loc;
  res->status = undeclared;
  for (int i = 0; i < CODE_PROPS_SIZE; ++i)
    code_props_none_init (&res->props[i]);

  return res;
}


/*-----------------.
| Print a symbol.  |
`-----------------*/

#define SYMBOL_ATTR_PRINT(Attr)                         \
  if (s->content && s->content->Attr)                   \
    fprintf (f, " %s { %s }", #Attr, s->content->Attr)

#define SYMBOL_CODE_PRINT(Attr)                                         \
  if (s->content && s->content->props[Attr].code)                       \
    fprintf (f, " %s { %s }", #Attr, s->content->props[Attr].code)

void
symbol_print (symbol const *s, FILE *f)
{
  if (s)
    {
      symbol_class c = s->content->class;
      fprintf (f, "%s: %s",
               c == unknown_sym    ? "unknown"
               : c == pct_type_sym ? "%type"
               : c == token_sym    ? "token"
               : c == nterm_sym    ? "nterm"
               : NULL, /* abort.  */
               s->tag);
      SYMBOL_ATTR_PRINT (type_name);
      SYMBOL_CODE_PRINT (destructor);
      SYMBOL_CODE_PRINT (printer);
    }
  else
    fputs ("<NULL>", f);
}

#undef SYMBOL_ATTR_PRINT
#undef SYMBOL_CODE_PRINT


/*----------------------------------.
| Whether S is a valid identifier.  |
`----------------------------------*/

static bool
is_identifier (uniqstr s)
{
  static char const alphanum[26 + 26 + 1 + 10] =
    "abcdefghijklmnopqrstuvwxyz"
    "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    "_"
    "0123456789";
  if (!s || ! memchr (alphanum, *s, sizeof alphanum - 10))
    return false;
  for (++s; *s; ++s)
    if (! memchr (alphanum, *s, sizeof alphanum))
      return false;
  return true;
}


/*-----------------------------------------------.
| Get the identifier associated to this symbol.  |
`-----------------------------------------------*/
uniqstr
symbol_id_get (symbol const *sym)
{
  if (sym->alias)
    sym = sym->alias;
  return is_identifier (sym->tag) ? sym->tag : 0;
}


/*------------------------------------------------------------------.
| Complain that S's WHAT is redeclared at SECOND, and was first set |
| at FIRST.                                                         |
`------------------------------------------------------------------*/

static void
complain_symbol_redeclared (symbol *s, const char *what, location first,
                            location second)
{
  int i = 0;
  locations_sort (&first, &second);
  complain_indent (&second, complaint, &i,
                   _("%s redeclaration for %s"), what, s->tag);
  i += SUB_INDENT;
  complain_indent (&first, complaint, &i,
                   _("previous declaration"));
}

static void
complain_semantic_type_redeclared (semantic_type *s, const char *what, location first,
                                   location second)
{
  int i = 0;
  locations_sort (&first, &second);
  complain_indent (&second, complaint, &i,
                   _("%s redeclaration for <%s>"), what, s->tag);
  i += SUB_INDENT;
  complain_indent (&first, complaint, &i,
                   _("previous declaration"));
}

static void
complain_class_redeclared (symbol *sym, symbol_class class, location second)
{
  int i = 0;
  complain_indent (&second, complaint, &i,
                   class == token_sym
                   ? _("symbol %s redeclared as a token")
                   : _("symbol %s redeclared as a nonterminal"), sym->tag);
  if (!location_empty (sym->location))
    {
      i += SUB_INDENT;
      complain_indent (&sym->location, complaint, &i,
                       _("previous definition"));
    }
}

static const symbol *
symbol_from_uniqstr_fuzzy (const uniqstr key)
{
  aver (symbols_sorted);
#define FSTRCMP_THRESHOLD 0.6
  double best_similarity = FSTRCMP_THRESHOLD;
  const symbol *res = NULL;
  size_t count = hash_get_n_entries (symbol_table);
  for (int i = 0; i < count; ++i)
    {
      symbol *sym = symbols_sorted[i];
      if (STRNEQ (key, sym->tag)
          && (sym->content->status == declared
              || sym->content->status == undeclared))
        {
          double similarity = fstrcmp_bounded (key, sym->tag, best_similarity);
          if (best_similarity < similarity)
            {
              res = sym;
              best_similarity = similarity;
            }
        }
    }
  return res;
}

static void
complain_symbol_undeclared (symbol *sym)
{
  assert (sym->content->status != declared);
  const symbol *best = symbol_from_uniqstr_fuzzy (sym->tag);
  if (best)
    {
      complain (&sym->location,
                sym->content->status == needed ? complaint : Wother,
                _("symbol %s is used, but is not defined as a token"
                  " and has no rules; did you mean %s?"),
                quote_n (0, sym->tag),
                quote_n (1, best->tag));
      if (feature_flag & feature_caret)
        location_caret_suggestion (sym->location, best->tag, stderr);
    }
  else
    complain (&sym->location,
              sym->content->status == needed ? complaint : Wother,
              _("symbol %s is used, but is not defined as a token"
                " and has no rules"),
              quote (sym->tag));
}

void
symbol_location_as_lhs_set (symbol *sym, location loc)
{
  if (!sym->location_of_lhs)
    sym->location = loc;
}


/*-----------------------------------------------------------------.
| Set the TYPE_NAME associated with SYM.  Does nothing if passed 0 |
| as TYPE_NAME.                                                    |
`-----------------------------------------------------------------*/

void
symbol_type_set (symbol *sym, uniqstr type_name, location loc)
{
  if (type_name)
    {
      tag_seen = true;
      if (sym->content->type_name)
        complain_symbol_redeclared (sym, "%type",
                                    sym->content->type_loc, loc);
      else
        {
          uniqstr_assert (type_name);
          sym->content->type_name = type_name;
          sym->content->type_loc = loc;
        }
    }
}

/*--------------------------------------------------------.
| Set the DESTRUCTOR or PRINTER associated with the SYM.  |
`--------------------------------------------------------*/

void
symbol_code_props_set (symbol *sym, code_props_type kind,
                       code_props const *code)
{
  if (sym->content->props[kind].code)
    complain_symbol_redeclared (sym, code_props_type_string (kind),
                                sym->content->props[kind].location,
                                code->location);
  else
    sym->content->props[kind] = *code;
}

/*-----------------------------------------------------.
| Set the DESTRUCTOR or PRINTER associated with TYPE.  |
`-----------------------------------------------------*/

void
semantic_type_code_props_set (semantic_type *type,
                              code_props_type kind,
                              code_props const *code)
{
  if (type->props[kind].code)
    complain_semantic_type_redeclared (type, code_props_type_string (kind),
                                       type->props[kind].location,
                                       code->location);
  else
    type->props[kind] = *code;
}

/*---------------------------------------------------.
| Get the computed %destructor or %printer for SYM.  |
`---------------------------------------------------*/

code_props *
symbol_code_props_get (symbol *sym, code_props_type kind)
{
  /* Per-symbol code props.  */
  if (sym->content->props[kind].code)
    return &sym->content->props[kind];

  /* Per-type code props.  */
  if (sym->content->type_name)
    {
      code_props *code =
        &semantic_type_get (sym->content->type_name, NULL)->props[kind];
      if (code->code)
        return code;
    }

  /* Apply default code props's only to user-defined symbols.  */
  if (sym->tag[0] != '$' && sym != errtoken)
    {
      code_props *code = &semantic_type_get (sym->content->type_name ? "*" : "",
                                             NULL)->props[kind];
      if (code->code)
        return code;
    }
  return &code_props_none;
}

/*-----------------------------------------------------------------.
| Set the PRECEDENCE associated with SYM.  Does nothing if invoked |
| with UNDEF_ASSOC as ASSOC.                                       |
`-----------------------------------------------------------------*/

void
symbol_precedence_set (symbol *sym, int prec, assoc a, location loc)
{
  if (a != undef_assoc)
    {
      sym_content *s = sym->content;
      if (s->prec)
        complain_symbol_redeclared (sym, assoc_to_string (a),
                                    s->prec_loc, loc);
      else
        {
          s->prec = prec;
          s->assoc = a;
          s->prec_loc = loc;
        }
    }

  /* Only terminals have a precedence. */
  symbol_class_set (sym, token_sym, loc, false);
}


/*------------------------------------.
| Set the CLASS associated with SYM.  |
`------------------------------------*/

static void
complain_pct_type_on_token (location *loc)
{
  complain (loc, Wyacc,
            _("POSIX yacc reserves %%type to nonterminals"));
}

void
symbol_class_set (symbol *sym, symbol_class class, location loc, bool declaring)
{
  aver (class != unknown_sym);
  sym_content *s = sym->content;
  if (class == pct_type_sym)
    {
      if (s->class == token_sym)
        complain_pct_type_on_token (&loc);
      else if (s->class == unknown_sym)
        s->class = class;
    }
  else if (s->class != unknown_sym && s->class != pct_type_sym
           && s->class != class)
    complain_class_redeclared (sym, class, loc);
  else
    {
      if (class == token_sym && s->class == pct_type_sym)
        complain_pct_type_on_token (&sym->location);

      if (class == nterm_sym && s->class != nterm_sym)
        s->number = nvars++;
      else if (class == token_sym && s->number == NUMBER_UNDEFINED)
        s->number = ntokens++;
      s->class = class;

      if (declaring)
        {
          if (s->status == declared)
            {
              int i = 0;
              complain_indent (&loc, Wother, &i,
                               _("symbol %s redeclared"), sym->tag);
              i += SUB_INDENT;
              complain_indent (&sym->location, Wother, &i,
                               _("previous declaration"));
            }
          else
            {
              sym->location = loc;
              s->status = declared;
            }
        }
    }
}


/*------------------------------------------------.
| Set the USER_TOKEN_NUMBER associated with SYM.  |
`------------------------------------------------*/

void
symbol_user_token_number_set (symbol *sym, int user_token_number, location loc)
{
  int *user_token_numberp = &sym->content->user_token_number;
  if (sym->content->class != token_sym)
    complain (&loc, complaint,
              _("nonterminals cannot be given an explicit number"));
  else if (*user_token_numberp != USER_NUMBER_UNDEFINED
           && *user_token_numberp != user_token_number)
    complain (&loc, complaint, _("redefining user token number of %s"),
              sym->tag);
  else if (user_token_number == INT_MAX)
    complain (&loc, complaint, _("user token number of %s too large"),
              sym->tag);
  else
    {
      *user_token_numberp = user_token_number;
      /* User defined $end token? */
      if (user_token_number == 0 && !endtoken)
        {
          endtoken = sym->content->symbol;
          /* It is always mapped to 0, so it was already counted in
             NTOKENS.  */
          if (endtoken->content->number != NUMBER_UNDEFINED)
            --ntokens;
          endtoken->content->number = 0;
        }
    }
}


/*----------------------------------------------------------.
| If SYM is not defined, report an error, and consider it a |
| nonterminal.                                              |
`----------------------------------------------------------*/

static void
symbol_check_defined (symbol *sym)
{
  sym_content *s = sym->content;
  if (s->class == unknown_sym || s->class == pct_type_sym)
    {
      complain_symbol_undeclared (sym);
      s->class = nterm_sym;
      s->number = nvars++;
    }

  if (s->class == token_sym
      && sym->tag[0] == '"'
      && !sym->is_alias)
    complain (&sym->location, Wdangling_alias,
              _("string literal %s not attached to a symbol"),
              sym->tag);

  for (int i = 0; i < 2; ++i)
    symbol_code_props_get (sym, i)->is_used = true;

  /* Set the semantic type status associated to the current symbol to
     'declared' so that we could check semantic types unnecessary uses. */
  if (s->type_name)
    {
      semantic_type *sem_type = semantic_type_get (s->type_name, NULL);
      if (sem_type)
        sem_type->status = declared;
    }
}

static void
semantic_type_check_defined (semantic_type *sem_type)
{
  /* <*> and <> do not have to be "declared".  */
  if (sem_type->status == declared
      || !*sem_type->tag
      || STREQ (sem_type->tag, "*"))
    {
      for (int i = 0; i < 2; ++i)
        if (sem_type->props[i].kind != CODE_PROPS_NONE
            && ! sem_type->props[i].is_used)
          complain (&sem_type->location, Wother,
                    _("useless %s for type <%s>"),
                    code_props_type_string (i), sem_type->tag);
    }
  else
    complain (&sem_type->location, Wother,
              _("type <%s> is used, but is not associated to any symbol"),
              sem_type->tag);
}

/*-------------------------------------------------------------------.
| Merge the properties (precedence, associativity, etc.) of SYM, and |
| its string-named alias STR; check consistency.                     |
`-------------------------------------------------------------------*/

static void
symbol_merge_properties (symbol *sym, symbol *str)
{
  if (str->content->type_name != sym->content->type_name)
    {
      if (str->content->type_name)
        symbol_type_set (sym,
                         str->content->type_name, str->content->type_loc);
      else
        symbol_type_set (str,
                         sym->content->type_name, sym->content->type_loc);
    }


  for (int i = 0; i < CODE_PROPS_SIZE; ++i)
    if (str->content->props[i].code)
      symbol_code_props_set (sym, i, &str->content->props[i]);
    else if (sym->content->props[i].code)
      symbol_code_props_set (str, i, &sym->content->props[i]);

  if (sym->content->prec || str->content->prec)
    {
      if (str->content->prec)
        symbol_precedence_set (sym, str->content->prec, str->content->assoc,
                               str->content->prec_loc);
      else
        symbol_precedence_set (str, sym->content->prec, sym->content->assoc,
                               sym->content->prec_loc);
    }
}

void
symbol_make_alias (symbol *sym, symbol *str, location loc)
{
  if (sym->content->class != token_sym)
    complain (&loc, complaint,
              _("nonterminals cannot be given a string alias"));
  else if (str->alias)
    complain (&loc, Wother,
              _("symbol %s used more than once as a literal string"), str->tag);
  else if (sym->alias)
    complain (&loc, Wother,
              _("symbol %s given more than one literal string"), sym->tag);
  else
    {
      symbol_merge_properties (sym, str);
      sym_content_free (str->content);
      str->content = sym->content;
      str->content->symbol = str;
      str->is_alias = true;
      str->alias = sym;
      sym->alias = str;
    }
}


/*-------------------------------------------------------------------.
| Assign a symbol number, and write the definition of the token name |
| into FDEFINES.  Put in SYMBOLS.                                    |
`-------------------------------------------------------------------*/

static void
symbol_pack (symbol *this)
{
  aver (this->content->number != NUMBER_UNDEFINED);
  if (this->content->class == nterm_sym)
    this->content->number += ntokens;

  symbols[this->content->number] = this->content->symbol;
}

static void
complain_user_token_number_redeclared (int num, symbol *first, symbol *second)
{
  int i = 0;
  symbols_sort (&first, &second);
  complain_indent (&second->location, complaint, &i,
                   _("user token number %d redeclaration for %s"),
                   num, second->tag);
  i += SUB_INDENT;
  complain_indent (&first->location, complaint, &i,
                   _("previous declaration for %s"),
                   first->tag);
}

/*--------------------------------------------------.
| Put THIS in TOKEN_TRANSLATIONS if it is a token.  |
`--------------------------------------------------*/

static void
symbol_translation (symbol *this)
{
  /* Nonterminal? */
  if (this->content->class == token_sym
      && !this->is_alias)
    {
      /* A token which translation has already been set?*/
      if (token_translations[this->content->user_token_number]
          != undeftoken->content->number)
        complain_user_token_number_redeclared
          (this->content->user_token_number,
           symbols[token_translations[this->content->user_token_number]], this);
      else
        token_translations[this->content->user_token_number]
          = this->content->number;
    }
}


/*---------------------------------------.
| Symbol and semantic type hash tables.  |
`---------------------------------------*/

/* Initial capacity of symbol and semantic type hash table.  */
#define HT_INITIAL_CAPACITY 257

static inline bool
hash_compare_symbol (const symbol *m1, const symbol *m2)
{
  /* Since tags are unique, we can compare the pointers themselves.  */
  return UNIQSTR_EQ (m1->tag, m2->tag);
}

static inline bool
hash_compare_semantic_type (const semantic_type *m1, const semantic_type *m2)
{
  /* Since names are unique, we can compare the pointers themselves.  */
  return UNIQSTR_EQ (m1->tag, m2->tag);
}

static bool
hash_symbol_comparator (void const *m1, void const *m2)
{
  return hash_compare_symbol (m1, m2);
}

static bool
hash_semantic_type_comparator (void const *m1, void const *m2)
{
  return hash_compare_semantic_type (m1, m2);
}

static inline size_t
hash_symbol (const symbol *m, size_t tablesize)
{
  /* Since tags are unique, we can hash the pointer itself.  */
  return ((uintptr_t) m->tag) % tablesize;
}

static inline size_t
hash_semantic_type (const semantic_type *m, size_t tablesize)
{
  /* Since names are unique, we can hash the pointer itself.  */
  return ((uintptr_t) m->tag) % tablesize;
}

static size_t
hash_symbol_hasher (void const *m, size_t tablesize)
{
  return hash_symbol (m, tablesize);
}

static size_t
hash_semantic_type_hasher (void const *m, size_t tablesize)
{
  return hash_semantic_type (m, tablesize);
}

/*-------------------------------.
| Create the symbol hash table.  |
`-------------------------------*/

void
symbols_new (void)
{
  symbol_table = hash_xinitialize (HT_INITIAL_CAPACITY,
                                   NULL,
                                   hash_symbol_hasher,
                                   hash_symbol_comparator,
                                   symbol_free);

  /* Construct the accept symbol. */
  accept = symbol_get ("$accept", empty_loc);
  accept->content->class = nterm_sym;
  accept->content->number = nvars++;

  /* Construct the error token */
  errtoken = symbol_get ("error", empty_loc);
  errtoken->content->class = token_sym;
  errtoken->content->number = ntokens++;

  /* Construct a token that represents all undefined literal tokens.
     It is always token number 2.  */
  undeftoken = symbol_get ("$undefined", empty_loc);
  undeftoken->content->class = token_sym;
  undeftoken->content->number = ntokens++;

  semantic_type_table = hash_xinitialize (HT_INITIAL_CAPACITY,
                                          NULL,
                                          hash_semantic_type_hasher,
                                          hash_semantic_type_comparator,
                                          free);
}


/*----------------------------------------------------------------.
| Find the symbol named KEY, and return it.  If it does not exist |
| yet, create it.                                                 |
`----------------------------------------------------------------*/

symbol *
symbol_from_uniqstr (const uniqstr key, location loc)
{
  symbol probe;

  probe.tag = key;
  symbol *entry = hash_lookup (symbol_table, &probe);

  if (!entry)
    {
      /* First insertion in the hash. */
      aver (!symbols_sorted);
      entry = symbol_new (key, loc);
      if (!hash_insert (symbol_table, entry))
        xalloc_die ();
    }
  return entry;
}


/*-----------------------------------------------------------------------.
| Find the semantic type named KEY, and return it.  If it does not exist |
| yet, create it.                                                        |
`-----------------------------------------------------------------------*/

semantic_type *
semantic_type_from_uniqstr (const uniqstr key, const location *loc)
{
  semantic_type probe;

  probe.tag = key;
  semantic_type *entry = hash_lookup (semantic_type_table, &probe);

  if (!entry)
    {
      /* First insertion in the hash. */
      entry = semantic_type_new (key, loc);
      if (!hash_insert (semantic_type_table, entry))
        xalloc_die ();
    }
  return entry;
}


/*----------------------------------------------------------------.
| Find the symbol named KEY, and return it.  If it does not exist |
| yet, create it.                                                 |
`----------------------------------------------------------------*/

symbol *
symbol_get (const char *key, location loc)
{
  return symbol_from_uniqstr (uniqstr_new (key), loc);
}


/*-----------------------------------------------------------------------.
| Find the semantic type named KEY, and return it.  If it does not exist |
| yet, create it.                                                        |
`-----------------------------------------------------------------------*/

semantic_type *
semantic_type_get (const char *key, const location *loc)
{
  return semantic_type_from_uniqstr (uniqstr_new (key), loc);
}


/*------------------------------------------------------------------.
| Generate a dummy nonterminal, whose name cannot conflict with the |
| user's names.                                                     |
`------------------------------------------------------------------*/

symbol *
dummy_symbol_get (location loc)
{
  /* Incremented for each generated symbol.  */
  static int dummy_count = 0;
  char buf[32];
  int len = snprintf (buf, sizeof buf, "$@%d", ++dummy_count);
  assure (len < sizeof buf);
  symbol *sym = symbol_get (buf, loc);
  sym->content->class = nterm_sym;
  sym->content->number = nvars++;
  return sym;
}

bool
symbol_is_dummy (const symbol *sym)
{
  return sym->tag[0] == '@' || (sym->tag[0] == '$' && sym->tag[1] == '@');
}

/*-------------------.
| Free the symbols.  |
`-------------------*/

void
symbols_free (void)
{
  hash_free (symbol_table);
  hash_free (semantic_type_table);
  free (symbols);
  free (symbols_sorted);
  free (semantic_types_sorted);
}


static int
symbol_cmp (void const *a, void const *b)
{
  return location_cmp ((*(symbol * const *)a)->location,
                       (*(symbol * const *)b)->location);
}

/* Store in *SORTED an array of pointers to the symbols contained in
   TABLE, sorted (alphabetically) by tag. */

static void
table_sort (struct hash_table *table, symbol ***sorted)
{
  aver (!*sorted);
  size_t count = hash_get_n_entries (table);
  *sorted = xnmalloc (count + 1, sizeof **sorted);
  hash_get_entries (table, (void**)*sorted, count);
  qsort (*sorted, count, sizeof **sorted, symbol_cmp);
  (*sorted)[count] = NULL;
}

/*--------------------------------------------------------------.
| Check that all the symbols are defined.  Report any undefined |
| symbols and consider them nonterminals.                       |
`--------------------------------------------------------------*/

void
symbols_check_defined (void)
{
  table_sort (symbol_table, &symbols_sorted);
  /* semantic_type, like symbol, starts with a 'tag' field and then a
     'location' field.  And here we only deal with arrays/hashes of
     pointers, sizeof is not an issue.

     So instead of implementing table_sort (and symbol_cmp) once for
     each type, let's lie a bit to the typing system, and treat
     'semantic_type' as if it were 'symbol'. */
  table_sort (semantic_type_table, (symbol ***) &semantic_types_sorted);

  for (int i = 0; symbols_sorted[i]; ++i)
    symbol_check_defined (symbols_sorted[i]);
  for (int i = 0; semantic_types_sorted[i]; ++i)
    semantic_type_check_defined (semantic_types_sorted[i]);
}

/*------------------------------------------------------------------.
| Set TOKEN_TRANSLATIONS.  Check that no two symbols share the same |
| number.                                                           |
`------------------------------------------------------------------*/

static void
symbols_token_translations_init (void)
{
  bool num_256_available_p = true;

  /* Find the highest user token number, and whether 256, the POSIX
     preferred user token number for the error token, is used.  */
  max_user_token_number = 0;
  for (int i = 0; i < ntokens; ++i)
    {
      sym_content *this = symbols[i]->content;
      if (this->user_token_number != USER_NUMBER_UNDEFINED)
        {
          if (this->user_token_number > max_user_token_number)
            max_user_token_number = this->user_token_number;
          if (this->user_token_number == 256)
            num_256_available_p = false;
        }
    }

  /* If 256 is not used, assign it to error, to follow POSIX.  */
  if (num_256_available_p
      && errtoken->content->user_token_number == USER_NUMBER_UNDEFINED)
    errtoken->content->user_token_number = 256;

  /* Set the missing user numbers. */
  if (max_user_token_number < 256)
    max_user_token_number = 256;

  for (int i = 0; i < ntokens; ++i)
    {
      sym_content *this = symbols[i]->content;
      if (this->user_token_number == USER_NUMBER_UNDEFINED)
        {
          IGNORE_TYPE_LIMITS_BEGIN
          if (INT_ADD_WRAPV (max_user_token_number, 1, &max_user_token_number))
            complain (NULL, fatal, _("token number too large"));
          IGNORE_TYPE_LIMITS_END
          this->user_token_number = max_user_token_number;
        }
      if (this->user_token_number > max_user_token_number)
        max_user_token_number = this->user_token_number;
    }

  token_translations = xnmalloc (max_user_token_number + 1,
                                 sizeof *token_translations);

  /* Initialize all entries for literal tokens to the internal token
     number for $undefined, which represents all invalid inputs.  */
  for (int i = 0; i < max_user_token_number + 1; ++i)
    token_translations[i] = undeftoken->content->number;
  for (int i = 0; symbols_sorted[i]; ++i)
    symbol_translation (symbols_sorted[i]);
}


/*----------------------------------------------------------------.
| Assign symbol numbers, and write definition of token names into |
| FDEFINES.  Set up vectors SYMBOL_TABLE, TAGS of symbols.        |
`----------------------------------------------------------------*/

void
symbols_pack (void)
{
  symbols = xcalloc (nsyms, sizeof *symbols);
  for (int i = 0; symbols_sorted[i]; ++i)
    symbol_pack (symbols_sorted[i]);

  /* Aliases leave empty slots in symbols, so remove them.  */
  {
    int nsyms_old = nsyms;
    for (int writei = 0, readi = 0; readi < nsyms_old; readi += 1)
      {
        if (symbols[readi] == NULL)
          {
            nsyms -= 1;
            ntokens -= 1;
          }
        else
          {
            symbols[writei] = symbols[readi];
            symbols[writei]->content->number = writei;
            writei += 1;
          }
      }
  }
  symbols = xnrealloc (symbols, nsyms, sizeof *symbols);

  symbols_token_translations_init ();

  if (startsymbol->content->class == unknown_sym)
    complain (&startsymbol_loc, fatal,
              _("the start symbol %s is undefined"),
              startsymbol->tag);
  else if (startsymbol->content->class == token_sym)
    complain (&startsymbol_loc, fatal,
              _("the start symbol %s is a token"),
              startsymbol->tag);
}

/*---------------------------------.
| Initialize relation graph nodes. |
`---------------------------------*/

static void
init_prec_nodes (void)
{
  prec_nodes = xcalloc (nsyms, sizeof *prec_nodes);
  for (int i = 0; i < nsyms; ++i)
    {
      prec_nodes[i] = xmalloc (sizeof *prec_nodes[i]);
      symgraph *s = prec_nodes[i];
      s->id = i;
      s->succ = 0;
      s->pred = 0;
    }
}

/*----------------.
| Create a link.  |
`----------------*/

static symgraphlink *
symgraphlink_new (graphid id, symgraphlink *next)
{
  symgraphlink *l = xmalloc (sizeof *l);
  l->id = id;
  l->next = next;
  return l;
}


/*------------------------------------------------------------------.
| Register the second symbol of the precedence relation, and return |
| whether this relation is new.  Use only in register_precedence.   |
`------------------------------------------------------------------*/

static bool
register_precedence_second_symbol (symgraphlink **first, graphid sym)
{
  if (!*first || sym < (*first)->id)
    *first = symgraphlink_new (sym, *first);
  else
    {
      symgraphlink *slist = *first;

      while (slist->next && slist->next->id <= sym)
        slist = slist->next;

      if (slist->id == sym)
        /* Relation already present. */
        return false;

      slist->next = symgraphlink_new (sym, slist->next);
    }
  return true;
}

/*------------------------------------------------------------------.
| Register a new relation between symbols as used. The first symbol |
| has a greater precedence than the second one.                     |
`------------------------------------------------------------------*/

void
register_precedence (graphid first, graphid snd)
{
  if (!prec_nodes)
    init_prec_nodes ();
  register_precedence_second_symbol (&(prec_nodes[first]->succ), snd);
  register_precedence_second_symbol (&(prec_nodes[snd]->pred), first);
}


/*---------------------------------------.
| Deep clear a linked / adjacency list). |
`---------------------------------------*/

static void
linkedlist_free (symgraphlink *node)
{
  if (node)
    {
      while (node->next)
        {
          symgraphlink *tmp = node->next;
          free (node);
          node = tmp;
        }
      free (node);
    }
}

/*----------------------------------------------.
| Clear and destroy association tracking table. |
`----------------------------------------------*/

static void
assoc_free (void)
{
  for (int i = 0; i < nsyms; ++i)
    {
      linkedlist_free (prec_nodes[i]->pred);
      linkedlist_free (prec_nodes[i]->succ);
      free (prec_nodes[i]);
    }
  free (prec_nodes);
}

/*---------------------------------------.
| Initialize association tracking table. |
`---------------------------------------*/

static void
init_assoc (void)
{
  used_assoc = xcalloc (nsyms, sizeof *used_assoc);
  for (graphid i = 0; i < nsyms; ++i)
    used_assoc[i] = false;
}

/*------------------------------------------------------------------.
| Test if the associativity for the symbols is defined and useless. |
`------------------------------------------------------------------*/

static inline bool
is_assoc_useless (symbol *s)
{
  return s
      && s->content->assoc != undef_assoc
      && s->content->assoc != precedence_assoc
      && !used_assoc[s->content->number];
}

/*-------------------------------.
| Register a used associativity. |
`-------------------------------*/

void
register_assoc (graphid i, graphid j)
{
  if (!used_assoc)
    init_assoc ();
  used_assoc[i] = true;
  used_assoc[j] = true;
}

/*--------------------------------------------------.
| Print a warning for unused precedence relations.  |
`--------------------------------------------------*/

void
print_precedence_warnings (void)
{
  if (!prec_nodes)
    init_prec_nodes ();
  if (!used_assoc)
    init_assoc ();
  for (int i = 0; i < nsyms; ++i)
    {
      symbol *s = symbols[i];
      if (s
          && s->content->prec != 0
          && !prec_nodes[i]->pred
          && !prec_nodes[i]->succ)
        {
          if (is_assoc_useless (s))
            complain (&s->content->prec_loc, Wprecedence,
                      _("useless precedence and associativity for %s"), s->tag);
          else if (s->content->assoc == precedence_assoc)
            complain (&s->content->prec_loc, Wprecedence,
                      _("useless precedence for %s"), s->tag);
        }
      else if (is_assoc_useless (s))
        complain (&s->content->prec_loc, Wprecedence,
                  _("useless associativity for %s, use %%precedence"), s->tag);
    }
  free (used_assoc);
  assoc_free ();
}
