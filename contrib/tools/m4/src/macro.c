/* GNU m4 -- A simple macro processor

   Copyright (C) 1989-1994, 2006-2007, 2009-2013 Free Software
   Foundation, Inc.

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

/* This file contains the functions, that performs the basic argument
   parsing and macro expansion.  */

#include "m4.h"

static void expand_macro (symbol *);
static void expand_token (struct obstack *, token_type, token_data *, int);

/* Current recursion level in expand_macro ().  */
int expansion_level = 0;

/* The number of the current call of expand_macro ().  */
static int macro_call_id = 0;

/* The shared stack of collected arguments for macro calls; as each
   argument is collected, it is finished and its location stored in
   argv_stack.  Normally, this stack can be used simultaneously by
   multiple macro calls; the exception is when an outer macro has
   generated some text, then calls a nested macro, in which case the
   nested macro must use a local stack to leave the unfinished text
   alone.  Too bad obstack.h does not provide an easy way to reopen a
   finished object for further growth, but in practice this does not
   hurt us too much.  */
static struct obstack argc_stack;

/* The shared stack of pointers to collected arguments for macro
   calls.  This object is never finished; we exploit the fact that
   obstack_blank is documented to take a negative size to reduce the
   size again.  */
static struct obstack argv_stack;

/*----------------------------------------------------------------------.
| This function read all input, and expands each token, one at a time.  |
`----------------------------------------------------------------------*/

void
expand_input (void)
{
  token_type t;
  token_data td;
  int line;

  obstack_init (&argc_stack);
  obstack_init (&argv_stack);

  while ((t = next_token (&td, &line)) != TOKEN_EOF)
    expand_token ((struct obstack *) NULL, t, &td, line);

  obstack_free (&argc_stack, NULL);
  obstack_free (&argv_stack, NULL);
}


/*----------------------------------------------------------------.
| Expand one token, according to its type.  Potential macro names |
| (TOKEN_WORD) are looked up in the symbol table, to see if they  |
| have a macro definition.  If they have, they are expanded as    |
| macros, otherwise the text is just copied to the output.        |
`----------------------------------------------------------------*/

static void
expand_token (struct obstack *obs, token_type t, token_data *td, int line)
{
  symbol *sym;

  switch (t)
    { /* TOKSW */
    case TOKEN_EOF:
    case TOKEN_MACDEF:
      break;

    case TOKEN_OPEN:
    case TOKEN_COMMA:
    case TOKEN_CLOSE:
    case TOKEN_SIMPLE:
    case TOKEN_STRING:
      shipout_text (obs, TOKEN_DATA_TEXT (td), strlen (TOKEN_DATA_TEXT (td)),
                    line);
      break;

    case TOKEN_WORD:
      sym = lookup_symbol (TOKEN_DATA_TEXT (td), SYMBOL_LOOKUP);
      if (sym == NULL || SYMBOL_TYPE (sym) == TOKEN_VOID
          || (SYMBOL_TYPE (sym) == TOKEN_FUNC
              && SYMBOL_BLIND_NO_ARGS (sym)
              && peek_token () != TOKEN_OPEN))
        {
#ifdef ENABLE_CHANGEWORD
          shipout_text (obs, TOKEN_DATA_ORIG_TEXT (td),
                        strlen (TOKEN_DATA_ORIG_TEXT (td)), line);
#else
          shipout_text (obs, TOKEN_DATA_TEXT (td),
                        strlen (TOKEN_DATA_TEXT (td)), line);
#endif
        }
      else
        expand_macro (sym);
      break;

    default:
      M4ERROR ((warning_status, 0,
                "INTERNAL ERROR: bad token type in expand_token ()"));
      abort ();
    }
}


/*-------------------------------------------------------------------.
| This function parses one argument to a macro call.  It expects the |
| first left parenthesis, or the separating comma, to have been read |
| by the caller.  It skips leading whitespace, and reads and expands |
| tokens, until it finds a comma or an right parenthesis at the same |
| level of parentheses.  It returns a flag indicating whether the    |
| argument read is the last for the active macro call.  The argument |
| is built on the obstack OBS, indirectly through expand_token ().   |
`-------------------------------------------------------------------*/

static bool
expand_argument (struct obstack *obs, token_data *argp)
{
  token_type t;
  token_data td;
  char *text;
  int paren_level;
  const char *file = current_file;
  int line = current_line;

  TOKEN_DATA_TYPE (argp) = TOKEN_VOID;

  /* Skip leading white space.  */
  do
    {
      t = next_token (&td, NULL);
    }
  while (t == TOKEN_SIMPLE && isspace (to_uchar (*TOKEN_DATA_TEXT (&td))));

  paren_level = 0;

  while (1)
    {

      switch (t)
        { /* TOKSW */
        case TOKEN_COMMA:
        case TOKEN_CLOSE:
          if (paren_level == 0)
            {
              /* The argument MUST be finished, whether we want it or not.  */
              obstack_1grow (obs, '\0');
              text = (char *) obstack_finish (obs);

              if (TOKEN_DATA_TYPE (argp) == TOKEN_VOID)
                {
                  TOKEN_DATA_TYPE (argp) = TOKEN_TEXT;
                  TOKEN_DATA_TEXT (argp) = text;
                }
              return t == TOKEN_COMMA;
            }
          /* fallthru */
        case TOKEN_OPEN:
        case TOKEN_SIMPLE:
          text = TOKEN_DATA_TEXT (&td);

          if (*text == '(')
            paren_level++;
          else if (*text == ')')
            paren_level--;
          expand_token (obs, t, &td, line);
          break;

        case TOKEN_EOF:
          /* current_file changed to "" if we see TOKEN_EOF, use the
             previous value we stored earlier.  */
          M4ERROR_AT_LINE ((EXIT_FAILURE, 0, file, line,
                            "ERROR: end of file in argument list"));
          break;

        case TOKEN_WORD:
        case TOKEN_STRING:
          expand_token (obs, t, &td, line);
          break;

        case TOKEN_MACDEF:
          if (obstack_object_size (obs) == 0)
            {
              TOKEN_DATA_TYPE (argp) = TOKEN_FUNC;
              TOKEN_DATA_FUNC (argp) = TOKEN_DATA_FUNC (&td);
            }
          break;

        default:
          M4ERROR ((warning_status, 0,
                    "INTERNAL ERROR: bad token type in expand_argument ()"));
          abort ();
        }

      t = next_token (&td, NULL);
    }
}

/*-------------------------------------------------------------.
| Collect all the arguments to a call of the macro SYM.  The   |
| arguments are stored on the obstack ARGUMENTS and a table of |
| pointers to the arguments on the obstack ARGPTR.             |
`-------------------------------------------------------------*/

static void
collect_arguments (symbol *sym, struct obstack *argptr,
                   struct obstack *arguments)
{
  token_data td;
  token_data *tdp;
  bool more_args;
  bool groks_macro_args = SYMBOL_MACRO_ARGS (sym);

  TOKEN_DATA_TYPE (&td) = TOKEN_TEXT;
  TOKEN_DATA_TEXT (&td) = SYMBOL_NAME (sym);
  tdp = (token_data *) obstack_copy (arguments, &td, sizeof td);
  obstack_ptr_grow (argptr, tdp);

  if (peek_token () == TOKEN_OPEN)
    {
      next_token (&td, NULL); /* gobble parenthesis */
      do
        {
          more_args = expand_argument (arguments, &td);

          if (!groks_macro_args && TOKEN_DATA_TYPE (&td) == TOKEN_FUNC)
            {
              TOKEN_DATA_TYPE (&td) = TOKEN_TEXT;
              TOKEN_DATA_TEXT (&td) = (char *) "";
            }
          tdp = (token_data *) obstack_copy (arguments, &td, sizeof td);
          obstack_ptr_grow (argptr, tdp);
        }
      while (more_args);
    }
}


/*-------------------------------------------------------------------.
| The actual call of a macro is handled by call_macro ().            |
| call_macro () is passed a symbol SYM, whose type is used to call   |
| either a builtin function, or the user macro expansion function    |
| expand_user_macro () (lives in builtin.c).  There are ARGC         |
| arguments to the call, stored in the ARGV table.  The expansion is |
| left on the obstack EXPANSION.  Macro tracing is also handled      |
| here.                                                              |
`-------------------------------------------------------------------*/

void
call_macro (symbol *sym, int argc, token_data **argv,
                 struct obstack *expansion)
{
  switch (SYMBOL_TYPE (sym))
    {
    case TOKEN_FUNC:
      (*SYMBOL_FUNC (sym)) (expansion, argc, argv);
      break;

    case TOKEN_TEXT:
      expand_user_macro (expansion, sym, argc, argv);
      break;

    case TOKEN_VOID:
    default:
      M4ERROR ((warning_status, 0,
                "INTERNAL ERROR: bad symbol type in call_macro ()"));
      abort ();
    }
}

/*-------------------------------------------------------------------.
| The macro expansion is handled by expand_macro ().  It parses the  |
| arguments, using collect_arguments (), and builds a table of       |
| pointers to the arguments.  The arguments themselves are stored on |
| a local obstack.  Expand_macro () uses call_macro () to do the     |
| call of the macro.                                                 |
|                                                                    |
| Expand_macro () is potentially recursive, since it calls           |
| expand_argument (), which might call expand_token (), which might  |
| call expand_macro ().                                              |
`-------------------------------------------------------------------*/

static void
expand_macro (symbol *sym)
{
  struct obstack arguments;     /* Alternate obstack if argc_stack is busy.  */
  unsigned argv_base;           /* Size of argv_stack on entry.  */
  bool use_argc_stack = true;   /* Whether argc_stack is safe.  */
  token_data **argv;
  int argc;
  struct obstack *expansion;
  const char *expanded;
  bool traced;
  int my_call_id;

  /* Report errors at the location where the open parenthesis (if any)
     was found, but after expansion, restore global state back to the
     location of the close parenthesis.  This is safe since we
     guarantee that macro expansion does not alter the state of
     current_file/current_line (dnl, include, and sinclude are special
     cased in the input engine to ensure this fact).  */
  const char *loc_open_file = current_file;
  int loc_open_line = current_line;
  const char *loc_close_file;
  int loc_close_line;

  SYMBOL_PENDING_EXPANSIONS (sym)++;
  expansion_level++;
  if (nesting_limit > 0 && expansion_level > nesting_limit)
    M4ERROR ((EXIT_FAILURE, 0,
              "recursion limit of %d exceeded, use -L<N> to change it",
              nesting_limit));

  macro_call_id++;
  my_call_id = macro_call_id;

  traced = (debug_level & DEBUG_TRACE_ALL) || SYMBOL_TRACED (sym);

  argv_base = obstack_object_size (&argv_stack);
  if (obstack_object_size (&argc_stack) > 0)
    {
      /* We cannot use argc_stack if this is a nested invocation, and an
         outer invocation has an unfinished argument being
         collected.  */
      obstack_init (&arguments);
      use_argc_stack = false;
    }

  if (traced && (debug_level & DEBUG_TRACE_CALL))
    trace_prepre (SYMBOL_NAME (sym), my_call_id);

  collect_arguments (sym, &argv_stack,
                     use_argc_stack ? &argc_stack : &arguments);

  argc = ((obstack_object_size (&argv_stack) - argv_base)
          / sizeof (token_data *));
  argv = (token_data **) ((char *) obstack_base (&argv_stack) + argv_base);

  loc_close_file = current_file;
  loc_close_line = current_line;
  current_file = loc_open_file;
  current_line = loc_open_line;

  if (traced)
    trace_pre (SYMBOL_NAME (sym), my_call_id, argc, argv);

  expansion = push_string_init ();
  call_macro (sym, argc, argv, expansion);
  expanded = push_string_finish ();

  if (traced)
    trace_post (SYMBOL_NAME (sym), my_call_id, argc, expanded);

  current_file = loc_close_file;
  current_line = loc_close_line;

  --expansion_level;
  --SYMBOL_PENDING_EXPANSIONS (sym);

  if (SYMBOL_DELETED (sym))
    free_symbol (sym);

  if (use_argc_stack)
    obstack_free (&argc_stack, argv[0]);
  else
    obstack_free (&arguments, NULL);
  obstack_blank (&argv_stack, -argc * sizeof (token_data *));
}
