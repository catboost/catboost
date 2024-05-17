/* GNU m4 -- A simple macro processor

   Copyright (C) 1989-1994, 2004-2013 Free Software Foundation, Inc.

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

/* Handling of different input sources, and lexical analysis.  */

#include "m4.h"

#include "memchr2.h"

/* Unread input can be either files, that should be read (eg. included
   files), strings, which should be rescanned (eg. macro expansion text),
   or quoted macro definitions (as returned by the builtin "defn").
   Unread input are organised in a stack, implemented with an obstack.
   Each input source is described by a "struct input_block".  The obstack
   is "current_input".  The top of the input stack is "isp".

   The macro "m4wrap" places the text to be saved on another input
   stack, on the obstack "wrapup_stack", whose top is "wsp".  When EOF
   is seen on normal input (eg, when "current_input" is empty), input is
   switched over to "wrapup_stack", and the original "current_input" is
   freed.  A new stack is allocated for "wrapup_stack", which will
   accept any text produced by calls to "m4wrap" from within the
   wrapped text.  This process of shuffling "wrapup_stack" to
   "current_input" can continue indefinitely, even generating infinite
   loops (e.g. "define(`f',`m4wrap(`f')')f"), without memory leaks.

   Pushing new input on the input stack is done by push_file (),
   push_string (), push_wrapup () (for wrapup text), and push_macro ()
   (for macro definitions).  Because macro expansion needs direct access
   to the current input obstack (for optimisation), push_string () are
   split in two functions, push_string_init (), which returns a pointer
   to the current input stack, and push_string_finish (), which return a
   pointer to the final text.  The input_block *next is used to manage
   the coordination between the different push routines.

   The current file and line number are stored in two global
   variables, for use by the error handling functions in m4.c.  Macro
   expansion wants to report the line where a macro name was detected,
   rather than where it finished collecting arguments.  This also
   applies to text resulting from macro expansions.  So each input
   block maintains its own notion of the current file and line, and
   swapping between input blocks updates the global variables
   accordingly.  */

#ifdef ENABLE_CHANGEWORD
#include <contrib/tools/bison/gnulib/src/regex.h>
#endif

enum input_type
{
  INPUT_STRING,         /* String resulting from macro expansion.  */
  INPUT_FILE,           /* File from command line or include.  */
  INPUT_MACRO           /* Builtin resulting from defn.  */
};

typedef enum input_type input_type;

struct input_block
{
  struct input_block *prev;     /* previous input_block on the input stack */
  input_type type;              /* see enum values */
  const char *file;             /* file where this input is from */
  int line;                     /* line where this input is from */
  union
    {
      struct
        {
          char *string;         /* remaining string value */
          char *end;            /* terminating NUL of string */
        }
        u_s;    /* INPUT_STRING */
      struct
        {
          FILE *fp;                  /* input file handle */
          bool_bitfield end : 1;     /* true if peek has seen EOF */
          bool_bitfield close : 1;   /* true if we should close file on pop */
          bool_bitfield advance : 1; /* track previous start_of_input_line */
        }
        u_f;    /* INPUT_FILE */
      builtin_func *func;       /* pointer to macro's function */
    }
  u;
};

typedef struct input_block input_block;


/* Current input file name.  */
const char *current_file;

/* Current input line number.  */
int current_line;

/* Obstack for storing individual tokens.  */
static struct obstack token_stack;

/* Obstack for storing file names.  */
static struct obstack file_names;

/* Wrapup input stack.  */
static struct obstack *wrapup_stack;

/* Current stack, from input or wrapup.  */
static struct obstack *current_input;

/* Bottom of token_stack, for obstack_free.  */
static void *token_bottom;

/* Pointer to top of current_input.  */
static input_block *isp;

/* Pointer to top of wrapup_stack.  */
static input_block *wsp;

/* Aux. for handling split push_string ().  */
static input_block *next;

/* Flag for next_char () to increment current_line.  */
static bool start_of_input_line;

/* Flag for next_char () to recognize change in input block.  */
static bool input_change;

#define CHAR_EOF        256     /* character return on EOF */
#define CHAR_MACRO      257     /* character return for MACRO token */

/* Quote chars.  */
STRING rquote;
STRING lquote;

/* Comment chars.  */
STRING bcomm;
STRING ecomm;

#ifdef ENABLE_CHANGEWORD

# define DEFAULT_WORD_REGEXP "[_a-zA-Z][_a-zA-Z0-9]*"

static struct re_pattern_buffer word_regexp;
static int default_word_regexp;
static struct re_registers regs;

#else /* ! ENABLE_CHANGEWORD */
# define default_word_regexp 1
#endif /* ! ENABLE_CHANGEWORD */

#ifdef DEBUG_INPUT
static const char *token_type_string (token_type);
#endif


/*-------------------------------------------------------------------.
| push_file () pushes an input file on the input stack, saving the   |
| current file name and line number.  If next is non-NULL, this push |
| invalidates a call to push_string_init (), whose storage is        |
| consequently released.  If CLOSE_WHEN_DONE, then close FP after    |
| EOF is detected.                                                   |
`-------------------------------------------------------------------*/

void
push_file (FILE *fp, const char *title, bool close_when_done)
{
  input_block *i;

  if (next != NULL)
    {
      obstack_free (current_input, next);
      next = NULL;
    }

  if (debug_level & DEBUG_TRACE_INPUT)
    DEBUG_MESSAGE1 ("input read from %s", title);

  i = (input_block *) obstack_alloc (current_input,
                                     sizeof (struct input_block));
  i->type = INPUT_FILE;
  i->file = (char *) obstack_copy0 (&file_names, title, strlen (title));
  i->line = 1;
  input_change = true;

  i->u.u_f.fp = fp;
  i->u.u_f.end = false;
  i->u.u_f.close = close_when_done;
  i->u.u_f.advance = start_of_input_line;
  output_current_line = -1;

  i->prev = isp;
  isp = i;
}

/*---------------------------------------------------------------.
| push_macro () pushes a builtin macro's definition on the input |
| stack.  If next is non-NULL, this push invalidates a call to   |
| push_string_init (), whose storage is consequently released.   |
`---------------------------------------------------------------*/

void
push_macro (builtin_func *func)
{
  input_block *i;

  if (next != NULL)
    {
      obstack_free (current_input, next);
      next = NULL;
    }

  i = (input_block *) obstack_alloc (current_input,
                                     sizeof (struct input_block));
  i->type = INPUT_MACRO;
  i->file = current_file;
  i->line = current_line;
  input_change = true;

  i->u.func = func;
  i->prev = isp;
  isp = i;
}

/*------------------------------------------------------------------.
| First half of push_string ().  The pointer next points to the new |
| input_block.                                                      |
`------------------------------------------------------------------*/

struct obstack *
push_string_init (void)
{
  if (next != NULL)
    {
      M4ERROR ((warning_status, 0,
                "INTERNAL ERROR: recursive push_string!"));
      abort ();
    }

  next = (input_block *) obstack_alloc (current_input,
                                        sizeof (struct input_block));
  next->type = INPUT_STRING;
  next->file = current_file;
  next->line = current_line;

  return current_input;
}

/*-------------------------------------------------------------------.
| Last half of push_string ().  If next is now NULL, a call to       |
| push_file () has invalidated the previous call to push_string_init |
| (), so we just give up.  If the new object is void, we do not push |
| it.  The function push_string_finish () returns a pointer to the   |
| finished object.  This pointer is only for temporary use, since    |
| reading the next token might release the memory used for the       |
| object.                                                            |
`-------------------------------------------------------------------*/

const char *
push_string_finish (void)
{
  const char *ret = NULL;

  if (next == NULL)
    return NULL;

  if (obstack_object_size (current_input) > 0)
    {
      size_t len = obstack_object_size (current_input);
      obstack_1grow (current_input, '\0');
      next->u.u_s.string = (char *) obstack_finish (current_input);
      next->u.u_s.end = next->u.u_s.string + len;
      next->prev = isp;
      isp = next;
      ret = isp->u.u_s.string; /* for immediate use only */
      input_change = true;
    }
  else
    obstack_free (current_input, next); /* people might leave garbage on it. */
  next = NULL;
  return ret;
}

/*------------------------------------------------------------------.
| The function push_wrapup () pushes a string on the wrapup stack.  |
| When the normal input stack gets empty, the wrapup stack will     |
| become the input stack, and push_string () and push_file () will  |
| operate on wrapup_stack.  Push_wrapup should be done as           |
| push_string (), but this will suffice, as long as arguments to    |
| m4_m4wrap () are moderate in size.                                |
`------------------------------------------------------------------*/

void
push_wrapup (const char *s)
{
  size_t len = strlen (s);
  input_block *i;
  i = (input_block *) obstack_alloc (wrapup_stack,
                                     sizeof (struct input_block));
  i->prev = wsp;
  i->type = INPUT_STRING;
  i->file = current_file;
  i->line = current_line;
  i->u.u_s.string = (char *) obstack_copy0 (wrapup_stack, s, len);
  i->u.u_s.end = i->u.u_s.string + len;
  wsp = i;
}


/*-------------------------------------------------------------------.
| The function pop_input () pops one level of input sources.  If the |
| popped input_block is a file, current_file and current_line are    |
| reset to the saved values before the memory for the input_block is |
| released.                                                          |
`-------------------------------------------------------------------*/

static void
pop_input (void)
{
  input_block *tmp = isp->prev;

  switch (isp->type)
    {
    case INPUT_STRING:
    case INPUT_MACRO:
      break;

    case INPUT_FILE:
      if (debug_level & DEBUG_TRACE_INPUT)
        {
          if (tmp)
            DEBUG_MESSAGE2 ("input reverted to %s, line %d",
                            tmp->file, tmp->line);
          else
            DEBUG_MESSAGE ("input exhausted");
        }

      if (ferror (isp->u.u_f.fp))
        {
          M4ERROR ((warning_status, 0, "read error"));
          if (isp->u.u_f.close)
            fclose (isp->u.u_f.fp);
          retcode = EXIT_FAILURE;
        }
      else if (isp->u.u_f.close && fclose (isp->u.u_f.fp) == EOF)
        {
          M4ERROR ((warning_status, errno, "error reading file"));
          retcode = EXIT_FAILURE;
        }
      start_of_input_line = isp->u.u_f.advance;
      output_current_line = -1;
      break;

    default:
      M4ERROR ((warning_status, 0,
                "INTERNAL ERROR: input stack botch in pop_input ()"));
      abort ();
    }
  obstack_free (current_input, isp);
  next = NULL; /* might be set in push_string_init () */

  isp = tmp;
  input_change = true;
}

/*-------------------------------------------------------------------.
| To switch input over to the wrapup stack, main calls pop_wrapup    |
| ().  Since wrapup text can install new wrapup text, pop_wrapup ()  |
| returns false when there is no wrapup text on the stack, and true  |
| otherwise.                                                         |
`-------------------------------------------------------------------*/

bool
pop_wrapup (void)
{
  next = NULL;
  obstack_free (current_input, NULL);
  free (current_input);

  if (wsp == NULL)
    {
      /* End of the program.  Free all memory even though we are about
         to exit, since it makes leak detection easier.  */
      obstack_free (&token_stack, NULL);
      obstack_free (&file_names, NULL);
      obstack_free (wrapup_stack, NULL);
      free (wrapup_stack);
#ifdef ENABLE_CHANGEWORD
      regfree (&word_regexp);
#endif /* ENABLE_CHANGEWORD */
      return false;
    }

  current_input = wrapup_stack;
  wrapup_stack = (struct obstack *) xmalloc (sizeof (struct obstack));
  obstack_init (wrapup_stack);

  isp = wsp;
  wsp = NULL;
  input_change = true;

  return true;
}

/*-------------------------------------------------------------------.
| When a MACRO token is seen, next_token () uses init_macro_token () |
| to retrieve the value of the function pointer.                     |
`-------------------------------------------------------------------*/

static void
init_macro_token (token_data *td)
{
  if (isp->type != INPUT_MACRO)
    {
      M4ERROR ((warning_status, 0,
                "INTERNAL ERROR: bad call to init_macro_token ()"));
      abort ();
    }

  TOKEN_DATA_TYPE (td) = TOKEN_FUNC;
  TOKEN_DATA_FUNC (td) = isp->u.func;
}


/*-----------------------------------------------------------------.
| Low level input is done a character at a time.  The function     |
| peek_input () is used to look at the next character in the input |
| stream.  At any given time, it reads from the input_block on the |
| top of the current input stack.                                  |
`-----------------------------------------------------------------*/

static int
peek_input (void)
{
  int ch;
  input_block *block = isp;

  while (1)
    {
      if (block == NULL)
        return CHAR_EOF;

      switch (block->type)
        {
        case INPUT_STRING:
          ch = to_uchar (block->u.u_s.string[0]);
          if (ch != '\0')
            return ch;
          break;

        case INPUT_FILE:
          ch = getc (block->u.u_f.fp);
          if (ch != EOF)
            {
              ungetc (ch, block->u.u_f.fp);
              return ch;
            }
          block->u.u_f.end = true;
          break;

        case INPUT_MACRO:
          return CHAR_MACRO;

        default:
          M4ERROR ((warning_status, 0,
                    "INTERNAL ERROR: input stack botch in peek_input ()"));
          abort ();
        }
      block = block->prev;
    }
}

/*-------------------------------------------------------------------.
| The function next_char () is used to read and advance the input to |
| the next character.  It also manages line numbers for error        |
| messages, so they do not get wrong, due to lookahead.  The token   |
| consisting of a newline alone is taken as belonging to the line it |
| ends, and the current line number is not incremented until the     |
| next character is read.  99.9% of all calls will read from a       |
| string, so factor that out into a macro for speed.                 |
`-------------------------------------------------------------------*/

#define next_char() \
  (isp && isp->type == INPUT_STRING && isp->u.u_s.string[0]     \
   && !input_change                                             \
   ? to_uchar (*isp->u.u_s.string++)                            \
   : next_char_1 ())

static int
next_char_1 (void)
{
  int ch;

  while (1)
    {
      if (isp == NULL)
        {
          current_file = "";
          current_line = 0;
          return CHAR_EOF;
        }

      if (input_change)
        {
          current_file = isp->file;
          current_line = isp->line;
          input_change = false;
        }

      switch (isp->type)
        {
        case INPUT_STRING:
          ch = to_uchar (*isp->u.u_s.string++);
          if (ch != '\0')
            return ch;
          break;

        case INPUT_FILE:
          if (start_of_input_line)
            {
              start_of_input_line = false;
              current_line = ++isp->line;
            }

          /* If stdin is a terminal, calling getc after peek_input
             already called it would make the user have to hit ^D
             twice to quit.  */
          ch = isp->u.u_f.end ? EOF : getc (isp->u.u_f.fp);
          if (ch != EOF)
            {
              if (ch == '\n')
                start_of_input_line = true;
              return ch;
            }
          break;

        case INPUT_MACRO:
          pop_input (); /* INPUT_MACRO input sources has only one token */
          return CHAR_MACRO;

        default:
          M4ERROR ((warning_status, 0,
                    "INTERNAL ERROR: input stack botch in next_char ()"));
          abort ();
        }

      /* End of input source --- pop one level.  */
      pop_input ();
    }
}

/*-------------------------------------------------------------------.
| skip_line () simply discards all immediately following characters, |
| upto the first newline.  It is only used from m4_dnl ().           |
`-------------------------------------------------------------------*/

void
skip_line (void)
{
  int ch;
  const char *file = current_file;
  int line = current_line;

  while ((ch = next_char ()) != CHAR_EOF && ch != '\n')
    ;
  if (ch == CHAR_EOF)
    /* current_file changed to "" if we see CHAR_EOF, use the
       previous value we stored earlier.  */
    M4ERROR_AT_LINE ((warning_status, 0, file, line,
                      "Warning: end of file treated as newline"));
  /* On the rare occasion that dnl crosses include file boundaries
     (either the input file did not end in a newline, or changeword
     was used), calling next_char can update current_file and
     current_line, and that update will be undone as we return to
     expand_macro.  This informs next_char to fix things again.  */
  if (file != current_file || line != current_line)
    input_change = true;
}


/*------------------------------------------------------------------.
| This function is for matching a string against a prefix of the    |
| input stream.  If the string matches the input and consume is     |
| true, the input is discarded; otherwise any characters read are   |
| pushed back again.  The function is used only when multicharacter |
| quotes or comment delimiters are used.                            |
`------------------------------------------------------------------*/

static bool
match_input (const char *s, bool consume)
{
  int n;                        /* number of characters matched */
  int ch;                       /* input character */
  const char *t;
  bool result = false;

  ch = peek_input ();
  if (ch != to_uchar (*s))
    return false;                       /* fail */

  if (s[1] == '\0')
    {
      if (consume)
        next_char ();
      return true;                      /* short match */
    }

  next_char ();
  for (n = 1, t = s++; peek_input () == to_uchar (*s++); )
    {
      next_char ();
      n++;
      if (*s == '\0')           /* long match */
        {
          if (consume)
            return true;
          result = true;
          break;
        }
    }

  /* Failed or shouldn't consume, push back input.  */
  {
    struct obstack *h = push_string_init ();

    /* `obstack_grow' may be macro evaluating its arg 1 several times. */
    obstack_grow (h, t, n);
  }
  push_string_finish ();
  return result;
}

/*--------------------------------------------------------------------.
| The macro MATCH() is used to match a string S against the input.    |
| The first character is handled inline, for speed.  Hopefully, this  |
| will not hurt efficiency too much when single character quotes and  |
| comment delimiters are used.  If CONSUME, then CH is the result of  |
| next_char, and a successful match will discard the matched string.  |
| Otherwise, CH is the result of peek_char, and the input stream is   |
| effectively unchanged.                                              |
`--------------------------------------------------------------------*/

#define MATCH(ch, s, consume)                                           \
  (to_uchar ((s)[0]) == (ch)                                            \
   && (ch) != '\0'                                                      \
   && ((s)[1] == '\0' || (match_input ((s) + (consume), consume))))


/*--------------------------------------------------------.
| Initialize input stacks, and quote/comment characters.  |
`--------------------------------------------------------*/

void
input_init (void)
{
  current_file = "";
  current_line = 0;

  current_input = (struct obstack *) xmalloc (sizeof (struct obstack));
  obstack_init (current_input);
  wrapup_stack = (struct obstack *) xmalloc (sizeof (struct obstack));
  obstack_init (wrapup_stack);

  obstack_init (&file_names);

  /* Allocate an object in the current chunk, so that obstack_free
     will always work even if the first token parsed spills to a new
     chunk.  */
  obstack_init (&token_stack);
  obstack_alloc (&token_stack, 1);
  token_bottom = obstack_base (&token_stack);

  isp = NULL;
  wsp = NULL;
  next = NULL;

  start_of_input_line = false;

  lquote.string = xstrdup (DEF_LQUOTE);
  lquote.length = strlen (lquote.string);
  rquote.string = xstrdup (DEF_RQUOTE);
  rquote.length = strlen (rquote.string);
  bcomm.string = xstrdup (DEF_BCOMM);
  bcomm.length = strlen (bcomm.string);
  ecomm.string = xstrdup (DEF_ECOMM);
  ecomm.length = strlen (ecomm.string);

#ifdef ENABLE_CHANGEWORD
  set_word_regexp (user_word_regexp);
#endif
}


/*------------------------------------------------------------------.
| Functions for setting quotes and comment delimiters.  Used by     |
| m4_changecom () and m4_changequote ().  Pass NULL if the argument |
| was not present, to distinguish from an explicit empty string.    |
`------------------------------------------------------------------*/

void
set_quotes (const char *lq, const char *rq)
{
  free (lquote.string);
  free (rquote.string);

  /* POSIX states that with 0 arguments, the default quotes are used.
     POSIX XCU ERN 112 states that behavior is implementation-defined
     if there was only one argument, or if there is an empty string in
     either position when there are two arguments.  We allow an empty
     left quote to disable quoting, but a non-empty left quote will
     always create a non-empty right quote.  See the texinfo for what
     some other implementations do.  */
  if (!lq)
    {
      lq = DEF_LQUOTE;
      rq = DEF_RQUOTE;
    }
  else if (!rq || (*lq && !*rq))
    rq = DEF_RQUOTE;

  lquote.string = xstrdup (lq);
  lquote.length = strlen (lquote.string);
  rquote.string = xstrdup (rq);
  rquote.length = strlen (rquote.string);
}

void
set_comment (const char *bc, const char *ec)
{
  free (bcomm.string);
  free (ecomm.string);

  /* POSIX requires no arguments to disable comments.  It requires
     empty arguments to be used as-is, but this is counter to
     traditional behavior, because a non-null begin and null end makes
     it impossible to end a comment.  An aardvark has been filed:
     http://www.opengroup.org/austin/mailarchives/ag-review/msg02168.html
     This implementation assumes the aardvark will be approved.  See
     the texinfo for what some other implementations do.  */
  if (!bc)
    bc = ec = "";
  else if (!ec || (*bc && !*ec))
    ec = DEF_ECOMM;

  bcomm.string = xstrdup (bc);
  bcomm.length = strlen (bcomm.string);
  ecomm.string = xstrdup (ec);
  ecomm.length = strlen (ecomm.string);
}

#ifdef ENABLE_CHANGEWORD

void
set_word_regexp (const char *regexp)
{
  const char *msg;
  struct re_pattern_buffer new_word_regexp;

  if (!*regexp || STREQ (regexp, DEFAULT_WORD_REGEXP))
    {
      default_word_regexp = true;
      return;
    }

  /* Dry run to see whether the new expression is compilable.  */
  init_pattern_buffer (&new_word_regexp, NULL);
  msg = re_compile_pattern (regexp, strlen (regexp), &new_word_regexp);
  regfree (&new_word_regexp);

  if (msg != NULL)
    {
      M4ERROR ((warning_status, 0,
                "bad regular expression `%s': %s", regexp, msg));
      return;
    }

  /* If compilation worked, retry using the word_regexp struct.  We
     can't rely on struct assigns working, so redo the compilation.
     The fastmap can be reused between compilations, and will be freed
     by the final regfree.  */
  if (!word_regexp.fastmap)
    word_regexp.fastmap = xcharalloc (UCHAR_MAX + 1);
  msg = re_compile_pattern (regexp, strlen (regexp), &word_regexp);
  assert (!msg);
  re_set_registers (&word_regexp, &regs, regs.num_regs, regs.start, regs.end);
  if (re_compile_fastmap (&word_regexp))
    assert (false);

  default_word_regexp = false;
}

#endif /* ENABLE_CHANGEWORD */


/*--------------------------------------------------------------------.
| Parse and return a single token from the input stream.  A token     |
| can either be TOKEN_EOF, if the input_stack is empty; it can be     |
| TOKEN_STRING for a quoted string; TOKEN_WORD for something that is  |
| a potential macro name; and TOKEN_SIMPLE for any single character   |
| that is not a part of any of the previous types.  If LINE is not    |
| NULL, set *LINE to the line where the token starts.                 |
|                                                                     |
| Next_token () return the token type, and passes back a pointer to   |
| the token data through TD.  The token text is collected on the      |
| obstack token_stack, which never contains more than one token text  |
| at a time.  The storage pointed to by the fields in TD is           |
| therefore subject to change the next time next_token () is called.  |
`--------------------------------------------------------------------*/

token_type
next_token (token_data *td, int *line)
{
  int ch;
  int quote_level;
  token_type type;
#ifdef ENABLE_CHANGEWORD
  int startpos;
  char *orig_text = NULL;
#endif
  const char *file;
  int dummy;

  obstack_free (&token_stack, token_bottom);
  if (!line)
    line = &dummy;

 /* Can't consume character until after CHAR_MACRO is handled.  */
  ch = peek_input ();
  if (ch == CHAR_EOF)
    {
#ifdef DEBUG_INPUT
      xfprintf (stderr, "next_token -> EOF\n");
#endif
      next_char ();
      return TOKEN_EOF;
    }
  if (ch == CHAR_MACRO)
    {
      init_macro_token (td);
      next_char ();
#ifdef DEBUG_INPUT
      xfprintf (stderr, "next_token -> MACDEF (%s)\n",
                find_builtin_by_addr (TOKEN_DATA_FUNC (td))->name);
#endif
      return TOKEN_MACDEF;
    }

  next_char (); /* Consume character we already peeked at.  */
  file = current_file;
  *line = current_line;
  if (MATCH (ch, bcomm.string, true))
    {
      obstack_grow (&token_stack, bcomm.string, bcomm.length);
      while ((ch = next_char ()) != CHAR_EOF
             && !MATCH (ch, ecomm.string, true))
        obstack_1grow (&token_stack, ch);
      if (ch != CHAR_EOF)
        obstack_grow (&token_stack, ecomm.string, ecomm.length);
      else
        /* current_file changed to "" if we see CHAR_EOF, use the
           previous value we stored earlier.  */
        M4ERROR_AT_LINE ((EXIT_FAILURE, 0, file, *line,
                          "ERROR: end of file in comment"));

      type = TOKEN_STRING;
    }
  else if (default_word_regexp && (isalpha (ch) || ch == '_'))
    {
      obstack_1grow (&token_stack, ch);
      while ((ch = peek_input ()) != CHAR_EOF && (isalnum (ch) || ch == '_'))
        {
          obstack_1grow (&token_stack, ch);
          next_char ();
        }
      type = TOKEN_WORD;
    }

#ifdef ENABLE_CHANGEWORD

  else if (!default_word_regexp && word_regexp.fastmap[ch])
    {
      obstack_1grow (&token_stack, ch);
      while (1)
        {
          ch = peek_input ();
          if (ch == CHAR_EOF)
            break;
          obstack_1grow (&token_stack, ch);
          startpos = re_search (&word_regexp,
                                (char *) obstack_base (&token_stack),
                                obstack_object_size (&token_stack), 0, 0,
                                &regs);
          if (startpos ||
              regs.end [0] != (regoff_t) obstack_object_size (&token_stack))
            {
              *(((char *) obstack_base (&token_stack)
                 + obstack_object_size (&token_stack)) - 1) = '\0';
              break;
            }
          next_char ();
        }

      obstack_1grow (&token_stack, '\0');
      orig_text = (char *) obstack_finish (&token_stack);

      if (regs.start[1] != -1)
        obstack_grow (&token_stack,orig_text + regs.start[1],
                      regs.end[1] - regs.start[1]);
      else
        obstack_grow (&token_stack, orig_text,regs.end[0]);

      type = TOKEN_WORD;
    }

#endif /* ENABLE_CHANGEWORD */

  else if (!MATCH (ch, lquote.string, true))
    {
      switch (ch)
        {
        case '(':
          type = TOKEN_OPEN;
          break;
        case ',':
          type = TOKEN_COMMA;
          break;
        case ')':
          type = TOKEN_CLOSE;
          break;
        default:
          type = TOKEN_SIMPLE;
          break;
        }
      obstack_1grow (&token_stack, ch);
    }
  else
    {
      bool fast = lquote.length == 1 && rquote.length == 1;
      quote_level = 1;
      while (1)
        {
          /* Try scanning a buffer first.  */
          const char *buffer = (isp && isp->type == INPUT_STRING
                                ? isp->u.u_s.string : NULL);
          if (buffer && *buffer)
            {
              size_t len = isp->u.u_s.end - buffer;
              const char *p = buffer;
              do
                {
                  p = (char *) memchr2 (p, *lquote.string, *rquote.string,
                                        buffer + len - p);
                }
              while (p && fast && (*p++ == *rquote.string
                                   ? --quote_level : ++quote_level));
              if (p)
                {
                  if (fast)
                    {
                      assert (!quote_level);
                      obstack_grow (&token_stack, buffer, p - buffer - 1);
                      isp->u.u_s.string += p - buffer;
                      break;
                    }
                  obstack_grow (&token_stack, buffer, p - buffer);
                  ch = to_uchar (*p);
                  isp->u.u_s.string += p - buffer + 1;
                }
              else
                {
                  obstack_grow (&token_stack, buffer, len);
                  isp->u.u_s.string += len;
                  continue;
                }
            }
          /* Fall back to a byte.  */
          else
            ch = next_char ();
          if (ch == CHAR_EOF)
            /* current_file changed to "" if we see CHAR_EOF, use
               the previous value we stored earlier.  */
            M4ERROR_AT_LINE ((EXIT_FAILURE, 0, file, *line,
                              "ERROR: end of file in string"));

          if (MATCH (ch, rquote.string, true))
            {
              if (--quote_level == 0)
                break;
              obstack_grow (&token_stack, rquote.string, rquote.length);
            }
          else if (MATCH (ch, lquote.string, true))
            {
              quote_level++;
              obstack_grow (&token_stack, lquote.string, lquote.length);
            }
          else
            obstack_1grow (&token_stack, ch);
        }
      type = TOKEN_STRING;
    }

  obstack_1grow (&token_stack, '\0');

  TOKEN_DATA_TYPE (td) = TOKEN_TEXT;
  TOKEN_DATA_TEXT (td) = (char *) obstack_finish (&token_stack);
#ifdef ENABLE_CHANGEWORD
  if (orig_text == NULL)
    orig_text = TOKEN_DATA_TEXT (td);
  TOKEN_DATA_ORIG_TEXT (td) = orig_text;
#endif
#ifdef DEBUG_INPUT
  xfprintf (stderr, "next_token -> %s (%s)\n",
            token_type_string (type), TOKEN_DATA_TEXT (td));
#endif
  return type;
}

/*-----------------------------------------------.
| Peek at the next token from the input stream.  |
`-----------------------------------------------*/

token_type
peek_token (void)
{
  token_type result;
  int ch = peek_input ();

  if (ch == CHAR_EOF)
    {
      result = TOKEN_EOF;
    }
  else if (ch == CHAR_MACRO)
    {
      result = TOKEN_MACDEF;
    }
  else if (MATCH (ch, bcomm.string, false))
    {
      result = TOKEN_STRING;
    }
  else if ((default_word_regexp && (isalpha (ch) || ch == '_'))
#ifdef ENABLE_CHANGEWORD
           || (! default_word_regexp && word_regexp.fastmap[ch])
#endif /* ENABLE_CHANGEWORD */
           )
    {
      result = TOKEN_WORD;
    }
  else if (MATCH (ch, lquote.string, false))
    {
      result = TOKEN_STRING;
    }
  else
    switch (ch)
      {
      case '(':
        result = TOKEN_OPEN;
        break;
      case ',':
        result = TOKEN_COMMA;
        break;
      case ')':
        result = TOKEN_CLOSE;
        break;
      default:
        result = TOKEN_SIMPLE;
      }

#ifdef DEBUG_INPUT
  xfprintf (stderr, "peek_token -> %s\n", token_type_string (result));
#endif /* DEBUG_INPUT */
  return result;
}


#ifdef DEBUG_INPUT

static const char *
token_type_string (token_type t)
{
 switch (t)
    { /* TOKSW */
    case TOKEN_EOF:
      return "EOF";
    case TOKEN_STRING:
      return "STRING";
    case TOKEN_WORD:
      return "WORD";
    case TOKEN_OPEN:
      return "OPEN";
    case TOKEN_COMMA:
      return "COMMA";
    case TOKEN_CLOSE:
      return "CLOSE";
    case TOKEN_SIMPLE:
      return "SIMPLE";
    case TOKEN_MACDEF:
      return "MACDEF";
    default:
      abort ();
    }
 }

static void
print_token (const char *s, token_type t, token_data *td)
{
  xfprintf (stderr, "%s: ", s);
  switch (t)
    { /* TOKSW */
    case TOKEN_OPEN:
    case TOKEN_COMMA:
    case TOKEN_CLOSE:
    case TOKEN_SIMPLE:
      xfprintf (stderr, "char:");
      break;

    case TOKEN_WORD:
      xfprintf (stderr, "word:");
      break;

    case TOKEN_STRING:
      xfprintf (stderr, "string:");
      break;

    case TOKEN_MACDEF:
      xfprintf (stderr, "macro: %p\n", TOKEN_DATA_FUNC (td));
      break;

    case TOKEN_EOF:
      xfprintf (stderr, "eof\n");
      break;
    }
  xfprintf (stderr, "\t\"%s\"\n", TOKEN_DATA_TEXT (td));
}

static void M4_GNUC_UNUSED
lex_debug (void)
{
  token_type t;
  token_data td;

  while ((t = next_token (&td, NULL)) != TOKEN_EOF)
    print_token ("lex", t, &td);
}
#endif /* DEBUG_INPUT */
