/* argmatch.h -- definitions and prototypes for argmatch.c

   Copyright (C) 1990, 1998-1999, 2001-2002, 2004-2005, 2009-2020 Free Software
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

/* Written by David MacKenzie <djm@ai.mit.edu>
   Modified by Akim Demaille <demaille@inf.enst.fr> */

#ifndef ARGMATCH_H_
# define ARGMATCH_H_ 1

# include <limits.h>
# include <stdbool.h>
# include <stddef.h>
# include <stdio.h>
# include <string.h> /* memcmp */

# include "gettext.h"
# include "quote.h"
# include "verify.h"

# ifdef  __cplusplus
extern "C" {
# endif

# define ARRAY_CARDINALITY(Array) (sizeof (Array) / sizeof *(Array))

/* Assert there are as many real arguments as there are values
   (argument list ends with a NULL guard).  */

# define ARGMATCH_VERIFY(Arglist, Vallist) \
    verify (ARRAY_CARDINALITY (Arglist) == ARRAY_CARDINALITY (Vallist) + 1)

/* Return the index of the element of ARGLIST (NULL terminated) that
   matches with ARG.  If VALLIST is not NULL, then use it to resolve
   false ambiguities (i.e., different matches of ARG but corresponding
   to the same values in VALLIST).  */

ptrdiff_t argmatch (char const *arg, char const *const *arglist,
                    void const *vallist, size_t valsize) _GL_ATTRIBUTE_PURE;

# define ARGMATCH(Arg, Arglist, Vallist) \
  argmatch (Arg, Arglist, (void const *) (Vallist), sizeof *(Vallist))

/* xargmatch calls this function when it fails.  This function should not
   return.  By default, this is a function that calls ARGMATCH_DIE which
   in turn defaults to 'exit (exit_failure)'.  */
typedef void (*argmatch_exit_fn) (void);
extern argmatch_exit_fn argmatch_die;

/* Report on stderr why argmatch failed.  Report correct values. */

void argmatch_invalid (char const *context, char const *value,
                       ptrdiff_t problem);

/* Left for compatibility with the old name invalid_arg */

# define invalid_arg(Context, Value, Problem) \
  argmatch_invalid (Context, Value, Problem)



/* Report on stderr the list of possible arguments.  */

void argmatch_valid (char const *const *arglist,
                     void const *vallist, size_t valsize);

# define ARGMATCH_VALID(Arglist, Vallist) \
  argmatch_valid (Arglist, (void const *) (Vallist), sizeof *(Vallist))



/* Same as argmatch, but upon failure, report an explanation of the
   failure, and exit using the function EXIT_FN. */

ptrdiff_t __xargmatch_internal (char const *context,
                                char const *arg, char const *const *arglist,
                                void const *vallist, size_t valsize,
                                argmatch_exit_fn exit_fn);

/* Programmer friendly interface to __xargmatch_internal. */

# define XARGMATCH(Context, Arg, Arglist, Vallist)              \
  ((Vallist) [__xargmatch_internal (Context, Arg, Arglist,      \
                                    (void const *) (Vallist),   \
                                    sizeof *(Vallist),          \
                                    argmatch_die)])

/* Convert a value into a corresponding argument. */

char const *argmatch_to_argument (void const *value,
                                  char const *const *arglist,
                                  void const *vallist, size_t valsize)
  _GL_ATTRIBUTE_PURE;

# define ARGMATCH_TO_ARGUMENT(Value, Arglist, Vallist)                  \
  argmatch_to_argument (Value, Arglist,                                 \
                        (void const *) (Vallist), sizeof *(Vallist))

# define ARGMATCH_DEFINE_GROUP(Name, Type)                              \
  /* The type of the values of this group.  */                          \
  typedef Type argmatch_##Name##_type;                                  \
                                                                        \
  /* The size of the type of the values of this group. */               \
  enum argmatch_##Name##_size_enum                                      \
  {                                                                     \
    argmatch_##Name##_size = sizeof (argmatch_##Name##_type)            \
  };                                                                    \
                                                                        \
  /* Argument mapping of this group.  */                                \
  typedef struct                                                        \
  {                                                                     \
    /* Argument (e.g., "simple").  */                                   \
    const char *arg;                                                    \
    /* Value (e.g., simple_backups).  */                                \
    const argmatch_##Name##_type val;                                   \
  } argmatch_##Name##_arg;                                              \
                                                                        \
  /* Documentation of this group.  */                                   \
  typedef struct                                                        \
  {                                                                     \
    /* Argument (e.g., "simple").  */                                   \
    const char *arg;                                                    \
    /* Documentation (e.g., N_("always make simple backups")).  */      \
    const char *doc;                                                    \
  } argmatch_##Name##_doc;                                              \
                                                                        \
  /* All the features of an argmatch group.  */                         \
  typedef struct                                                        \
  {                                                                     \
    const argmatch_##Name##_arg* args;                                  \
    const argmatch_##Name##_doc* docs;                                  \
                                                                        \
    /* Printed before the usage message.  */                            \
    const char *doc_pre;                                                \
    /* Printed after the usage message.  */                             \
    const char *doc_post;                                               \
  } argmatch_##Name##_group_type;                                       \
                                                                        \
  /* The structure the user must build.  */                             \
  extern const argmatch_##Name##_group_type argmatch_##Name##_group;    \
                                                                        \
  /* Print the documentation of this group.  */                         \
  void argmatch_##Name##_usage (FILE *out);                             \
                                                                        \
  /* If nonnegative, the index I of ARG in ARGS, i.e,                   \
     ARGS[I] == ARG.                                                    \
     Return -1 for invalid argument, -2 for ambiguous argument. */      \
  ptrdiff_t argmatch_##Name##_choice (const char *arg);                 \
                                                                        \
  /* A pointer to the corresponding value if it exists, or              \
     report an error and exit with failure if the argument was          \
     not recognized. */                                                 \
  const argmatch_##Name##_type*                                         \
  argmatch_##Name##_value (const char *context, const char *arg);       \
                                                                        \
  /* The first argument in ARGS that matches this value, or NULL.  */   \
  const char *                                                          \
  argmatch_##Name##_argument (const argmatch_##Name##_type *val);       \
                                                                        \
  ptrdiff_t                                                             \
  argmatch_##Name##_choice (const char *arg)                            \
  {                                                                     \
    const argmatch_##Name##_group_type *g = &argmatch_##Name##_group;   \
    size_t size = argmatch_##Name##_size;                               \
    ptrdiff_t res = -1;      /* Index of first nonexact match.  */      \
    bool ambiguous = false;  /* Whether multiple nonexact match(es). */ \
    size_t arglen = strlen (arg);                                       \
                                                                        \
    /* Test all elements for either exact match or abbreviated          \
       matches.  */                                                     \
    for (size_t i = 0; g->args[i].arg; i++)                             \
      if (!strncmp (g->args[i].arg, arg, arglen))                       \
        {                                                               \
          if (strlen (g->args[i].arg) == arglen)                        \
            /* Exact match found.  */                                   \
            return i;                                                   \
          else if (res == -1)                                           \
            /* First nonexact match found.  */                          \
            res = i;                                                    \
          else if (memcmp (&g->args[res].val, &g->args[i].val, size))   \
            /* Second nonexact match found.  */                         \
            /* There is a real ambiguity, or we could not               \
               disambiguate. */                                         \
            ambiguous = true;                                           \
        }                                                               \
    return ambiguous ? -2 : res;                                        \
  }                                                                     \
                                                                        \
  const char *                                                          \
  argmatch_##Name##_argument (const argmatch_##Name##_type *val)        \
  {                                                                     \
    const argmatch_##Name##_group_type *g = &argmatch_##Name##_group;   \
    size_t size = argmatch_##Name##_size;                               \
    for (size_t i = 0; g->args[i].arg; i++)                             \
      if (!memcmp (val, &g->args[i].val, size))                         \
        return g->args[i].arg;                                          \
    return NULL;                                                        \
  }                                                                     \
                                                                        \
  /* List the valid values of this group. */                            \
  static void                                                           \
  argmatch_##Name##_valid (FILE *out)                                   \
  {                                                                     \
    const argmatch_##Name##_group_type *g = &argmatch_##Name##_group;   \
    size_t size = argmatch_##Name##_size;                               \
                                                                        \
    /* Try to put synonyms on the same line.  Synonyms are expected     \
       to follow each other. */                                         \
    fputs (gettext ("Valid arguments are:"), out);                      \
    for (int i = 0; g->args[i].arg; i++)                                \
      if (i == 0                                                        \
          || memcmp (&g->args[i-1].val, &g->args[i].val, size))         \
        fprintf (out, "\n  - %s", quote (g->args[i].arg));              \
      else                                                              \
        fprintf (out, ", %s", quote (g->args[i].arg));                  \
    putc ('\n', out);                                                   \
  }                                                                     \
                                                                        \
  const argmatch_##Name##_type*                                         \
  argmatch_##Name##_value (const char *context, const char *arg)        \
  {                                                                     \
    const argmatch_##Name##_group_type *g = &argmatch_##Name##_group;   \
    ptrdiff_t res = argmatch_##Name##_choice (arg);                     \
    if (res < 0)                                                        \
      {                                                                 \
        argmatch_invalid (context, arg, res);                           \
        argmatch_##Name##_valid (stderr);                               \
        argmatch_die ();                                                \
      }                                                                 \
    return &g->args[res].val;                                           \
  }                                                                     \
                                                                        \
  /* The column in which the documentation is displayed.                \
     The leftmost possible, but no more than 20. */                     \
  static int                                                            \
  argmatch_##Name##_doc_col (void)                                      \
  {                                                                     \
    const argmatch_##Name##_group_type *g = &argmatch_##Name##_group;   \
    size_t size = argmatch_##Name##_size;                               \
    int res = 0;                                                        \
    for (int i = 0; g->docs[i].arg; ++i)                                \
      {                                                                 \
        int col = 4;                                                    \
        int ival = argmatch_##Name##_choice (g->docs[i].arg);           \
        if (ival < 0)                                                   \
          /* Pseudo argument, display it. */                            \
          col += strlen (g->docs[i].arg);                               \
        else                                                            \
          /* Genuine argument, display it with its synonyms. */         \
          for (int j = 0; g->args[j].arg; ++j)                          \
            if (! memcmp (&g->args[ival].val, &g->args[j].val, size))   \
              col += (col == 4 ? 0 : 2) + strlen (g->args[j].arg);      \
        if (res <= col)                                                 \
          res = col <= 20 ? col : 20;                                   \
      }                                                                 \
    return res ? res : 20;                                              \
  }                                                                     \
                                                                        \
  void                                                                  \
  argmatch_##Name##_usage (FILE *out)                                   \
  {                                                                     \
    const argmatch_##Name##_group_type *g = &argmatch_##Name##_group;   \
    size_t size = argmatch_##Name##_size;                               \
    /* Width of the screen.  Help2man does not seem to support          \
       arguments on several lines, so in that case pretend a very       \
       large width. */                                                  \
    const int screen_width = getenv ("HELP2MAN") ? INT_MAX : 80;        \
    if (g->doc_pre)                                                     \
      fprintf (out, "%s\n", gettext (g->doc_pre));                      \
    int doc_col = argmatch_##Name##_doc_col ();                         \
    for (int i = 0; g->docs[i].arg; ++i)                                \
      {                                                                 \
        int col = 0;                                                    \
        bool first = true;                                              \
        int ival = argmatch_##Name##_choice (g->docs[i].arg);           \
        if (ival < 0)                                                   \
          /* Pseudo argument, display it. */                            \
          col += fprintf (out,  "  %s", g->docs[i].arg);                \
        else                                                            \
          /* Genuine argument, display it with its synonyms. */         \
          for (int j = 0; g->args[j].arg; ++j)                          \
            if (! memcmp (&g->args[ival].val, &g->args[j].val, size))   \
              {                                                         \
                if (!first                                              \
                    && screen_width < col + 2 + strlen (g->args[j].arg)) \
                  {                                                     \
                    fprintf (out, ",\n");                               \
                    col = 0;                                            \
                    first = true;                                       \
                  }                                                     \
                if (first)                                              \
                  {                                                     \
                    col += fprintf (out, " ");                          \
                    first = false;                                      \
                  }                                                     \
                else                                                    \
                  col += fprintf (out, ",");                            \
                col += fprintf (out,  " %s", g->args[j].arg);           \
              }                                                         \
        /* The doc.  Separated by at least two spaces. */               \
        if (doc_col < col + 2)                                          \
          {                                                             \
            fprintf (out, "\n");                                        \
            col = 0;                                                    \
          }                                                             \
        fprintf (out, "%*s%s\n",                                        \
                 doc_col - col, "", gettext (g->docs[i].doc));          \
      }                                                                 \
    if (g->doc_post)                                                    \
      fprintf (out, "%s\n", gettext (g->doc_post));                     \
  }

# ifdef  __cplusplus
}
# endif

#endif /* ARGMATCH_H_ */
