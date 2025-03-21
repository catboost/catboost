/* ----------------------------------------------------------------------------- 
 * This file is part of SWIG, which is licensed as a whole under version 3 
 * (or any later version) of the GNU General Public License. Some additional
 * terms also apply to certain portions of SWIG. The full details of the SWIG
 * license and copyrights can be found in the LICENSE and COPYRIGHT files
 * included with the SWIG source code as distributed by the SWIG developers
 * and at https://www.swig.org/legal.html.
 *
 * swigmain.cxx
 *
 * Simplified Wrapper and Interface Generator  (SWIG)
 *
 * This file is the main entry point to SWIG.  It collects the command
 * line options, registers built-in language modules, and instantiates
 * a module for code generation.   If adding new language modules
 * to SWIG, you would modify this file.
 * ----------------------------------------------------------------------------- */

#include "swigmod.h"
#include <ctype.h>

/* Module factories.  These functions are used to instantiate
   the built-in language modules.    If adding a new language
   module to SWIG, place a similar function here. Make sure
   the function has "C" linkage.  This is required so that modules
   can be dynamically loaded in future versions. */

extern "C" {
  Language *swig_c(void);
  Language *swig_csharp(void);
  Language *swig_d(void);
  Language *swig_go(void);
  Language *swig_guile(void);
  Language *swig_java(void);
  Language *swig_javascript(void);
  Language *swig_lua(void);
  Language *swig_mzscheme(void);
  Language *swig_ocaml(void);
  Language *swig_octave(void);
  Language *swig_perl5(void);
  Language *swig_php(void);
  Language *swig_python(void);
  Language *swig_r(void);
  Language *swig_ruby(void);
  Language *swig_scilab(void);
  Language *swig_tcl(void);
  Language *swig_xml(void);
}

/* Association of command line options to language modules.
   Place an entry for new language modules here, keeping the
   list sorted alphabetically. */

static TargetLanguageModule modules[] = {
  {"-allegrocl", NULL, "ALLEGROCL", Disabled},
  {"-c", swig_c, "C", Experimental},
  {"-chicken", NULL, "CHICKEN", Disabled},
  {"-clisp", NULL, "CLISP", Disabled},
  {"-csharp", swig_csharp, "C#", Supported},
  {"-d", swig_d, "D", Supported},
  {"-go", swig_go, "Go", Supported},
  {"-guile", swig_guile, "Guile", Supported},
  {"-java", swig_java, "Java", Supported},
  {"-javascript", swig_javascript, "Javascript", Supported},
  {"-lua", swig_lua, "Lua", Supported},
  {"-modula3", NULL, "Modula 3", Disabled},
  {"-mzscheme", swig_mzscheme, "MzScheme/Racket", Deprecated},
  {"-ocaml", swig_ocaml, "OCaml", Experimental},
  {"-octave", swig_octave, "Octave", Supported},
  {"-perl", swig_perl5, NULL, Supported},
  {"-perl5", swig_perl5, "Perl 5", Supported},
  {"-php", swig_php, NULL, Supported},
  {"-php5", NULL, "PHP 5", Disabled},
  {"-php7", swig_php, "PHP 8 or later", Supported},
  {"-pike", NULL, "Pike", Disabled},
  {"-python", swig_python, "Python", Supported},
  {"-r", swig_r, "R (aka GNU S)", Supported},
  {"-ruby", swig_ruby, "Ruby", Supported},
  {"-scilab", swig_scilab, "Scilab", Supported},
  {"-sexp", NULL, "Lisp S-Expressions", Disabled},
  {"-tcl", swig_tcl, NULL, Supported},
  {"-tcl8", swig_tcl, "Tcl 8", Supported},
  {"-uffi", NULL, "Common Lisp / UFFI", Disabled},
  {"-xml", swig_xml, "XML", Supported},
  {NULL, NULL, NULL, Disabled}
};

//-----------------------------------------------------------------
// main()
//
// Main program.    Initializes the files and starts the parser.
//-----------------------------------------------------------------

static void SWIG_merge_envopt(const char *env, int oargc, char *oargv[], int *nargc, char ***nargv) {
  if (!env) {
    *nargc = oargc;
    *nargv = (char **)Malloc(sizeof(char *) * (oargc + 1));
    memcpy(*nargv, oargv, sizeof(char *) * (oargc + 1));
    return;
  }

  int argc = 1;
  int arge = oargc + 1024;
  char **argv = (char **) Malloc(sizeof(char *) * (arge + 1));
  char *buffer = (char *) Malloc(2048);
  char *b = buffer;
  char *be = b + 1023;
  const char *c = env;
  while ((b != be) && *c && (argc < arge)) {
    while (isspace(*c) && *c)
      ++c;
    if (*c) {
      argv[argc] = b;
      ++argc;
    }
    while ((b != be) && *c && !isspace(*c)) {
      *(b++) = *(c++);
    }
    *b++ = 0;
  }

  argv[0] = oargv[0];
  for (int i = 1; (i < oargc) && (argc < arge); ++i, ++argc) {
    argv[argc] = oargv[i];
  }
  argv[argc] = NULL;

  *nargc = argc;
  *nargv = argv;
}

static void insert_option(int *argc, char ***argv, int index, char const *start, char const *end) {
  int new_argc = *argc;
  char **new_argv = *argv;
  size_t option_len = end - start;

  // Preserve the NULL pointer at argv[argc]
  new_argv = (char **)Realloc(new_argv, (new_argc + 2) * sizeof(char *));
  memmove(&new_argv[index + 1], &new_argv[index], sizeof(char *) * (new_argc + 1 - index));
  new_argc++;

  new_argv[index] = (char *)Malloc(option_len + 1);
  memcpy(new_argv[index], start, option_len);
  new_argv[index][option_len] = '\0';

  *argc = new_argc;
  *argv = new_argv;
}

static void merge_options_files(int *argc, char ***argv) {
  static const int BUFFER_SIZE = 4096;
  char buffer[BUFFER_SIZE];
  int i;
  int insert;
  char **new_argv = *argv;
  int new_argc = *argc;
  FILE *f;

  i = 1;
  while (i < new_argc) {
    if (new_argv[i] && new_argv[i][0] == '@' && (f = fopen(&new_argv[i][1], "r"))) {
      int ci;
      char *b;
      char *be = &buffer[BUFFER_SIZE];
      int quote = 0;
      bool escape = false;

      new_argc--;
      memmove(&new_argv[i], &new_argv[i + 1], sizeof(char *) * (new_argc - i));
      insert = i;
      b = buffer;

      while ((ci = fgetc(f)) != EOF) {
        const char c = static_cast<char>(ci);
        if (escape) {
          if (b != be) {
            *b = c;
            ++b;
          }
          escape = false;
        } else if (c == '\\') {
          escape = true;
        } else if (!quote && (c == '\'' || c == '"')) {
          quote = c;
        } else if (quote && c == quote) {
          quote = 0;
        } else if (isspace(c) && !quote) {
          if (b != buffer) {
            insert_option(&new_argc, &new_argv, insert, buffer, b);
            insert++;

            b = buffer;
          }
        } else if (b != be) {
          *b = c;
          ++b;
        }
      }
      if (b != buffer)
        insert_option(&new_argc, &new_argv, insert, buffer, b);
      fclose(f);
    } else {
      ++i;
    }
  }

  *argv = new_argv;
  *argc = new_argc;
}

int main(int margc, char **margv) {
  int i;
  const TargetLanguageModule *language_module = 0;

  int argc;
  char **argv;

  /* Check for SWIG_FEATURES environment variable */

  SWIG_merge_envopt(getenv("SWIG_FEATURES"), margc, margv, &argc, &argv);
  merge_options_files(&argc, &argv);

  Swig_init_args(argc, argv);

  /* Get options */
  for (i = 1; i < argc; i++) {
    if (argv[i]) {
      bool is_target_language_module = false;
      for (int j = 0; modules[j].name; j++) {
	if (strcmp(modules[j].name, argv[i]) == 0) {
	  if (!language_module) {
	    language_module = &modules[j];
	    is_target_language_module = true;
	  } else {
	    Printf(stderr, "Only one target language can be supported at a time (both %s and %s were specified).\n", language_module->name, argv[i]);
	    Exit(EXIT_FAILURE);
	  }
	}
      }
      if (is_target_language_module) {
	Swig_mark_arg(i);
	if (language_module->status == Disabled) {
	  if (language_module->help)
	    Printf(stderr, "Target language option %s (%s) is no longer supported.\n", language_module->name, language_module->help);
	  else
	    Printf(stderr, "Target language option %s is no longer supported.\n", language_module->name);
	  Exit(EXIT_FAILURE);
	}
      } else if ((strcmp(argv[i], "-help") == 0) || (strcmp(argv[i], "--help") == 0)) {
	if (strcmp(argv[i], "--help") == 0)
	  strcpy(argv[i], "-help");
	Printf(stdout, "Supported Target Language Options\n");
	for (int j = 0; modules[j].name; j++) {
	  if (modules[j].help && modules[j].status == Supported) {
	    Printf(stdout, "     %-15s - Generate %s wrappers\n", modules[j].name, modules[j].help);
	  }
	}
	Printf(stdout, "\nExperimental Target Language Options\n");
	for (int j = 0; modules[j].name; j++) {
	  if (modules[j].help && modules[j].status == Experimental) {
	    Printf(stdout, "     %-15s - Generate %s wrappers\n", modules[j].name, modules[j].help);
	  }
	}
	Printf(stdout, "\nDeprecated Target Language Options\n");
	for (int j = 0; modules[j].name; j++) {
	  if (modules[j].help && modules[j].status == Deprecated) {
	    Printf(stdout, "     %-15s - Generate %s wrappers\n", modules[j].name, modules[j].help);
	  }
	}
	// Swig_mark_arg not called as the general -help options also need to be displayed later on
      }
    }
  }

  int res = SWIG_main(argc, argv, language_module);

  return res;
}
