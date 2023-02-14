/* -----------------------------------------------------------------------------
 * This file is part of SWIG, which is licensed as a whole under version 3
 * (or any later version) of the GNU General Public License. Some additional
 * terms also apply to certain portions of SWIG. The full details of the SWIG
 * license and copyrights can be found in the LICENSE and COPYRIGHT files
 * included with the SWIG source code as distributed by the SWIG developers
 * and at http://www.swig.org/legal.html.
 *
 * python.cxx
 *
 * Python language module for SWIG.
 * ----------------------------------------------------------------------------- */

#include "swigmod.h"
#include <limits.h>
#include "cparse.h"
#include <ctype.h>
#include <errno.h>
#include "pydoc.h"

#include <stdint.h>

#define PYSHADOW_MEMBER  0x2
#define WARN_PYTHON_MULTIPLE_INH 405

#define PYTHON_INT_MAX (2147483647)
#define PYTHON_INT_MIN (-2147483647-1)

static String *const_code = 0;
static String *module = 0;
static String *package = 0;
static String *mainmodule = 0;
static String *interface = 0;
static String *global_name = 0;
static int shadow = 1;
static int use_kw = 0;
static int director_method_index = 0;
static int builtin = 0;

static File *f_begin = 0;
static File *f_runtime = 0;
static File *f_runtime_h = 0;
static File *f_header = 0;
static File *f_wrappers = 0;
static File *f_directors = 0;
static File *f_directors_h = 0;
static File *f_init = 0;
static File *f_shadow_py = 0;
static String *f_shadow = 0;
static String *f_shadow_begin = 0;
static Hash *f_shadow_imports = 0;
static String *f_shadow_after_begin = 0;
static String *f_shadow_stubs = 0;
static Hash *builtin_getset = 0;
static Hash *builtin_closures = 0;
static Hash *class_members = 0;
static File *f_builtins = 0;
static String *builtin_tp_init = 0;
static String *builtin_methods = 0;
static String *builtin_default_unref = 0;
static String *builtin_closures_code = 0;

static String *methods;
static String *methods_proxydocs;
static String *class_name;
static String *shadow_indent = 0;
static int in_class = 0;
static int no_header_file = 0;
static int max_bases = 0;
static int builtin_bases_needed = 0;

static int py3 = 0;

/* C++ Support + Shadow Classes */

static int have_constructor;
static int have_repr;
static String *real_classname;

/* Thread Support */
static int threads = 0;
static int nothreads = 0;

/* Other options */
static int dirvtable = 0;
static int doxygen = 0;
static int fastunpack = 1;
static int fastproxy = 0;
static int olddefs = 0;
static int castmode = 0;
static int extranative = 0;
static int nortti = 0;
static int relativeimport = 0;

/* flags for the make_autodoc function */
enum autodoc_t {
  AUTODOC_CLASS,
  AUTODOC_CTOR,
  AUTODOC_DTOR,
  AUTODOC_STATICFUNC,
  AUTODOC_FUNC,
  AUTODOC_METHOD,
  AUTODOC_CONST,
  AUTODOC_VAR
};


static const char *usage1 = "\
Python Options (available with -python)\n\
     -builtin        - Create Python built-in types rather than proxy classes, for better performance\n\
     -castmode       - Enable the casting mode, which allows implicit cast between types in Python\n\
     -debug-doxygen-parser     - Display doxygen parser module debugging information\n\
     -debug-doxygen-translator - Display doxygen translator module debugging information\n\
     -dirvtable      - Generate a pseudo virtual table for directors for faster dispatch\n\
     -doxygen        - Convert C++ doxygen comments to pydoc comments in proxy classes\n\
     -extranative    - Return extra native wrappers for C++ std containers wherever possible\n\
     -fastproxy      - Use fast proxy mechanism for member methods\n\
     -globals <name> - Set <name> used to access C global variable (default: 'cvar')\n\
     -interface <mod>- Set low-level C/C++ module name to <mod> (default: module name prefixed by '_')\n\
     -keyword        - Use keyword arguments\n";
static const char *usage2 = "\
     -nofastunpack   - Use traditional UnpackTuple method to parse the argument functions\n\
     -noh            - Don't generate the output header file\n";
static const char *usage3 = "\
     -noproxy        - Don't generate proxy classes\n\
     -nortti         - Disable the use of the native C++ RTTI with directors\n\
     -nothreads      - Disable thread support for the entire interface\n\
     -olddefs        - Keep the old method definitions when using -fastproxy\n\
     -py3            - Generate code with Python 3 specific features and syntax\n\
     -relativeimport - Use relative Python imports\n\
     -threads        - Add thread support for all the interface\n\
     -O              - Enable the following optimization options:\n\
                         -fastdispatch -fastproxy -fvirtual\n\
\n";

static String *getSlot(Node *n = NULL, const char *key = NULL, String *default_slot = NULL) {
  static String *zero = NewString("0");
  String *val = n && key && *key ? Getattr(n, key) : NULL;
  return val ? val : default_slot ? default_slot : zero;
}

static void printSlot(File *f, String *slotval, const char *slotname, const char *functype = NULL) {
  String *slotval_override = 0;
  if (functype && Strcmp(slotval, "0") == 0)
    slotval = slotval_override = NewStringf("(%s) %s", functype, slotval);
  int len = Len(slotval);
  int fieldwidth = len > 41 ? (len > 61 ? 0 : 61 - len) : 41 - len;
  Printf(f, "    %s,%*s/* %s */\n", slotval, fieldwidth, "", slotname);
  Delete(slotval_override);
}

static String *getClosure(String *functype, String *wrapper, int funpack = 0) {
  static const char *functypes[] = {
    "unaryfunc", "SWIGPY_UNARYFUNC_CLOSURE",
    "destructor", "SWIGPY_DESTRUCTOR_CLOSURE",
    "inquiry", "SWIGPY_INQUIRY_CLOSURE",
    "getiterfunc", "SWIGPY_GETITERFUNC_CLOSURE",
    "binaryfunc", "SWIGPY_BINARYFUNC_CLOSURE",
    "ternaryfunc", "SWIGPY_TERNARYFUNC_CLOSURE",
    "ternarycallfunc", "SWIGPY_TERNARYCALLFUNC_CLOSURE",
    "lenfunc", "SWIGPY_LENFUNC_CLOSURE",
    "ssizeargfunc", "SWIGPY_SSIZEARGFUNC_CLOSURE",
    "ssizessizeargfunc", "SWIGPY_SSIZESSIZEARGFUNC_CLOSURE",
    "ssizeobjargproc", "SWIGPY_SSIZEOBJARGPROC_CLOSURE",
    "ssizessizeobjargproc", "SWIGPY_SSIZESSIZEOBJARGPROC_CLOSURE",
    "objobjproc", "SWIGPY_OBJOBJPROC_CLOSURE",
    "objobjargproc", "SWIGPY_OBJOBJARGPROC_CLOSURE",
    "reprfunc", "SWIGPY_REPRFUNC_CLOSURE",
    "hashfunc", "SWIGPY_HASHFUNC_CLOSURE",
    "iternextfunc", "SWIGPY_ITERNEXTFUNC_CLOSURE",
    NULL
  };

  static const char *funpack_functypes[] = {
    "unaryfunc", "SWIGPY_UNARYFUNC_CLOSURE",
    "destructor", "SWIGPY_DESTRUCTOR_CLOSURE",
    "inquiry", "SWIGPY_INQUIRY_CLOSURE",
    "getiterfunc", "SWIGPY_GETITERFUNC_CLOSURE",
    "ternaryfunc", "SWIGPY_TERNARYFUNC_CLOSURE",
    "ternarycallfunc", "SWIGPY_TERNARYCALLFUNC_CLOSURE",
    "lenfunc", "SWIGPY_LENFUNC_CLOSURE",
    "ssizeargfunc", "SWIGPY_FUNPACK_SSIZEARGFUNC_CLOSURE",
    "ssizessizeargfunc", "SWIGPY_SSIZESSIZEARGFUNC_CLOSURE",
    "ssizeobjargproc", "SWIGPY_SSIZEOBJARGPROC_CLOSURE",
    "ssizessizeobjargproc", "SWIGPY_SSIZESSIZEOBJARGPROC_CLOSURE",
    "objobjproc", "SWIGPY_FUNPACK_OBJOBJPROC_CLOSURE",
    "objobjargproc", "SWIGPY_OBJOBJARGPROC_CLOSURE",
    "reprfunc", "SWIGPY_REPRFUNC_CLOSURE",
    "hashfunc", "SWIGPY_HASHFUNC_CLOSURE",
    "iternextfunc", "SWIGPY_ITERNEXTFUNC_CLOSURE",
    NULL
  };

  if (!functype)
    return NULL;
  char *c = Char(functype);
  int i;
  if (funpack) {
    for (i = 0; funpack_functypes[i] != NULL; i += 2) {
      if (!strcmp(c, funpack_functypes[i]))
	return NewStringf("%s(%s)", funpack_functypes[i + 1], wrapper);
    }
  } else {
    for (i = 0; functypes[i] != NULL; i += 2) {
      if (!strcmp(c, functypes[i]))
	return NewStringf("%s(%s)", functypes[i + 1], wrapper);
    }
  }
  return NULL;
}

class PYTHON:public Language {
public:
  PYTHON() {
    /* Add code to manage protected constructors and directors */
    director_prot_ctor_code = NewString("");
    Printv(director_prot_ctor_code,
	   "if ( $comparison ) { /* subclassed */\n",
	   "  $director_new \n",
	   "} else {\n", "  SWIG_SetErrorMsg(PyExc_RuntimeError,\"accessing abstract class or protected constructor\"); \n", "  SWIG_fail;\n", "}\n", NIL);
    director_multiple_inheritance = 1;
    director_language = 1;
  }

  ~PYTHON() {
    delete doxygenTranslator;
  }

  /* ------------------------------------------------------------
   * Thread Implementation
   * ------------------------------------------------------------ */
  int threads_enable(Node *n) const {
    return threads && !GetFlagAttr(n, "feature:nothread");
  }

  int initialize_threads(String *f_init) {
    if (!threads) {
      return SWIG_OK;
    }
    Printf(f_init, "\n");
    Printf(f_init, "/* Initialize threading */\n");
    Printf(f_init, "SWIG_PYTHON_INITIALIZE_THREADS;\n");

    return SWIG_OK;
  }

  virtual void thread_begin_block(Node *n, String *f) {
    if (!GetFlag(n, "feature:nothreadblock")) {
      String *bb = Getattr(n, "feature:threadbeginblock");
      if (bb) {
	Append(f, bb);
      } else {
	Append(f, "SWIG_PYTHON_THREAD_BEGIN_BLOCK;\n");
      }
    }
  }

  virtual void thread_end_block(Node *n, String *f) {
    if (!GetFlag(n, "feature:nothreadblock")) {
      String *eb = Getattr(n, "feature:threadendblock");
      if (eb) {
	Append(f, eb);
      } else {
	Append(f, "SWIG_PYTHON_THREAD_END_BLOCK;\n");
      }
    }
  }

  virtual void thread_begin_allow(Node *n, String *f) {
    if (!GetFlag(n, "feature:nothreadallow")) {
      String *bb = Getattr(n, "feature:threadbeginallow");
      Append(f, "{\n");
      if (bb) {
	Append(f, bb);
      } else {
	Append(f, "SWIG_PYTHON_THREAD_BEGIN_ALLOW;\n");
      }
    }
  }

  virtual void thread_end_allow(Node *n, String *f) {
    if (!GetFlag(n, "feature:nothreadallow")) {
      String *eb = Getattr(n, "feature:threadendallow");
      Append(f, "\n");
      if (eb) {
	Append(f, eb);
      } else {
	Append(f, "SWIG_PYTHON_THREAD_END_ALLOW;");
      }
      Append(f, "\n}");
    }
  }


  /* ------------------------------------------------------------
   * main()
   * ------------------------------------------------------------ */

  virtual void main(int argc, char *argv[]) {

    SWIG_library_directory("python");

    int doxygen_translator_flags = 0;

    for (int i = 1; i < argc; i++) {
      if (argv[i]) {
	if (strcmp(argv[i], "-interface") == 0) {
	  if (argv[i + 1]) {
	    interface = NewString(argv[i + 1]);
	    Swig_mark_arg(i);
	    Swig_mark_arg(i + 1);
	    i++;
	  } else {
	    Swig_arg_error();
	  }
	  /* end added */
	} else if (strcmp(argv[i], "-globals") == 0) {
	  if (argv[i + 1]) {
	    global_name = NewString(argv[i + 1]);
	    Swig_mark_arg(i);
	    Swig_mark_arg(i + 1);
	    i++;
	  } else {
	    Swig_arg_error();
	  }
	} else if ((strcmp(argv[i], "-shadow") == 0) || ((strcmp(argv[i], "-proxy") == 0))) {
	  shadow = 1;
	  Swig_mark_arg(i);
	} else if ((strcmp(argv[i], "-noproxy") == 0)) {
	  shadow = 0;
	  Swig_mark_arg(i);
	} else if (strcmp(argv[i], "-keyword") == 0) {
	  use_kw = 1;
	  SWIG_cparse_set_compact_default_args(1);
	  Swig_mark_arg(i);
	} else if (strcmp(argv[i], "-nortti") == 0) {
	  nortti = 1;
	  Swig_mark_arg(i);
	} else if (strcmp(argv[i], "-threads") == 0) {
	  threads = 1;
	  Swig_mark_arg(i);
	} else if (strcmp(argv[i], "-nothreads") == 0) {
	  /* Turn off thread support mode */
	  nothreads = 1;
	  Swig_mark_arg(i);
	} else if (strcmp(argv[i], "-dirvtable") == 0) {
	  dirvtable = 1;
	  Swig_mark_arg(i);
	} else if (strcmp(argv[i], "-doxygen") == 0) {
	  doxygen = 1;
	  scan_doxygen_comments = 1;
	  Swig_mark_arg(i);
	} else if (strcmp(argv[i], "-debug-doxygen-translator") == 0) {
	  doxygen_translator_flags |= DoxygenTranslator::debug_translator;
	  Swig_mark_arg(i);
	} else if (strcmp(argv[i], "-debug-doxygen-parser") == 0) {
	  doxygen_translator_flags |= DoxygenTranslator::debug_parser;
	  Swig_mark_arg(i);
	} else if (strcmp(argv[i], "-nofastunpack") == 0) {
	  fastunpack = 0;
	  Swig_mark_arg(i);
	} else if (strcmp(argv[i], "-fastproxy") == 0) {
	  fastproxy = 1;
	  Swig_mark_arg(i);
	} else if (strcmp(argv[i], "-olddefs") == 0) {
	  olddefs = 1;
	  Swig_mark_arg(i);
	} else if (strcmp(argv[i], "-castmode") == 0) {
	  castmode = 1;
	  Swig_mark_arg(i);
	} else if (strcmp(argv[i], "-extranative") == 0) {
	  extranative = 1;
	  Swig_mark_arg(i);
	} else if (strcmp(argv[i], "-noh") == 0) {
	  no_header_file = 1;
	  Swig_mark_arg(i);
	} else if (strcmp(argv[i], "-newvwm") == 0) {
	  /* Turn on new value wrapper mode */
	  /* Undocumented option, did have -help text: New value wrapper mode, use only when everything else fails */
	  Swig_value_wrapper_mode(1);
	  no_header_file = 1;
	  Swig_mark_arg(i);
	} else if (strcmp(argv[i], "-O") == 0) {
	  fastproxy = 1;
	  Wrapper_fast_dispatch_mode_set(1);
	  Wrapper_virtual_elimination_mode_set(1);
	  Swig_mark_arg(i);
	} else if (strcmp(argv[i], "-help") == 0) {
	  fputs(usage1, stdout);
	  fputs(usage2, stdout);
	  fputs(usage3, stdout);
	} else if (strcmp(argv[i], "-py3") == 0) {
	  py3 = 1;
	  Preprocessor_define("SWIGPYTHON_PY3", 0);
	  Swig_mark_arg(i);
	} else if (strcmp(argv[i], "-builtin") == 0) {
	  builtin = 1;
	  Preprocessor_define("SWIGPYTHON_BUILTIN", 0);
	  Swig_mark_arg(i);
	} else if (strcmp(argv[i], "-relativeimport") == 0) {
	  relativeimport = 1;
	  Swig_mark_arg(i);
	} else if (strcmp(argv[i], "-cppcast") == 0 ||
		   strcmp(argv[i], "-fastinit") == 0 ||
		   strcmp(argv[i], "-fastquery") == 0 ||
		   strcmp(argv[i], "-fastunpack") == 0 ||
		   strcmp(argv[i], "-modern") == 0 ||
		   strcmp(argv[i], "-modernargs") == 0 ||
		   strcmp(argv[i], "-noproxydel") == 0 ||
		   strcmp(argv[i], "-safecstrings") == 0) {
	  Printf(stderr, "Deprecated command line option: %s. This option is now always on.\n", argv[i]);
	  Swig_mark_arg(i);
	} else if (strcmp(argv[i], "-aliasobj0") == 0 ||
		   strcmp(argv[i], "-buildnone") == 0 ||
		   strcmp(argv[i], "-classic") == 0 ||
		   strcmp(argv[i], "-classptr") == 0 ||
		   strcmp(argv[i], "-new_repr") == 0 ||
		   strcmp(argv[i], "-new_vwm") == 0 ||
		   strcmp(argv[i], "-newrepr") == 0 ||
		   strcmp(argv[i], "-noaliasobj0") == 0 ||
		   strcmp(argv[i], "-nobuildnone") == 0 ||
		   strcmp(argv[i], "-nocastmode") == 0 ||
		   strcmp(argv[i], "-nocppcast") == 0 ||
		   strcmp(argv[i], "-nodirvtable") == 0 ||
		   strcmp(argv[i], "-noextranative") == 0 ||
		   strcmp(argv[i], "-nofastinit") == 0 ||
		   strcmp(argv[i], "-nofastproxy") == 0 ||
		   strcmp(argv[i], "-nofastquery") == 0 ||
		   strcmp(argv[i], "-nomodern") == 0 ||
		   strcmp(argv[i], "-nomodernargs") == 0 ||
		   strcmp(argv[i], "-noolddefs") == 0 ||
		   strcmp(argv[i], "-nooutputtuple") == 0 ||
		   strcmp(argv[i], "-noproxyimport") == 0 ||
		   strcmp(argv[i], "-nosafecstrings") == 0 ||
		   strcmp(argv[i], "-old_repr") == 0 ||
		   strcmp(argv[i], "-oldrepr") == 0 ||
		   strcmp(argv[i], "-outputtuple") == 0 ||
		   strcmp(argv[i], "-proxydel") == 0) {
	  Printf(stderr, "Deprecated command line option: %s. This option is no longer supported.\n", argv[i]);
	  Swig_mark_arg(i);
	  SWIG_exit(EXIT_FAILURE);
	}

      }
    }

    if (doxygen)
      doxygenTranslator = new PyDocConverter(doxygen_translator_flags);

    if (!global_name)
      global_name = NewString("cvar");
    Preprocessor_define("SWIGPYTHON 1", 0);
    SWIG_typemap_lang("python");
    SWIG_config_file("python.swg");
    allow_overloading();
  }


  /* ------------------------------------------------------------
   * top()
   * ------------------------------------------------------------ */

  virtual int top(Node *n) {
    /* check if directors are enabled for this module.  note: this
     * is a "master" switch, without which no director code will be
     * emitted.  %feature("director") statements are also required
     * to enable directors for individual classes or methods.
     *
     * use %module(directors="1") modulename at the start of the
     * interface file to enable director generation.
     */
    String *mod_docstring = NULL;
    String *moduleimport = NULL;
    {
      Node *mod = Getattr(n, "module");
      if (mod) {
	Node *options = Getattr(mod, "options");
	if (options) {
	  int dirprot = 0;
	  if (Getattr(options, "dirprot")) {
	    dirprot = 1;
	  }
	  if (Getattr(options, "nodirprot")) {
	    dirprot = 0;
	  }
	  if (Getattr(options, "directors")) {
	    allow_directors();
	    if (dirprot)
	      allow_dirprot();
	  }
	  if (Getattr(options, "threads")) {
	    threads = 1;
	  }
	  if (Getattr(options, "castmode")) {
	    castmode = 1;
	  }
	  if (Getattr(options, "nocastmode")) {
	    Printf(stderr, "Deprecated module option: %s. This option is no longer supported.\n", "nocastmode");
	    SWIG_exit(EXIT_FAILURE);
	  }
	  if (Getattr(options, "extranative")) {
	    extranative = 1;
	  }
	  if (Getattr(options, "noextranative")) {
	    Printf(stderr, "Deprecated module option: %s. This option is no longer supported.\n", "noextranative");
	    SWIG_exit(EXIT_FAILURE);
	  }
	  if (Getattr(options, "outputtuple")) {
	    Printf(stderr, "Deprecated module option: %s. This option is no longer supported.\n", "outputtuple");
	    SWIG_exit(EXIT_FAILURE);
	  }
	  if (Getattr(options, "nooutputtuple")) {
	    Printf(stderr, "Deprecated module option: %s. This option is no longer supported.\n", "nooutputtuple");
	    SWIG_exit(EXIT_FAILURE);
	  }
	  mod_docstring = Getattr(options, "docstring");
	  package = Getattr(options, "package");
	  moduleimport = Getattr(options, "moduleimport");
	}
      }
    }

    /* Set comparison with none for ConstructorToFunction */
    setSubclassInstanceCheck(NewString("$arg != Py_None"));

    /* Initialize all of the output files */
    String *outfile = Getattr(n, "outfile");
    String *outfile_h = !no_header_file ? Getattr(n, "outfile_h") : 0;

    f_begin = NewFile(outfile, "w", SWIG_output_files());
    if (!f_begin) {
      FileErrorDisplay(outfile);
      SWIG_exit(EXIT_FAILURE);
    }
    f_runtime = NewString("");
    f_init = NewString("");
    f_header = NewString("");
    f_wrappers = NewString("");
    f_directors_h = NewString("");
    f_directors = NewString("");
    builtin_getset = NewHash();
    builtin_closures = NewHash();
    builtin_closures_code = NewString("");
    class_members = NewHash();
    builtin_methods = NewString("");
    builtin_default_unref = NewString("delete $self;");

    if (builtin) {
      f_builtins = NewString("");
    }

    if (directorsEnabled()) {
      if (!no_header_file) {
	f_runtime_h = NewFile(outfile_h, "w", SWIG_output_files());
	if (!f_runtime_h) {
	  FileErrorDisplay(outfile_h);
	  SWIG_exit(EXIT_FAILURE);
	}
      } else {
	f_runtime_h = f_runtime;
      }
    }

    /* Register file targets with the SWIG file handler */
    Swig_register_filebyname("header", f_header);
    Swig_register_filebyname("wrapper", f_wrappers);
    Swig_register_filebyname("begin", f_begin);
    Swig_register_filebyname("runtime", f_runtime);
    Swig_register_filebyname("init", f_init);
    Swig_register_filebyname("director", f_directors);
    Swig_register_filebyname("director_h", f_directors_h);

    const_code = NewString("");
    methods = NewString("");
    methods_proxydocs = NewString("");

    Swig_banner(f_begin);

    Printf(f_runtime, "\n\n#ifndef SWIGPYTHON\n#define SWIGPYTHON\n#endif\n\n");

    if (directorsEnabled()) {
      Printf(f_runtime, "#define SWIG_DIRECTORS\n");
    }

    if (nothreads) {
      Printf(f_runtime, "#define SWIG_PYTHON_NO_THREADS\n");
    } else if (threads) {
      Printf(f_runtime, "#define SWIG_PYTHON_THREADS\n");
    }

    if (!dirvtable) {
      Printf(f_runtime, "#define SWIG_PYTHON_DIRECTOR_NO_VTABLE\n");
    }

    if (nortti) {
      Printf(f_runtime, "#ifndef SWIG_DIRECTOR_NORTTI\n");
      Printf(f_runtime, "#define SWIG_DIRECTOR_NORTTI\n");
      Printf(f_runtime, "#endif\n");
    }

    if (castmode) {
      Printf(f_runtime, "#define SWIG_CASTRANK_MODE\n");
      Printf(f_runtime, "#define SWIG_PYTHON_CAST_MODE\n");
    }

    if (extranative) {
      Printf(f_runtime, "#define SWIG_PYTHON_EXTRA_NATIVE_CONTAINERS\n");
    }

    if (builtin) {
      Printf(f_runtime, "#define SWIGPYTHON_BUILTIN\n");
    }

    Printf(f_runtime, "\n");

    Printf(f_header, "#ifdef SWIG_TypeQuery\n");
    Printf(f_header, "# undef SWIG_TypeQuery\n");
    Printf(f_header, "#endif\n");
    Printf(f_header, "#define SWIG_TypeQuery SWIG_Python_TypeQuery\n");


    /* Set module name */
    module = Copy(Getattr(n, "name"));
    mainmodule = Getattr(n, "name");

    if (directorsEnabled()) {
      Swig_banner(f_directors_h);
      Printf(f_directors_h, "\n");
      Printf(f_directors_h, "#ifndef SWIG_%s_WRAP_H_\n", module);
      Printf(f_directors_h, "#define SWIG_%s_WRAP_H_\n\n", module);
      if (dirprot_mode()) {
	Printf(f_directors_h, "#include <map>\n");
	Printf(f_directors_h, "#include <string>\n\n");
      }

      Printf(f_directors, "\n\n");
      Printf(f_directors, "/* ---------------------------------------------------\n");
      Printf(f_directors, " * C++ director class methods\n");
      Printf(f_directors, " * --------------------------------------------------- */\n\n");
      if (outfile_h) {
	String *filename = Swig_file_filename(outfile_h);
	Printf(f_directors, "#include \"%s\"\n\n", filename);
	Delete(filename);
      }
    }

    /* If shadow classing is enabled, we're going to change the module name to "_module" */
    String *default_import_code = NewString("");
    if (shadow) {
      String *filen = NewStringf("%s%s.py", SWIG_output_directory(), Char(module));
      // If we don't have an interface then change the module name X to _X
      if (interface)
	module = interface;
      else
	Insert(module, 0, "_");
      if ((f_shadow_py = NewFile(filen, "w", SWIG_output_files())) == 0) {
	FileErrorDisplay(filen);
	SWIG_exit(EXIT_FAILURE);
      }
      Delete(filen);
      filen = NULL;

      f_shadow = NewString("");
      f_shadow_begin = NewString("");
      f_shadow_imports = NewHash();
      f_shadow_after_begin = NewString("");
      f_shadow_stubs = NewString("");

      Swig_register_filebyname("shadow", f_shadow);
      Swig_register_filebyname("python", f_shadow);

      if (!builtin) {
	/* Import the low-level C/C++ module.  This should be a relative import,
	 * since the shadow module may also have been imported by a relative
	 * import, and there is thus no guarantee that the low-level C/C++ module is on
	 * sys.path.  Relative imports must be explicitly specified from 2.6.0
	 * onwards (implicit relative imports raised a DeprecationWarning in 2.6,
	 * and fail in 2.7 onwards).
	 *
	 * First check for __package__ which is available from 2.6 onwards, see PEP366.
	 * Next try determine the shadow wrapper's package based on the __name__ it
	 * was given by the importer that loaded it.
	 * If the module is in a package, load the low-level C/C++ module from the
	 * same package, otherwise load it as a global module.
	 */
        Printv(default_import_code, "# Import the low-level C/C++ module\n", NULL);
        Printv(default_import_code, "if __package__ or \".\" in __name__:\n", NULL);
        Printv(default_import_code, tab4, "from . import ", module, "\n", NULL);
        Printv(default_import_code, "else:\n", NULL);
        Printv(default_import_code, tab4, "import ", module, "\n", NULL);
      } else {
        Printv(default_import_code, "# Pull in all the attributes from the low-level C/C++ module\n", NULL);
        Printv(default_import_code, "if __package__ or \".\" in __name__:\n", NULL);
        Printv(default_import_code, tab4, "from .", module, " import *\n", NULL);
        Printv(default_import_code, "else:\n", NULL);
        Printv(default_import_code, tab4, "from ", module, " import *\n", NULL);
      }

      /* Need builtins to qualify names like Exception that might also be
         defined in this module (try both Python 3 and Python 2 names) */
      Printv(f_shadow, "try:\n", tab4, "import builtins as __builtin__\n", "except ImportError:\n", tab4, "import __builtin__\n", NULL);

      if (!builtin && fastproxy) {
	Printf(f_shadow, "\n");
	Printf(f_shadow, "_swig_new_instance_method = %s.SWIG_PyInstanceMethod_New\n", module);
	Printf(f_shadow, "_swig_new_static_method = %s.SWIG_PyStaticMethod_New\n", module);
      }

      Printv(f_shadow, "\n",
	     "def _swig_repr(self):\n",
	     tab4, "try:\n",
	     tab4, tab4, "strthis = \"proxy of \" + self.this.__repr__()\n",
	     tab4, "except __builtin__.Exception:\n",
	     tab4, tab4, "strthis = \"\"\n",
	     tab4, "return \"<%s.%s; %s >\" % (self.__class__.__module__, self.__class__.__name__, strthis,)\n\n", NIL);

      Printv(f_shadow,  "\n",
	     "def _swig_setattr_nondynamic_instance_variable(set):\n",
	     tab4, "def set_instance_attr(self, name, value):\n",
#ifdef USE_THISOWN
	     tab4, tab4, "if name in (\"this\", \"thisown\"):\n",
	     tab4, tab4, tab4, "set(self, name, value)\n",
#else
	     tab4, tab4, "if name == \"thisown\":\n",
	     tab4, tab4, tab4, "self.this.own(value)\n",
	     tab4, tab4, "elif name == \"this\":\n",
	     tab4, tab4, tab4, "set(self, name, value)\n",
#endif
	     tab4, tab4, "elif hasattr(self, name) and isinstance(getattr(type(self), name), property):\n",
	     tab4, tab4, tab4, "set(self, name, value)\n",
	     tab4, tab4, "else:\n",
	     tab4, tab4, tab4, "raise AttributeError(\"You cannot add instance attributes to %s\" % self)\n",
	     tab4, "return set_instance_attr\n\n", NIL);

      Printv(f_shadow,  "\n",
	     "def _swig_setattr_nondynamic_class_variable(set):\n",
	     tab4, "def set_class_attr(cls, name, value):\n",
	     tab4, tab4, "if hasattr(cls, name) and not isinstance(getattr(cls, name), property):\n",
	     tab4, tab4, tab4, "set(cls, name, value)\n",
	     tab4, tab4, "else:\n",
	     tab4, tab4, tab4, "raise AttributeError(\"You cannot add class attributes to %s\" % cls)\n",
	     tab4, "return set_class_attr\n\n", NIL);

      Printv(f_shadow,  "\n",
	     "def _swig_add_metaclass(metaclass):\n",
	     tab4, "\"\"\"Class decorator for adding a metaclass to a SWIG wrapped class - a slimmed down version of six.add_metaclass\"\"\"\n",
	     tab4, "def wrapper(cls):\n",
	     tab4, tab4, "return metaclass(cls.__name__, cls.__bases__, cls.__dict__.copy())\n",
	     tab4, "return wrapper\n\n", NIL);

      Printv(f_shadow,  "\n",
	     "class _SwigNonDynamicMeta(type):\n",
	     tab4, "\"\"\"Meta class to enforce nondynamic attributes (no new attributes) for a class\"\"\"\n",
	     tab4, "__setattr__ = _swig_setattr_nondynamic_class_variable(type.__setattr__)\n",
	     "\n", NIL);

      Printv(f_shadow, "\n", NIL);

      if (directorsEnabled()) {
	Printv(f_shadow, "import weakref\n\n", NIL);
      }
    }
    // Include some information in the code
    Printf(f_header, "\n/*-----------------------------------------------\n              @(target):= %s.so\n\
  ------------------------------------------------*/\n", module);

    Printf(f_header, "#if PY_VERSION_HEX >= 0x03000000\n");
    Printf(f_header, "#  define SWIG_init    PyInit_%s\n\n", module);
    Printf(f_header, "#else\n");
    Printf(f_header, "#  define SWIG_init    init%s\n\n", module);
    Printf(f_header, "#endif\n");
    Printf(f_header, "#define SWIG_name    \"%s\"\n", module);

    Printf(f_wrappers, "#ifdef __cplusplus\n");
    Printf(f_wrappers, "extern \"C\" {\n");
    Printf(f_wrappers, "#endif\n");
    Append(const_code, "static swig_const_info swig_const_table[] = {\n");
    Append(methods, "static PyMethodDef SwigMethods[] = {\n");
    Append(methods_proxydocs, "static PyMethodDef SwigMethods_proxydocs[] = {\n");

    /* the method exported for replacement of new.instancemethod in Python 3 */
    add_pyinstancemethod_new();
    add_pystaticmethod_new();

    if (builtin) {
      SwigType *s = NewString("SwigPyObject");
      SwigType_add_pointer(s);
      SwigType_remember(s);
      Delete(s);
    }

    /* emit code */
    Language::top(n);

    if (directorsEnabled()) {
      // Insert director runtime into the f_runtime file (make it occur before %header section)
      Swig_insert_file("director_common.swg", f_runtime);
      Swig_insert_file("director.swg", f_runtime);
    }

    /* Close language module */
    Append(methods, "\t { NULL, NULL, 0, NULL }\n");
    Append(methods, "};\n");
    Printf(f_wrappers, "%s\n", methods);
    Append(methods_proxydocs, "\t { NULL, NULL, 0, NULL }\n");
    Append(methods_proxydocs, "};\n");
    Printf(f_wrappers, "%s\n", methods_proxydocs);

    if (builtin) {
      Dump(f_builtins, f_wrappers);
    }

    SwigType_emit_type_table(f_runtime, f_wrappers);

    Append(const_code, "{0, 0, 0, 0.0, 0, 0}};\n");
    Printf(f_wrappers, "%s\n", const_code);
    initialize_threads(f_init);

    Printf(f_init, "#if PY_VERSION_HEX >= 0x03000000\n");
    Printf(f_init, "  return m;\n");
    Printf(f_init, "#else\n");
    Printf(f_init, "  return;\n");
    Printf(f_init, "#endif\n");
    Printf(f_init, "}\n");

    Printf(f_wrappers, "#ifdef __cplusplus\n");
    Printf(f_wrappers, "}\n");
    Printf(f_wrappers, "#endif\n");

    if (shadow) {
      Swig_banner_target_lang(f_shadow_py, "#");

      if (mod_docstring) {
	if (Len(mod_docstring)) {
	  const char *triple_double = "\"\"\"";
	  // follow PEP257 rules: https://www.python.org/dev/peps/pep-0257/
	  // reported by pep257: https://github.com/GreenSteam/pep257
	  bool multi_line_ds = Strchr(mod_docstring, '\n') != 0;
	  Printv(f_shadow_py, "\n", triple_double, multi_line_ds ? "\n":"", mod_docstring, multi_line_ds ? "\n":"", triple_double, "\n", NIL);
	}
	Delete(mod_docstring);
	mod_docstring = NULL;
      }

      if (Len(f_shadow_begin) > 0)
	Printv(f_shadow_py, "\n", f_shadow_begin, "\n", NIL);

      Printv(f_shadow_py, "\nfrom sys import version_info as _swig_python_version_info\n", NULL);
      Printv(f_shadow_py, "if _swig_python_version_info < (2, 7, 0):\n", NULL);
      Printv(f_shadow_py, tab4, "raise RuntimeError(\"Python 2.7 or later required\")\n\n", NULL);

      if (Len(f_shadow_after_begin) > 0)
	Printv(f_shadow_py, f_shadow_after_begin, "\n", NIL);

      if (moduleimport) {
	Replaceall(moduleimport, "$module", module);
	Printv(f_shadow_py, moduleimport, "\n", NIL);
      } else {
	Printv(f_shadow_py, default_import_code, NIL);
      }

      Printv(f_shadow_py, "\n", f_shadow, "\n", NIL);
      Printv(f_shadow_py, f_shadow_stubs, "\n", NIL);
      Delete(f_shadow_py);
    }

    /* Close all of the files */
    Dump(f_runtime, f_begin);
    Dump(f_header, f_begin);

    if (directorsEnabled()) {
      Dump(f_directors_h, f_runtime_h);
      Printf(f_runtime_h, "\n");
      Printf(f_runtime_h, "#endif\n");
      if (f_runtime_h != f_begin)
	Delete(f_runtime_h);
      Dump(f_directors, f_begin);
    }

    Dump(f_wrappers, f_begin);
    if (builtin && builtin_bases_needed)
      Printf(f_begin, "static PyTypeObject *builtin_bases[%d];\n\n", max_bases + 2);
    Wrapper_pretty_print(f_init, f_begin);

    Delete(default_import_code);
    Delete(f_shadow_after_begin);
    Delete(f_shadow_imports);
    Delete(f_shadow_begin);
    Delete(f_shadow);
    Delete(f_header);
    Delete(f_wrappers);
    Delete(f_builtins);
    Delete(f_init);
    Delete(f_directors);
    Delete(f_directors_h);
    Delete(f_runtime);
    Delete(f_begin);

    return SWIG_OK;
  }

  /* ------------------------------------------------------------
   * Emit the wrapper for PyInstanceMethod_New to MethodDef array.
   * This wrapper is used to implement -fastproxy,
   * as a replacement of new.instancemethod in Python 3.
   * ------------------------------------------------------------ */
  int add_pyinstancemethod_new() {
    String *name = NewString("SWIG_PyInstanceMethod_New");
    String *line = NewString("");
    Printf(line, "\t { \"%s\", %s, METH_O, NULL},\n", name, name);
    Append(methods, line);
    if (fastproxy) {
      Append(methods_proxydocs, line);
    }
    Delete(line);
    Delete(name);
    return 0;
  }

  /* ------------------------------------------------------------
   * Emit the wrapper for PyStaticMethod_New to MethodDef array.
   * This wrapper is used to ensure the correct documentation is
   * generated for static methods when using -fastproxy
   * ------------------------------------------------------------ */
  int add_pystaticmethod_new() {
    if (fastproxy) {
      String *name = NewString("SWIG_PyStaticMethod_New");
      String *line = NewString("");
      Printf(line, "\t { \"%s\", %s, METH_O, NULL},\n", name, name);
      Append(methods, line);
      Append(methods_proxydocs, line);
      Delete(line);
      Delete(name);
    }
    return 0;
  }

  /* ------------------------------------------------------------
   * subpkg_tail()
   *
   * Return the name of 'other' package relative to 'base'.
   *
   * 1. If 'other' is a sub-package of 'base', returns the 'other' relative to
   *    'base'.
   * 2. If 'other' and 'base' are equal, returns empty string "".
   * 3. In any other case, NULL pointer is returned.
   *
   * The 'base' and 'other' are expected to be fully qualified names.
   *
   * NOTE: none of 'base' nor 'other' can be null.
   *
   * Examples:
   *
   *  #  base       other         tail
   * --  ----       -----         ----
   *  1  "Foo"      "Foo.Bar" ->  "Bar"
   *  2	 "Foo"      "Foo."    ->  ""
   *  3	 "Foo"      "FooB.ar" ->  NULL
   *  4	 "Foo.Bar"  "Foo.Bar" ->  ""
   *  5  "Foo.Bar"  "Foo"     ->  NULL
   *  6  "Foo.Bar"  "Foo.Gez" ->  NULL
   *
   *  NOTE: the example #2 is actually a syntax error (at input). I believe
   *        swig parser prevents us from this case happening here.
   * ------------------------------------------------------------ */

  static String *subpkg_tail(const String *base, const String *other) {
    int baselen = Len(base);
    int otherlen = Len(other);

    if (Strncmp(other, base, baselen) == 0) {
      if ((baselen < otherlen) && (Char(other))[baselen] == '.') {
        return NewString((Char(other)) + baselen + 1);
      } else if (baselen == otherlen) {
        return NewString("");
      } else {
        return 0;
      }
    } else {
      return 0;
    }
  }

  /* ------------------------------------------------------------
   * abs_import_directive_string()
   *
   * Return a string containing python code to import module.
   *
   * 	pkg     package name or the module being imported
   * 	mod     module name of the module being imported
   * 	pfx     optional prefix to module name
   *
   * NOTE: keep this function consistent with abs_import_name_string().
   * ------------------------------------------------------------ */

  static String *abs_import_directive_string(const String *pkg, const String *mod, const char *pfx = "") {
    String *out = NewString("");

    if (pkg && *Char(pkg)) {
      Printf(out, "import %s.%s%s\n", pkg, pfx, mod);
    } else {
      Printf(out, "import %s%s\n", pfx, mod);
    }
    return out;
  }

  /* ------------------------------------------------------------
   * rel_import_directive_string()
   *
   * Return a string containing python code to import module that
   * is potentially within a package.
   *
   * 	mainpkg	package name of the module which imports the other module
   * 	pkg     package name or the module being imported
   * 	mod     module name of the module being imported
   * 	pfx     optional prefix to module name
   *
   * NOTE: keep this function consistent with rel_import_name_string().
   * ------------------------------------------------------------ */

  static String *rel_import_directive_string(const String *mainpkg, const String *pkg, const String *mod, const char *pfx = "") {

    /* NOTE: things are not so trivial. This is what we do here (by examples):
     *
     * 0. To import module 'foo', which is not in any package, we do absolute
     *    import:
     *
     *       import foo
     *
     * 1. To import 'pkg1.pkg2.foo', when mainpkg != "pkg1" and
     *    mainpkg != "pkg1.pkg2" or when mainpkg is not given we do absolute
     *    import:
     *
     *          import pkg1.pkg2.foo
     *
     * 2. To import module pkg1.foo, when mainpkg == "pkg1", we do:
     *
     *    - for py3 = 0:
     *
     *          import foo
     *
     *    - for py3 = 1:
     *
     *          from . import foo
     *
     * 3. To import "pkg1.pkg2.pkg3.foo", when mainpkg = "pkg1", we do:
     *
     *    - for py3 == 0:
     *
     *          import pkg2.pkg3.foo
     *
     *    - for py3 == 1:
     *
     *          from . import pkg2  # [1]
     *          import pkg1.pkg2.pkg3.foo
     *
     * NOTE: [1] is necessary for pkg2.foo to be present in the importing module
     */

    String *apkg = 0; // absolute (FQDN) package name of pkg
    String *rpkg = 0; // relative package name
    int py3_rlen1 = 0; // length of 1st level sub-package name, used by py3
    String *out = NewString("");

    if (pkg && *Char(pkg)) {
      if (mainpkg) {
	String *tail = subpkg_tail(mainpkg, pkg);
	if (tail) {
	  if (*Char(tail)) {
	    rpkg = NewString(tail);
	    const char *py3_end1 = Strchr(rpkg, '.');
	    if (!py3_end1)
	      py3_end1 = (Char(rpkg)) + Len(rpkg);
	    py3_rlen1 = (int)(py3_end1 - Char(rpkg));
	  } else {
	    rpkg = NewString("");
	  }
	  Delete(tail);
	} else {
	  apkg = NewString(pkg);
	}
      } else {
	apkg = NewString(pkg);
      }
    } else {
      apkg = NewString("");
    }

    if (apkg) {
      Printf(out, "import %s%s%s%s\n", apkg, *Char(apkg) ? "." : "", pfx, mod);
      Delete(apkg);
    } else {
      if (py3_rlen1)
	Printf(out, "from . import %.*s\n", py3_rlen1, rpkg);
      Printf(out, "from .%s import %s%s\n", rpkg, pfx, mod);
      Delete(rpkg);
    }
    return out;
  }

  /* ------------------------------------------------------------
   * import_directive_string()
   * ------------------------------------------------------------ */

  static String *import_directive_string(const String *mainpkg, const String *pkg, const String *mod, const char *pfx = "") {
    if (!relativeimport) {
      return abs_import_directive_string(pkg, mod, pfx);
    } else {
      return rel_import_directive_string(mainpkg, pkg, mod, pfx);
    }
  }

  /* ------------------------------------------------------------
   * abs_import_name_string()
   *
   * Return a string with the name of a symbol (perhaps imported
   * from external module by absolute import directive).
   *
   * mainpkg  package name of current module
   * mainmod  module name of current module
   * pkg      package name of (perhaps other) module
   * mod      module name of (perhaps other) module
   * sym      symbol name
   *
   * NOTE: mainmod, mod, and sym can't be NULL.
   * NOTE: keep this function consistent with abs_import_directive_string()
   * ------------------------------------------------------------ */

  static String *abs_import_name_string(const String *mainpkg, const String *mainmod, const String *pkg, const String *mod, const String *sym) {
    String *out = NewString("");
    if (pkg && *Char(pkg)) {
      if (mainpkg && *Char(mainpkg)) {
        if (Strcmp(mainpkg,pkg) != 0 || Strcmp(mainmod, mod) != 0) {
          Printf(out, "%s.%s.", pkg, mod);
        }
      } else {
        Printf(out, "%s.%s.", pkg, mod);
      }
    } else if ((mainpkg && *Char(mainpkg)) || Strcmp(mainmod, mod) != 0) {
      Printf(out, "%s.", mod);
    }
    Append(out, sym);
    return out;
  }

  /* ------------------------------------------------------------
   * rel_import_name_string()
   *
   * Return a string with the name of a symbol (perhaps imported
   * from external module by relative import directive).
   *
   * mainpkg  package name of current module
   * mainmod  module name of current module
   * pkg      package name of (perhaps other) module
   * mod      module name of (perhaps other) module
   * sym      symbol name
   *
   * NOTE: mainmod, mod, and sym can't be NULL.
   * NOTE: keep this function consistent with rel_import_directive_string()
   * ------------------------------------------------------------ */

  static String *rel_import_name_string(const String *mainpkg, const String *mainmod, const String *pkg, const String *mod, const String *sym) {
    String *out = NewString("");
    if (pkg && *Char(pkg)) {
      String *tail = 0;
      if (mainpkg)
        tail = subpkg_tail(mainpkg, pkg);
      if (!tail)
        tail = NewString(pkg);
      if (*Char(tail)) {
        Printf(out, "%s.%s.", tail, mod);
      } else if (Strcmp(mainmod, mod) != 0) {
        Printf(out, "%s.", mod);
      }
      Delete(tail);
    } else if ((mainpkg && *Char(mainpkg)) || Strcmp(mainmod, mod) != 0) {
      Printf(out, "%s.", mod);
    }
    Append(out, sym);
    return out;
  }

  /* ------------------------------------------------------------
   * import_name_string()
   * ------------------------------------------------------------ */

  static String *import_name_string(const String *mainpkg, const String *mainmod, const String *pkg, const String *mod, const String *sym) {
    if (!relativeimport) {
      return abs_import_name_string(mainpkg,mainmod,pkg,mod,sym);
    } else {
      return rel_import_name_string(mainpkg,mainmod,pkg,mod,sym);
    }
  }

  /* ------------------------------------------------------------
   * importDirective()
   * ------------------------------------------------------------ */

  virtual int importDirective(Node *n) {
    if (shadow) {
      String *modname = Getattr(n, "module");

      if (modname) {
	// Find the module node for this imported module.  It should be the
	// first child but search just in case.
	Node *mod = firstChild(n);
	while (mod && Strcmp(nodeType(mod), "module") != 0)
	  mod = nextSibling(mod);

	Node *options = Getattr(mod, "options");
	String *pkg = options ? Getattr(options, "package") : 0;

	if (!options || (!Getattr(options, "noshadow") && !Getattr(options, "noproxy"))) {
	  String *_import = import_directive_string(package, pkg, modname, "_");
	  if (!GetFlagAttr(f_shadow_imports, _import)) {
	    String *import = import_directive_string(package, pkg, modname);
	    Printf(builtin ? f_shadow_after_begin : f_shadow, "%s", import);
	    Delete(import);
	    SetFlag(f_shadow_imports, _import);
	  }
	  Delete(_import);
	}

      }
    }
    return Language::importDirective(n);
  }

  /* ------------------------------------------------------------
   * funcCall()
   *
   * Emit shadow code to call a function in the extension
   * module. Using proper argument and calling style for
   * given node n.
   * ------------------------------------------------------------ */
  String *funcCall(String *name, String *parms) {
    String *str = NewString("");

    Printv(str, module, ".", name, "(", parms, ")", NIL);
    return str;
  }

  /* ------------------------------------------------------------
   * indent_pythoncode()
   *
   * Format (indent) Python code.
   * Remove leading whitespace from 'code' and re-indent using
   * the indentation string in 'indent'.
   * ------------------------------------------------------------ */

  String *indent_pythoncode(const String *code, const_String_or_char_ptr indent, String *file, int line, const char *directive_name) {
    String *out = NewString("");
    String *temp;
    char *t;
    if (!indent)
      indent = "";

    temp = NewString(code);

    t = Char(temp);
    if (*t == '{') {
      Delitem(temp, 0);
      Delitem(temp, DOH_END);
    }

    /* Split the input text into lines */
    List *clist = SplitLines(temp);
    Delete(temp);

    // Line number within the pythoncode.
    int py_line = 0;

    String *initial = 0;
    Iterator si;

    /* Get the initial indentation.  Skip lines which only contain whitespace
     * and/or a comment, as the indentation of those doesn't matter:
     *
     *     A logical line that contains only spaces, tabs, formfeeds and
     *     possibly a comment, is ignored (i.e., no NEWLINE token is
     *     generated).
     *
     * see:
     * https://docs.python.org/2/reference/lexical_analysis.html#blank-lines
     * https://docs.python.org/3/reference/lexical_analysis.html#blank-lines
     */
    for (si = First(clist); si.item; si = Next(si), ++py_line) {
      const char *c = Char(si.item);
      int i;
      for (i = 0; isspace((unsigned char)c[i]); i++) {
	// Scan forward until we find a non-space (which may be a null byte).
      }
      char ch = c[i];
      if (ch && ch != '#') {
	// Found a line with actual content.
	initial = NewStringWithSize(c, i);
	break;
      }
      if (ch) {
	Printv(out, indent, c, NIL);
      }
      Putc('\n', out);
    }

    // Process remaining lines.
    for ( ; si.item; si = Next(si), ++py_line) {
      const char *c = Char(si.item);
      // If no prefixed line was found, the above loop should have completed.
      assert(initial);

      int i;
      for (i = 0; isspace((unsigned char)c[i]); i++) {
	// Scan forward until we find a non-space (which may be a null byte).
      }
      char ch = c[i];
      if (!ch) {
	// Line is just whitespace - emit an empty line.
	Putc('\n', out);
	continue;
      }

      if (ch == '#') {
	// Comment - the indentation doesn't matter to python, but try to
	// adjust the whitespace for the benefit of human readers (though SWIG
	// currently seems to always remove any whitespace before a '#' before
	// we get here, in which case we'll just leave the comment at the start
	// of the line).
	if (i >= Len(initial)) {
	  Printv(out, indent, NIL);
	}

	Printv(out, c + i, "\n", NIL);
	continue;
      }

      if (i < Len(initial)) {
	// There's non-whitespace in the initial prefix of this line.
	Swig_error(file, line, "Line indented less than expected (line %d of %s) as no line should be indented less than the indentation in line 1\n", py_line, directive_name);
	Printv(out, indent, c, "\n", NIL);
      } else {
	if (memcmp(c, Char(initial), Len(initial)) == 0) {
	  // Prefix matches initial, so just remove it.
	  Printv(out, indent, c + Len(initial), "\n", NIL);
	  continue;
	}
	Swig_warning(WARN_PYTHON_INDENT_MISMATCH,
		     file, line, "Whitespace indentation is inconsistent compared to earlier lines (line %d of %s)\n", py_line, directive_name);
	// To avoid gratuitously breaking interface files which worked with
	// SWIG <= 3.0.5, we remove a prefix of the same number of bytes for
	// lines which start with different whitespace to the line we got
	// 'initial' from.
	Printv(out, indent, c + Len(initial), "\n", NIL);
      }
    }
    Delete(clist);
    return out;
  }

  /* ------------------------------------------------------------
   * indent_docstring()
   *
   * Format (indent) a Python docstring.
   * Remove leading whitespace from 'code' and re-indent using
   * the indentation string in 'indent'.
   * ------------------------------------------------------------ */

  String *indent_docstring(const String *code, const_String_or_char_ptr indent) {
    String *out = NewString("");
    String *temp;
    char *t;
    if (!indent)
      indent = "";

    temp = NewString(code);

    t = Char(temp);
    if (*t == '{') {
      Delitem(temp, 0);
      Delitem(temp, DOH_END);
    }

    /* Split the input text into lines */
    List *clist = SplitLines(temp);
    Delete(temp);

    Iterator si;

    int truncate_characters_count = INT_MAX;
    for (si = First(clist); si.item; si = Next(si)) {
      const char *c = Char(si.item);
      int i;
      for (i = 0; isspace((unsigned char)c[i]); i++) {
	// Scan forward until we find a non-space (which may be a null byte).
      }
      char ch = c[i];
      if (ch) {
	// Found a line which isn't just whitespace
	if (i < truncate_characters_count)
	  truncate_characters_count = i;
      }
    }

    if (truncate_characters_count == INT_MAX)
      truncate_characters_count = 0;

    for (si = First(clist); si.item; si = Next(si)) {
      const char *c = Char(si.item);

      int i;
      for (i = 0; isspace((unsigned char)c[i]); i++) {
	// Scan forward until we find a non-space (which may be a null byte).
      }
      char ch = c[i];
      if (!ch) {
	// Line is just whitespace - emit an empty line.
	Putc('\n', out);
	continue;
      }

      Printv(out, indent, c + truncate_characters_count, "\n", NIL);
    }
    Delete(clist);
    return out;
  }

  /* ------------------------------------------------------------
   * autodoc level declarations
   * ------------------------------------------------------------ */

  enum autodoc_l {
    NO_AUTODOC = -2,		// no autodoc
    STRING_AUTODOC = -1,	// use provided string
    NAMES_AUTODOC = 0,		// only parameter names
    TYPES_AUTODOC = 1,		// parameter names and types
    EXTEND_AUTODOC = 2,		// extended documentation and parameter names
    EXTEND_TYPES_AUTODOC = 3	// extended documentation and parameter types + names
  };


  autodoc_l autodoc_level(String *autodoc) {
    autodoc_l dlevel = NO_AUTODOC;
    char *c = Char(autodoc);
    if (c) {
      if (isdigit(c[0])) {
	dlevel = (autodoc_l) atoi(c);
      } else {
	if (strcmp(c, "extended") == 0) {
	  dlevel = EXTEND_AUTODOC;
	} else {
	  dlevel = STRING_AUTODOC;
	}
      }
    }
    return dlevel;
  }


  /* ------------------------------------------------------------
   * have_docstring()
   *
   * Check if there is a docstring directive and it has text,
   * or there is an autodoc flag set
   * ------------------------------------------------------------ */

  bool have_docstring(Node *n) {
    String *str = Getattr(n, "feature:docstring");
    return ((str && Len(str) > 0)
	|| (Getattr(n, "feature:autodoc") && !GetFlag(n, "feature:noautodoc"))
	|| (doxygen && doxygenTranslator->hasDocumentation(n))
      );
  }

  /* ------------------------------------------------------------
   * build_combined_docstring()
   *
   * Build the full docstring which may be a combination of the
   * explicit docstring and autodoc string or, if none of them
   * is specified, obtained by translating Doxygen comment to
   * Python.
   *
   * Return new string to be deleted by caller (never NIL but
   * may be empty if there is no docstring).
   * ------------------------------------------------------------ */

  String *build_combined_docstring(Node *n, autodoc_t ad_type, const String *indent = "", bool low_level = false) {
    String *docstr = Getattr(n, "feature:docstring");
    if (docstr) {
      // Simplify the code below by just ignoring empty docstrings.
      if (!Len(docstr))
	docstr = NULL;
      else
	docstr = Copy(docstr);
    }

    if (docstr) {
      char *t = Char(docstr);
      if (*t == '{') {
	Delitem(docstr, 0);
	Delitem(docstr, DOH_END);
      }
    }

    if (Getattr(n, "feature:autodoc") && !GetFlag(n, "feature:noautodoc")) {
      String *autodoc = make_autodoc(n, ad_type, low_level);
      if (autodoc && Len(autodoc) > 0) {
	if (docstr) {
	  Append(autodoc, "\n");
	  Append(autodoc, docstr);
	}

	String *tmp = autodoc;
	autodoc = docstr;
	docstr = tmp;
      }

      Delete(autodoc);
    }

    if (!docstr) {
      if (doxygen) {
	docstr = Getattr(n, "python:docstring");
	if (!docstr && doxygenTranslator->hasDocumentation(n)) {
	  docstr = doxygenTranslator->getDocumentation(n, 0);

	  // Avoid rebuilding it again the next time: notice that we can't do
	  // this for the combined doc string as autodoc part of it depends on
	  // the sym:name of the node and it is changed while handling it, so
	  // the cached results become incorrect. But Doxygen docstring only
	  // depends on the comment which is not going to change, so we can
	  // safely cache it.
	  Setattr(n, "python:docstring", Copy(docstr));
	} else {
	  // Must copy here since if the docstring is multi-line, the String*
	  // here will get Deleted below, which is bad if it is a pointer to
	  // the cached object!
	  docstr = Copy(docstr);
	}
      }
    }

    if (!docstr)
      docstr = NewString("");

    // If there is more than one line then make docstrings like this:
    //
    //      """
    //      This is line1
    //      And here is line2 followed by the rest of them
    //      """
    //
    // otherwise, put it all on a single line
    if (Strchr(docstr, '\n')) {
      String *tmp = NewString("");
      Append(tmp, "\n");
      Append(tmp, indent_docstring(docstr, indent));
      Append(tmp, indent);
      Delete(docstr);
      docstr = tmp;
    }

    return docstr;
  }

  /* ------------------------------------------------------------
   * docstring()
   *
   * Get the docstring text, stripping off {} if necessary,
   * and enclose in triple double quotes.  If autodoc is also
   * set then it will build a combined docstring.
   * ------------------------------------------------------------ */

  String *docstring(Node *n, autodoc_t ad_type, const String *indent, bool low_level = false) {
    String *docstr = build_combined_docstring(n, ad_type, indent, low_level);
    const int len = Len(docstr);
    if (!len)
      return docstr;

    // Notice that all comments are created as raw strings (prefix "r"),
    // because '\' is used often in comments, but may break Python module from
    // loading. For example, in doxy comment one may write path in quotes:
    //
    //     This is path to file "C:\x\file.txt"
    //
    // Python will not load the module with such comment because of illegal
    // escape '\x'. '\' may additionally appear in verbatim or htmlonly sections
    // of doxygen doc, Latex expressions, ...
    String *doc = NewString("");

    // Determine which kind of quotes to use as delimiters: for single line
    // strings we can avoid problems with having a quote as the last character
    // of the docstring by using different kind of quotes as delimiters. For
    // multi-line strings this problem doesn't arise, as we always have a new
    // line or spaces at the end of it, but it still does no harm to do it for
    // them too.
    //
    // Note: we use double quotes by default, i.e. if there is no reason to
    // prefer using single ones, for consistency with the older SWIG versions.
    const bool useSingleQuotes = (Char(docstr))[len - 1] == '"';

    Append(doc, useSingleQuotes ? "r'''" : "r\"\"\"");

    // We also need to avoid having triple quotes of whichever type we use, as
    // this would break Python doc string syntax too. Unfortunately there is no
    // way to have triple quotes inside of raw-triple-quoted string, so we have
    // to break the string in parts and rely on concatenation of the adjacent
    // string literals.
    if (useSingleQuotes)
      Replaceall(docstr, "'''", "''' \"'''\" '''");
    else
      Replaceall(docstr, "\"\"\"", "\"\"\" '\"\"\"' \"\"\"");

    Append(doc, docstr);
    Append(doc, useSingleQuotes ? "'''" : "\"\"\"");
    Delete(docstr);

    return doc;
  }

  /* ------------------------------------------------------------
   * cdocstring()
   *
   * Get the docstring text as it would appear in C-language
   * source code (but without quotes around it).
   * ------------------------------------------------------------ */

  String *cdocstring(Node *n, autodoc_t ad_type, bool low_level = false) {
    String *ds = build_combined_docstring(n, ad_type, "", low_level);
    Replaceall(ds, "\\", "\\\\");
    Replaceall(ds, "\"", "\\\"");
    Replaceall(ds, "\n", "\\n\"\n\t\t\"");
    return ds;
  }

  /* -----------------------------------------------------------------------------
   * addMissingParameterNames()
   *
   * For functions that have not had nameless parameters set in the Language class.
   *
   * Inputs:
   *   plist - entire parameter list
   *   arg_num - the number to start from when naming arguments
   * Side effects:
   *   The "lname" attribute in each parameter in plist will be contain a parameter name
   * ----------------------------------------------------------------------------- */

  void addMissingParameterNames(Node *n, ParmList *plist, int arg_num) {
    Parm *p = plist;
    int i = arg_num;
    while (p) {
      if (!Getattr(p, "lname")) {
	String *name = makeParameterName(n, p, i);
	Setattr(p, "lname", name);
	Delete(name);
      }
      i++;
      p = nextSibling(p);
    }
  }

  /* ------------------------------------------------------------
   * make_autodocParmList()
   *
   * Generate the documentation for the function parameters
   * Parameters:
   *    arg_num:         The number to start assigning unnamed arguments from
   *    func_annotation: Function annotation support
   * ------------------------------------------------------------ */

  String *make_autodocParmList(Node *n, bool showTypes, int arg_num = 1, bool calling = false, bool func_annotation = false) {

    String *doc = NewString("");
    String *pdocs = 0;
    ParmList *plist = CopyParmList(Getattr(n, "parms"));
    Parm *p;
    Parm *pnext;

    if (calling)
      func_annotation = false;

    addMissingParameterNames(n, plist, arg_num); // for $1_name substitutions done in Swig_typemap_attach_parms
    Swig_typemap_attach_parms("in", plist, 0);
    Swig_typemap_attach_parms("doc", plist, 0);

    if (Strcmp(ParmList_protostr(plist), "void") == 0) {
      //No parameters actually
      return doc;
    }

    for (p = plist; p; p = pnext) {
      String *tm = Getattr(p, "tmap:in");
      if (tm) {
	pnext = Getattr(p, "tmap:in:next");
	if (checkAttribute(p, "tmap:in:numinputs", "0")) {
	  continue;
	}
      } else {
	pnext = nextSibling(p);
      }

      String *name = 0;
      String *type = 0;
      String *value = 0;
      String *pdoc = Getattr(p, "tmap:doc");
      if (pdoc) {
	name = Getattr(p, "tmap:doc:name");
	type = Getattr(p, "tmap:doc:type");
	value = Getattr(p, "tmap:doc:value");
      }

      // Skip the "self" argument - it is added to the parameter list automatically
      // and shouldn't be included in the Parameters block
      if (Getattr(p, "self")) {
	continue;
      }

      // Note: the generated name should be consistent with that in kwnames[]
      String *made_name = 0;
      if (!name) {
	name = made_name = makeParameterName(n, p, arg_num);
      }

      // Increment the argument number once we are sure this is a real argument to count
      arg_num++;

      type = type ? type : Getattr(p, "type");
      value = value ? value : Getattr(p, "value");

      if (SwigType_isvarargs(type)) {
	Delete(made_name);
	break;
      }

      if (Len(doc)) {
	// add a comma to the previous one if any
	Append(doc, ", ");
      }

      // Do the param type too?
      Node *nn = classLookup(Getattr(p, "type"));
      String *type_str = nn ? Copy(Getattr(nn, "sym:name")) : SwigType_str(type, 0);
      if (showTypes)
	Printf(doc, "%s ", type_str);

      Append(doc, name);
      if (pdoc) {
	if (!pdocs)
	  // numpydoc style: https://github.com/numpy/numpy/blob/master/doc/HOWTO_DOCUMENT.rst.txt
	  pdocs = NewString("\nParameters\n----------\n");
	Printf(pdocs, "%s\n", pdoc);
      }
      // Write the function annotation
      if (func_annotation)
	Printf(doc, ": \"%s\"", type_str);

      // Write default value
      if (value && !calling) {
	String *new_value = convertValue(value, Getattr(p, "type"));
	if (new_value) {
	  value = new_value;
	} else {
	  // Even if the value is not representable in the target language, still use it in the documentation, for compatibility with the previous SWIG versions
	  // and because it can still be useful to see the C++ expression there.
	  Node *lookup = Swig_symbol_clookup(value, 0);
	  if (lookup)
	    value = Getattr(lookup, "sym:name");
	}
	Printf(doc, "=%s", value);

	if (new_value)
	  Delete(new_value);
      }
      Delete(type_str);
      Delete(made_name);
    }
    if (pdocs)
      Setattr(n, "feature:pdocs", pdocs);
    Delete(plist);
    return doc;
  }

  /* ------------------------------------------------------------
   * make_autodoc()
   *
   * Build a docstring for the node, using parameter and other
   * info in the parse tree.  If the value of the autodoc
   * attribute is "0" then do not include parameter types, if
   * it is "1" (the default) then do.  If it has some other
   * value then assume it is supplied by the extension writer
   * and use it directly.
   * ------------------------------------------------------------ */

  String *make_autodoc(Node *n, autodoc_t ad_type, bool low_level = false) {
    int extended = 0;
    bool first_func = true;
    // If the function is overloaded then this function is called
    // for the last one.  Rewind to the first so the docstrings are
    // in order.
    while (Getattr(n, "sym:previousSibling"))
      n = Getattr(n, "sym:previousSibling");

    String *doc = NewString("");
    while (n) {
      bool showTypes = false;
      bool skipAuto = false;
      String *autodoc = Getattr(n, "feature:autodoc");
      autodoc_l dlevel = autodoc_level(autodoc);
      switch (dlevel) {
      case NO_AUTODOC:
	break;
      case NAMES_AUTODOC:
	showTypes = false;
	break;
      case TYPES_AUTODOC:
	showTypes = true;
	break;
      case EXTEND_AUTODOC:
	extended = 1;
	showTypes = false;
	break;
      case EXTEND_TYPES_AUTODOC:
	extended = 1;
	showTypes = true;
	break;
      case STRING_AUTODOC:
	Append(doc, autodoc);
	skipAuto = true;
	break;
      }

      if (!skipAuto) {
	/* Check if a documentation name was given for either the low-level C API or high-level Python shadow API */
	String *symname = Getattr(n, low_level ? "doc:low:name" : "doc:high:name");
	if (!symname) {
	  symname = Getattr(n, "sym:name");
	}

	SwigType *type = Getattr(n, "type");
	String *type_str = NULL;

	// If the function has default arguments, then that documentation covers this version too
	if (Getattr(n, "defaultargs") != NULL) {
	  n = Getattr(n, "sym:nextSibling");
	  continue;
	}

	if (!first_func)
	  Append(doc, "\n");

	if (type) {
	  if (Strcmp(type, "void") == 0) {
	    type_str = NULL;
	  } else {
	    Node *nn = classLookup(type);
	    type_str = nn ? Copy(Getattr(nn, "sym:name")) : SwigType_str(type, 0);
	  }
	}

	/* Treat the low-level C API functions for getting/setting variables as methods for documentation purposes */
	String *kind = Getattr(n, "kind");
	if (kind && Strcmp(kind, "variable") == 0) {
	  if (ad_type == AUTODOC_FUNC) {
	    ad_type = AUTODOC_METHOD;
	  }
	}
	/* Treat destructors as methods for documentation purposes */
	String *nodeType = Getattr(n, "nodeType");
	if (nodeType && Strcmp(nodeType, "destructor") == 0) {
	  if (ad_type == AUTODOC_FUNC) {
	    ad_type = AUTODOC_METHOD;
	  }
	}

	switch (ad_type) {
	case AUTODOC_CLASS:
	  {
	    // Only do the autodoc if there isn't a docstring for the class
	    String *str = Getattr(n, "feature:docstring");
	    if (!str || Len(str) == 0) {
	      if (builtin) {
		String *name = Getattr(n, "name");
		String *rname = add_explicit_scope(SwigType_namestr(name));
		Printf(doc, "%s", rname);
		Delete(rname);
	      } else {
		if (CPlusPlus) {
		  Printf(doc, "Proxy of C++ %s class.", SwigType_namestr(real_classname));
		} else {
		  Printf(doc, "Proxy of C %s struct.", SwigType_namestr(real_classname));
		}
	      }
	    }
	  }
	  break;
	case AUTODOC_CTOR:
	  if (Strcmp(class_name, symname) == 0) {
	    String *paramList = make_autodocParmList(n, showTypes, 2);
	    Printf(doc, "__init__(");
	    if (showTypes)
	      Printf(doc, "%s ", class_name);
	    if (Len(paramList))
	      Printf(doc, "self, %s) -> %s", paramList, class_name);
	    else
	      Printf(doc, "self) -> %s", class_name);
	  } else
	    Printf(doc, "%s(%s) -> %s", symname, make_autodocParmList(n, showTypes), class_name);
	  break;

	case AUTODOC_DTOR:
	  if (showTypes)
	    Printf(doc, "__del__(%s self)", class_name);
	  else
	    Printf(doc, "__del__(self)");
	  break;

	case AUTODOC_STATICFUNC:
	  Printf(doc, "%s(%s)", symname, make_autodocParmList(n, showTypes));
	  if (type_str)
	    Printf(doc, " -> %s", type_str);
	  break;

	case AUTODOC_FUNC:
	  Printf(doc, "%s(%s)", symname, make_autodocParmList(n, showTypes));
	  if (type_str)
	    Printf(doc, " -> %s", type_str);
	  break;

	case AUTODOC_METHOD:
	  {
	    String *paramList = make_autodocParmList(n, showTypes, 2);
	    Printf(doc, "%s(", symname);
	    if (showTypes)
	      Printf(doc, "%s ", class_name);
	    if (Len(paramList))
	      Printf(doc, "self, %s)", paramList);
	    else
	      Printf(doc, "self)");
	    if (type_str)
	      Printf(doc, " -> %s", type_str);
	  }
	  break;

	case AUTODOC_CONST:
	  // There is no autodoc support for constants currently, this enum
	  // element only exists to allow calling docstring() with it.
	  return NULL;
	case AUTODOC_VAR:
	  // Variables can also be documented (e.g. through the property() function in python)
	  Printf(doc, "%s", symname);
	  if (showTypes) {
	    String *type = Getattr(n, "tmap:doc:type");
	    if (!type)
	      type = Getattr(n, "membervariableHandler:type");
	    if (!type)
	      type = Getattr(n, "type");
	    Printf(doc, " : %s", type);
	  }
	  break;
	}
	Delete(type_str);

	// Special case: wrapper functions to get a variable should have no parameters.
	// Because the node is re-used for the setter and getter, the feature:pdocs field will
	// exist for the getter function, so explicitly avoid printing parameters in this case.
	bool variable_getter = kind && Strcmp(kind, "variable") == 0 && Getattr(n, "memberget");
	if (extended && ad_type != AUTODOC_VAR && !variable_getter) {
	  String *pdocs = Getattr(n, "feature:pdocs");
	  if (pdocs) {
	    Printv(doc, "\n", pdocs, NULL);
	  }
	}
      }
      // if it's overloaded then get the next decl and loop around again
      n = Getattr(n, "sym:nextSibling");
      if (n)
	first_func = false;
    }

    return doc;
  }

  /* ------------------------------------------------------------
   * convertIntegerValue()
   *
   * Check if string v is an integer and can be represented in
   * Python. If so, return an appropriate Python representation,
   * otherwise (or if we are unsure), return NIL.
   * ------------------------------------------------------------ */
  String *convertIntegerValue(String *v, SwigType *resolved_type) {
    const char *const s = Char(v);
    char *end;
    String *result = NIL;

    // Check if this is an integer number in any base.
    errno = 0;
    long value = strtol(s, &end, 0);
    if (errno == ERANGE || end == s)
      return NIL;

    if (*end != '\0') {
      // If there is a suffix after the number, we can safely ignore "l"
      // and (provided the number is unsigned) "u", and also combinations of
      // these, but not anything else.
      for (char *p = end; *p != '\0'; ++p) {
        switch (*p) {
          case 'l':
          case 'L':
	    break;
          case 'u':
          case 'U':
	    if (value < 0)
	      return NIL;
            break;
          default:
            return NIL;
        }
      }
    }
    // So now we are certain that we are indeed dealing with an integer
    // that has a representation as long given by value.

    // Restrict to guaranteed supported range in Python, see maxint docs: https://docs.python.org/2/library/sys.html#sys.maxint
    // Don't do this pointless check when long is 32 bits or smaller as strtol will have already failed with ERANGE
#if LONG_MAX > PYTHON_INT_MAX || LONG_MIN < PYTHON_INT_MIN
    if (value > PYTHON_INT_MAX || value < PYTHON_INT_MIN) {
      return NIL;
    }
#endif

    if (Cmp(resolved_type, "bool") == 0)
      // Allow integers as the default value for a bool parameter.
      return NewString(value ? "True" : "False");

    if (value == 0)
      return NewString(SwigType_ispointer(resolved_type) ? "None" : "0");

    // v may still be octal or hexadecimal:
    const char *p = s;
    if (*p == '+' || *p == '-')
      ++p;
    if (*p == '0' && *(p+1) != 'x' && *(p+1) != 'X') {
      // This must have been an octal number. This is the only case we
      // cannot use in Python directly, since Python 2 and 3 use non-
      // compatible representations.
      result = NewString(*s == '-' ? "int(\"-" : "int(\"");
      String *octal_string = NewStringWithSize(p, (int) (end - p));
      Append(result, octal_string);
      Append(result, "\", 8)");
      Delete(octal_string);
      return result;
    }
    result = *end == '\0' ? Copy(v) : NewStringWithSize(s, (int) (end - s));
    return result;
  }

  /* ------------------------------------------------------------
   * convertDoubleValue()
   *
   * Check if the given string looks like a decimal floating point constant
   * and return it if it does, otherwise return NIL.
   * ------------------------------------------------------------ */
  String *convertDoubleValue(String *v) {
    const char *const s = Char(v);
    char *end;

    errno = 0;
    double value = strtod(s, &end);
    (void) value;
    if (errno != ERANGE && end != s) {
      // An added complication: at least some versions of strtod() recognize
      // hexadecimal floating point numbers which don't exist in Python, so
      // detect them ourselves and refuse to convert them (this can't be done
      // without loss of precision in general).
      //
      // Also don't accept neither "NAN" nor "INFINITY" (both of which
      // conveniently contain "n").
      if (strpbrk(s, "xXnN"))
	return NIL;

      // Disregard optional "f" suffix, it can be just dropped in Python as it
      // uses doubles for everything anyhow.
      for (char * p = end; *p != '\0'; ++p) {
	switch (*p) {
	  case 'f':
	  case 'F':
	    break;

	  default:
	    return NIL;
	}
      }

      // Avoid unnecessary string allocation in the common case when we don't
      // need to remove any suffix.
      return *end == '\0' ? Copy(v) : NewStringWithSize(s, (int)(end - s));
    }

    return NIL;
  }

  /* ------------------------------------------------------------
   * convertValue()
   *
   * Check if string v can be a Python value literal or a
   * constant. Return an equivalent Python representation,
   * or NIL if it isn't, or we are unsure.
   * ------------------------------------------------------------ */
  String *convertValue(String *v, SwigType *type) {
    const char *const s = Char(v);
    String *result = NIL;
    SwigType *resolved_type = SwigType_typedef_resolve_all(type);

    result = convertIntegerValue(v, resolved_type);
    if (!result) {
      result = convertDoubleValue(v);
      if (!result) {
	if (Strcmp(v, "true") == 0)
	  result = NewString("True");
	else if (Strcmp(v, "false") == 0)
	  result = NewString("False");
	else if (Strcmp(v, "NULL") == 0 || Strcmp(v, "nullptr") == 0)
	  result = SwigType_ispointer(resolved_type) ? NewString("None") : NewString("0");
	// This could also be an enum type, default value of which could be
	// representable in Python if it doesn't include any scope (which could,
	// but currently is not, translated).
	else if (!Strchr(s, ':')) {
	  Node *lookup = Swig_symbol_clookup(v, 0);
	  if (lookup) {
	    if (Cmp(Getattr(lookup, "nodeType"), "enumitem") == 0)
	      result = Copy(Getattr(lookup, "sym:name"));
	  }
	}
      }
    }

    Delete(resolved_type);
    return result;
  }

  /* ------------------------------------------------------------
   * is_representable_as_pyargs()
   *
   * Check if the function parameters default argument values
   * can be represented in Python.
   *
   * If this method returns false, the parameters will be translated
   * to a generic "*args" which allows us to deal with default values
   * at C++ code level where they can always be handled.
   * ------------------------------------------------------------ */
  bool is_representable_as_pyargs(Node *n) {
    ParmList *plist = CopyParmList(Getattr(n, "parms"));
    Swig_typemap_attach_parms("default", plist, NULL);

    Parm *p;
    Parm *pnext;

    for (p = plist; p; p = pnext) {
      pnext = nextSibling(p);
      String *tm = Getattr(p, "tmap:in");
      if (tm) {
	Parm *in_next = Getattr(p, "tmap:in:next");
	if (in_next)
	  pnext = in_next;
	if (checkAttribute(p, "tmap:in:numinputs", "0")) {
	  continue;
	}
      }

      // "default" typemap can contain arbitrary C++ code, so while it could, in
      // principle, be possible to examine it and check if it's just something
      // simple of the form "$1 = expression" and then use convertValue() to
      // check if expression can be used in Python, but for now we just
      // pessimistically give up and prefer to handle this at C++ level only.
      if (Getattr(p, "tmap:default"))
	return false;

      String *value = Getattr(p, "value");
      if (value) {
	String *convertedValue = convertValue(value, Getattr(p, "type"));
	if (!convertedValue)
	  return false;
	Delete(convertedValue);
      }
    }

    return true;
  }


  /* ------------------------------------------------------------
   * is_real_overloaded()
   *
   * Check if the function is overloaded, but not just have some
   * siblings generated due to the original function having
   * default arguments.
   * ------------------------------------------------------------ */
  bool is_real_overloaded(Node *n) {
    Node *h = Getattr(n, "sym:overloaded");
    Node *i;
    if (!h)
      return false;

    i = Getattr(h, "sym:nextSibling");
    while (i) {
      Node *nn = Getattr(i, "defaultargs");
      if (nn != h) {
	/* Check if overloaded function has defaultargs and
	 * pointed to the first overloaded. */
	return true;
      }
      i = Getattr(i, "sym:nextSibling");
    }

    return false;
  }

  /* ------------------------------------------------------------
   * make_pyParmList()
   *
   * Generate parameter list for Python functions or methods,
   * reuse make_autodocParmList() to do so.
   * ------------------------------------------------------------ */
  String *make_pyParmList(Node *n, bool in_class, bool is_calling, int kw, bool has_self_for_count = false) {
    /* Get the original function for a defaultargs copy,
     * see default_arguments() in parser.y. */
    Node *nn = Getattr(n, "defaultargs");
    if (nn)
      n = nn;

    Parm *parms = Getattr(n, "parms");
    int varargs = parms ? emit_isvarargs(parms) : 0;

    /* We prefer to explicitly list all parameters of the C function in the
       generated Python code as this makes the function more convenient to use,
       however in some cases we must replace the real parameters list with just
       the catch all "*args". This happens when:

	1. The function is overloaded as Python doesn't support this.
	2. We were explicitly asked to use the "compact" arguments form.
	3. We were explicitly asked to use default args from C via the "python:cdefaultargs" feature.
	4. One of the default argument values can't be represented in Python.
	5. Varargs that haven't been forced to use a fixed number of arguments with %varargs.
     */
    if (is_real_overloaded(n) || GetFlag(n, "feature:compactdefaultargs") || GetFlag(n, "feature:python:cdefaultargs") || !is_representable_as_pyargs(n) || varargs) {
      String *parms = NewString("");
      if (in_class)
	Printf(parms, "self, ");
      Printf(parms, "*args");
      if (kw)
	Printf(parms, ", **kwargs");
      return parms;
    }

    bool funcanno = py3 ? true : false;
    String *params = NewString("");
    String *_params = make_autodocParmList(n, false, ((in_class || has_self_for_count)? 2 : 1), is_calling, funcanno);

    if (in_class) {
      Printf(params, "self");
      if (Len(_params) > 0)
	Printf(params, ", ");
    }

    Printv(params, _params, NULL);

    return params;
  }

  /* ------------------------------------------------------------
   * have_pythonprepend()
   *
   * Check if there is a %pythonprepend directive and it has text
   * ------------------------------------------------------------ */

  bool have_pythonprepend(Node *n) {
    String *str = Getattr(n, "feature:pythonprepend");
    return (str && Len(str) > 0);
  }

  /* ------------------------------------------------------------
   * pythonprepend()
   *
   * Get the %pythonprepend code, stripping off {} if necessary
   * ------------------------------------------------------------ */

  String *pythonprepend(Node *n) {
    String *str = Getattr(n, "feature:pythonprepend");
    char *t = Char(str);
    if (*t == '{') {
      Delitem(str, 0);
      Delitem(str, DOH_END);
    }
    return str;
  }

  /* ------------------------------------------------------------
   * have_pythonappend()
   *
   * Check if there is a %pythonappend directive and it has text
   * ------------------------------------------------------------ */

  bool have_pythonappend(Node *n) {
    String *str = Getattr(n, "feature:pythonappend");
    if (!str)
      str = Getattr(n, "feature:addtofunc");
    return (str && Len(str) > 0);
  }

  /* ------------------------------------------------------------
   * pythonappend()
   *
   * Get the %pythonappend code, stripping off {} if necessary
   * ------------------------------------------------------------ */

  String *pythonappend(Node *n) {
    String *str = Getattr(n, "feature:pythonappend");
    if (!str)
      str = Getattr(n, "feature:addtofunc");

    char *t = Char(str);
    if (*t == '{') {
      Delitem(str, 0);
      Delitem(str, DOH_END);
    }
    return str;
  }

  /* ------------------------------------------------------------
   * have_addtofunc()
   *
   * Check if there is a %addtofunc directive and it has text
   * ------------------------------------------------------------ */

  bool have_addtofunc(Node *n) {
    return have_pythonappend(n) || have_pythonprepend(n);
  }


  /* ------------------------------------------------------------
   * returnTypeAnnotation()
   *
   * Helper function for constructing the function annotation
   * of the returning type, return a empty string for Python 2.x
   * ------------------------------------------------------------ */
  String *returnTypeAnnotation(Node *n) {
    String *ret = 0;
    Parm *p = Getattr(n, "parms");
    String *tm;
    /* Try to guess the returning type by argout typemap,
     * however the result may not accurate. */
    while (p) {
      if ((tm = Getattr(p, "tmap:argout:match_type"))) {
	tm = SwigType_str(tm, 0);
	if (ret)
	  Printv(ret, ", ", tm, NULL);
	else
	  ret = tm;
	p = Getattr(p, "tmap:argout:next");
      } else {
	p = nextSibling(p);
      }
    }
    /* If no argout typemap, then get the returning type from
     * the function prototype. */
    if (!ret) {
      ret = Getattr(n, "type");
      if (ret)
	ret = SwigType_str(ret, 0);
    }
    return (ret && py3) ? NewStringf(" -> \"%s\"", ret)
	: NewString("");
  }

  /* ------------------------------------------------------------
   * emitFunctionShadowHelper()
   *
   * Refactoring some common code out of functionWrapper and
   * dispatchFunction that writes the proxy code for non-member
   * functions.
   * ------------------------------------------------------------ */

  void emitFunctionShadowHelper(Node *n, File *f_dest, String *name, int kw) {
    String *parms = make_pyParmList(n, false, false, kw);
    String *callParms = make_pyParmList(n, false, true, kw);

    // Callbacks need the C function in order to extract the pointer from the swig_ptr: string
    bool fast = (fastproxy && !have_addtofunc(n)) || Getattr(n, "feature:callback");

    if (!fast || olddefs) {
      /* Make a wrapper function to insert the code into */
      Printv(f_dest, "\n", "def ", name, "(", parms, ")", returnTypeAnnotation(n), ":\n", NIL);
      if (have_docstring(n))
	Printv(f_dest, tab4, docstring(n, AUTODOC_FUNC, tab4, true), "\n", NIL);
      if (have_pythonprepend(n))
	Printv(f_dest, indent_pythoncode(pythonprepend(n), tab4, Getfile(n), Getline(n), "%pythonprepend or %feature(\"pythonprepend\")"), "\n", NIL);
      if (have_pythonappend(n)) {
	Printv(f_dest, tab4 "val = ", funcCall(name, callParms), "\n", NIL);
	Printv(f_dest, indent_pythoncode(pythonappend(n), tab4, Getfile(n), Getline(n), "%pythonappend or %feature(\"pythonappend\")"), "\n", NIL);
	Printv(f_dest, tab4 "return val\n", NIL);
      } else {
	Printv(f_dest, tab4 "return ", funcCall(name, callParms), "\n", NIL);
      }
    }

    // Below may result in a 2nd definition of the method when -olddefs is used. The Python interpreter will use the second definition as it overwrites the first.
    if (fast) {
      /* If there is no addtofunc directive then just assign from the extension module (for speed up) */
      Printv(f_dest, name, " = ", module, ".", name, "\n", NIL);
    }
  }


  /* ------------------------------------------------------------
   * check_kwargs()
   *
   * check if using kwargs is allowed for this Node
   * ------------------------------------------------------------ */

  int check_kwargs(Node *n) const {
    return (use_kw || GetFlag(n, "feature:kwargs"))
	&& !GetFlag(n, "memberset") && !GetFlag(n, "memberget");
  }



  /* ------------------------------------------------------------
   * add_method()
   * ------------------------------------------------------------ */

  void add_method(String *name, String *function, int kw, Node *n = 0, int funpack = 0, int num_required = -1, int num_arguments = -1) {
    String * meth_str = NewString("");
    if (!kw) {
      if (funpack) {
	if (num_required == 0 && num_arguments == 0) {
	  Printf(meth_str, "\t { \"%s\", %s, METH_NOARGS, ", name, function);
	} else if (num_required == 1 && num_arguments == 1) {
	  Printf(meth_str, "\t { \"%s\", %s, METH_O, ", name, function);
	} else {
	  Printf(meth_str, "\t { \"%s\", %s, METH_VARARGS, ", name, function);
	}
      } else {
	Printf(meth_str, "\t { \"%s\", %s, METH_VARARGS, ", name, function);
      }
    } else {
      // Cast via void(*)(void) to suppress GCC -Wcast-function-type warning.
      // Python should always call the function correctly, but the Python C API
      // requires us to store it in function pointer of a different type.
      Printf(meth_str, "\t { \"%s\", (PyCFunction)(void(*)(void))%s, METH_VARARGS|METH_KEYWORDS, ", name, function);
    }
    Append(methods, meth_str);
    if (fastproxy) {
      Append(methods_proxydocs, meth_str);
    }
    Delete(meth_str);

    if (!n) {
      Append(methods, "NULL");
      if (fastproxy) {
	Append(methods_proxydocs, "NULL");
      }
    } else if (have_docstring(n)) {
      /* Use the low-level docstring here since this is the docstring that will be used for the C API */
      String *ds = cdocstring(n, Getattr(n, "memberfunction") ? AUTODOC_METHOD : AUTODOC_FUNC, true);
      Printf(methods, "\"%s\"", ds);
      if (fastproxy) {
	/* In the fastproxy case, we must also record the high-level docstring for use in the Python shadow API */
	Delete(ds);
        ds = cdocstring(n, Getattr(n, "memberfunction") ? AUTODOC_METHOD : AUTODOC_FUNC);
	Printf(methods_proxydocs, "\"%s\"", ds);
      }
      Delete(ds);
    } else if (Getattr(n, "feature:callback")) {
      Printf(methods, "\"swig_ptr: %s\"", Getattr(n, "feature:callback:name"));
      if (fastproxy) {
	Printf(methods_proxydocs, "\"swig_ptr: %s\"", Getattr(n, "feature:callback:name"));
      }
    } else {
      Append(methods, "NULL");
      if (fastproxy) {
	Append(methods_proxydocs, "NULL");
      }
    }

    Append(methods, "},\n");
    if (fastproxy) {
      Append(methods_proxydocs, "},\n");
    }
  }

  /* ------------------------------------------------------------
   * dispatchFunction()
   * ------------------------------------------------------------ */
  void dispatchFunction(Node *n, String *linkage, int funpack = 0, bool builtin_self = false, bool builtin_ctor = false, bool director_class = false) {
    /* Last node in overloaded chain */

    bool add_self = builtin_self && (!builtin_ctor || director_class);

    int maxargs;

    String *tmp = NewString("");
    String *dispatch;

    const char *dispatch_call = funpack ? "%s(self, argc, argv);" : (builtin_ctor ? "%s(self, args, NULL);" : "%s(self, args);");
    String *dispatch_code = NewStringf("return %s", dispatch_call);

    if (castmode) {
      dispatch = Swig_overload_dispatch_cast(n, dispatch_code, &maxargs);
    } else {
      String *fastdispatch_code;
      if (builtin_ctor)
	fastdispatch_code = NewStringf("int retval = %s\nif (retval == 0 || !SWIG_Python_TypeErrorOccurred(NULL)) return retval;\nSWIG_fail;", dispatch_call);
      else
	fastdispatch_code = NewStringf("PyObject *retobj = %s\nif (!SWIG_Python_TypeErrorOccurred(retobj)) return retobj;\nSWIG_fail;", dispatch_call);
      if (!CPlusPlus) {
	Insert(fastdispatch_code, 0, "{\n");
	Append(fastdispatch_code, "\n}");
      }
      dispatch = Swig_overload_dispatch(n, dispatch_code, &maxargs, fastdispatch_code);
      Delete(fastdispatch_code);
    }

    /* Generate a dispatch wrapper for all overloaded functions */

    Wrapper *f = NewWrapper();
    String *symname = Getattr(n, "sym:name");
    String *wname = Swig_name_wrapper(symname);

    const char *builtin_kwargs = builtin_ctor ? ", PyObject *kwargs" : "";
    Printv(f->def, linkage, builtin_ctor ? "int " : "PyObject *", wname, "(PyObject *self, PyObject *args", builtin_kwargs, ") {", NIL);

    Wrapper_add_local(f, "argc", "Py_ssize_t argc");
    Printf(tmp, "PyObject *argv[%d] = {0}", maxargs + 1);
    Wrapper_add_local(f, "argv", tmp);

    if (!fastunpack) {
      Wrapper_add_local(f, "ii", "Py_ssize_t ii");

      if (builtin_ctor)
	Printf(f->code, "if (!SWIG_Python_CheckNoKeywords(kwargs, \"%s\")) SWIG_fail;\n", symname);

      if (maxargs - (add_self ? 1 : 0) > 0) {
        Append(f->code, "if (!PyTuple_Check(args)) SWIG_fail;\n");
        Append(f->code, "argc = PyObject_Length(args);\n");
      } else {
        Append(f->code, "argc = args ? PyObject_Length(args) : 0;\n");
      }

      if (add_self)
	Append(f->code, "argv[0] = self;\n");
      Printf(f->code, "for (ii = 0; (ii < %d) && (ii < argc); ii++) {\n", add_self ? maxargs - 1 : maxargs);
      Printf(f->code, "argv[ii%s] = PyTuple_GET_ITEM(args,ii);\n", add_self ? " + 1" : "");
      Append(f->code, "}\n");
      if (add_self)
	Append(f->code, "argc++;\n");
    } else {
      if (builtin_ctor)
	Printf(f->code, "if (!SWIG_Python_CheckNoKeywords(kwargs, \"%s\")) SWIG_fail;\n", symname);
      Printf(f->code, "if (!(argc = SWIG_Python_UnpackTuple(args, \"%s\", 0, %d, argv%s))) SWIG_fail;\n", symname, maxargs, add_self ? "+1" : "");
      if (add_self)
	Append(f->code, "argv[0] = self;\n");
      else
	Append(f->code, "--argc;\n");
    }

    Replaceall(dispatch, "$args", "self, args");

    Printv(f->code, dispatch, "\n", NIL);

    if (GetFlag(n, "feature:python:maybecall")) {
      Append(f->code, "fail:\n");
      Append(f->code, "  Py_INCREF(Py_NotImplemented);\n");
      Append(f->code, "  return Py_NotImplemented;\n");
    } else {
      Node *sibl = n;
      while (Getattr(sibl, "sym:previousSibling"))
	sibl = Getattr(sibl, "sym:previousSibling");	// go all the way up
      String *protoTypes = NewString("");
      do {
	String *fulldecl = Swig_name_decl(sibl);
	Printf(protoTypes, "\n\"    %s\\n\"", fulldecl);
	Delete(fulldecl);
      } while ((sibl = Getattr(sibl, "sym:nextSibling")));
      Append(f->code, "fail:\n");
      Printf(f->code, "  SWIG_Python_RaiseOrModifyTypeError("
	     "\"Wrong number or type of arguments for overloaded function '%s'.\\n\"" "\n\"  Possible C/C++ prototypes are:\\n\"%s);\n", symname, protoTypes);
      Printf(f->code, "return %s;\n", builtin_ctor ? "-1" : "0");
      Delete(protoTypes);
    }
    Printv(f->code, "}\n", NIL);
    Wrapper_print(f, f_wrappers);
    Node *p = Getattr(n, "sym:previousSibling");
    if (!builtin_self)
      add_method(symname, wname, 0, p);

    /* Create a shadow for this function (if enabled and not in a member function) */
    if (!builtin && (shadow) && (!(shadow & PYSHADOW_MEMBER))) {
      emitFunctionShadowHelper(n, in_class ? f_shadow_stubs : f_shadow, symname, 0);
    }
    DelWrapper(f);
    Delete(dispatch);
    Delete(dispatch_code);
    Delete(tmp);
    Delete(wname);
  }

  /* ------------------------------------------------------------
   * functionWrapper()
   * ------------------------------------------------------------ */

  /*
    A note about argument marshalling with built-in types.
    There are three distinct cases for member (non-static) methods:

    1) An ordinary member function.  In this case, the first param in
    the param list is 'this'.  For builtin types, 'this' is taken from
    the first argument to the wrapper (usually called 'self); it's not
    extracted from the second argument (which is usually a tuple).

    2) A constructor for a non-director class.  In this case, the
    param list doesn't contain an entry for 'this', but the first ('self')
    argument to the wrapper *does* contain the newly-allocated,
    uninitialized object.

    3) A constructor for a director class.  In this case, the param
    list contains a 'self' param, which comes from the first argument
    to the wrapper function.
  */

  const char *get_implicitconv_flag(Node *klass) {
    int conv = 0;
    if (klass && GetFlag(klass, "feature:implicitconv")) {
      conv = 1;
    }
    return conv ? "SWIG_POINTER_IMPLICIT_CONV" : "0";
  }


  virtual int functionWrapper(Node *n) {

    String *name = Getattr(n, "name");
    String *iname = Getattr(n, "sym:name");
    SwigType *d = Getattr(n, "type");
    ParmList *l = Getattr(n, "parms");
    Node *parent = Swig_methodclass(n);

    int director_method = 0;

    Parm *p;
    int i;
    char source[64];
    Wrapper *f;
    String *self_parse;
    String *parse_args;
    String *arglist;
    String *get_pointers;
    String *cleanup;
    String *outarg;
    String *kwargs;
    String *tm;
    String *overname = 0;

    int num_required;
    int num_arguments;
    int num_fixed_arguments;
    int tuple_required;
    int tuple_arguments;
    int varargs = 0;
    int allow_kwargs = check_kwargs(n);

    String *nodeType = Getattr(n, "nodeType");
    int constructor = (!Cmp(nodeType, "constructor"));
    int destructor = (!Cmp(nodeType, "destructor"));
    String *storage = Getattr(n, "storage");
    /* Only the first constructor is handled as init method. Others
       constructor can be emitted via %rename */
    int handled_as_init = 0;
    if (!have_constructor && (constructor || Getattr(n, "handled_as_constructor"))
	&& ((shadow & PYSHADOW_MEMBER))) {
      String *nname = Getattr(n, "sym:name");
      String *sname = Getattr(getCurrentClass(), "sym:name");
      String *cname = Swig_name_construct(NSPACE_TODO, sname);
      handled_as_init = (Strcmp(nname, sname) == 0) || (Strcmp(nname, cname) == 0);
      Delete(cname);
    }
    bool builtin_self = builtin && in_class && (constructor || (l && Getattr(l, "self")));
    bool builtin_ctor = false;
    if (builtin_self && constructor) {
      String *class_mname = Getattr(getCurrentClass(), "sym:name");
      String *mrename = Swig_name_construct(getNSpace(), class_mname);
      if (Cmp(iname, mrename))
	builtin_self = false;
      else
	builtin_ctor = true;
      Delete(mrename);
    }
    bool director_class = (getCurrentClass() && Swig_directorclass(getCurrentClass()));
    bool add_self = builtin_self && (!builtin_ctor || director_class);
    bool builtin_getter = (builtin && GetFlag(n, "memberget"));
    bool builtin_setter = (builtin && GetFlag(n, "memberset") && !builtin_getter);
    char const *self_param = builtin ? "self" : "SWIGUNUSEDPARM(self)";
    char const *wrap_return = builtin_ctor ? "int " : "PyObject *";
    String *linkage = NewString("SWIGINTERN ");
    String *wrapper_name = Swig_name_wrapper(iname);

    if (Getattr(n, "sym:overloaded")) {
      overname = Getattr(n, "sym:overname");
    } else {
      if (!addSymbol(iname, n))
	return SWIG_ERROR;
    }

    f = NewWrapper();
    self_parse = NewString("");
    parse_args = NewString("");
    arglist = NewString("");
    get_pointers = NewString("");
    cleanup = NewString("");
    outarg = NewString("");
    kwargs = NewString("");

    int allow_thread = threads_enable(n);

    Wrapper_add_local(f, "resultobj", "PyObject *resultobj = 0");

    // Emit all of the local variables for holding arguments.
    emit_parameter_variables(l, f);

    /* Attach the standard typemaps */
    emit_attach_parmmaps(l, f);
    Setattr(n, "wrap:parms", l);
    /* Get number of required and total arguments */
    tuple_arguments = num_arguments = emit_num_arguments(l);
    tuple_required = num_required = emit_num_required(l);
    if (add_self) {
      --tuple_arguments;
      --tuple_required;
    }
    num_fixed_arguments = tuple_required;

    // builtin handles/checks kwargs by default except in constructor wrappers so we need to explicitly handle them in the C constructor wrapper
    // The check below is for zero arguments. Sometimes (eg directors) self is the first argument for a method with zero arguments.
    if (((num_arguments == 0) && (num_required == 0)) || ((num_arguments == 1) && (num_required == 1) && Getattr(l, "self")))
      if (!builtin_ctor)
	allow_kwargs = 0;
    varargs = emit_isvarargs(l);

    String *wname = Copy(wrapper_name);
    if (overname) {
      Append(wname, overname);
    }

    const char *builtin_kwargs = builtin_ctor ? ", PyObject *kwargs" : "";
    if (!allow_kwargs || overname) {
      if (!varargs) {
	Printv(f->def, linkage, wrap_return, wname, "(PyObject *", self_param, ", PyObject *args", builtin_kwargs, ") {", NIL);
      } else {
	Printv(f->def, linkage, wrap_return, wname, "__varargs__", "(PyObject *", self_param, ", PyObject *args, PyObject *varargs) {", NIL);
      }
      if (allow_kwargs) {
	Swig_warning(WARN_LANG_OVERLOAD_KEYWORD, input_file, line_number, "Can't use keyword arguments with overloaded functions (%s).\n", Swig_name_decl(n));
	allow_kwargs = 0;
      }
    } else {
      if (varargs) {
	Swig_warning(WARN_LANG_VARARGS_KEYWORD, input_file, line_number, "Can't wrap varargs with keyword arguments enabled\n");
	varargs = 0;
      }
      Printv(f->def, linkage, wrap_return, wname, "(PyObject *", self_param, ", PyObject *args, PyObject *kwargs) {", NIL);
    }
    if (!builtin || !in_class || tuple_arguments > 0 || builtin_ctor) {
      if (!allow_kwargs) {
	Append(parse_args, "    if (!PyArg_ParseTuple(args, \"");
      } else {
	Append(parse_args, "    if (!PyArg_ParseTupleAndKeywords(args, kwargs, \"");
	Append(arglist, ", kwnames");
      }
    }

    bool over_varargs = emit_isvarargs_function(n);

    int funpack = fastunpack && !varargs && !over_varargs && !allow_kwargs;
    int noargs = funpack && (tuple_required == 0 && tuple_arguments == 0);
    int onearg = funpack && (tuple_required == 1 && tuple_arguments == 1);

    if (builtin && funpack && !overname && !builtin_ctor) {
      int compactdefargs = ParmList_is_compactdefargs(l);
      if (!(compactdefargs && (tuple_arguments > tuple_required || varargs))) {
	String *argattr = NewStringf("%d", tuple_arguments);
	Setattr(n, "python:argcount", argattr);
	Delete(argattr);
      }
    }

    /* Generate code for argument marshalling */
    if (funpack) {
      if (num_arguments > 0 && !overname) {
	sprintf(source, "PyObject *swig_obj[%d]", num_arguments);
	Wrapper_add_localv(f, "swig_obj", source, NIL);
      }
    }


    if (constructor && num_arguments == 1 && num_required == 1) {
      if (Cmp(storage, "explicit") == 0) {
	if (GetFlag(parent, "feature:implicitconv")) {
	  String *desc = NewStringf("SWIGTYPE%s", SwigType_manglestr(Getattr(n, "type")));
	  Printf(f->code, "if (SWIG_CheckImplicit(%s)) SWIG_fail;\n", desc);
	  Delete(desc);
	}
      }
    }

    if (builtin_ctor && checkAttribute(n, "access", "protected")) {
      String *tmp_none_comparison = Copy(none_comparison);
      Replaceall(tmp_none_comparison, "$arg", "self");
      Printf(self_parse, "if (!(%s)) {\n", tmp_none_comparison);
      Printv(self_parse, "  SWIG_SetErrorMsg(PyExc_RuntimeError, \"accessing abstract class or protected constructor\");\n  SWIG_fail;\n}\n", NIL);
      Delete(tmp_none_comparison);
    }

    int use_parse = 0;
    Append(kwargs, "{");
    for (i = 0, p = l; i < num_arguments; i++) {
      while (checkAttribute(p, "tmap:in:numinputs", "0")) {
	p = Getattr(p, "tmap:in:next");
      }

      SwigType *pt = Getattr(p, "type");
      String *ln = Getattr(p, "lname");
      bool parse_from_tuple = (i > 0 || !add_self);
      if (SwigType_type(pt) == T_VARARGS) {
	parse_from_tuple = false;
	num_fixed_arguments -= atoi(Char(Getattr(p, "tmap:in:numinputs")));
      }
      if (!parse_from_tuple)
	sprintf(source, "self");
      else if (funpack)
	sprintf(source, "swig_obj[%d]", add_self && !overname ? i - 1 : i);
      else
	sprintf(source, "obj%d", builtin_ctor ? i + 1 : i);

      if (parse_from_tuple) {
	Printf(arglist, ", ");
	if (i == num_required)
	  Putc('|', parse_args);	/* Optional argument separator */
      }

      /* Keyword argument handling */
      if (allow_kwargs && parse_from_tuple) {
	String *name = makeParameterName(n, p, i + 1);
	Printf(kwargs, " (char *)\"%s\", ", name);
	Delete(name);
      }

      /* Look for an input typemap */
      if ((tm = Getattr(p, "tmap:in"))) {
	String *parse = Getattr(p, "tmap:in:parse");
	if (!parse) {
	  if (builtin_self) {
	    Replaceall(tm, "$self", "self");
	  } else if (funpack) {
	    Replaceall(tm, "$self", "swig_obj[0]");
	  } else {
	    Replaceall(tm, "$self", "obj0");
	  }
	  Replaceall(tm, "$source", source);
	  Replaceall(tm, "$target", ln);
	  Replaceall(tm, "$input", source);
	  Setattr(p, "emit:input", source);	/* Save the location of the object */

	  if (Getattr(p, "wrap:disown") || (Getattr(p, "tmap:in:disown"))) {
	    Replaceall(tm, "$disown", "SWIG_POINTER_DISOWN");
	  } else {
	    Replaceall(tm, "$disown", "0");
	  }

	  if (Getattr(p, "tmap:in:implicitconv")) {
	    const char *convflag = "0";
	    if (!Getattr(p, "hidden")) {
	      SwigType *ptype = Getattr(p, "type");
	      convflag = get_implicitconv_flag(classLookup(ptype));
	    }
	    Replaceall(tm, "$implicitconv", convflag);
	    Setattr(p, "implicitconv", convflag);
	  }

	  if (parse_from_tuple)
	    Putc('O', parse_args);
	  if (!funpack && parse_from_tuple) {
	    Wrapper_add_localv(f, source, "PyObject *", source, "= 0", NIL);
	    Printf(arglist, "&%s", source);
	  }
	  if (i >= num_required)
	    Printv(get_pointers, "if (", source, ") {\n", NIL);
	  Printv(get_pointers, tm, "\n", NIL);
	  if (i >= num_required)
	    Printv(get_pointers, "}\n", NIL);

	} else {
	  use_parse = 1;
	  Append(parse_args, parse);
	  if (parse_from_tuple)
	    Printf(arglist, "&%s", ln);
	}
	p = Getattr(p, "tmap:in:next");
	continue;
      } else {
	Swig_warning(WARN_TYPEMAP_IN_UNDEF, input_file, line_number, "Unable to use type %s as a function argument.\n", SwigType_str(pt, 0));
	break;
      }
    }

    /* finish argument marshalling */
    Append(kwargs, " NULL }");
    if (allow_kwargs) {
      Printv(f->locals, "  char * kwnames[] = ", kwargs, ";\n", NIL);
    }

    if (use_parse || allow_kwargs) {
      Printf(parse_args, ":%s\"", iname);
      Printv(parse_args, arglist, ")) SWIG_fail;\n", NIL);
      funpack = 0;
    } else {
      Clear(parse_args);

      if (funpack) {
	Clear(f->def);
	if (overname) {
	  if (noargs) {
	    Printv(f->def, linkage, wrap_return, wname, "(PyObject *", self_param, ", Py_ssize_t nobjs, PyObject **SWIGUNUSEDPARM(swig_obj)) {", NIL);
	  } else {
	    Printv(f->def, linkage, wrap_return, wname, "(PyObject *", self_param, ", Py_ssize_t nobjs, PyObject **swig_obj) {", NIL);
	  }
	  Printf(parse_args, "if ((nobjs < %d) || (nobjs > %d)) SWIG_fail;\n", num_required, num_arguments);
	} else {
	  int is_tp_call = Equal(Getattr(n, "feature:python:slot"), "tp_call");
	  Printv(f->def, linkage, wrap_return, wname, "(PyObject *", self_param, ", PyObject *args", builtin_kwargs, ") {", NIL);
	  if (builtin_ctor)
	    Printf(parse_args, "if (!SWIG_Python_CheckNoKeywords(kwargs, \"%s\")) SWIG_fail;\n", iname);
	  if (onearg && !builtin_ctor && !is_tp_call) {
	    Printf(parse_args, "if (!args) SWIG_fail;\n");
	    Append(parse_args, "swig_obj[0] = args;\n");
	  } else if (!noargs) {
	    Printf(parse_args, "if (!SWIG_Python_UnpackTuple(args, \"%s\", %d, %d, swig_obj)) SWIG_fail;\n", iname, num_fixed_arguments, tuple_arguments);
	  } else if (noargs) {
	    Printf(parse_args, "if (!SWIG_Python_UnpackTuple(args, \"%s\", %d, %d, 0)) SWIG_fail;\n", iname, num_fixed_arguments, tuple_arguments);
	  }
	}
      } else {
	if (builtin_ctor)
	  Printf(parse_args, "if (!SWIG_Python_CheckNoKeywords(kwargs, \"%s\")) SWIG_fail;\n", iname);
	if (builtin && in_class && tuple_arguments == 0) {
	  Printf(parse_args, "    if (args && PyTuple_Check(args) && PyTuple_GET_SIZE(args) > 0) SWIG_exception_fail(SWIG_TypeError, \"%s takes no arguments\");\n", iname);
	} else {
	  Printf(parse_args, "if (!PyArg_UnpackTuple(args, \"%s\", %d, %d", iname, num_fixed_arguments, tuple_arguments);
	  Printv(parse_args, arglist, ")) SWIG_fail;\n", NIL);
	}
      }
    }

    /* Now piece together the first part of the wrapper function */
    Printv(f->code, self_parse, parse_args, get_pointers, NIL);

    /* Check for trailing varargs */
    if (varargs) {
      if (p && (tm = Getattr(p, "tmap:in"))) {
	Replaceall(tm, "$input", "varargs");
	Printv(f->code, tm, "\n", NIL);
      }
    }

    /* Insert constraint checking code */
    for (p = l; p;) {
      if ((tm = Getattr(p, "tmap:check"))) {
	Replaceall(tm, "$target", Getattr(p, "lname"));
	Printv(f->code, tm, "\n", NIL);
	p = Getattr(p, "tmap:check:next");
      } else {
	p = nextSibling(p);
      }
    }

    /* Insert cleanup code */
    for (p = l; p;) {
      if (!Getattr(p, "tmap:in:parse") && (tm = Getattr(p, "tmap:freearg"))) {
	if (Getattr(p, "tmap:freearg:implicitconv")) {
	  const char *convflag = "0";
	  if (!Getattr(p, "hidden")) {
	    SwigType *ptype = Getattr(p, "type");
	    convflag = get_implicitconv_flag(classLookup(ptype));
	  }
	  if (strcmp(convflag, "0") == 0) {
	    tm = 0;
	  }
	}
	if (tm && (Len(tm) != 0)) {
	  Replaceall(tm, "$source", Getattr(p, "lname"));
	  Printv(cleanup, tm, "\n", NIL);
	}
	p = Getattr(p, "tmap:freearg:next");
      } else {
	p = nextSibling(p);
      }
    }

    /* Insert argument output code */
    for (p = l; p;) {
      if ((tm = Getattr(p, "tmap:argout"))) {
	Replaceall(tm, "$source", Getattr(p, "lname"));
	Replaceall(tm, "$target", "resultobj");
	Replaceall(tm, "$arg", Getattr(p, "emit:input"));
	Replaceall(tm, "$input", Getattr(p, "emit:input"));
	Printv(outarg, tm, "\n", NIL);
	p = Getattr(p, "tmap:argout:next");
      } else {
	p = nextSibling(p);
      }
    }

    /* if the object is a director, and the method call originated from its
     * underlying python object, resolve the call by going up the c++
     * inheritance chain.  otherwise try to resolve the method in python.
     * without this check an infinite loop is set up between the director and
     * shadow class method calls.
     */

    // NOTE: this code should only be inserted if this class is the
    // base class of a director class.  however, in general we haven't
    // yet analyzed all classes derived from this one to see if they are
    // directors.  furthermore, this class may be used as the base of
    // a director class defined in a completely different module at a
    // later time, so this test must be included whether or not directorbase
    // is true.  we do skip this code if directors have not been enabled
    // at the command line to preserve source-level compatibility with
    // non-polymorphic swig.  also, if this wrapper is for a smart-pointer
    // method, there is no need to perform the test since the calling object
    // (the smart-pointer) and the director object (the "pointee") are
    // distinct.

    director_method = is_member_director(n) && !is_smart_pointer() && !destructor;
    if (director_method) {
      Wrapper_add_local(f, "director", "Swig::Director *director = 0");
      Append(f->code, "director = SWIG_DIRECTOR_CAST(arg1);\n");
      if (dirprot_mode() && !is_public(n)) {
	Printf(f->code, "if (!director || !(director->swig_get_inner(\"%s\"))) {\n", name);
	Printf(f->code, "SWIG_SetErrorMsg(PyExc_RuntimeError,\"accessing protected member %s\");\n", name);
	Append(f->code, "SWIG_fail;\n");
	Append(f->code, "}\n");
      }
      Wrapper_add_local(f, "upcall", "bool upcall = false");
      if (funpack) {
	const char *self_parm = builtin_self ? "self" : "swig_obj[0]";
	Printf(f->code, "upcall = (director && (director->swig_get_self()==%s));\n", self_parm);
      } else {
	const char *self_parm = builtin_self ? "self" : "obj0";
	Printf(f->code, "upcall = (director && (director->swig_get_self()==%s));\n", self_parm);
      }
    }

    /* Emit the function call */
    if (director_method) {
      Append(f->code, "try {\n");
    } else {
      if (allow_thread) {
	String *preaction = NewString("");
	thread_begin_allow(n, preaction);
	Setattr(n, "wrap:preaction", preaction);

	String *postaction = NewString("");
	thread_end_allow(n, postaction);
	Setattr(n, "wrap:postaction", postaction);
      }
    }

    Setattr(n, "wrap:name", wname);

    Swig_director_emit_dynamic_cast(n, f);
    String *actioncode = emit_action(n);

    if (director_method) {
      Append(actioncode, "} catch (Swig::DirectorException&) {\n");
      Append(actioncode, "  SWIG_fail;\n");
      Append(actioncode, "}\n");
    }

    /* This part below still needs cleanup */

    /* Return the function value */
    tm = Swig_typemap_lookup_out("out", n, Swig_cresult_name(), f, actioncode);

    if (tm) {
      if (builtin_self) {
	Replaceall(tm, "$self", "self");
      } else if (funpack) {
	Replaceall(tm, "$self", "swig_obj[0]");
      } else {
	Replaceall(tm, "$self", "obj0");
      }
      Replaceall(tm, "$source", Swig_cresult_name());
      Replaceall(tm, "$target", "resultobj");
      Replaceall(tm, "$result", "resultobj");
      if (builtin_ctor) {
	Replaceall(tm, "$owner", "SWIG_BUILTIN_INIT");
      } else if (handled_as_init) {
	Replaceall(tm, "$owner", "SWIG_POINTER_NEW");
      } else {
	if (GetFlag(n, "feature:new")) {
	  Replaceall(tm, "$owner", "SWIG_POINTER_OWN");
	} else {
	  Replaceall(tm, "$owner", "0");
	}
      }
      // FIXME: this will not try to unwrap directors returned as non-director
      //        base class pointers!

      /* New addition to unwrap director return values so that the original
       * python object is returned instead.
       */
#if 1
      int unwrap = 0;
      String *decl = Getattr(n, "decl");
      int is_pointer = SwigType_ispointer_return(decl);
      int is_reference = SwigType_isreference_return(decl);
      if (is_pointer || is_reference) {
	String *type = Getattr(n, "type");
	//Node *classNode = Swig_methodclass(n);
	//Node *module = Getattr(classNode, "module");
	Node *module = Getattr(parent, "module");
	Node *target = Swig_directormap(module, type);
	if (target)
	  unwrap = 1;
      }
      if (unwrap) {
	Wrapper_add_local(f, "director", "Swig::Director *director = 0");
	Printf(f->code, "director = SWIG_DIRECTOR_CAST(%s);\n", Swig_cresult_name());
	Append(f->code, "if (director) {\n");
	Append(f->code, "  resultobj = director->swig_get_self();\n");
	Append(f->code, "  Py_INCREF(resultobj);\n");
	Append(f->code, "} else {\n");
	Printf(f->code, "%s\n", tm);
	Append(f->code, "}\n");
      } else {
	Printf(f->code, "%s\n", tm);
      }
#else
      Printf(f->code, "%s\n", tm);
#endif
      Delete(tm);
    } else {
      Swig_warning(WARN_TYPEMAP_OUT_UNDEF, input_file, line_number, "Unable to use return type %s in function %s.\n", SwigType_str(d, 0), name);
    }
    emit_return_variable(n, d, f);

    /* Output argument output code */
    Printv(f->code, outarg, NIL);

    /* Output cleanup code */
    int need_cleanup = Len(cleanup) != 0;
    if (need_cleanup) {
      Printv(f->code, cleanup, NIL);
    }

    /* Look to see if there is any newfree cleanup code */
    if (GetFlag(n, "feature:new")) {
      if ((tm = Swig_typemap_lookup("newfree", n, Swig_cresult_name(), 0))) {
	Replaceall(tm, "$source", Swig_cresult_name());
	Printf(f->code, "%s\n", tm);
	Delete(tm);
      }
    }

    /* See if there is any return cleanup code */
    if ((tm = Swig_typemap_lookup("ret", n, Swig_cresult_name(), 0))) {
      Replaceall(tm, "$source", Swig_cresult_name());
      Printf(f->code, "%s\n", tm);
      Delete(tm);
    }

    if (director_method) {
      if ((tm = Swig_typemap_lookup("directorfree", n, Swig_cresult_name(), 0))) {
	Replaceall(tm, "$input", Swig_cresult_name());
	Replaceall(tm, "$result", "resultobj");
	Printf(f->code, "%s\n", tm);
	Delete(tm);
      }
    }

    if (builtin_ctor)
      Append(f->code, "    return resultobj == Py_None ? -1 : 0;\n");
    else
      Append(f->code, "    return resultobj;\n");

    /* Error handling code */

    Append(f->code, "fail:\n");
    if (need_cleanup) {
      Printv(f->code, cleanup, NIL);
    }
    if (builtin_ctor) {
      Printv(f->code, "  return -1;\n", NIL);
    } else {
      if (GetFlag(n, "feature:python:maybecall")) {
	Append(f->code, "  PyErr_Clear();\n");
	Append(f->code, "  Py_INCREF(Py_NotImplemented);\n");
	Append(f->code, "  return Py_NotImplemented;\n");
      } else {
        Printv(f->code, "  return NULL;\n", NIL);
      }
    }

    Append(f->code, "}\n");

    /* Substitute the cleanup code */
    Replaceall(f->code, "$cleanup", cleanup);

    /* Substitute the function name */
    Replaceall(f->code, "$symname", iname);
    Replaceall(f->code, "$result", "resultobj");

    if (builtin_self) {
      Replaceall(f->code, "$self", "self");
    } else if (funpack) {
      Replaceall(f->code, "$self", "swig_obj[0]");
    } else {
      Replaceall(f->code, "$self", "obj0");
    }

    /* Dump the function out */
    Wrapper_print(f, f_wrappers);

    /* If varargs.  Need to emit a varargs stub */
    if (varargs) {
      DelWrapper(f);
      f = NewWrapper();
      if (funpack) {
	// Note: funpack is currently always false for varargs
	Printv(f->def, linkage, wrap_return, wname, "(PyObject *", self_param, ", Py_ssize_t nobjs, PyObject **swig_obj) {", NIL);
      } else {
	Printv(f->def, linkage, wrap_return, wname, "(PyObject *", self_param, ", PyObject *args", builtin_kwargs, ") {", NIL);
      }
      Wrapper_add_local(f, "resultobj", builtin_ctor ? "int resultobj" : "PyObject *resultobj");
      Wrapper_add_local(f, "varargs", "PyObject *varargs");
      Wrapper_add_local(f, "newargs", "PyObject *newargs");
      if (funpack) {
	Wrapper_add_local(f, "i", "int i");
	Printf(f->code, "newargs = PyTuple_New(%d);\n", num_fixed_arguments);
	Printf(f->code, "for (i = 0; i < %d; ++i) {\n", num_fixed_arguments);
	Printf(f->code, "  PyTuple_SET_ITEM(newargs, i, swig_obj[i]);\n");
	Printf(f->code, "  Py_XINCREF(swig_obj[i]);\n");
	Printf(f->code, "}\n");
	Printf(f->code, "varargs = PyTuple_New(nobjs > %d ? nobjs - %d : 0);\n", num_fixed_arguments, num_fixed_arguments);
	Printf(f->code, "for (i = 0; i < nobjs - %d; ++i) {\n", num_fixed_arguments);
	Printf(f->code, "  PyTuple_SET_ITEM(newargs, i, swig_obj[i + %d]);\n", num_fixed_arguments);
	Printf(f->code, "  Py_XINCREF(swig_obj[i + %d]);\n", num_fixed_arguments);
	Printf(f->code, "}\n");
      } else {
	Printf(f->code, "newargs = PyTuple_GetSlice(args,0,%d);\n", num_fixed_arguments);
	Printf(f->code, "varargs = PyTuple_GetSlice(args,%d,PyTuple_Size(args));\n", num_fixed_arguments);
      }
      Printf(f->code, "resultobj = %s__varargs__(%s,newargs,varargs);\n", wname, builtin ? "self" : "NULL");
      Append(f->code, "Py_XDECREF(newargs);\n");
      Append(f->code, "Py_XDECREF(varargs);\n");
      Append(f->code, "return resultobj;\n");
      Append(f->code, "}\n");
      Wrapper_print(f, f_wrappers);
    }

    /* Now register the function with the interpreter.   */
    if (!Getattr(n, "sym:overloaded")) {
      if (!builtin_self)
	add_method(iname, wname, allow_kwargs, n, funpack, num_required, num_arguments);

      /* Create a shadow for this function (if enabled and not in a member function) */
      if (!builtin && (shadow) && (!(shadow & PYSHADOW_MEMBER))) {
	emitFunctionShadowHelper(n, in_class ? f_shadow_stubs : f_shadow, iname, allow_kwargs);
      }
    } else {
      if (!Getattr(n, "sym:nextSibling")) {
	dispatchFunction(n, linkage, funpack, builtin_self, builtin_ctor, director_class);
      }
    }

    // Put this in tp_init of the PyTypeObject
    if (builtin_ctor) {
      if ((director_method || !is_private(n)) && !Getattr(class_members, iname)) {
	Setattr(class_members, iname, n);
	if (!builtin_tp_init)
	  builtin_tp_init = Swig_name_wrapper(iname);
      }
    }

    /* If this is a builtin type, create a PyGetSetDef entry for this member variable. */
    if (builtin) {
      const char *memname = "__dict__";
      Hash *h = Getattr(builtin_getset, memname);
      if (!h) {
        h = NewHash();
        Setattr(builtin_getset, memname, h);
        Delete(h);
      }
      Setattr(h, "getter", "SwigPyObject_get___dict__");
      if (!Getattr(h, "doc")) {
	Setattr(n, "doc:high:name", Getattr(n, "name"));
	Setattr(h, "doc", cdocstring(n, AUTODOC_VAR));
      }
    }

    if (builtin_getter) {
      String *memname = Getattr(n, "membervariableHandler:sym:name");
      if (!memname)
	memname = iname;
      Hash *h = Getattr(builtin_getset, memname);
      if (!h) {
	h = NewHash();
	Setattr(builtin_getset, memname, h);
	Delete(h);
      }
      Setattr(h, "getter", wrapper_name);
      Delattr(n, "memberget");
      if (!Getattr(h, "doc")) {
	Setattr(n, "doc:high:name", Getattr(n, "name"));
	String *ds = cdocstring(n, AUTODOC_VAR);
	Setattr(h, "doc", ds);
	Delete(ds);
      }
    }
    if (builtin_setter) {
      String *memname = Getattr(n, "membervariableHandler:sym:name");
      if (!memname)
	memname = iname;
      Hash *h = Getattr(builtin_getset, memname);
      if (!h) {
	h = NewHash();
	Setattr(builtin_getset, memname, h);
	Delete(h);
      }
      Setattr(h, "setter", wrapper_name);
      Delattr(n, "memberset");
      if (!Getattr(h, "doc")) {
	Setattr(n, "doc:high:name", Getattr(n, "name"));
	String *ds = cdocstring(n, AUTODOC_VAR);
	Setattr(h, "doc", ds);
	Delete(ds);
      }
    }

    if (in_class && builtin) {
      /* Handle operator overloads for builtin types */
      String *slot = Getattr(n, "feature:python:slot");
      if (slot) {
	String *func_type = Getattr(n, "feature:python:slot:functype");
	String *closure_decl = getClosure(func_type, wrapper_name, overname ? 0 : funpack);
	String *feature_name = NewStringf("feature:python:%s", slot);
	String *closure_name = 0;
	if (closure_decl) {
	  closure_name = NewStringf("%s_%s_closure", wrapper_name, func_type);
	  if (!GetFlag(builtin_closures, closure_name))
	    Printf(builtin_closures_code, "%s /* defines %s */\n\n", closure_decl, closure_name);
	  SetFlag(builtin_closures, closure_name);
	  Delete(closure_decl);
	} else {
	  closure_name = Copy(wrapper_name);
	}
	if (func_type) {
	  String *s = NewStringf("%s", closure_name);
	  Delete(closure_name);
	  closure_name = s;
	}
	Setattr(parent, feature_name, closure_name);
	Delete(feature_name);
	Delete(closure_name);
      }

      /* Handle comparison operators for builtin types */
      String *compare = Getattr(n, "feature:python:compare");
      if (compare) {
	Hash *richcompare = Getattr(parent, "python:richcompare");
	assert(richcompare);
	Setattr(richcompare, compare, wrapper_name);
      }
    }

    Delete(self_parse);
    Delete(parse_args);
    Delete(linkage);
    Delete(arglist);
    Delete(get_pointers);
    Delete(cleanup);
    Delete(outarg);
    Delete(kwargs);
    Delete(wname);
    DelWrapper(f);
    Delete(wrapper_name);
    return SWIG_OK;
  }



  /* ------------------------------------------------------------
   * variableWrapper()
   * ------------------------------------------------------------ */

  virtual int variableWrapper(Node *n) {
    String *name = Getattr(n, "name");
    String *iname = Getattr(n, "sym:name");
    SwigType *t = Getattr(n, "type");

    static int have_globals = 0;
    String *tm;
    Wrapper *getf, *setf;

    if (!addSymbol(iname, n))
      return SWIG_ERROR;

    getf = NewWrapper();
    setf = NewWrapper();

    /* If this is our first call, add the globals variable to the
       Python dictionary. */

    if (!have_globals) {
      Printf(f_init, "\t globals = SWIG_globals();\n");
      Printf(f_init, "\t if (!globals) {\n");
      Printf(f_init, "     PyErr_SetString(PyExc_TypeError, \"Failure to create SWIG globals.\");\n");
      Printf(f_init, "#if PY_VERSION_HEX >= 0x03000000\n");
      Printf(f_init, "\t   return NULL;\n");
      Printf(f_init, "#else\n");
      Printf(f_init, "\t   return;\n");
      Printf(f_init, "#endif\n");
      Printf(f_init, "\t }\n");
      Printf(f_init, "\t PyDict_SetItemString(md, \"%s\", globals);\n", global_name);
      Printf(f_init, "\t Py_DECREF(globals);\n");
      if (builtin)
	Printf(f_init, "\t SwigPyBuiltin_AddPublicSymbol(public_interface, \"%s\");\n", global_name);
      have_globals = 1;
      if (!builtin && (shadow) && (!(shadow & PYSHADOW_MEMBER))) {
	Printf(f_shadow_stubs, "%s = %s.%s\n", global_name, module, global_name);
      }
    }
    int assignable = is_assignable(n);

    if (!builtin && shadow && !assignable && !in_class)
      Printf(f_shadow_stubs, "%s = %s.%s\n", iname, global_name, iname);

    String *getname = Swig_name_get(NSPACE_TODO, iname);
    String *setname = Swig_name_set(NSPACE_TODO, iname);
    String *vargetname = NewStringf("Swig_var_%s", getname);
    String *varsetname = NewStringf("Swig_var_%s", setname);

    /* Create a function for setting the value of the variable */
    if (assignable) {
      Setattr(n, "wrap:name", varsetname);
      if (builtin && in_class) {
	String *set_wrapper = Swig_name_wrapper(setname);
	Setattr(n, "pybuiltin:setter", set_wrapper);
	Delete(set_wrapper);
      }
      Printf(setf->def, "SWIGINTERN int %s(PyObject *_val) {", varsetname);
      if ((tm = Swig_typemap_lookup("varin", n, name, 0))) {
	Replaceall(tm, "$source", "_val");
	Replaceall(tm, "$target", name);
	Replaceall(tm, "$input", "_val");
	if (Getattr(n, "tmap:varin:implicitconv")) {
	  Replaceall(tm, "$implicitconv", get_implicitconv_flag(n));
	}
	emit_action_code(n, setf->code, tm);
	Delete(tm);
      } else {
	Swig_warning(WARN_TYPEMAP_VARIN_UNDEF, input_file, line_number, "Unable to set variable of type %s.\n", SwigType_str(t, 0));
      }
      Printv(setf->code, "  return 0;\n", NULL);
      Append(setf->code, "fail:\n");
      Printv(setf->code, "  return 1;\n", NULL);
    } else {
      /* Is a readonly variable.  Issue an error */
      if (CPlusPlus) {
	Printf(setf->def, "SWIGINTERN int %s(PyObject *) {", varsetname);
      } else {
	Printf(setf->def, "SWIGINTERN int %s(PyObject *_val SWIGUNUSED) {", varsetname);
      }
      Printv(setf->code, "  SWIG_Error(SWIG_AttributeError,\"Variable ", iname, " is read-only.\");\n", "  return 1;\n", NIL);
    }

    Append(setf->code, "}\n");
    Wrapper_print(setf, f_wrappers);

    /* Create a function for getting the value of a variable */
    Setattr(n, "wrap:name", vargetname);
    if (builtin && in_class) {
      String *get_wrapper = Swig_name_wrapper(getname);
      Setattr(n, "pybuiltin:getter", get_wrapper);
      Delete(get_wrapper);
    }
    int addfail = 0;
    Printf(getf->def, "SWIGINTERN PyObject *%s(void) {", vargetname);
    Wrapper_add_local(getf, "pyobj", "PyObject *pyobj = 0");
    if (builtin) {
      Wrapper_add_local(getf, "self", "PyObject *self = 0");
      Append(getf->code, "  (void)self;\n");
    }
    if ((tm = Swig_typemap_lookup("varout", n, name, 0))) {
      Replaceall(tm, "$source", name);
      Replaceall(tm, "$target", "pyobj");
      Replaceall(tm, "$result", "pyobj");
      addfail = emit_action_code(n, getf->code, tm);
      Delete(tm);
    } else {
      Swig_warning(WARN_TYPEMAP_VAROUT_UNDEF, input_file, line_number, "Unable to read variable of type %s\n", SwigType_str(t, 0));
    }
    Append(getf->code, "  return pyobj;\n");
    if (addfail) {
      Append(getf->code, "fail:\n");
      Append(getf->code, "  return NULL;\n");
    }
    Append(getf->code, "}\n");

    Wrapper_print(getf, f_wrappers);

    /* Now add this to the variable linking mechanism */
    Printf(f_init, "\t SWIG_addvarlink(globals, \"%s\", %s, %s);\n", iname, vargetname, varsetname);
    if (builtin && shadow && !assignable && !in_class) {
      Printf(f_init, "\t PyDict_SetItemString(md, \"%s\", PyObject_GetAttrString(globals, \"%s\"));\n", iname, iname);
      Printf(f_init, "\t SwigPyBuiltin_AddPublicSymbol(public_interface, \"%s\");\n", iname);
    }
    Delete(vargetname);
    Delete(varsetname);
    Delete(getname);
    Delete(setname);
    DelWrapper(setf);
    DelWrapper(getf);
    return SWIG_OK;
  }

  /* ------------------------------------------------------------
   * constantWrapper()
   * ------------------------------------------------------------ */

  /* Determine if the node requires the _swigconstant code to be generated */
  bool needs_swigconstant(Node *n) {
    SwigType *type = Getattr(n, "type");
    SwigType *qtype = SwigType_typedef_resolve_all(type);
    SwigType *uqtype = SwigType_strip_qualifiers(qtype);
    bool result = false;

    /* Note, that we need special handling for function pointers, as
     * SwigType_base(fptr) does not return the underlying pointer-to-function
     * type but the return-type of function. */
    if (!SwigType_isfunction(uqtype) && !SwigType_isfunctionpointer(uqtype)) {
      SwigType *basetype = SwigType_base(uqtype);
      result = SwigType_isclass(basetype) != 0;
      Delete(basetype);
    }

    Delete(qtype);
    Delete(uqtype);

    return result;
  }

  virtual int constantWrapper(Node *n) {
    String *name = Getattr(n, "name");
    String *iname = Getattr(n, "sym:name");
    SwigType *type = Getattr(n, "type");
    String *rawval = Getattr(n, "rawval");
    String *value = rawval ? rawval : Getattr(n, "value");
    String *tm;
    int have_tm = 0;
    int have_builtin_symname = 0;

    if (!addSymbol(iname, n))
      return SWIG_ERROR;

    /* Special hook for member pointer */
    if (SwigType_type(type) == T_MPOINTER) {
      String *wname = Swig_name_wrapper(iname);
      String *str = SwigType_str(type, wname);
      Printf(f_header, "static %s = %s;\n", str, value);
      Delete(str);
      value = wname;
    }

    if ((tm = Swig_typemap_lookup("consttab", n, name, 0))) {
      Replaceall(tm, "$source", value);
      Replaceall(tm, "$target", name);
      Replaceall(tm, "$value", value);
      Printf(const_code, "%s,\n", tm);
      Delete(tm);
      have_tm = 1;
    }


    if (builtin && in_class && Getattr(n, "pybuiltin:symname")) {
      have_builtin_symname = 1;
      Swig_require("builtin_constantWrapper", n, "*sym:name", "pybuiltin:symname", NIL);
      Setattr(n, "sym:name", Getattr(n, "pybuiltin:symname"));
    }

    if ((tm = Swig_typemap_lookup("constcode", n, name, 0))) {
      Replaceall(tm, "$source", value);
      Replaceall(tm, "$target", name);
      Replaceall(tm, "$value", value);
      if (needs_swigconstant(n) && !builtin && (shadow) && (!(shadow & PYSHADOW_MEMBER)) && (!in_class || !Getattr(n, "feature:python:callback"))) {
	// Generate `*_swigconstant()` method which registers the new constant.
	//
	// *_swigconstant methods are required for constants of class type.
	// Class types are registered in shadow file (see *_swigregister). The
	// instances of class must be created (registered) after the type is
	// registered, so we can't let SWIG_init() to register constants of
	// class type (the SWIG_init() is called before shadow classes are
	// defined and registered).
        Printf(f_wrappers, "SWIGINTERN PyObject *%s_swigconstant(PyObject *SWIGUNUSEDPARM(self), PyObject *args) {\n", iname);
        Printf(f_wrappers, tab2 "PyObject *module;\n", tm);
        Printf(f_wrappers, tab2 "PyObject *d;\n");
	Printf(f_wrappers, tab2 "if (!SWIG_Python_UnpackTuple(args, \"swigconstant\", 1, 1, &module)) return NULL;\n");
        Printf(f_wrappers, tab2 "d = PyModule_GetDict(module);\n");
        Printf(f_wrappers, tab2 "if (!d) return NULL;\n");
        Printf(f_wrappers, tab2 "%s\n", tm);
        Printf(f_wrappers, tab2 "return SWIG_Py_Void();\n");
        Printf(f_wrappers, "}\n\n\n");

        // Register the method in SwigMethods array
	String *cname = NewStringf("%s_swigconstant", iname);
	add_method(cname, cname, 0, 0, 1, 1, 1);
	Delete(cname);
      } else {
        Printf(f_init, "%s\n", tm);
      }
      Delete(tm);
      have_tm = 1;
    }

    if (have_builtin_symname)
      Swig_restore(n);

    if (!have_tm) {
      Swig_warning(WARN_TYPEMAP_CONST_UNDEF, input_file, line_number, "Unsupported constant value.\n");
      return SWIG_NOWRAP;
    }

    if (!builtin && (shadow) && (!(shadow & PYSHADOW_MEMBER))) {
      String *f_s;
      if (!in_class) {
	f_s = f_shadow;
      } else {
	f_s = Getattr(n, "feature:python:callback") ? NIL : f_shadow_stubs;
      }

      if (f_s) {
	if (needs_swigconstant(n)) {
	  Printv(f_s, "\n",NIL);
	  Printv(f_s, module, ".", iname, "_swigconstant(",module,")\n", NIL);
	}
	Printv(f_s, iname, " = ", module, ".", iname, "\n", NIL);
	if (have_docstring(n))
	  Printv(f_s, docstring(n, AUTODOC_CONST, tab4), "\n", NIL);
      }
    }
    return SWIG_OK;
  }


  /* ------------------------------------------------------------
   * nativeWrapper()
   * ------------------------------------------------------------ */

  virtual int nativeWrapper(Node *n) {
    String *name = Getattr(n, "sym:name");
    String *wrapname = Getattr(n, "wrap:name");

    if (!addSymbol(wrapname, n))
      return SWIG_ERROR;

    add_method(name, wrapname, 0);
    if (!builtin && shadow) {
      Printv(f_shadow_stubs, name, " = ", module, ".", name, "\n", NIL);
    }
    return SWIG_OK;
  }



  /* ----------------------------------------------------------------------------
   * BEGIN C++ Director Class modifications
   * ------------------------------------------------------------------------- */

  /* C++/Python polymorphism demo code
   *
   * TODO
   *
   * Move some boilerplate code generation to Swig_...() functions.
   *
   */

  /* ---------------------------------------------------------------
   * classDirectorMethod()
   *
   * Emit a virtual director method to pass a method call on to the
   * underlying Python object.
   * ** Moved down due to gcc-2.96 internal error **
   * --------------------------------------------------------------- */

  int classDirectorMethods(Node *n);

  int classDirectorMethod(Node *n, Node *parent, String *super);

  /* ------------------------------------------------------------
   * classDirectorConstructor()
   * ------------------------------------------------------------ */

  int classDirectorConstructor(Node *n) {
    Node *parent = Getattr(n, "parentNode");
    String *sub = NewString("");
    String *decl = Getattr(n, "decl");
    String *supername = Swig_class_name(parent);
    String *classname = NewString("");
    Printf(classname, "SwigDirector_%s", supername);

    /* insert self parameter */
    Parm *p;
    ParmList *superparms = Getattr(n, "parms");
    ParmList *parms = CopyParmList(superparms);
    String *type = NewString("PyObject");
    SwigType_add_pointer(type);
    p = NewParm(type, NewString("self"), n);
    set_nextSibling(p, parms);
    parms = p;

    if (!Getattr(n, "defaultargs")) {
      /* constructor */
      {
	Wrapper *w = NewWrapper();
	String *call;
	String *basetype = Getattr(parent, "classtype");
	String *target = Swig_method_decl(0, decl, classname, parms, 0);
	call = Swig_csuperclass_call(0, basetype, superparms);
	Printf(w->def, "%s::%s: %s, Swig::Director(self) { \n", classname, target, call);
	Printf(w->def, "   SWIG_DIRECTOR_RGTR((%s *)this, this); \n", basetype);
	Append(w->def, "}\n");
	Delete(target);
	Wrapper_print(w, f_directors);
	Delete(call);
	DelWrapper(w);
      }

      /* constructor header */
      {
	String *target = Swig_method_decl(0, decl, classname, parms, 1);
	Printf(f_directors_h, "    %s;\n", target);
	Delete(target);
      }
    }

    Delete(sub);
    Delete(classname);
    Delete(supername);
    Delete(parms);
    return Language::classDirectorConstructor(n);
  }

  /* ------------------------------------------------------------
   * classDirectorDefaultConstructor()
   * ------------------------------------------------------------ */

  int classDirectorDefaultConstructor(Node *n) {
    String *classname = Swig_class_name(n);
    {
      Node *parent = Swig_methodclass(n);
      String *basetype = Getattr(parent, "classtype");
      Wrapper *w = NewWrapper();
      Printf(w->def, "SwigDirector_%s::SwigDirector_%s(PyObject *self) : Swig::Director(self) { \n", classname, classname);
      Printf(w->def, "   SWIG_DIRECTOR_RGTR((%s *)this, this); \n", basetype);
      Append(w->def, "}\n");
      Wrapper_print(w, f_directors);
      DelWrapper(w);
    }
    Printf(f_directors_h, "    SwigDirector_%s(PyObject *self);\n", classname);
    Delete(classname);
    return Language::classDirectorDefaultConstructor(n);
  }


  /* ------------------------------------------------------------
   * classDirectorInit()
   * ------------------------------------------------------------ */

  int classDirectorInit(Node *n) {
    String *declaration = Swig_director_declaration(n);
    Printf(f_directors_h, "\n");
    Printf(f_directors_h, "%s\n", declaration);
    Printf(f_directors_h, "public:\n");
    Delete(declaration);
    return Language::classDirectorInit(n);
  }

  /* ------------------------------------------------------------
   * classDirectorEnd()
   * ------------------------------------------------------------ */

  int classDirectorEnd(Node *n) {
    String *classname = Swig_class_name(n);

    if (dirprot_mode()) {
      /*
         This implementation uses a std::map<std::string,int>.

         It should be possible to rewrite it using a more elegant way,
         like copying the Java approach for the 'override' array.

         But for now, this seems to be the least intrusive way.
       */
      Printf(f_directors_h, "\n");
      Printf(f_directors_h, "/* Internal director utilities */\n");
      Printf(f_directors_h, "public:\n");
      Printf(f_directors_h, "    bool swig_get_inner(const char *swig_protected_method_name) const {\n");
      Printf(f_directors_h, "      std::map<std::string, bool>::const_iterator iv = swig_inner.find(swig_protected_method_name);\n");
      Printf(f_directors_h, "      return (iv != swig_inner.end() ? iv->second : false);\n");
      Printf(f_directors_h, "    }\n");

      Printf(f_directors_h, "    void swig_set_inner(const char *swig_protected_method_name, bool swig_val) const {\n");
      Printf(f_directors_h, "      swig_inner[swig_protected_method_name] = swig_val;\n");
      Printf(f_directors_h, "    }\n");
      Printf(f_directors_h, "private:\n");
      Printf(f_directors_h, "    mutable std::map<std::string, bool> swig_inner;\n");

    }
    if (director_method_index) {
      Printf(f_directors_h, "\n");
      Printf(f_directors_h, "#if defined(SWIG_PYTHON_DIRECTOR_VTABLE)\n");
      Printf(f_directors_h, "/* VTable implementation */\n");
      Printf(f_directors_h, "    PyObject *swig_get_method(size_t method_index, const char *method_name) const {\n");
      Printf(f_directors_h, "      PyObject *method = vtable[method_index];\n");
      Printf(f_directors_h, "      if (!method) {\n");
      Printf(f_directors_h, "        swig::SwigVar_PyObject name = SWIG_Python_str_FromChar(method_name);\n");
      Printf(f_directors_h, "        method = PyObject_GetAttr(swig_get_self(), name);\n");
      Printf(f_directors_h, "        if (!method) {\n");
      Printf(f_directors_h, "          std::string msg = \"Method in class %s doesn't exist, undefined \";\n", classname);
      Printf(f_directors_h, "          msg += method_name;\n");
      Printf(f_directors_h, "          Swig::DirectorMethodException::raise(msg.c_str());\n");
      Printf(f_directors_h, "        }\n");
      Printf(f_directors_h, "        vtable[method_index] = method;\n");
      Printf(f_directors_h, "      }\n");
      Printf(f_directors_h, "      return method;\n");
      Printf(f_directors_h, "    }\n");
      Printf(f_directors_h, "private:\n");
      Printf(f_directors_h, "    mutable swig::SwigVar_PyObject vtable[%d];\n", director_method_index);
      Printf(f_directors_h, "#endif\n\n");
    }

    Printf(f_directors_h, "};\n\n");
    return Language::classDirectorEnd(n);
  }


  /* ------------------------------------------------------------
   * classDirectorDisown()
   * ------------------------------------------------------------ */

  int classDirectorDisown(Node *n) {
    int result;
    int oldshadow = shadow;
    /* disable shadowing */
    if (shadow)
      shadow = shadow | PYSHADOW_MEMBER;
    result = Language::classDirectorDisown(n);
    shadow = oldshadow;
    if (shadow) {
      if (builtin) {
	String *rname = SwigType_namestr(real_classname);
	Printf(builtin_methods, "  { \"__disown__\", Swig::Director::swig_pyobj_disown< %s >, METH_NOARGS, \"\" },\n", rname);
	Delete(rname);
      } else {
	String *symname = Getattr(n, "sym:name");
	String *mrename = Swig_name_disown(NSPACE_TODO, symname);	//Getattr(n, "name"));
	Printv(f_shadow, tab4, "def __disown__(self):\n", NIL);
#ifdef USE_THISOWN
	Printv(f_shadow, tab8, "self.thisown = 0\n", NIL);
#else
	Printv(f_shadow, tab8, "self.this.disown()\n", NIL);
#endif
	Printv(f_shadow, tab8, module, ".", mrename, "(self)\n", NIL);
	Printv(f_shadow, tab8, "return weakref.proxy(self)\n", NIL);
	Delete(mrename);
      }
    }
    return result;
  }

  /* ----------------------------------------------------------------------------
   * END of C++ Director Class modifications
   * ------------------------------------------------------------------------- */


  /* ------------------------------------------------------------
   * classDeclaration()
   * ------------------------------------------------------------ */

  virtual int classDeclaration(Node *n) {
    if (shadow && !Getattr(n, "feature:onlychildren")) {
      Node *mod = Getattr(n, "module");
      if (mod) {
	String *modname = Getattr(mod, "name");
	Node *options = Getattr(mod, "options");
	String *pkg = options ? Getattr(options, "package") : 0;
	String *sym = Getattr(n, "sym:name");
	String *importname = import_name_string(package, mainmodule, pkg, modname, sym);
	Setattr(n, "python:proxy", importname);
	Delete(importname);
      }
    }
    int result = Language::classDeclaration(n);
    return result;
  }

  /* ------------------------------------------------------------
   * classHandler()
   * ------------------------------------------------------------ */

  String *add_explicit_scope(String *s) {
    if (!Strstr(s, "::")) {
      String *ss = NewStringf("::%s", s);
      Delete(s);
      s = ss;
    }
    return s;
  }

  void builtin_pre_decl(Node *n) {
    String *name = Getattr(n, "name");
    String *rname = add_explicit_scope(SwigType_namestr(name));
    String *mname = SwigType_manglestr(rname);

    Printf(f_init, "\n/* type '%s' */\n", rname);
    Printf(f_init, "    builtin_pytype = (PyTypeObject *)&SwigPyBuiltin_%s_type;\n", mname);
    Printf(f_init, "    builtin_pytype->tp_dict = d = PyDict_New();\n");

    Delete(rname);
    Delete(mname);
  }

  void builtin_post_decl(File *f, Node *n) {
    String *name = Getattr(n, "name");
    String *pname = Copy(name);
    SwigType_add_pointer(pname);
    String *symname = Getattr(n, "sym:name");
    String *rname = add_explicit_scope(SwigType_namestr(name));
    String *mname = SwigType_manglestr(rname);
    String *pmname = SwigType_manglestr(pname);
    String *templ = NewStringf("SwigPyBuiltin_%s", mname);
    int funpack = fastunpack;
    static String *tp_new = NewString("PyType_GenericNew");

    Printv(f_init, "  SwigPyBuiltin_SetMetaType(builtin_pytype, metatype);\n", NIL);

    // We can’t statically initialize a structure member with a function defined in another C module
    // So this is done in the initialization function instead, see https://docs.python.org/2/extending/newtypes.html
    Printf(f_init, "  builtin_pytype->tp_new = %s;\n", getSlot(n, "feature:python:tp_new", tp_new));

    Printv(f_init, "  builtin_base_count = 0;\n", NIL);
    List *baselist = Getattr(n, "bases");
    if (baselist) {
      int base_count = 0;
      for (Iterator b = First(baselist); b.item; b = Next(b)) {
	String *bname = Getattr(b.item, "name");
	if (!bname || GetFlag(b.item, "feature:ignore"))
	  continue;
	base_count++;
	String *base_name = Copy(bname);
	SwigType_add_pointer(base_name);
	String *base_mname = SwigType_manglestr(base_name);
	Printf(f_init, "  builtin_basetype = SWIG_MangledTypeQuery(\"%s\");\n", base_mname);
	Printv(f_init, "  if (builtin_basetype && builtin_basetype->clientdata && ((SwigPyClientData *) builtin_basetype->clientdata)->pytype) {\n", NIL);
	Printv(f_init, "    builtin_bases[builtin_base_count++] = ((SwigPyClientData *) builtin_basetype->clientdata)->pytype;\n", NIL);
	Printv(f_init, "  } else {\n", NIL);
	Printf(f_init, "    PyErr_SetString(PyExc_TypeError, \"Could not create type '%s' as base '%s' has not been initialized.\\n\");\n", symname, bname);
	Printv(f_init, "#if PY_VERSION_HEX >= 0x03000000\n", NIL);
	Printv(f_init, "      return NULL;\n", NIL);
	Printv(f_init, "#else\n", NIL);
	Printv(f_init, "      return;\n", NIL);
	Printv(f_init, "#endif\n", NIL);
	Printv(f_init, "  }\n", NIL);
	Delete(base_name);
	Delete(base_mname);
      }
      if (base_count > max_bases)
	max_bases = base_count;
    }
    Printv(f_init, "  builtin_bases[builtin_base_count] = NULL;\n", NIL);
    Printv(f_init, "  SwigPyBuiltin_InitBases(builtin_pytype, builtin_bases);\n", NIL);
    builtin_bases_needed = 1;

    // Check for non-public destructor, in which case tp_dealloc will issue
    // a warning and allow the memory to leak.  Any class that doesn't explicitly
    // have a private/protected destructor has an implicit public destructor.
    static String *tp_dealloc_bad = NewString("SwigPyBuiltin_BadDealloc");

    String *getset_name = NewStringf("%s_getset", templ);
    String *methods_name = NewStringf("%s_methods", templ);
    String *getset_def = NewString("");
    Printf(getset_def, "SWIGINTERN PyGetSetDef %s[] = {\n", getset_name);

    // All objects have 'this' and 'thisown' attributes
    Printv(f_init, "PyDict_SetItemString(d, \"this\", this_descr);\n", NIL);
    Printv(f_init, "PyDict_SetItemString(d, \"thisown\", thisown_descr);\n", NIL);

    // Now, the rest of the attributes
    for (Iterator member_iter = First(builtin_getset); member_iter.item; member_iter = Next(member_iter)) {
      String *memname = member_iter.key;
      Hash *mgetset = member_iter.item;
      String *getter = Getattr(mgetset, "getter");
      String *setter = Getattr(mgetset, "setter");
      const char *getter_closure = getter ? funpack ? "SwigPyBuiltin_FunpackGetterClosure" : "SwigPyBuiltin_GetterClosure" : "0";
      const char *setter_closure = setter ? funpack ? "SwigPyBuiltin_FunpackSetterClosure" : "SwigPyBuiltin_SetterClosure" : "0";
      String *gspair = NewStringf("%s_%s_getset", symname, memname);
      Printf(f, "static SwigPyGetSet %s = { %s, %s };\n", gspair, getter ? getter : "0", setter ? setter : "0");
      String *doc = Getattr(mgetset, "doc");
      if (!doc)
	doc = NewStringf("%s.%s", name, memname);
      String *entry = NewStringf("{ (char *)\"%s\", %s, %s, (char *)\"%s\", &%s }", memname, getter_closure, setter_closure, doc, gspair);
      if (GetFlag(mgetset, "static")) {
	Printf(f, "static PyGetSetDef %s_def = %s;\n", gspair, entry);
	Printf(f_init, "static_getset = SwigPyStaticVar_new_getset(metatype, &%s_def);\n", gspair);
	Printf(f_init, "PyDict_SetItemString(d, static_getset->d_getset->name, (PyObject *) static_getset);\n", memname);
	Printf(f_init, "Py_DECREF(static_getset);\n");
      } else {
	Printf(getset_def, "    %s,\n", entry);
      }
      Delete(gspair);
      Delete(entry);
    }
    Printv(f, getset_def, "    { NULL, NULL, NULL, NULL, NULL } /* Sentinel */\n", "};\n\n", NIL);

    // Rich compare function
    Hash *richcompare = Getattr(n, "python:richcompare");
    String *richcompare_func = NewStringf("%s_richcompare", templ);
    assert(richcompare);
    Printf(f, "SWIGINTERN PyObject *\n");
    Printf(f, "%s(PyObject *self, PyObject *other, int op) {\n", richcompare_func);
    Printf(f, "  PyObject *result = NULL;\n");
    if (!funpack) {
      Printf(f, "  PyObject *tuple = PyTuple_New(1);\n");
      Printf(f, "  assert(tuple);\n");
      Printf(f, "  PyTuple_SET_ITEM(tuple, 0, other);\n");
      Printf(f, "  Py_XINCREF(other);\n");
    }
    Iterator rich_iter = First(richcompare);
    if (rich_iter.item) {
      Printf(f, "  switch (op) {\n");
      for (; rich_iter.item; rich_iter = Next(rich_iter))
	Printf(f, "    case %s : result = %s(self, %s); break;\n", rich_iter.key, rich_iter.item, funpack ? "other" : "tuple");
      Printv(f, "    default : break;\n", NIL);
      Printf(f, "  }\n");
    }
    Printv(f, "  if (!result) {\n", NIL);
    Printv(f, "    if (SwigPyObject_Check(self) && SwigPyObject_Check(other)) {\n", NIL);
    Printv(f, "      result = SwigPyObject_richcompare((SwigPyObject *)self, (SwigPyObject *)other, op);\n", NIL);
    Printv(f, "    } else {\n", NIL);
    Printv(f, "      result = Py_NotImplemented;\n", NIL);
    Printv(f, "      Py_INCREF(result);\n", NIL);
    Printv(f, "    }\n", NIL);
    Printv(f, "  }\n", NIL);
    if (!funpack)
      Printf(f, "  Py_DECREF(tuple);\n");
    Printf(f, "  return result;\n");
    Printf(f, "}\n\n");

    // Methods
    Printf(f, "SWIGINTERN PyMethodDef %s_methods[] = {\n", templ);
    Dump(builtin_methods, f);
    Printf(f, "  { NULL, NULL, 0, NULL } /* Sentinel */\n};\n\n");

    // No instance dict for nondynamic objects
    if (GetFlag(n, "feature:python:nondynamic"))
      Setattr(n, "feature:python:tp_setattro", "SWIG_Python_NonDynamicSetAttr");

    Node *mod = Getattr(n, "module");
    String *modname = mod ? Getattr(mod, "name") : 0;
    String *quoted_symname;
    if (package) {
      if (modname)
	quoted_symname = NewStringf("\"%s.%s.%s\"", package, modname, symname);
      else
	quoted_symname = NewStringf("\"%s.%s\"", package, symname);
    } else {
      if (modname)
	quoted_symname = NewStringf("\"%s.%s\"", modname, symname);
      else
	quoted_symname = NewStringf("\"%s\"", symname);
    }
    String *quoted_tp_doc_str = NewStringf("\"%s\"", getSlot(n, "feature:python:tp_doc"));
    String *tp_init = NewString(builtin_tp_init ? Char(builtin_tp_init) : Swig_directorclass(n) ? "0" : "SwigPyBuiltin_BadInit");
    String *tp_flags = NewString("Py_TPFLAGS_DEFAULT|Py_TPFLAGS_BASETYPE|Py_TPFLAGS_CHECKTYPES");
    String *tp_flags_py3 = NewString("Py_TPFLAGS_DEFAULT|Py_TPFLAGS_BASETYPE");

    static String *tp_basicsize = NewStringf("sizeof(SwigPyObject)");
    static String *tp_dictoffset_default = NewString("offsetof(SwigPyObject, dict)");
    static String *tp_hash = NewString("SwigPyObject_hash");
    String *tp_as_number = NewStringf("&%s_type.as_number", templ);
    String *tp_as_sequence = NewStringf("&%s_type.as_sequence", templ);
    String *tp_as_mapping = NewStringf("&%s_type.as_mapping", templ);
    String *tp_as_buffer = NewStringf("&%s_type.as_buffer", templ);

    Printf(f, "static PyHeapTypeObject %s_type = {\n", templ);

    // PyTypeObject ht_type
    Printf(f, "  {\n");
    Printv(f, "#if PY_VERSION_HEX >= 0x03000000\n", NIL);
    Printv(f, "    PyVarObject_HEAD_INIT(NULL, 0)\n", NIL);
    Printv(f, "#else\n", NIL);
    Printf(f, "    PyObject_HEAD_INIT(NULL)\n");
    printSlot(f, getSlot(), "ob_size");
    Printv(f, "#endif\n", NIL);
    printSlot(f, quoted_symname, "tp_name");
    printSlot(f, getSlot(n, "feature:python:tp_basicsize", tp_basicsize), "tp_basicsize");
    printSlot(f, getSlot(n, "feature:python:tp_itemsize"), "tp_itemsize");
    printSlot(f, getSlot(n, "feature:python:tp_dealloc", tp_dealloc_bad), "tp_dealloc", "destructor");
    printSlot(f, getSlot(n, "feature:python:tp_print"), "tp_print", "printfunc");
    printSlot(f, getSlot(n, "feature:python:tp_getattr"), "tp_getattr", "getattrfunc");
    printSlot(f, getSlot(n, "feature:python:tp_setattr"), "tp_setattr", "setattrfunc");
    Printv(f, "#if PY_VERSION_HEX >= 0x03000000\n", NIL);
    printSlot(f, getSlot(n, "feature:python:tp_compare"), "tp_compare");
    Printv(f, "#else\n", NIL);
    printSlot(f, getSlot(n, "feature:python:tp_compare"), "tp_compare", "cmpfunc");
    Printv(f, "#endif\n", NIL);
    printSlot(f, getSlot(n, "feature:python:tp_repr"), "tp_repr", "reprfunc");
    printSlot(f, getSlot(n, "feature:python:tp_as_number", tp_as_number), "tp_as_number");
    printSlot(f, getSlot(n, "feature:python:tp_as_sequence", tp_as_sequence), "tp_as_sequence");
    printSlot(f, getSlot(n, "feature:python:tp_as_mapping", tp_as_mapping), "tp_as_mapping");
    printSlot(f, getSlot(n, "feature:python:tp_hash", tp_hash), "tp_hash", "hashfunc");
    printSlot(f, getSlot(n, "feature:python:tp_call"), "tp_call", "ternaryfunc");
    printSlot(f, getSlot(n, "feature:python:tp_str"), "tp_str", "reprfunc");
    printSlot(f, getSlot(n, "feature:python:tp_getattro"), "tp_getattro", "getattrofunc");
    printSlot(f, getSlot(n, "feature:python:tp_setattro"), "tp_setattro", "setattrofunc");
    printSlot(f, getSlot(n, "feature:python:tp_as_buffer", tp_as_buffer), "tp_as_buffer");
    Printv(f, "#if PY_VERSION_HEX >= 0x03000000\n", NIL);
    printSlot(f, getSlot(n, "feature:python:tp_flags", tp_flags_py3), "tp_flags");
    Printv(f, "#else\n", NIL);
    printSlot(f, getSlot(n, "feature:python:tp_flags", tp_flags), "tp_flags");
    Printv(f, "#endif\n", NIL);
    if (have_docstring(n)) {
      String *ds = cdocstring(n, AUTODOC_CLASS);
      String *tp_doc = NewString("");
      Printf(tp_doc, "\"%s\"", ds);
      Delete(ds);
      printSlot(f, tp_doc, "tp_doc");
      Delete(tp_doc);
    } else {
      printSlot(f, quoted_tp_doc_str, "tp_doc");
    }
    printSlot(f, getSlot(n, "feature:python:tp_traverse"), "tp_traverse", "traverseproc");
    printSlot(f, getSlot(n, "feature:python:tp_clear"), "tp_clear", "inquiry");
    printSlot(f, getSlot(n, "feature:python:tp_richcompare", richcompare_func), "tp_richcompare", "richcmpfunc");
    printSlot(f, getSlot(n, "feature:python:tp_weaklistoffset"), "tp_weaklistoffset");
    printSlot(f, getSlot(n, "feature:python:tp_iter"), "tp_iter", "getiterfunc");
    printSlot(f, getSlot(n, "feature:python:tp_iternext"), "tp_iternext", "iternextfunc");
    printSlot(f, getSlot(n, "feature:python:tp_methods", methods_name), "tp_methods");
    printSlot(f, getSlot(n, "feature:python:tp_members"), "tp_members");
    printSlot(f, getSlot(n, "feature:python:tp_getset", getset_name), "tp_getset");
    printSlot(f, getSlot(n, "feature:python:tp_base"), "tp_base");
    printSlot(f, getSlot(n, "feature:python:tp_dict"), "tp_dict");
    printSlot(f, getSlot(n, "feature:python:tp_descr_get"), "tp_descr_get", "descrgetfunc");
    printSlot(f, getSlot(n, "feature:python:tp_descr_set"), "tp_descr_set", "descrsetfunc");
    printSlot(f, getSlot(n, "feature:python:tp_dictoffset", tp_dictoffset_default), "tp_dictoffset", "Py_ssize_t");
    printSlot(f, getSlot(n, "feature:python:tp_init", tp_init), "tp_init", "initproc");
    printSlot(f, getSlot(n, "feature:python:tp_alloc"), "tp_alloc", "allocfunc");
    printSlot(f, getSlot(), "tp_new", "newfunc");
    printSlot(f, getSlot(n, "feature:python:tp_free"), "tp_free", "freefunc");
    printSlot(f, getSlot(n, "feature:python:tp_is_gc"), "tp_is_gc", "inquiry");
    printSlot(f, getSlot(n, "feature:python:tp_bases"), "tp_bases", "PyObject *");
    printSlot(f, getSlot(n, "feature:python:tp_mro"), "tp_mro", "PyObject *");
    printSlot(f, getSlot(n, "feature:python:tp_cache"), "tp_cache", "PyObject *");
    printSlot(f, getSlot(n, "feature:python:tp_subclasses"), "tp_subclasses", "PyObject *");
    printSlot(f, getSlot(n, "feature:python:tp_weaklist"), "tp_weaklist", "PyObject *");
    printSlot(f, getSlot(n, "feature:python:tp_del"), "tp_del", "destructor");
    printSlot(f, getSlot(n, "feature:python:tp_version_tag"), "tp_version_tag", "int");
    Printv(f, "#if PY_VERSION_HEX >= 0x03040000\n", NIL);
    printSlot(f, getSlot(n, "feature:python:tp_finalize"), "tp_finalize", "destructor");
    Printv(f, "#endif\n", NIL);
    Printv(f, "#if PY_VERSION_HEX >= 0x03080000\n", NIL);
    printSlot(f, getSlot(n, "feature:python:tp_vectorcall"), "tp_vectorcall", "vectorcallfunc");
    Printv(f, "#endif\n", NIL);
    Printv(f, "#if (PY_VERSION_HEX >= 0x03080000) && (PY_VERSION_HEX < 0x03090000)\n", NIL);
    printSlot(f, getSlot(), "tp_print");
    Printv(f, "#endif\n", NIL);

    Printv(f, "#ifdef COUNT_ALLOCS\n", NIL);
    printSlot(f, getSlot(n, "feature:python:tp_allocs"), "tp_allocs", "Py_ssize_t");
    printSlot(f, getSlot(n, "feature:python:tp_frees"), "tp_frees", "Py_ssize_t");
    printSlot(f, getSlot(n, "feature:python:tp_maxalloc"), "tp_maxalloc", "Py_ssize_t");
    printSlot(f, getSlot(n, "feature:python:tp_prev"), "tp_prev");
    printSlot(f, getSlot(n, "feature:python:tp_next"), "tp_next");
    Printv(f, "#endif\n", NIL);
    Printf(f, "  },\n");

    // PyAsyncMethods as_async
    Printv(f, "#if PY_VERSION_HEX >= 0x03050000\n", NIL);
    Printf(f, "  {\n");
    printSlot(f, getSlot(n, "feature:python:am_await"), "am_await", "unaryfunc");
    printSlot(f, getSlot(n, "feature:python:am_aiter"), "am_aiter", "unaryfunc");
    printSlot(f, getSlot(n, "feature:python:am_anext"), "am_anext", "unaryfunc");
    Printf(f, "  },\n");
    Printv(f, "#endif\n", NIL);

    // PyNumberMethods as_number
    Printf(f, "  {\n");
    printSlot(f, getSlot(n, "feature:python:nb_add"), "nb_add", "binaryfunc");
    printSlot(f, getSlot(n, "feature:python:nb_subtract"), "nb_subtract", "binaryfunc");
    printSlot(f, getSlot(n, "feature:python:nb_multiply"), "nb_multiply", "binaryfunc");
    Printv(f, "#if PY_VERSION_HEX < 0x03000000\n", NIL);
    printSlot(f, getSlot(n, "feature:python:nb_divide"), "nb_divide", "binaryfunc");
    Printv(f, "#endif\n", NIL);
    printSlot(f, getSlot(n, "feature:python:nb_remainder"), "nb_remainder", "binaryfunc");
    printSlot(f, getSlot(n, "feature:python:nb_divmod"), "nb_divmod", "binaryfunc");
    printSlot(f, getSlot(n, "feature:python:nb_power"), "nb_power", "ternaryfunc");
    printSlot(f, getSlot(n, "feature:python:nb_negative"), "nb_negative", "unaryfunc");
    printSlot(f, getSlot(n, "feature:python:nb_positive"), "nb_positive", "unaryfunc");
    printSlot(f, getSlot(n, "feature:python:nb_absolute"), "nb_absolute", "unaryfunc");
    printSlot(f, getSlot(n, "feature:python:nb_nonzero"), "nb_nonzero", "inquiry");
    printSlot(f, getSlot(n, "feature:python:nb_invert"), "nb_invert", "unaryfunc");
    printSlot(f, getSlot(n, "feature:python:nb_lshift"), "nb_lshift", "binaryfunc");
    printSlot(f, getSlot(n, "feature:python:nb_rshift"), "nb_rshift", "binaryfunc");
    printSlot(f, getSlot(n, "feature:python:nb_and"), "nb_and", "binaryfunc");
    printSlot(f, getSlot(n, "feature:python:nb_xor"), "nb_xor", "binaryfunc");
    printSlot(f, getSlot(n, "feature:python:nb_or"), "nb_or", "binaryfunc");
    Printv(f, "#if PY_VERSION_HEX < 0x03000000\n", NIL);
    printSlot(f, getSlot(n, "feature:python:nb_coerce"), "nb_coerce", "coercion");
    Printv(f, "#endif\n", NIL);
    printSlot(f, getSlot(n, "feature:python:nb_int"), "nb_int", "unaryfunc");
    Printv(f, "#if PY_VERSION_HEX >= 0x03000000\n", NIL);
    printSlot(f, getSlot(n, "feature:python:nb_reserved"), "nb_reserved", "void *");
    Printv(f, "#else\n", NIL);
    printSlot(f, getSlot(n, "feature:python:nb_long"), "nb_long", "unaryfunc");
    Printv(f, "#endif\n", NIL);
    printSlot(f, getSlot(n, "feature:python:nb_float"), "nb_float", "unaryfunc");
    Printv(f, "#if PY_VERSION_HEX < 0x03000000\n", NIL);
    printSlot(f, getSlot(n, "feature:python:nb_oct"), "nb_oct", "unaryfunc");
    printSlot(f, getSlot(n, "feature:python:nb_hex"), "nb_hex", "unaryfunc");
    Printv(f, "#endif\n", NIL);
    printSlot(f, getSlot(n, "feature:python:nb_inplace_add"), "nb_inplace_add", "binaryfunc");
    printSlot(f, getSlot(n, "feature:python:nb_inplace_subtract"), "nb_inplace_subtract", "binaryfunc");
    printSlot(f, getSlot(n, "feature:python:nb_inplace_multiply"), "nb_inplace_multiply", "binaryfunc");
    Printv(f, "#if PY_VERSION_HEX < 0x03000000\n", NIL);
    printSlot(f, getSlot(n, "feature:python:nb_inplace_divide"), "nb_inplace_divide", "binaryfunc");
    Printv(f, "#endif\n", NIL);
    printSlot(f, getSlot(n, "feature:python:nb_inplace_remainder"), "nb_inplace_remainder", "binaryfunc");
    printSlot(f, getSlot(n, "feature:python:nb_inplace_power"), "nb_inplace_power", "ternaryfunc");
    printSlot(f, getSlot(n, "feature:python:nb_inplace_lshift"), "nb_inplace_lshift", "binaryfunc");
    printSlot(f, getSlot(n, "feature:python:nb_inplace_rshift"), "nb_inplace_rshift", "binaryfunc");
    printSlot(f, getSlot(n, "feature:python:nb_inplace_and"), "nb_inplace_and", "binaryfunc");
    printSlot(f, getSlot(n, "feature:python:nb_inplace_xor"), "nb_inplace_xor", "binaryfunc");
    printSlot(f, getSlot(n, "feature:python:nb_inplace_or"), "nb_inplace_or", "binaryfunc");
    printSlot(f, getSlot(n, "feature:python:nb_floor_divide"), "nb_floor_divide", "binaryfunc");
    printSlot(f, getSlot(n, "feature:python:nb_divide"), "nb_true_divide", "binaryfunc");
    printSlot(f, getSlot(n, "feature:python:nb_inplace_floor_divide"), "nb_inplace_floor_divide", "binaryfunc");
    printSlot(f, getSlot(n, "feature:python:nb_inplace_divide"), "nb_inplace_true_divide", "binaryfunc");
    printSlot(f, getSlot(n, "feature:python:nb_index"), "nb_index", "unaryfunc");
    Printv(f, "#if PY_VERSION_HEX >= 0x03050000\n", NIL);
    printSlot(f, getSlot(n, "feature:python:nb_matrix_multiply"), "nb_matrix_multiply", "binaryfunc");
    printSlot(f, getSlot(n, "feature:python:nb_inplace_matrix_multiply"), "nb_inplace_matrix_multiply", "binaryfunc");
    Printv(f, "#endif\n", NIL);
    Printf(f, "  },\n");

    // PyMappingMethods as_mapping;
    Printf(f, "  {\n");
    printSlot(f, getSlot(n, "feature:python:mp_length"), "mp_length", "lenfunc");
    printSlot(f, getSlot(n, "feature:python:mp_subscript"), "mp_subscript", "binaryfunc");
    printSlot(f, getSlot(n, "feature:python:mp_ass_subscript"), "mp_ass_subscript", "objobjargproc");
    Printf(f, "  },\n");

    // PySequenceMethods as_sequence;
    Printf(f, "  {\n");
    printSlot(f, getSlot(n, "feature:python:sq_length"), "sq_length", "lenfunc");
    printSlot(f, getSlot(n, "feature:python:sq_concat"), "sq_concat", "binaryfunc");
    printSlot(f, getSlot(n, "feature:python:sq_repeat"), "sq_repeat", "ssizeargfunc");
    printSlot(f, getSlot(n, "feature:python:sq_item"), "sq_item", "ssizeargfunc");
    Printv(f, "#if PY_VERSION_HEX >= 0x03000000\n", NIL);
    printSlot(f, getSlot(n, "feature:python:was_sq_slice"), "was_sq_slice", "void *");
    Printv(f, "#else\n", NIL);
    printSlot(f, getSlot(n, "feature:python:sq_slice"), "sq_slice", "ssizessizeargfunc");
    Printv(f, "#endif\n", NIL);
    printSlot(f, getSlot(n, "feature:python:sq_ass_item"), "sq_ass_item", "ssizeobjargproc");
    Printv(f, "#if PY_VERSION_HEX >= 0x03000000\n", NIL);
    printSlot(f, getSlot(n, "feature:python:was_sq_ass_slice"), "was_sq_ass_slice", "void *");
    Printv(f, "#else\n", NIL);
    printSlot(f, getSlot(n, "feature:python:sq_ass_slice"), "sq_ass_slice", "ssizessizeobjargproc");
    Printv(f, "#endif\n", NIL);
    printSlot(f, getSlot(n, "feature:python:sq_contains"), "sq_contains", "objobjproc");
    printSlot(f, getSlot(n, "feature:python:sq_inplace_concat"), "sq_inplace_concat", "binaryfunc");
    printSlot(f, getSlot(n, "feature:python:sq_inplace_repeat"), "sq_inplace_repeat", "ssizeargfunc");
    Printf(f, "  },\n");

    // PyBufferProcs as_buffer;
    Printf(f, "  {\n");
    Printv(f, "#if PY_VERSION_HEX < 0x03000000\n", NIL);
    printSlot(f, getSlot(n, "feature:python:bf_getreadbuffer"), "bf_getreadbuffer", "readbufferproc");
    printSlot(f, getSlot(n, "feature:python:bf_getwritebuffer"), "bf_getwritebuffer", "writebufferproc");
    printSlot(f, getSlot(n, "feature:python:bf_getsegcount"), "bf_getsegcount", "segcountproc");
    printSlot(f, getSlot(n, "feature:python:bf_getcharbuffer"), "bf_getcharbuffer", "charbufferproc");
    Printv(f, "#endif\n", NIL);
    printSlot(f, getSlot(n, "feature:python:bf_getbuffer"), "bf_getbuffer", "getbufferproc");
    printSlot(f, getSlot(n, "feature:python:bf_releasebuffer"), "bf_releasebuffer", "releasebufferproc");
    Printf(f, "  },\n");

    // PyObject *ht_name, *ht_slots, *ht_qualname;
    printSlot(f, getSlot(n, "feature:python:ht_name"), "ht_name", "PyObject *");
    printSlot(f, getSlot(n, "feature:python:ht_slots"), "ht_slots", "PyObject *");
    Printv(f, "#if PY_VERSION_HEX >= 0x03030000\n", NIL);
    printSlot(f, getSlot(n, "feature:python:ht_qualname"), "ht_qualname", "PyObject *");

    // struct _dictkeysobject *ht_cached_keys;
    printSlot(f, getSlot(n, "feature:python:ht_cached_keys"), "ht_cached_keys");
    Printv(f, "#endif\n", NIL);
    Printf(f, "};\n\n");

    String *clientdata = NewString("");
    Printf(clientdata, "&%s_clientdata", templ);
    SwigType_remember_mangleddata(pmname, clientdata);

    SwigType *smart = Swig_cparse_smartptr(n);
    if (smart) {
      SwigType_add_pointer(smart);
      String *smart_pmname = SwigType_manglestr(smart);
      SwigType_remember_mangleddata(smart_pmname, clientdata);
      Delete(smart_pmname);
    }

    String *clientdata_klass = NewString("0");
    if (GetFlag(n, "feature:implicitconv")) {
      Clear(clientdata_klass);
      Printf(clientdata_klass, "(PyObject *) &%s_type", templ);
    }

    Printf(f, "SWIGINTERN SwigPyClientData %s_clientdata = {%s, 0, 0, 0, 0, 0, (PyTypeObject *)&%s_type};\n\n", templ, clientdata_klass, templ);

    Printv(f_init, "    if (PyType_Ready(builtin_pytype) < 0) {\n", NIL);
    Printf(f_init, "      PyErr_SetString(PyExc_TypeError, \"Could not create type '%s'.\");\n", symname);
    Printv(f_init, "#if PY_VERSION_HEX >= 0x03000000\n", NIL);
    Printv(f_init, "      return NULL;\n", NIL);
    Printv(f_init, "#else\n", NIL);
    Printv(f_init, "      return;\n", NIL);
    Printv(f_init, "#endif\n", NIL);
    Printv(f_init, "    }\n", NIL);
    Printv(f_init, "    Py_INCREF(builtin_pytype);\n", NIL);
    Printf(f_init, "    PyModule_AddObject(m, \"%s\", (PyObject *)builtin_pytype);\n", symname);
    Printf(f_init, "    SwigPyBuiltin_AddPublicSymbol(public_interface, \"%s\");\n", symname);
    Printv(f_init, "    d = md;\n", NIL);

    Delete(clientdata);
    Delete(smart);
    Delete(rname);
    Delete(pname);
    Delete(mname);
    Delete(pmname);
    Delete(templ);
    Delete(tp_flags);
    Delete(tp_flags_py3);
    Delete(tp_as_buffer);
    Delete(tp_as_mapping);
    Delete(tp_as_sequence);
    Delete(tp_as_number);
    Delete(quoted_symname);
    Delete(quoted_tp_doc_str);
    Delete(tp_init);
    Delete(clientdata_klass);
    Delete(richcompare_func);
    Delete(getset_name);
    Delete(methods_name);
  }

  virtual int classHandler(Node *n) {
    File *f_shadow_file = f_shadow;
    Node *base_node = NULL;

    if (shadow) {

      /* Create new strings for building up a wrapper function */
      have_constructor = 0;
      have_repr = 0;

      class_name = Getattr(n, "sym:name");
      real_classname = Getattr(n, "name");

      if (!addSymbol(class_name, n))
	return SWIG_ERROR;

      if (builtin) {
	List *baselist = Getattr(n, "bases");
	if (baselist && Len(baselist) > 0) {
	  Iterator b = First(baselist);
	  base_node = b.item;
	}
      }

      shadow_indent = (String *) tab4;

      /* Handle inheritance */
      String *base_class = NewString("");
      List *baselist = Getattr(n, "bases");
      if (baselist && Len(baselist)) {
	Iterator b;
	b = First(baselist);
	while (b.item) {
	  String *bname = Getattr(b.item, "python:proxy");
	  bool ignore = GetFlag(b.item, "feature:ignore") ? true : false;
	  if (!bname || ignore) {
	    if (!bname && !ignore) {
	      Swig_warning(WARN_TYPE_UNDEFINED_CLASS, Getfile(n), Getline(n),
			   "Base class '%s' ignored - unknown module name for base. Either import the appropriate module interface file or specify the name of the module in the %%import directive.\n",
			   SwigType_namestr(Getattr(b.item, "name")));
	    }
	    b = Next(b);
	    continue;
	  }
	  Printv(base_class, bname, NIL);
	  b = Next(b);
	  if (b.item) {
            Printv(base_class, ", ", NIL);
	  }
	}
      }

      if (builtin) {
	Hash *base_richcompare = NULL;
	Hash *richcompare = NULL;
	if (base_node)
	  base_richcompare = Getattr(base_node, "python:richcompare");
	if (base_richcompare)
	  richcompare = Copy(base_richcompare);
	else
	  richcompare = NewHash();
	Setattr(n, "python:richcompare", richcompare);
      }

      /* dealing with abstract base class */
      String *abcs = Getattr(n, "feature:python:abc");
      if (py3 && abcs) {
	if (Len(base_class)) {
	  Printv(base_class, ", ", NIL);
	}
	Printv(base_class, abcs, NIL);
      }

      if (builtin) {
	if (have_docstring(n)) {
	  String *ds = cdocstring(n, AUTODOC_CLASS);
	  Setattr(n, "feature:python:tp_doc", ds);
	  Delete(ds);
	} else {
	  String *name = Getattr(n, "name");
	  String *rname = add_explicit_scope(SwigType_namestr(name));
	  Setattr(n, "feature:python:tp_doc", rname);
	  Delete(rname);
	}
      } else {
	if (!py3) {
	  if (GetFlag(n, "feature:python:nondynamic"))
	    Printv(f_shadow, "@_swig_add_metaclass(_SwigNonDynamicMeta)\n", NIL);
	}
	Printv(f_shadow, "class ", class_name, NIL);

	if (Len(base_class)) {
	  Printf(f_shadow, "(%s)", base_class);
	} else {
	  if (GetFlag(n, "feature:exceptionclass")) {
	    Printf(f_shadow, "(Exception)");
	  } else {
	    Printf(f_shadow, "(object");
	    Printf(f_shadow, py3 && GetFlag(n, "feature:python:nondynamic") ? ", metaclass=_SwigNonDynamicMeta" : "", ")");
	    Printf(f_shadow, ")");
	  }
	}

	Printf(f_shadow, ":\n");

	// write docstrings if requested
	if (have_docstring(n)) {
	  String *str = docstring(n, AUTODOC_CLASS, tab4);
	  if (str && Len(str))
	    Printv(f_shadow, tab4, str, "\n\n", NIL);
	}

	Printv(f_shadow, tab4, "thisown = property(lambda x: x.this.own(), ", "lambda x, v: x.this.own(v), doc=\"The membership flag\")\n", NIL);
	/* Add static attribute */
	if (GetFlag(n, "feature:python:nondynamic")) {
	  Printv(f_shadow_file, tab4, "__setattr__ = _swig_setattr_nondynamic_instance_variable(object.__setattr__)\n", NIL);
	}
      }
    }

    /* Emit all of the members */

    in_class = 1;
    if (builtin)
      builtin_pre_decl(n);

    /* Override the shadow file so we can capture its methods */
    f_shadow = NewString("");

    // Set up type check for director class constructor
    Clear(none_comparison);
    if (builtin && Swig_directorclass(n)) {
      String *p_real_classname = Copy(real_classname);
      SwigType_add_pointer(p_real_classname);
      String *mangle = SwigType_manglestr(p_real_classname);
      String *descriptor = NewStringf("SWIGTYPE%s", mangle);
      Printv(none_comparison, "self->ob_type != ((SwigPyClientData *)(", descriptor, ")->clientdata)->pytype", NIL);
      Delete(descriptor);
      Delete(mangle);
      Delete(p_real_classname);
    } else {
      Printv(none_comparison, "$arg != Py_None", NIL);
    }

    Language::classHandler(n);

    in_class = 0;

    /* Complete the class */
    if (shadow) {
      /* Generate a class registration function */
      // Replace storing a pointer to underlying class with a smart pointer (intended for use with non-intrusive smart pointers)
      SwigType *smart = Swig_cparse_smartptr(n);
      SwigType *ct = Copy(smart ? smart : real_classname);
      SwigType_add_pointer(ct);
      SwigType *realct = Copy(real_classname);
      SwigType_add_pointer(realct);
      SwigType_remember(realct);
      if (builtin) {
	Printv(f_wrappers, builtin_closures_code, NIL);
	Delete(builtin_closures_code);
	builtin_closures_code = NewString("");
	Clear(builtin_closures);
      } else {
	Printv(f_wrappers, "SWIGINTERN PyObject *", class_name, "_swigregister(PyObject *SWIGUNUSEDPARM(self), PyObject *args) {\n", NIL);
	Printv(f_wrappers, "  PyObject *obj;\n", NIL);
	Printv(f_wrappers, "  if (!SWIG_Python_UnpackTuple(args, \"swigregister\", 1, 1, &obj)) return NULL;\n", NIL);

	Printv(f_wrappers,
	       "  SWIG_TypeNewClientData(SWIGTYPE", SwigType_manglestr(ct), ", SWIG_NewClientData(obj));\n", "  return SWIG_Py_Void();\n", "}\n\n", NIL);
	String *cname = NewStringf("%s_swigregister", class_name);
	add_method(cname, cname, 0, 0, 1, 1, 1);
	Delete(cname);
      }
      Delete(smart);
      Delete(ct);
      Delete(realct);
      if (!have_constructor) {
	if (!builtin)
	  Printv(f_shadow_file, "\n", tab4, "def __init__(self, *args, **kwargs):\n", tab8, "raise AttributeError(\"", "No constructor defined",
		 (Getattr(n, "abstracts") ? " - class is abstract" : ""), "\")\n", NIL);
      } else if (!builtin) {

	Printv(f_wrappers, "SWIGINTERN PyObject *", class_name, "_swiginit(PyObject *SWIGUNUSEDPARM(self), PyObject *args) {\n", NIL);
	Printv(f_wrappers, "  return SWIG_Python_InitShadowInstance(args);\n", "}\n\n", NIL);
	String *cname = NewStringf("%s_swiginit", class_name);
	add_method(cname, cname, 0);
	Delete(cname);
      }
      if (!have_repr && !builtin) {
	/* Supply a repr method for this class  */
	String *rname = SwigType_namestr(real_classname);
	Printv(f_shadow_file, tab4, "__repr__ = _swig_repr\n", NIL);
	Delete(rname);
      }

      if (builtin)
	builtin_post_decl(f_builtins, n);

      if (builtin_tp_init) {
	Delete(builtin_tp_init);
	builtin_tp_init = 0;
      }

      if (!builtin) {
	/* Now emit methods */
	Printv(f_shadow_file, f_shadow, NIL);
	Printf(f_shadow_file, "\n");
	Printf(f_shadow_file, "# Register %s in %s:\n", class_name, module);
	Printf(f_shadow_file, "%s.%s_swigregister(%s)\n", module, class_name, class_name);
      }

      shadow_indent = 0;
      Printf(f_shadow_file, "%s\n", f_shadow_stubs);
      Clear(f_shadow_stubs);
    }

    if (builtin) {
      Clear(class_members);
      Clear(builtin_getset);
      Clear(builtin_methods);
    }

    /* Restore shadow file back to original version */
    Delete(f_shadow);
    f_shadow = f_shadow_file;

    return SWIG_OK;
  }

  /* ------------------------------------------------------------
   * functionHandler()  -  Mainly overloaded for callback handling
   * ------------------------------------------------------------ */

  virtual int functionHandler(Node *n) {
    String *pcb = GetFlagAttr(n, "feature:python:callback");
    if (pcb) {
      if (Strcmp(pcb, "1") == 0) {
	SetFlagAttr(n, "feature:callback", "%s_cb_ptr");
      } else {
	SetFlagAttr(n, "feature:callback", pcb);
      }
      autodoc_l dlevel = autodoc_level(Getattr(n, "feature:autodoc"));
      if (dlevel != NO_AUTODOC && dlevel > TYPES_AUTODOC) {
	Setattr(n, "feature:autodoc", "1");
      }
    }
    return Language::functionHandler(n);
  }

  /* ------------------------------------------------------------
   * memberfunctionHandler()
   * ------------------------------------------------------------ */

  virtual int memberfunctionHandler(Node *n) {
    String *symname = Getattr(n, "sym:name");
    int oldshadow;

    if (builtin)
      Swig_save("builtin_memberfunc", n, "python:argcount", NIL);

    /* Create the default member function */
    oldshadow = shadow;		/* Disable shadowing when wrapping member functions */
    if (shadow)
      shadow = shadow | PYSHADOW_MEMBER;
    Language::memberfunctionHandler(n);
    shadow = oldshadow;

    if (builtin && in_class) {
      // Can't use checkAttribute(n, "access", "public") because
      // "access" attr isn't set on %extend methods
      if (!checkAttribute(n, "access", "private") && strncmp(Char(symname), "operator ", 9) && !Getattr(class_members, symname)) {
	String *fullname = Swig_name_member(NSPACE_TODO, class_name, symname);
	String *wname = Swig_name_wrapper(fullname);
	Setattr(class_members, symname, n);
	int argcount = Getattr(n, "python:argcount") ? atoi(Char(Getattr(n, "python:argcount"))) : 2;
	String *ds = have_docstring(n) ? cdocstring(n, AUTODOC_METHOD) : NewString("");
	if (check_kwargs(n)) {
	  // Cast via void(*)(void) to suppress GCC -Wcast-function-type
	  // warning.  Python should always call the function correctly, but
	  // the Python C API requires us to store it in function pointer of a
	  // different type.
	  Printf(builtin_methods, "  { \"%s\", (PyCFunction)(void(*)(void))%s, METH_VARARGS|METH_KEYWORDS, \"%s\" },\n", symname, wname, ds);
	} else if (argcount == 0) {
	  Printf(builtin_methods, "  { \"%s\", %s, METH_NOARGS, \"%s\" },\n", symname, wname, ds);
	} else if (argcount == 1) {
	  Printf(builtin_methods, "  { \"%s\", %s, METH_O, \"%s\" },\n", symname, wname, ds);
	} else {
	  Printf(builtin_methods, "  { \"%s\", %s, METH_VARARGS, \"%s\" },\n", symname, wname, ds);
	}
	Delete(fullname);
	Delete(wname);
	Delete(ds);
      }
    }

    if (builtin)
      Swig_restore(n);

    if (!Getattr(n, "sym:nextSibling")) {
      if (shadow && !builtin) {
	int fproxy = fastproxy;
	String *fullname = Swig_name_member(NSPACE_TODO, class_name, symname);
	if (Strcmp(symname, "__repr__") == 0) {
	  have_repr = 1;
	}
	if (Getattr(n, "feature:shadow")) {
	  String *pycode = indent_pythoncode(Getattr(n, "feature:shadow"), tab4, Getfile(n), Getline(n), "%feature(\"shadow\")");
	  String *pyaction = NewStringf("%s.%s", module, fullname);
	  Replaceall(pycode, "$action", pyaction);
	  Delete(pyaction);
	  Printv(f_shadow, pycode, "\n", NIL);
	  Delete(pycode);
	  fproxy = 0;
	} else {
	  int allow_kwargs = (check_kwargs(n) && !Getattr(n, "sym:overloaded")) ? 1 : 0;
	  String *parms = make_pyParmList(n, true, false, allow_kwargs);
	  String *callParms = make_pyParmList(n, true, true, allow_kwargs);
	  if (!have_addtofunc(n)) {
	    if (!fastproxy || olddefs) {
	      Printv(f_shadow, "\n", tab4, "def ", symname, "(", parms, ")", returnTypeAnnotation(n), ":\n", NIL);
	      if (have_docstring(n))
		Printv(f_shadow, tab8, docstring(n, AUTODOC_METHOD, tab8), "\n", NIL);
	      Printv(f_shadow, tab8, "return ", funcCall(fullname, callParms), "\n", NIL);
	    }
	  } else {
	    Printv(f_shadow, "\n", tab4, "def ", symname, "(", parms, ")", returnTypeAnnotation(n), ":\n", NIL);
	    if (have_docstring(n))
	      Printv(f_shadow, tab8, docstring(n, AUTODOC_METHOD, tab8), "\n", NIL);
	    if (have_pythonprepend(n)) {
	      fproxy = 0;
	      Printv(f_shadow, indent_pythoncode(pythonprepend(n), tab8, Getfile(n), Getline(n), "%pythonprepend or %feature(\"pythonprepend\")"), "\n", NIL);
	    }
	    if (have_pythonappend(n)) {
	      fproxy = 0;
	      Printv(f_shadow, tab8, "val = ", funcCall(fullname, callParms), "\n", NIL);
	      Printv(f_shadow, indent_pythoncode(pythonappend(n), tab8, Getfile(n), Getline(n), "%pythonappend or %feature(\"pythonappend\")"), "\n", NIL);
	      Printv(f_shadow, tab8, "return val\n\n", NIL);
	    } else {
	      Printv(f_shadow, tab8, "return ", funcCall(fullname, callParms), "\n\n", NIL);
	    }
	  }
	}
	if (fproxy) {
	  Printf(f_shadow, tab4);
	  Printf(f_shadow, "%s = _swig_new_instance_method(%s.%s)\n", symname, module, Swig_name_member(NSPACE_TODO, class_name, symname));
	}
	Delete(fullname);
      }
    }
    return SWIG_OK;
  }


  /* ------------------------------------------------------------
   * staticmemberfunctionHandler()
   * ------------------------------------------------------------ */

  virtual int staticmemberfunctionHandler(Node *n) {
    String *symname = Getattr(n, "sym:name");
    if (builtin && in_class) {
      Swig_save("builtin_memberconstantHandler", n, "pybuiltin:symname", NIL);
      Setattr(n, "pybuiltin:symname", symname);
    }
    Language::staticmemberfunctionHandler(n);
    if (builtin && in_class) {
      Swig_restore(n);
    }

    if (builtin && in_class) {
      if ((GetFlagAttr(n, "feature:extend") || checkAttribute(n, "access", "public"))
	  && !Getattr(class_members, symname)) {
	String *fullname = Swig_name_member(NSPACE_TODO, class_name, symname);
	String *wname = Swig_name_wrapper(fullname);
	Setattr(class_members, symname, n);
	int funpack = fastunpack && !Getattr(n, "sym:overloaded");
	String *pyflags = NewString("METH_STATIC|");
	int argcount = Getattr(n, "python:argcount") ? atoi(Char(Getattr(n, "python:argcount"))) : 2;
	if (funpack && argcount == 0)
	  Append(pyflags, "METH_NOARGS");
	else if (funpack && argcount == 1)
	  Append(pyflags, "METH_O");
	else
	  Append(pyflags, "METH_VARARGS");
	// Cast via void(*)(void) to suppress GCC -Wcast-function-type warning.
	// Python should always call the function correctly, but the Python C
	// API requires us to store it in function pointer of a different type.
	if (have_docstring(n)) {
	  String *ds = cdocstring(n, AUTODOC_STATICFUNC);
	  Printf(builtin_methods, "  { \"%s\", (PyCFunction)(void(*)(void))%s, %s, \"%s\" },\n", symname, wname, pyflags, ds);
	  Delete(ds);
	} else {
	  Printf(builtin_methods, "  { \"%s\", (PyCFunction)(void(*)(void))%s, %s, \"\" },\n", symname, wname, pyflags);
	}
	Delete(fullname);
	Delete(wname);
	Delete(pyflags);
      }
      return SWIG_OK;
    }

    if (Getattr(n, "sym:nextSibling")) {
      return SWIG_OK;
    }

    if (shadow) {
      String *staticfunc_name = NewString(fastproxy ? "_swig_new_static_method" : "staticmethod");
      bool fast = (fastproxy && !have_addtofunc(n)) || Getattr(n, "feature:callback");
      if (!fast || olddefs) {
	int kw = (check_kwargs(n) && !Getattr(n, "sym:overloaded")) ? 1 : 0;
	String *parms = make_pyParmList(n, false, false, kw);
	String *callParms = make_pyParmList(n, false, true, kw);
	Printv(f_shadow, "\n", tab4, "@staticmethod", NIL);
	Printv(f_shadow, "\n", tab4, "def ", symname, "(", parms, ")", returnTypeAnnotation(n), ":\n", NIL);
	if (have_docstring(n))
	  Printv(f_shadow, tab8, docstring(n, AUTODOC_STATICFUNC, tab8), "\n", NIL);
	if (have_pythonprepend(n))
	  Printv(f_shadow, indent_pythoncode(pythonprepend(n), tab8, Getfile(n), Getline(n), "%pythonprepend or %feature(\"pythonprepend\")"), "\n", NIL);
	if (have_pythonappend(n)) {
	  Printv(f_shadow, tab8, "val = ", funcCall(Swig_name_member(NSPACE_TODO, class_name, symname), callParms), "\n", NIL);
	  Printv(f_shadow, indent_pythoncode(pythonappend(n), tab8, Getfile(n), Getline(n), "%pythonappend or %feature(\"pythonappend\")"), "\n", NIL);
	  Printv(f_shadow, tab8, "return val\n", NIL);
	} else {
	  Printv(f_shadow, tab8, "return ", funcCall(Swig_name_member(NSPACE_TODO, class_name, symname), callParms), "\n", NIL);
	}
      }

      // Below may result in a 2nd definition of the method when -olddefs is used. The Python interpreter will use the second definition as it overwrites the first.
      if (fast) {
	Printv(f_shadow, tab4, symname, " = ", staticfunc_name, "(", module, ".", Swig_name_member(NSPACE_TODO, class_name, symname),
	       ")\n", NIL);
      }
      Delete(staticfunc_name);
    }
    return SWIG_OK;
  }

  /* ------------------------------------------------------------
   * constructorDeclaration()
   * ------------------------------------------------------------ */

  virtual int constructorHandler(Node *n) {
    String *symname = Getattr(n, "sym:name");
    int oldshadow = shadow;
    int use_director = Swig_directorclass(n);

    /*
     * If we're wrapping the constructor of a C++ director class, prepend a new parameter
     * to receive the scripting language object (e.g. 'self')
     *
     */
    Swig_save("python:constructorHandler", n, "parms", NIL);
    if (use_director) {
      Parm *parms = Getattr(n, "parms");
      Parm *self;
      String *name = NewString("self");
      String *type = NewString("PyObject");
      SwigType_add_pointer(type);
      self = NewParm(type, name, n);
      Delete(type);
      Delete(name);
      Setattr(self, "lname", "O");
      if (parms)
	set_nextSibling(self, parms);
      Setattr(n, "parms", self);
      Setattr(n, "wrap:self", "1");
      Setattr(n, "hidden", "1");
      Delete(self);
    }

    if (shadow)
      shadow = shadow | PYSHADOW_MEMBER;
    Language::constructorHandler(n);
    shadow = oldshadow;

    Delattr(n, "wrap:self");
    Swig_restore(n);

    if (!Getattr(n, "sym:nextSibling")) {
      if (shadow) {
	int allow_kwargs = (check_kwargs(n) && (!Getattr(n, "sym:overloaded"))) ? 1 : 0;
	int handled_as_init = 0;
	if (!have_constructor) {
	  String *nname = Getattr(n, "sym:name");
	  String *sname = Getattr(getCurrentClass(), "sym:name");
	  String *cname = Swig_name_construct(NSPACE_TODO, sname);
	  handled_as_init = (Strcmp(nname, sname) == 0) || (Strcmp(nname, cname) == 0);
	  Delete(cname);
	}

	String *subfunc = Swig_name_construct(NSPACE_TODO, symname);
	if (!have_constructor && handled_as_init) {
	  if (!builtin) {
	    if (Getattr(n, "feature:shadow")) {
	      String *pycode = indent_pythoncode(Getattr(n, "feature:shadow"), tab4, Getfile(n), Getline(n), "%feature(\"shadow\")");
	      String *pyaction = NewStringf("%s.%s", module, subfunc);
	      Replaceall(pycode, "$action", pyaction);
	      Delete(pyaction);
	      Printv(f_shadow, pycode, "\n", NIL);
	      Delete(pycode);
	    } else {
	      String *pass_self = NewString("");
	      Node *parent = Swig_methodclass(n);
	      String *classname = Swig_class_name(parent);
	      String *rclassname = Swig_class_name(getCurrentClass());
	      assert(rclassname);

	      String *parms = make_pyParmList(n, true, false, allow_kwargs);
	      /* Pass 'self' only if using director */
	      String *callParms = make_pyParmList(n, false, true, allow_kwargs, true);

	      if (use_director) {
		Insert(callParms, 0, "_self, ");
		Printv(pass_self, tab8, NIL);
		Printf(pass_self, "if self.__class__ == %s:\n", classname);
		//Printv(pass_self, tab8, tab4, "args = (None,) + args\n", tab8, "else:\n", tab8, tab4, "args = (self,) + args\n", NIL);
		Printv(pass_self, tab8, tab4, "_self = None\n", tab8, "else:\n", tab8, tab4, "_self = self\n", NIL);
	      }

	      Printv(f_shadow, "\n", tab4, "def __init__(", parms, ")", returnTypeAnnotation(n), ":\n", NIL);
	      if (have_docstring(n))
		Printv(f_shadow, tab8, docstring(n, AUTODOC_CTOR, tab8), "\n", NIL);
	      if (have_pythonprepend(n))
		Printv(f_shadow, indent_pythoncode(pythonprepend(n), tab8, Getfile(n), Getline(n), "%pythonprepend or %feature(\"pythonprepend\")"), "\n", NIL);
	      Printv(f_shadow, pass_self, NIL);
	      Printv(f_shadow, tab8, module, ".", class_name, "_swiginit(self, ", funcCall(subfunc, callParms), ")\n", NIL);
	      if (have_pythonappend(n))
		Printv(f_shadow, indent_pythoncode(pythonappend(n), tab8, Getfile(n), Getline(n), "%pythonappend or %feature(\"pythonappend\")"), "\n\n", NIL);
	      Delete(pass_self);
	    }
	    have_constructor = 1;
	  }
	} else {
	  /* Hmmm. We seem to be creating a different constructor.  We're just going to create a
	     function for it. */
	  if (!builtin) {
	    if (Getattr(n, "feature:shadow")) {
	      String *pycode = indent_pythoncode(Getattr(n, "feature:shadow"), "", Getfile(n), Getline(n), "%feature(\"shadow\")");
	      String *pyaction = NewStringf("%s.%s", module, subfunc);
	      Replaceall(pycode, "$action", pyaction);
	      Delete(pyaction);
	      Printv(f_shadow_stubs, pycode, "\n", NIL);
	      Delete(pycode);
	    } else {
	      String *parms = make_pyParmList(n, false, false, allow_kwargs);
	      String *callParms = make_pyParmList(n, false, true, allow_kwargs);

	      Printv(f_shadow_stubs, "\ndef ", symname, "(", parms, ")", returnTypeAnnotation(n), ":\n", NIL);
	      if (have_docstring(n))
		Printv(f_shadow_stubs, tab4, docstring(n, AUTODOC_CTOR, tab4), "\n", NIL);
	      if (have_pythonprepend(n))
		Printv(f_shadow_stubs, indent_pythoncode(pythonprepend(n), tab4, Getfile(n), Getline(n), "%pythonprepend or %feature(\"pythonprepend\")"), "\n", NIL);
	      Printv(f_shadow_stubs, tab4, "val = ", funcCall(subfunc, callParms), "\n", NIL);
#ifdef USE_THISOWN
	      Printv(f_shadow_stubs, tab4, "val.thisown = 1\n", NIL);
#endif
	      if (have_pythonappend(n))
		Printv(f_shadow_stubs, indent_pythoncode(pythonappend(n), tab4, Getfile(n), Getline(n), "%pythonappend or %feature(\"pythonappend\")"), "\n", NIL);
	      Printv(f_shadow_stubs, tab4, "return val\n", NIL);
	    }
	  } else {
	    Printf(f_shadow_stubs, "%s = %s\n", symname, subfunc);
	  }
	}
	Delete(subfunc);
      }
    }
    return SWIG_OK;
  }

  /* ------------------------------------------------------------
   * destructorHandler()
   * ------------------------------------------------------------ */

  virtual int destructorHandler(Node *n) {
    String *symname = Getattr(n, "sym:name");
    int oldshadow = shadow;

    if (builtin && in_class) {
      Node *cls = Swig_methodclass(n);
      // Use the destructor for the tp_dealloc slot unless a user overrides it with another method
      if (!Getattr(cls, "feature:python:tp_dealloc")) {
	Setattr(n, "feature:python:slot", "tp_dealloc");
	Setattr(n, "feature:python:slot:functype", "destructor");
      }
    }

    if (shadow)
      shadow = shadow | PYSHADOW_MEMBER;
    //Setattr(n,"emit:dealloc","1");
    Language::destructorHandler(n);
    shadow = oldshadow;

    if (shadow) {
      if (Getattr(n, "feature:shadow")) {
	String *pycode = indent_pythoncode(Getattr(n, "feature:shadow"), tab4, Getfile(n), Getline(n), "%feature(\"shadow\")");
	String *pyaction = NewStringf("%s.%s", module, Swig_name_destroy(NSPACE_TODO, symname));
	Replaceall(pycode, "$action", pyaction);
	Delete(pyaction);
	Printv(f_shadow, pycode, "\n", NIL);
	Delete(pycode);
      } else {
	Printv(f_shadow, tab4, "__swig_destroy__ = ", module, ".", Swig_name_destroy(NSPACE_TODO, symname), "\n", NIL);
	if (!have_pythonprepend(n) && !have_pythonappend(n)) {
	  return SWIG_OK;
	}
	Printv(f_shadow, tab4, "def __del__(self):\n", NIL);
	if (have_docstring(n))
	  Printv(f_shadow, tab8, docstring(n, AUTODOC_DTOR, tab8), "\n", NIL);
	if (have_pythonprepend(n))
	  Printv(f_shadow, indent_pythoncode(pythonprepend(n), tab8, Getfile(n), Getline(n), "%pythonprepend or %feature(\"pythonprepend\")"), "\n", NIL);
#ifdef USE_THISOWN
	Printv(f_shadow, tab8, "try:\n", NIL);
	Printv(f_shadow, tab8, tab4, "if self.thisown:", module, ".", Swig_name_destroy(NSPACE_TODO, symname), "(self)\n", NIL);
	Printv(f_shadow, tab8, "except __builtin__.Exception: pass\n", NIL);
#else
#endif
	if (have_pythonappend(n))
	  Printv(f_shadow, indent_pythoncode(pythonappend(n), tab8, Getfile(n), Getline(n), "%pythonappend or %feature(\"pythonappend\")"), "\n", NIL);
	Printv(f_shadow, tab8, "pass\n", NIL);
	Printv(f_shadow, "\n", NIL);
      }
    }
    return SWIG_OK;
  }

  /* ------------------------------------------------------------
   * membervariableHandler()
   * ------------------------------------------------------------ */

  virtual int membervariableHandler(Node *n) {
    String *symname = Getattr(n, "sym:name");

    int oldshadow = shadow;
    if (shadow)
      shadow = shadow | PYSHADOW_MEMBER;
    Language::membervariableHandler(n);
    shadow = oldshadow;

    if (shadow && !builtin) {
      String *mname = Swig_name_member(NSPACE_TODO, class_name, symname);
      String *setname = Swig_name_set(NSPACE_TODO, mname);
      String *getname = Swig_name_get(NSPACE_TODO, mname);
      int assignable = is_assignable(n);
      Printv(f_shadow, tab4, symname, " = property(", module, ".", getname, NIL);
      if (assignable)
	Printv(f_shadow, ", ", module, ".", setname, NIL);
      if (have_docstring(n))
	Printv(f_shadow, ", doc=", docstring(n, AUTODOC_VAR, tab4), NIL);
      Printv(f_shadow, ")\n", NIL);
      Delete(mname);
      Delete(setname);
      Delete(getname);
    }

    return SWIG_OK;
  }

  /* ------------------------------------------------------------
   * staticmembervariableHandler()
   * ------------------------------------------------------------ */

  virtual int staticmembervariableHandler(Node *n) {
    Swig_save("builtin_staticmembervariableHandler", n, "builtin_symname", NIL);
    Language::staticmembervariableHandler(n);
    Swig_restore(n);

    if (GetFlag(n, "wrappedasconstant"))
      return SWIG_OK;

    String *symname = Getattr(n, "sym:name");

    if (shadow) {
      if (!builtin && GetFlag(n, "hasconsttype")) {
	String *mname = Swig_name_member(NSPACE_TODO, class_name, symname);
	Printf(f_shadow_stubs, "%s.%s = %s.%s.%s\n", class_name, symname, module, global_name, mname);
	Delete(mname);
      } else {
	String *mname = Swig_name_member(NSPACE_TODO, class_name, symname);
	String *getname = Swig_name_get(NSPACE_TODO, mname);
	String *wrapgetname = Swig_name_wrapper(getname);
	String *vargetname = NewStringf("Swig_var_%s", getname);
	String *setname = Swig_name_set(NSPACE_TODO, mname);
	String *wrapsetname = Swig_name_wrapper(setname);
	String *varsetname = NewStringf("Swig_var_%s", setname);

	Wrapper *f = NewWrapper();
	Printv(f->def, "SWIGINTERN PyObject *", wrapgetname, "(PyObject *SWIGUNUSEDPARM(self), PyObject *SWIGUNUSEDPARM(args)) {", NIL);
	Printv(f->code, "  return ", vargetname, "();\n", NIL);
	Append(f->code, "}\n");
	add_method(getname, wrapgetname, 0);
	Wrapper_print(f, f_wrappers);
	DelWrapper(f);
	int assignable = is_assignable(n);
	if (assignable) {
	  int funpack = fastunpack;
	  Wrapper *f = NewWrapper();
	  Printv(f->def, "SWIGINTERN PyObject *", wrapsetname, "(PyObject *SWIGUNUSEDPARM(self), PyObject *args) {", NIL);
	  Wrapper_add_local(f, "res", "int res");
	  if (!funpack) {
	    Wrapper_add_local(f, "value", "PyObject *value");
	    Append(f->code, "if (!PyArg_ParseTuple(args, \"O:set\", &value)) return NULL;\n");
	  }
	  Printf(f->code, "res = %s(%s);\n", varsetname, funpack ? "args" : "value");
	  Append(f->code, "return !res ? SWIG_Py_Void() : NULL;\n");
	  Append(f->code, "}\n");
	  Wrapper_print(f, f_wrappers);
	  add_method(setname, wrapsetname, 0, 0, funpack, 1, 1);
	  DelWrapper(f);
	}
	if (!builtin) {
	  Printv(f_shadow, tab4, symname, " = property(", module, ".", getname, NIL);
	  if (assignable)
	    Printv(f_shadow, ", ", module, ".", setname, NIL);
	  if (have_docstring(n))
	    Printv(f_shadow, ", doc=", docstring(n, AUTODOC_VAR, tab4), NIL);
	  Printv(f_shadow, ")\n", NIL);
	}
	String *getter = Getattr(n, "pybuiltin:getter");
	String *setter = Getattr(n, "pybuiltin:setter");
	Hash *h = NULL;
	if (getter || setter) {
	  h = Getattr(builtin_getset, symname);
	  if (!h) {
	    h = NewHash();
	    Setattr(h, "static", "1");
	    Setattr(builtin_getset, symname, h);
	  }
	}
	if (getter)
	  Setattr(h, "getter", getter);
	if (setter)
	  Setattr(h, "setter", setter);
	if (h)
	  Delete(h);
	Delete(mname);
	Delete(getname);
	Delete(wrapgetname);
	Delete(vargetname);
	Delete(setname);
	Delete(wrapsetname);
	Delete(varsetname);
      }
    }
    return SWIG_OK;
  }

  /* ------------------------------------------------------------
   * memberconstantHandler()
   * ------------------------------------------------------------ */

  virtual int memberconstantHandler(Node *n) {
    String *symname = Getattr(n, "sym:name");
    if (builtin && in_class) {
      Swig_save("builtin_memberconstantHandler", n, "pybuiltin:symname", NIL);
      Setattr(n, "pybuiltin:symname", symname);
    }
    int oldshadow = shadow;
    if (shadow)
      shadow = shadow | PYSHADOW_MEMBER;
    Language::memberconstantHandler(n);
    shadow = oldshadow;

    if (builtin && in_class) {
      Swig_restore(n);
    } else if (shadow) {
      Printv(f_shadow, tab4, symname, " = ", module, ".", Swig_name_member(NSPACE_TODO, class_name, symname), "\n", NIL);
      if (have_docstring(n))
	Printv(f_shadow, tab4, docstring(n, AUTODOC_CONST, tab4), "\n", NIL);
    }
    return SWIG_OK;
  }

  /* ------------------------------------------------------------
   * insertDirective()
   *
   * Hook for %insert directive.   We're going to look for special %shadow inserts
   * as a special case so we can do indenting correctly
   * ------------------------------------------------------------ */

  virtual int insertDirective(Node *n) {
    String *code = Getattr(n, "code");
    String *section = Getattr(n, "section");

    if (!ImportMode && (Cmp(section, "python") == 0 || Cmp(section, "shadow") == 0)) {
      if (shadow) {
	String *pycode = indent_pythoncode(code, shadow_indent, Getfile(n), Getline(n), "%pythoncode or %insert(\"python\") block");
	Printv(f_shadow, pycode, NIL);
	Delete(pycode);
      }
    } else if (!ImportMode && (Cmp(section, "pythonbegin") == 0)) {
      if (shadow) {
	String *pycode = indent_pythoncode(code, "", Getfile(n), Getline(n), "%pythonbegin or %insert(\"pythonbegin\") block");
	Printv(f_shadow_begin, pycode, NIL);
	Delete(pycode);
      }
    } else {
      Language::insertDirective(n);
    }
    return SWIG_OK;
  }

  virtual String *runtimeCode() {
    String *s = NewString("");
    String *shead = Swig_include_sys("pyhead.swg");
    if (!shead) {
      Printf(stderr, "*** Unable to open 'pyhead.swg'\n");
    } else {
      Append(s, shead);
      Delete(shead);
    }
    String *serrors = Swig_include_sys("pyerrors.swg");
    if (!serrors) {
      Printf(stderr, "*** Unable to open 'pyerrors.swg'\n");
    } else {
      Append(s, serrors);
      Delete(serrors);
    }
    String *sthread = Swig_include_sys("pythreads.swg");
    if (!sthread) {
      Printf(stderr, "*** Unable to open 'pythreads.swg'\n");
    } else {
      Append(s, sthread);
      Delete(sthread);
    }
    String *sapi = Swig_include_sys("pyapi.swg");
    if (!sapi) {
      Printf(stderr, "*** Unable to open 'pyapi.swg'\n");
    } else {
      Append(s, sapi);
      Delete(sapi);
    }
    String *srun = Swig_include_sys("pyrun.swg");
    if (!srun) {
      Printf(stderr, "*** Unable to open 'pyrun.swg'\n");
    } else {
      Append(s, srun);
      Delete(srun);
    }
    return s;
  }

  virtual String *defaultExternalRuntimeFilename() {
    return NewString("swigpyrun.h");
  }

  /*----------------------------------------------------------------------
   * kwargsSupport()
   *--------------------------------------------------------------------*/

  bool kwargsSupport() const {
    return true;
  }
};

/* ---------------------------------------------------------------
 * classDirectorMethod()
 *
 * Emit a virtual director method to pass a method call on to the
 * underlying Python object.
 *
 * ** Moved it here due to internal error on gcc-2.96 **
 * --------------------------------------------------------------- */
int PYTHON::classDirectorMethods(Node *n) {
  director_method_index = 0;
  return Language::classDirectorMethods(n);
}


int PYTHON::classDirectorMethod(Node *n, Node *parent, String *super) {
  int is_void = 0;
  int is_pointer = 0;
  String *decl = Getattr(n, "decl");
  String *name = Getattr(n, "name");
  String *classname = Getattr(parent, "sym:name");
  String *c_classname = Getattr(parent, "name");
  String *symname = Getattr(n, "sym:name");
  String *declaration = NewString("");
  ParmList *l = Getattr(n, "parms");
  Wrapper *w = NewWrapper();
  String *tm;
  String *wrap_args = NewString("");
  String *returntype = Getattr(n, "type");
  String *value = Getattr(n, "value");
  String *storage = Getattr(n, "storage");
  bool pure_virtual = false;
  int status = SWIG_OK;
  int idx;
  bool ignored_method = GetFlag(n, "feature:ignore") ? true : false;

  if (builtin) {
    // Rename any wrapped parameters called 'self' as the generated code contains a variable with same name
    Parm *p;
    for (p = l; p; p = nextSibling(p)) {
      String *arg = Getattr(p, "name");
      if (arg && Cmp(arg, "self") == 0)
	Delattr(p, "name");
    }
  }

  if (Cmp(storage, "virtual") == 0) {
    if (Cmp(value, "0") == 0) {
      pure_virtual = true;
    }
  }

  /* determine if the method returns a pointer */
  is_pointer = SwigType_ispointer_return(decl);
  is_void = (!Cmp(returntype, "void") && !is_pointer);

  /* virtual method definition */
  String *target;
  String *pclassname = NewStringf("SwigDirector_%s", classname);
  String *qualified_name = NewStringf("%s::%s", pclassname, name);
  SwigType *rtype = Getattr(n, "conversion_operator") ? 0 : Getattr(n, "classDirectorMethods:type");
  target = Swig_method_decl(rtype, decl, qualified_name, l, 0);
  Printf(w->def, "%s", target);
  Delete(qualified_name);
  Delete(target);
  /* header declaration */
  target = Swig_method_decl(rtype, decl, name, l, 1);
  Printf(declaration, "    virtual %s", target);
  Delete(target);

  // Get any exception classes in the throws typemap
  if (Getattr(n, "noexcept")) {
    Append(w->def, " noexcept");
    Append(declaration, " noexcept");
  }
  ParmList *throw_parm_list = 0;
  if ((throw_parm_list = Getattr(n, "throws")) || Getattr(n, "throw")) {
    Parm *p;
    int gencomma = 0;

    Append(w->def, " throw(");
    Append(declaration, " throw(");

    if (throw_parm_list)
      Swig_typemap_attach_parms("throws", throw_parm_list, 0);
    for (p = throw_parm_list; p; p = nextSibling(p)) {
      if (Getattr(p, "tmap:throws")) {
	if (gencomma++) {
	  Append(w->def, ", ");
	  Append(declaration, ", ");
	}
	String *str = SwigType_str(Getattr(p, "type"), 0);
	Append(w->def, str);
	Append(declaration, str);
	Delete(str);
      }
    }

    Append(w->def, ")");
    Append(declaration, ")");
  }

  Append(w->def, " {");
  Append(declaration, ";\n");

  /* declare method return value
   * if the return value is a reference or const reference, a specialized typemap must
   * handle it, including declaration of c_result ($result).
   */
  if (!is_void && (!ignored_method || pure_virtual)) {
    if (!SwigType_isclass(returntype)) {
      if (!(SwigType_ispointer(returntype) || SwigType_isreference(returntype))) {
	String *construct_result = NewStringf("= SwigValueInit< %s >()", SwigType_lstr(returntype, 0));
	Wrapper_add_localv(w, "c_result", SwigType_lstr(returntype, "c_result"), construct_result, NIL);
	Delete(construct_result);
      } else {
	Wrapper_add_localv(w, "c_result", SwigType_lstr(returntype, "c_result"), "= 0", NIL);
      }
    } else {
      String *cres = SwigType_lstr(returntype, "c_result");
      Printf(w->code, "%s;\n", cres);
      Delete(cres);
    }
  }

  if (builtin) {
    Printv(w->code, "PyObject *self = NULL;\n", NIL);
    Printv(w->code, "(void)self;\n", NIL);
  }

  if (ignored_method) {
    if (!pure_virtual) {
      if (!is_void)
	Printf(w->code, "return ");
      String *super_call = Swig_method_call(super, l);
      Printf(w->code, "%s;\n", super_call);
      Delete(super_call);
    } else {
      Printf(w->code, "Swig::DirectorPureVirtualException::raise(\"Attempted to invoke pure virtual method %s::%s\");\n", SwigType_namestr(c_classname),
	     SwigType_namestr(name));
    }
  } else {
    /* attach typemaps to arguments (C/C++ -> Python) */
    String *arglist = NewString("");
    String *parse_args = NewString("");

    Swig_director_parms_fixup(l);

    /* remove the wrapper 'w' since it was producing spurious temps */
    Swig_typemap_attach_parms("in", l, 0);
    Swig_typemap_attach_parms("directorin", l, w);
    Swig_typemap_attach_parms("directorargout", l, w);

    Parm *p;
    char source[256];

    int outputs = 0;
    if (!is_void)
      outputs++;

    /* build argument list and type conversion string */
    idx = 0;
    p = l;
    int use_parse = 0;
    while (p) {
      if (checkAttribute(p, "tmap:in:numinputs", "0")) {
	p = Getattr(p, "tmap:in:next");
	continue;
      }

      /* old style?  caused segfaults without the p!=0 check
         in the for() condition, and seems dangerous in the
         while loop as well.
         while (Getattr(p, "tmap:ignore")) {
         p = Getattr(p, "tmap:ignore:next");
         }
       */

      if (Getattr(p, "tmap:directorargout") != 0)
	outputs++;

      String *pname = Getattr(p, "name");
      String *ptype = Getattr(p, "type");

      Putc(',', arglist);
      if ((tm = Getattr(p, "tmap:directorin")) != 0) {
	String *parse = Getattr(p, "tmap:directorin:parse");
	if (!parse) {
	  sprintf(source, "obj%d", idx++);
	  String *input = NewString(source);
	  Setattr(p, "emit:directorinput", input);
	  Replaceall(tm, "$input", input);
	  Delete(input);
	  Replaceall(tm, "$owner", "0");
	  /* Wrapper_add_localv(w, source, "swig::SwigVar_PyObject", source, "= 0", NIL); */
	  Printv(wrap_args, "swig::SwigVar_PyObject ", source, ";\n", NIL);

	  Printv(wrap_args, tm, "\n", NIL);
	  Printv(arglist, "(PyObject *)", source, NIL);
	  Putc('O', parse_args);
	} else {
	  use_parse = 1;
	  Append(parse_args, parse);
	  Setattr(p, "emit:directorinput", pname);
	  Replaceall(tm, "$input", pname);
	  Replaceall(tm, "$owner", "0");
	  if (Len(tm) == 0)
	    Append(tm, pname);
	  Append(arglist, tm);
	}
	p = Getattr(p, "tmap:directorin:next");
	continue;
      } else if (Cmp(ptype, "void")) {
	/* special handling for pointers to other C++ director classes.
	 * ideally this would be left to a typemap, but there is currently no
	 * way to selectively apply the dynamic_cast<> to classes that have
	 * directors.  in other words, the type "SwigDirector_$1_lname" only exists
	 * for classes with directors.  we avoid the problem here by checking
	 * module.wrap::directormap, but it's not clear how to get a typemap to
	 * do something similar.  perhaps a new default typemap (in addition
	 * to SWIGTYPE) called DIRECTORTYPE?
	 */
	if (SwigType_ispointer(ptype) || SwigType_isreference(ptype)) {
	  Node *module = Getattr(parent, "module");
	  Node *target = Swig_directormap(module, ptype);
	  sprintf(source, "obj%d", idx++);
	  String *nonconst = 0;
	  /* strip pointer/reference --- should move to Swig/stype.c */
	  String *nptype = NewString(Char(ptype) + 2);
	  /* name as pointer */
	  String *ppname = Copy(pname);
	  if (SwigType_isreference(ptype)) {
	    Insert(ppname, 0, "&");
	  }
	  /* if necessary, cast away const since Python doesn't support it! */
	  if (SwigType_isconst(nptype)) {
	    nonconst = NewStringf("nc_tmp_%s", pname);
	    String *nonconst_i = NewStringf("= const_cast< %s >(%s)", SwigType_lstr(ptype, 0), ppname);
	    Wrapper_add_localv(w, nonconst, SwigType_lstr(ptype, 0), nonconst, nonconst_i, NIL);
	    Delete(nonconst_i);
	    Swig_warning(WARN_LANG_DISCARD_CONST, input_file, line_number,
			 "Target language argument '%s' discards const in director method %s::%s.\n",
			 SwigType_str(ptype, pname), SwigType_namestr(c_classname), SwigType_namestr(name));
	  } else {
	    nonconst = Copy(ppname);
	  }
	  Delete(nptype);
	  Delete(ppname);
	  String *mangle = SwigType_manglestr(ptype);
	  if (target) {
	    String *director = NewStringf("director_%s", mangle);
	    Wrapper_add_localv(w, director, "Swig::Director *", director, "= 0", NIL);
	    Wrapper_add_localv(w, source, "swig::SwigVar_PyObject", source, "= 0", NIL);
	    Printf(wrap_args, "%s = SWIG_DIRECTOR_CAST(%s);\n", director, nonconst);
	    Printf(wrap_args, "if (!%s) {\n", director);
	    Printf(wrap_args, "%s = SWIG_InternalNewPointerObj(%s, SWIGTYPE%s, 0);\n", source, nonconst, mangle);
	    Append(wrap_args, "} else {\n");
	    Printf(wrap_args, "%s = %s->swig_get_self();\n", source, director);
	    Printf(wrap_args, "Py_INCREF((PyObject *)%s);\n", source);
	    Append(wrap_args, "}\n");
	    Delete(director);
	    Printv(arglist, source, NIL);
	  } else {
	    Wrapper_add_localv(w, source, "swig::SwigVar_PyObject", source, "= 0", NIL);
	    Printf(wrap_args, "%s = SWIG_InternalNewPointerObj(%s, SWIGTYPE%s, 0);\n", source, nonconst, mangle);
	    //Printf(wrap_args, "%s = SWIG_NewPointerObj(%s, SWIGTYPE_p_%s, 0);\n",
	    //       source, nonconst, base);
	    Printv(arglist, source, NIL);
	  }
	  Putc('O', parse_args);
	  Delete(mangle);
	  Delete(nonconst);
	} else {
	  Swig_warning(WARN_TYPEMAP_DIRECTORIN_UNDEF, input_file, line_number,
		       "Unable to use type %s as a function argument in director method %s::%s (skipping method).\n", SwigType_str(ptype, 0),
		       SwigType_namestr(c_classname), SwigType_namestr(name));
	  status = SWIG_NOWRAP;
	  break;
	}
      }
      p = nextSibling(p);
    }

    /* add the method name as a PyString */
    String *pyname = Getattr(n, "sym:name");

    int allow_thread = threads_enable(n);

    if (allow_thread) {
      thread_begin_block(n, w->code);
      Append(w->code, "{\n");
    }

    /* wrap complex arguments to PyObjects */
    Printv(w->code, wrap_args, NIL);

    /* pass the method call on to the Python object */
    if (dirprot_mode() && !is_public(n)) {
      Printf(w->code, "swig_set_inner(\"%s\", true);\n", name);
    }


    Append(w->code, "if (!swig_get_self()) {\n");
    Printf(w->code, "  Swig::DirectorException::raise(\"'self' uninitialized, maybe you forgot to call %s.__init__.\");\n", classname);
    Append(w->code, "}\n");
    Append(w->code, "#if defined(SWIG_PYTHON_DIRECTOR_VTABLE)\n");
    Printf(w->code, "const size_t swig_method_index = %d;\n", director_method_index++);
    Printf(w->code, "const char *const swig_method_name = \"%s\";\n", pyname);

    Append(w->code, "PyObject *method = swig_get_method(swig_method_index, swig_method_name);\n");
    if (Len(parse_args) > 0) {
      if (use_parse) {
	Printf(w->code, "swig::SwigVar_PyObject %s = PyObject_CallFunction(method, (char *)\"(%s)\" %s);\n", Swig_cresult_name(), parse_args, arglist);
      } else {
	Printf(w->code, "swig::SwigVar_PyObject %s = PyObject_CallFunctionObjArgs(method %s, NULL);\n", Swig_cresult_name(), arglist);
      }
    } else {
      Append(w->code, "swig::SwigVar_PyObject args = PyTuple_New(0);\n");
      Printf(w->code, "swig::SwigVar_PyObject %s = PyObject_Call(method, (PyObject *) args, NULL);\n", Swig_cresult_name());
    }
    Append(w->code, "#else\n");
    if (Len(parse_args) > 0) {
      if (use_parse) {
	Printf(w->code, "swig::SwigVar_PyObject %s = PyObject_CallMethod(swig_get_self(), (char *)\"%s\", (char *)\"(%s)\" %s);\n", Swig_cresult_name(), pyname, parse_args, arglist);
      } else {
	Printf(w->code, "swig::SwigVar_PyObject swig_method_name = SWIG_Python_str_FromChar(\"%s\");\n", pyname);
	Printf(w->code, "swig::SwigVar_PyObject %s = PyObject_CallMethodObjArgs(swig_get_self(), (PyObject *) swig_method_name %s, NULL);\n", Swig_cresult_name(), arglist);
      }
    } else {
      Printf(w->code, "swig::SwigVar_PyObject swig_method_name = SWIG_Python_str_FromChar(\"%s\");\n", pyname);
      Printf(w->code, "swig::SwigVar_PyObject %s = PyObject_CallMethodObjArgs(swig_get_self(), (PyObject *) swig_method_name, NULL);\n", Swig_cresult_name());
    }
    Append(w->code, "#endif\n");

    if (dirprot_mode() && !is_public(n))
      Printf(w->code, "swig_set_inner(\"%s\", false);\n", name);

    /* exception handling */
    tm = Swig_typemap_lookup("director:except", n, Swig_cresult_name(), 0);
    if (!tm) {
      tm = Getattr(n, "feature:director:except");
      if (tm)
	tm = Copy(tm);
    }
    Printf(w->code, "if (!%s) {\n", Swig_cresult_name());
    Append(w->code, "  PyObject *error = PyErr_Occurred();\n");
    if ((tm) && Len(tm) && (Strcmp(tm, "1") != 0)) {
      Replaceall(tm, "$error", "error");
      Printv(w->code, Str(tm), "\n", NIL);
    } else {
      Append(w->code, "  if (error) {\n");
      Printf(w->code, "    Swig::DirectorMethodException::raise(\"Error detected when calling '%s.%s'\");\n", classname, pyname);
      Append(w->code, "  }\n");
    }
    Append(w->code, "}\n");
    Delete(tm);

    /*
     * Python method may return a simple object, or a tuple.
     * for in/out arguments, we have to extract the appropriate PyObjects from the tuple,
     * then marshal everything back to C/C++ (return value and output arguments).
     *
     */

    /* marshal return value and other outputs (if any) from PyObject to C/C++ type */

    String *cleanup = NewString("");
    String *outarg = NewString("");

    if (outputs > 1) {
      Wrapper_add_local(w, "output", "PyObject *output");
      Printf(w->code, "if (!PyTuple_Check(%s)) {\n", Swig_cresult_name());
      Printf(w->code, "  Swig::DirectorTypeMismatchException::raise(\"Python method %s.%sfailed to return a tuple.\");\n", classname, pyname);
      Append(w->code, "}\n");
    }

    idx = 0;

    /* marshal return value */
    if (!is_void) {
      tm = Swig_typemap_lookup("directorout", n, Swig_cresult_name(), w);
      if (tm != 0) {
	if (outputs > 1) {
	  Printf(w->code, "output = PyTuple_GetItem(%s, %d);\n", Swig_cresult_name(), idx++);
	  Replaceall(tm, "$input", "output");
	} else {
	  Replaceall(tm, "$input", Swig_cresult_name());
	}
	char temp[24];
	sprintf(temp, "%d", idx);
	Replaceall(tm, "$argnum", temp);

	/* TODO check this */
	if (Getattr(n, "wrap:disown")) {
	  Replaceall(tm, "$disown", "SWIG_POINTER_DISOWN");
	} else {
	  Replaceall(tm, "$disown", "0");
	}
	if (Getattr(n, "tmap:directorout:implicitconv")) {
	  Replaceall(tm, "$implicitconv", get_implicitconv_flag(n));
	}
	Replaceall(tm, "$result", "c_result");
	Printv(w->code, tm, "\n", NIL);
	Delete(tm);
      } else {
	Swig_warning(WARN_TYPEMAP_DIRECTOROUT_UNDEF, input_file, line_number,
		     "Unable to use return type %s in director method %s::%s (skipping method).\n", SwigType_str(returntype, 0), SwigType_namestr(c_classname),
		     SwigType_namestr(name));
	status = SWIG_ERROR;
      }
    }

    /* marshal outputs */
    for (p = l; p;) {
      if ((tm = Getattr(p, "tmap:directorargout")) != 0) {
	if (outputs > 1) {
	  Printf(w->code, "output = PyTuple_GetItem(%s, %d);\n", Swig_cresult_name(), idx++);
	  Replaceall(tm, "$result", "output");
	} else {
	  Replaceall(tm, "$result", Swig_cresult_name());
	}
	Replaceall(tm, "$input", Getattr(p, "emit:directorinput"));
	Printv(w->code, tm, "\n", NIL);
	p = Getattr(p, "tmap:directorargout:next");
      } else {
	p = nextSibling(p);
      }
    }

    /* any existing helper functions to handle this? */
    if (allow_thread) {
      Append(w->code, "}\n");
      thread_end_block(n, w->code);
    }

    Delete(parse_args);
    Delete(arglist);
    Delete(cleanup);
    Delete(outarg);
  }

  if (!is_void) {
    if (!(ignored_method && !pure_virtual)) {
      String *rettype = SwigType_str(returntype, 0);
      if (!SwigType_isreference(returntype)) {
	Printf(w->code, "return (%s) c_result;\n", rettype);
      } else {
	Printf(w->code, "return (%s) *c_result;\n", rettype);
      }
      Delete(rettype);
    }
  }

  Append(w->code, "}\n");

  // We expose protected methods via an extra public inline method which makes a straight call to the wrapped class' method
  String *inline_extra_method = NewString("");
  if (dirprot_mode() && !is_public(n) && !pure_virtual) {
    Printv(inline_extra_method, declaration, NIL);
    String *extra_method_name = NewStringf("%sSwigPublic", name);
    Replaceall(inline_extra_method, name, extra_method_name);
    Replaceall(inline_extra_method, ";\n", " {\n      ");
    if (!is_void)
      Printf(inline_extra_method, "return ");
    String *methodcall = Swig_method_call(super, l);
    Printv(inline_extra_method, methodcall, ";\n    }\n", NIL);
    Delete(methodcall);
    Delete(extra_method_name);
  }

  /* emit the director method */
  if (status == SWIG_OK) {
    if (!Getattr(n, "defaultargs")) {
      Replaceall(w->code, "$symname", symname);
      Wrapper_print(w, f_directors);
      Printv(f_directors_h, declaration, NIL);
      Printv(f_directors_h, inline_extra_method, NIL);
    }
  }

  /* clean up */
  Delete(wrap_args);
  Delete(pclassname);
  DelWrapper(w);
  return status;
}

/* -----------------------------------------------------------------------------
 * swig_python()    - Instantiate module
 * ----------------------------------------------------------------------------- */

static Language *new_swig_python() {
  return new PYTHON();
}
extern "C" Language *swig_python(void) {
  return new_swig_python();
}
