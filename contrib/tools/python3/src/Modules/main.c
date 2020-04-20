/* Python interpreter main program */

#include "Python.h"
#include "osdefs.h"
#include "internal/import.h"
#include "internal/pygetopt.h"
#include "internal/pystate.h"

#include <locale.h>

#if defined(MS_WINDOWS) || defined(__CYGWIN__)
#  include <windows.h>
#  ifdef HAVE_IO_H
#    include <io.h>
#  endif
#  ifdef HAVE_FCNTL_H
#    include <fcntl.h>
#  endif
#endif

#ifdef MS_WINDOWS
#  include <windows.h>  /* STATUS_CONTROL_C_EXIT */
#endif

#ifdef __FreeBSD__
#  include <fenv.h>
#endif

#if defined(MS_WINDOWS)
#  define PYTHONHOMEHELP "<prefix>\\python{major}{minor}"
#else
#  define PYTHONHOMEHELP "<prefix>/lib/pythonX.X"
#endif

#define COPYRIGHT \
    "Type \"help\", \"copyright\", \"credits\" or \"license\" " \
    "for more information."

#ifdef __cplusplus
extern "C" {
#endif

#define DECODE_LOCALE_ERR(NAME, LEN) \
    (((LEN) == -2) \
     ? _Py_INIT_USER_ERR("cannot decode " NAME) \
     : _Py_INIT_NO_MEMORY())


#define SET_DECODE_ERROR(NAME, LEN) \
    do { \
        if ((LEN) == (size_t)-2) { \
            pymain->err = _Py_INIT_USER_ERR("cannot decode " NAME); \
        } \
        else { \
            pymain->err = _Py_INIT_NO_MEMORY(); \
        } \
    } while (0)

#ifdef MS_WINDOWS
#define WCSTOK wcstok_s
#else
#define WCSTOK wcstok
#endif

/* For Py_GetArgcArgv(); set by main() */
static wchar_t **orig_argv = NULL;
static int orig_argc = 0;

/* command line options */
#define BASE_OPTS L"bBc:dEhiIJm:OqRsStuvVW:xX:?"

#define PROGRAM_OPTS BASE_OPTS

static const _PyOS_LongOption longoptions[] = {
    {L"check-hash-based-pycs", 1, 0},
    {NULL, 0, 0},
};

/* Short usage message (with %s for argv0) */
static const char usage_line[] =
"usage: %ls [option] ... [-c cmd | -m mod | file | -] [arg] ...\n";

/* Long usage message, split into parts < 512 bytes */
static const char usage_1[] = "\
Options and arguments (and corresponding environment variables):\n\
-b     : issue warnings about str(bytes_instance), str(bytearray_instance)\n\
         and comparing bytes/bytearray with str. (-bb: issue errors)\n\
-B     : don't write .pyc files on import; also PYTHONDONTWRITEBYTECODE=x\n\
-c cmd : program passed in as string (terminates option list)\n\
-d     : debug output from parser; also PYTHONDEBUG=x\n\
-E     : ignore PYTHON* environment variables (such as PYTHONPATH)\n\
-h     : print this help message and exit (also --help)\n\
";
static const char usage_2[] = "\
-i     : inspect interactively after running script; forces a prompt even\n\
         if stdin does not appear to be a terminal; also PYTHONINSPECT=x\n\
-I     : isolate Python from the user's environment (implies -E and -s)\n\
-m mod : run library module as a script (terminates option list)\n\
-O     : remove assert and __debug__-dependent statements; add .opt-1 before\n\
         .pyc extension; also PYTHONOPTIMIZE=x\n\
-OO    : do -O changes and also discard docstrings; add .opt-2 before\n\
         .pyc extension\n\
-q     : don't print version and copyright messages on interactive startup\n\
-s     : don't add user site directory to sys.path; also PYTHONNOUSERSITE\n\
-S     : don't imply 'import site' on initialization\n\
";
static const char usage_3[] = "\
-u     : force the stdout and stderr streams to be unbuffered;\n\
         this option has no effect on stdin; also PYTHONUNBUFFERED=x\n\
-v     : verbose (trace import statements); also PYTHONVERBOSE=x\n\
         can be supplied multiple times to increase verbosity\n\
-V     : print the Python version number and exit (also --version)\n\
         when given twice, print more information about the build\n\
-W arg : warning control; arg is action:message:category:module:lineno\n\
         also PYTHONWARNINGS=arg\n\
-x     : skip first line of source, allowing use of non-Unix forms of #!cmd\n\
-X opt : set implementation-specific option. The following options are available:\n\
\n\
         -X faulthandler: enable faulthandler\n\
         -X showrefcount: output the total reference count and number of used\n\
             memory blocks when the program finishes or after each statement in the\n\
             interactive interpreter. This only works on debug builds\n\
         -X tracemalloc: start tracing Python memory allocations using the\n\
             tracemalloc module. By default, only the most recent frame is stored in a\n\
             traceback of a trace. Use -X tracemalloc=NFRAME to start tracing with a\n\
             traceback limit of NFRAME frames\n\
         -X showalloccount: output the total count of allocated objects for each\n\
             type when the program finishes. This only works when Python was built with\n\
             COUNT_ALLOCS defined\n\
         -X importtime: show how long each import takes. It shows module name,\n\
             cumulative time (including nested imports) and self time (excluding\n\
             nested imports). Note that its output may be broken in multi-threaded\n\
             application. Typical usage is python3 -X importtime -c 'import asyncio'\n\
         -X dev: enable CPython’s “development mode”, introducing additional runtime\n\
             checks which are too expensive to be enabled by default. Effect of the\n\
             developer mode:\n\
                * Add default warning filter, as -W default\n\
                * Install debug hooks on memory allocators: see the PyMem_SetupDebugHooks() C function\n\
                * Enable the faulthandler module to dump the Python traceback on a crash\n\
                * Enable asyncio debug mode\n\
                * Set the dev_mode attribute of sys.flags to True\n\
         -X utf8: enable UTF-8 mode for operating system interfaces, overriding the default\n\
             locale-aware mode. -X utf8=0 explicitly disables UTF-8 mode (even when it would\n\
             otherwise activate automatically)\n\
\n\
--check-hash-based-pycs always|default|never:\n\
    control how Python invalidates hash-based .pyc files\n\
";
static const char usage_4[] = "\
file   : program read from script file\n\
-      : program read from stdin (default; interactive mode if a tty)\n\
arg ...: arguments passed to program in sys.argv[1:]\n\n\
Other environment variables:\n\
PYTHONSTARTUP: file executed on interactive startup (no default)\n\
PYTHONPATH   : '%lc'-separated list of directories prefixed to the\n\
               default module search path.  The result is sys.path.\n\
";
static const char usage_5[] =
"PYTHONHOME   : alternate <prefix> directory (or <prefix>%lc<exec_prefix>).\n"
"               The default module search path uses %s.\n"
"PYTHONCASEOK : ignore case in 'import' statements (Windows).\n"
"PYTHONUTF8: if set to 1, enable the UTF-8 mode.\n"
"PYTHONIOENCODING: Encoding[:errors] used for stdin/stdout/stderr.\n"
"PYTHONFAULTHANDLER: dump the Python traceback on fatal errors.\n";
static const char usage_6[] =
"PYTHONHASHSEED: if this variable is set to 'random', a random value is used\n"
"   to seed the hashes of str, bytes and datetime objects.  It can also be\n"
"   set to an integer in the range [0,4294967295] to get hash values with a\n"
"   predictable seed.\n"
"PYTHONMALLOC: set the Python memory allocators and/or install debug hooks\n"
"   on Python memory allocators. Use PYTHONMALLOC=debug to install debug\n"
"   hooks.\n"
"PYTHONCOERCECLOCALE: if this variable is set to 0, it disables the locale\n"
"   coercion behavior. Use PYTHONCOERCECLOCALE=warn to request display of\n"
"   locale coercion and locale compatibility warnings on stderr.\n"
"PYTHONBREAKPOINT: if this variable is set to 0, it disables the default\n"
"   debugger. It can be set to the callable of your debugger of choice.\n"
"PYTHONDEVMODE: enable the development mode.\n";

static void
pymain_usage(int error, const wchar_t* program)
{
    FILE *f = error ? stderr : stdout;

    fprintf(f, usage_line, program);
    if (error)
        fprintf(f, "Try `python -h' for more information.\n");
    else {
        fputs(usage_1, f);
        fputs(usage_2, f);
        fputs(usage_3, f);
        fprintf(f, usage_4, (wint_t)DELIM);
        fprintf(f, usage_5, (wint_t)DELIM, PYTHONHOMEHELP);
        fputs(usage_6, f);
    }
}


static const char*
config_get_env_var(const char *name)
{
    const char *var = Py_GETENV(name);
    if (var && var[0] != '\0') {
        return var;
    }
    else {
        return NULL;
    }
}


static int
config_get_env_var_dup(wchar_t **dest, wchar_t *wname, char *name)
{
    if (Py_IgnoreEnvironmentFlag) {
        *dest = NULL;
        return 0;
    }

#ifdef MS_WINDOWS
    const wchar_t *var = _wgetenv(wname);
    if (!var || var[0] == '\0') {
        *dest = NULL;
        return 0;
    }

    wchar_t *copy = _PyMem_RawWcsdup(var);
    if (copy == NULL) {
        return -1;
    }

    *dest = copy;
#else
    const char *var = getenv(name);
    if (!var || var[0] == '\0') {
        *dest = NULL;
        return 0;
    }

    size_t len;
    wchar_t *wvar = Py_DecodeLocale(var, &len);
    if (!wvar) {
        if (len == (size_t)-2) {
            return -2;
        }
        else {
            return -1;
        }
    }
    *dest = wvar;
#endif
    return 0;
}


static void
pymain_run_startup(PyCompilerFlags *cf)
{
    const char *startup = config_get_env_var("PYTHONSTARTUP");
    if (startup == NULL) {
        return;
    }

    FILE *fp = _Py_fopen(startup, "r");
    if (fp == NULL) {
        int save_errno = errno;
        PySys_WriteStderr("Could not open PYTHONSTARTUP\n");
        errno = save_errno;

        PyErr_SetFromErrnoWithFilename(PyExc_OSError,
                        startup);
        PyErr_Print();
        PyErr_Clear();
        return;
    }

    (void) PyRun_SimpleFileExFlags(fp, startup, 0, cf);
    PyErr_Clear();
    fclose(fp);
}

static void
pymain_run_interactive_hook(void)
{
    PyObject *sys, *hook, *result;
    sys = PyImport_ImportModule("sys");
    if (sys == NULL) {
        goto error;
    }

    hook = PyObject_GetAttrString(sys, "__interactivehook__");
    Py_DECREF(sys);
    if (hook == NULL) {
        PyErr_Clear();
        return;
    }

    result = _PyObject_CallNoArg(hook);
    Py_DECREF(hook);
    if (result == NULL) {
        goto error;
    }
    Py_DECREF(result);

    return;

error:
    PySys_WriteStderr("Failed calling sys.__interactivehook__\n");
    PyErr_Print();
    PyErr_Clear();
}


static int
pymain_run_module(const wchar_t *modname, int set_argv0)
{
    PyObject *module, *runpy, *runmodule, *runargs, *result;
    runpy = PyImport_ImportModule("runpy");
    if (runpy == NULL) {
        fprintf(stderr, "Could not import runpy module\n");
        PyErr_Print();
        return -1;
    }
    runmodule = PyObject_GetAttrString(runpy, "_run_module_as_main");
    if (runmodule == NULL) {
        fprintf(stderr, "Could not access runpy._run_module_as_main\n");
        PyErr_Print();
        Py_DECREF(runpy);
        return -1;
    }
    module = PyUnicode_FromWideChar(modname, wcslen(modname));
    if (module == NULL) {
        fprintf(stderr, "Could not convert module name to unicode\n");
        PyErr_Print();
        Py_DECREF(runpy);
        Py_DECREF(runmodule);
        return -1;
    }
    runargs = Py_BuildValue("(Oi)", module, set_argv0);
    if (runargs == NULL) {
        fprintf(stderr,
            "Could not create arguments for runpy._run_module_as_main\n");
        PyErr_Print();
        Py_DECREF(runpy);
        Py_DECREF(runmodule);
        Py_DECREF(module);
        return -1;
    }
    result = PyObject_Call(runmodule, runargs, NULL);
    if (result == NULL) {
        PyErr_Print();
    }
    Py_DECREF(runpy);
    Py_DECREF(runmodule);
    Py_DECREF(module);
    Py_DECREF(runargs);
    if (result == NULL) {
        return -1;
    }
    Py_DECREF(result);
    return 0;
}

static PyObject *
pymain_get_importer(const wchar_t *filename)
{
    PyObject *sys_path0 = NULL, *importer;

    sys_path0 = PyUnicode_FromWideChar(filename, wcslen(filename));
    if (sys_path0 == NULL) {
        goto error;
    }

    importer = PyImport_GetImporter(sys_path0);
    if (importer == NULL) {
        goto error;
    }

    if (importer == Py_None) {
        Py_DECREF(sys_path0);
        Py_DECREF(importer);
        return NULL;
    }

    Py_DECREF(importer);
    return sys_path0;

error:
    Py_XDECREF(sys_path0);
    PySys_WriteStderr("Failed checking if argv[0] is an import path entry\n");
    PyErr_Print();
    PyErr_Clear();
    return NULL;
}


static int
pymain_run_command(wchar_t *command, PyCompilerFlags *cf)
{
    PyObject *unicode, *bytes;
    int ret;

    unicode = PyUnicode_FromWideChar(command, -1);
    if (unicode == NULL) {
        goto error;
    }

    bytes = PyUnicode_AsUTF8String(unicode);
    Py_DECREF(unicode);
    if (bytes == NULL) {
        goto error;
    }

    ret = PyRun_SimpleStringFlags(PyBytes_AsString(bytes), cf);
    Py_DECREF(bytes);
    return (ret != 0);

error:
    PySys_WriteStderr("Unable to decode the command from the command line:\n");
    PyErr_Print();
    return 1;
}


static int
pymain_run_file(FILE *fp, const wchar_t *filename, PyCompilerFlags *p_cf)
{
    PyObject *unicode, *bytes = NULL;
    const char *filename_str;
    int run;

    /* call pending calls like signal handlers (SIGINT) */
    if (Py_MakePendingCalls() == -1) {
        PyErr_Print();
        return 1;
    }

    if (filename) {
        unicode = PyUnicode_FromWideChar(filename, wcslen(filename));
        if (unicode != NULL) {
            bytes = PyUnicode_EncodeFSDefault(unicode);
            Py_DECREF(unicode);
        }
        if (bytes != NULL) {
            filename_str = PyBytes_AsString(bytes);
        }
        else {
            PyErr_Clear();
            filename_str = "<encoding error>";
        }
    }
    else {
        filename_str = "<stdin>";
    }

    run = PyRun_AnyFileExFlags(fp, filename_str, filename != NULL, p_cf);
    Py_XDECREF(bytes);
    return run != 0;
}


/* Main program */

typedef struct {
    wchar_t **argv;
    int nwarnoption;             /* Number of -W options */
    wchar_t **warnoptions;       /* -W options */
    int nenv_warnoption;         /* Number of PYTHONWARNINGS options */
    wchar_t **env_warnoptions;   /* PYTHONWARNINGS options */
    int print_help;              /* -h, -? options */
    int print_version;           /* -V option */
    int bytes_warning;           /* Py_BytesWarningFlag, -b */
    int debug;                   /* Py_DebugFlag, -b, PYTHONDEBUG */
    int inspect;                 /* Py_InspectFlag, -i, PYTHONINSPECT */
    int interactive;             /* Py_InteractiveFlag, -i */
    int isolated;                /* Py_IsolatedFlag, -I */
    int optimization_level;      /* Py_OptimizeFlag, -O, PYTHONOPTIMIZE */
    int dont_write_bytecode;     /* Py_DontWriteBytecodeFlag, -B, PYTHONDONTWRITEBYTECODE */
    int no_user_site_directory;  /* Py_NoUserSiteDirectory, -I, -s, PYTHONNOUSERSITE */
    int no_site_import;          /* Py_NoSiteFlag, -S */
    int use_unbuffered_io;       /* Py_UnbufferedStdioFlag, -u, PYTHONUNBUFFERED */
    int verbosity;               /* Py_VerboseFlag, -v, PYTHONVERBOSE */
    int quiet_flag;              /* Py_QuietFlag, -q */
    const char *check_hash_pycs_mode; /* --check-hash-based-pycs */
#ifdef MS_WINDOWS
    int legacy_windows_fs_encoding;  /* Py_LegacyWindowsFSEncodingFlag,
                                        PYTHONLEGACYWINDOWSFSENCODING */
    int legacy_windows_stdio;        /* Py_LegacyWindowsStdioFlag,
                                        PYTHONLEGACYWINDOWSSTDIO */
#endif
} _PyCmdline;

/* Structure used by Py_Main() to pass data to subfunctions */
typedef struct {
    /* Input arguments */
    int argc;
    int use_bytes_argv;
    char **bytes_argv;
    wchar_t **wchar_argv;

    /* Exit status or "exit code": result of pymain_main() */
    int status;
    /* Error message if a function failed */
    _PyInitError err;

    /* non-zero is stdin is a TTY or if -i option is used */
    int stdin_is_interactive;
    int skip_first_line;         /* -x option */
    wchar_t *filename;           /* Trailing arg without -c or -m */
    wchar_t *command;            /* -c argument */
    wchar_t *module;             /* -m argument */

    PyObject *main_importer_path;
} _PyMain;

#define _PyMain_INIT {.err = _Py_INIT_OK()}
/* Note: _PyMain_INIT sets other fields to 0/NULL */


/* Non-zero if filename, command (-c) or module (-m) is set
   on the command line */
#define RUN_CODE(pymain) \
    (pymain->command != NULL || pymain->filename != NULL \
     || pymain->module != NULL)


static wchar_t*
pymain_wstrdup(_PyMain *pymain, const wchar_t *str)
{
    wchar_t *str2 = _PyMem_RawWcsdup(str);
    if (str2 == NULL) {
        pymain->err = _Py_INIT_NO_MEMORY();
        return NULL;
    }
    return str2;
}


static void
clear_wstrlist(int len, wchar_t **list)
{
    for (int i=0; i < len; i++) {
        PyMem_RawFree(list[i]);
    }
    PyMem_RawFree(list);
}


static int
pymain_init_cmdline_argv(_PyMain *pymain, _PyCoreConfig *config,
                         _PyCmdline *cmdline)
{
    assert(cmdline->argv == NULL);

    if (pymain->use_bytes_argv) {
        /* +1 for a the NULL terminator */
        size_t size = sizeof(wchar_t*) * (pymain->argc + 1);
        wchar_t** argv = (wchar_t **)PyMem_RawMalloc(size);
        if (argv == NULL) {
            pymain->err = _Py_INIT_NO_MEMORY();
            return -1;
        }

        for (int i = 0; i < pymain->argc; i++) {
            size_t len;
            wchar_t *arg = Py_DecodeLocale(pymain->bytes_argv[i], &len);
            if (arg == NULL) {
                clear_wstrlist(i, argv);
                pymain->err = DECODE_LOCALE_ERR("command line arguments",
                                                (Py_ssize_t)len);
                return -1;
            }
            argv[i] = arg;
        }
        argv[pymain->argc] = NULL;

        cmdline->argv = argv;
    }
    else {
        cmdline->argv = pymain->wchar_argv;
    }

    wchar_t *program;
    if (pymain->argc >= 1 && cmdline->argv != NULL) {
        program = cmdline->argv[0];
    }
    else {
        program = L"";
    }
    config->program = pymain_wstrdup(pymain, program);
    if (config->program == NULL) {
        return -1;
    }

    return 0;
}


static void
pymain_clear_cmdline(_PyMain *pymain, _PyCmdline *cmdline)
{
    PyMemAllocatorEx old_alloc;
    _PyMem_SetDefaultAllocator(PYMEM_DOMAIN_RAW, &old_alloc);

    clear_wstrlist(cmdline->nwarnoption, cmdline->warnoptions);
    cmdline->nwarnoption = 0;
    cmdline->warnoptions = NULL;

    clear_wstrlist(cmdline->nenv_warnoption, cmdline->env_warnoptions);
    cmdline->nenv_warnoption = 0;
    cmdline->env_warnoptions = NULL;

    if (pymain->use_bytes_argv && cmdline->argv != NULL) {
        clear_wstrlist(pymain->argc, cmdline->argv);
    }
    cmdline->argv = NULL;

    PyMem_SetAllocator(PYMEM_DOMAIN_RAW, &old_alloc);
}


static void
pymain_clear_pymain(_PyMain *pymain)
{
#define CLEAR(ATTR) \
    do { \
        PyMem_RawFree(ATTR); \
        ATTR = NULL; \
    } while (0)

    CLEAR(pymain->filename);
    CLEAR(pymain->command);
    CLEAR(pymain->module);
#undef CLEAR
}

static void
pymain_clear_config(_PyCoreConfig *config)
{
    /* Clear core config with the memory allocator
       used by pymain_read_conf() */
    PyMemAllocatorEx old_alloc;
    _PyMem_SetDefaultAllocator(PYMEM_DOMAIN_RAW, &old_alloc);

    _PyCoreConfig_Clear(config);

    PyMem_SetAllocator(PYMEM_DOMAIN_RAW, &old_alloc);
}


static void
pymain_free_python(_PyMain *pymain)
{
    Py_CLEAR(pymain->main_importer_path);

#ifdef __INSURE__
    /* Insure++ is a memory analysis tool that aids in discovering
     * memory leaks and other memory problems.  On Python exit, the
     * interned string dictionaries are flagged as being in use at exit
     * (which it is).  Under normal circumstances, this is fine because
     * the memory will be automatically reclaimed by the system.  Under
     * memory debugging, it's a huge source of useless noise, so we
     * trade off slower shutdown for less distraction in the memory
     * reports.  -baw
     */
    _Py_ReleaseInternedUnicodeStrings();
#endif /* __INSURE__ */
}


static void
pymain_free_raw(_PyMain *pymain)
{
    _PyImport_Fini2();

    /* Free global variables which cannot be freed in Py_Finalize():
       configuration options set before Py_Initialize() which should
       remain valid after Py_Finalize(), since
       Py_Initialize()-Py_Finalize() can be called multiple times. */
    _PyPathConfig_Clear(&_Py_path_config);

    /* Force the allocator used by pymain_read_conf() */
    PyMemAllocatorEx old_alloc;
    _PyMem_SetDefaultAllocator(PYMEM_DOMAIN_RAW, &old_alloc);

    pymain_clear_pymain(pymain);

    clear_wstrlist(orig_argc, orig_argv);
    orig_argc = 0;
    orig_argv = NULL;

    _PyRuntime_Finalize();

    PyMem_SetAllocator(PYMEM_DOMAIN_RAW, &old_alloc);
}


static void
pymain_free(_PyMain *pymain)
{
    pymain_free_python(pymain);
    pymain_free_raw(pymain);
}


static int
pymain_run_main_from_importer(_PyMain *pymain)
{
    /* Assume sys_path0 has already been checked by pymain_get_importer(),
     * so put it in sys.path[0] and import __main__ */
    PyObject *sys_path = PySys_GetObject("path");
    if (sys_path == NULL) {
        PyErr_SetString(PyExc_RuntimeError, "unable to get sys.path");
        goto error;
    }

    if (PyList_Insert(sys_path, 0, pymain->main_importer_path)) {
        goto error;
    }

    int sts = pymain_run_module(L"__main__", 0);
    return (sts != 0);

error:
    Py_CLEAR(pymain->main_importer_path);
    PyErr_Print();
    return 1;
}


static _PyInitError
wstrlist_append(int *len, wchar_t ***list, const wchar_t *str)
{
    wchar_t *str2 = _PyMem_RawWcsdup(str);
    if (str2 == NULL) {
        return _Py_INIT_NO_MEMORY();
    }

    size_t size = (*len + 1) * sizeof(list[0]);
    wchar_t **list2 = (wchar_t **)PyMem_RawRealloc(*list, size);
    if (list2 == NULL) {
        PyMem_RawFree(str2);
        return _Py_INIT_NO_MEMORY();
    }
    list2[*len] = str2;
    *list = list2;
    (*len)++;
    return _Py_INIT_OK();
}


static int
pymain_wstrlist_append(_PyMain *pymain, int *len, wchar_t ***list, const wchar_t *str)
{
    _PyInitError err = wstrlist_append(len, list, str);
    if (_Py_INIT_FAILED(err)) {
        pymain->err = err;
        return -1;
    }
    return 0;
}


/* Parse the command line arguments
   Return 0 on success.
   Return 1 if parsing failed.
   Set pymain->err and return -1 on other errors. */
static int
pymain_parse_cmdline_impl(_PyMain *pymain, _PyCoreConfig *config,
                          _PyCmdline *cmdline)
{
    _PyOS_ResetGetOpt();
    do {
        int longindex = -1;
        int c = _PyOS_GetOpt(pymain->argc, cmdline->argv, PROGRAM_OPTS,
                             longoptions, &longindex);
        if (c == EOF) {
            break;
        }

        if (c == 'c') {
            /* -c is the last option; following arguments
               that look like options are left for the
               command to interpret. */
            size_t len = wcslen(_PyOS_optarg) + 1 + 1;
            wchar_t *command = PyMem_RawMalloc(sizeof(wchar_t) * len);
            if (command == NULL) {
                pymain->err = _Py_INIT_NO_MEMORY();
                return -1;
            }
            memcpy(command, _PyOS_optarg, (len - 2) * sizeof(wchar_t));
            command[len - 2] = '\n';
            command[len - 1] = 0;
            pymain->command = command;
            break;
        }

        if (c == 'm') {
            /* -m is the last option; following arguments
               that look like options are left for the
               module to interpret. */
            pymain->module = pymain_wstrdup(pymain, _PyOS_optarg);
            if (pymain->module == NULL) {
                return -1;
            }
            break;
        }

        switch (c) {
        case 0:
            // Handle long option.
            assert(longindex == 0); // Only one long option now.
            if (!wcscmp(_PyOS_optarg, L"always")) {
                cmdline->check_hash_pycs_mode = "always";
            } else if (!wcscmp(_PyOS_optarg, L"never")) {
                cmdline->check_hash_pycs_mode = "never";
            } else if (!wcscmp(_PyOS_optarg, L"default")) {
                cmdline->check_hash_pycs_mode = "default";
            } else {
                fprintf(stderr, "--check-hash-based-pycs must be one of "
                        "'default', 'always', or 'never'\n");
                return 1;
            }
            break;

        case 'b':
            cmdline->bytes_warning++;
            break;

        case 'd':
            cmdline->debug++;
            break;

        case 'i':
            cmdline->inspect++;
            cmdline->interactive++;
            break;

        case 'I':
            config->ignore_environment++;
            cmdline->isolated++;
            cmdline->no_user_site_directory++;
            break;

        /* case 'J': reserved for Jython */

        case 'O':
            cmdline->optimization_level++;
            break;

        case 'B':
            cmdline->dont_write_bytecode++;
            break;

        case 's':
            cmdline->no_user_site_directory++;
            break;

        case 'S':
            cmdline->no_site_import++;
            break;

        case 'E':
            config->ignore_environment++;
            break;

        case 't':
            /* ignored for backwards compatibility */
            break;

        case 'u':
            cmdline->use_unbuffered_io = 1;
            break;

        case 'v':
            cmdline->verbosity++;
            break;

        case 'x':
            pymain->skip_first_line = 1;
            break;

        case 'h':
        case '?':
            cmdline->print_help++;
            break;

        case 'V':
            cmdline->print_version++;
            break;

        case 'W':
            if (pymain_wstrlist_append(pymain,
                                       &cmdline->nwarnoption,
                                       &cmdline->warnoptions,
                                       _PyOS_optarg) < 0) {
                return -1;
            }
            break;

        case 'X':
            if (pymain_wstrlist_append(pymain,
                                       &config->nxoption,
                                       &config->xoptions,
                                       _PyOS_optarg) < 0) {
                return -1;
            }
            break;

        case 'q':
            cmdline->quiet_flag++;
            break;

        case 'R':
            config->use_hash_seed = 0;
            break;

        /* This space reserved for other options */

        default:
            /* unknown argument: parsing failed */
            return 1;
        }
    } while (1);

    if (pymain->command == NULL && pymain->module == NULL
        && _PyOS_optind < pymain->argc
        && wcscmp(cmdline->argv[_PyOS_optind], L"-") != 0)
    {
        pymain->filename = pymain_wstrdup(pymain, cmdline->argv[_PyOS_optind]);
        if (pymain->filename == NULL) {
            return -1;
        }
    }

    /* -c and -m options are exclusive */
    assert(!(pymain->command != NULL && pymain->module != NULL));

    return 0;
}


static int
add_xoption(PyObject *opts, const wchar_t *s)
{
    PyObject *name, *value;

    const wchar_t *name_end = wcschr(s, L'=');
    if (!name_end) {
        name = PyUnicode_FromWideChar(s, -1);
        value = Py_True;
        Py_INCREF(value);
    }
    else {
        name = PyUnicode_FromWideChar(s, name_end - s);
        value = PyUnicode_FromWideChar(name_end + 1, -1);
    }
    if (name == NULL || value == NULL) {
        goto error;
    }
    if (PyDict_SetItem(opts, name, value) < 0) {
        goto error;
    }
    Py_DECREF(name);
    Py_DECREF(value);
    return 0;

error:
    Py_XDECREF(name);
    Py_XDECREF(value);
    return -1;
}


static PyObject*
config_create_xoptions_dict(const _PyCoreConfig *config)
{
    int nxoption = config->nxoption;
    wchar_t **xoptions = config->xoptions;
    PyObject *dict = PyDict_New();
    if (dict == NULL) {
        return NULL;
    }

    for (int i=0; i < nxoption; i++) {
        wchar_t *option = xoptions[i];
        if (add_xoption(dict, option) < 0) {
            Py_DECREF(dict);
            return NULL;
        }
    }

    return dict;
}


static _PyInitError
config_add_warnings_optlist(_PyCoreConfig *config, int len, wchar_t **options)
{
    for (int i = 0; i < len; i++) {
        _PyInitError err = wstrlist_append(&config->nwarnoption,
                                           &config->warnoptions,
                                           options[i]);
        if (_Py_INIT_FAILED(err)) {
            return err;
        }
    }
    return _Py_INIT_OK();
}


static _PyInitError
config_init_warnoptions(_PyCoreConfig *config, _PyCmdline *cmdline)
{
    _PyInitError err;

    assert(config->nwarnoption == 0);

    /* The priority order for warnings configuration is (highest precedence
     * first):
     *
     * - the BytesWarning filter, if needed ('-b', '-bb')
     * - any '-W' command line options; then
     * - the 'PYTHONWARNINGS' environment variable; then
     * - the dev mode filter ('-X dev', 'PYTHONDEVMODE'); then
     * - any implicit filters added by _warnings.c/warnings.py
     *
     * All settings except the last are passed to the warnings module via
     * the `sys.warnoptions` list. Since the warnings module works on the basis
     * of "the most recently added filter will be checked first", we add
     * the lowest precedence entries first so that later entries override them.
     */

    if (config->dev_mode) {
        err = wstrlist_append(&config->nwarnoption,
                              &config->warnoptions,
                              L"default");
        if (_Py_INIT_FAILED(err)) {
            return err;
        }
    }

    err = config_add_warnings_optlist(config,
                                      cmdline->nenv_warnoption,
                                      cmdline->env_warnoptions);
    if (_Py_INIT_FAILED(err)) {
        return err;
    }

    err = config_add_warnings_optlist(config,
                                      cmdline->nwarnoption,
                                      cmdline->warnoptions);
    if (_Py_INIT_FAILED(err)) {
        return err;
    }

    /* If the bytes_warning_flag isn't set, bytesobject.c and bytearrayobject.c
     * don't even try to emit a warning, so we skip setting the filter in that
     * case.
     */
    if (cmdline->bytes_warning) {
        wchar_t *filter;
        if (cmdline->bytes_warning> 1) {
            filter = L"error::BytesWarning";
        }
        else {
            filter = L"default::BytesWarning";
        }
        err = wstrlist_append(&config->nwarnoption,
                              &config->warnoptions,
                              filter);
        if (_Py_INIT_FAILED(err)) {
            return err;
        }
    }
    return _Py_INIT_OK();
}


/* Get warning options from PYTHONWARNINGS environment variable.
   Return 0 on success.
   Set pymain->err and return -1 on error. */
static _PyInitError
cmdline_init_env_warnoptions(_PyCmdline *cmdline)
{
    if (Py_IgnoreEnvironmentFlag) {
        return _Py_INIT_OK();
    }

    wchar_t *env;
    int res = config_get_env_var_dup(&env, L"PYTHONWARNINGS", "PYTHONWARNINGS");
    if (res < 0) {
        return DECODE_LOCALE_ERR("PYTHONWARNINGS", res);
    }

    if (env == NULL) {
        return _Py_INIT_OK();
    }


    wchar_t *warning, *context = NULL;
    for (warning = WCSTOK(env, L",", &context);
         warning != NULL;
         warning = WCSTOK(NULL, L",", &context))
    {
        _PyInitError err = wstrlist_append(&cmdline->nenv_warnoption,
                                           &cmdline->env_warnoptions,
                                          warning);
        if (_Py_INIT_FAILED(err)) {
            PyMem_RawFree(env);
            return err;
        }
    }
    PyMem_RawFree(env);
    return _Py_INIT_OK();
}


static void
pymain_init_stdio(_PyMain *pymain)
{
    pymain->stdin_is_interactive = (isatty(fileno(stdin))
                                    || Py_InteractiveFlag);

#if defined(MS_WINDOWS) || defined(__CYGWIN__)
    /* don't translate newlines (\r\n <=> \n) */
    _setmode(fileno(stdin), O_BINARY);
    _setmode(fileno(stdout), O_BINARY);
    _setmode(fileno(stderr), O_BINARY);
#endif

    if (Py_UnbufferedStdioFlag) {
#ifdef HAVE_SETVBUF
        setvbuf(stdin,  (char *)NULL, _IONBF, BUFSIZ);
        setvbuf(stdout, (char *)NULL, _IONBF, BUFSIZ);
        setvbuf(stderr, (char *)NULL, _IONBF, BUFSIZ);
#else /* !HAVE_SETVBUF */
        setbuf(stdin,  (char *)NULL);
        setbuf(stdout, (char *)NULL);
        setbuf(stderr, (char *)NULL);
#endif /* !HAVE_SETVBUF */
    }
    else if (Py_InteractiveFlag) {
#ifdef MS_WINDOWS
        /* Doesn't have to have line-buffered -- use unbuffered */
        /* Any set[v]buf(stdin, ...) screws up Tkinter :-( */
        setvbuf(stdout, (char *)NULL, _IONBF, BUFSIZ);
#else /* !MS_WINDOWS */
#ifdef HAVE_SETVBUF
        setvbuf(stdin,  (char *)NULL, _IOLBF, BUFSIZ);
        setvbuf(stdout, (char *)NULL, _IOLBF, BUFSIZ);
#endif /* HAVE_SETVBUF */
#endif /* !MS_WINDOWS */
        /* Leave stderr alone - it should be unbuffered anyway. */
    }
}


/* Get the program name: use PYTHONEXECUTABLE and __PYVENV_LAUNCHER__
   environment variables on macOS if available. */
static _PyInitError
config_init_program_name(_PyCoreConfig *config)
{
    assert(config->program_name == NULL);

    /* If Py_SetProgramName() was called, use its value */
    const wchar_t *program_name = _Py_path_config.program_name;
    if (program_name != NULL) {
        config->program_name = _PyMem_RawWcsdup(program_name);
        if (config->program_name == NULL) {
            return _Py_INIT_NO_MEMORY();
        }
        return _Py_INIT_OK();
    }

#ifdef __APPLE__
    /* On MacOS X, when the Python interpreter is embedded in an
       application bundle, it gets executed by a bootstrapping script
       that does os.execve() with an argv[0] that's different from the
       actual Python executable. This is needed to keep the Finder happy,
       or rather, to work around Apple's overly strict requirements of
       the process name. However, we still need a usable sys.executable,
       so the actual executable path is passed in an environment variable.
       See Lib/plat-mac/bundlebuiler.py for details about the bootstrap
       script. */
    const char *p = config_get_env_var("PYTHONEXECUTABLE");
    if (p != NULL) {
        size_t len;
        wchar_t* program_name = Py_DecodeLocale(p, &len);
        if (program_name == NULL) {
            return DECODE_LOCALE_ERR("PYTHONEXECUTABLE environment "
                                     "variable", (Py_ssize_t)len);
        }
        config->program_name = program_name;
        return _Py_INIT_OK();
    }
#ifdef WITH_NEXT_FRAMEWORK
    else {
        const char* pyvenv_launcher = getenv("__PYVENV_LAUNCHER__");
        if (pyvenv_launcher && *pyvenv_launcher) {
            /* Used by Mac/Tools/pythonw.c to forward
             * the argv0 of the stub executable
             */
            size_t len;
            wchar_t* program_name = Py_DecodeLocale(pyvenv_launcher, &len);
            if (program_name == NULL) {
                return DECODE_LOCALE_ERR("__PYVENV_LAUNCHER__ environment "
                                         "variable", (Py_ssize_t)len);
            }
            config->program_name = program_name;
            return _Py_INIT_OK();
        }
    }
#endif   /* WITH_NEXT_FRAMEWORK */
#endif   /* __APPLE__ */

    /* Use argv[0] by default, if available */
    if (config->program != NULL) {
        config->program_name = _PyMem_RawWcsdup(config->program);
        if (config->program_name == NULL) {
            return _Py_INIT_NO_MEMORY();
        }
        return _Py_INIT_OK();
    }

    /* Last fall back: hardcoded string */
#ifdef MS_WINDOWS
    const wchar_t *default_program_name = L"python";
#else
    const wchar_t *default_program_name = L"python3";
#endif
    config->program_name = _PyMem_RawWcsdup(default_program_name);
    if (config->program_name == NULL) {
        return _Py_INIT_NO_MEMORY();
    }
    return _Py_INIT_OK();
}


static _PyInitError
config_init_executable(_PyCoreConfig *config)
{
    assert(config->executable == NULL);

    /* If Py_SetProgramFullPath() was called, use its value */
    const wchar_t *program_full_path = _Py_path_config.program_full_path;
    if (program_full_path != NULL) {
        config->executable = _PyMem_RawWcsdup(program_full_path);
        if (config->executable == NULL) {
            return _Py_INIT_NO_MEMORY();
        }
        return _Py_INIT_OK();
    }

    return _Py_INIT_OK();
}


static void
pymain_header(_PyMain *pymain)
{
    if (Py_QuietFlag) {
        return;
    }

    if (!Py_VerboseFlag && (RUN_CODE(pymain) || !pymain->stdin_is_interactive)) {
        return;
    }

    fprintf(stderr, "Python %s on %s\n", Py_GetVersion(), Py_GetPlatform());
    if (!Py_NoSiteFlag) {
        fprintf(stderr, "%s\n", COPYRIGHT);
    }
}


static wchar_t**
copy_wstrlist(int len, wchar_t **list)
{
    assert((len > 0 && list != NULL) || len == 0);
    size_t size = len * sizeof(list[0]);
    wchar_t **list_copy = PyMem_RawMalloc(size);
    if (list_copy == NULL) {
        return NULL;
    }
    for (int i=0; i < len; i++) {
        wchar_t* arg = _PyMem_RawWcsdup(list[i]);
        if (arg == NULL) {
            clear_wstrlist(i, list_copy);
            return NULL;
        }
        list_copy[i] = arg;
    }
    return list_copy;
}


static int
pymain_init_core_argv(_PyMain *pymain, _PyCoreConfig *config,
                      _PyCmdline *cmdline)
{
    /* Copy argv to be able to modify it (to force -c/-m) */
    int argc = pymain->argc - _PyOS_optind;
    wchar_t **argv;

    if (argc <= 0 || cmdline->argv == NULL) {
        /* Ensure at least one (empty) argument is seen */
        static wchar_t *empty_argv[1] = {L""};
        argc = 1;
        argv = copy_wstrlist(1, empty_argv);
    }
    else {
        argv = copy_wstrlist(argc, &cmdline->argv[_PyOS_optind]);
    }

    if (argv == NULL) {
        pymain->err = _Py_INIT_NO_MEMORY();
        return -1;
    }

    wchar_t *arg0 = NULL;
    if (pymain->command != NULL) {
        /* Force sys.argv[0] = '-c' */
        arg0 = L"-c";
    }
    else if (pymain->module != NULL) {
        /* Force sys.argv[0] = '-m'*/
        arg0 = L"-m";
    }
    if (arg0 != NULL) {
        arg0 = _PyMem_RawWcsdup(arg0);
        if (arg0 == NULL) {
            clear_wstrlist(argc, argv);
            pymain->err = _Py_INIT_NO_MEMORY();
            return -1;
        }

        assert(argc >= 1);
        PyMem_RawFree(argv[0]);
        argv[0] = arg0;
    }

    config->argc = argc;
    config->argv = argv;
    return 0;
}


static PyObject*
_Py_wstrlist_as_pylist(int len, wchar_t **list)
{
    assert(list != NULL || len < 1);

    PyObject *pylist = PyList_New(len);
    if (pylist == NULL) {
        return NULL;
    }

    for (int i = 0; i < len; i++) {
        PyObject *v = PyUnicode_FromWideChar(list[i], -1);
        if (v == NULL) {
            Py_DECREF(pylist);
            return NULL;
        }
        PyList_SET_ITEM(pylist, i, v);
    }
    return pylist;
}


static int
pymain_update_sys_path(_PyMain *pymain, PyObject *path0)
{
    /* Prepend argv[0] to sys.path.
       If argv[0] is a symlink, use the real path. */
    PyObject *sys_path = PySys_GetObject("path");
    if (sys_path == NULL) {
        pymain->err = _Py_INIT_ERR("can't get sys.path");
        return -1;
    }

    /* Prepend path0 to sys.path */
    if (PyList_Insert(sys_path, 0, path0) < 0) {
        pymain->err = _Py_INIT_ERR("sys.path.insert(0, path0) failed");
        return -1;
    }
    return 0;
}


PyObject *
_Py_GetGlobalVariablesAsDict(void)
{
    PyObject *dict, *obj;

    dict = PyDict_New();
    if (dict == NULL) {
        return NULL;
    }

#define SET_ITEM(KEY, EXPR) \
        do { \
            obj = (EXPR); \
            if (obj == NULL) { \
                return NULL; \
            } \
            int res = PyDict_SetItemString(dict, (KEY), obj); \
            Py_DECREF(obj); \
            if (res < 0) { \
                goto fail; \
            } \
        } while (0)
#define SET_ITEM_INT(VAR) \
    SET_ITEM(#VAR, PyLong_FromLong(VAR))
#define FROM_STRING(STR) \
    ((STR != NULL) ? \
        PyUnicode_FromString(STR) \
        : (Py_INCREF(Py_None), Py_None))
#define SET_ITEM_STR(VAR) \
    SET_ITEM(#VAR, FROM_STRING(VAR))

    SET_ITEM_STR(Py_FileSystemDefaultEncoding);
    SET_ITEM_INT(Py_HasFileSystemDefaultEncoding);
    SET_ITEM_STR(Py_FileSystemDefaultEncodeErrors);

    SET_ITEM_INT(Py_UTF8Mode);
    SET_ITEM_INT(Py_DebugFlag);
    SET_ITEM_INT(Py_VerboseFlag);
    SET_ITEM_INT(Py_QuietFlag);
    SET_ITEM_INT(Py_InteractiveFlag);
    SET_ITEM_INT(Py_InspectFlag);

    SET_ITEM_INT(Py_OptimizeFlag);
    SET_ITEM_INT(Py_NoSiteFlag);
    SET_ITEM_INT(Py_BytesWarningFlag);
    SET_ITEM_INT(Py_FrozenFlag);
    SET_ITEM_INT(Py_IgnoreEnvironmentFlag);
    SET_ITEM_INT(Py_DontWriteBytecodeFlag);
    SET_ITEM_INT(Py_NoUserSiteDirectory);
    SET_ITEM_INT(Py_UnbufferedStdioFlag);
    SET_ITEM_INT(Py_HashRandomizationFlag);
    SET_ITEM_INT(Py_IsolatedFlag);

#ifdef MS_WINDOWS
    SET_ITEM_INT(Py_LegacyWindowsFSEncodingFlag);
    SET_ITEM_INT(Py_LegacyWindowsStdioFlag);
#endif

    return dict;

fail:
    Py_DECREF(dict);
    return NULL;

#undef FROM_STRING
#undef SET_ITEM
#undef SET_ITEM_INT
#undef SET_ITEM_STR
}


void
_PyCoreConfig_GetGlobalConfig(_PyCoreConfig *config)
{
#define COPY_FLAG(ATTR, VALUE) \
        if (config->ATTR == -1) { \
            config->ATTR = VALUE; \
        }

    COPY_FLAG(ignore_environment, Py_IgnoreEnvironmentFlag);
    COPY_FLAG(utf8_mode, Py_UTF8Mode);

#undef COPY_FLAG
}


/* Get Py_xxx global configuration variables */
static void
cmdline_get_global_config(_PyCmdline *cmdline)
{
    cmdline->bytes_warning = Py_BytesWarningFlag;
    cmdline->debug = Py_DebugFlag;
    cmdline->inspect = Py_InspectFlag;
    cmdline->interactive = Py_InteractiveFlag;
    cmdline->isolated = Py_IsolatedFlag;
    cmdline->optimization_level = Py_OptimizeFlag;
    cmdline->dont_write_bytecode = Py_DontWriteBytecodeFlag;
    cmdline->no_user_site_directory = Py_NoUserSiteDirectory;
    cmdline->no_site_import = Py_NoSiteFlag;
    cmdline->use_unbuffered_io = Py_UnbufferedStdioFlag;
    cmdline->verbosity = Py_VerboseFlag;
    cmdline->quiet_flag = Py_QuietFlag;
#ifdef MS_WINDOWS
    cmdline->legacy_windows_fs_encoding = Py_LegacyWindowsFSEncodingFlag;
    cmdline->legacy_windows_stdio = Py_LegacyWindowsStdioFlag;
#endif
    cmdline->check_hash_pycs_mode = _Py_CheckHashBasedPycsMode ;
}


void
_PyCoreConfig_SetGlobalConfig(const _PyCoreConfig *config)
{
    Py_IgnoreEnvironmentFlag = config->ignore_environment;
    Py_UTF8Mode = config->utf8_mode;

    /* Random or non-zero hash seed */
    Py_HashRandomizationFlag = (config->use_hash_seed == 0 ||
                                config->hash_seed != 0);
}


/* Set Py_xxx global configuration variables */
static void
cmdline_set_global_config(_PyCmdline *cmdline)
{
    Py_BytesWarningFlag = cmdline->bytes_warning;
    Py_DebugFlag = cmdline->debug;
    Py_InspectFlag = cmdline->inspect;
    Py_InteractiveFlag = cmdline->interactive;
    Py_IsolatedFlag = cmdline->isolated;
    Py_OptimizeFlag = cmdline->optimization_level;
    Py_DontWriteBytecodeFlag = cmdline->dont_write_bytecode;
    Py_NoUserSiteDirectory = cmdline->no_user_site_directory;
    Py_NoSiteFlag = cmdline->no_site_import;
    Py_UnbufferedStdioFlag = cmdline->use_unbuffered_io;
    Py_VerboseFlag = cmdline->verbosity;
    Py_QuietFlag = cmdline->quiet_flag;
    _Py_CheckHashBasedPycsMode = cmdline->check_hash_pycs_mode;
#ifdef MS_WINDOWS
    Py_LegacyWindowsFSEncodingFlag = cmdline->legacy_windows_fs_encoding;
    Py_LegacyWindowsStdioFlag = cmdline->legacy_windows_stdio;
#endif
}


static void
pymain_import_readline(_PyMain *pymain)
{
    if (Py_IsolatedFlag) {
        return;
    }
    if (!Py_InspectFlag && RUN_CODE(pymain)) {
        return;
    }
    if (!isatty(fileno(stdin))) {
        return;
    }

    PyObject *mod = PyImport_ImportModule("readline");
    if (mod == NULL) {
        PyErr_Clear();
    }
    else {
        Py_DECREF(mod);
    }
}


static FILE*
pymain_open_filename(_PyMain *pymain)
{
    const _PyCoreConfig *config = &_PyGILState_GetInterpreterStateUnsafe()->core_config;
    FILE* fp;

    fp = _Py_wfopen(pymain->filename, L"rb");
    if (fp == NULL) {
        char *cfilename_buffer;
        const char *cfilename;
        int err = errno;
        cfilename_buffer = _Py_EncodeLocaleRaw(pymain->filename, NULL);
        if (cfilename_buffer != NULL)
            cfilename = cfilename_buffer;
        else
            cfilename = "<unprintable file name>";
        fprintf(stderr, "%ls: can't open file '%s': [Errno %d] %s\n",
                config->program, cfilename, err, strerror(err));
        PyMem_RawFree(cfilename_buffer);
        pymain->status = 2;
        return NULL;
    }

    if (pymain->skip_first_line) {
        int ch;
        /* Push back first newline so line numbers
           remain the same */
        while ((ch = getc(fp)) != EOF) {
            if (ch == '\n') {
                (void)ungetc(ch, fp);
                break;
            }
        }
    }

    struct _Py_stat_struct sb;
    if (_Py_fstat_noraise(fileno(fp), &sb) == 0 &&
            S_ISDIR(sb.st_mode)) {
        fprintf(stderr,
                "%ls: '%ls' is a directory, cannot continue\n",
                config->program, pymain->filename);
        fclose(fp);
        pymain->status = 1;
        return NULL;
    }

    return fp;
}


static void
pymain_run_filename(_PyMain *pymain, PyCompilerFlags *cf)
{
    if (pymain->filename == NULL && pymain->stdin_is_interactive) {
        Py_InspectFlag = 0; /* do exit on SystemExit */
        pymain_run_startup(cf);
        pymain_run_interactive_hook();
    }

    if (pymain->main_importer_path != NULL) {
        pymain->status = pymain_run_main_from_importer(pymain);
        return;
    }

    FILE *fp;
    if (pymain->filename != NULL) {
        fp = pymain_open_filename(pymain);
        if (fp == NULL) {
            return;
        }
    }
    else {
        fp = stdin;
    }

    pymain->status = pymain_run_file(fp, pymain->filename, cf);
}


static void
pymain_repl(_PyMain *pymain, PyCompilerFlags *cf)
{
    /* Check this environment variable at the end, to give programs the
       opportunity to set it from Python. */
    if (!Py_InspectFlag && config_get_env_var("PYTHONINSPECT")) {
        Py_InspectFlag = 1;
    }

    if (!(Py_InspectFlag && pymain->stdin_is_interactive && RUN_CODE(pymain))) {
        return;
    }

    Py_InspectFlag = 0;
    pymain_run_interactive_hook();

    int res = PyRun_AnyFileFlags(stdin, "<stdin>", cf);
    pymain->status = (res != 0);
}


/* Parse the command line.
   Handle --version and --help options directly.

   Return 1 if Python must exit.
   Return 0 on success.
   Set pymain->err and return -1 on failure. */
static int
pymain_parse_cmdline(_PyMain *pymain, _PyCoreConfig *config,
                     _PyCmdline *cmdline)
{
    int res = pymain_parse_cmdline_impl(pymain, config, cmdline);
    if (res < 0) {
        return -1;
    }
    if (res) {
        pymain_usage(1, config->program);
        pymain->status = 2;
        return 1;
    }

    if (pymain->command != NULL || pymain->module != NULL) {
        /* Backup _PyOS_optind */
        _PyOS_optind--;
    }

    return 0;
}


static const wchar_t*
config_get_xoption(_PyCoreConfig *config, wchar_t *name)
{
    int nxoption = config->nxoption;
    wchar_t **xoptions = config->xoptions;
    for (int i=0; i < nxoption; i++) {
        wchar_t *option = xoptions[i];
        size_t len;
        wchar_t *sep = wcschr(option, L'=');
        if (sep != NULL) {
            len = (sep - option);
        }
        else {
            len = wcslen(option);
        }
        if (wcsncmp(option, name, len) == 0 && name[len] == L'\0') {
            return option;
        }
    }
    return NULL;
}


static int
pymain_str_to_int(const char *str, int *result)
{
    errno = 0;
    const char *endptr = str;
    long value = strtol(str, (char **)&endptr, 10);
    if (*endptr != '\0' || errno == ERANGE) {
        return -1;
    }
    if (value < INT_MIN || value > INT_MAX) {
        return -1;
    }

    *result = (int)value;
    return 0;
}


static int
pymain_wstr_to_int(const wchar_t *wstr, int *result)
{
    errno = 0;
    const wchar_t *endptr = wstr;
    long value = wcstol(wstr, (wchar_t **)&endptr, 10);
    if (*endptr != '\0' || errno == ERANGE) {
        return -1;
    }
    if (value < INT_MIN || value > INT_MAX) {
        return -1;
    }

    *result = (int)value;
    return 0;
}


static _PyInitError
config_init_tracemalloc(_PyCoreConfig *config)
{
    int nframe;
    int valid;

    const char *env = config_get_env_var("PYTHONTRACEMALLOC");
    if (env) {
        if (!pymain_str_to_int(env, &nframe)) {
            valid = (nframe >= 1);
        }
        else {
            valid = 0;
        }
        if (!valid) {
            return _Py_INIT_USER_ERR("PYTHONTRACEMALLOC: invalid number "
                                     "of frames");
        }
        config->tracemalloc = nframe;
    }

    const wchar_t *xoption = config_get_xoption(config, L"tracemalloc");
    if (xoption) {
        const wchar_t *sep = wcschr(xoption, L'=');
        if (sep) {
            if (!pymain_wstr_to_int(sep + 1, &nframe)) {
                valid = (nframe >= 1);
            }
            else {
                valid = 0;
            }
            if (!valid) {
                return _Py_INIT_USER_ERR("-X tracemalloc=NFRAME: "
                                         "invalid number of frames");
            }
        }
        else {
            /* -X tracemalloc behaves as -X tracemalloc=1 */
            nframe = 1;
        }
        config->tracemalloc = nframe;
    }
    return _Py_INIT_OK();
}


static void
get_env_flag(int *flag, const char *name)
{
    const char *var = config_get_env_var(name);
    if (!var) {
        return;
    }
    int value;
    if (pymain_str_to_int(var, &value) < 0 || value < 0) {
        /* PYTHONDEBUG=text and PYTHONDEBUG=-2 behave as PYTHONDEBUG=1 */
        value = 1;
    }
    if (*flag < value) {
        *flag = value;
    }
}


static void
cmdline_get_env_flags(_PyCmdline *cmdline)
{
    get_env_flag(&cmdline->debug, "PYTHONDEBUG");
    get_env_flag(&cmdline->verbosity, "PYTHONVERBOSE");
    get_env_flag(&cmdline->optimization_level, "PYTHONOPTIMIZE");
    get_env_flag(&cmdline->inspect, "PYTHONINSPECT");
    get_env_flag(&cmdline->dont_write_bytecode, "PYTHONDONTWRITEBYTECODE");
    get_env_flag(&cmdline->no_user_site_directory, "PYTHONNOUSERSITE");
    get_env_flag(&cmdline->use_unbuffered_io, "PYTHONUNBUFFERED");
#ifdef MS_WINDOWS
    get_env_flag(&cmdline->legacy_windows_fs_encoding,
                 "PYTHONLEGACYWINDOWSFSENCODING");
    get_env_flag(&cmdline->legacy_windows_stdio,
                 "PYTHONLEGACYWINDOWSSTDIO");
#endif
}


/* Set global variable variables from environment variables */
void
_Py_Initialize_ReadEnvVarsNoAlloc(void)
{
    _PyCmdline cmdline;
    memset(&cmdline, 0, sizeof(cmdline));

    cmdline_get_global_config(&cmdline);
    if (cmdline.isolated) {
        Py_IgnoreEnvironmentFlag = 1;
        cmdline.no_user_site_directory = 1;
    }
    if (!Py_IgnoreEnvironmentFlag) {
        cmdline_get_env_flags(&cmdline);
    }
    cmdline_set_global_config(&cmdline);

    /* no need to call pymain_clear_cmdline(), no memory has been allocated */
}


static _PyInitError
config_init_home(_PyCoreConfig *config)
{
    wchar_t *home;

    /* If Py_SetPythonHome() was called, use its value */
    home = _Py_path_config.home;
    if (home) {
        config->home = _PyMem_RawWcsdup(home);
        if (config->home == NULL) {
            return _Py_INIT_NO_MEMORY();
        }
        return _Py_INIT_OK();
    }

    int res = config_get_env_var_dup(&home, L"PYTHONHOME", "PYTHONHOME");
    if (res < 0) {
        return DECODE_LOCALE_ERR("PYTHONHOME", res);
    }
    config->home = home;
    return _Py_INIT_OK();
}


static _PyInitError
config_init_hash_seed(_PyCoreConfig *config)
{
    const char *seed_text = config_get_env_var("PYTHONHASHSEED");
    int use_hash_seed;
    unsigned long hash_seed;
    if (_Py_ReadHashSeed(seed_text, &use_hash_seed, &hash_seed) < 0) {
        return _Py_INIT_USER_ERR("PYTHONHASHSEED must be \"random\" "
                                 "or an integer in range [0; 4294967295]");
    }
    config->use_hash_seed = use_hash_seed;
    config->hash_seed = hash_seed;
    return _Py_INIT_OK();
}


static _PyInitError
config_init_utf8_mode(_PyCoreConfig *config)
{
    const wchar_t *xopt = config_get_xoption(config, L"utf8");
    if (xopt) {
        wchar_t *sep = wcschr(xopt, L'=');
        if (sep) {
            xopt = sep + 1;
            if (wcscmp(xopt, L"1") == 0) {
                config->utf8_mode = 1;
            }
            else if (wcscmp(xopt, L"0") == 0) {
                config->utf8_mode = 0;
            }
            else {
                return _Py_INIT_USER_ERR("invalid -X utf8 option value");
            }
        }
        else {
            config->utf8_mode = 1;
        }
        return _Py_INIT_OK();
    }

    const char *opt = config_get_env_var("PYTHONUTF8");
    if (opt) {
        if (strcmp(opt, "1") == 0) {
            config->utf8_mode = 1;
        }
        else if (strcmp(opt, "0") == 0) {
            config->utf8_mode = 0;
        }
        else {
            return _Py_INIT_USER_ERR("invalid PYTHONUTF8 environment "
                                     "variable value");
        }
        return _Py_INIT_OK();
    }

    return _Py_INIT_OK();
}


static _PyInitError
config_read_env_vars(_PyCoreConfig *config)
{
    assert(!config->ignore_environment);

    if (config->allocator == NULL) {
        config->allocator = config_get_env_var("PYTHONMALLOC");
    }

    if (config_get_env_var("PYTHONDUMPREFS")) {
        config->dump_refs = 1;
    }
    if (config_get_env_var("PYTHONMALLOCSTATS")) {
        config->malloc_stats = 1;
    }

    const char *env = config_get_env_var("PYTHONCOERCECLOCALE");
    if (env) {
        if (strcmp(env, "0") == 0) {
            if (config->coerce_c_locale < 0) {
                config->coerce_c_locale = 0;
            }
        }
        else if (strcmp(env, "warn") == 0) {
            config->coerce_c_locale_warn = 1;
        }
        else {
            if (config->coerce_c_locale < 0) {
                config->coerce_c_locale = 1;
            }
        }
    }

    wchar_t *path;
    int res = config_get_env_var_dup(&path, L"PYTHONPATH", "PYTHONPATH");
    if (res < 0) {
        return DECODE_LOCALE_ERR("PYTHONPATH", res);
    }
    config->module_search_path_env = path;

    if (config->use_hash_seed < 0) {
        _PyInitError err = config_init_hash_seed(config);
        if (_Py_INIT_FAILED(err)) {
            return err;
        }
    }

    return _Py_INIT_OK();
}


static _PyInitError
config_read_complex_options(_PyCoreConfig *config)
{
    /* More complex options configured by env var and -X option */
    if (config->faulthandler < 0) {
        if (config_get_env_var("PYTHONFAULTHANDLER")
           || config_get_xoption(config, L"faulthandler")) {
            config->faulthandler = 1;
        }
    }
    if (config_get_env_var("PYTHONPROFILEIMPORTTIME")
       || config_get_xoption(config, L"importtime")) {
        config->import_time = 1;
    }
    if (config_get_xoption(config, L"dev" ) ||
        config_get_env_var("PYTHONDEVMODE"))
    {
        config->dev_mode = 1;
    }

    if (config->tracemalloc < 0) {
        _PyInitError err = config_init_tracemalloc(config);
        if (_Py_INIT_FAILED(err)) {
            return err;
        }
    }
    return _Py_INIT_OK();
}


/* Parse command line options and environment variables.
   This code must not use Python runtime apart PyMem_Raw memory allocator.

   Return 0 on success.
   Return 1 if Python is done and must exit.
   Set pymain->err and return -1 on error. */
static int
pymain_read_conf_impl(_PyMain *pymain, _PyCoreConfig *config,
                      _PyCmdline *cmdline)
{
    _PyInitError err;

    int res = pymain_parse_cmdline(pymain, config, cmdline);
    if (res != 0) {
        return res;
    }

    /* Set Py_IgnoreEnvironmentFlag for Py_GETENV() */
    Py_IgnoreEnvironmentFlag = config->ignore_environment || cmdline->isolated;

    /* Get environment variables */
    if (!Py_IgnoreEnvironmentFlag) {
        cmdline_get_env_flags(cmdline);
    }

    err = cmdline_init_env_warnoptions(cmdline);
    if (_Py_INIT_FAILED(err)) {
        pymain->err = err;
        return -1;
    }

#ifdef MS_WINDOWS
    if (cmdline->legacy_windows_fs_encoding) {
        config->utf8_mode = 0;
    }
#endif

    if (pymain_init_core_argv(pymain, config, cmdline) < 0) {
        return -1;
    }

    /* On Windows, _PyPathConfig_Init() modifies Py_IsolatedFlag and
       Py_NoSiteFlag variables if a "._pth" file is found. */
    int init_isolated = Py_IsolatedFlag;
    int init_no_site = Py_NoSiteFlag;
    Py_IsolatedFlag = cmdline->isolated;
    Py_NoSiteFlag = cmdline->no_site_import;

    err = _PyCoreConfig_Read(config);

    cmdline->isolated = Py_IsolatedFlag;
    cmdline->no_site_import = Py_NoSiteFlag;
    Py_IsolatedFlag = init_isolated;
    Py_NoSiteFlag = init_no_site;

    if (_Py_INIT_FAILED(err)) {
        pymain->err = err;
        return -1;
    }
    return 0;
}


/* Read the configuration, but initialize also the LC_CTYPE locale:
   enable UTF-8 mode (PEP 540) and/or coerce the C locale (PEP 538) */
static int
pymain_read_conf(_PyMain *pymain, _PyCoreConfig *config, _PyCmdline *cmdline)
{
    int init_utf8_mode = Py_UTF8Mode;
    _PyCoreConfig save_config = _PyCoreConfig_INIT;
    int res = -1;

    char *oldloc = _PyMem_RawStrdup(setlocale(LC_ALL, NULL));
    if (oldloc == NULL) {
        pymain->err = _Py_INIT_NO_MEMORY();
        goto done;
    }

    /* Reconfigure the locale to the default for this process */
    _Py_SetLocaleFromEnv(LC_ALL);

    int locale_coerced = 0;
    int loops = 0;
    int init_ignore_env = config->ignore_environment;

    if (_PyCoreConfig_Copy(&save_config, config) < 0) {
        pymain->err = _Py_INIT_NO_MEMORY();
        goto done;
    }

    while (1) {
        int init_utf8_mode = config->utf8_mode;
        int encoding_changed = 0;

        /* Watchdog to prevent an infinite loop */
        loops++;
        if (loops == 3) {
            pymain->err = _Py_INIT_ERR("Encoding changed twice while "
                                       "reading the configuration");
            goto done;
        }

        /* bpo-34207: Py_DecodeLocale(), Py_EncodeLocale() and similar
           functions depend on Py_UTF8Mode. */
        Py_UTF8Mode = config->utf8_mode;

        if (pymain_init_cmdline_argv(pymain, config, cmdline) < 0) {
            goto done;
        }

        int conf_res = pymain_read_conf_impl(pymain, config, cmdline);
        if (conf_res != 0) {
            res = conf_res;
            goto done;
        }

        /* The legacy C locale assumes ASCII as the default text encoding, which
         * causes problems not only for the CPython runtime, but also other
         * components like GNU readline.
         *
         * Accordingly, when the CLI detects it, it attempts to coerce it to a
         * more capable UTF-8 based alternative.
         *
         * See the documentation of the PYTHONCOERCECLOCALE setting for more
         * details.
         */
        if (config->coerce_c_locale && !locale_coerced) {
            locale_coerced = 1;
            _Py_CoerceLegacyLocale(config);
            encoding_changed = 1;
        }

        if (init_utf8_mode == -1) {
            if (config->utf8_mode == 1) {
                /* UTF-8 Mode enabled */
                encoding_changed = 1;
            }
        }
        else {
            if (config->utf8_mode != init_utf8_mode) {
                encoding_changed = 1;
            }
        }

        if (!encoding_changed) {
            break;
        }

        /* Reset the configuration, except UTF-8 Mode. Set Py_UTF8Mode for
           Py_DecodeLocale(). Reset Py_IgnoreEnvironmentFlag, modified by
           pymain_read_conf_impl(). Reset Py_IsolatedFlag and Py_NoSiteFlag
           modified by _PyCoreConfig_Read(). */
        int new_utf8_mode = config->utf8_mode;
        int new_coerce_c_locale = config->coerce_c_locale;
        Py_IgnoreEnvironmentFlag = init_ignore_env;
        if (_PyCoreConfig_Copy(config, &save_config) < 0) {
            pymain->err = _Py_INIT_NO_MEMORY();
            goto done;
        }
        pymain_clear_cmdline(pymain, cmdline);
        pymain_clear_pymain(pymain);
        memset(cmdline, 0, sizeof(*cmdline));

        cmdline_get_global_config(cmdline);
        _PyCoreConfig_GetGlobalConfig(config);
        config->utf8_mode = new_utf8_mode;
        config->coerce_c_locale = new_coerce_c_locale;

        /* The encoding changed: read again the configuration
           with the new encoding */
    }
    res = 0;

done:
    _PyCoreConfig_Clear(&save_config);
    if (oldloc != NULL) {
        setlocale(LC_ALL, oldloc);
        PyMem_RawFree(oldloc);
    }
    Py_UTF8Mode = init_utf8_mode ;
    return res;
}


static void
config_init_locale(_PyCoreConfig *config)
{
    /* Test also if coerce_c_locale equals 1: PYTHONCOERCECLOCALE=1 doesn't
       imply that the C locale is always coerced. It is only coerced if
       if the LC_CTYPE locale is "C". */
    if (config->coerce_c_locale != 0) {
        /* The C locale enables the C locale coercion (PEP 538) */
        if (_Py_LegacyLocaleDetected()) {
            config->coerce_c_locale = 1;
        }
        else {
            config->coerce_c_locale = 0;
        }
    }

#ifndef MS_WINDOWS
    if (config->utf8_mode < 0) {
        /* The C locale and the POSIX locale enable the UTF-8 Mode (PEP 540) */
        const char *ctype_loc = setlocale(LC_CTYPE, NULL);
        if (ctype_loc != NULL
           && (strcmp(ctype_loc, "C") == 0
               || strcmp(ctype_loc, "POSIX") == 0))
        {
            config->utf8_mode = 1;
        }
    }
#endif
}


static _PyInitError
config_init_module_search_paths(_PyCoreConfig *config)
{
    assert(config->module_search_paths == NULL);
    assert(config->nmodule_search_path < 0);

    config->nmodule_search_path = 0;

    const wchar_t *sys_path = Py_GetPath();
    const wchar_t delim = DELIM;
    const wchar_t *p = sys_path;
    while (1) {
        p = wcschr(sys_path, delim);
        if (p == NULL) {
            p = sys_path + wcslen(sys_path); /* End of string */
        }

        size_t path_len = (p - sys_path);
        wchar_t *path = PyMem_RawMalloc((path_len + 1) * sizeof(wchar_t));
        if (path == NULL) {
            return _Py_INIT_NO_MEMORY();
        }
        memcpy(path, sys_path, path_len * sizeof(wchar_t));
        path[path_len] = L'\0';

        _PyInitError err = wstrlist_append(&config->nmodule_search_path,
                                           &config->module_search_paths,
                                           path);
        PyMem_RawFree(path);
        if (_Py_INIT_FAILED(err)) {
            return err;
        }

        if (*p == '\0') {
            break;
        }
        sys_path = p + 1;
    }
    return _Py_INIT_OK();
}


static _PyInitError
config_init_path_config(_PyCoreConfig *config)
{
    _PyInitError err = _PyPathConfig_Init(config);
    if (_Py_INIT_FAILED(err)) {
        return err;
    }

    if (config->nmodule_search_path < 0) {
        err = config_init_module_search_paths(config);
        if (_Py_INIT_FAILED(err)) {
            return err;
        }
    }

    if (config->executable == NULL) {
        config->executable = _PyMem_RawWcsdup(Py_GetProgramFullPath());
        if (config->executable == NULL) {
            return _Py_INIT_NO_MEMORY();
        }
    }

    if (config->prefix == NULL) {
        config->prefix = _PyMem_RawWcsdup(Py_GetPrefix());
        if (config->prefix == NULL) {
            return _Py_INIT_NO_MEMORY();
        }
    }

    if (config->exec_prefix == NULL) {
        config->exec_prefix = _PyMem_RawWcsdup(Py_GetExecPrefix());
        if (config->exec_prefix == NULL) {
            return _Py_INIT_NO_MEMORY();
        }
    }

    if (config->base_prefix == NULL) {
        config->base_prefix = _PyMem_RawWcsdup(config->prefix);
        if (config->base_prefix == NULL) {
            return _Py_INIT_NO_MEMORY();
        }
    }

    if (config->base_exec_prefix == NULL) {
        config->base_exec_prefix = _PyMem_RawWcsdup(config->exec_prefix);
        if (config->base_exec_prefix == NULL) {
            return _Py_INIT_NO_MEMORY();
        }
    }

    return _Py_INIT_OK();
}

/* Read configuration settings from standard locations
 *
 * This function doesn't make any changes to the interpreter state - it
 * merely populates any missing configuration settings. This allows an
 * embedding application to completely override a config option by
 * setting it before calling this function, or else modify the default
 * setting before passing the fully populated config to Py_EndInitialization.
 *
 * More advanced selective initialization tricks are possible by calling
 * this function multiple times with various preconfigured settings.
 */

_PyInitError
_PyCoreConfig_Read(_PyCoreConfig *config)
{
    _PyInitError err;

    _PyCoreConfig_GetGlobalConfig(config);

    assert(config->ignore_environment >= 0);
    if (!config->ignore_environment) {
        err = config_read_env_vars(config);
        if (_Py_INIT_FAILED(err)) {
            return err;
        }
    }

    /* -X options */
    if (config_get_xoption(config, L"showrefcount")) {
        config->show_ref_count = 1;
    }
    if (config_get_xoption(config, L"showalloccount")) {
        config->show_alloc_count = 1;
    }

    err = config_read_complex_options(config);
    if (_Py_INIT_FAILED(err)) {
        return err;
    }

    if (config->utf8_mode < 0) {
        err = config_init_utf8_mode(config);
        if (_Py_INIT_FAILED(err)) {
            return err;
        }
    }

    if (config->home == NULL) {
        err = config_init_home(config);
        if (_Py_INIT_FAILED(err)) {
            return err;
        }
    }

    if (config->program_name == NULL) {
        err = config_init_program_name(config);
        if (_Py_INIT_FAILED(err)) {
            return err;
        }
    }

    if (config->executable == NULL) {
        err = config_init_executable(config);
        if (_Py_INIT_FAILED(err)) {
            return err;
        }
    }

    if (config->coerce_c_locale != 0 || config->utf8_mode < 0) {
        config_init_locale(config);
    }

    if (!config->_disable_importlib) {
        err = config_init_path_config(config);
        if (_Py_INIT_FAILED(err)) {
            return err;
        }
    }

    /* default values */
    if (config->dev_mode) {
        if (config->faulthandler < 0) {
            config->faulthandler = 1;
        }
        if (config->allocator == NULL) {
            config->allocator = "debug";
        }
    }
    if (config->install_signal_handlers < 0) {
        config->install_signal_handlers = 1;
    }
    if (config->use_hash_seed < 0) {
        config->use_hash_seed = 0;
        config->hash_seed = 0;
    }
    if (config->faulthandler < 0) {
        config->faulthandler = 0;
    }
    if (config->tracemalloc < 0) {
        config->tracemalloc = 0;
    }
    if (config->coerce_c_locale < 0) {
        config->coerce_c_locale = 0;
    }
    if (config->utf8_mode < 0) {
        config->utf8_mode = 0;
    }
    if (config->argc < 0) {
        config->argc = 0;
    }

    return _Py_INIT_OK();
}


void
_PyCoreConfig_Clear(_PyCoreConfig *config)
{
#define CLEAR(ATTR) \
    do { \
        PyMem_RawFree(ATTR); \
        ATTR = NULL; \
    } while (0)
#define CLEAR_WSTRLIST(LEN, LIST) \
    do { \
        clear_wstrlist(LEN, LIST); \
        LEN = 0; \
        LIST = NULL; \
    } while (0)

    CLEAR(config->module_search_path_env);
    CLEAR(config->home);
    CLEAR(config->program_name);
    CLEAR(config->program);

    CLEAR_WSTRLIST(config->argc, config->argv);
    config->argc = -1;

    CLEAR_WSTRLIST(config->nwarnoption, config->warnoptions);
    CLEAR_WSTRLIST(config->nxoption, config->xoptions);
    CLEAR_WSTRLIST(config->nmodule_search_path, config->module_search_paths);
    config->nmodule_search_path = -1;

    CLEAR(config->executable);
    CLEAR(config->prefix);
    CLEAR(config->base_prefix);
    CLEAR(config->exec_prefix);
    CLEAR(config->base_exec_prefix);
#undef CLEAR
#undef CLEAR_WSTRLIST
}


int
_PyCoreConfig_Copy(_PyCoreConfig *config, const _PyCoreConfig *config2)
{
    _PyCoreConfig_Clear(config);

#define COPY_ATTR(ATTR) config->ATTR = config2->ATTR
#define COPY_STR_ATTR(ATTR) \
    do { \
        if (config2->ATTR != NULL) { \
            config->ATTR = _PyMem_RawWcsdup(config2->ATTR); \
            if (config->ATTR == NULL) { \
                return -1; \
            } \
        } \
    } while (0)
#define COPY_WSTRLIST(LEN, LIST) \
    do { \
        if (config2->LIST != NULL) { \
            config->LIST = copy_wstrlist(config2->LEN, config2->LIST); \
            if (config->LIST == NULL) { \
                return -1; \
            } \
        } \
        config->LEN = config2->LEN; \
    } while (0)

    COPY_ATTR(install_signal_handlers);
    COPY_ATTR(ignore_environment);
    COPY_ATTR(use_hash_seed);
    COPY_ATTR(hash_seed);
    COPY_ATTR(_disable_importlib);
    COPY_ATTR(allocator);
    COPY_ATTR(dev_mode);
    COPY_ATTR(faulthandler);
    COPY_ATTR(tracemalloc);
    COPY_ATTR(import_time);
    COPY_ATTR(show_ref_count);
    COPY_ATTR(show_alloc_count);
    COPY_ATTR(dump_refs);
    COPY_ATTR(malloc_stats);

    COPY_ATTR(coerce_c_locale);
    COPY_ATTR(coerce_c_locale_warn);
    COPY_ATTR(utf8_mode);

    COPY_STR_ATTR(module_search_path_env);
    COPY_STR_ATTR(home);
    COPY_STR_ATTR(program_name);
    COPY_STR_ATTR(program);

    COPY_WSTRLIST(argc, argv);
    COPY_WSTRLIST(nwarnoption, warnoptions);
    COPY_WSTRLIST(nxoption, xoptions);
    COPY_WSTRLIST(nmodule_search_path, module_search_paths);

    COPY_STR_ATTR(executable);
    COPY_STR_ATTR(prefix);
    COPY_STR_ATTR(base_prefix);
    COPY_STR_ATTR(exec_prefix);
    COPY_STR_ATTR(base_exec_prefix);

#undef COPY_ATTR
#undef COPY_STR_ATTR
#undef COPY_WSTRLIST
    return 0;
}


PyObject *
_PyCoreConfig_AsDict(const _PyCoreConfig *config)
{
    PyObject *dict, *obj;

    dict = PyDict_New();
    if (dict == NULL) {
        return NULL;
    }

#define SET_ITEM(KEY, EXPR) \
        do { \
            obj = (EXPR); \
            if (obj == NULL) { \
                return NULL; \
            } \
            int res = PyDict_SetItemString(dict, (KEY), obj); \
            Py_DECREF(obj); \
            if (res < 0) { \
                goto fail; \
            } \
        } while (0)
#define FROM_STRING(STR) \
    ((STR != NULL) ? \
        PyUnicode_FromString(STR) \
        : (Py_INCREF(Py_None), Py_None))
#define SET_ITEM_INT(ATTR) \
    SET_ITEM(#ATTR, PyLong_FromLong(config->ATTR))
#define SET_ITEM_UINT(ATTR) \
    SET_ITEM(#ATTR, PyLong_FromUnsignedLong(config->ATTR))
#define SET_ITEM_STR(ATTR) \
    SET_ITEM(#ATTR, FROM_STRING(config->ATTR))
#define FROM_WSTRING(STR) \
    ((STR != NULL) ? \
        PyUnicode_FromWideChar(STR, -1) \
        : (Py_INCREF(Py_None), Py_None))
#define SET_ITEM_WSTR(ATTR) \
    SET_ITEM(#ATTR, FROM_WSTRING(config->ATTR))
#define SET_ITEM_WSTRLIST(NOPTION, OPTIONS) \
    SET_ITEM(#OPTIONS, _Py_wstrlist_as_pylist(config->NOPTION, config->OPTIONS))

    SET_ITEM_INT(install_signal_handlers);
    SET_ITEM_INT(ignore_environment);
    SET_ITEM_INT(use_hash_seed);
    SET_ITEM_UINT(hash_seed);
    SET_ITEM_STR(allocator);
    SET_ITEM_INT(dev_mode);
    SET_ITEM_INT(faulthandler);
    SET_ITEM_INT(tracemalloc);
    SET_ITEM_INT(import_time);
    SET_ITEM_INT(show_ref_count);
    SET_ITEM_INT(show_alloc_count);
    SET_ITEM_INT(dump_refs);
    SET_ITEM_INT(malloc_stats);
    SET_ITEM_INT(coerce_c_locale);
    SET_ITEM_INT(coerce_c_locale_warn);
    SET_ITEM_INT(utf8_mode);
    SET_ITEM_WSTR(program_name);
    SET_ITEM_WSTRLIST(argc, argv);
    SET_ITEM_WSTR(program);
    SET_ITEM_WSTRLIST(nxoption, xoptions);
    SET_ITEM_WSTRLIST(nwarnoption, warnoptions);
    SET_ITEM_WSTR(module_search_path_env);
    SET_ITEM_WSTR(home);
    SET_ITEM_WSTRLIST(nmodule_search_path, module_search_paths);
    SET_ITEM_WSTR(executable);
    SET_ITEM_WSTR(prefix);
    SET_ITEM_WSTR(base_prefix);
    SET_ITEM_WSTR(exec_prefix);
    SET_ITEM_WSTR(base_exec_prefix);
    SET_ITEM_INT(_disable_importlib);

    return dict;

fail:
    Py_DECREF(dict);
    return NULL;

#undef FROM_STRING
#undef FROM_WSTRING
#undef SET_ITEM
#undef SET_ITEM_INT
#undef SET_ITEM_UINT
#undef SET_ITEM_STR
#undef SET_ITEM_WSTR
#undef SET_ITEM_WSTRLIST
}


void
_PyMainInterpreterConfig_Clear(_PyMainInterpreterConfig *config)
{
    Py_CLEAR(config->argv);
    Py_CLEAR(config->executable);
    Py_CLEAR(config->prefix);
    Py_CLEAR(config->base_prefix);
    Py_CLEAR(config->exec_prefix);
    Py_CLEAR(config->base_exec_prefix);
    Py_CLEAR(config->warnoptions);
    Py_CLEAR(config->xoptions);
    Py_CLEAR(config->module_search_path);
}


static PyObject*
config_copy_attr(PyObject *obj)
{
    if (PyUnicode_Check(obj)) {
        Py_INCREF(obj);
        return obj;
    }
    else if (PyList_Check(obj)) {
        return PyList_GetSlice(obj, 0, Py_SIZE(obj));
    }
    else if (PyDict_Check(obj)) {
        /* The dict type is used for xoptions. Make the assumption that keys
           and values are immutables */
        return PyDict_Copy(obj);
    }
    else {
        PyErr_Format(PyExc_TypeError,
                     "cannot copy config attribute of type %.200s",
                     Py_TYPE(obj)->tp_name);
        return NULL;
    }
}


int
_PyMainInterpreterConfig_Copy(_PyMainInterpreterConfig *config,
                              const _PyMainInterpreterConfig *config2)
{
    _PyMainInterpreterConfig_Clear(config);

#define COPY_ATTR(ATTR) config->ATTR = config2->ATTR
#define COPY_OBJ_ATTR(ATTR) \
    do { \
        if (config2->ATTR != NULL) { \
            config->ATTR = config_copy_attr(config2->ATTR); \
            if (config->ATTR == NULL) { \
                return -1; \
            } \
        } \
    } while (0)

    COPY_ATTR(install_signal_handlers);
    COPY_OBJ_ATTR(argv);
    COPY_OBJ_ATTR(executable);
    COPY_OBJ_ATTR(prefix);
    COPY_OBJ_ATTR(base_prefix);
    COPY_OBJ_ATTR(exec_prefix);
    COPY_OBJ_ATTR(base_exec_prefix);
    COPY_OBJ_ATTR(warnoptions);
    COPY_OBJ_ATTR(xoptions);
    COPY_OBJ_ATTR(module_search_path);
#undef COPY_ATTR
#undef COPY_OBJ_ATTR
    return 0;
}


PyObject*
_PyMainInterpreterConfig_AsDict(const _PyMainInterpreterConfig *config)
{
    PyObject *dict, *obj;
    int res;

    dict = PyDict_New();
    if (dict == NULL) {
        return NULL;
    }

#define SET_ITEM_INT(ATTR) \
    do { \
        obj = PyLong_FromLong(config->ATTR); \
        if (obj == NULL) { \
            goto fail; \
        } \
        res = PyDict_SetItemString(dict, #ATTR, obj); \
        Py_DECREF(obj); \
        if (res < 0) { \
            goto fail; \
        } \
    } while (0)

#define SET_ITEM_OBJ(ATTR) \
    do { \
        obj = config->ATTR; \
        if (obj == NULL) { \
            obj = Py_None; \
        } \
        res = PyDict_SetItemString(dict, #ATTR, obj); \
        if (res < 0) { \
            goto fail; \
        } \
    } while (0)

    SET_ITEM_INT(install_signal_handlers);
    SET_ITEM_OBJ(argv);
    SET_ITEM_OBJ(executable);
    SET_ITEM_OBJ(prefix);
    SET_ITEM_OBJ(base_prefix);
    SET_ITEM_OBJ(exec_prefix);
    SET_ITEM_OBJ(base_exec_prefix);
    SET_ITEM_OBJ(warnoptions);
    SET_ITEM_OBJ(xoptions);
    SET_ITEM_OBJ(module_search_path);

    return dict;

fail:
    Py_DECREF(dict);
    return NULL;

#undef SET_ITEM_OBJ
}


_PyInitError
_PyMainInterpreterConfig_Read(_PyMainInterpreterConfig *main_config,
                              const _PyCoreConfig *config)
{
    if (main_config->install_signal_handlers < 0) {
        main_config->install_signal_handlers = config->install_signal_handlers;
    }

    if (main_config->xoptions == NULL) {
        main_config->xoptions = config_create_xoptions_dict(config);
        if (main_config->xoptions == NULL) {
            return _Py_INIT_NO_MEMORY();
        }
    }

#define COPY_WSTR(ATTR) \
    do { \
        if (main_config->ATTR == NULL) { \
            main_config->ATTR = PyUnicode_FromWideChar(config->ATTR, -1); \
            if (main_config->ATTR == NULL) { \
                return _Py_INIT_NO_MEMORY(); \
            } \
        } \
    } while (0)
#define COPY_WSTRLIST(ATTR, LEN, LIST) \
    do { \
        if (ATTR == NULL) { \
            ATTR = _Py_wstrlist_as_pylist(LEN, LIST); \
            if (ATTR == NULL) { \
                return _Py_INIT_NO_MEMORY(); \
            } \
        } \
    } while (0)

    COPY_WSTRLIST(main_config->warnoptions,
                  config->nwarnoption, config->warnoptions);
    if (config->argc >= 0) {
        COPY_WSTRLIST(main_config->argv,
                      config->argc, config->argv);
    }

    if (!config->_disable_importlib) {
        COPY_WSTR(executable);
        COPY_WSTR(prefix);
        COPY_WSTR(base_prefix);
        COPY_WSTR(exec_prefix);
        COPY_WSTR(base_exec_prefix);

        COPY_WSTRLIST(main_config->module_search_path,
                      config->nmodule_search_path, config->module_search_paths);
    }

    return _Py_INIT_OK();
#undef COPY_WSTR
#undef COPY_WSTRLIST
}


static int
pymain_init_python_main(_PyMain *pymain, PyInterpreterState *interp)
{
    _PyInitError err;

    _PyMainInterpreterConfig main_config = _PyMainInterpreterConfig_INIT;
    err = _PyMainInterpreterConfig_Read(&main_config, &interp->core_config);
    if (!_Py_INIT_FAILED(err)) {
        err = _Py_InitializeMainInterpreter(interp, &main_config);
    }
    _PyMainInterpreterConfig_Clear(&main_config);

    if (_Py_INIT_FAILED(err)) {
        pymain->err = err;
        return -1;
    }
    return 0;
}


static int
pymain_init_sys_path(_PyMain *pymain, _PyCoreConfig *config)
{
    if (pymain->filename != NULL) {
        /* If filename is a package (ex: directory or ZIP file) which contains
           __main__.py, main_importer_path is set to filename and will be
           prepended to sys.path by pymain_run_main_from_importer(). Otherwise,
           main_importer_path is set to NULL. */
        pymain->main_importer_path = pymain_get_importer(pymain->filename);
    }

    if (pymain->main_importer_path != NULL) {
        /* Let pymain_run_main_from_importer() adjust sys.path[0] later */
        return 0;
    }

    if (Py_IsolatedFlag || 1) { // Do not add current directory.
        return 0;
    }

    PyObject *path0 = NULL;
    if (!_PyPathConfig_ComputeArgv0(config->argc, config->argv, &path0)) {
        return 0;
    }
    if (path0 == NULL) {
        pymain->err = _Py_INIT_NO_MEMORY();
        return -1;
    }

    if (pymain_update_sys_path(pymain, path0) < 0) {
        Py_DECREF(path0);
        return -1;
    }
    Py_DECREF(path0);
    return 0;
}


static void
pymain_run_python(_PyMain *pymain)
{
    PyCompilerFlags cf = {.cf_flags = 0};

    pymain_header(pymain);
    pymain_import_readline(pymain);

    if (pymain->command) {
        pymain->status = pymain_run_command(pymain->command, &cf);
    }
    else if (pymain->module) {
        pymain->status = (pymain_run_module(pymain->module, 1) != 0);
    }
    else {
        pymain_run_filename(pymain, &cf);
    }

    pymain_repl(pymain, &cf);
}


static int
pymain_cmdline_impl(_PyMain *pymain, _PyCoreConfig *config,
                    _PyCmdline *cmdline)
{
    pymain->err = _PyRuntime_Initialize();
    if (_Py_INIT_FAILED(pymain->err)) {
        return -1;
    }

    int res = pymain_read_conf(pymain, config, cmdline);
    if (res < 0) {
        return -1;
    }
    if (res > 0) {
        /* --help or --version command: we are done */
        return 1;
    }

    if (cmdline->print_help) {
        pymain_usage(0, config->program);
        return 1;
    }

    if (cmdline->print_version) {
        printf("Python %s\n",
               (cmdline->print_version >= 2) ? Py_GetVersion() : PY_VERSION);
        return 1;
    }

    /* For Py_GetArgcArgv(). Cleared by pymain_free(). */
    orig_argv = copy_wstrlist(pymain->argc, cmdline->argv);
    if (orig_argv == NULL) {
        pymain->err = _Py_INIT_NO_MEMORY();
        return -1;
    }
    orig_argc = pymain->argc;

    _PyInitError err = config_init_warnoptions(config, cmdline);
    if (_Py_INIT_FAILED(err)) {
        pymain->err = err;
        return -1;
    }
    return 0;
}


/* Read the configuration into _PyCoreConfig and _PyMain, initialize the
   LC_CTYPE locale and Py_DecodeLocale().

   Configuration:

   * Command line arguments
   * Environment variables
   * Py_xxx global configuration variables

   _PyCmdline is a temporary structure used to prioritize these
   variables. */
static int
pymain_cmdline(_PyMain *pymain, _PyCoreConfig *config)
{
    /* Force default allocator, since pymain_free() and pymain_clear_config()
       must use the same allocator than this function. */
    PyMemAllocatorEx old_alloc;
    _PyMem_SetDefaultAllocator(PYMEM_DOMAIN_RAW, &old_alloc);
#ifdef Py_DEBUG
    PyMemAllocatorEx default_alloc;
    PyMem_GetAllocator(PYMEM_DOMAIN_RAW, &default_alloc);
#endif

    _PyCmdline cmdline;
    memset(&cmdline, 0, sizeof(cmdline));

    cmdline_get_global_config(&cmdline);

    int res = pymain_cmdline_impl(pymain, config, &cmdline);

    cmdline_set_global_config(&cmdline);
    _PyCoreConfig_SetGlobalConfig(config);
    if (Py_IsolatedFlag) {
        Py_IgnoreEnvironmentFlag = 1;
        Py_NoUserSiteDirectory = 1;
    }

    pymain_clear_cmdline(pymain, &cmdline);

#ifdef Py_DEBUG
    /* Make sure that PYMEM_DOMAIN_RAW has not been modified */
    PyMemAllocatorEx cur_alloc;
    PyMem_GetAllocator(PYMEM_DOMAIN_RAW, &cur_alloc);
    assert(memcmp(&cur_alloc, &default_alloc, sizeof(cur_alloc)) == 0);
#endif
    PyMem_SetAllocator(PYMEM_DOMAIN_RAW, &old_alloc);
    return res;
}


static int
pymain_init(_PyMain *pymain)
{
    _PyCoreConfig local_config = _PyCoreConfig_INIT;
    _PyCoreConfig *config = &local_config;

    /* 754 requires that FP exceptions run in "no stop" mode by default,
     * and until C vendors implement C99's ways to control FP exceptions,
     * Python requires non-stop mode.  Alas, some platforms enable FP
     * exceptions by default.  Here we disable them.
     */
#ifdef __FreeBSD__
    fedisableexcept(FE_OVERFLOW);
#endif

    config->_disable_importlib = 0;
    config->install_signal_handlers = 1;
    _PyCoreConfig_GetGlobalConfig(config);

    int res = pymain_cmdline(pymain, config);
    if (res < 0) {
        _Py_FatalInitError(pymain->err);
    }
    if (res == 1) {
        pymain_clear_config(&local_config);
        return res;
    }

    pymain_init_stdio(pymain);

    PyInterpreterState *interp;
    pymain->err = _Py_InitializeCore(&interp, config);
    if (_Py_INIT_FAILED(pymain->err)) {
        _Py_FatalInitError(pymain->err);
    }

    pymain_clear_config(&local_config);
    config = &interp->core_config;

    if (pymain_init_python_main(pymain, interp) < 0) {
        _Py_FatalInitError(pymain->err);
    }

    if (pymain_init_sys_path(pymain, config) < 0) {
        _Py_FatalInitError(pymain->err);
    }
    return 0;
}


static int
pymain_main(_PyMain *pymain)
{
    int res = pymain_init(pymain);
    if (res == 1) {
        goto done;
    }

    pymain_run_python(pymain);

    if (Py_FinalizeEx() < 0) {
        /* Value unlikely to be confused with a non-error exit status or
           other special meaning */
        pymain->status = 120;
    }

done:
    pymain_free(pymain);

    return pymain->status;
}


int
Py_Main(int argc, wchar_t **argv)
{
    _PyMain pymain = _PyMain_INIT;
    pymain.use_bytes_argv = 0;
    pymain.argc = argc;
    pymain.wchar_argv = argv;

    return pymain_main(&pymain);
}


int
_Py_UnixMain(int argc, char **argv)
{
    _PyMain pymain = _PyMain_INIT;
    pymain.use_bytes_argv = 1;
    pymain.argc = argc;
    pymain.bytes_argv = argv;

    return pymain_main(&pymain);
}


/* this is gonna seem *real weird*, but if you put some other code between
   Py_Main() and Py_GetArgcArgv() you will need to adjust the test in the
   while statement in Misc/gdbinit:ppystack */

/* Make the *original* argc/argv available to other modules.
   This is rare, but it is needed by the secureware extension. */

void
Py_GetArgcArgv(int *argc, wchar_t ***argv)
{
    *argc = orig_argc;
    *argv = orig_argv;
}

void
Py_InitArgcArgv(int argc, wchar_t **argv)
{
    if (!orig_argc) {
        orig_argc = argc;
        orig_argv = argv;
    }
}

#ifdef __cplusplus
}
#endif
