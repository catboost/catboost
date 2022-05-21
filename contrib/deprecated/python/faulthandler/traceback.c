#ifdef __gnu_linux__
#  include <sys/prctl.h>
#endif

#include "Python.h"
#include <frameobject.h>

#if PY_MAJOR_VERSION >= 3
#  define PYSTRING_CHECK PyUnicode_Check
#else
#  define PYSTRING_CHECK PyString_Check
#endif

#define PUTS(fd, str) _Py_write_noraise(fd, str, (int)strlen(str))
#define MAX_STRING_LENGTH 500
#define MAX_FRAME_DEPTH 100
#define MAX_NTHREADS 100

/* Write count bytes of buf into fd.
 *
 * On success, return the number of written bytes, it can be lower than count
 * including 0. On error, set errno and return -1.
 *
 * When interrupted by a signal (write() fails with EINTR), retry the syscall
 * without calling the Python signal handler. */
Py_ssize_t
_Py_write_noraise(int fd, const char *buf, size_t count)
{
    Py_ssize_t res;

    do {
#ifdef MS_WINDOWS
        assert(count < INT_MAX);
        res = write(fd, buf, (int)count);
#else
        res = write(fd, buf, count);
#endif
    /* retry write() if it was interrupted by a signal */
    } while (res < 0 && errno == EINTR);

    return res;
}

/* Reverse a string. For example, "abcd" becomes "dcba".

   This function is signal safe. */

void
reverse_string(char *text, const size_t len)
{
    char tmp;
    size_t i, j;
    if (len == 0)
        return;
    for (i=0, j=len-1; i < j; i++, j--) {
        tmp = text[i];
        text[i] = text[j];
        text[j] = tmp;
    }
}

/* Format an integer in range [0; 999999999] to decimal,
   and write it into the file fd.

   This function is signal safe. */

void
dump_decimal(int fd, int value)
{
    char buffer[10];
    int len;
    if (value < 0 || 999999999 < value)
        return;
    len = 0;
    do {
        buffer[len] = '0' + (value % 10);
        value /= 10;
        len++;
    } while (value);
    reverse_string(buffer, len);
    _Py_write_noraise(fd, buffer, len);
}

/* Format an integer in range [0; 0xffffffff] to hexadecimal of 'width' digits,
   and write it into the file fd.

   This function is signal safe. */

void
_Py_dump_hexadecimal(int fd, unsigned long value, size_t bytes)
{
    const char *hexdigits = "0123456789abcdef";
    size_t width = bytes * 2;
    size_t len;
    char buffer[sizeof(unsigned long) * 2 + 1];
    len = 0;
    do {
        buffer[len] = hexdigits[value & 15];
        value >>= 4;
        len++;
    } while (len < width || value);
    reverse_string(buffer, len);
    _Py_write_noraise(fd, buffer, len);
}

/* Write an unicode object into the file fd using ascii+backslashreplace.

   This function is signal safe. */

static void
dump_ascii(int fd, PyObject *text)
{
    Py_ssize_t i, size;
    int truncated;
    unsigned long ch;
#if PY_MAJOR_VERSION >= 3
    Py_UNICODE *u;

    size = PyUnicode_GET_SIZE(text);
    u = PyUnicode_AS_UNICODE(text);
#else
    char *s;

    size = PyString_GET_SIZE(text);
    s = PyString_AS_STRING(text);
#endif

    if (MAX_STRING_LENGTH < size) {
        size = MAX_STRING_LENGTH;
        truncated = 1;
    }
    else
        truncated = 0;

#if PY_MAJOR_VERSION >= 3
    for (i=0; i < size; i++, u++) {
        ch = *u;
        if (' ' <= ch && ch < 0x7f) {
            /* printable ASCII character */
            char c = (char)ch;
            _Py_write_noraise(fd, &c, 1);
        }
        else if (ch <= 0xff) {
            PUTS(fd, "\\x");
            _Py_dump_hexadecimal(fd, ch, 1);
        }
        else
#ifdef Py_UNICODE_WIDE
        if (ch <= 0xffff)
#endif
        {
            PUTS(fd, "\\u");
            _Py_dump_hexadecimal(fd, ch, 2);
#ifdef Py_UNICODE_WIDE
        }
        else {
            PUTS(fd, "\\U");
            _Py_dump_hexadecimal(fd, ch, 4);
#endif
        }
    }
#else
    for (i=0; i < size; i++, s++) {
        ch = *s;
        if (' ' <= ch && ch <= 126) {
            /* printable ASCII character */
            _Py_write_noraise(fd, s, 1);
        }
        else {
            PUTS(fd, "\\x");
            _Py_dump_hexadecimal(fd, ch, 1);
        }
    }
#endif
    if (truncated)
        PUTS(fd, "...");
}

/* Write a frame into the file fd: "File "xxx", line xxx in xxx".

   This function is signal safe. */

static void
dump_frame(int fd, PyFrameObject *frame)
{
    PyCodeObject *code;
    int lineno;

    code = frame->f_code;
    PUTS(fd, "  File ");
    if (code != NULL && code->co_filename != NULL
        && PYSTRING_CHECK(code->co_filename))
    {
        PUTS(fd, "\"");
        dump_ascii(fd, code->co_filename);
        PUTS(fd, "\"");
    } else {
        PUTS(fd, "???");
    }

#if (PY_MAJOR_VERSION <= 2 && PY_MINOR_VERSION < 7) \
||  (PY_MAJOR_VERSION >= 3 && PY_MINOR_VERSION < 2)
    /* PyFrame_GetLineNumber() was introduced in Python 2.7.0 and 3.2.0 */
    lineno = PyCode_Addr2Line(code, frame->f_lasti);
#else
    lineno = PyFrame_GetLineNumber(frame);
#endif
    PUTS(fd, ", line ");
    dump_decimal(fd, lineno);
    PUTS(fd, " in ");

    if (code != NULL && code->co_name != NULL
        && PYSTRING_CHECK(code->co_name))
        dump_ascii(fd, code->co_name);
    else
        PUTS(fd, "???");

    PUTS(fd, "\n");
}

static void
dump_traceback(int fd, PyThreadState *tstate, int write_header)
{
    PyFrameObject *frame;
    unsigned int depth;

    if (write_header)
        PUTS(fd, "Stack (most recent call first):\n");

    frame = _PyThreadState_GetFrame(tstate);
    if (frame == NULL)
        return;

    depth = 0;
    while (frame != NULL) {
        if (MAX_FRAME_DEPTH <= depth) {
            PUTS(fd, "  ...\n");
            break;
        }
        if (!PyFrame_Check(frame))
            break;
        dump_frame(fd, frame);
        frame = frame->f_back;
        depth++;
    }
}

/* Dump the traceback of a Python thread into fd. Use write() to write the
   traceback and retry if write() is interrupted by a signal (failed with
   EINTR), but don't call the Python signal handler.

   The caller is responsible to call PyErr_CheckSignals() to call Python signal
   handlers if signals were received. */
void
_Py_DumpTraceback(int fd, PyThreadState *tstate)
{
    dump_traceback(fd, tstate, 1);
}

/* Write the thread identifier into the file 'fd': "Current thread 0xHHHH:\" if
   is_current is true, "Thread 0xHHHH:\n" otherwise.

   This function is signal safe. */

static void
write_thread_id(int fd, PyThreadState *tstate, int is_current)
{
    if (is_current)
        PUTS(fd, "Current thread 0x");
    else
        PUTS(fd, "Thread 0x");
    _Py_dump_hexadecimal(fd, (unsigned long)tstate->thread_id, sizeof(unsigned long));

#ifdef __gnu_linux__
    /* Linux only, get and print thread name */
    static char thread_name[16];
    if (0 == prctl(PR_GET_NAME, (unsigned long) thread_name, 0, 0, 0)) {
        if (0 != strlen(thread_name)) {
            PUTS(fd, " <");
            PUTS(fd, thread_name);
            PUTS(fd, ">");
        }
    }
#endif

    PUTS(fd, " (most recent call first):\n");
}

/* Dump the traceback of all Python threads into fd. Use write() to write the
   traceback and retry if write() is interrupted by a signal (failed with
   EINTR), but don't call the Python signal handler.

   The caller is responsible to call PyErr_CheckSignals() to call Python signal
   handlers if signals were received. */
const char*
_Py_DumpTracebackThreads(int fd, PyInterpreterState *interp,
                         PyThreadState *current_thread)
{
    PyThreadState *tstate;
    unsigned int nthreads;

    /* Get the current interpreter from the current thread */
    tstate = PyInterpreterState_ThreadHead(interp);
    if (tstate == NULL)
        return "unable to get the thread head state";

    /* Dump the traceback of each thread */
    tstate = PyInterpreterState_ThreadHead(interp);
    nthreads = 0;
    do
    {
        if (nthreads != 0)
            PUTS(fd, "\n");
        if (nthreads >= MAX_NTHREADS) {
            PUTS(fd, "...\n");
            break;
        }
        write_thread_id(fd, tstate, tstate == current_thread);
        dump_traceback(fd, tstate, 0);
        tstate = PyThreadState_Next(tstate);
        nthreads++;
    } while (tstate != NULL);

    return NULL;
}

