/* C speedups for scandir module

This is divided into four sections (each prefixed with a "SECTION:"
comment):

1) Python 2/3 compatibility
2) Helper utilities from posixmodule.c, fileutils.h, etc
3) SECTION: Main DirEntry and scandir implementation, taken from
   Python 3.5's posixmodule.c
4) Module and method definitions and initialization code

*/

#include <Python.h>
#include <structseq.h>
#include <structmember.h>
#include "osdefs.h"

#ifdef MS_WINDOWS
#include <windows.h>
#include <winioctl.h>
#include "winreparse.h"
#else
#include <dirent.h>
#ifndef HAVE_DIRENT_H
#define HAVE_DIRENT_H 1
#endif
#endif

#define MODNAME "scandir"


/* SECTION: Python 2/3 compatibility */

#if PY_MAJOR_VERSION >= 3
#define INIT_ERROR return NULL
#else
#define INIT_ERROR return
// Because on PyPy, Py_FileSystemDefaultEncoding is (was) defined to be NULL
// (see PyPy Bitbucket issue #2669)
#define FS_ENCODING (Py_FileSystemDefaultEncoding ? Py_FileSystemDefaultEncoding : "UTF-8")
#endif

#if PY_MAJOR_VERSION < 3 || PY_MAJOR_VERSION == 3 && PY_MINOR_VERSION <= 2
#define _Py_IDENTIFIER(name) static char * PyId_##name = #name;
#define _PyObject_GetAttrId(obj, pyid_name) PyObject_GetAttrString((obj), *(pyid_name))
#define PyExc_FileNotFoundError PyExc_OSError
#define PyUnicode_AsUnicodeAndSize(unicode, addr_length) \
    PyUnicode_AsUnicode(unicode); *(addr_length) = PyUnicode_GetSize(unicode)
#endif

// Because on PyPy not working without
#if PY_MAJOR_VERSION == 3 && PY_MINOR_VERSION > 2 && defined(PYPY_VERSION_NUM)
#define _Py_IDENTIFIER(name) static char * PyId_##name = #name;
#define _PyObject_GetAttrId(obj, pyid_name) PyObject_GetAttrString((obj), *(pyid_name))
#endif

/* SECTION: Helper utilities from posixmodule.c, fileutils.h, etc */

#if !defined(MS_WINDOWS) && defined(DT_UNKNOWN)
#define HAVE_DIRENT_D_TYPE 1
#endif

#ifdef HAVE_DIRENT_H
#include <dirent.h>
#define NAMLEN(dirent) strlen((dirent)->d_name)
#else
#if defined(__WATCOMC__) && !defined(__QNX__)
#include <direct.h>
#define NAMLEN(dirent) strlen((dirent)->d_name)
#else
#define dirent direct
#define NAMLEN(dirent) (dirent)->d_namlen
#endif
#ifdef HAVE_SYS_NDIR_H
#include <sys/ndir.h>
#endif
#ifdef HAVE_SYS_DIR_H
#include <sys/dir.h>
#endif
#ifdef HAVE_NDIR_H
#include <ndir.h>
#endif
#endif

#ifndef Py_CLEANUP_SUPPORTED
#define Py_CLEANUP_SUPPORTED 0x20000
#endif

#ifndef S_IFLNK
/* Windows doesn't define S_IFLNK but posixmodule.c maps
 * IO_REPARSE_TAG_SYMLINK to S_IFLNK */
#  define S_IFLNK 0120000
#endif

// _Py_stat_struct is already defined in fileutils.h on Python 3.5+
// But not in PyPy
#if PY_MAJOR_VERSION < 3 || (PY_MAJOR_VERSION == 3 && PY_MINOR_VERSION < 5) || defined(PYPY_VERSION_NUM)
#ifdef MS_WINDOWS
struct _Py_stat_struct {
    unsigned long st_dev;
    unsigned __int64 st_ino;
    unsigned short st_mode;
    int st_nlink;
    int st_uid;
    int st_gid;
    unsigned long st_rdev;
    __int64 st_size;
    time_t st_atime;
    int st_atime_nsec;
    time_t st_mtime;
    int st_mtime_nsec;
    time_t st_ctime;
    int st_ctime_nsec;
    unsigned long st_file_attributes;
};
#else
#  define _Py_stat_struct stat
#endif
#endif

/* choose the appropriate stat and fstat functions and return structs */
#undef STAT
#undef FSTAT
#undef STRUCT_STAT
#ifdef MS_WINDOWS
#       define STAT win32_stat
#       define LSTAT win32_lstat
#       define FSTAT _Py_fstat_noraise
#       define STRUCT_STAT struct _Py_stat_struct
#else
#       define STAT stat
#       define LSTAT lstat
#       define FSTAT fstat
#       define STRUCT_STAT struct stat
#endif

#ifdef MS_WINDOWS

static __int64 secs_between_epochs = 11644473600; /* Seconds between 1.1.1601 and 1.1.1970 */

static void
FILE_TIME_to_time_t_nsec(FILETIME *in_ptr, time_t *time_out, int* nsec_out)
{
    /* XXX endianness. Shouldn't matter, as all Windows implementations are little-endian */
    /* Cannot simply cast and dereference in_ptr,
       since it might not be aligned properly */
    __int64 in;
    memcpy(&in, in_ptr, sizeof(in));
    *nsec_out = (int)(in % 10000000) * 100; /* FILETIME is in units of 100 nsec. */
    *time_out = Py_SAFE_DOWNCAST((in / 10000000) - secs_between_epochs, __int64, time_t);
}

/* Below, we *know* that ugo+r is 0444 */
#if _S_IREAD != 0400
#error Unsupported C library
#endif
static int
attributes_to_mode(DWORD attr)
{
    int m = 0;
    if (attr & FILE_ATTRIBUTE_DIRECTORY)
        m |= _S_IFDIR | 0111; /* IFEXEC for user,group,other */
    else
        m |= _S_IFREG;
    if (attr & FILE_ATTRIBUTE_READONLY)
        m |= 0444;
    else
        m |= 0666;
    return m;
}

void
_Py_attribute_data_to_stat(BY_HANDLE_FILE_INFORMATION *info, ULONG reparse_tag,
                           struct _Py_stat_struct *result)
{
    memset(result, 0, sizeof(*result));
    result->st_mode = attributes_to_mode(info->dwFileAttributes);
    result->st_size = (((__int64)info->nFileSizeHigh)<<32) + info->nFileSizeLow;
    result->st_dev = info->dwVolumeSerialNumber;
    result->st_rdev = result->st_dev;
    FILE_TIME_to_time_t_nsec(&info->ftCreationTime, &result->st_ctime, &result->st_ctime_nsec);
    FILE_TIME_to_time_t_nsec(&info->ftLastWriteTime, &result->st_mtime, &result->st_mtime_nsec);
    FILE_TIME_to_time_t_nsec(&info->ftLastAccessTime, &result->st_atime, &result->st_atime_nsec);
    result->st_nlink = info->nNumberOfLinks;
    result->st_ino = (((unsigned __int64)info->nFileIndexHigh)<<32) + info->nFileIndexLow;
    if (reparse_tag == IO_REPARSE_TAG_SYMLINK) {
        /* first clear the S_IFMT bits */
        result->st_mode ^= (result->st_mode & S_IFMT);
        /* now set the bits that make this a symlink */
        result->st_mode |= S_IFLNK;
    }
    result->st_file_attributes = info->dwFileAttributes;
}

static BOOL
get_target_path(HANDLE hdl, wchar_t **target_path)
{
    int buf_size, result_length;
    wchar_t *buf;

    /* We have a good handle to the target, use it to determine
       the target path name (then we'll call lstat on it). */
    buf_size = GetFinalPathNameByHandleW(hdl, 0, 0,
                                         VOLUME_NAME_DOS);
    if(!buf_size)
        return FALSE;

    buf = PyMem_New(wchar_t, buf_size+1);
    if (!buf) {
        SetLastError(ERROR_OUTOFMEMORY);
        return FALSE;
    }

    result_length = GetFinalPathNameByHandleW(hdl,
                       buf, buf_size, VOLUME_NAME_DOS);

    if(!result_length) {
        PyMem_Free(buf);
        return FALSE;
    }

    if(!CloseHandle(hdl)) {
        PyMem_Free(buf);
        return FALSE;
    }

    buf[result_length] = 0;

    *target_path = buf;
    return TRUE;
}

static int
win32_get_reparse_tag(HANDLE reparse_point_handle, ULONG *reparse_tag)
{
    char target_buffer[MAXIMUM_REPARSE_DATA_BUFFER_SIZE];
    REPARSE_DATA_BUFFER *rdb = (REPARSE_DATA_BUFFER *)target_buffer;
    DWORD n_bytes_returned;

    if (0 == DeviceIoControl(
        reparse_point_handle,
        FSCTL_GET_REPARSE_POINT,
        NULL, 0, /* in buffer */
        target_buffer, sizeof(target_buffer),
        &n_bytes_returned,
        NULL)) /* we're not using OVERLAPPED_IO */
        return FALSE;

    if (reparse_tag)
        *reparse_tag = rdb->ReparseTag;

    return TRUE;
}

static void
find_data_to_file_info_w(WIN32_FIND_DATAW *pFileData,
                         BY_HANDLE_FILE_INFORMATION *info,
                         ULONG *reparse_tag)
{
    memset(info, 0, sizeof(*info));
    info->dwFileAttributes = pFileData->dwFileAttributes;
    info->ftCreationTime   = pFileData->ftCreationTime;
    info->ftLastAccessTime = pFileData->ftLastAccessTime;
    info->ftLastWriteTime  = pFileData->ftLastWriteTime;
    info->nFileSizeHigh    = pFileData->nFileSizeHigh;
    info->nFileSizeLow     = pFileData->nFileSizeLow;
/*  info->nNumberOfLinks   = 1; */
    if (pFileData->dwFileAttributes & FILE_ATTRIBUTE_REPARSE_POINT)
        *reparse_tag = pFileData->dwReserved0;
    else
        *reparse_tag = 0;
}

static BOOL
attributes_from_dir_w(LPCWSTR pszFile, BY_HANDLE_FILE_INFORMATION *info, ULONG *reparse_tag)
{
    HANDLE hFindFile;
    WIN32_FIND_DATAW FileData;
    hFindFile = FindFirstFileW(pszFile, &FileData);
    if (hFindFile == INVALID_HANDLE_VALUE)
        return FALSE;
    FindClose(hFindFile);
    find_data_to_file_info_w(&FileData, info, reparse_tag);
    return TRUE;
}

static int
win32_xstat_impl_w(const wchar_t *path, struct _Py_stat_struct *result,
                   BOOL traverse)
{
    int code;
    HANDLE hFile, hFile2;
    BY_HANDLE_FILE_INFORMATION info;
    ULONG reparse_tag = 0;
    wchar_t *target_path;
    const wchar_t *dot;

    hFile = CreateFileW(
        path,
        FILE_READ_ATTRIBUTES, /* desired access */
        0, /* share mode */
        NULL, /* security attributes */
        OPEN_EXISTING,
        /* FILE_FLAG_BACKUP_SEMANTICS is required to open a directory */
        /* FILE_FLAG_OPEN_REPARSE_POINT does not follow the symlink.
           Because of this, calls like GetFinalPathNameByHandle will return
           the symlink path again and not the actual final path. */
        FILE_ATTRIBUTE_NORMAL|FILE_FLAG_BACKUP_SEMANTICS|
            FILE_FLAG_OPEN_REPARSE_POINT,
        NULL);

    if (hFile == INVALID_HANDLE_VALUE) {
        /* Either the target doesn't exist, or we don't have access to
           get a handle to it. If the former, we need to return an error.
           If the latter, we can use attributes_from_dir. */
        if (GetLastError() != ERROR_SHARING_VIOLATION)
            return -1;
        /* Could not get attributes on open file. Fall back to
           reading the directory. */
        if (!attributes_from_dir_w(path, &info, &reparse_tag))
            /* Very strange. This should not fail now */
            return -1;
        if (info.dwFileAttributes & FILE_ATTRIBUTE_REPARSE_POINT) {
            if (traverse) {
                /* Should traverse, but could not open reparse point handle */
                SetLastError(ERROR_SHARING_VIOLATION);
                return -1;
            }
        }
    } else {
        if (!GetFileInformationByHandle(hFile, &info)) {
            CloseHandle(hFile);
            return -1;
        }
        if (info.dwFileAttributes & FILE_ATTRIBUTE_REPARSE_POINT) {
            if (!win32_get_reparse_tag(hFile, &reparse_tag))
                return -1;

            /* Close the outer open file handle now that we're about to
               reopen it with different flags. */
            if (!CloseHandle(hFile))
                return -1;

            if (traverse) {
                /* In order to call GetFinalPathNameByHandle we need to open
                   the file without the reparse handling flag set. */
                hFile2 = CreateFileW(
                           path, FILE_READ_ATTRIBUTES, FILE_SHARE_READ,
                           NULL, OPEN_EXISTING,
                           FILE_ATTRIBUTE_NORMAL|FILE_FLAG_BACKUP_SEMANTICS,
                           NULL);
                if (hFile2 == INVALID_HANDLE_VALUE)
                    return -1;

                if (!get_target_path(hFile2, &target_path))
                    return -1;

                code = win32_xstat_impl_w(target_path, result, FALSE);
                PyMem_Free(target_path);
                return code;
            }
        } else
            CloseHandle(hFile);
    }
    _Py_attribute_data_to_stat(&info, reparse_tag, result);

    /* Set S_IEXEC if it is an .exe, .bat, ... */
    dot = wcsrchr(path, '.');
    if (dot) {
        if (_wcsicmp(dot, L".bat") == 0 || _wcsicmp(dot, L".cmd") == 0 ||
            _wcsicmp(dot, L".exe") == 0 || _wcsicmp(dot, L".com") == 0)
            result->st_mode |= 0111;
    }
    return 0;
}

static int
win32_xstat_w(const wchar_t *path, struct _Py_stat_struct *result, BOOL traverse)
{
    /* Protocol violation: we explicitly clear errno, instead of
       setting it to a POSIX error. Callers should use GetLastError. */
    int code = win32_xstat_impl_w(path, result, traverse);
    errno = 0;
    return code;
}

static int
win32_lstat_w(const wchar_t* path, struct _Py_stat_struct *result)
{
    return win32_xstat_w(path, result, FALSE);
}

static int
win32_stat_w(const wchar_t* path, struct _Py_stat_struct *result)
{
    return win32_xstat_w(path, result, TRUE);
}

#endif /* MS_WINDOWS */

static PyTypeObject StatResultType;

static PyObject *billion = NULL;

static newfunc structseq_new;

static PyObject *
statresult_new(PyTypeObject *type, PyObject *args, PyObject *kwds)
{
    PyStructSequence *result;
    int i;

    result = (PyStructSequence*)structseq_new(type, args, kwds);
    if (!result)
        return NULL;
    /* If we have been initialized from a tuple,
       st_?time might be set to None. Initialize it
       from the int slots.  */
    for (i = 7; i <= 9; i++) {
        if (result->ob_item[i+3] == Py_None) {
            Py_DECREF(Py_None);
            Py_INCREF(result->ob_item[i]);
            result->ob_item[i+3] = result->ob_item[i];
        }
    }
    return (PyObject*)result;
}

/* If true, st_?time is float. */
static int _stat_float_times = 1;

static void
fill_time(PyObject *v, int index, time_t sec, unsigned long nsec)
{
#if SIZEOF_TIME_T > SIZEOF_LONG
    PyObject *s = PyLong_FromLongLong((PY_LONG_LONG)sec);
#else
#if PY_MAJOR_VERSION >= 3
    PyObject *s = PyLong_FromLong((long)sec);
#else
    PyObject *s = PyInt_FromLong((long)sec);
#endif
#endif
    PyObject *ns_fractional = PyLong_FromUnsignedLong(nsec);
    PyObject *s_in_ns = NULL;
    PyObject *ns_total = NULL;
    PyObject *float_s = NULL;

    if (!(s && ns_fractional))
        goto exit;

    s_in_ns = PyNumber_Multiply(s, billion);
    if (!s_in_ns)
        goto exit;

    ns_total = PyNumber_Add(s_in_ns, ns_fractional);
    if (!ns_total)
        goto exit;

    if (_stat_float_times) {
        float_s = PyFloat_FromDouble(sec + 1e-9*nsec);
        if (!float_s)
            goto exit;
    }
    else {
        float_s = s;
        Py_INCREF(float_s);
    }

    PyStructSequence_SET_ITEM(v, index, s);
    PyStructSequence_SET_ITEM(v, index+3, float_s);
    PyStructSequence_SET_ITEM(v, index+6, ns_total);
    s = NULL;
    float_s = NULL;
    ns_total = NULL;
exit:
    Py_XDECREF(s);
    Py_XDECREF(ns_fractional);
    Py_XDECREF(s_in_ns);
    Py_XDECREF(ns_total);
    Py_XDECREF(float_s);
}

#ifdef MS_WINDOWS
#define HAVE_STAT_NSEC 1
#define HAVE_STRUCT_STAT_ST_FILE_ATTRIBUTES 1
#endif

#ifdef HAVE_STRUCT_STAT_ST_BLKSIZE
#define ST_BLKSIZE_IDX 16
#else
#define ST_BLKSIZE_IDX 15
#endif

#ifdef HAVE_STRUCT_STAT_ST_BLOCKS
#define ST_BLOCKS_IDX (ST_BLKSIZE_IDX+1)
#else
#define ST_BLOCKS_IDX ST_BLKSIZE_IDX
#endif

#ifdef HAVE_STRUCT_STAT_ST_RDEV
#define ST_RDEV_IDX (ST_BLOCKS_IDX+1)
#else
#define ST_RDEV_IDX ST_BLOCKS_IDX
#endif

#ifdef HAVE_STRUCT_STAT_ST_FLAGS
#define ST_FLAGS_IDX (ST_RDEV_IDX+1)
#else
#define ST_FLAGS_IDX ST_RDEV_IDX
#endif

#ifdef HAVE_STRUCT_STAT_ST_GEN
#define ST_GEN_IDX (ST_FLAGS_IDX+1)
#else
#define ST_GEN_IDX ST_FLAGS_IDX
#endif

#ifdef HAVE_STRUCT_STAT_ST_BIRTHTIME
#define ST_BIRTHTIME_IDX (ST_GEN_IDX+1)
#else
#define ST_BIRTHTIME_IDX ST_GEN_IDX
#endif

#ifdef HAVE_STRUCT_STAT_ST_FILE_ATTRIBUTES
#define ST_FILE_ATTRIBUTES_IDX (ST_BIRTHTIME_IDX+1)
#else
#define ST_FILE_ATTRIBUTES_IDX ST_BIRTHTIME_IDX
#endif

#ifdef HAVE_LONG_LONG
#  define _PyLong_FromDev PyLong_FromLongLong
#else
#  define _PyLong_FromDev PyLong_FromLong
#endif

#ifndef MS_WINDOWS
PyObject *
_PyLong_FromUid(uid_t uid)
{
    if (uid == (uid_t)-1)
        return PyLong_FromLong(-1);
    return PyLong_FromUnsignedLong(uid);
}

PyObject *
_PyLong_FromGid(gid_t gid)
{
    if (gid == (gid_t)-1)
        return PyLong_FromLong(-1);
    return PyLong_FromUnsignedLong(gid);
}
#endif

/* pack a system stat C structure into the Python stat tuple
   (used by posix_stat() and posix_fstat()) */
static PyObject*
_pystat_fromstructstat(STRUCT_STAT *st)
{
    unsigned long ansec, mnsec, cnsec;
    PyObject *v = PyStructSequence_New(&StatResultType);
    if (v == NULL)
        return NULL;

    PyStructSequence_SET_ITEM(v, 0, PyLong_FromLong((long)st->st_mode));
#ifdef HAVE_LARGEFILE_SUPPORT
    PyStructSequence_SET_ITEM(v, 1,
                              PyLong_FromUnsignedLongLong(st->st_ino));
#else
    PyStructSequence_SET_ITEM(v, 1, PyLong_FromUnsignedLong((unsigned long)st->st_ino));
#endif
#ifdef MS_WINDOWS
    PyStructSequence_SET_ITEM(v, 2, PyLong_FromUnsignedLong(st->st_dev));
#else
    PyStructSequence_SET_ITEM(v, 2, _PyLong_FromDev(st->st_dev));
#endif
    PyStructSequence_SET_ITEM(v, 3, PyLong_FromLong((long)st->st_nlink));
#if defined(MS_WINDOWS)
    PyStructSequence_SET_ITEM(v, 4, PyLong_FromLong(0));
    PyStructSequence_SET_ITEM(v, 5, PyLong_FromLong(0));
#else
    PyStructSequence_SET_ITEM(v, 4, _PyLong_FromUid(st->st_uid));
    PyStructSequence_SET_ITEM(v, 5, _PyLong_FromGid(st->st_gid));
#endif
#ifdef HAVE_LARGEFILE_SUPPORT
    PyStructSequence_SET_ITEM(v, 6,
                              PyLong_FromLongLong((PY_LONG_LONG)st->st_size));
#else
    PyStructSequence_SET_ITEM(v, 6, PyLong_FromLong(st->st_size));
#endif

#if defined(HAVE_STAT_TV_NSEC)
    ansec = st->st_atim.tv_nsec;
    mnsec = st->st_mtim.tv_nsec;
    cnsec = st->st_ctim.tv_nsec;
#elif defined(HAVE_STAT_TV_NSEC2)
    ansec = st->st_atimespec.tv_nsec;
    mnsec = st->st_mtimespec.tv_nsec;
    cnsec = st->st_ctimespec.tv_nsec;
#elif defined(HAVE_STAT_NSEC)
    ansec = st->st_atime_nsec;
    mnsec = st->st_mtime_nsec;
    cnsec = st->st_ctime_nsec;
#else
    ansec = mnsec = cnsec = 0;
#endif
    fill_time(v, 7, st->st_atime, ansec);
    fill_time(v, 8, st->st_mtime, mnsec);
    fill_time(v, 9, st->st_ctime, cnsec);

#ifdef HAVE_STRUCT_STAT_ST_BLKSIZE
    PyStructSequence_SET_ITEM(v, ST_BLKSIZE_IDX,
                              PyLong_FromLong((long)st->st_blksize));
#endif
#ifdef HAVE_STRUCT_STAT_ST_BLOCKS
    PyStructSequence_SET_ITEM(v, ST_BLOCKS_IDX,
                              PyLong_FromLong((long)st->st_blocks));
#endif
#ifdef HAVE_STRUCT_STAT_ST_RDEV
    PyStructSequence_SET_ITEM(v, ST_RDEV_IDX,
                              PyLong_FromLong((long)st->st_rdev));
#endif
#ifdef HAVE_STRUCT_STAT_ST_GEN
    PyStructSequence_SET_ITEM(v, ST_GEN_IDX,
                              PyLong_FromLong((long)st->st_gen));
#endif
#ifdef HAVE_STRUCT_STAT_ST_BIRTHTIME
    {
      PyObject *val;
      unsigned long bsec,bnsec;
      bsec = (long)st->st_birthtime;
#ifdef HAVE_STAT_TV_NSEC2
      bnsec = st->st_birthtimespec.tv_nsec;
#else
      bnsec = 0;
#endif
      if (_stat_float_times) {
        val = PyFloat_FromDouble(bsec + 1e-9*bnsec);
      } else {
        val = PyLong_FromLong((long)bsec);
      }
      PyStructSequence_SET_ITEM(v, ST_BIRTHTIME_IDX,
                                val);
    }
#endif
#ifdef HAVE_STRUCT_STAT_ST_FLAGS
    PyStructSequence_SET_ITEM(v, ST_FLAGS_IDX,
                              PyLong_FromLong((long)st->st_flags));
#endif
#ifdef HAVE_STRUCT_STAT_ST_FILE_ATTRIBUTES
    PyStructSequence_SET_ITEM(v, ST_FILE_ATTRIBUTES_IDX,
                              PyLong_FromUnsignedLong(st->st_file_attributes));
#endif

    if (PyErr_Occurred()) {
        Py_DECREF(v);
        return NULL;
    }

    return v;
}

//char *PyStructSequence_UnnamedField = "unnamed field";

PyDoc_STRVAR(stat_result__doc__,
"stat_result: Result from stat, fstat, or lstat.\n\n\
This object may be accessed either as a tuple of\n\
  (mode, ino, dev, nlink, uid, gid, size, atime, mtime, ctime)\n\
or via the attributes st_mode, st_ino, st_dev, st_nlink, st_uid, and so on.\n\
\n\
Posix/windows: If your platform supports st_blksize, st_blocks, st_rdev,\n\
or st_flags, they are available as attributes only.\n\
\n\
See os.stat for more information.");

static PyStructSequence_Field stat_result_fields[] = {
    {"st_mode",    "protection bits"},
    {"st_ino",     "inode"},
    {"st_dev",     "device"},
    {"st_nlink",   "number of hard links"},
    {"st_uid",     "user ID of owner"},
    {"st_gid",     "group ID of owner"},
    {"st_size",    "total size, in bytes"},
    /* The NULL is replaced with PyStructSequence_UnnamedField later. */
    {NULL,         "integer time of last access"},
    {NULL,         "integer time of last modification"},
    {NULL,         "integer time of last change"},
    {"st_atime",   "time of last access"},
    {"st_mtime",   "time of last modification"},
    {"st_ctime",   "time of last change"},
    {"st_atime_ns",   "time of last access in nanoseconds"},
    {"st_mtime_ns",   "time of last modification in nanoseconds"},
    {"st_ctime_ns",   "time of last change in nanoseconds"},
#ifdef HAVE_STRUCT_STAT_ST_BLKSIZE
    {"st_blksize", "blocksize for filesystem I/O"},
#endif
#ifdef HAVE_STRUCT_STAT_ST_BLOCKS
    {"st_blocks",  "number of blocks allocated"},
#endif
#ifdef HAVE_STRUCT_STAT_ST_RDEV
    {"st_rdev",    "device type (if inode device)"},
#endif
#ifdef HAVE_STRUCT_STAT_ST_FLAGS
    {"st_flags",   "user defined flags for file"},
#endif
#ifdef HAVE_STRUCT_STAT_ST_GEN
    {"st_gen",    "generation number"},
#endif
#ifdef HAVE_STRUCT_STAT_ST_BIRTHTIME
    {"st_birthtime",   "time of creation"},
#endif
#ifdef HAVE_STRUCT_STAT_ST_FILE_ATTRIBUTES
    {"st_file_attributes", "Windows file attribute bits"},
#endif
    {0}
};

static PyStructSequence_Desc stat_result_desc = {
    "scandir.stat_result", /* name */
    stat_result__doc__, /* doc */
    stat_result_fields,
    10
};


#ifdef MS_WINDOWS
static int
win32_warn_bytes_api()
{
    return PyErr_WarnEx(PyExc_DeprecationWarning,
        "The Windows bytes API has been deprecated, "
        "use Unicode filenames instead",
        1);
}
#endif

typedef struct {
    const char *function_name;
    const char *argument_name;
    int nullable;
    wchar_t *wide;
    char *narrow;
    int fd;
    Py_ssize_t length;
    PyObject *object;
    PyObject *cleanup;
} path_t;

static void
path_cleanup(path_t *path) {
    if (path->cleanup) {
        Py_CLEAR(path->cleanup);
    }
}

static int
path_converter(PyObject *o, void *p) {
    path_t *path = (path_t *)p;
    PyObject *unicode, *bytes;
    Py_ssize_t length;
    char *narrow;

#define FORMAT_EXCEPTION(exc, fmt) \
    PyErr_Format(exc, "%s%s" fmt, \
        path->function_name ? path->function_name : "", \
        path->function_name ? ": "                : "", \
        path->argument_name ? path->argument_name : "path")

    /* Py_CLEANUP_SUPPORTED support */
    if (o == NULL) {
        path_cleanup(path);
        return 1;
    }

    /* ensure it's always safe to call path_cleanup() */
    path->cleanup = NULL;

    if (o == Py_None) {
        if (!path->nullable) {
            FORMAT_EXCEPTION(PyExc_TypeError,
                             "can't specify None for %s argument");
            return 0;
        }
        path->wide = NULL;
        path->narrow = NULL;
        path->length = 0;
        path->object = o;
        path->fd = -1;
        return 1;
    }

    unicode = PyUnicode_FromObject(o);
    if (unicode) {
#ifdef MS_WINDOWS
        wchar_t *wide;

        wide = PyUnicode_AsUnicodeAndSize(unicode, &length);
        if (!wide) {
            Py_DECREF(unicode);
            return 0;
        }
        if (length > 32767) {
            FORMAT_EXCEPTION(PyExc_ValueError, "%s too long for Windows");
            Py_DECREF(unicode);
            return 0;
        }
        if (wcslen(wide) != length) {
            FORMAT_EXCEPTION(PyExc_ValueError, "embedded null character");
            Py_DECREF(unicode);
            return 0;
        }

        path->wide = wide;
        path->narrow = NULL;
        path->length = length;
        path->object = o;
        path->fd = -1;
        path->cleanup = unicode;
        return Py_CLEANUP_SUPPORTED;
#else
#if PY_MAJOR_VERSION >= 3
        if (!PyUnicode_FSConverter(unicode, &bytes))
            bytes = NULL;
#else
        bytes = PyUnicode_AsEncodedString(unicode, FS_ENCODING, "strict");
#endif
        Py_DECREF(unicode);
#endif
    }
    else {
        PyErr_Clear();
#if PY_MAJOR_VERSION >= 3
        if (PyObject_CheckBuffer(o)) {
            bytes = PyBytes_FromObject(o);
        }
#else
        if (PyString_Check(o)) {
            bytes = o;
            Py_INCREF(bytes);
        }
#endif
        else
            bytes = NULL;
        if (!bytes) {
            PyErr_Clear();
        }
    }

    if (!bytes) {
        if (!PyErr_Occurred())
            FORMAT_EXCEPTION(PyExc_TypeError, "illegal type for %s parameter");
        return 0;
    }

#ifdef MS_WINDOWS
    if (win32_warn_bytes_api()) {
        Py_DECREF(bytes);
        return 0;
    }
#endif

    length = PyBytes_GET_SIZE(bytes);
#ifdef MS_WINDOWS
    if (length > MAX_PATH-1) {
        FORMAT_EXCEPTION(PyExc_ValueError, "%s too long for Windows");
        Py_DECREF(bytes);
        return 0;
    }
#endif

    narrow = PyBytes_AS_STRING(bytes);
    if ((size_t)length != strlen(narrow)) {
        FORMAT_EXCEPTION(PyExc_ValueError, "embedded null character in %s");
        Py_DECREF(bytes);
        return 0;
    }

    path->wide = NULL;
    path->narrow = narrow;
    path->length = length;
    path->object = o;
    path->fd = -1;
    path->cleanup = bytes;
    return Py_CLEANUP_SUPPORTED;
}

static PyObject *
path_error(path_t *path)
{
#ifdef MS_WINDOWS
    return PyErr_SetExcFromWindowsErrWithFilenameObject(PyExc_OSError,
                                                        0, path->object);
#else
    return PyErr_SetFromErrnoWithFilenameObject(PyExc_OSError, path->object);
#endif
}


/* SECTION: Main DirEntry and scandir implementation, taken from
   Python 3.5's posixmodule.c */

PyDoc_STRVAR(posix_scandir__doc__,
"scandir(path='.') -> iterator of DirEntry objects for given path");

static char *follow_symlinks_keywords[] = {"follow_symlinks", NULL};
#if PY_MAJOR_VERSION >= 3 && PY_MINOR_VERSION >= 3
static char *follow_symlinks_format = "|$p:DirEntry.stat";
#else
static char *follow_symlinks_format = "|i:DirEntry.stat";
#endif

typedef struct {
    PyObject_HEAD
    PyObject *name;
    PyObject *path;
    PyObject *stat;
    PyObject *lstat;
#ifdef MS_WINDOWS
    struct _Py_stat_struct win32_lstat;
    unsigned __int64 win32_file_index;
    int got_file_index;
#if PY_MAJOR_VERSION < 3
    int name_path_bytes;
#endif
#else /* POSIX */
#ifdef HAVE_DIRENT_D_TYPE
    unsigned char d_type;
#endif
    ino_t d_ino;
#endif
} DirEntry;

static void
DirEntry_dealloc(DirEntry *entry)
{
    Py_XDECREF(entry->name);
    Py_XDECREF(entry->path);
    Py_XDECREF(entry->stat);
    Py_XDECREF(entry->lstat);
    Py_TYPE(entry)->tp_free((PyObject *)entry);
}

/* Forward reference */
static int
DirEntry_test_mode(DirEntry *self, int follow_symlinks, unsigned short mode_bits);

/* Set exception and return -1 on error, 0 for False, 1 for True */
static int
DirEntry_is_symlink(DirEntry *self)
{
#ifdef MS_WINDOWS
    return (self->win32_lstat.st_mode & S_IFMT) == S_IFLNK;
#elif defined(HAVE_DIRENT_D_TYPE)
    /* POSIX */
    if (self->d_type != DT_UNKNOWN)
        return self->d_type == DT_LNK;
    else
        return DirEntry_test_mode(self, 0, S_IFLNK);
#else
    /* POSIX without d_type */
    return DirEntry_test_mode(self, 0, S_IFLNK);
#endif
}

static PyObject *
DirEntry_py_is_symlink(DirEntry *self)
{
    int result;

    result = DirEntry_is_symlink(self);
    if (result == -1)
        return NULL;
    return PyBool_FromLong(result);
}

static PyObject *
DirEntry_fetch_stat(DirEntry *self, int follow_symlinks)
{
    int result;
    struct _Py_stat_struct st;

#ifdef MS_WINDOWS
    wchar_t *path;

    path = PyUnicode_AsUnicode(self->path);
    if (!path)
        return NULL;

    if (follow_symlinks)
        result = win32_stat_w(path, &st);
    else
        result = win32_lstat_w(path, &st);

    if (result != 0) {
        return PyErr_SetExcFromWindowsErrWithFilenameObject(PyExc_OSError,
                                                            0, self->path);
    }
#else /* POSIX */
    PyObject *bytes;
    char *path;

#if PY_MAJOR_VERSION >= 3
    if (!PyUnicode_FSConverter(self->path, &bytes))
        return NULL;
#else
    if (PyString_Check(self->path)) {
        bytes = self->path;
        Py_INCREF(bytes);
    } else {
        bytes = PyUnicode_AsEncodedString(self->path, FS_ENCODING, "strict");
        if (!bytes)
            return NULL;
    }
#endif
    path = PyBytes_AS_STRING(bytes);

    if (follow_symlinks)
        result = STAT(path, &st);
    else
        result = LSTAT(path, &st);
    Py_DECREF(bytes);

    if (result != 0)
        return PyErr_SetFromErrnoWithFilenameObject(PyExc_OSError, self->path);
#endif

    return _pystat_fromstructstat(&st);
}

static PyObject *
DirEntry_get_lstat(DirEntry *self)
{
    if (!self->lstat) {
#ifdef MS_WINDOWS
        self->lstat = _pystat_fromstructstat(&self->win32_lstat);
#else /* POSIX */
        self->lstat = DirEntry_fetch_stat(self, 0);
#endif
    }
    Py_XINCREF(self->lstat);
    return self->lstat;
}

static PyObject *
DirEntry_get_stat(DirEntry *self, int follow_symlinks)
{
    if (!follow_symlinks)
        return DirEntry_get_lstat(self);

    if (!self->stat) {
        int result = DirEntry_is_symlink(self);
        if (result == -1)
            return NULL;
        else if (result)
            self->stat = DirEntry_fetch_stat(self, 1);
        else
            self->stat = DirEntry_get_lstat(self);
    }

    Py_XINCREF(self->stat);
    return self->stat;
}

static PyObject *
DirEntry_stat(DirEntry *self, PyObject *args, PyObject *kwargs)
{
    int follow_symlinks = 1;

    if (!PyArg_ParseTupleAndKeywords(args, kwargs, follow_symlinks_format,
                                     follow_symlinks_keywords, &follow_symlinks))
        return NULL;

    return DirEntry_get_stat(self, follow_symlinks);
}

/* Set exception and return -1 on error, 0 for False, 1 for True */
static int
DirEntry_test_mode(DirEntry *self, int follow_symlinks, unsigned short mode_bits)
{
    PyObject *stat = NULL;
    PyObject *st_mode = NULL;
    long mode;
    int result;
#if defined(MS_WINDOWS) || defined(HAVE_DIRENT_D_TYPE)
    int is_symlink;
    int need_stat;
#endif
#ifdef MS_WINDOWS
    unsigned long dir_bits;
#endif
    _Py_IDENTIFIER(st_mode);

#ifdef MS_WINDOWS
    is_symlink = (self->win32_lstat.st_mode & S_IFMT) == S_IFLNK;
    need_stat = follow_symlinks && is_symlink;
#elif defined(HAVE_DIRENT_D_TYPE)
    is_symlink = self->d_type == DT_LNK;
    need_stat = self->d_type == DT_UNKNOWN || (follow_symlinks && is_symlink);
#endif

#if defined(MS_WINDOWS) || defined(HAVE_DIRENT_D_TYPE)
    if (need_stat) {
#endif
        stat = DirEntry_get_stat(self, follow_symlinks);
        if (!stat) {
            if (PyErr_ExceptionMatches(PyExc_FileNotFoundError)) {
                /* If file doesn't exist (anymore), then return False
                   (i.e., say it's not a file/directory) */
                PyErr_Clear();
                return 0;
            }
            goto error;
        }
        st_mode = _PyObject_GetAttrId(stat, &PyId_st_mode);
        if (!st_mode)
            goto error;

        mode = PyLong_AsLong(st_mode);
        if (mode == -1 && PyErr_Occurred())
            goto error;
        Py_CLEAR(st_mode);
        Py_CLEAR(stat);
        result = (mode & S_IFMT) == mode_bits;
#if defined(MS_WINDOWS) || defined(HAVE_DIRENT_D_TYPE)
    }
    else if (is_symlink) {
        assert(mode_bits != S_IFLNK);
        result = 0;
    }
    else {
        assert(mode_bits == S_IFDIR || mode_bits == S_IFREG);
#ifdef MS_WINDOWS
        dir_bits = self->win32_lstat.st_file_attributes & FILE_ATTRIBUTE_DIRECTORY;
        if (mode_bits == S_IFDIR)
            result = dir_bits != 0;
        else
            result = dir_bits == 0;
#else /* POSIX */
        if (mode_bits == S_IFDIR)
            result = self->d_type == DT_DIR;
        else
            result = self->d_type == DT_REG;
#endif
    }
#endif

    return result;

error:
    Py_XDECREF(st_mode);
    Py_XDECREF(stat);
    return -1;
}

static PyObject *
DirEntry_py_test_mode(DirEntry *self, int follow_symlinks, unsigned short mode_bits)
{
    int result;

    result = DirEntry_test_mode(self, follow_symlinks, mode_bits);
    if (result == -1)
        return NULL;
    return PyBool_FromLong(result);
}

static PyObject *
DirEntry_is_dir(DirEntry *self, PyObject *args, PyObject *kwargs)
{
    int follow_symlinks = 1;

    if (!PyArg_ParseTupleAndKeywords(args, kwargs, follow_symlinks_format,
                                     follow_symlinks_keywords, &follow_symlinks))
        return NULL;

    return DirEntry_py_test_mode(self, follow_symlinks, S_IFDIR);
}

static PyObject *
DirEntry_is_file(DirEntry *self, PyObject *args, PyObject *kwargs)
{
    int follow_symlinks = 1;

    if (!PyArg_ParseTupleAndKeywords(args, kwargs, follow_symlinks_format,
                                     follow_symlinks_keywords, &follow_symlinks))
        return NULL;

    return DirEntry_py_test_mode(self, follow_symlinks, S_IFREG);
}

static PyObject *
DirEntry_inode(DirEntry *self)
{
#ifdef MS_WINDOWS
    if (!self->got_file_index) {
        wchar_t *path;
        struct _Py_stat_struct stat;

        path = PyUnicode_AsUnicode(self->path);
        if (!path)
            return NULL;

        if (win32_lstat_w(path, &stat) != 0) {
            return PyErr_SetExcFromWindowsErrWithFilenameObject(PyExc_OSError,
                                                                0, self->path);
        }

        self->win32_file_index = stat.st_ino;
        self->got_file_index = 1;
    }
    return PyLong_FromUnsignedLongLong(self->win32_file_index);
#else /* POSIX */
#ifdef HAVE_LARGEFILE_SUPPORT
    return PyLong_FromUnsignedLongLong(self->d_ino);
#else
    return PyLong_FromUnsignedLong((unsigned long)self->d_ino);
#endif
#endif
}

#if PY_MAJOR_VERSION < 3 && defined(MS_WINDOWS)

PyObject *DirEntry_name_getter(DirEntry *self, void *closure) {
    if (self->name_path_bytes) {
        return PyUnicode_EncodeMBCS(PyUnicode_AS_UNICODE(self->name),
                                    PyUnicode_GetSize(self->name), "strict");
    } else {
        Py_INCREF(self->name);
        return self->name;
    }
}

PyObject *DirEntry_path_getter(DirEntry *self, void *closure) {
    if (self->name_path_bytes) {
        return PyUnicode_EncodeMBCS(PyUnicode_AS_UNICODE(self->path),
                                    PyUnicode_GetSize(self->path), "strict");
    } else {
        Py_INCREF(self->path);
        return self->path;
    }
}

static PyGetSetDef DirEntry_getset[] = {
    {"name", (getter)DirEntry_name_getter, NULL,
     "the entry's base filename, relative to scandir() \"path\" argument", NULL},
    {"path", (getter)DirEntry_path_getter, NULL,
     "the entry's full path name; equivalent to os.path.join(scandir_path, entry.name)", NULL},
    {NULL}
};

#else

static PyMemberDef DirEntry_members[] = {
    {"name", T_OBJECT_EX, offsetof(DirEntry, name), READONLY,
     "the entry's base filename, relative to scandir() \"path\" argument"},
    {"path", T_OBJECT_EX, offsetof(DirEntry, path), READONLY,
     "the entry's full path name; equivalent to os.path.join(scandir_path, entry.name)"},
    {NULL}
};

#endif

static PyObject *
DirEntry_repr(DirEntry *self)
{
#if PY_MAJOR_VERSION >= 3
    return PyUnicode_FromFormat("<DirEntry %R>", self->name);
#elif defined(MS_WINDOWS)
    PyObject *name;
    PyObject *name_repr;
    PyObject *entry_repr;

    name = DirEntry_name_getter(self, NULL);
    if (!name)
        return NULL;
    name_repr = PyObject_Repr(name);
    Py_DECREF(name);
    if (!name_repr)
        return NULL;
    entry_repr = PyString_FromFormat("<DirEntry %s>", PyString_AsString(name_repr));
    Py_DECREF(name_repr);
    return entry_repr;
#else
    PyObject *name_repr;
    PyObject *entry_repr;

    name_repr = PyObject_Repr(self->name);
    if (!name_repr)
        return NULL;
    entry_repr = PyString_FromFormat("<DirEntry %s>", PyString_AsString(name_repr));
    Py_DECREF(name_repr);
    return entry_repr;
#endif
}

static PyMethodDef DirEntry_methods[] = {
    {"is_dir", (PyCFunction)DirEntry_is_dir, METH_VARARGS | METH_KEYWORDS,
     "return True if the entry is a directory; cached per entry"
    },
    {"is_file", (PyCFunction)DirEntry_is_file, METH_VARARGS | METH_KEYWORDS,
     "return True if the entry is a file; cached per entry"
    },
    {"is_symlink", (PyCFunction)DirEntry_py_is_symlink, METH_NOARGS,
     "return True if the entry is a symbolic link; cached per entry"
    },
    {"stat", (PyCFunction)DirEntry_stat, METH_VARARGS | METH_KEYWORDS,
     "return stat_result object for the entry; cached per entry"
    },
    {"inode", (PyCFunction)DirEntry_inode, METH_NOARGS,
     "return inode of the entry; cached per entry",
    },
    {NULL}
};

static PyTypeObject DirEntryType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    MODNAME ".DirEntry",                    /* tp_name */
    sizeof(DirEntry),                       /* tp_basicsize */
    0,                                      /* tp_itemsize */
    /* methods */
    (destructor)DirEntry_dealloc,           /* tp_dealloc */
    0,                                      /* tp_print */
    0,                                      /* tp_getattr */
    0,                                      /* tp_setattr */
    0,                                      /* tp_compare */
    (reprfunc)DirEntry_repr,                /* tp_repr */
    0,                                      /* tp_as_number */
    0,                                      /* tp_as_sequence */
    0,                                      /* tp_as_mapping */
    0,                                      /* tp_hash */
    0,                                      /* tp_call */
    0,                                      /* tp_str */
    0,                                      /* tp_getattro */
    0,                                      /* tp_setattro */
    0,                                      /* tp_as_buffer */
    Py_TPFLAGS_DEFAULT,                     /* tp_flags */
    0,                                      /* tp_doc */
    0,                                      /* tp_traverse */
    0,                                      /* tp_clear */
    0,                                      /* tp_richcompare */
    0,                                      /* tp_weaklistoffset */
    0,                                      /* tp_iter */
    0,                                      /* tp_iternext */
    DirEntry_methods,                       /* tp_methods */
#if PY_MAJOR_VERSION < 3 && defined(MS_WINDOWS)
    NULL,                                   /* tp_members */
    DirEntry_getset,                        /* tp_getset */
#else
    DirEntry_members,                       /* tp_members */
    NULL,                                   /* tp_getset */
#endif
};

#ifdef MS_WINDOWS

static wchar_t *
join_path_filenameW(wchar_t *path_wide, wchar_t* filename)
{
    Py_ssize_t path_len;
    Py_ssize_t size;
    wchar_t *result;
    wchar_t ch;

    if (!path_wide) { /* Default arg: "." */
        path_wide = L".";
        path_len = 1;
    }
    else {
        path_len = wcslen(path_wide);
    }

    /* The +1's are for the path separator and the NUL */
    size = path_len + 1 + wcslen(filename) + 1;
    result = PyMem_New(wchar_t, size);
    if (!result) {
        PyErr_NoMemory();
        return NULL;
    }
    wcscpy(result, path_wide);
    if (path_len > 0) {
        ch = result[path_len - 1];
        if (ch != SEP && ch != ALTSEP && ch != L':')
            result[path_len++] = SEP;
        wcscpy(result + path_len, filename);
    }
    return result;
}

static PyObject *
DirEntry_from_find_data(path_t *path, WIN32_FIND_DATAW *dataW)
{
    DirEntry *entry;
    BY_HANDLE_FILE_INFORMATION file_info;
    ULONG reparse_tag;
    wchar_t *joined_path;

    entry = PyObject_New(DirEntry, &DirEntryType);
    if (!entry)
        return NULL;
    entry->name = NULL;
    entry->path = NULL;
    entry->stat = NULL;
    entry->lstat = NULL;
    entry->got_file_index = 0;
#if PY_MAJOR_VERSION < 3
    entry->name_path_bytes = path->object && PyBytes_Check(path->object);
#endif

    entry->name = PyUnicode_FromWideChar(dataW->cFileName, wcslen(dataW->cFileName));
    if (!entry->name)
        goto error;

    joined_path = join_path_filenameW(path->wide, dataW->cFileName);
    if (!joined_path)
        goto error;

    entry->path = PyUnicode_FromWideChar(joined_path, wcslen(joined_path));
    PyMem_Free(joined_path);
    if (!entry->path)
        goto error;

    find_data_to_file_info_w(dataW, &file_info, &reparse_tag);
    _Py_attribute_data_to_stat(&file_info, reparse_tag, &entry->win32_lstat);

    return (PyObject *)entry;

error:
    Py_DECREF(entry);
    return NULL;
}

#else /* POSIX */

static char *
join_path_filename(char *path_narrow, char* filename, Py_ssize_t filename_len)
{
    Py_ssize_t path_len;
    Py_ssize_t size;
    char *result;

    if (!path_narrow) { /* Default arg: "." */
        path_narrow = ".";
        path_len = 1;
    }
    else {
        path_len = strlen(path_narrow);
    }

    if (filename_len == -1)
        filename_len = strlen(filename);

    /* The +1's are for the path separator and the NUL */
    size = path_len + 1 + filename_len + 1;
    result = PyMem_New(char, size);
    if (!result) {
        PyErr_NoMemory();
        return NULL;
    }
    strcpy(result, path_narrow);
    if (path_len > 0 && result[path_len - 1] != '/')
        result[path_len++] = '/';
    strcpy(result + path_len, filename);
    return result;
}

static PyObject *
DirEntry_from_posix_info(path_t *path, char *name, Py_ssize_t name_len,
                         ino_t d_ino
#ifdef HAVE_DIRENT_D_TYPE
                         , unsigned char d_type
#endif
                         )
{
    DirEntry *entry;
    char *joined_path;

    entry = PyObject_New(DirEntry, &DirEntryType);
    if (!entry)
        return NULL;
    entry->name = NULL;
    entry->path = NULL;
    entry->stat = NULL;
    entry->lstat = NULL;

    joined_path = join_path_filename(path->narrow, name, name_len);
    if (!joined_path)
        goto error;

    if (!path->narrow || !PyBytes_Check(path->object)) {
#if PY_MAJOR_VERSION >= 3
        entry->name = PyUnicode_DecodeFSDefaultAndSize(name, name_len);
        entry->path = PyUnicode_DecodeFSDefault(joined_path);
#else
        entry->name = PyUnicode_Decode(name, name_len,
                                       FS_ENCODING, "strict");
        entry->path = PyUnicode_Decode(joined_path, strlen(joined_path),
                                       FS_ENCODING, "strict");
#endif
    }
    else {
        entry->name = PyBytes_FromStringAndSize(name, name_len);
        entry->path = PyBytes_FromString(joined_path);
    }
    PyMem_Free(joined_path);
    if (!entry->name || !entry->path)
        goto error;

#ifdef HAVE_DIRENT_D_TYPE
    entry->d_type = d_type;
#endif
    entry->d_ino = d_ino;

    return (PyObject *)entry;

error:
    Py_XDECREF(entry);
    return NULL;
}

#endif


typedef struct {
    PyObject_HEAD
    path_t path;
#ifdef MS_WINDOWS
    HANDLE handle;
    WIN32_FIND_DATAW file_data;
    int first_time;
#else /* POSIX */
    DIR *dirp;
#endif
} ScandirIterator;

#ifdef MS_WINDOWS

static void
ScandirIterator_close(ScandirIterator *iterator)
{
    if (iterator->handle == INVALID_HANDLE_VALUE)
        return;

    Py_BEGIN_ALLOW_THREADS
    FindClose(iterator->handle);
    Py_END_ALLOW_THREADS
    iterator->handle = INVALID_HANDLE_VALUE;
}

static PyObject *
ScandirIterator_iternext(ScandirIterator *iterator)
{
    WIN32_FIND_DATAW *file_data = &iterator->file_data;
    BOOL success;

    /* Happens if the iterator is iterated twice */
    if (iterator->handle == INVALID_HANDLE_VALUE) {
        PyErr_SetNone(PyExc_StopIteration);
        return NULL;
    }

    while (1) {
        if (!iterator->first_time) {
            Py_BEGIN_ALLOW_THREADS
            success = FindNextFileW(iterator->handle, file_data);
            Py_END_ALLOW_THREADS
            if (!success) {
                if (GetLastError() != ERROR_NO_MORE_FILES)
                    return path_error(&iterator->path);
                /* No more files found in directory, stop iterating */
                break;
            }
        }
        iterator->first_time = 0;

        /* Skip over . and .. */
        if (wcscmp(file_data->cFileName, L".") != 0 &&
                wcscmp(file_data->cFileName, L"..") != 0)
            return DirEntry_from_find_data(&iterator->path, file_data);

        /* Loop till we get a non-dot directory or finish iterating */
    }

    ScandirIterator_close(iterator);

    PyErr_SetNone(PyExc_StopIteration);
    return NULL;
}

#else /* POSIX */

static void
ScandirIterator_close(ScandirIterator *iterator)
{
    if (!iterator->dirp)
        return;

    Py_BEGIN_ALLOW_THREADS
    closedir(iterator->dirp);
    Py_END_ALLOW_THREADS
    iterator->dirp = NULL;
    return;
}

static PyObject *
ScandirIterator_iternext(ScandirIterator *iterator)
{
    struct dirent *direntp;
    Py_ssize_t name_len;
    int is_dot;

    /* Happens if the iterator is iterated twice */
    if (!iterator->dirp) {
        PyErr_SetNone(PyExc_StopIteration);
        return NULL;
    }

    while (1) {
        errno = 0;
        Py_BEGIN_ALLOW_THREADS
        direntp = readdir(iterator->dirp);
        Py_END_ALLOW_THREADS

        if (!direntp) {
            if (errno != 0)
                return path_error(&iterator->path);
            /* No more files found in directory, stop iterating */
            break;
        }

        /* Skip over . and .. */
        name_len = NAMLEN(direntp);
        is_dot = direntp->d_name[0] == '.' &&
                 (name_len == 1 || (direntp->d_name[1] == '.' && name_len == 2));
        if (!is_dot) {
            return DirEntry_from_posix_info(&iterator->path, direntp->d_name,
                                            name_len, direntp->d_ino
#ifdef HAVE_DIRENT_D_TYPE
                                            , direntp->d_type
#endif
                                            );
        }

        /* Loop till we get a non-dot directory or finish iterating */
    }

    ScandirIterator_close(iterator);

    PyErr_SetNone(PyExc_StopIteration);
    return NULL;
}

#endif

static void
ScandirIterator_dealloc(ScandirIterator *iterator)
{
    ScandirIterator_close(iterator);
    Py_XDECREF(iterator->path.object);
    path_cleanup(&iterator->path);
    Py_TYPE(iterator)->tp_free((PyObject *)iterator);
}

static PyTypeObject ScandirIteratorType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    MODNAME ".ScandirIterator",             /* tp_name */
    sizeof(ScandirIterator),                /* tp_basicsize */
    0,                                      /* tp_itemsize */
    /* methods */
    (destructor)ScandirIterator_dealloc,    /* tp_dealloc */
    0,                                      /* tp_print */
    0,                                      /* tp_getattr */
    0,                                      /* tp_setattr */
    0,                                      /* tp_compare */
    0,                                      /* tp_repr */
    0,                                      /* tp_as_number */
    0,                                      /* tp_as_sequence */
    0,                                      /* tp_as_mapping */
    0,                                      /* tp_hash */
    0,                                      /* tp_call */
    0,                                      /* tp_str */
    0,                                      /* tp_getattro */
    0,                                      /* tp_setattro */
    0,                                      /* tp_as_buffer */
    Py_TPFLAGS_DEFAULT,                     /* tp_flags */
    0,                                      /* tp_doc */
    0,                                      /* tp_traverse */
    0,                                      /* tp_clear */
    0,                                      /* tp_richcompare */
    0,                                      /* tp_weaklistoffset */
    PyObject_SelfIter,                      /* tp_iter */
    (iternextfunc)ScandirIterator_iternext, /* tp_iternext */
};

static PyObject *
posix_scandir(PyObject *self, PyObject *args, PyObject *kwargs)
{
    ScandirIterator *iterator;
    static char *keywords[] = {"path", NULL};
#ifdef MS_WINDOWS
    wchar_t *path_strW;
#else
    char *path;
#endif

    iterator = PyObject_New(ScandirIterator, &ScandirIteratorType);
    if (!iterator)
        return NULL;
    memset(&iterator->path, 0, sizeof(path_t));
    iterator->path.function_name = "scandir";
    iterator->path.nullable = 1;

#ifdef MS_WINDOWS
    iterator->handle = INVALID_HANDLE_VALUE;
#else
    iterator->dirp = NULL;
#endif

    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "|O&:scandir", keywords,
                                     path_converter, &iterator->path))
        goto error;

    /* path_converter doesn't keep path.object around, so do it
       manually for the lifetime of the iterator here (the refcount
       is decremented in ScandirIterator_dealloc)
    */
    Py_XINCREF(iterator->path.object);

#ifdef MS_WINDOWS
    if (iterator->path.narrow) {
        PyErr_SetString(PyExc_TypeError,
                        "os.scandir() doesn't support bytes path on Windows, use Unicode instead");
        goto error;
    }
    iterator->first_time = 1;

    path_strW = join_path_filenameW(iterator->path.wide, L"*.*");
    if (!path_strW)
        goto error;

    Py_BEGIN_ALLOW_THREADS
    iterator->handle = FindFirstFileW(path_strW, &iterator->file_data);
    Py_END_ALLOW_THREADS

    PyMem_Free(path_strW);

    if (iterator->handle == INVALID_HANDLE_VALUE) {
        path_error(&iterator->path);
        goto error;
    }
#else /* POSIX */
    if (iterator->path.narrow)
        path = iterator->path.narrow;
    else
        path = ".";

    errno = 0;
    Py_BEGIN_ALLOW_THREADS
    iterator->dirp = opendir(path);
    Py_END_ALLOW_THREADS

    if (!iterator->dirp) {
        path_error(&iterator->path);
        goto error;
    }
#endif

    return (PyObject *)iterator;

error:
    Py_DECREF(iterator);
    return NULL;
}


/* SECTION: Module and method definitions and initialization code */

static PyMethodDef scandir_methods[] = {
    {"scandir",         (PyCFunction)posix_scandir,
                        METH_VARARGS | METH_KEYWORDS,
                        posix_scandir__doc__},
    {NULL, NULL},
};

#if PY_MAJOR_VERSION >= 3
static struct PyModuleDef moduledef = {
        PyModuleDef_HEAD_INIT,
        "_scandir",
        NULL,
        0,
        scandir_methods,
        NULL,
        NULL,
        NULL,
        NULL,
};
#endif

#if PY_MAJOR_VERSION >= 3
PyObject *
PyInit__scandir(void)
{
    PyObject *module = PyModule_Create(&moduledef);
#else
void
init_scandir(void)
{
    PyObject *module = Py_InitModule("_scandir", scandir_methods);
#endif
    if (module == NULL) {
        INIT_ERROR;
    }

    billion = PyLong_FromLong(1000000000);
    if (!billion)
        INIT_ERROR;

    stat_result_desc.fields[7].name = PyStructSequence_UnnamedField;
    stat_result_desc.fields[8].name = PyStructSequence_UnnamedField;
    stat_result_desc.fields[9].name = PyStructSequence_UnnamedField;
    PyStructSequence_InitType(&StatResultType, &stat_result_desc);
    structseq_new = StatResultType.tp_new;
    StatResultType.tp_new = statresult_new;

    if (PyType_Ready(&ScandirIteratorType) < 0)
        INIT_ERROR;
    if (PyType_Ready(&DirEntryType) < 0)
        INIT_ERROR;

    PyModule_AddObject(module, "DirEntry", (PyObject *)&DirEntryType);

#if PY_MAJOR_VERSION >= 3
    return module;
#endif
}
