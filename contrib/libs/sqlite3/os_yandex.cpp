#include "bridge.h"

#include <util/folder/dirut.h>
#include <util/random/entropy.h>
#include <util/datetime/systime.h>
#include <util/system/file.h>
#include <util/system/fs.h>
#include <util/system/maxlen.h>
#include <util/system/mktemp.h>
#include <util/stream/input.h>

extern "C" {
#include "sqlite3.h"
}

struct TYandexFile {
    const sqlite3_io_methods* Methods;  /* Always the first entry */
    TString FileName;
    TFileHandle Handle;
    int LockType;
    bool DeleteOnClose;

    TYandexFile(const sqlite3_io_methods* methods, const TString& fname, EOpenMode omode, bool deleteOnClose)
        : Methods(methods)
        , FileName(fname)
        , Handle(fname, omode)
        , LockType(NO_LOCK)
        , DeleteOnClose(deleteOnClose)
    {
    }

    ~TYandexFile() {
        if (Handle.IsOpen()) {
            Handle.Close();
            if (DeleteOnClose) {
                NFs::Remove(FileName);
            }
        }
    }

    static void* operator new(size_t, sqlite3_file* f) {
        return (TYandexFile*)f;
    }
};

/*****************************************************************************
** The next group of routines implement the I/O methods specified
** by the sqlite3_io_methods object.
******************************************************************************/

static int YandexUnlock(sqlite3_file* id, int locktype);

/*
** Close a file.
*/
static int YandexClose(sqlite3_file* id) {
    YandexUnlock(id, NO_LOCK);

    TYandexFile* file = (TYandexFile*)id;
    file->~TYandexFile();

    return SQLITE_OK;
}

/*
** Read data from a file into a buffer.  Return SQLITE_OK if all
** bytes were read successfully and SQLITE_IOERR if anything goes
** wrong.
*/
static int YandexRead(
    sqlite3_file* id,               /* File to read from */
    void* pBuf,                     /* Write content into this buffer */
    int amt,                        /* Number of bytes to read */
    sqlite3_int64 offset            /* Begin reading at this offset */
) {
    i32 read = ((TYandexFile*)id)->Handle.Pread(pBuf, amt, offset);
    if (read < 0) {
        return SQLITE_IOERR_READ;
    }
    if (read < amt) {
        memset((char*)pBuf + read, 0, amt - read); // Required by sqlite3
    }
    return read == amt ? SQLITE_OK : SQLITE_IOERR_SHORT_READ;
}

/*
** Write data from a buffer into a file.  Return SQLITE_OK on success
** or some other error code on failure.
*/
static int YandexWrite(
    sqlite3_file* id,               /* File to write into */
    const void* pBuf,               /* The bytes to be written */
    int amt,                        /* Number of bytes to write */
    sqlite3_int64 offset            /* Offset into the file to begin writing at */
) {
    int wrote;
    while(amt  > 0 && (wrote = ((TYandexFile*)id)->Handle.Pwrite(pBuf, amt, offset)) > 0) {
        amt -= wrote;
        offset += wrote;
        pBuf = &((char*)pBuf)[wrote];
    }
    if (amt > 0) {
        return wrote < 0 ? SQLITE_IOERR_WRITE : SQLITE_FULL;
    }
    return SQLITE_OK;
}

/*
** Truncate an open file to a specified size
*/
static int YandexTruncate(sqlite3_file* id, i64 nByte) {
    if (!((TYandexFile*)id)->Handle.Resize(nByte)) {
        return SQLITE_IOERR_TRUNCATE;
    }
    return SQLITE_OK;
}

/*
** Make sure all writes to a particular file are committed to disk.
*/
static int YandexSync(sqlite3_file* id, int /*flags*/) {
    if (!((TYandexFile*)id)->Handle.Flush()) {
        return SQLITE_IOERR_FSYNC;
    }
    return SQLITE_OK;
}

/*
** Determine the current size of a file in bytes
*/
static int TYandexFileSize(sqlite3_file* id, sqlite3_int64 *pSize) {
    *pSize = ((TYandexFile*)id)->Handle.GetLength();
    return SQLITE_OK;
}

/*
** Lock the file with the lock specified by parameter locktype - one
** of the following:
**
**     (1) SHARED_LOCK
**     (2) RESERVED_LOCK
**     (3) PENDING_LOCK
**     (4) EXCLUSIVE_LOCK
**
** Sometimes when requesting one lock state, additional lock states
** are inserted in between.  The locking might fail on one of the later
** transitions leaving the lock state different from what it started but
** still short of its goal.  The following chart shows the allowed
** transitions and the inserted intermediate states:
**
**    UNLOCKED -> SHARED
**    SHARED -> RESERVED
**    SHARED -> (PENDING) -> EXCLUSIVE
**    RESERVED -> (PENDING) -> EXCLUSIVE
**    PENDING -> EXCLUSIVE
**
** This routine will only increase a lock.  The os2Unlock() routine
** erases all locks at once and returns us immediately to locking level 0.
** It is not possible to lower the locking level one step at a time.  You
** must go straight to locking level 0.
*/
static int YandexLock(sqlite3_file* id, int locktype) {
    TYandexFile* file = (TYandexFile*)id;

    /* if we already have a lock, it is exclusive.
    ** Just adjust level and punt on outta here. */
    if (file->LockType > NO_LOCK) {
        file->LockType = locktype;
        return SQLITE_OK;
    }

    /* grab an exclusive lock */
    if (file->Handle.Flock(LOCK_EX | LOCK_NB)) {
        return SQLITE_BUSY;
    }

    /* got it, set the type and return ok */
    file->LockType = locktype;
    return SQLITE_OK;
}

/*
** This routine checks if there is a RESERVED lock held on the specified
** file by this or any other process. If such a lock is held, return
** non-zero, otherwise zero.
*/
static int YandexCheckReservedLock(sqlite3_file* id, int* out) {
    TYandexFile* file = (TYandexFile*)id;

    *out = 0;

    /* Check if a thread in this process holds such a lock */
    if (file->LockType > SHARED_LOCK) {
        *out = 1;
        return SQLITE_OK;
    }

    /* Otherwise see if some other process holds it. */

    /* attempt to get the lock */
    if (!file->Handle.Flock(LOCK_EX | LOCK_NB)) {
        /* got the lock, unlock it */
        if (file->Handle.Flock(LOCK_UN)) {
            return SQLITE_IOERR_UNLOCK;
        }
        *out = 0;
        return SQLITE_OK;
    }

    /* someone else might have it reserved */
    *out = 1;
    return SQLITE_OK;
}

/*
** Lower the locking level on file descriptor id to locktype.  locktype
** must be either NO_LOCK or SHARED_LOCK.
**
** If the locking level of the file descriptor is already at or below
** the requested locking level, this routine is a no-op.
**
** It is not possible for this routine to fail if the second argument
** is NO_LOCK.  If the second argument is SHARED_LOCK then this routine
** might return SQLITE_IOERR;
*/
static int YandexUnlock(sqlite3_file* id, int locktype) {
    TYandexFile* file = (TYandexFile*)id;

    /* no-op if possible */
    if (file->LockType == locktype) {
        return SQLITE_OK;
    }

    /* shared can just be set because we always have an exclusive */
    if (locktype == SHARED_LOCK) {
        file->LockType = locktype;
        return SQLITE_OK;
    }

    /* no, really, unlock. */
    if (file->Handle.Flock(LOCK_UN)) {
        return SQLITE_IOERR_UNLOCK;
    }

    file->LockType = NO_LOCK;
    return SQLITE_OK;
}

/*
** Control and query of the open file handle.
*/
static int TYandexFileControl(sqlite3_file* id, int op, void* pArg){
    switch(op) {
    case SQLITE_FCNTL_LOCKSTATE:
        *(int*)pArg = ((TYandexFile*)id)->LockType;
        return SQLITE_OK;
    default:
        return SQLITE_NOTFOUND;
    }
}

/*
** Return the sector size in bytes of the underlying block device for
** the specified file. This is almost always 512 bytes, but may be
** larger for some devices.
**
** SQLite code assumes this function cannot fail. It also assumes that
** if two files are created in the same file-system directory (i.e.
** a database and its journal file) that the sector size will be the
** same for both.
*/
static int YandexSectorSize(sqlite3_file* /*id*/) {
  return SQLITE_DEFAULT_SECTOR_SIZE;
}

/*
** Return a vector of device characteristics.
*/
static int YandexDeviceCharacteristics(sqlite3_file* /*id*/) {
  return 0;
}

/*
** This vector defines all the methods that can operate on an
** sqlite3_file for yandex.
*/
static const sqlite3_io_methods YandexIoMethods = {
    1,                        /* iVersion */
    YandexClose,
    YandexRead,
    YandexWrite,
    YandexTruncate,
    YandexSync,
    TYandexFileSize,
    YandexLock,
    YandexUnlock,
    YandexCheckReservedLock,
    TYandexFileControl,
    YandexSectorSize,
    YandexDeviceCharacteristics,
};

/***************************************************************************
** Here ends the I/O methods that form the sqlite3_io_methods object.
**
** The next block of code implements the VFS methods.
****************************************************************************/


/*
** Turn a relative pathname into a full pathname.  Write the full
** pathname into zFull[].  zFull[] will be at least pVfs->mxPathname
** bytes in size.
*/
static int YandexFullPathname(
    sqlite3_vfs* /*pVfs*/,      /* Pointer to vfs object */
    const char* zRelative,      /* Possibly relative input path */
    int nFull,                  /* Size of output buffer in bytes */
    char* zFull                 /* Output buffer */
) {
    zFull[nFull - 1] = '\0';
    try {
#ifdef _win_
        TString full = RealPath(zRelative);
        sqlite3_snprintf(nFull, zFull, "%s", ~full);
        return SQLITE_OK;
#else
        if (zRelative[0] == '/') {
            sqlite3_snprintf(nFull, zFull, "%s", zRelative);
        } else {
            TString cwd = NFs::CurrentWorkingDirectory();
            sqlite3_snprintf(nFull, zFull, "%s", ~(cwd + "/" + zRelative));
        }
        return SQLITE_OK;
#endif
    } catch (...) {
        return SQLITE_IOERR;
    }
    return SQLITE_IOERR;
}


/*
** Open the file zPath.
**
** Previously, the SQLite OS layer used three functions in place of this
** one:
**
**     sqlite3OsOpenReadWrite();
**     sqlite3OsOpenReadOnly();
**     sqlite3OsOpenExclusive();
**
** These calls correspond to the following combinations of flags:
**
**     ReadWrite() ->     (READWRITE | CREATE)
**     ReadOnly()  ->     (READONLY)
**     OpenExclusive() -> (READWRITE | CREATE | EXCLUSIVE)
**
** The old OpenExclusive() accepted a boolean argument - "delFlag". If
** true, the file was configured to be automatically deleted when the
** file handle closed. To achieve the same effect using this new
** interface, add the DELETEONCLOSE flag to those specified above for
** OpenExclusive().
*/
static int YandexOpen(
    sqlite3_vfs* /*pVfs*/,        /* Not used */
    const char* zName,            /* Name of the file */
    sqlite3_file* id,             /* Write the SQLite file handle here */
    int flags,                    /* Open mode flags */
    int* pOutFlags                /* Status return flags */
) {
    TString fname(zName);
    bool tmp = false;
    if (!fname) {
        fname = MakeTempName(NULL, "yandex.sqlite");
        tmp = true;
    }

    EOpenMode omode = 0;

    // MakeTempName creates a new file and new TYandexFile fails to open it with mode CreateNew
    if ((flags & SQLITE_OPEN_CREATE) != 0 && !tmp) {
        if (flags & SQLITE_OPEN_EXCLUSIVE) {
            omode |= CreateNew;
        } else {
            omode |= OpenAlways;
        }
    }

    if (flags & SQLITE_OPEN_READONLY) {
        omode |= RdOnly;
    }
    if (flags & SQLITE_OPEN_READWRITE) {
        omode |= RdWr;
    }

    bool deleteOnClose = (flags & SQLITE_OPEN_DELETEONCLOSE);

    TYandexFile* file = new (id) TYandexFile(&YandexIoMethods, fname, omode, deleteOnClose);

    if (!file->Handle.IsOpen()) {
        file->~TYandexFile();
        file->Methods = NULL; // Otherwise unneeded Close() may be called by sqlite3
        return SQLITE_IOERR;
    }

    if (pOutFlags) {
        *pOutFlags = flags & SQLITE_OPEN_READWRITE ? SQLITE_OPEN_READWRITE : SQLITE_OPEN_READONLY;
    }

    return SQLITE_OK;
}

/*
** Delete the named file.
*/
static int YandexDelete(
    sqlite3_vfs* /*pVfs*/,                 /* Not used */
    const char* zFilename,                 /* Name of file to delete */
    int /*syncDir*/                        /* Not used */
) {
    if (NFs::Exists(zFilename)) {
        if (!NFs::Remove(zFilename)) {
            return SQLITE_IOERR_DELETE;
        }
    }
    return SQLITE_OK;
}

/*
** Test the existance of or access permissions of file zPath. The
** test performed depends on the value of flags:
**
**     SQLITE_ACCESS_EXISTS: Return 1 if the file exists
**     SQLITE_ACCESS_READWRITE: Return 1 if the file is read and writable.
**     SQLITE_ACCESS_READONLY: Return 1 if the file is readable.
**
** Otherwise return 0.
*/
static int YandexAccess(
    sqlite3_vfs* /*pVfs*/,    /* Not used */
    const char* zFilename,    /* Name of file to check */
    int /*flags*/,            /* Type of test to make on this file */
    int* pOut                 /* Write results here */
) {
    // XXX: Fake implementation, but should be sufficient
    *pOut = NFs::Exists(zFilename);
    return SQLITE_OK;
}

/*
** Write up to nBuf bytes of randomness into zBuf.
*/
static int YandexRandomness(sqlite3_vfs* /*pVfs*/, int nBuf, char* zBuf) {
    return EntropyPool().Load(zBuf, nBuf);
}

/*
** Sleep for a little while.  Return the amount of time slept.
** The argument is the number of microseconds we want to sleep.
** The return value is the number of microseconds of sleep actually
** requested from the underlying operating system, a number which
** might be greater than or equal to the argument, but not less
** than the argument.
*/
static int YandexSleep(sqlite3_vfs* /*pVfs*/, int microsec) {
  usleep(microsec);
  return microsec;
}

/*
** Find the current time (in Universal Coordinated Time).  Write the
** current time and date as a Julian Day number into *prNow and
** return 0.  Return 1 if the time and date cannot be found.
*/
int YandexCurrentTime(sqlite3_vfs* /*pVfs*/, double* prNow) {
  struct timeval sNow;
  gettimeofday(&sNow, 0);
  *prNow = 2440587.5 + sNow.tv_sec/86400.0 + sNow.tv_usec/86400000000.0;
  return 0;
}

static int YandexGetLastError(sqlite3_vfs* /*pVfs*/, int /*nBuf*/, char* /*zBuf*/) {
    // Fake impl, because it's never called in sqlite3 source
    return 0;
}

extern "C" {

/*
** Initialize and deinitialize the operating system interface.
*/
int sqlite3_os_init() {
    static sqlite3_vfs YandexVfs = {
        1,                 /* iVersion */
        sizeof(TYandexFile), /* szOsFile */
        MAX_PATH,          /* mxPathname */
        0,                 /* pNext */
        "Yandex",          /* zName */
        0,                 /* pAppData */

        YandexOpen,        /* xOpen */
        YandexDelete,      /* xDelete */
        YandexAccess,      /* xAccess */
        YandexFullPathname, /* xFullPathname */
        0,                 /* xDlOpen */
        0,                 /* xDlError */
        0,                 /* xDlSym */
        0,                 /* xDlClose */
        YandexRandomness,  /* xRandomness */
        YandexSleep,       /* xSleep */
        YandexCurrentTime, /* xCurrentTime */
        YandexGetLastError /* xGetLastError */
    };
    sqlite3_vfs_register(&YandexVfs, 1);
    return SQLITE_OK;
}

int sqlite3_os_end() {
  return SQLITE_OK;
}

}
