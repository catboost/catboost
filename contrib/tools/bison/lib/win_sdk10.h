#pragma once

#if defined(_MSC_VER)

#include <ntverp.h>

// Check for Windows SDK 10+
#if defined(VER_PRODUCTBUILD) && VER_PRODUCTBUILD >= 9600
#define WIN_SDK10 1
#else
#define WIN_SDK10 0
#endif

// Since Windows SDK 10 FILE is an internal opaque structure with no backward compatibility.
// This code has been transplanted from Windows SDK
// corecrt_internal_stdio.h

// __crt_stdio_stream_data
#if WIN_SDK10
typedef struct {
    union {
        void* _public_file;
        char* _ptr;
    };

    char* _base;
    int _cnt;
    long _flags;
    long _file;
    int _charbuf;
    int _bufsiz;
    char* _tmpfname;
    //CRITICAL_SECTION _lock;
} TWinSdk10File;

enum EWinSdk10ModeBits {
    WIN_SDK10_IOREAD = 0x0001,
    WIN_SDK10_IOWRITE = 0x0002,
    WIN_SDK10_IOUPDATE = 0x0004,
    WIN_SDK10_IOEOF = 0x0008,
    WIN_SDK10_IOERROR = 0x0010,
    WIN_SDK10_IOCTRLZ = 0x0020,
    WIN_SDK10_IOBUFFER_CRT = 0x0040,
    WIN_SDK10_IOBUFFER_USER = 0x0080,
    WIN_SDK10_IOBUFFER_SETVBUF = 0x0100,
    WIN_SDK10_IOBUFFER_STBUF = 0x0200,
    WIN_SDK10_IOBUFFER_NONE = 0x0400,
    WIN_SDK10_IOCOMMIT = 0x0800,
    WIN_SDK10_IOSTRING = 0x1000,
    WIN_SDK10_IOALLOCATED = 0x2000,
};
#endif

#endif

