/*
 * Copyright (c) 1987, 1993
 *      The Regents of the University of California.  All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 * 1. Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 * 3. All advertising materials mentioning features or use of this software
 *    must display the following acknowledgement:
 *      This product includes software developed by the University of
 *      California, Berkeley and its contributors.
 * 4. Neither the name of the University nor the names of its contributors
 *    may be used to endorse or promote products derived from this software
 *    without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE REGENTS AND CONTRIBUTORS ``AS IS'' AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED.  IN NO EVENT SHALL THE REGENTS OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS
 * OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
 * HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 * LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY
 * OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF
 * SUCH DAMAGE.
 */

#include "defaults.h"

#include <fcntl.h>
#include <errno.h>
#include <string.h>

#ifdef _win32_
    #include "winint.h"
    #include <util/folder/dirut.h>
#endif

#include <util/random/random.h>
#include "sysstat.h"

static const unsigned char padchar[] =
    "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz";

static int
GetTemp(char* path, int* doopen, int domkdir, int slen)
{
    char *start, *trv, *suffp;
    char* pad;
#ifndef _win32_
    struct stat sbuf;
    int rval;
#endif
    ui32 rand;

    if (doopen != nullptr && domkdir) {
        errno = EINVAL;
        return (0);
    }

    trv = path;
    while (*trv != 0) {
        ++trv;
    }
    trv -= slen;
    suffp = trv;
    --trv;
    if (trv < path) {
        errno = EINVAL;
        return (0);
    }

    /* Fill space with random characters */
    while (trv >= path && *trv == 'X') {
        rand = (RandomNumber<ui32>()) % (sizeof(padchar) - 1);
        *trv-- = padchar[rand];
    }
    start = trv + 1;

    /*
         * check the target directory.
         */
    if (doopen != nullptr || domkdir) {
        for (; trv > path; --trv) {
            if (*trv == '/') {
                *trv = '\0';
#ifdef _win32_
                ui32 attr = ::GetFileAttributesA(path);
                *trv = '/';
                if (attr == 0xFFFFFFFF)
                    return (0);
                if (!(attr & FILE_ATTRIBUTE_DIRECTORY)) {
                    errno = ENOTDIR;
                    return (0);
                }
#else
                rval = stat(path, &sbuf);
                *trv = '/';
                if (rval != 0) {
                    return (0);
                }
                if (!S_ISDIR(sbuf.st_mode)) {
                    errno = ENOTDIR;
                    return (0);
                }
#endif
                break;
            }
        }
    }

    for (;;) {
        if (doopen) {
            if ((*doopen =
                     open(path, O_CREAT | O_EXCL | O_RDWR, 0600)) >= 0) {
                return (1);
            }
            if (errno != EEXIST) {
                return (0);
            }
        } else if (domkdir) {
            if (Mkdir(path, S_IRWXU) == 0) {
                return (1);
            }
            if (errno != EEXIST) {
                return (0);
            }
        } else
#ifdef _win32_
            if (::GetFileAttributesA(path) == INVALID_FILE_ATTRIBUTES)
            return (errno == ENOENT);
#else
            if (lstat(path, &sbuf)) {
            return (errno == ENOENT);
        }
#endif
        /* If we have a collision, cycle through the space of filenames */
        for (trv = start;;) {
            if (*trv == '\0' || trv == suffp) {
                return (0);
            }
            pad = strchr((char*)padchar, *trv);
            if (pad == nullptr || *++pad == '\0') {
                *trv++ = padchar[0];
            } else {
                *trv++ = *pad;
                break;
            }
        }
    }
    /*NOTREACHED*/
}

extern "C" int mkstemps(char* path, int slen) {
    int fd;

    return (GetTemp(path, &fd, 0, slen) ? fd : -1);
}

#if defined(_win_)
char* mkdtemp(char* path) {
    return (GetTemp(path, (int*)nullptr, 1, 0) ? path : (char*)nullptr);
}
#endif
