/*-
 * Copyright (c) 1990, 1993, 1994
 *    The Regents of the University of California.  All rights reserved.
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
 *    This product includes software developed by the University of
 *    California, Berkeley and its contributors.
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
 *
 * $OpenBSD: fts.c,v 1.22 1999/10/03 19:22:22 millert Exp $
 */

#include <util/memory/tempbuf.h>
#include <util/system/compat.h>
#include <util/system/compiler.h>
#include <util/system/defaults.h>
#include <util/system/error.h>

#include <stdlib.h>
#ifndef _win_
    #include <inttypes.h>
    #include <sys/param.h>
    #include <dirent.h>
    #include <errno.h>
    #include <string.h>
    #include <unistd.h>
#else
    #include <direct.h>
    #include "dirent_win.h"
    #include "lstat_win.h"
#endif

#include <sys/stat.h>

#include <fcntl.h>

#include "fts.h"
#include <limits.h>

#ifndef _win_

static const dird invalidDirD = -1;

dird get_cwdd() {
    return open(".", O_RDONLY, 0);
}

dird get_dird(char* path) {
    return open(path, O_RDONLY, 0);
}

int valid_dird(dird fd) {
    return fd < 0;
}

void close_dird(dird fd) {
    (void)close(fd);
}

int chdir_dird(dird fd) {
    return fchdir(fd);
}

int cmp_dird(dird fd1, dird fd2) {
    return fd1 - fd2;
}

#else // ndef _win_

int stat64UTF(const char* path, struct _stat64* _Stat) {
    int len_converted = MultiByteToWideChar(CP_UTF8, 0, path, -1, 0, 0);
    if (len_converted == 0) {
        return -1;
    }
    WCHAR* buf = (WCHAR*)malloc(sizeof(WCHAR) * (len_converted));
    if (buf == nullptr) {
        return -1;
    }
    MultiByteToWideChar(CP_UTF8, 0, path, -1, buf, len_converted);

    int ret = _wstat64(buf, _Stat);
    free(buf);
    return ret;
}

int stat64UTF(dird path, struct _stat64* _Stat) {
    return _wstat64(path, _Stat);
}

const dird invalidDirD = nullptr;

dird get_cwdd() {
    return _wgetcwd(nullptr, 0);
}

int valid_dird(dird fd) {
    return fd == nullptr;
}

void close_dird(dird fd) {
    free(fd);
}

int chdir_dird(dird fd) {
    return _wchdir(fd);
}

int chdir_dird(const char* path) {
    int len_converted = MultiByteToWideChar(CP_UTF8, 0, path, -1, 0, 0);
    if (len_converted == 0) {
        return -1;
    }
    WCHAR* buf = (WCHAR*)malloc(sizeof(WCHAR) * (len_converted));
    if (buf == nullptr) {
        return -1;
    }
    MultiByteToWideChar(CP_UTF8, 0, path, -1, buf, len_converted);
    int ret = _wchdir(buf);
    free(buf);
    return ret;
}

int cmp_dird(dird fd1, dird fd2) {
    return lstrcmpW(fd1, fd2);
}

dird get_dird(char* path) {
    int len_converted = MultiByteToWideChar(CP_UTF8, 0, path, -1, 0, 0);
    if (len_converted == 0) {
        return nullptr;
    }
    WCHAR* buf = (WCHAR*)malloc(sizeof(WCHAR) * (len_converted));
    if (buf == nullptr) {
        return nullptr;
    }
    MultiByteToWideChar(CP_UTF8, 0, path, -1, buf, len_converted);

    WCHAR* ret = _wfullpath(0, buf, 0);

    free(buf);
    return ret;
}

#endif // ndef _win_

#ifdef _win_
    #define S_ISDIR(st_mode) ((st_mode & _S_IFMT) == _S_IFDIR)
    #define S_ISREG(st_mode) ((st_mode & _S_IFMT) == _S_IFREG)
    #define S_ISLNK(st_mode) ((st_mode & _S_IFMT) == _S_IFLNK)
    #define O_RDONLY _O_RDONLY
static int fts_safe_changedir(FTS*, FTSENT*, int, dird);
#endif

#if defined(__svr4__) || defined(__linux__) || defined(__CYGWIN__) || defined(_win_) || defined(_emscripten_)
    #ifdef MAX
        #undef MAX
    #endif
    #define MAX(a, b) ((a) > (b) ? (a) : (b))
    #undef ALIGNBYTES
    #undef ALIGN
    #define ALIGNBYTES (sizeof(long long) - 1)
    #define ALIGN(p) (((uintptr_t)(p) + ALIGNBYTES) & ~ALIGNBYTES)
    #if !defined(__linux__) && !defined(__CYGWIN__) && !defined(_emscripten_)
        #define dirfd(dirp) ((dirp)->dd_fd)
    #endif
    #define D_NAMLEN(dirp) (strlen(dirp->d_name))
#else
    #define D_NAMLEN(dirp) (dirp->d_namlen)
#endif

static FTSENT* fts_alloc(FTS*, const char*, int);
static FTSENT* fts_build(FTS*, int);
static void fts_lfree(FTSENT*);
static void fts_load(FTS*, FTSENT*);
static size_t fts_maxarglen(char* const*);
static void fts_padjust(FTS*);
static int fts_palloc(FTS*, size_t);
static FTSENT* fts_sort(FTS*, FTSENT*, int);
static u_short fts_stat(FTS*, FTSENT*, int);
static int fts_safe_changedir(FTS*, FTSENT*, int, const char*);

#define ISDOT(a) (a[0] == '.' && (!a[1] || (a[1] == '.' && !a[2])))

#define CLR(opt) (sp->fts_options &= ~(opt))
#define ISSET(opt) (sp->fts_options & (opt))
#define SET(opt) (sp->fts_options |= (opt))

#define FCHDIR(sp, fd) (!ISSET(FTS_NOCHDIR) && chdir_dird(fd))

/* fts_build flags */
#define BCHILD 1 /* yfts_children */
#define BNAMES 2 /* yfts_children, names only */
#define BREAD 3  /* yfts_read */

static u_short
yfts_type_from_info(u_short info) {
    if (info == FTS_D || info == FTS_DP || info == FTS_DOT) {
        return FTS_D;
    } else if (info == FTS_F) {
        return FTS_F;
    } else if (info == FTS_SL || info == FTS_SLNONE) {
        return FTS_SL;
    }
    return FTS_NSOK;
}

static void*
yreallocf(void* ptr, size_t size)
{
    void* nptr;

    nptr = realloc(ptr, size);
    if (!nptr && ptr) {
        free(ptr);
    }
    return (nptr);
}

FTS* yfts_open(char* const* argv, int options, int (*compar)(const FTSENT**, const FTSENT**))
{
    FTS* sp;
    FTSENT *p, *root;
    int nitems;
    FTSENT *parent, *tmp;
    int len;

    errno = 0;

    Y_ASSERT(argv);
    if (!*argv) {
        errno = ENOENT;
        return nullptr;
    }

    /* Options check. */
    if (options & ~FTS_OPTIONMASK) {
        errno = EINVAL;
        return nullptr;
    }

    /* Allocate/initialize the stream */
    if ((sp = (FTS*)malloc(sizeof(FTS))) == nullptr) {
        return nullptr;
    }
    memset(sp, 0, sizeof(FTS));
    sp->fts_compar = compar;
    sp->fts_options = options;

    /* Shush, GCC. */
    tmp = nullptr;

    /* Logical walks turn on NOCHDIR; symbolic links are too hard. */
    if (ISSET(FTS_LOGICAL)) {
        SET(FTS_NOCHDIR);
    }

    /*
     * Start out with 1K of path space, and enough, in any case,
     * to hold the user's paths.
     */
    if (fts_palloc(sp, MAX(fts_maxarglen(argv), MAXPATHLEN))) {
        goto mem1;
    }

    /* Allocate/initialize root's parent. */
    if ((parent = fts_alloc(sp, "", 0)) == nullptr) {
        goto mem2;
    }
    parent->fts_level = FTS_ROOTPARENTLEVEL;

    /* Allocate/initialize root(s). */
    for (root = nullptr, nitems = 0; *argv; ++argv, ++nitems) {
        /* Don't allow zero-length paths. */

        len = strlen(*argv);

//Any subsequent windows call will expect no trailing slashes so we will remove them here
#ifdef _win_
        while (len && ((*argv)[len - 1] == '\\' || (*argv)[len - 1] == '/')) {
            --len;
        }
#endif

        if (len == 0) {
            errno = ENOENT;
            goto mem3;
        }

        p = fts_alloc(sp, *argv, len);
        p->fts_level = FTS_ROOTLEVEL;
        p->fts_parent = parent;
        p->fts_accpath = p->fts_name;
        p->fts_info = fts_stat(sp, p, ISSET(FTS_COMFOLLOW));
        p->fts_type = yfts_type_from_info(p->fts_info);

        /* Command-line "." and ".." are real directories. */
        if (p->fts_info == FTS_DOT) {
            p->fts_info = FTS_D;
        }

        /*
         * If comparison routine supplied, traverse in sorted
         * order; otherwise traverse in the order specified.
         */
        if (compar) {
            p->fts_link = root;
            root = p;
        } else {
            p->fts_link = nullptr;
            if (root == nullptr) {
                tmp = root = p;
            } else {
                tmp->fts_link = p;
                tmp = p;
            }
        }
    }
    if (compar && nitems > 1) {
        root = fts_sort(sp, root, nitems);
    }

    /*
     * Allocate a dummy pointer and make yfts_read think that we've just
     * finished the node before the root(s); set p->fts_info to FTS_INIT
     * so that everything about the "current" node is ignored.
     */
    if ((sp->fts_cur = fts_alloc(sp, "", 0)) == nullptr) {
        goto mem3;
    }
    sp->fts_cur->fts_level = FTS_ROOTLEVEL;
    sp->fts_cur->fts_link = root;
    sp->fts_cur->fts_info = FTS_INIT;

    /*
     * If using chdir(2), grab a file descriptor pointing to dot to ensure
     * that we can get back here; this could be avoided for some paths,
     * but almost certainly not worth the effort.  Slashes, symbolic links,
     * and ".." are all fairly nasty problems.  Note, if we can't get the
     * descriptor we run anyway, just more slowly.
     */

    if (!ISSET(FTS_NOCHDIR) && valid_dird(sp->fts_rfd = get_cwdd())) {
        SET(FTS_NOCHDIR);
    }

    return (sp);

mem3:
    fts_lfree(root);
    free(parent);
mem2:
    free(sp->fts_path);
mem1:
    free(sp);
    return nullptr;
}

static void
fts_load(FTS* sp, FTSENT* p)
{
    size_t len;
    char* cp;

    /*
     * Load the stream structure for the next traversal.  Since we don't
     * actually enter the directory until after the preorder visit, set
     * the fts_accpath field specially so the chdir gets done to the right
     * place and the user can access the first node.  From yfts_open it's
     * known that the path will fit.
     */
    len = p->fts_pathlen = p->fts_namelen;
    memmove((void*)sp->fts_path, (void*)p->fts_name, len + 1);
    if ((cp = strrchr(p->fts_name, LOCSLASH_C)) != nullptr && (cp != p->fts_name || cp[1])) {
        len = strlen(++cp);
        memmove((void*)p->fts_name, (void*)cp, len + 1);
        p->fts_namelen = (u_short)len;
    }
    p->fts_accpath = p->fts_path = sp->fts_path;
    sp->fts_dev = p->fts_dev;
}

int yfts_close(FTS* sp)
{
    FTSENT *freep, *p;
    int saved_errno;

    /*
     * This still works if we haven't read anything -- the dummy structure
     * points to the root list, so we step through to the end of the root
     * list which has a valid parent pointer.
     */
    if (sp->fts_cur) {
        for (p = sp->fts_cur; p->fts_level >= FTS_ROOTLEVEL;) {
            freep = p;
            p = p->fts_link ? p->fts_link : p->fts_parent;
            free(freep);
        }
        free(p);
    }

    /* Free up child linked list, sort array, path buffer. */
    if (sp->fts_child) {
        fts_lfree(sp->fts_child);
    }
    if (sp->fts_array) {
        free(sp->fts_array);
    }
    free(sp->fts_path);

    /* Return to original directory, save errno if necessary. */
    if (!ISSET(FTS_NOCHDIR)) {
        saved_errno = chdir_dird(sp->fts_rfd) ? errno : 0;
        close_dird(sp->fts_rfd);

        /* Set errno and return. */
        if (saved_errno != 0) {
            /* Free up the stream pointer. */
            free(sp);
            errno = saved_errno;
            return (-1);
        }
    }

    /* Free up the stream pointer. */
    free(sp);
    return (0);
}

/*
 * Special case of "/" at the end of the path so that slashes aren't
 * appended which would cause paths to be written as "....//foo".
 */
#define NAPPEND(p)                                 \
    (p->fts_path[p->fts_pathlen - 1] == LOCSLASH_C \
         ? p->fts_pathlen - 1                      \
         : p->fts_pathlen)

FTSENT*
yfts_read(FTS* sp) {
    FTSENT *p, *tmp;
    int instr;
    char* t;
    int saved_errno;

    ClearLastSystemError();

    /* If finished or unrecoverable error, return NULL. */
    if (sp->fts_cur == nullptr || ISSET(FTS_STOP)) {
        return nullptr;
    }

    /* Set current node pointer. */
    p = sp->fts_cur;

    /* Save and zero out user instructions. */
    instr = p->fts_instr;
    p->fts_instr = FTS_NOINSTR;

    /* Any type of file may be re-visited; re-stat and re-turn. */
    if (instr == FTS_AGAIN) {
        p->fts_info = fts_stat(sp, p, 0);
        p->fts_type = yfts_type_from_info(p->fts_info);
        return (p);
    }

    /*
     * Following a symlink -- SLNONE test allows application to see
     * SLNONE and recover.  If indirecting through a symlink, have
     * keep a pointer to current location.  If unable to get that
     * pointer, follow fails.
     */
    if (instr == FTS_FOLLOW &&
        (p->fts_info == FTS_SL || p->fts_info == FTS_SLNONE)) {
        p->fts_info = fts_stat(sp, p, 1);
        p->fts_type = yfts_type_from_info(p->fts_info);
        if (p->fts_info == FTS_D && !ISSET(FTS_NOCHDIR)) {
            if (valid_dird(p->fts_symfd = get_cwdd())) {
                p->fts_errno = errno;
                p->fts_info = FTS_ERR;
            } else {
                p->fts_flags |= FTS_SYMFOLLOW;
            }
        }
        return (p);
    }

    /* Directory in pre-order. */
    if (p->fts_info == FTS_D) {
        /* If skipped or crossed mount point, do post-order visit. */
        if (instr == FTS_SKIP ||
            (ISSET(FTS_XDEV) && p->fts_dev != sp->fts_dev)) {
            if (p->fts_flags & FTS_SYMFOLLOW) {
                close_dird(p->fts_symfd);
            }
            if (sp->fts_child) {
                fts_lfree(sp->fts_child);
                sp->fts_child = nullptr;
            }
            p->fts_info = FTS_DP;
            return (p);
        }

        /* Rebuild if only read the names and now traversing. */
        if (sp->fts_child && ISSET(FTS_NAMEONLY)) {
            CLR(FTS_NAMEONLY);
            fts_lfree(sp->fts_child);
            sp->fts_child = nullptr;
        }

        /*
         * Cd to the subdirectory.
         *
         * If have already read and now fail to chdir, whack the list
         * to make the names come out right, and set the parent errno
         * so the application will eventually get an error condition.
         * Set the FTS_DONTCHDIR flag so that when we logically change
         * directories back to the parent we don't do a chdir.
         *
         * If haven't read do so.  If the read fails, fts_build sets
         * FTS_STOP or the fts_info field of the node.
         */
        if (sp->fts_child) {
            if (fts_safe_changedir(sp, p, -1, p->fts_accpath)) {
                p->fts_errno = errno;
                p->fts_flags |= FTS_DONTCHDIR;
                for (p = sp->fts_child; p; p = p->fts_link) {
                    p->fts_accpath =
                        p->fts_parent->fts_accpath;
                }
            }
        } else if ((sp->fts_child = fts_build(sp, BREAD)) == nullptr) {
            if (ISSET(FTS_STOP)) {
                return nullptr;
            }
            return (p);
        }
        p = sp->fts_child;
        sp->fts_child = nullptr;
        goto name;
    }

    /* Move to the next node on this level. */
next:
    tmp = p;
    if ((p = p->fts_link) != nullptr) {
        free(tmp);

        /*
         * If reached the top, return to the original directory (or
         * the root of the tree), and load the paths for the next root.
         */
        if (p->fts_level == FTS_ROOTLEVEL) {
            if (FCHDIR(sp, sp->fts_rfd)) {
                SET(FTS_STOP);
                return nullptr;
            }
            fts_load(sp, p);
            return (sp->fts_cur = p);
        }

        /*
         * User may have called yfts_set on the node.  If skipped,
         * ignore.  If followed, get a file descriptor so we can
         * get back if necessary.
         */
        if (p->fts_instr == FTS_SKIP) {
            goto next;
        }
        if (p->fts_instr == FTS_FOLLOW) {
            p->fts_info = fts_stat(sp, p, 1);
            p->fts_type = yfts_type_from_info(p->fts_info);
            if (p->fts_info == FTS_D && !ISSET(FTS_NOCHDIR)) {
                if (valid_dird(p->fts_symfd =
                                   get_cwdd())) {
                    p->fts_errno = errno;
                    p->fts_info = FTS_ERR;
                } else {
                    p->fts_flags |= FTS_SYMFOLLOW;
                }
            }
            p->fts_instr = FTS_NOINSTR;
        }

    name:
        t = sp->fts_path + NAPPEND(p->fts_parent);
        *t++ = LOCSLASH_C;
        memmove(t, p->fts_name, (size_t)p->fts_namelen + 1);
        return (sp->fts_cur = p);
    }

    /* Move up to the parent node. */
    p = tmp->fts_parent;
    free(tmp);

    if (p->fts_level == FTS_ROOTPARENTLEVEL) {
        /*
         * Done; free everything up and set errno to 0 so the user
         * can distinguish between error and EOF.
         */
        free(p);
        errno = 0;
        return (sp->fts_cur = nullptr);
    }

    /* NUL terminate the pathname. */
    sp->fts_path[p->fts_pathlen] = '\0';

    /*
     * Return to the parent directory.  If at a root node or came through
     * a symlink, go back through the file descriptor.  Otherwise, cd up
     * one directory.
     */
    if (p->fts_level == FTS_ROOTLEVEL) {
        if (FCHDIR(sp, sp->fts_rfd)) {
            SET(FTS_STOP);
            return nullptr;
        }
    } else if (p->fts_flags & FTS_SYMFOLLOW) {
        if (FCHDIR(sp, p->fts_symfd)) {
            saved_errno = errno;
            close_dird(p->fts_symfd);
            errno = saved_errno;
            SET(FTS_STOP);
            return nullptr;
        }
        close_dird(p->fts_symfd);
    } else if (!(p->fts_flags & FTS_DONTCHDIR) &&
               fts_safe_changedir(sp, p->fts_parent, -1, "..")) {
        SET(FTS_STOP);
        return nullptr;
    }
    p->fts_info = p->fts_errno ? FTS_ERR : FTS_DP;
    return (sp->fts_cur = p);
}

/*
 * Fts_set takes the stream as an argument although it's not used in this
 * implementation; it would be necessary if anyone wanted to add global
 * semantics to fts using yfts_set.  An error return is allowed for similar
 * reasons.
 */
/* ARGSUSED */
int yfts_set(FTS* sp, FTSENT* p, int instr)
{
    (void)sp; //Unused
    if (instr && instr != FTS_AGAIN && instr != FTS_FOLLOW &&
        instr != FTS_NOINSTR && instr != FTS_SKIP) {
        errno = EINVAL;
        return (1);
    }
    p->fts_instr = (u_short)instr;
    return (0);
}

FTSENT*
yfts_children(FTS* sp, int instr)
{
    FTSENT* p;
    dird fd;
    if (instr && instr != FTS_NAMEONLY) {
        errno = EINVAL;
        return nullptr;
    }

    /* Set current node pointer. */
    p = sp->fts_cur;

    /*
     * Errno set to 0 so user can distinguish empty directory from
     * an error.
     */
    errno = 0;

    /* Fatal errors stop here. */
    if (ISSET(FTS_STOP)) {
        return nullptr;
    }

    /* Return logical hierarchy of user's arguments. */
    if (p->fts_info == FTS_INIT) {
        return (p->fts_link);
    }

    /*
     * If not a directory being visited in pre-order, stop here.  Could
     * allow FTS_DNR, assuming the user has fixed the problem, but the
     * same effect is available with FTS_AGAIN.
     */
    if (p->fts_info != FTS_D /* && p->fts_info != FTS_DNR */) {
        return nullptr;
    }

    /* Free up any previous child list. */
    if (sp->fts_child) {
        fts_lfree(sp->fts_child);
    }

    if (instr == FTS_NAMEONLY) {
        SET(FTS_NAMEONLY);
        instr = BNAMES;
    } else {
        instr = BCHILD;
    }

    /*
     * If using chdir on a relative path and called BEFORE yfts_read does
     * its chdir to the root of a traversal, we can lose -- we need to
     * chdir into the subdirectory, and we don't know where the current
     * directory is, so we can't get back so that the upcoming chdir by
     * yfts_read will work.
     */
    if (p->fts_level != FTS_ROOTLEVEL || p->fts_accpath[0] == LOCSLASH_C ||
        ISSET(FTS_NOCHDIR)) {
        return (sp->fts_child = fts_build(sp, instr));
    }

    if (valid_dird(fd = get_cwdd())) {
        return nullptr;
    }
    sp->fts_child = fts_build(sp, instr);
    if (chdir_dird(fd)) {
        close_dird(fd);
        return nullptr;
    }
    close_dird(fd);
    return (sp->fts_child);
}

static inline struct dirent* yreaddir(DIR* dir, struct dirent* de) {
    // TODO(yazevnul|IGNIETFERRO-1070): remove these macroses by replacing `readdir_r` with proper
    // alternative
    Y_PRAGMA_DIAGNOSTIC_PUSH
    Y_PRAGMA_NO_DEPRECATED
    if (readdir_r(dir, de, &de) == 0) {
        Y_PRAGMA_DIAGNOSTIC_POP
        return de;
    }

    return nullptr;
}

/*
 * This is the tricky part -- do not casually change *anything* in here.  The
 * idea is to build the linked list of entries that are used by yfts_children
 * and yfts_read.  There are lots of special cases.
 *
 * The real slowdown in walking the tree is the stat calls.  If FTS_NOSTAT is
 * set and it's a physical walk (so that symbolic links can't be directories),
 * we can do things quickly.  First, if it's a 4.4BSD file system, the type
 * of the file is in the directory entry.  Otherwise, we assume that the number
 * of subdirectories in a node is equal to the number of links to the parent.
 * The former skips all stat calls.  The latter skips stat calls in any leaf
 * directories and for any files after the subdirectories in the directory have
 * been found, cutting the stat calls by about 2/3.
 */
static FTSENT*
fts_build(FTS* sp, int type)
{
    struct dirent* dp;
    FTSENT *p, *head;
    int nitems;
    FTSENT *cur, *tail;

#ifdef _win_
    dird dirpd;
    struct DIR* dirp;
#else
    DIR* dirp;
#endif

    void* oldaddr;
    int cderrno, descend, len, level, maxlen, nlinks, saved_errno,
        nostat, doadjust;
    char* cp;

    /* Set current node pointer. */
    cur = sp->fts_cur;

    /*
     * Open the directory for reading.  If this fails, we're done.
     * If being called from yfts_read, set the fts_info field.
     */
#ifdef FTS_WHITEOUT
    if (ISSET(FTS_WHITEOUT))
        oflag = DTF_NODUP | DTF_REWIND;
    else
        oflag = DTF_HIDEW | DTF_NODUP | DTF_REWIND;
#else
    #define __opendir2(path, flag) opendir(path)
#endif
    if ((dirp = __opendir2(cur->fts_accpath, oflag)) == nullptr) {
        if (type == BREAD) {
            cur->fts_info = FTS_DNR;
            cur->fts_errno = errno;
        }
        return nullptr;
    }

#ifdef _win_
    dirpd = get_dird(cur->fts_accpath);
#endif

    /*
     * Nlinks is the number of possible entries of type directory in the
     * directory if we're cheating on stat calls, 0 if we're not doing
     * any stat calls at all, -1 if we're doing stats on everything.
     */
    if (type == BNAMES) {
        nlinks = 0;
        /* Be quiet about nostat, GCC. */
        nostat = 0;
    } else if (ISSET(FTS_NOSTAT) && ISSET(FTS_PHYSICAL)) {
        nlinks = cur->fts_nlink - (ISSET(FTS_SEEDOT) ? 0 : 2);
        nostat = 1;
    } else {
        nlinks = -1;
        nostat = 0;
    }

    /*
     * If we're going to need to stat anything or we want to descend
     * and stay in the directory, chdir.  If this fails we keep going,
     * but set a flag so we don't chdir after the post-order visit.
     * We won't be able to stat anything, but we can still return the
     * names themselves.  Note, that since yfts_read won't be able to
     * chdir into the directory, it will have to return different path
     * names than before, i.e. "a/b" instead of "b".  Since the node
     * has already been visited in pre-order, have to wait until the
     * post-order visit to return the error.  There is a special case
     * here, if there was nothing to stat then it's not an error to
     * not be able to stat.  This is all fairly nasty.  If a program
     * needed sorted entries or stat information, they had better be
     * checking FTS_NS on the returned nodes.
     */
    cderrno = 0;
    if (nlinks || type == BREAD) {
#ifndef _win_
        if (fts_safe_changedir(sp, cur, dirfd(dirp), nullptr)) {
#else
        if (fts_safe_changedir(sp, cur, -1, dirpd)) {
#endif

            if (nlinks && type == BREAD) {
                cur->fts_errno = errno;
            }
            cur->fts_flags |= FTS_DONTCHDIR;
            descend = 0;
            cderrno = errno;
            (void)closedir(dirp);
            dirp = nullptr;
#ifdef _win_
            close_dird(dirpd);
            dirpd = invalidDirD;
#else
            Y_UNUSED(invalidDirD);
#endif
        } else {
            descend = 1;
        }
    } else {
        descend = 0;
    }

    /*
     * Figure out the max file name length that can be stored in the
     * current path -- the inner loop allocates more path as necessary.
     * We really wouldn't have to do the maxlen calculations here, we
     * could do them in yfts_read before returning the path, but it's a
     * lot easier here since the length is part of the dirent structure.
     *
     * If not changing directories set a pointer so that can just append
     * each new name into the path.
     */
    len = NAPPEND(cur);
    if (ISSET(FTS_NOCHDIR)) {
        cp = sp->fts_path + len;
        *cp++ = LOCSLASH_C;
    } else {
        /* GCC, you're too verbose. */
        cp = nullptr;
    }
    ++len;
    maxlen = sp->fts_pathlen - len;

    level = cur->fts_level + 1;

    /* Read the directory, attaching each entry to the `link' pointer. */
    doadjust = 0;

    //to ensure enough buffer
    TTempBuf dpe;

    for (head = tail = nullptr, nitems = 0; dirp && (dp = yreaddir(dirp, (struct dirent*)dpe.Data())) != nullptr;) {
        if (!ISSET(FTS_SEEDOT) && ISDOT(dp->d_name)) {
            continue;
        }

        if ((p = fts_alloc(sp, dp->d_name, (int)strlen(dp->d_name))) == nullptr) {
            goto mem1;
        }
        if (strlen(dp->d_name) >= (size_t)maxlen) { /* include space for NUL */
            oldaddr = sp->fts_path;
            if (fts_palloc(sp, strlen(dp->d_name) + len + 1)) {
                /*
                 * No more memory for path or structures.  Save
                 * errno, free up the current structure and the
                 * structures already allocated.
                 */
            mem1:
                saved_errno = errno;
                if (p) {
                    free(p);
                }
                fts_lfree(head);
                (void)closedir(dirp);
#ifdef _win_
                close_dird(dirpd);
#endif
                cur->fts_info = FTS_ERR;
                SET(FTS_STOP);
                errno = saved_errno;
                return nullptr;
            }
            /* Did realloc() change the pointer? */
            if (oldaddr != sp->fts_path) {
                doadjust = 1;
                if (ISSET(FTS_NOCHDIR)) {
                    cp = sp->fts_path + len;
                }
            }
            maxlen = sp->fts_pathlen - len;
        }

        if (len + strlen(dp->d_name) >= USHRT_MAX) {
            /*
             * In an FTSENT, fts_pathlen is a u_short so it is
             * possible to wraparound here.  If we do, free up
             * the current structure and the structures already
             * allocated, then error out with ENAMETOOLONG.
             */
            free(p);
            fts_lfree(head);
            (void)closedir(dirp);
#ifdef _win_
            close_dird(dirpd);
#endif
            cur->fts_info = FTS_ERR;
            SET(FTS_STOP);
            errno = ENAMETOOLONG;
            return nullptr;
        }
        p->fts_level = (short)level;
        p->fts_parent = sp->fts_cur;
        p->fts_pathlen = u_short(len + strlen(dp->d_name));

#ifdef FTS_WHITEOUT
        if (dp->d_type == DT_WHT)
            p->fts_flags |= FTS_ISW;
#endif

#ifdef _DIRENT_HAVE_D_TYPE
        if (dp->d_type == DT_DIR) {
            p->fts_type = FTS_D;
        } else if (dp->d_type == DT_REG) {
            p->fts_type = FTS_F;
        } else if (dp->d_type == DT_LNK) {
            p->fts_type = FTS_SL;
        }
#endif

        // coverity[dead_error_line]: false positive
        if (cderrno) {
            if (nlinks) {
                p->fts_info = FTS_NS;
                p->fts_errno = cderrno;
            } else {
                p->fts_info = FTS_NSOK;
            }
            p->fts_accpath = cur->fts_accpath;
        } else if (nlinks == 0
#ifdef DT_DIR
                   || (nostat &&
                       dp->d_type != DT_DIR && dp->d_type != DT_UNKNOWN)
#endif
        ) {
            p->fts_accpath =
                ISSET(FTS_NOCHDIR) ? p->fts_path : p->fts_name;
            p->fts_info = FTS_NSOK;
        } else {
            /* Build a file name for fts_stat to stat. */
            if (ISSET(FTS_NOCHDIR)) {
                p->fts_accpath = p->fts_path;
                memmove((void*)cp, (void*)p->fts_name, (size_t)p->fts_namelen + 1);
            } else {
                p->fts_accpath = p->fts_name;
            }
            /* Stat it. */
            p->fts_info = fts_stat(sp, p, 0);
            p->fts_type = yfts_type_from_info(p->fts_info);

            /* Decrement link count if applicable. */
            if (nlinks > 0 && (p->fts_info == FTS_D ||
                               p->fts_info == FTS_DC || p->fts_info == FTS_DOT)) {
                --nlinks;
            }
        }

        /* We walk in directory order so "ls -f" doesn't get upset. */
        p->fts_link = nullptr;
        if (head == nullptr) {
            head = tail = p;
        } else {
            tail->fts_link = p;
            tail = p;
        }
        ++nitems;
    }
    if (dirp) {
        (void)closedir(dirp);
#ifdef _win_
        close_dird(dirpd);
#endif
    }

    /*
     * If realloc() changed the address of the path, adjust the
     * addresses for the rest of the tree and the dir list.
     */
    if (doadjust) {
        fts_padjust(sp);
    }

    /*
     * If not changing directories, reset the path back to original
     * state.
     */
    if (ISSET(FTS_NOCHDIR)) {
        if (len == sp->fts_pathlen || nitems == 0) {
            --cp;
        }
        *cp = '\0';
    }

    /*
     * If descended after called from yfts_children or after called from
     * yfts_read and nothing found, get back.  At the root level we use
     * the saved fd; if one of yfts_open()'s arguments is a relative path
     * to an empty directory, we wind up here with no other way back.  If
     * can't get back, we're done.
     */
    if (descend && (type == BCHILD || !nitems) &&
        (cur->fts_level == FTS_ROOTLEVEL ? FCHDIR(sp, sp->fts_rfd) : fts_safe_changedir(sp, cur->fts_parent, -1, ".."))) {
        cur->fts_info = FTS_ERR;
        SET(FTS_STOP);
        fts_lfree(head);
        return nullptr;
    }

    /* If didn't find anything, return NULL. */
    if (!nitems) {
        if (type == BREAD) {
            cur->fts_info = FTS_DP;
        }
        fts_lfree(head);
        return nullptr;
    }

    /* Sort the entries. */
    if (sp->fts_compar && nitems > 1) {
        head = fts_sort(sp, head, nitems);
    }
    return (head);
}

static u_short
fts_stat(FTS* sp, FTSENT* p, int follow)
{
    dev_t dev;
    ino_t ino;
    stat_struct *sbp, sb;
    int saved_errno;
    /* If user needs stat info, stat buffer already allocated. */
    sbp = ISSET(FTS_NOSTAT) ? &sb : p->fts_statp;

#ifdef FTS_WHITEOUT
    /* check for whiteout */
    if (p->fts_flags & FTS_ISW) {
        if (sbp != &sb) {
            memset(sbp, '\0', sizeof(*sbp));
            sbp->st_mode = S_IFWHT;
        }
        return (FTS_W);
    }
#endif

    /*
     * If doing a logical walk, or application requested FTS_FOLLOW, do
     * a stat(2).  If that fails, check for a non-existent symlink.  If
     * fail, set the errno from the stat call.
     */
    if (ISSET(FTS_LOGICAL) || follow) {
        if (STAT_FUNC(p->fts_accpath, sbp)) {
            saved_errno = errno;
            if (!lstat(p->fts_accpath, sbp)) {
                errno = 0;
                return (FTS_SLNONE);
            }
            p->fts_errno = saved_errno;
            memset(sbp, 0, sizeof(stat_struct));
            return (FTS_NS);
        }
    } else if (lstat(p->fts_accpath, sbp)) {
        p->fts_errno = errno;
        memset(sbp, 0, sizeof(stat_struct));
        return (FTS_NS);
    }

    if (S_ISDIR(sbp->st_mode)) {
        /*
         * Set the device/inode.  Used to find cycles and check for
         * crossing mount points.  Also remember the link count, used
         * in fts_build to limit the number of stat calls.  It is
         * understood that these fields are only referenced if fts_info
         * is set to FTS_D.
         */
        dev = p->fts_dev = sbp->st_dev;
        ino = p->fts_ino = sbp->st_ino;
        p->fts_nlink = sbp->st_nlink;

        const char* fts_name_x = p->fts_name;
        if (ISDOT(fts_name_x)) {
            return (FTS_DOT);
        }

        /*
         * Cycle detection is done by brute force when the directory
         * is first encountered.  If the tree gets deep enough or the
         * number of symbolic links to directories is high enough,
         * something faster might be worthwhile.
         */

        //There is no way to detect symlink or mount cycles on win32

#ifndef _win_
        FTSENT* t;
        for (t = p->fts_parent;
             t->fts_level >= FTS_ROOTLEVEL; t = t->fts_parent) {
            if (ino == t->fts_ino && dev == t->fts_dev) {
                p->fts_cycle = t;
                return (FTS_DC);
            }
        }
#endif /*_win_*/
        return (FTS_D);
    }
    if (S_ISLNK(sbp->st_mode)) {
        return (FTS_SL);
    }
    if (S_ISREG(sbp->st_mode)) {
        return (FTS_F);
    }
    return (FTS_DEFAULT);
}

static FTSENT*
fts_sort(FTS* sp, FTSENT* head, int nitems)
{
    FTSENT **ap, *p;

    /*
     * Construct an array of pointers to the structures and call qsort(3).
     * Reassemble the array in the order returned by qsort.  If unable to
     * sort for memory reasons, return the directory entries in their
     * current order.  Allocate enough space for the current needs plus
     * 40 so don't realloc one entry at a time.
     */
    if (nitems > sp->fts_nitems) {
        struct _ftsent** a;

        sp->fts_nitems = nitems + 40;
        if ((a = (struct _ftsent**)realloc(sp->fts_array,
                                           sp->fts_nitems * sizeof(FTSENT*))) == nullptr) {
            if (sp->fts_array) {
                free(sp->fts_array);
            }
            sp->fts_array = nullptr;
            sp->fts_nitems = 0;
            return (head);
        }
        sp->fts_array = a;
    }
    for (ap = sp->fts_array, p = head; p; p = p->fts_link) {
        *ap++ = p;
    }
    qsort((void*)sp->fts_array, (size_t)nitems, sizeof(FTSENT*), (int (*)(const void*, const void*))sp->fts_compar);
    for (head = *(ap = sp->fts_array); --nitems; ++ap) {
        ap[0]->fts_link = ap[1];
    }
    ap[0]->fts_link = nullptr;
    return (head);
}

static FTSENT*
fts_alloc(FTS* sp, const char* name, int namelen)
{
    FTSENT* p;
    size_t len;

    /*
     * The file name is a variable length array and no stat structure is
     * necessary if the user has set the nostat bit.  Allocate the FTSENT
     * structure, the file name and the stat structure in one chunk, but
     * be careful that the stat structure is reasonably aligned.  Since the
     * fts_name field is declared to be of size 1, the fts_name pointer is
     * namelen + 2 before the first possible address of the stat structure.
     */
    len = sizeof(FTSENT) + namelen;
    if (!ISSET(FTS_NOSTAT)) {
        len += sizeof(stat_struct) + ALIGNBYTES;
    }
    if ((p = (FTSENT*)malloc(len)) == nullptr) {
        return nullptr;
    }

    /* Copy the name and guarantee NUL termination. */
    memmove((void*)p->fts_name, (void*)name, (size_t)namelen);
    p->fts_name[namelen] = '\0';

    if (!ISSET(FTS_NOSTAT)) {
        p->fts_statp = (stat_struct*)ALIGN(p->fts_name + namelen + 2);
    } else {
        p->fts_statp = nullptr;
    }
    p->fts_namelen = (u_short)namelen;
    p->fts_path = sp->fts_path;
    p->fts_errno = 0;
    p->fts_flags = 0;
    p->fts_instr = FTS_NOINSTR;
    p->fts_number = 0;
    p->fts_pointer = nullptr;
    p->fts_type = FTS_NSOK;
    return (p);
}

static void
fts_lfree(FTSENT* head)
{
    FTSENT* p;

    /* Free a linked list of structures. */
    while ((p = head) != nullptr) {
        head = head->fts_link;
        free(p);
    }
}

/*
 * Allow essentially unlimited paths; find, rm, ls should all work on any tree.
 * Most systems will allow creation of paths much longer than MAXPATHLEN, even
 * though the kernel won't resolve them.  Add the size (not just what's needed)
 * plus 256 bytes so don't realloc the path 2 bytes at a time.
 */
static int
fts_palloc(FTS* sp, size_t more)
{
    sp->fts_pathlen += more + 256;
    sp->fts_path = (char*)yreallocf(sp->fts_path, (size_t)sp->fts_pathlen);
    return (sp->fts_path == nullptr);
}

static void
ADJUST(FTSENT* p, void* addr)
{
    if ((p)->fts_accpath >= (p)->fts_path &&
        (p)->fts_accpath < (p)->fts_path + (p)->fts_pathlen) {
        if (p->fts_accpath != p->fts_path) {
            errx(1, "fts ADJUST: accpath %p path %p",
                 p->fts_accpath, p->fts_path);
        }
        if (p->fts_level != 0) {
            errx(1, "fts ADJUST: level %d not 0", p->fts_level);
        }
        (p)->fts_accpath =
            (char*)addr + ((p)->fts_accpath - (p)->fts_path);
    }
    (p)->fts_path = (char*)addr;
}

/*
 * When the path is realloc'd, have to fix all of the pointers in structures
 * already returned.
 */
static void
fts_padjust(FTS* sp)
{
    FTSENT* p;
    char* addr = sp->fts_path;

#define ADJUST1(p)                             \
    {                                          \
        if ((p)->fts_accpath == (p)->fts_path) \
            (p)->fts_accpath = (addr);         \
        (p)->fts_path = addr;                  \
    }
    /* Adjust the current set of children. */
    for (p = sp->fts_child; p; p = p->fts_link) {
        ADJUST(p, addr);
    }

    /* Adjust the rest of the tree. */
    for (p = sp->fts_cur; p->fts_level >= FTS_ROOTLEVEL;) {
        ADJUST(p, addr);
        p = p->fts_link ? p->fts_link : p->fts_parent;
    }
}

static size_t
fts_maxarglen(char* const* argv)
{
    size_t len, max;

    for (max = 0; *argv; ++argv) {
        if ((len = strlen(*argv)) > max) {
            max = len;
        }
    }
    return (max + 1);
}

/*
 * Change to dir specified by fd or p->fts_accpath without getting
 * tricked by someone changing the world out from underneath us.
 * Assumes p->fts_dev and p->fts_ino are filled in.
 */

#ifndef _win_
static int
fts_safe_changedir(FTS* sp, FTSENT* p, int fd, const char* path)
{
    int ret, oerrno, newfd;
    stat_struct sb;

    newfd = fd;
    if (ISSET(FTS_NOCHDIR)) {
        return (0);
    }
    if (fd < 0 && (newfd = open(path, O_RDONLY, 0)) < 0) {
        return (-1);
    }
    if (fstat(newfd, &sb)) {
        ret = -1;
        goto bail;
    }
    if (p->fts_dev != sb.st_dev || p->fts_ino != sb.st_ino) {
        errno = ENOENT; /* disinformation */
        ret = -1;
        goto bail;
    }
    ret = fchdir(newfd);
bail:
    oerrno = errno;
    if (fd < 0) {
        (void)close(newfd);
    }
    errno = oerrno;
    return (ret);
}
#else
static int
fts_safe_changedir(FTS* sp, FTSENT* p, int /*fd*/, const char* path)
{
    int ret;
    stat_struct sb;

    if (ISSET(FTS_NOCHDIR))
        return (0);
    if (STAT_FUNC(path, &sb)) {
        ret = -1;
        goto bail;
    }
    if (p->fts_dev != sb.st_dev) {
        errno = ENOENT; /* disinformation */
        ret = -1;
        goto bail;
    }
    ret = chdir_dird(path);
bail:
    return (ret);
}

static int
fts_safe_changedir(FTS* sp, FTSENT* p, int /*fd*/, dird path) {
    int ret;
    stat_struct sb;

    if (ISSET(FTS_NOCHDIR))
        return (0);
    if (STAT_FUNC(path, &sb)) {
        ret = -1;
        goto bail;
    }
    if (p->fts_dev != sb.st_dev) {
        errno = ENOENT; /* disinformation */
        ret = -1;
        goto bail;
    }
    ret = chdir_dird(path);
bail:
    return (ret);
}
#endif
