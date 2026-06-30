#pragma once

#include <sys/types.h>

#include <util/system/defaults.h>

#ifndef _win32_
typedef int dird;
typedef struct stat stat_struct;
    #define STAT_FUNC stat
#else
    #include <util/folder/dirent_win.h>
typedef WCHAR* dird;
typedef unsigned short u_short;
typedef unsigned int nlink_t;
typedef struct _stat64 stat_struct;
    #define STAT_FUNC stat64UTF
    // TODO: remove from global scope stat64UTF stat64UTF
    #ifdef __cplusplus
int stat64UTF(const char* path, struct _stat64* _Stat);
int stat64UTF(dird path, struct _stat64* _Stat);
    #endif
#endif

typedef struct {
    struct _ftsent* fts_cur;    /* current node */
    struct _ftsent* fts_child;  /* linked list of children */
    struct _ftsent** fts_array; /* sort array */
    dev_t fts_dev;              /* starting device # */
    char* fts_path;             /* path for this descent */
    dird fts_rfd;               /* fd for root */
    int fts_pathlen;            /* sizeof(path) */
    int fts_nitems;             /* elements in the sort array */
    int(*fts_compar)            /* compare function */
        (const struct _ftsent**, const struct _ftsent**);

#define FTS_COMFOLLOW 0x001  /* follow command line symlinks */
#define FTS_LOGICAL 0x002    /* logical walk */
#define FTS_NOCHDIR 0x004    /* don't change directories */
#define FTS_NOSTAT 0x008     /* don't get stat info */
#define FTS_PHYSICAL 0x010   /* physical walk */
#define FTS_SEEDOT 0x020     /* return dot and dot-dot */
#define FTS_XDEV 0x040       /* don't cross devices */
#define FTS_OPTIONMASK 0x0ff /* valid user option mask */

#define FTS_NAMEONLY 0x100 /* (private) child names only */
#define FTS_STOP 0x200     /* (private) unrecoverable error */
    int fts_options;       /* yfts_open options, global flags */
} FTS;

typedef struct _ftsent {
    struct _ftsent* fts_cycle;  /* cycle node */
    struct _ftsent* fts_parent; /* parent directory */
    struct _ftsent* fts_link;   /* next file in directory */
    long fts_number;            /* local numeric value */
    void* fts_pointer;          /* local address value */
    char* fts_accpath;          /* access path */
    char* fts_path;             /* root path */
    int fts_errno;              /* errno for this node */
    dird fts_symfd;             /* fd for symlink */
    u_short fts_pathlen;        /* strlen(fts_path) */
    u_short fts_namelen;        /* strlen(fts_name) */

    ino_t fts_ino;     /* inode */
    dev_t fts_dev;     /* device */
    nlink_t fts_nlink; /* link count */

#define FTS_ROOTPARENTLEVEL -1
#define FTS_ROOTLEVEL 0
    short fts_level; /* depth (-1 to N) */

#define FTS_D 1       /* preorder directory */
#define FTS_DC 2      /* directory that causes cycles */
#define FTS_DEFAULT 3 /* none of the above */
#define FTS_DNR 4     /* unreadable directory */
#define FTS_DOT 5     /* dot or dot-dot */
#define FTS_DP 6      /* postorder directory */
#define FTS_ERR 7     /* error; errno is set */
#define FTS_F 8       /* regular file */
#define FTS_INIT 9    /* initialized only */
#define FTS_NS 10     /* stat(2) failed */
#define FTS_NSOK 11   /* no stat(2) requested */
#define FTS_SL 12     /* symbolic link */
#define FTS_SLNONE 13 /* symbolic link without target */
#define FTS_W 14      /* whiteout object */
    u_short fts_info; /* user flags for FTSENT structure */
    u_short fts_type; /* type of fs node; one of FTS_D, FTS_F, FTS_SL */

#define FTS_DONTCHDIR 0x01 /* don't chdir .. to the parent */
#define FTS_SYMFOLLOW 0x02 /* followed a symlink to get here */
#define FTS_ISW 0x04       /* this is a whiteout object */
    u_short fts_flags;     /* private flags for FTSENT structure */

#define FTS_AGAIN 1    /* read node again */
#define FTS_FOLLOW 2   /* follow symbolic link */
#define FTS_NOINSTR 3  /* no instructions */
#define FTS_SKIP 4     /* discard node */
    u_short fts_instr; /* yfts_set() instructions */

    stat_struct* fts_statp; /* stat(2) information */
    char fts_name[1];       /* file name */
} FTSENT;

FTSENT* yfts_children(FTS*, int);
int yfts_close(FTS*);
FTS* yfts_open(char* const*, int, int (*)(const FTSENT**, const FTSENT**));
FTSENT* yfts_read(FTS*);
int yfts_set(FTS*, FTSENT*, int);
