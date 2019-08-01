// Do not alias __xstat to __xstat64.
#undef _FILE_OFFSET_BITS

#include <dlfcn.h>
#include <sys/stat.h>

int __xstat(int ver, const char* path, struct stat* buf) {
    static int (*xstat)(int, const char*, struct stat*) = 0;
    if (!xstat) {
        xstat = dlsym(RTLD_NEXT, "__xstat");
    }
    int rc = xstat(ver, path, buf);
    if (rc == 0 && ver == _STAT_VER) {
        buf->st_mtime = 0;
    }
    return rc;
}
