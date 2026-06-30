#include "../system/utime.h"

#ifdef _MSC_VER
    #include <sys/utime.h>
#else
    #include <utime.h>
#endif

int TouchFile(const char* filePath) {
    return utime(filePath, nullptr);
}

int SetModTime(const char* filePath, time_t modtime, time_t actime) {
    struct utimbuf buf;
    buf.modtime = modtime;
    buf.actime = actime;
    return utime(filePath, &buf);
}
