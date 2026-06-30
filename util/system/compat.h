#pragma once

#include "defaults.h"

#include <cstdarg>

#include <csignal>

#if defined(_unix_)
    #include <unistd.h>
#endif

#if defined(_win_)
    #include <process.h>
#endif

extern "C" {
#if defined(_win_)
    using pid_t = int;

    inline unsigned int alarm(unsigned int /*seconds*/) {
        return 0; // no alarm is currently set :)
    }

    #define SIGQUIT SIGBREAK // instead of 3
    #define SIGKILL SIGTERM  // instead of 9
    #define SIGPIPE 13       // will not receive under win?
    #define SIGALRM 14       // will not receive under win?
#endif

#if defined(__FreeBSD__) || defined(_darwin_)
    #define HAVE_NATIVE_GETPROGNAME
#endif

#ifndef HAVE_NATIVE_GETPROGNAME
    const char* getprogname();
#endif

#if defined(_MSC_VER)
    void warn(const char* m, ...);
    void warnx(const char* m, ...);
    void vwarnx(const char* format, va_list ap);
    void vwarn(const char* format, va_list ap);

    [[noreturn]] void err(int e, const char* m, ...);
    [[noreturn]] void errx(int e, const char* m, ...);
    [[noreturn]] void verr(int status, const char* fmt, va_list args);
    [[noreturn]] void verrx(int status, const char* format, va_list ap);
#else
    #include <err.h>
#endif
}

#ifdef _MSC_VER
    #define popen _popen
    #define pclose _pclose
#endif

#ifdef _win_
    #define NAME_MAX FILENAME_MAX
#endif
#ifdef _sun_
    #define NAME_MAX PATH_MAX
#endif

#ifdef _win_

    #ifdef sleep // may be defined by perl
        #undef sleep
    #endif

void sleep(i64 len);
void usleep(i64 len);

#endif

#ifdef _win_
int ftruncate(int fd, i64 length);
int truncate(const char* name, i64 length);
#endif

#if defined(GNUC)
    #ifndef va_copy
        #define va_copy __va_copy
    #endif
#endif
