#include "platform.h"

#if defined(_solaris_)
    #include <stdlib.h>
#elif defined(_darwin_)
    #include <mach-o/dyld.h>
    #include <util/generic/function.h>
#elif defined(_win_)
    #include "winint.h"
    #include <io.h>
#elif defined(_linux_)
#elif defined(_freebsd_)
    #include <string.h>
    #include <sys/types.h> // for u_int not defined in sysctl.h
    #include <sys/sysctl.h>
    #include <unistd.h>
#endif

#include <util/generic/singleton.h>

#include "execpath.h"
#include "fs.h"

#if defined(_freebsd_)
static inline bool GoodPath(const TString& path) {
    return path.find('/') != TString::npos;
}

static inline int FreeBSDSysCtl(int* mib, size_t mibSize, TTempBuf& res) {
    for (size_t i = 0; i < 2; ++i) {
        size_t cb = res.Size();
        if (sysctl(mib, mibSize, res.Data(), &cb, nullptr, 0) == 0) {
            res.Proceed(cb);
            return 0;
        } else if (errno == ENOMEM) {
            res = TTempBuf(cb);
        } else {
            return errno;
        }
    }
    return errno;
}

static inline TString FreeBSDGetExecPath() {
    int mib[] = {CTL_KERN, KERN_PROC, KERN_PROC_PATHNAME, -1};
    TTempBuf buf;
    int r = FreeBSDSysCtl(mib, Y_ARRAY_SIZE(mib), buf);
    if (r == 0) {
        return TString(buf.Data(), buf.Filled() - 1);
    } else if (r == ENOTSUP) { // older FreeBSD version
        /*
         * BSD analogue for /proc/self is /proc/curproc.
         * See:
         * https://www.freebsd.org/cgi/man.cgi?query=procfs&sektion=5&format=html
         */
        TString path("/proc/curproc/file");
        return NFs::ReadLink(path);
    } else {
        return TString();
    }
}

static inline TString FreeBSDGetArgv0() {
    int mib[] = {CTL_KERN, KERN_PROC, KERN_PROC_ARGS, getpid()};
    TTempBuf buf;
    int r = FreeBSDSysCtl(mib, Y_ARRAY_SIZE(mib), buf);
    if (r == 0) {
        return TString(buf.Data());
    } else if (r == ENOTSUP) {
        return TString();
    } else {
        ythrow yexception() << "FreeBSDGetArgv0() failed: " << LastSystemErrorText();
    }
}

static inline bool FreeBSDGuessExecPath(const TString& guessPath, TString& execPath) {
    if (NFs::Exists(guessPath)) {
        // now it should work for real
        execPath = FreeBSDGetExecPath();
        if (RealPath(execPath) == RealPath(guessPath)) {
            return true;
        }
    }
    return false;
}

static inline bool FreeBSDGuessExecBasePath(const TString& guessBasePath, TString& execPath) {
    return FreeBSDGuessExecPath(TString(guessBasePath) + "/" + getprogname(), execPath);
}

#endif

static TString GetExecPathImpl() {
#if defined(_solaris_)
    return execname();
#elif defined(_darwin_)
    TTempBuf execNameBuf;
    for (size_t i = 0; i < 2; ++i) {
        std::remove_pointer_t<TFunctionArg<decltype(_NSGetExecutablePath), 1>> bufsize = execNameBuf.Size();
        int r = _NSGetExecutablePath(execNameBuf.Data(), &bufsize);
        if (r == 0) {
            return execNameBuf.Data();
        } else if (r == -1) {
            execNameBuf = TTempBuf(bufsize);
        }
    }
    ythrow yexception() << "GetExecPathImpl() failed";
#elif defined(_win_)
    TTempBuf execNameBuf;
    for (;;) {
        DWORD r = GetModuleFileName(nullptr, execNameBuf.Data(), execNameBuf.Size());
        if (r == execNameBuf.Size()) {
            execNameBuf = TTempBuf(execNameBuf.Size() * 2);
        } else if (r == 0) {
            ythrow yexception() << "GetExecPathImpl() failed: " << LastSystemErrorText();
        } else {
            return execNameBuf.Data();
        }
    }
#elif defined(_linux_) || defined(_cygwin_)
    TString path("/proc/self/exe");
    return NFs::ReadLink(path);
// TODO(yoda): check if the filename ends with " (deleted)"
#elif defined(_freebsd_)
    TString execPath = FreeBSDGetExecPath();
    if (GoodPath(execPath)) {
        return execPath;
    }
    if (FreeBSDGuessExecPath(FreeBSDGetArgv0(), execPath)) {
        return execPath;
    }
    if (FreeBSDGuessExecPath(getenv("_"), execPath)) {
        return execPath;
    }
    if (FreeBSDGuessExecBasePath(getenv("PWD"), execPath)) {
        return execPath;
    }
    if (FreeBSDGuessExecBasePath(NFs::CurrentWorkingDirectory(), execPath)) {
        return execPath;
    }

    ythrow yexception() << "can not resolve exec path";
#else
    #error dont know how to implement GetExecPath on this platform
#endif
}

static bool GetPersistentExecPathImpl(TString& to) {
#if defined(_solaris_)
    to = TString("/proc/self/object/a.out");
    return true;
#elif defined(_linux_) || defined(_cygwin_)
    to = TString("/proc/self/exe");
    return true;
#elif defined(_freebsd_)
    to = TString("/proc/curproc/file");
    return true;
#else // defined(_win_) || defined(_darwin_)  or unknown
    Y_UNUSED(to);
    return false;
#endif
}

namespace {
    struct TExecPathsHolder {
        inline TExecPathsHolder() {
            ExecPath = GetExecPathImpl();

            if (!GetPersistentExecPathImpl(PersistentExecPath)) {
                PersistentExecPath = ExecPath;
            }
        }

        static inline auto Instance() {
            return SingletonWithPriority<TExecPathsHolder, 1>();
        }

        TString ExecPath;
        TString PersistentExecPath;
    };
}

const TString& GetExecPath() {
    return TExecPathsHolder::Instance()->ExecPath;
}

const TString& GetPersistentExecPath() {
    return TExecPathsHolder::Instance()->PersistentExecPath;
}
