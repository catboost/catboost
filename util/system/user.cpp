#include "user.h"
#include "env.h"

#include <util/generic/yexception.h>

#ifdef _win_
    #include "winint.h"
#else
    #include <cerrno>
    #include <pwd.h>
    #include <unistd.h>
#endif

TString GetUsername() {
    for (const auto& var : {"LOGNAME", "USER", "LNAME", "USERNAME"}) {
        TString val = GetEnv(var);
        if (val) {
            return val;
        }
    }

    TTempBuf nameBuf;
    for (;;) {
#if defined(_win_)
        DWORD len = (DWORD)Min(nameBuf.Size(), size_t(32767));
        if (!GetUserNameA(nameBuf.Data(), &len)) {
            DWORD err = GetLastError();
            if ((err == ERROR_INSUFFICIENT_BUFFER) && (nameBuf.Size() <= 32767))
                nameBuf = TTempBuf((size_t)len);
            else
                ythrow TSystemError(err) << " GetUserName failed";
        } else {
            return TString(nameBuf.Data(), (size_t)(len - 1));
        }
#elif defined(_bionic_)
        const passwd* pwd = getpwuid(geteuid());

        if (pwd) {
            return TString(pwd->pw_name);
        }

        ythrow TSystemError() << TStringBuf(" getpwuid failed");
#else
        passwd pwd;
        passwd* tmpPwd;
        int err = getpwuid_r(geteuid(), &pwd, nameBuf.Data(), nameBuf.Size(), &tmpPwd);
        if (err == 0 && tmpPwd) {
            return TString(pwd.pw_name);
        } else if (err == ERANGE) {
            nameBuf = TTempBuf(nameBuf.Size() * 2);
        } else {
            ythrow TSystemError(err) << " getpwuid_r failed";
        }
#endif
    }
}
