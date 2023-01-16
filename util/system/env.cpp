#include "env.h"

#include <util/generic/string.h>
#include <util/generic/yexception.h>

#ifdef _win_
    #include <util/generic/vector.h>
    #include "winint.h"
#else
    #include <cerrno>
    #include <cstdlib>
#endif

/**
 * On Windows there may be many copies of enviroment variables, there at least two known, one is
 * manipulated by Win32 API, another by C runtime, so we must be consistent in the choice of
 * functions used to manipulate them.
 *
 * Relevant links:
 *  - http://bugs.python.org/issue16633
 *  - https://a.yandex-team.ru/review/108892/details
 */

TString GetEnv(const TString& key, const TString& def) {
#ifdef _win_
    size_t len = GetEnvironmentVariableA(key.data(), nullptr, 0);

    if (len == 0) {
        if (GetLastError() == ERROR_ENVVAR_NOT_FOUND) {
            return def;
        }
        return TString{};
    }

    TVector<char> buffer(len);
    size_t bufferSize;
    do {
        bufferSize = buffer.size();
        len = GetEnvironmentVariableA(key.data(), buffer.data(), static_cast<DWORD>(bufferSize));
        if (len > bufferSize) {
            buffer.resize(len);
        }
    } while (len > bufferSize);

    return TString(buffer.data(), len);
#else
    const char* env = getenv(key.data());
    return env ? TString(env) : def;
#endif
}

void SetEnv(const TString& key, const TString& value) {
    bool isOk = false;
    int errorCode = 0;
#ifdef _win_
    isOk = SetEnvironmentVariable(key.data(), value.data());
    if (!isOk) {
        errorCode = GetLastError();
    }
#else
    isOk = (0 == setenv(key.data(), value.data(), true /*replace*/));
    if (!isOk) {
        errorCode = errno;
    }
#endif
    Y_ENSURE_EX(isOk, TSystemError() << "failed to SetEnv with error-code " << errorCode);
}
