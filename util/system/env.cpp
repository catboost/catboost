#include "env.h"

#include <util/generic/string.h>

#ifdef _win_
#include <util/generic/vector.h>
#include "winint.h"
#else
#include <cstdlib>
#endif

/**
 * On Windows there may be many copies of enviroment variables, there at least two known, one is
 * manipulated by Win32 API, another by C runtime, so we must be consistent in the choice of
 * functions used to manipulate them.
 *
 * Relevant links:
 *  - http://bugs.python.org/issue16633
 *  - https://rb.yandex-team.ru/arc/r/108892/
 */

TString GetEnv(const TString& key, const TString& def) {
#ifdef _win_
    size_t len = GetEnvironmentVariableA(~key, nullptr, 0);

    if (len == 0) {
        if (GetLastError() == ERROR_ENVVAR_NOT_FOUND) {
            return def;
        }
        return TString{};
    }

    yvector<char> buffer(len);
    size_t bufferSize;
    do {
        bufferSize = buffer.size();
        len = GetEnvironmentVariableA(~key, buffer.data(), static_cast<DWORD>(bufferSize));
        if (len > bufferSize) {
            buffer.resize(len);
        }
    } while (len > bufferSize);

    return TString(buffer.data(), len);
#else
    const char* env = getenv(~key);
    return env ? TString(env) : def;
#endif
}

void SetEnv(const TString& key, const TString& value) {
#ifdef _win_
    SetEnvironmentVariable(~key, ~value);
#else
    setenv(~key, ~value, true /*replace*/);
#endif
}
