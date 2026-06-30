#include "env.h"

#include <util/generic/maybe.h>
#include <util/generic/string.h>
#include <util/generic/yexception.h>

#ifdef _win_
    #include <util/generic/scope.h>
    #include <util/generic/vector.h>
    #include <util/system/yassert.h>
    #include "winint.h"
#else
    #ifndef _linux_
        #include <util/generic/vector.h> // ClearEnv impl
    #endif

    #include <cerrno>
    #include <cstdlib>

extern char** environ;
#endif

/**
 * On Windows there may be many copies of environment variables, there at least two known, one is
 * manipulated by Win32 API, another by C runtime, so we must be consistent in the choice of
 * functions used to manipulate them.
 *
 * Relevant links:
 *  - http://bugs.python.org/issue16633
 *  - https://a.yandex-team.ru/review/108892/details
 */

TMaybe<TString> TryGetEnv(const TString& key) {
#ifdef _win_
    size_t len = GetEnvironmentVariableA(key.data(), nullptr, 0);

    if (len == 0) {
        if (GetLastError() == ERROR_ENVVAR_NOT_FOUND) {
            return Nothing();
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
    if (!env) {
        return Nothing();
    }
    return TString(env);
#endif
}

TString GetEnv(const TString& key, const TString& def) {
    TMaybe<TString> value = TryGetEnv(key);
    if (value.Defined()) {
        return *std::move(value);
    }
    return def;
}

void SetEnv(const TString& key, const TString& value) {
    bool isOk = false;
#ifdef _win_
    isOk = SetEnvironmentVariable(key.data(), value.data());
#else
    isOk = (0 == setenv(key.data(), value.data(), true /*replace*/));
#endif
    Y_ENSURE_EX(isOk, TSystemError() << "failed to SetEnv");
}

void UnsetEnv(const TString& key) {
    bool notFound = false;
#ifdef _win_
    bool ok = SetEnvironmentVariable(key.c_str(), NULL);
    notFound = !ok && (GetLastError() == ERROR_ENVVAR_NOT_FOUND);
#else
    bool ok = (0 == unsetenv(key.c_str()));
    #if defined(_darwin_)
    notFound = !ok && (errno == EINVAL);
    #endif
#endif
    Y_ENSURE_EX(ok || notFound, TSystemError() << "failed to unset environment variable " << key.Quote());
}

void IterateEnv(const std::function<void(TStringBuf, TStringBuf)>& f, bool ignoreMalformedStrings) {
#ifdef _win_
    // Env block format:
    // Var1=Value1\0
    // Var2=Value2\0
    // ...
    // VarN=ValueN\0\0
    // Or "\0" if empty

    auto envBlock = GetEnvironmentStringsA();
    if (!envBlock) {
        ythrow TSystemError() << "failed to get environment variables";
    }
    Y_DEFER {
        bool ok = FreeEnvironmentStringsA(envBlock);
        Y_ABORT_UNLESS(ok, "failed to free env block"); // ¯\_(ツ)_/¯
    };
    const char* charEnv = envBlock;
    while (*charEnv) {
        TStringBuf varStr(charEnv);
        TStringBuf name, value;
        if (varStr.TrySplit('=', name, value)) { // Not optimal (we could scan for null byte and '=' in one pass), but less bugprone
            f(name, value);
        } else {
            Y_ENSURE(ignoreMalformedStrings, "Environment contains a string without '=': \"" << varStr << '"');
        }
        charEnv = varStr.data() + varStr.size() + 1;
    }
#else
    if (!environ) { // may be null after clearenv
        return;
    }
    for (char** var = environ; *var; ++var) {
        TStringBuf varStr(*var);
        TStringBuf name, value;
        if (varStr.TrySplit('=', name, value)) {
            f(name, value);
        } else {
            Y_ENSURE(ignoreMalformedStrings, "Environment contains a string without '=': \"" << varStr << '"');
        }
    }
#endif
}

void ClearEnv() {
#if defined(_win_)
    wchar_t emptyEnv[] = {L'\0'};
    // At least on Wine, SetEnvironmentStringsA expects a multi-byte string that is internally converted to UTF-16.
    // So it's easier to use SetEnvironmentStringsW directly.
    bool ok = SetEnvironmentStringsW(emptyEnv);
    Y_ENSURE_EX(ok, TSystemError() << "failed to clear environment");
#elif defined(_linux_)
    bool ok = !clearenv();
    Y_ENSURE_EX(ok, TSystemError() << "failed to clear environment");
#else
    // Darwin or other unix platform that may not have clearenv
    TVector<TString> keys;
    IterateEnv(
        [&keys](TStringBuf name, TStringBuf) {
            keys.emplace_back(name);
        },
        true // Ignore malformed strings - they don't seem to be accessible anyway.
             // IterateEnv won't expose these strings, and GetEnv/SetEnv/UnsetEnv ignore them (at least in glibc).
    );
    for (const auto& key : keys) {
        UnsetEnv(key); // duplicate UnsetEnv is ok.
    }
#endif
}
