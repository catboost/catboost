#pragma once

#include <util/generic/strbuf.h>
#include <util/generic/string.h>
#include <util/system/compiler.h>

// RAII primitive to set/unset thread name within scope

namespace NThreading {
    class TThreadNameGuard {
    public:
        TThreadNameGuard(TThreadNameGuard&&);
        ~TThreadNameGuard();

        static TThreadNameGuard Make(TStringBuf name);
    private:
        TThreadNameGuard(TStringBuf name);

        TThreadNameGuard(const TThreadNameGuard&) = delete;
        TThreadNameGuard& operator =(const TThreadNameGuard&) = delete;
        TThreadNameGuard& operator =(TThreadNameGuard&&) = delete;

    private:
        TString PreviousThreadName_;
    };
}

#define Y_THREAD_NAME_GUARD(name) \
    const auto Y_GENERATE_UNIQUE_ID(threadNameGuard) = ::NThreading::TThreadNameGuard::Make(name);
