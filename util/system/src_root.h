#pragma once

#include "compiler.h"
#include "defaults.h"

#include <type_traits>

namespace NPrivate {
    struct TStaticBuf {
        constexpr TStaticBuf(const char* data, unsigned len) noexcept
            : Data(data)
            , Len(len)
        {
        }

        template <class T>
        constexpr T As() const noexcept {
            return T(Data, Len);
        }

        template <class T>
        constexpr operator T() const noexcept {
            return this->As<T>();
        }

        const char* Data;
        unsigned Len;
    };

#define STATIC_BUF(x) ::NPrivate::TStaticBuf(x, sizeof(x) - 1)

    constexpr TStaticBuf ArcRoot = STATIC_BUF(Y_STRINGIZE(ARCADIA_ROOT));
    constexpr TStaticBuf BuildRoot = STATIC_BUF(Y_STRINGIZE(ARCADIA_BUILD_ROOT));

    constexpr Y_FORCE_INLINE bool IsProperPrefix(const TStaticBuf prefix, const TStaticBuf string) noexcept {
        if (prefix.Len < string.Len) {
            for (unsigned i = prefix.Len; i-- > 0;) {
                if (prefix.Data[i] != string.Data[i]) {
#if defined(_MSC_VER) && !defined(__clang__)
                    // cl.exe uses back slashes for __FILE__ but ARCADIA_ROOT, ARCADIA_BUILD_ROOT are
                    // defined with forward slashes
                    if ((prefix.Data[i] == '/') && (string.Data[i] == '\\')) {
                        continue;
                    }
#endif
                    return false;
                }
            }
            return true;
        } else {
            return false;
        }
    }

    constexpr unsigned RootPrefixLength(const TStaticBuf& f) noexcept {
        if (IsProperPrefix(ArcRoot, f)) {
            return ArcRoot.Len + 1;
        }
        if (IsProperPrefix(BuildRoot, f)) {
            return BuildRoot.Len + 1;
        }
        return 0;
    }

    constexpr Y_FORCE_INLINE TStaticBuf StripRoot(const TStaticBuf& f, unsigned prefixLength) noexcept {
        return TStaticBuf(f.Data + prefixLength, f.Len - prefixLength);
    }

    //$(SRC_ROOT)/prj/blah.cpp -> prj/blah.cpp
    constexpr Y_FORCE_INLINE TStaticBuf StripRoot(const TStaticBuf& f) noexcept {
        return StripRoot(f, RootPrefixLength(f));
    }
} // namespace NPrivate

#define __SOURCE_FILE_IMPL__ ::NPrivate::StripRoot(STATIC_BUF(__FILE__), std::integral_constant<unsigned, ::NPrivate::RootPrefixLength(STATIC_BUF(__FILE__))>::value)
