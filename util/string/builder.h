#pragma once

#include <util/stream/str.h>
#include <utility>
#include <util/generic/string.h>

namespace NPrivateStringBuilder {
    class TStringBuilder: public TString {
    public:
        inline TStringBuilder()
            : Out(*this)
        {
        }

        TStringBuilder(TStringBuilder&& rhs) noexcept
            : TString(std::move(rhs))
            , Out(*this)
        {
        }

        TStringOutput Out;
    };

    template <class T>
    static inline TStringBuilder& operator<<(TStringBuilder& builder, const T& t) {
        builder.Out << t;

        return builder;
    }

    template <class T>
    static inline TStringBuilder&& operator<<(TStringBuilder&& builder, const T& t) {
        builder.Out << t;

        return std::move(builder);
    }
}

using TStringBuilder = NPrivateStringBuilder::TStringBuilder;
