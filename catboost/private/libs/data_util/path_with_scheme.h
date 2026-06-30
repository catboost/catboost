#pragma once

#include <catboost/libs/helpers/exception.h>

#include <library/cpp/binsaver/bin_saver.h>
#include <library/cpp/object_factory/object_factory.h>

#include <util/digest/multi.h>
#include <util/generic/hash.h>
#include <util/generic/ptr.h>
#include <util/generic/strbuf.h>
#include <util/generic/string.h>
#include <util/ysaveload.h>


namespace NCB {

    // Represents '[scheme://]path', empty scheme is ok
    struct TPathWithScheme {
        TString Scheme; // empty is ok
        TString Path;

    public:
        TPathWithScheme() = default;

        explicit TPathWithScheme(TStringBuf pathWithScheme, TStringBuf defaultScheme = "") {
            TStringBuf part1, part2;
            pathWithScheme.Split(TStringBuf("://"), part1, part2);
            if (part1 == pathWithScheme) { // no scheme in pathWithScheme
                Scheme = defaultScheme;
                Path = part1;
            } else {
                CB_ENSURE(!part1.empty(),
                          "Empty scheme part for path with scheme: " << pathWithScheme);
                Scheme = part1;
                Path = part2;
            }
            CB_ENSURE(!Path.empty(), "Empty path part for path with scheme: " << pathWithScheme);
        }

        SAVELOAD(Scheme, Path);
        Y_SAVELOAD_DEFINE(Scheme, Path);

        bool Inited() const noexcept {
            return !Path.empty();
        }

        bool operator==(const TPathWithScheme& rhs) const {
            return std::tie(Scheme, Path) == std::tie(rhs.Scheme, rhs.Path);
        }

        bool operator!=(const TPathWithScheme& rhs) const {
            return !(rhs == *this);
        }

        ui64 GetHash() const {
            return MultiHash(Scheme, Path);
        }
    };

    template <class ISchemeDependentProcessor, class... TArgs>
    THolder<ISchemeDependentProcessor> GetProcessor(TPathWithScheme pathWithScheme, TArgs&&... args) {
        auto res = NObjectFactory::TParametrizedObjectFactory<ISchemeDependentProcessor, TString, TArgs...>::Construct(
                pathWithScheme.Scheme, std::forward<TArgs>(args)...
            );
        CB_ENSURE(res != nullptr,
                  "Processor for scheme [" << pathWithScheme.Scheme << "] not found");
        return THolder<ISchemeDependentProcessor>(res);
    }

}

template <>
struct THash<NCB::TPathWithScheme> {
    inline size_t operator()(const NCB::TPathWithScheme& value) const {
        return value.GetHash();
    }
};
