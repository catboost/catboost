#pragma once

#include <util/generic/hash.h>

namespace NVariant {
    template <class T>
    class TVisitorEquals {
    public:
        TVisitorEquals(const T& left)
            : Left_(left)
        {
        }

        template <class T2>
        bool operator()(const T2& right) const {
            return Left_ == right;
        }

    private:
        const T& Left_;
    };

    struct TVisitorHash {
        template <class T>
        size_t operator()(const T& value) const {
            return THash<T>()(value);
        };
    };

    struct TVisitorDestroy {
        template <class T>
        void operator()(T& value) const {
            Y_UNUSED(value);
            value.~T();
        };
    };

    template <class T>
    class TVisitorCopyConstruct {
    public:
        TVisitorCopyConstruct(T* var)
            : Var_(var)
        {
        }

        template <class T2>
        void operator()(const T2& value) {
            new (Var_) T(value);
        };

    private:
        T* const Var_;
    };

    template <class T>
    class TVisitorMoveConstruct {
    public:
        TVisitorMoveConstruct(T* var)
            : Var_(var)
        {
        }

        template <class T2>
        void operator()(T2& value) {
            new (Var_) T(std::move(value));
        };

    private:
        T* const Var_;
    };
}
