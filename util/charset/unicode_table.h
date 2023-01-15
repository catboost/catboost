#pragma once

#include <util/system/yassert.h>
#include <util/system/defaults.h>
#include <util/generic/typetraits.h>

namespace NUnicodeTable {
    template <class Value>
    struct TValueSelector;

    template <class Value>
    struct TValueSelector {
        using TStored = const Value;
        using TValueRef = const Value&;
        using TValuePtr = const Value*;

        static inline TValueRef Get(TValuePtr val) {
            return *val;
        }
    };

    template <class Value>
    struct TValueSelector<const Value*> {
        using TStored = const Value[];
        using TValueRef = const Value*;
        using TValuePtr = const Value*;

        static inline TValueRef Get(TValuePtr val) {
            return val;
        }
    };

    template <class Value>
    struct TValues {
        using TSelector = TValueSelector<Value>;

        using TStored = typename TSelector::TStored;
        using TValueRef = typename TSelector::TValueRef;
        using TValuePtr = typename TSelector::TValuePtr;

        using TData = const TValuePtr*;

        static inline TValuePtr Get(TData table, size_t index) {
            static_assert(std::is_pointer<TData>::value, "expect std::is_pointer<TData>::value");
            return table[index];
        }

        static inline TValueRef Get(TValuePtr val) {
            return TSelector::Get(val);
        }
    };

    template <int Shift, class TChild>
    struct TSubtable {
        using TStored = typename TChild::TStored;
        using TValueRef = typename TChild::TValueRef;
        using TValuePtr = typename TChild::TValuePtr;
        using TData = const typename TChild::TData*;

        static inline TValuePtr Get(TData table, size_t key) {
            static_assert(std::is_pointer<TData>::value, "expect std::is_pointer<TData>::value");
            return TChild::Get(table[key >> Shift], key & ((1 << Shift) - 1));
        }

        static inline TValueRef Get(TValuePtr val) {
            return TChild::Get(val);
        }
    };

    template <class T>
    class TTable {
    private:
        using TImpl = T;
        using TData = typename TImpl::TData;

        const TData Data;
        const size_t MSize;

    public:
        using TStored = typename TImpl::TStored;
        using TValueRef = typename TImpl::TValueRef;
        using TValuePtr = typename TImpl::TValuePtr;

    private:
        inline TValueRef GetImpl(size_t key) const {
            TValuePtr val = TImpl::Get(Data, key);

            return TImpl::Get(val);
        }

        inline TValueRef Get(size_t key) const {
            return GetImpl(key);
        }

    public:
        TTable(TData data, size_t size)
            : Data(data)
            , MSize(size)
        {
            static_assert(std::is_pointer<TData>::value, "expect std::is_pointer<TData>::value");
        }

        inline TValueRef Get(size_t key, TValueRef value) const {
            if (key >= Size())
                return value;

            return GetImpl(key);
        }

        inline TValueRef Get(size_t key, size_t defaultKey) const {
            if (key >= Size())
                return Get(defaultKey);

            return GetImpl(key);
        }

        inline size_t Size() const {
            return MSize;
        }
    };

    const size_t UNICODE_TABLE_SHIFT = 5;
}
