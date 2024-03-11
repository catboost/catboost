#pragma once

#include "resource_holder.h"
#include "serialization.h"

#include <library/cpp/binsaver/bin_saver.h>

#include <util/generic/array_ref.h>
#include <util/generic/cast.h>
#include <util/system/compiler.h>

#include <type_traits>


namespace NCB {

    template <class T>
    class TMaybeOwningArrayHolder {
    public:
        using value_type = T;
        using iterator = typename TArrayRef<T>::iterator;
        using const_iterator = typename TArrayRef<T>::const_iterator;

    public:
        TMaybeOwningArrayHolder() = default;

        static TMaybeOwningArrayHolder CreateNonOwning(TArrayRef<T> arrayRef)
        {
            return TMaybeOwningArrayHolder(arrayRef, nullptr);
        }

        static TMaybeOwningArrayHolder CreateOwning(
            TArrayRef<T> arrayRef,
            TIntrusivePtr<IResourceHolder> resourceHolder
        ) {
            return TMaybeOwningArrayHolder(arrayRef, std::move(resourceHolder));
        }

        template <class T2>
        static TMaybeOwningArrayHolder CreateOwning(TVector<T2>&& data) {
            auto vectorHolder = MakeIntrusive<NCB::TVectorHolder<T2>>(std::move(data));
            const TArrayRef<T> dataRef{vectorHolder->Data};
            return TMaybeOwningArrayHolder(dataRef, std::move(vectorHolder));
        }

        // for Cython that does not support move semantics
        template <class T2>
        static TMaybeOwningArrayHolder CreateOwningMovedFrom(TVector<T2>& data) {
            auto vectorHolder = MakeIntrusive<NCB::TVectorHolder<T2>>(std::move(data));
            const TArrayRef<T> dataRef{vectorHolder->Data};
            return TMaybeOwningArrayHolder(dataRef, std::move(vectorHolder));
        }

        template <class T2>
        static TMaybeOwningArrayHolder CreateOwningReinterpretCast(TMaybeOwningArrayHolder<T2>& data) {
            auto arrayRef = *data;
            return TMaybeOwningArrayHolder(
                TArrayRef<T>((T*)arrayRef.begin(), (T*)arrayRef.end()),
                data.GetResourceHolder());
        }

        int operator&(IBinSaver& binSaver) {
            IBinSaver::TStoredSize serializedSize;
            if (!binSaver.IsReading()) {
                serializedSize = SafeIntegerCast<IBinSaver::TStoredSize>(ArrayRef.size());
            }
            binSaver.Add(1, &serializedSize);

            using TNonConstValue = std::remove_const_t<T>;
            if (binSaver.IsReading()) {
                TVector<TNonConstValue> data;
                data.yresize(serializedSize);
                LoadArrayData<TNonConstValue>(data, &binSaver);
                *this = TMaybeOwningArrayHolder<T>::CreateOwning(std::move(data));
            } else {
                SaveArrayData<TNonConstValue>(ArrayRef, &binSaver);
            }
            return 0;
        }

        // if strict is true compare bit-by-bit, else compare values
        // implemented here for interface uniformity
        bool EqualTo(const TMaybeOwningArrayHolder& rhs, bool strict = true) const {
            Y_UNUSED(strict);
            return ArrayRef == rhs.ArrayRef;
        }

        bool operator==(const TMaybeOwningArrayHolder& rhs) const {
            return EqualTo(rhs);
        }

        TArrayRef<T> operator*() noexcept {
            return ArrayRef;
        }

        TConstArrayRef<T> operator*() const noexcept {
            return ArrayRef;
        }

        T* data() noexcept {
            return ArrayRef.data();
        }

        const T* data() const noexcept {
            return ArrayRef.data();
        }

        T& operator[] (size_t idx) {
            return ArrayRef[idx];
        }

        const T& operator[] (size_t idx) const {
            return ArrayRef[idx];
        }

        TIntrusivePtr<IResourceHolder> GetResourceHolder() const {
            return ResourceHolder;
        }

        size_t GetSize() const noexcept {
            return ArrayRef.size();
        }

        iterator begin() {
            return ArrayRef.begin();
        }

        iterator end() {
            return ArrayRef.end();
        }

        const_iterator begin() const {
            return ArrayRef.begin();
        }

        const_iterator end() const {
            return ArrayRef.end();
        }

        TMaybeOwningArrayHolder Slice(size_t offset, size_t size) const {
            return TMaybeOwningArrayHolder(ArrayRef.Slice(offset, size), ResourceHolder);
        }

    private:
        TMaybeOwningArrayHolder(
            TArrayRef<T> arrayRef,
            TIntrusivePtr<IResourceHolder> resourceHolder
        )
            : ArrayRef(arrayRef)
            , ResourceHolder(std::move(resourceHolder))
        {}

    private:
        TArrayRef<T> ArrayRef;
        TIntrusivePtr<IResourceHolder> ResourceHolder;
    };

    template <class T>
    using TMaybeOwningConstArrayHolder = TMaybeOwningArrayHolder<const T>;

    template <class TDst, class TSrc>
    TMaybeOwningArrayHolder<TDst> CreateOwningWithMaybeTypeCast(TMaybeOwningArrayHolder<TSrc> src) {
        if constexpr (std::is_same_v<std::remove_const_t<TDst>, TSrc>) {
            return TMaybeOwningArrayHolder<TDst>::CreateOwningReinterpretCast(src);
        } else {
            TVector<TDst> dstData(src.begin(), src.end());
            return TMaybeOwningArrayHolder<TDst>::CreateOwning(std::move(dstData));
        }
    }

    // for Cython that has problems with const template parameters
    template <class TDst, class TSrc>
    TMaybeOwningArrayHolder<const TDst> CreateConstOwningWithMaybeTypeCast(TMaybeOwningArrayHolder<TSrc> src) {
        return CreateOwningWithMaybeTypeCast<const TDst, TSrc>(std::move(src));
    }
}
