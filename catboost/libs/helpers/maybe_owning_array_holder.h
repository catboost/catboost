#pragma once

#include "resource_holder.h"
#include "serialization.h"

#include <library/binsaver/bin_saver.h>

#include <util/generic/array_ref.h>
#include <util/generic/cast.h>


namespace NCB {

    template <class T>
    class TMaybeOwningArrayHolder {
    public:
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
            return TMaybeOwningArrayHolder(vectorHolder->Data, std::move(vectorHolder));
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
            if (binSaver.IsReading()) {
                TVector<T> data;
                data.yresize(serializedSize);
                LoadArrayData<T>(data, &binSaver);
                *this = TMaybeOwningArrayHolder<T>::CreateOwning(std::move(data));
            } else {
                SaveArrayData<T>(ArrayRef, &binSaver);
            }
            return 0;
        }

        bool operator==(const TMaybeOwningArrayHolder& rhs) const {
            return ArrayRef == rhs.ArrayRef;
        }

        TArrayRef<T> operator*() {
            return ArrayRef;
        }

        TConstArrayRef<T> operator*() const {
            return ArrayRef;
        }

        T* data() {
            return ArrayRef.data();
        }

        const T* data() const {
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

        size_t GetSize() const {
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
}

