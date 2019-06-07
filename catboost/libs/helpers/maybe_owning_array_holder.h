#pragma once

#include "resource_holder.h"

#include <util/generic/array_ref.h>


namespace NCB {

    template <class T>
    class TMaybeOwningArrayHolder {
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

        TArrayRef<T> operator*() {
            return ArrayRef;
        }

        TConstArrayRef<T> operator*() const {
            return ArrayRef;
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

