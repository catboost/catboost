#pragma once

#include <catboost/libs/helpers/exception.h>
#include <catboost/libs/helpers/maybe_owning_array_holder.h>

#include <util/generic/array_ref.h>
#include <util/generic/type_name.h>
#include <util/system/unaligned_mem.h>

#include <cstring>
#include <cstdlib>


namespace NCB {

    template <class T, unsigned Align = alignof(T)>
    class TUnalignedArrayBuf {
    public:
        TUnalignedArrayBuf(const void* begin, size_t sizeInBytes)
            : Begin(begin)
            , SizeInBytes(sizeInBytes)
        {
            CB_ENSURE_INTERNAL(
                !(sizeInBytes % sizeof(T)),
                LabeledOutput(sizeInBytes) << " does not correspond to size of array of type "
                << TypeName<T>());
        }

        explicit TUnalignedArrayBuf(TConstArrayRef<ui8> memoryRegion)
            : TUnalignedArrayBuf(memoryRegion.data(), memoryRegion.size())
        {}

        explicit TUnalignedArrayBuf(TConstArrayRef<T> alignedData)
            : TUnalignedArrayBuf(alignedData.data(), alignedData.size() * sizeof(T))
        {}

        size_t GetSize() const {
            return SizeInBytes / sizeof(T);
        }

        void WriteTo(TArrayRef<T>* dst) const {
            CB_ENSURE_INTERNAL(
                dst->size() == GetSize(),
                "TUnalignedArrayBuf::WriteTo: Wrong destination array size; "
                LabeledOutput(dst->size(), GetSize()));
            memcpy(dst->data(), Begin, SizeInBytes);
        }

        void WriteTo(TVector<T>* dst) const {
            dst->yresize(GetSize());
            memcpy(dst->data(), Begin, SizeInBytes);
        }

        // allocates and copies only if Begin is unaligned
        TMaybeOwningArrayHolder<T> GetAlignedData() const {
            if (reinterpret_cast<const size_t>(Begin) % Align) {
                TVector<T> alignedData;
                WriteTo(&alignedData);
                return TMaybeOwningArrayHolder<T>::CreateOwning(std::move(alignedData));
            } else {
                return TMaybeOwningArrayHolder<T>::CreateNonOwning(TArrayRef<T>((T*)Begin, GetSize()));
            }
        }

        TUnalignedMemoryIterator<T> GetIterator() const {
            return TUnalignedMemoryIterator<T>(Begin, SizeInBytes);
        }


    private:
        const void* Begin;
        size_t SizeInBytes;
    };

}
