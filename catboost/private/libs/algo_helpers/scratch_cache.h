#pragma once

#include <library/cpp/containers/dense_hash/dense_hash.h>

#include <util/generic/array_ref.h>
#include <util/generic/ptr.h>
#include <util/generic/vector.h>
#include <util/thread/lfqueue.h>

namespace NCB {

    template <typename T>
    inline TArrayRef<T> GrowScratchBlob(size_t count, TVector<ui8>* scratch) {
        if (count * sizeof(T) > scratch->size()) {
            scratch->yresize(count * sizeof(T));
        }
        return TArrayRef<T>((T*)scratch->data(), count);
    }

    template <typename T>
    inline TArrayRef<T> PrepareScratchBlob(size_t count, TVector<ui8>* scratch) {
        auto array = GrowScratchBlob<T>(count, scratch);
        Fill(scratch->data(), scratch->data() + count * sizeof(T), 0);
        return array;
    }

    struct TScratchCache {
        TAtomicSharedPtr<TVector<ui8>> GetScratchBlob() {
            TAtomicSharedPtr<TVector<ui8>> scratch;
            if (BlobStorage.Dequeue(&scratch)) {
                return scratch;
            } else {
                return MakeAtomicShared<TVector<ui8>>();
            }
        }
        void ReleaseScratchBlob(TAtomicSharedPtr<TVector<ui8>> scratch) {
            BlobStorage.Enqueue(scratch);
        }

        TAtomicSharedPtr<TDenseHash<ui64, ui32>> GetScratchHash() {
            TAtomicSharedPtr<TDenseHash<ui64, ui32>> scratch;
            if (HashStorage.Dequeue(&scratch)) {
                return scratch;
            } else {
                return MakeAtomicShared<TDenseHash<ui64, ui32>>();
            }
        }
        void ReleaseScratchHash(TAtomicSharedPtr<TDenseHash<ui64, ui32>> scratch) {
            HashStorage.Enqueue(scratch);
        }
    private:
        TLockFreeQueue<TAtomicSharedPtr<TVector<ui8>>> BlobStorage;
        TLockFreeQueue<TAtomicSharedPtr<TDenseHash<ui64, ui32>>> HashStorage;
    };

}
