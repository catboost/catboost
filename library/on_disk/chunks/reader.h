#pragma once

#include <util/generic/array_ref.h>
#include <util/generic/vector.h>
#include <util/generic/yexception.h>

class TBlob;

class TChunkedDataReader {
public:
    TChunkedDataReader(const TBlob& blob);

    inline const void* GetBlock(size_t index) const {
        CheckIndex(index);
        return Offsets[index];
    }

    inline size_t GetBlockLen(size_t index) const {
        CheckIndex(index);

        if (Version == 0) {
            if (index + 1 < Offsets.size()) {
                return Offsets[index + 1] - Offsets[index];
            }

            return Size - (Offsets.back() - Offsets.front());
        }

        return Lengths[index];
    }

    TBlob GetBlob(size_t index) const;

    template <typename T>
    TArrayRef<const T> GetRegion(size_t index) const {
        size_t len = GetBlockLen(index);
        Y_ENSURE(len % sizeof(T) == 0, "wrong data padding");
        return TArrayRef<const T>(reinterpret_cast<const T*>(GetBlock(index)), len / sizeof(T));
    }

    inline size_t GetBlocksCount() const {
        return Offsets.size();
    }

private:
    inline void CheckIndex(size_t index) const {
        if (index >= GetBlocksCount()) {
            ythrow yexception() << "requested block " << index << " of " << GetBlocksCount() << " blocks";
        }
    }

private:
    ui64 Version = 0;
    TVector<const char*> Offsets;
    TVector<size_t> Lengths;
    size_t Size = 0;
};
