#pragma once

#include <catboost/libs/helpers/exception.h>

#include <util/generic/utility.h>
#include <util/generic/vector.h>
#include <util/system/yassert.h>


const int BS_MASK = 0x7fffffff;
const int BS_NEW_ELEM = 0x80000000;

struct TBinTracker {
    struct THashElem {
        ui64 HashValue = 0;
        int Version = 0;
        int ElemId = 0;

    public:
        THashElem() = default;
    };

public:
    int ElemCount = 0;
    TVector<THashElem> Hash;
    ui64 HashMask = 0;
    int CurrentVersion = 0;

public:
    TBinTracker() = default;
    void Alloc(int maxElemCount) {
        CB_ENSURE(maxElemCount < BS_MASK, "fail");
        HashMask = 1;
        while (((ui64)maxElemCount) * 3 > HashMask) {
            HashMask = HashMask * 2 + 1;
        }
        Hash.clear();
        Hash.resize(HashMask + 1, THashElem());
        CurrentVersion = 0;
        Clear();
    }
    void Clear() {
        ++CurrentVersion;
        if (CurrentVersion == 2 * 1000 * 1000 * 1000) {
            CurrentVersion = 1;
            for (int i = 0; i < Hash.ysize(); ++i) {
                Hash[i].Version = 0;
            }
        }
        ElemCount = 0;
    }
    int Add(const ui64 hashValue) {
        ui64 idx = (hashValue >> 1) & HashMask; // hash value is always odd
        int elemId = 0;
        for (;;) {
            THashElem& he = Hash[idx];
            if (he.Version < CurrentVersion) {
                he.Version = CurrentVersion;
                he.HashValue = hashValue;
                elemId = ElemCount++;
                he.ElemId = elemId;
                elemId |= BS_NEW_ELEM;
                break;
            }
            Y_ASSERT(he.Version == CurrentVersion);
            if (he.HashValue == hashValue) {
                elemId = he.ElemId;
                Y_ASSERT(elemId < ElemCount);
                break;
            }
            idx = (idx + 1) & HashMask;
        }
        return elemId;
    }
    int Find(const ui64 hashValue) const {
        ui64 idx = (hashValue >> 1) & HashMask; // hash value is always odd
        for (;;) {
            const THashElem& he = Hash[idx];
            if (he.Version < CurrentVersion) {
                break;
            }
            Y_ASSERT(he.Version == CurrentVersion);
            if (he.HashValue == hashValue) {
                Y_ASSERT(he.ElemId < ElemCount);
                return he.ElemId;
            }
            idx = (idx + 1) & HashMask;
        }
        return -1;
    }
};
