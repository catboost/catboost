#pragma once

#include "feature_index.h"

#include <catboost/libs/helpers/checksum.h>
#include <catboost/libs/helpers/exception.h>

#include <library/binsaver/bin_saver.h>
#include <library/dbg_output/dump.h>

#include <util/folder/path.h>
#include <util/generic/guid.h>
#include <util/generic/map.h>
#include <util/generic/maybe.h>
#include <util/generic/ptr.h>
#include <util/generic/string.h>
#include <util/generic/typetraits.h>
#include <util/generic/vector.h>
#include <util/stream/file.h>
#include <util/system/fs.h>
#include <util/system/mktemp.h>
#include <util/system/spinlock.h>
#include <util/system/tempfile.h>
#include <util/system/types.h>
#include <util/system/yassert.h>
#include <util/ysaveload.h>


namespace NCB {
    struct TCatFeatureUniqueValuesCounts {
        ui32 OnLearnOnly = 0;
        ui32 OnAll = 0;

    public:
        bool operator==(const TCatFeatureUniqueValuesCounts rhs) const {
            return (OnLearnOnly == rhs.OnLearnOnly) && (OnAll == rhs.OnAll);
        }
    };

    // for some reason TCatFeatureUniqueValuesCounts is not std::is_trivial
    inline ui32 UpdateCheckSumImpl(ui32 init, const TCatFeatureUniqueValuesCounts& data) {
        ui32 checkSum = UpdateCheckSum(init, data.OnLearnOnly);
        return UpdateCheckSum(checkSum, data.OnAll);
    }

    inline ui8 CalcHistogramWidthForUniqueValuesCount(ui32 count) {
        if (count <= 1ULL << 8) {
            return 8;
        } else if (count <= 1ULL << 16) {
            return 16;
        } else { //TODO
            return 32;
        }
    }

    struct TValueWithCount {
        ui32 Value = 0;
        ui32 Count = 0;

    public:
        bool operator==(const TValueWithCount rhs) const {
            return (Value == rhs.Value) && (Count == rhs.Count);
        }
    };

    // for some reason TValueWithCount is not std::is_trivial
    inline ui32 UpdateCheckSumImpl(ui32 init, const TValueWithCount& data) {
        ui32 checkSum = UpdateCheckSum(init, data.Value);
        return UpdateCheckSum(checkSum, data.Count);
    }


    struct TCatFeaturePerfectHashDefaultValue {
        ui32 SrcValue;
        TValueWithCount DstValueWithCount;
        float Fraction; // on Learn

    public:
        bool operator==(const TCatFeaturePerfectHashDefaultValue& rhs) const;
    };

    inline ui32 UpdateCheckSumImpl(ui32 init, const TCatFeaturePerfectHashDefaultValue& data) {
        return UpdateCheckSum(init, data.SrcValue, data.DstValueWithCount, data.Fraction);
    }

    struct TCatFeaturePerfectHash {
        TMaybe<TCatFeaturePerfectHashDefaultValue> DefaultMap;
        TMap<ui32, TValueWithCount> Map;

    public:
        bool operator==(const TCatFeaturePerfectHash& rhs) const {
            return (DefaultMap == rhs.DefaultMap) && (Map == rhs.Map);
        }

        SAVELOAD(DefaultMap, Map);

        Y_SAVELOAD_DEFINE(DefaultMap, Map);

        size_t GetSize() const {
            return DefaultMap.Defined() ? (Map.size() + 1) : Map.size();
        }

        bool Empty() const {
            return GetSize() == 0;
        }

        TMaybe<TValueWithCount> Find(ui32 key) const {
            if (DefaultMap && (DefaultMap->SrcValue == key)) {
                return DefaultMap->DstValueWithCount;
            }
            const auto it = Map.find(key);
            return (it != Map.end()) ? MakeMaybe(it->second) : Nothing();
        }
    };

    inline ui32 UpdateCheckSumImpl(ui32 init, const TCatFeaturePerfectHash& data) {
        return UpdateCheckSum(init, data.DefaultMap, data.Map);
    }

}

Y_DECLARE_PODTYPE(NCB::TCatFeatureUniqueValuesCounts);
Y_DECLARE_PODTYPE(NCB::TValueWithCount);
Y_DECLARE_PODTYPE(NCB::TCatFeaturePerfectHashDefaultValue);


template <>
struct TDumper<NCB::TValueWithCount> {
    template <class S>
    static inline void Dump(
        S& s,
        NCB::TValueWithCount valueWithCount
    ) {
        s << "{Value=" << valueWithCount.Value << ",Count=" << valueWithCount.Count << '}';
    }
};

template <>
struct TDumper<NCB::TCatFeaturePerfectHashDefaultValue> {
    template <class S>
    static inline void Dump(
        S& s,
        const NCB::TCatFeaturePerfectHashDefaultValue& catFeaturePerfectHashDefaultValue
    ) {
        s << "{SrcValue=" << catFeaturePerfectHashDefaultValue.SrcValue
            << ",DstValueWithCount=" << DbgDump(catFeaturePerfectHashDefaultValue.DstValueWithCount)
            << ",Fraction=" << catFeaturePerfectHashDefaultValue.Fraction << '}';
    }
};

template <>
struct TDumper<NCB::TCatFeaturePerfectHash> {
    template <class S>
    static inline void Dump(S& s, const NCB::TCatFeaturePerfectHash& catFeaturePerfectHash) {
        s << "{DefaultMap=";
        if (catFeaturePerfectHash.DefaultMap) {
            s << DbgDump(*catFeaturePerfectHash.DefaultMap);
        } else {
            s << "None";
        }
        s << "Map=" << DbgDump(catFeaturePerfectHash.Map) << "}\n";
    }
};


namespace NCB {

    class TCatFeaturesPerfectHash {
    public:
        // for IBinSaver
        TCatFeaturesPerfectHash() = default;

        TCatFeaturesPerfectHash(ui32 catFeatureCount)
            : CatFeatureUniqValuesCountsVector(catFeatureCount)
            , FeaturesPerfectHash(catFeatureCount)
        {}

        ~TCatFeaturesPerfectHash() = default;

        bool operator==(const TCatFeaturesPerfectHash& rhs) const;

        const TCatFeaturePerfectHash& GetFeaturePerfectHash(const TCatFeatureIdx catFeatureIdx) const {
            CheckHasFeature(catFeatureIdx);
            if (!HasHashInRam) {
                Load();
            }
            return FeaturesPerfectHash[*catFeatureIdx];
        }

        // for testing or setting from external sources
        void UpdateFeaturePerfectHash(const TCatFeatureIdx catFeatureIdx, TCatFeaturePerfectHash&& perfectHash);

        TCatFeatureUniqueValuesCounts GetUniqueValuesCounts(const TCatFeatureIdx catFeatureIdx) const {
            CheckHasFeature(catFeatureIdx);
            const auto uniqValuesCounts = CatFeatureUniqValuesCountsVector[*catFeatureIdx];
            return uniqValuesCounts.OnAll > 1 ? uniqValuesCounts : TCatFeatureUniqueValuesCounts();
        }

        bool HasFeature(const TCatFeatureIdx catFeatureIdx) const {
            return (size_t)*catFeatureIdx < CatFeatureUniqValuesCountsVector.size();
        }

        void FreeRam(const TString& tmpDir) const {
            if (!StorageTempFile) {
                StorageTempFile = MakeHolder<TTempFile>(
                    JoinFsPaths(tmpDir, TString::Join("cat_feature_index.", CreateGuidAsString(), ".tmp"))
                );
            }
            Save();
            TVector<TCatFeaturePerfectHash> empty;
            FeaturesPerfectHash.swap(empty);
            HasHashInRam = false;
        }

        void Load() const {
            if (!HasHashInRam) {
                Y_VERIFY(StorageTempFile);
                TIFStream inputStream(StorageTempFile->Name());
                FeaturesPerfectHash.clear();
                ::Load(&inputStream, FeaturesPerfectHash);
                HasHashInRam = true;
            }
        }

        Y_SAVELOAD_DEFINE(CatFeatureUniqValuesCountsVector, FeaturesPerfectHash, HasHashInRam);

        int operator&(IBinSaver& binSaver);

        ui32 CalcCheckSum() const;

    private:
        void Save() const {
            Y_VERIFY(StorageTempFile);
            TOFStream out(StorageTempFile->Name());
            ::Save(&out, FeaturesPerfectHash);
        }

    private:
        friend class TCatFeaturesPerfectHashHelper;

    private:
        void CheckHasFeature(const TCatFeatureIdx catFeatureIdx) const {
            CB_ENSURE_INTERNAL(
                HasFeature(catFeatureIdx),
                "Error: unknown categorical feature #" << catFeatureIdx
            );
        }

    private:
        TVector<TCatFeatureUniqueValuesCounts> CatFeatureUniqValuesCountsVector; // [catFeatureIdx]
        mutable TVector<TCatFeaturePerfectHash> FeaturesPerfectHash; // [catFeatureIdx]
        mutable bool HasHashInRam = true;
        mutable THolder<TTempFile> StorageTempFile;
    };
}
