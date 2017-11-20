#pragma once

#include "online_ctr.h"
#include <catboost/libs/model/flatbuffers/ctr_data.fbs.h>

#include <catboost/libs/helpers/dense_hash_view.h>
#include <util/generic/vector.h>
#include <util/generic/variant.h>
#include <tuple>
#include <util/stream/input.h>
#include <util/stream/output.h>

class TCtrValueTable {
    struct TSolidTable {
        TVector<NCatboost::TBucket> IndexBuckets;
        TVector<ui8> CTRBlob;
        bool operator==(const TSolidTable& other) const {
            return std::tie(IndexBuckets, CTRBlob) == std::tie(other.IndexBuckets, other.CTRBlob);
        }
    };
    struct TThinTable {
        TConstArrayRef<NCatboost::TBucket> IndexBuckets;
        TConstArrayRef<ui8> CTRBlob;

        bool operator==(const TThinTable& other) const {
            return std::tie(IndexBuckets, CTRBlob) == std::tie(other.IndexBuckets, other.CTRBlob);
        }

        void ToSolidTable(TSolidTable* table) {
            table->IndexBuckets.assign(IndexBuckets.begin(), IndexBuckets.end());
            table->CTRBlob.assign(CTRBlob.begin(), CTRBlob.end());
        }
    };
public:

    TCtrValueTable()
        : Impl(TSolidTable())
    {
    }

    template<typename T>
    TConstArrayRef<T> GetTypedArrayRefForBlobData() const {
        if (Impl.Is<TSolidTable>()) {
            auto& solid = Impl.As<TSolidTable>();
            return MakeArrayRef(
                reinterpret_cast<const T*>(solid.CTRBlob.data()),
                solid.CTRBlob.size() / sizeof(T)
            );
        } else {
            auto& thin = Impl.As<TThinTable>();
            return MakeArrayRef(
                reinterpret_cast<const T*>(thin.CTRBlob.data()),
                thin.CTRBlob.size() / sizeof(T)
            );
        }
    }

    template<typename T>
    TArrayRef<T> AllocateBlobAndGetArrayRef(size_t elementCount) {
        auto& solid = Impl.As<TSolidTable>();
        solid.CTRBlob.resize(elementCount * sizeof(T));
        std::fill(solid.CTRBlob.begin(), solid.CTRBlob.end(), 0);
        return MakeArrayRef(
            reinterpret_cast<T*>(solid.CTRBlob.data()),
            elementCount
        );
    }

    NCatboost::TDenseIndexHashView GetIndexHashViewer() const {
        if (Impl.Is<TSolidTable>()) {
            auto& solid = Impl.As<TSolidTable>();
            return NCatboost::TDenseIndexHashView(solid.IndexBuckets);
        } else {
            auto& thin = Impl.As<TThinTable>();
            return NCatboost::TDenseIndexHashView(thin.IndexBuckets);
        }
    }

    NCatboost::TDenseIndexHashBuilder GetIndexHashBuilder(size_t uniqueValuesCount) {
        auto& solid = Impl.As<TSolidTable>();
        auto bucketCount = NCatboost::TDenseIndexHashBuilder::GetProperBucketsCount(uniqueValuesCount);
        solid.IndexBuckets.resize(bucketCount);
        return NCatboost::TDenseIndexHashBuilder(solid.IndexBuckets);
    }
    void Save(IOutputStream* s) const;

    void Load(IInputStream* s);

    void LoadSolid(void* buf, size_t length);

    bool operator==(const TCtrValueTable& other) const {
        return std::tie(CounterDenominator, TargetClassesCount, Impl) ==
               std::tie(other.CounterDenominator, other.TargetClassesCount, other.Impl);
    }

public:
    TModelCtrBase ModelCtrBase;
    int CounterDenominator = 0;
    int TargetClassesCount = 0;
private:
    TVariant<TSolidTable, TThinTable> Impl;
};
