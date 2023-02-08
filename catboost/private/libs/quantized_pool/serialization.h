#pragma once

#include <catboost/libs/data/loader.h>
#include <catboost/private/libs/data_util/path_with_scheme.h>
#include <catboost/libs/data/data_provider.h>

#include <util/generic/fwd.h>
#include <util/stream/fwd.h>


namespace NCB {
    struct TQuantizedPool;
    struct TQuantizedPoolDigest;

    class TSrcColumnBase {
    public:
        EColumn Type;

    public:
        explicit TSrcColumnBase(EColumn type)
            : Type(type)
        {}

        virtual ~TSrcColumnBase() = default;
    };

    template <class T>
    class TSrcColumn : public TSrcColumnBase {
    public:
        TVector<TVector<T>> Data;

    public:
        explicit TSrcColumn(EColumn type)
            : TSrcColumnBase(type)
        {}

        TSrcColumn(EColumn type, TVector<TVector<T>>&& data)
            : TSrcColumnBase(type)
            , Data(std::move(data))
        {}
    };

    struct TSrcData {
        size_t DocumentCount;

        TVector<size_t> LocalIndexToColumnIndex; // [localIndex]
        TPoolQuantizationSchema PoolQuantizationSchema;
        TVector<TString> ColumnNames;

        // Objects data
        TMaybe<TSrcColumn<TGroupId>> GroupIds;
        TMaybe<TSrcColumn<TSubgroupId>> SubgroupIds;

        // TODO(akhropov): not yet supported by quantized pools format. MLTOOLS-2412.
        // TMaybe<TSrcColumn<ui64>> Timestamp;

        TVector<THolder<TSrcColumnBase>> FloatFeatures;

        TVector<THolder<TSrcColumnBase>> CatFeatures;

        // Target data
        TMaybe<TSrcColumn<float>> Target;
        TVector<TSrcColumn<float>> Baseline;
        TMaybe<TSrcColumn<float>> Weights;
        TMaybe<TSrcColumn<float>> GroupWeights;

        TStringBuf PairsFileData;
        TString PairsFilePathScheme = "dsv-flat";

        TStringBuf GroupWeightsFileData;
        TStringBuf BaselineFileData;

        TVector<size_t> IgnoredColumnIndices; // saved in quantized pool

        TVector<ui32> IgnoredFeatures; // passed in args to loader

        EObjectsOrder ObjectsOrder = EObjectsOrder::Undefined;
    };

    namespace NIdl {
        class TPoolMetainfo;
        class TPoolQuantizationSchema;
    }
}

namespace NCB {
    void SaveQuantizedPool(const TQuantizedPool& pool, IOutputStream* output);
    void SaveQuantizedPool(const TSrcData& srcData, TString fileName);
    void SaveQuantizedPool(const TDataProviderPtr& dataProvider, TString fileName);

    static constexpr size_t QUANTIZED_POOL_COLUMN_DEFAULT_SLICE_COUNT = 512 * 1024;

    template<class T>
    TSrcColumn<T> GenerateSrcColumn(TConstArrayRef<T> data, EColumn columnType) {
        TSrcColumn<T> dst(columnType);

        for (size_t idx = 0; idx < data.size(); ) {
            size_t chunkSize = Min(
                data.size() - idx,
                QUANTIZED_POOL_COLUMN_DEFAULT_SLICE_COUNT
            );
            dst.Data.push_back(TVector<T>(data.begin() + idx, data.begin() + idx + chunkSize));
            idx += chunkSize;
        }
        return dst;
    }

    struct TLoadQuantizedPoolParameters {
        bool LockMemory = true;
        bool Precharge = true;
        TDatasetSubset DatasetSubset;
    };

    // Load quantized pool saved by `SaveQuantizedPool` from file.
    TQuantizedPool LoadQuantizedPool(const TPathWithScheme& pathWithScheme, const TLoadQuantizedPoolParameters& params);

    NIdl::TPoolQuantizationSchema LoadQuantizationSchemaFromPool(TStringBuf path);
    NIdl::TPoolMetainfo LoadPoolMetainfo(TStringBuf path);
    TQuantizedPoolDigest CalculateQuantizedPoolDigest(TStringBuf path);
    TQuantizedPoolDigest GetQuantizedPoolDigest(
        const NIdl::TPoolMetainfo& poolMetainfo,
        const NIdl::TPoolQuantizationSchema& quantizationSchema
    );
    size_t EstimateIdsLength(const TStringBuf path);
    void EstimateGroupSize(const TStringBuf path, double* groupSize, double* sqrGroupSize, size_t* maxGroupSize);
    void AddPoolMetainfo(const NIdl::TPoolMetainfo& metainfo, TQuantizedPool* const pool);
}
