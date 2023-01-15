#pragma once

#include <catboost/libs/data/loader.h>
#include <catboost/private/libs/data_util/path_with_scheme.h>
#include <catboost/libs/data/data_provider.h>

#include <util/generic/fwd.h>
#include <util/stream/fwd.h>


namespace NCB {
    struct TQuantizedPool;
    struct TQuantizedPoolDigest;

    template <class T>
    struct TSrcColumn {
        EColumn Type;
        TVector<TVector<T>> Data;
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

        TVector<TMaybe<TSrcColumn<ui8>>> FloatFeatures;

        // TODO(akhropov): not yet supported by quantized pools format. MLTOOLS-1957.
        // TVector<TMaybe<TSrcColumn<TStringBuf>>> CatFeatures;

        // Target data
        TMaybe<TSrcColumn<float>> Target;
        TVector<TSrcColumn<float>> Baseline;
        TMaybe<TSrcColumn<float>> Weights;
        TMaybe<TSrcColumn<float>> GroupWeights;

        TStringBuf PairsFileData;
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
    //only for used C++
    void SaveQuantizedPool(const TQuantizedPool& pool, IOutputStream* output);
    void SaveQuantizedPool(const TSrcData& srcData, TString fileName);
    //only for python
    void SaveQuantizedPool(const TDataProviderPtr& dataProvider, TString fileName);

    template<class T>
    TSrcColumn<T> GenerateSrcColumn(TConstArrayRef<T> data, EColumn columnType);

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
    void AddPoolMetainfo(const NIdl::TPoolMetainfo& metainfo, TQuantizedPool* const pool);
}
