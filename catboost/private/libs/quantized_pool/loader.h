#pragma once

#include "pool.h"
#include "serialization.h"

#include <catboost/libs/data/loader.h>
#include <catboost/private/libs/index_range/index_range.h>

#include <library/object_factory/object_factory.h>

#include <util/generic/ylimits.h>

namespace NCB {
    class TCBQuantizedDataLoader : public IQuantizedFeaturesDatasetLoader {
    public:
        explicit TCBQuantizedDataLoader(TDatasetLoaderPullArgs&& args);

        void Do(IQuantizedFeaturesDataVisitor* visitor) override;

    private:
        void AddChunk(
            const TQuantizedPool::TChunkDescription& chunk,
            EColumn columnType,
            const size_t* const multiTargetIdx,
            const size_t* flatFeatureIdx,
            const size_t* baselineIdx,
            IQuantizedFeaturesDataVisitor* visitor) const;

        void AddQuantizedFeatureChunk(
            const TQuantizedPool::TChunkDescription& chunk,
            const size_t flatFeatureIdx,
            IQuantizedFeaturesDataVisitor* visitor) const;

        void AddQuantizedCatFeatureChunk(
            const TQuantizedPool::TChunkDescription& chunk,
            const size_t flatFeatureIdx,
            IQuantizedFeaturesDataVisitor* visitor) const;

        TConstArrayRef<ui8> ClipByDatasetSubset(const TQuantizedPool::TChunkDescription& chunk) const;
        ui32 GetDatasetOffset(const TQuantizedPool::TChunkDescription& chunk) const;

        static TLoadQuantizedPoolParameters GetLoadParameters(NCB::TDatasetSubset loadSubset) {
            return {/*LockMemory*/ false, /*Precharge*/ false, loadSubset};
        }

    private:
        ui32 ObjectCount;
        TVector<bool> IsFeatureIgnored;
        TQuantizedPool QuantizedPool;
        TPathWithScheme PairsPath;
        TPathWithScheme GroupWeightsPath;
        TPathWithScheme BaselinePath;
        TPathWithScheme TimestampsPath;
        TPathWithScheme FeatureNamesPath;
        TDataMetaInfo DataMetaInfo;
        EObjectsOrder ObjectsOrder;
        TDatasetSubset DatasetSubset;
    };

    struct IQuantizedPoolLoader {
        virtual ~IQuantizedPoolLoader() = default;
        virtual TQuantizedPool LoadQuantizedPool(TLoadQuantizedPoolParameters params) = 0;
        virtual TVector<ui8> LoadQuantizedColumn(ui32 columnIdx) = 0;
    };

    using TQuantizedPoolLoaderFactory =
        NObjectFactory::TParametrizedObjectFactory<IQuantizedPoolLoader, TString, const TPathWithScheme&>;
}
