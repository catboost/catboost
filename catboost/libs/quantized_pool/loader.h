#pragma once

#include "pool.h"
#include "serialization.h"

#include <catboost/libs/data_new/loader.h>
#include <catboost/libs/index_range/index_range.h>

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
            const size_t* flatFeatureIdx,
            const size_t* baselineIdx,
            IQuantizedFeaturesDataVisitor* visitor) const;

        void AddQuantizedFeatureChunk(
            const TQuantizedPool::TChunkDescription& chunk,
            const size_t flatFeatureIdx,
            IQuantizedFeaturesDataVisitor* visitor) const;

        static TLoadQuantizedPoolParameters GetLoadParameters() {
            return {/*LockMemory*/ false, /*Precharge*/ false};
        }

    private:
        ui32 ObjectCount;
        TVector<bool> IsFeatureIgnored;
        TQuantizedPool QuantizedPool;
        TPathWithScheme PairsPath;
        TPathWithScheme GroupWeightsPath;
        TPathWithScheme BaselinePath;
        TDataMetaInfo DataMetaInfo;
        EObjectsOrder ObjectsOrder;
    };

    struct TLoadSubset {
        TIndexRange<ui32> Range = {0, Max<ui32>()};
        bool SkipFeatures = false;
    };

    struct IQuantizedPoolLoader {
        virtual ~IQuantizedPoolLoader() = default;
        virtual TQuantizedPool LoadQuantizedPool(
            TLoadQuantizedPoolParameters params,
            TLoadSubset loadSubset) = 0;
    };

    using TQuantizedPoolLoaderFactory =
        NObjectFactory::TParametrizedObjectFactory<IQuantizedPoolLoader, TString, const TPathWithScheme&>;
}
