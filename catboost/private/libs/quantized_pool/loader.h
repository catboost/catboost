#pragma once

#include "pool.h"
#include "serialization.h"

#include <catboost/libs/data/loader.h>
#include <catboost/private/libs/index_range/index_range.h>

#include <library/cpp/object_factory/object_factory.h>

#include <util/generic/ptr.h>
#include <util/generic/singleton.h>
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
        TPathWithScheme GraphPath;
        TPathWithScheme GroupWeightsPath;
        TPathWithScheme BaselinePath;
        TPathWithScheme TimestampsPath;
        TPathWithScheme FeatureNamesPath;
        TPathWithScheme PoolMetaInfoPath;
        TDataMetaInfo DataMetaInfo;
        EObjectsOrder ObjectsOrder;
        TDatasetSubset DatasetSubset;
    };

    struct IQuantizedPoolLoader {
        virtual ~IQuantizedPoolLoader() = default;
        virtual void LoadQuantizedPool(TLoadQuantizedPoolParameters params) = 0;
        virtual TQuantizedPool ExtractQuantizedPool() = 0;
        virtual TVector<ui8> LoadQuantizedColumn(ui32 columnIdx) = 0;
        virtual TVector<ui8> LoadQuantizedColumn(ui32 columnIdx, ui64 offset, ui64 count) = 0;
        virtual TPathWithScheme GetPoolPathWithScheme() const = 0;
    };

    using TQuantizedPoolLoaderFactory =
        NObjectFactory::TParametrizedObjectFactory<IQuantizedPoolLoader, TString, const TPathWithScheme&>;

    class TQuantizedPoolLoadersCache {
    public:
        static TAtomicSharedPtr<IQuantizedPoolLoader> GetLoader(
            const TPathWithScheme& pathWithScheme,
            TDatasetSubset loadSubset);
        static bool HaveLoader(const TPathWithScheme& pathWithScheme, TDatasetSubset loadSubset);
        static void DropAllLoaders();

    private:
        THashMap<std::pair<TPathWithScheme, TDatasetSubset>, TAtomicSharedPtr<IQuantizedPoolLoader>> Cache;
        TAdaptiveLock Lock;
        inline static TQuantizedPoolLoadersCache& GetRef() {
            return *Singleton<TQuantizedPoolLoadersCache>();
        }
        Y_DECLARE_SINGLETON_FRIEND();
    };
}
