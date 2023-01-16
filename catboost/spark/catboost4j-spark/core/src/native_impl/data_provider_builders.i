%{
#include <catboost/spark/catboost4j-spark/core/src/native_impl/data_provider_builders.h>
#include <catboost/libs/data/data_provider_builders.h>
#include <catboost/libs/data/quantized_features_info.h>
#include <catboost/libs/data/unaligned_mem.h>
#include <catboost/libs/data/visitor.h>
#include <catboost/libs/helpers/exception.h>

#include <util/generic/fwd.h>
#include <util/generic/array_ref.h>
#include <util/generic/cast.h>
#include <util/generic/yexception.h>
%}

%include <enums.swg>

%include "data_provider.i"
%include "maybe_owning_array_holder.i"
%include "intrusive_ptr.i"
%include "tvector.i"

%include <bindings/swiglib/stroka.swg>


%template(TVector_TMaybeOwningConstArrayHolder_float) TVector<NCB::TMaybeOwningConstArrayHolder<float>>;
%template(TVector_TMaybeOwningConstArrayHolder_i32) TVector<NCB::TMaybeOwningConstArrayHolder<i32>>;


%{

template <class TDst, class TSrc>
NCB::TUnalignedArrayBuf<TDst> AsUnalignedBuf(TConstArrayRef<TSrc> data) {
    return NCB::TUnalignedArrayBuf<TDst>((const TDst*)data.data(), data.size() * sizeof(TSrc));
}

%}


namespace NCB {

    %javaconst(1);
    enum class EDatasetVisitorType {
        RawObjectsOrder,
        RawFeaturesOrder,
        QuantizedFeatures
    };

    // Adapt for SWIG + JVM
    class IQuantizedFeaturesDataVisitor {
    public:
        %extend {
            void Start(
                const TDataMetaInfo& metaInfo,
                i32 objectCount,
                const NCB::TQuantizedFeaturesInfo& quantizedFeaturesInfo
            ) {
                self->Start(
                    metaInfo,
                    SafeIntegerCast<ui32>(objectCount),
                    NCB::EObjectsOrder::Undefined,
                    TVector<TIntrusivePtr<NCB::IResourceHolder>>(),
                    NCB::GetPoolQuantizationSchema(quantizedFeaturesInfo, metaInfo.ClassLabels),
                    /*wholeColumns*/ true
                );
            }

            // TCommonObjectsData

            void AddGroupId(TConstArrayRef<i64> groupIdData) {
                self->AddGroupIdPart(0, AsUnalignedBuf<TGroupId>(groupIdData));
            }

            void AddSubgroupId(TConstArrayRef<i32> subgroupIdData) {
                self->AddSubgroupIdPart(0, AsUnalignedBuf<TSubgroupId>(subgroupIdData));
            }

            void AddTimestamp(TConstArrayRef<i64> timestampData) {
                self->AddTimestampPart(0, AsUnalignedBuf<ui64>(timestampData));
            }


            // TQuantizedObjectsData

            void AddFeature(
                const NCB::TFeaturesLayout& featuresLayout,
                i32 flatFeatureIdx,
                i32 objectCount,
                i8 bitsPerDocumentFeature,
                TVector<i64>* featureDataBuffer // moved into per-object data size depends on BitsPerKey
            ) {
                size_t sizeInBytes =
                    SafeIntegerCast<size_t>(objectCount)
                    * CeilDiv<size_t>(bitsPerDocumentFeature, CHAR_BIT);

                auto dataArray = TConstArrayRef<ui8>((ui8*)featureDataBuffer->data(), sizeInBytes);
                auto dataArrayHolder = NCB::TMaybeOwningConstArrayHolder<ui8>::CreateOwning(
                    dataArray,
                    MakeIntrusive<NCB::TVectorHolder<i64>>(std::move(*featureDataBuffer))
                );
                auto featureType = featuresLayout.GetExternalFeatureType(flatFeatureIdx);
                switch (featureType) {
                    case EFeatureType::Float:
                        self->AddFloatFeaturePart(
                            SafeIntegerCast<ui32>(flatFeatureIdx),
                            0,
                            SafeIntegerCast<ui8>(bitsPerDocumentFeature),
                            dataArrayHolder
                        );
                        break;
                    case EFeatureType::Categorical:
                        self->AddCatFeaturePart(
                            SafeIntegerCast<ui32>(flatFeatureIdx),
                            0,
                            SafeIntegerCast<ui8>(bitsPerDocumentFeature),
                            dataArrayHolder
                        );
                        break;
                    default:
                        CB_ENSURE(false, "AddFeature: Unsupported feature type: " << featureType);
                }
            }


            // TRawTargetData

            void AddTarget(TConstArrayRef<float> targetData) {
                self->AddTargetPart(0, AsUnalignedBuf<float>(targetData));
            }

            void AddTarget(TVector<TString>* targetData) {
                self->AddTargetPart(
                    0,
                    NCB::TMaybeOwningConstArrayHolder<TString>::CreateOwning(std::move(*targetData))
                );
            }

            void AddBaseline(i32 baselineIdx, TConstArrayRef<float> baselineData) {
                self->AddBaselinePart(
                    0,
                    SafeIntegerCast<i32>(baselineIdx),
                    AsUnalignedBuf<float>(baselineData)
                );
            }

            void AddWeight(TConstArrayRef<float> weightData) {
                self->AddWeightPart(0, AsUnalignedBuf<float>(weightData));
            }

            void AddGroupWeight(TConstArrayRef<float> groupWeightData) {
                self->AddGroupWeightPart(0, AsUnalignedBuf<float>(groupWeightData));
            }
        }

        virtual void Finish() = 0;
    };

    struct TDataProviderBuilderOptions {
        //bool GpuDistributedFormat = false;
        //TPathWithScheme PoolPath = TPathWithScheme();
        ui64 MaxCpuRamUsage = Max<ui64>();
        bool SkipCheck = false; // to increase speed, esp. when applying
        //ESparseArrayIndexingType SparseArrayIndexingType = ESparseArrayIndexingType::Undefined;
    };

}

%include "data_provider_builders.h"
