#pragma once

#include <catboost/private/libs/options/enums.h>
#include <catboost/private/libs/text_features/feature_calcer.h>
#include <catboost/libs/helpers/exception.h>
#include <catboost/libs/helpers/guid.h>
#include <util/system/types.h>
#include <util/generic/vector.h>
#include <util/generic/maybe.h>
#include <util/generic/array_ref.h>
#include <library/cpp/threading/local_executor/local_executor.h>

/*
 * Feature estimators - how to build FeatureCalculator from DataProvider
 */
namespace NCB {

    class TCalculatedFeatureVisitor {
    public:
        using TSingleFeatureWriter = std::function<void(ui32, TConstArrayRef<float>)>;
        using TPackedFeatureWriter = std::function<void(TConstArrayRef<ui32>, TConstArrayRef<ui32>)>;

        explicit TCalculatedFeatureVisitor(TSingleFeatureWriter&& singleFeatureWriter)
            : SingleFeatureWriter(std::move(singleFeatureWriter))
        {}

        explicit TCalculatedFeatureVisitor(TPackedFeatureWriter&& packedFeatureWriter)
            : PackedFeatureWriter(std::move(packedFeatureWriter))
        {}

        void operator()(ui32 featureIndex, TConstArrayRef<float> values) const {
            CB_ENSURE(
                SingleFeatureWriter.Defined(),
                "Attempt to call single feature writer on packed feature writer"
            );
            (*SingleFeatureWriter)(featureIndex, values);
        }

        void operator()(TConstArrayRef<ui32> featureIds, TConstArrayRef<ui32> packedValues) const {
            CB_ENSURE(
                PackedFeatureWriter.Defined(),
                "Attempt to call packed feature writer on single feature writer"
            );
            (*PackedFeatureWriter)(featureIds, packedValues);
        }

    private:
        TMaybe<TSingleFeatureWriter> SingleFeatureWriter;
        TMaybe<TPackedFeatureWriter> PackedFeatureWriter;
    };

    //We need this so we could prepare layout before computation of features
    struct TEstimatedFeaturesMeta {
        ui32 FeaturesCount = 0;
        TMaybe<TVector<ui32>> UniqueValuesUpperBoundHint;
        TVector<EFeatureCalcerType> Type;
    };


    /*
     * Extracts `derived` features
     * For examples, could be CTRs (but i don't think will transfer current logic here in near future)
     * Or TF-IDF based on Text
     * Or FTLR model (online linear model)
     * And so on
     *
     * This class is not stateless. Will compute learn and test features for fixed input data (specified on creation)
     */
    class IFeatureEstimator : public TThrRefBase {
    public:

        //Meta to allocate memory for datasets, for example
        //Or to estimate necessary memory
        virtual TEstimatedFeaturesMeta FeaturesMeta() const = 0;


        //for some type of features it is not efficient ot compute learn/test features in 2 different functions
        //especially on GPU
        virtual void ComputeFeatures(TCalculatedFeatureVisitor learnVisitor,
                                     TConstArrayRef<TCalculatedFeatureVisitor> testVisitors,
                                     NPar::ILocalExecutor* executor) const = 0;


        virtual THolder<IFeatureCalcer> MakeFinalFeatureCalcer(TConstArrayRef<ui32> featureIndices,
                                                               NPar::ILocalExecutor* executor) const {
            Y_UNUSED(featureIndices);
            Y_UNUSED(executor);
            CB_ENSURE(false, "Final feature calcer is unimplemented yet");
        }

        virtual EFeatureType GetSourceType() const = 0;

        virtual TGuid Id() const = 0;
    };


    //so we could compare online setting vs using all data
    class IOnlineFeatureEstimator : public IFeatureEstimator {
    public:

        //target/weights/permutation are for learn only
        virtual void ComputeOnlineFeatures(TConstArrayRef<ui32> learnPermutation,
                                           TCalculatedFeatureVisitor learnVisitor,
                                           TConstArrayRef<TCalculatedFeatureVisitor> testVisitors,
                                           NPar::ILocalExecutor* executor) const = 0;


    };


    using TFeatureEstimatorPtr = TIntrusiveConstPtr<IFeatureEstimator>;
    using TOnlineFeatureEstimatorPtr = TIntrusiveConstPtr<IOnlineFeatureEstimator>;
}
