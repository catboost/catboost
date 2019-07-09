#pragma once

#include "fwd.h"

#include "features.h"

#include <util/generic/array_ref.h>
#include <util/generic/maybe.h>
#include <util/generic/ptr.h>

namespace NCB {  // split due to CUDA-compiler inability to parse nested namespace definitions
    namespace NModelEvaluation {

        class IQuantizedData : public TThrRefBase {
        public:
            virtual size_t GetObjectsCount() const = 0;
        };

        class TFeatureLayout {
        public:
            TMaybe<TVector<ui32>> FloatFeatureIndexes;
            TMaybe<TVector<ui32>> CatFeatureIndexes;
            TMaybe<TVector<ui32>> FlatIndexes;
        public:
            inline TFeaturePosition AdjustFeature(const TFloatFeature& feature) const {
                TFeaturePosition position = feature.Position;
                if (FloatFeatureIndexes.Defined()) {
                    position.Index = FloatFeatureIndexes->at(position.Index);
                }
                if (FlatIndexes.Defined()) {
                    position.FlatIndex = FlatIndexes->at(position.FlatIndex);
                }
                return position;
            }

            inline TFeaturePosition AdjustFeature(const TCatFeature& feature) const {
                TFeaturePosition position = feature.Position;
                if (CatFeatureIndexes.Defined()) {
                    position.Index = CatFeatureIndexes->at(position.Index);
                }
                if (FlatIndexes.Defined()) {
                    position.FlatIndex = FlatIndexes->at(position.FlatIndex);
                }
                return position;
            }
        };

        class IModelEvaluator {
        public:
            virtual ~IModelEvaluator() = default;

            virtual void SetPredictionType(EPredictionType type) = 0;
            virtual EPredictionType GetPredictionType() const = 0;

            virtual TModelEvaluatorPtr Clone() const = 0;

            virtual i32 GetApproxDimension() const = 0;
            virtual size_t GetTreeCount() const = 0;

            virtual void SetFeatureLayout(const TFeatureLayout& featureLayout) = 0;

            virtual void CalcFlatTransposed(
                TConstArrayRef<TConstArrayRef<float>> transposedFeatures,
                size_t treeStart,
                size_t treeEnd,
                TArrayRef<double> results,
                const TFeatureLayout* featureInfo = nullptr
            ) const = 0;

            virtual void CalcFlat(
                TConstArrayRef<TConstArrayRef<float>> features,
                size_t treeStart,
                size_t treeEnd,
                TArrayRef<double> results,
                const TFeatureLayout* featureInfo = nullptr
            ) const = 0;

            void CalcFlat(
                TConstArrayRef<TConstArrayRef<float>> features,
                TArrayRef<double> results,
                const TFeatureLayout* featureInfo = nullptr
            ) const {
                CalcFlat(features, 0, GetTreeCount(), results, featureInfo);
            }

            virtual void CalcFlatSingle(
                TConstArrayRef<float> features,
                size_t treeStart,
                size_t treeEnd,
                TArrayRef<double> results,
                const TFeatureLayout* featureInfo = nullptr
            ) const = 0;

            virtual void Calc(
                TConstArrayRef<TConstArrayRef<float>> floatFeatures,
                TConstArrayRef<TConstArrayRef<int>> catFeatures,
                size_t treeStart,
                size_t treeEnd,
                TArrayRef<double> results,
                const TFeatureLayout* featureInfo = nullptr
            ) const = 0;

            virtual void Calc(
                TConstArrayRef<TConstArrayRef<float>> floatFeatures,
                TConstArrayRef<TVector<TStringBuf>> catFeatures,
                size_t treeStart,
                size_t treeEnd,
                TArrayRef<double> results,
                const TFeatureLayout* featureInfo = nullptr
            ) const = 0;

            template <typename TCatFeatureType>
            void Calc(
                TConstArrayRef<TConstArrayRef<float>> floatFeatures,
                TConstArrayRef<TConstArrayRef<TCatFeatureType>> catFeatures,
                TArrayRef<double> results,
                const TFeatureLayout* featureInfo = nullptr
            ) const {
                Calc(floatFeatures, catFeatures, 0, GetTreeCount(), results, featureInfo);
            }

            virtual void Calc(
                const IQuantizedData* quantizedFeatures,
                size_t treeStart,
                size_t treeEnd,
                TArrayRef<double> results
            ) const = 0;

            virtual void CalcLeafIndexesSingle(
                TConstArrayRef<float> floatFeatures,
                TConstArrayRef<TStringBuf> catFeatures,
                size_t treeStart,
                size_t treeEnd,
                TArrayRef<TCalcerIndexType> indexes,
                const TFeatureLayout* featureInfo = nullptr
            ) const = 0;

            virtual void CalcLeafIndexes(
                TConstArrayRef<TConstArrayRef<float>> floatFeatures,
                TConstArrayRef<TVector<TStringBuf>> catFeatures,
                size_t treeStart,
                size_t treeEnd,
                TArrayRef<TCalcerIndexType> indexes,
                const TFeatureLayout* featureInfo = nullptr
            ) const = 0;

            virtual void CalcLeafIndexes(
                const IQuantizedData* quantizedFeatures,
                size_t treeStart,
                size_t treeEnd,
                TArrayRef<TCalcerIndexType> indexes
            ) const = 0;
        };

        TModelEvaluatorPtr CreateCpuEvaluator(const TFullModel& model);

        bool CudaEvaluationPossible(const TFullModel& model);
        TModelEvaluatorPtr CreateGpuEvaluator(const TFullModel& model);

        class ILeafIndexCalcer {
        public:
            virtual ~ILeafIndexCalcer() = default;

            virtual bool Next() = 0;
            virtual bool CanGet() const = 0;
            virtual TVector<TCalcerIndexType> Get() const = 0;
        };
    }
}
