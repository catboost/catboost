#pragma once

#include "enums.h"
#include "split.h"

#include <catboost/libs/helpers/exception.h>
#include <catboost/libs/model/model.h>
#include <catboost/libs/model/split.h>

#include <util/generic/fwd.h>
#include <util/generic/hash.h>
#include <util/generic/vector.h>
#include <util/generic/xrange.h>
#include <util/string/cast.h>

namespace NMonoForest {
    class IGrid {
    public:
        virtual ~IGrid() {
        }
        virtual int FeatureCount() const = 0;
        virtual int BinFeatureCount() const = 0;
        virtual int BorderCount(int featureIdx) const = 0;
        virtual float Border(int featureIdx, int borderIdx) const = 0;
        virtual EFeatureType FeatureType(int featureIdx) const = 0;
        virtual int FeatureIndex(int binFeatureIdx) const = 0;
        virtual int BinFeatureIndex(int featureIdx, int borderIndex) const = 0;
        virtual int ExternalFlatFeatureIndex(int featureIdx) const = 0;
    };

    class TCatBoostGrid: public IGrid {
    public:
        TCatBoostGrid() = default;
        explicit TCatBoostGrid(const TFullModel& model);

        int FeatureCount() const override {
            return Borders.size();
        }

        int BorderCount(int featureIdx) const override {
            CB_ENSURE(featureIdx < FeatureCount(), "Feature index exceeds features count");
            return Borders[featureIdx].size();
        }

        float Border(int featureIdx, int borderIdx) const override {
            CB_ENSURE(featureIdx < FeatureCount(), "Feature index exceeds features count");
            CB_ENSURE(borderIdx < static_cast<int>(Borders[featureIdx].size()), "Border index exceeds borders count");
            return Borders[featureIdx][borderIdx];
        }

        EFeatureType FeatureType(int featureIdx) const override {
            CB_ENSURE(featureIdx < FeatureCount(), "Feature index exceeds features count");
            return FeatureTypes[featureIdx];
        }

        int BinFeatureCount() const override {
            return BinFeatures.size();
        }

        int FeatureIndex(int binFeatureIndex) const override {
            CB_ENSURE(binFeatureIndex < BinFeatureCount(), "BinFeature index exceeds binFeatures count");
            return BinFeatures[binFeatureIndex].FeatureId;
        }

        int BinFeatureIndex(int featureIdx, int borderIdx) const override {
            CB_ENSURE(featureIdx < FeatureCount(), "Feature index exceeds features count");
            CB_ENSURE(borderIdx < static_cast<int>(Borders[featureIdx].size()), "Border index exceeds borders count");
            return BorderIdxToBinFeature[featureIdx][borderIdx];
        }

        int ExternalFlatFeatureIndex(int featureIdx) const override {
            CB_ENSURE(featureIdx < FeatureCount(), "Feature index exceeds features count");
            return InternalToExternalFeature[featureIdx];
        }

        TBinarySplit ToBinarySplit(const TModelSplit& split) const {
            switch (split.Type) {
                case ESplitType::FloatFeature: {
                    const auto featureIdx = FloatFeatureToInternalFeature.at(split.FloatFeature.FloatFeature);
                    const auto borderIdx = BorderToIdx[featureIdx].at(split.FloatFeature.Split);
                    return TBinarySplit(featureIdx, borderIdx, EBinSplitType::TakeGreater);
                }
                case ESplitType::OneHotFeature: {
                    const auto featureIdx = CatFeatureToInternalFeature.at(split.OneHotFeature.CatFeatureIdx);
                    const auto borderIdx = BorderToIdx[featureIdx].at(split.OneHotFeature.Value);
                    return TBinarySplit(featureIdx, borderIdx, EBinSplitType::TakeBin);
                }
                default:
                    CB_ENSURE(false, "Unimplemented");
            }
        }

    private:
        THashMap<int, int> FloatFeatureToInternalFeature;
        THashMap<int, int> CatFeatureToInternalFeature;
        TVector<int> InternalToExternalFeature;
        TVector<TVector<float>> Borders;
        TVector<THashMap<float, int>> BorderToIdx;
        TVector<TVector<int>> BorderIdxToBinFeature;
        TVector<EFeatureType> FeatureTypes;
        TVector<TBinarySplit> BinFeatures;
    };
}
