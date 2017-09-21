#pragma once

#include "feature.h"

#include <util/system/types.h>
#include <library/grid_creator/binarization.h>
#include <library/binsaver/bin_saver.h>

struct TBinarizationDescription {
    EBorderSelectionType BorderSelectionType = EBorderSelectionType::GreedyLogSum;
    ui32 Discretization = 16;
};

struct TFeatureBinarizationConfiguration {
    TBinarizationDescription BinarizationDescription;

    yvector<float> Borders;
    bool HasBordersFlag = false;

    bool HasBorders() const {
        return (bool)HasBordersFlag;
    }

    const yvector<float>& GetBorders() const {
        return Borders;
    }
};

struct TFloatFeatureBinarizationConfiguration {
    TFeatureBinarizationConfiguration DefaultBinarization;
    ymap<ui32, TFeatureBinarizationConfiguration> CustomBinarization;

    const TFeatureBinarizationConfiguration& GetForFeature(ui32 featureId) const {
        if (CustomBinarization.has(featureId)) {
            return CustomBinarization.at(featureId);
        }
        return DefaultBinarization;
    }
};

struct TBinarizationConfiguration {
    TBinarizationDescription DefaultFloatBinarization;
    TBinarizationDescription DefaultCtrBinarization;
    TBinarizationDescription DefaultTreeCtrBinarization;
    TBinarizationDescription TargetBinarization;
    TBinarizationDescription FreqTreeCtrBinarization = {EBorderSelectionType::Median, 16};
    TBinarizationDescription FreqCtrBinarization = {EBorderSelectionType::GreedyLogSum, 16};

    ymap<ui32, TBinarizationDescription> CustomBinarization;

    SAVELOAD(DefaultFloatBinarization, DefaultCtrBinarization,
             DefaultTreeCtrBinarization, CustomBinarization, TargetBinarization);
};
