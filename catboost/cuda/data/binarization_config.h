#pragma once

#include "feature.h"

#include <util/system/types.h>
#include <library/grid_creator/binarization.h>
#include <library/binsaver/bin_saver.h>

namespace NCatboostCuda
{
    struct TBinarizationDescription
    {
        TBinarizationDescription() = default;

        TBinarizationDescription(EBorderSelectionType borderSelectionType, const ui32& discretization)
                : BorderSelectionType(borderSelectionType)
                  , Discretization(discretization)
        {
        }

        EBorderSelectionType BorderSelectionType = EBorderSelectionType::MinEntropy;
        ui32 Discretization = 15;
    };

    struct TFeatureBinarizationConfiguration
    {
        TBinarizationDescription BinarizationDescription;

        yvector<float> Borders;
        bool HasBordersFlag = false;

        bool HasBorders() const
        {
            return (bool) HasBordersFlag;
        }

        const yvector<float>& GetBorders() const
        {
            return Borders;
        }
    };

    struct TFloatFeatureBinarizationConfiguration
    {
        TFeatureBinarizationConfiguration DefaultBinarization;
        ymap<ui32, TFeatureBinarizationConfiguration> CustomBinarization;

        const TFeatureBinarizationConfiguration& GetForFeature(ui32 featureId) const
        {
            if (CustomBinarization.has(featureId))
            {
                return CustomBinarization.at(featureId);
            }
            return DefaultBinarization;
        }
    };

    struct TBinarizationConfiguration
    {
        TBinarizationDescription DefaultFloatBinarization = TBinarizationDescription(EBorderSelectionType::MinEntropy, 128);
        TBinarizationDescription DefaultCtrBinarization = TBinarizationDescription(EBorderSelectionType::Uniform, 15);
        TBinarizationDescription DefaultTreeCtrBinarization = TBinarizationDescription(EBorderSelectionType::Uniform, 15);
        TBinarizationDescription TargetBinarization = TBinarizationDescription(EBorderSelectionType::GreedyLogSum, 1);
        TBinarizationDescription FreqTreeCtrBinarization = TBinarizationDescription(EBorderSelectionType::Median, 15);
        TBinarizationDescription FreqCtrBinarization = TBinarizationDescription(EBorderSelectionType::GreedyLogSum, 15);

        ymap<ui32, TBinarizationDescription> CustomBinarization;

        SAVELOAD(DefaultFloatBinarization, DefaultCtrBinarization,
                 DefaultTreeCtrBinarization, CustomBinarization, TargetBinarization);
    };
}
