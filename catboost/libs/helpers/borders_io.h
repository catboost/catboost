#pragma once

#include <catboost/private/libs/options/enums.h>

#include <util/generic/maybe.h>
#include <util/generic/vector.h>
#include <util/stream/output.h>


namespace NCB {
    void ParseBordersFileLine(
        const TString& line,
        ui32* flatFeatureIdx,
        float* border,
        TMaybe<ENanMode>* nanMode);

    void OutputFeatureBorders(
        ui32 flatFeatureIdx,
        const TVector<float>& borders,
        ENanMode nanMode,
        IOutputStream* output);
}
