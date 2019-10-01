#pragma once

#include "quantized_features_info.h"

#include <util/generic/string.h>


namespace NCB {
    void LoadBordersAndNanModesFromFromFileInMatrixnetFormat(
        const TString& path,
        TQuantizedFeaturesInfo* quantizedFeaturesInfo);

    void SaveBordersAndNanModesToFileInMatrixnetFormat(
        const TString& file,
        const TQuantizedFeaturesInfo& quantizedFeaturesInfo);
}
