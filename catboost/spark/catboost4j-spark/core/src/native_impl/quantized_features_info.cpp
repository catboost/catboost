#include "quantized_features_info.h"

#include <library/cpp/dbg_output/dump.h>

#include <util/generic/ptr.h>
#include <util/stream/file.h>


using namespace NCB;


TQuantizedFeaturesInfoPtr MakeQuantizedFeaturesInfo(
    const TFeaturesLayout& featuresLayout
) throw(yexception) {
    return MakeIntrusive<TQuantizedFeaturesInfo>(
        featuresLayout,
        /*ignoredFeatures*/ TConstArrayRef<ui32>(),
        NCatboostOptions::TBinarizationOptions()
    );
}


void DbgDump(const NCB::TQuantizedFeaturesInfo& quantizedFeaturesInfo, const TString& fileName) {
    TFileOutput out(fileName);
    out << DbgDump(quantizedFeaturesInfo);
}


