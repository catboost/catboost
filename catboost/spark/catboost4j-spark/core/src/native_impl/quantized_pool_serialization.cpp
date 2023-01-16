#include "quantized_pool_serialization.h"

#include <catboost/private/libs/quantized_pool/serialization.h>


void SaveQuantizedPoolWrapper(const NCB::TDataProviderPtr& dataProvider, TString fileName) {
    NCB::SaveQuantizedPool(dataProvider, std::move(fileName));
}
