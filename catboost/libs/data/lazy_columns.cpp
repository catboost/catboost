#include "lazy_columns.h"

template <typename IQuantizedValuesHolder>
const NCB::TLazyQuantizedFloatValuesHolder* NCB::CastToLazyQuantizedFloatValuesHolder(
    const IQuantizedValuesHolder* quantizedFeatureColumn
) {
    return dynamic_cast<const NCB::TLazyQuantizedFloatValuesHolder*>(quantizedFeatureColumn);
}

template
const NCB::TLazyQuantizedFloatValuesHolder* NCB::CastToLazyQuantizedFloatValuesHolder(
    const NCB::IQuantizedFloatValuesHolder* quantizedFeatureColumn);

template
const NCB::TLazyQuantizedFloatValuesHolder* NCB::CastToLazyQuantizedFloatValuesHolder(
    const NCB::IQuantizedCatValuesHolder* quantizedFeatureColumn);

template
const NCB::TLazyQuantizedFloatValuesHolder* NCB::CastToLazyQuantizedFloatValuesHolder(
    const NCB::IExclusiveFeatureBundleArray* quantizedFeatureColumn);
