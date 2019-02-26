#include "data_provider.h"


namespace NCB {

    TQuantizedBuilderData CastToBase(TQuantizedForCPUBuilderData&& builderData) {
        TQuantizedBuilderData baseData;
        baseData.MetaInfo = std::move(builderData.MetaInfo);
        baseData.TargetData = std::move(builderData.TargetData);
        baseData.CommonObjectsData = std::move(builderData.CommonObjectsData);
        baseData.ObjectsData = std::move(builderData.ObjectsData.Data);
        return baseData;
    }

}

