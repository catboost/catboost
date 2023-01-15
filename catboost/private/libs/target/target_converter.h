#pragma once

#include <catboost/libs/data/target.h>

#include <catboost/private/libs/options/enums.h>

#include <library/cpp/json/json_value.h>

#include <util/generic/fwd.h>
#include <util/generic/maybe.h>
#include <util/generic/ptr.h>
#include <util/generic/string.h>
#include <util/generic/vector.h>

#include <util/system/types.h>


namespace NPar {
    class TLocalExecutor;
}


namespace NCB {

    template <class T>
    class ITypedSequence;

    class ITargetConverter {
    public:
        virtual ~ITargetConverter() = default;

        virtual TVector<float> Process(ERawTargetType targetType,
                                       const TRawTarget& rawTarget,
                                       NPar::TLocalExecutor* localExecutor) = 0;

        // call after all processing
        virtual ui32 GetClassCount() const = 0;

        virtual TMaybe<TVector<NJson::TJsonValue>> GetClassLabels() {
            return Nothing();
        }
    };


    /*
     *  only one of targetBorder, classCount or inputClassNames should be specified
     */
    THolder<ITargetConverter> MakeTargetConverter(bool isRealTarget,
                                                  bool isClass,
                                                  bool isMultiClass,
                                                  TMaybe<float> targetBorder,
                                                  TMaybe<ui32> classCount,
                                                  const TVector<NJson::TJsonValue>& inputClassNames);

}
