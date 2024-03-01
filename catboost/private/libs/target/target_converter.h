#pragma once

#include <catboost/libs/data/target.h>
#include <catboost/libs/helpers/exception.h>

#include <catboost/private/libs/options/enums.h>

#include <library/cpp/json/json_value.h>

#include <util/generic/fwd.h>
#include <util/generic/maybe.h>
#include <util/generic/ptr.h>
#include <util/generic/strbuf.h>
#include <util/generic/string.h>
#include <util/generic/vector.h>

#include <util/system/types.h>


namespace NPar {
    class ILocalExecutor;
}


namespace NCB {

    class TUnknownClassLabelException : public TCatBoostException {
    public:
        TUnknownClassLabelException(const TString& classLabelAsString)
            : ClassLabelAsString(classLabelAsString)
            , ErrorMessage("Unknown class label: \"" + classLabelAsString + "\"")
        {}

        TStringBuf GetUnknownClassLabel() const {
            return ClassLabelAsString;
        }

        const char* what() const noexcept override {
            return ErrorMessage.c_str();
        }

    private:
        TString ClassLabelAsString;
        TString ErrorMessage;
    };


    template <class T>
    class ITypedSequence;

    class ITargetConverter {
    public:
        virtual ~ITargetConverter() = default;

        virtual TVector<float> Process(ERawTargetType targetType,
                                       const TRawTarget& rawTarget,
                                       NPar::ILocalExecutor* localExecutor) = 0;

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
                                                  bool isMultiLabel,
                                                  TMaybe<float> targetBorder,
                                                  size_t targetDim,
                                                  TMaybe<ui32> classCount,
                                                  const TVector<NJson::TJsonValue>& inputClassNames,
                                                  bool allowConstLabel);

}
