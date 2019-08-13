#pragma once

#include <catboost/libs/options/data_processing_options.h>
#include <catboost/libs/options/enums.h>
#include <catboost/libs/options/loss_description.h>

#include <util/generic/array_ref.h>
#include <util/generic/hash.h>
#include <util/generic/hash_set.h>
#include <util/generic/maybe.h>
#include <util/generic/strbuf.h>
#include <util/generic/string.h>
#include <util/generic/vector.h>


namespace NCB {

    class TTargetConverter {
    public:
        TTargetConverter(const bool isClassTarget,
                         const bool isMultiClassTarget,
                         const EConvertTargetPolicy readingPoolTargetPolicy,
                         const TVector<TString>& inputClassNames,
                         TVector<TString>* const outputClassNames);

        float ConvertLabel(const TStringBuf& label);
        float ProcessLabel(const TString& label);

        TVector<float> PostprocessLabels(TConstArrayRef<TString> labels);
        void SetOutputClassNames() const;

        EConvertTargetPolicy GetTargetPolicy() const;

        const TVector<TString>& GetInputClassNames() const;

        // call after all processing
        ui32 GetClassCount() const;
    private:
        const bool IsClassTarget;
        const bool IsMultiClassTarget;
        const EConvertTargetPolicy TargetPolicy;
        const TVector<TString>& InputClassNames;
        TVector<TString>* const OutputClassNames;
        THashMap<TString, int> LabelToClass; // used with targetPolicy = MakeClassNames/UseClassNames
        THashSet<float> UniqueLabels; // used only if IsClassTarget
    };

    /*
     *  if inputClassNames is nonempty, isMultiClass and classCountUnknown
     *  are not used
     */
    TTargetConverter MakeTargetConverter(bool isClass,
                                         bool isMultiClass,
                                         bool classCountUnknown,
                                         const TVector<TString>& inputClassNames,
                                         TVector<TString>* outputClassNames);

}
