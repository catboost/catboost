#pragma once

#include <catboost/private/libs/options/data_processing_options.h>
#include <catboost/private/libs/options/enums.h>
#include <catboost/private/libs/options/loss_description.h>

#include <catboost/libs/data/target.h>

#include <util/generic/array_ref.h>
#include <util/generic/hash.h>
#include <util/generic/hash_set.h>
#include <util/generic/maybe.h>
#include <util/generic/strbuf.h>
#include <util/generic/string.h>
#include <util/generic/vector.h>


namespace NPar {
    class TLocalExecutor;
}


namespace NCB {

    template <class T>
    class ITypedSequence;


    class TTargetConverter {
    public:
        TTargetConverter(const bool isClassTarget,
                         const bool isMultiClassTarget,
                         const EConvertTargetPolicy readingPoolTargetPolicy,
                         const TVector<TString>& inputClassNames,
                         TVector<TString>* const outputClassNames);

        TVector<float> Process(const TRawTarget& rawTarget, NPar::TLocalExecutor* localExecutor);

        // call after all processing
        ui32 GetClassCount() const;

    private:
        float CastFloatLabel(float label);
        float CastFloatLabel(TStringBuf label);

        TVector<float> ProcessCastFloat(const TRawTarget& rawTarget, NPar::TLocalExecutor* localExecutor);
        TVector<float> ProcessUseClassNames(const TRawTarget& rawTarget, NPar::TLocalExecutor* localExecutor);

        TVector<float> ProcessMakeClassNames(const TRawTarget& rawTarget,
                                             NPar::TLocalExecutor* localExecutor);
        TVector<float> ProcessMakeClassNamesImpl(const ITypedSequencePtr<float>& labels,
                                                 NPar::TLocalExecutor* localExecutor);
        TVector<float> ProcessMakeClassNamesImpl(TConstArrayRef<TString> labels,
                                                 NPar::TLocalExecutor* localExecutor);

        void UpdateStringLabelToClass();
        void UpdateFloatLabelToClass();

        void SetOutputClassNames();

    private:
        const bool IsClassTarget;
        const bool IsMultiClassTarget;
        const EConvertTargetPolicy TargetPolicy;
        const TVector<TString>& InputClassNames;
        TVector<TString>* const OutputClassNames;

        /* used with targetPolicy = MakeClassNames/UseClassNames
         * which map is used depends on source target data type
         * UpdateStringLabelToClass initializes StringLabelToClass from FloatLabelToClass if necessary
         */
        THashMap<TString, int> StringLabelToClass;
        THashMap<float, int> FloatLabelToClass;

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
