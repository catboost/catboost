#pragma once

#include <catboost/libs/data_util/path_with_scheme.h>
#include <catboost/libs/options/load_options.h>

#include <library/object_factory/object_factory.h>

#include <util/generic/vector.h>

namespace NCB {
    class IModelEvaluator {
    public:
        virtual void Apply(
            int argc,
            const char** argv,
            const NCB::TPathWithScheme& inputPath,
            const NCatboostOptions::TDsvPoolFormatParams& dsvPoolFormatParams,
            const TString& modelPath,
            EModelType modelFormat,
            TConstArrayRef<TString> outputColumns,
            ui32 iterationsLimit,
            const NCB::TPathWithScheme& outputPath) const = 0;

        virtual bool IsAcceptable(const NCB::TPathWithScheme& inputPath) const = 0;
        virtual bool IsReasonable(const NCB::TPathWithScheme& inputPath) const = 0;

        virtual ~IModelEvaluator() = default;
    };

    using TModelEvaluatorFactory = NObjectFactory::TParametrizedObjectFactory<IModelEvaluator, TString>;
}

