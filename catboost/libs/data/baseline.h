#pragma once

#include <catboost/private/libs/data_util/line_data_reader.h>
#include <catboost/private/libs/data_util/path_with_scheme.h>
#include <catboost/private/libs/index_range/index_range.h>

#include <library/cpp/object_factory/object_factory.h>

#include <util/generic/string.h>
#include <util/generic/vector.h>
#include <util/generic/ylimits.h>
#include <util/system/types.h>


namespace NJson {
    class TJsonValue;
}


namespace NCB {
    struct TObjectBaselineData {
        TVector<float> Baseline;    // single-element for regression or binclass, multiple elements for multiclass
    };


    struct IBaselineReader {
        virtual ~IBaselineReader() = default;

        virtual ui32 GetBaselineCount() const = 0;

        // will return empty vector if no class names are present in the baseline data (i.e. it's for regresssion)
        virtual TVector<TString> GetClassNames() = 0;

        // read next object baseline data
        // returns false if end of data is reached, true otherwise
        virtual bool Read(TObjectBaselineData* data, ui64* objectIdx) = 0;
    };


    struct TBaselineReaderArgs {
        TPathWithScheme PathWithScheme;
        TVector<TString> ClassNames;
        TIndexRange<ui64> Range = {0, Max<ui64>()};
    };

    using TBaselineReaderFactory =
        NObjectFactory::TParametrizedObjectFactory<IBaselineReader, TString, TBaselineReaderArgs>;

    class TDsvBaselineReader : public IBaselineReader {
    public:
        TDsvBaselineReader(TBaselineReaderArgs&& args);

        ui32 GetBaselineCount() const override;
        TVector<TString> GetClassNames() override;

        bool Read(TObjectBaselineData* data, ui64* objectIdx) override;

    private:
        TIndexRange<ui64> Range_;

        THolder<ILineDataReader> Reader_;
        TVector<ui32> BaselineIndexes_;
        TVector<TString> ClassNames_;   // will be empty for regression

        ui32 BaselineSize_ = 0;
        constexpr static char DELIMITER_ = '\t';
    };

    /**
     * If classLabels are empty init them from baseline header,
     * check classLabels and baseline file header consistency otherwise
     */
    void UpdateClassLabelsFromBaselineFile(
        const TPathWithScheme& baselineFilePath,
        TVector<NJson::TJsonValue>* classLabels
    );
}
