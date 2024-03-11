#pragma once

#include "baseline.h"
#include "loader.h"

#include <catboost/libs/column_description/column.h>
#include <catboost/private/libs/data_util/line_data_reader.h>
#include <catboost/libs/helpers/exception.h>

#include <util/generic/maybe.h>
#include <util/generic/ptr.h>
#include <util/generic/string.h>
#include <util/generic/vector.h>
#include <util/generic/ylimits.h>
#include <util/system/mutex.h>
#include <util/system/types.h>


namespace NCB {

    // expose the declaration to allow to derive from it in other modules
    class TCBDsvDataLoader : public IRawObjectsOrderDatasetLoader
                           , protected TAsyncProcDataLoaderBase<TString>
    {
    public:
        using TBase = TAsyncProcDataLoaderBase<TString>;

    protected:
        decltype(auto) GetReadFunc() {
            return [this](TString* line) -> bool {
                return LineDataReader->ReadLine(line);
            };
        }

        decltype(auto) GetReadBaselineFunc() {
            return [this](TObjectBaselineData *line) -> bool {
                ui64 objectIdx = 0;
                return BaselineReader->Read(line, &objectIdx);
            };
        }

    public:
        explicit TCBDsvDataLoader(TDatasetLoaderPullArgs&& args);

        explicit TCBDsvDataLoader(TLineDataLoaderPushArgs&& args);

        ~TCBDsvDataLoader() {
            AsyncRowProcessor.FinishAsyncProcessing();
        }

        void Do(IRawObjectsOrderDataVisitor* visitor) override {
            TBase::Do(GetReadFunc(), GetReadBaselineFunc(), visitor);
        }

        bool DoBlock(IRawObjectsOrderDataVisitor* visitor) override {
            return TBase::DoBlock(GetReadFunc(), GetReadBaselineFunc(), visitor);
        }

        TVector<TColumn> CreateColumnsDescription(ui32 columnsCount);

        ui32 GetObjectCountSynchronized() override;

        void StartBuilder(
            bool inBlock,
            ui32 objectCount,
            ui32 offset,
            IRawObjectsOrderDataVisitor* visitor
        ) override;

        void ProcessBlock(IRawObjectsOrderDataVisitor* visitor) override;

    protected:
        TVector<bool> FeatureIgnored; // init in process
        char FieldDelimiter;
        char NumVectorDelimiter;
        char CsvSplitterQuote;
        THolder<NCB::ILineDataReader> LineDataReader;
        THolder<NCB::IBaselineReader> BaselineReader;

        // cached
        TMutex ObjectCountMutex;
        TMaybe<ui32> ObjectCount;
    };

    size_t GetDsvColumnCount(
        const TPathWithScheme& pathWithScheme,
        const TDsvFormatOptions& format = TDsvFormatOptions()
    );

    TVector<TColumn> CreateDsvColumnsDescription(
        const TPathWithScheme& datasetPathWithScheme,
        const TPathWithScheme& cdPathWithScheme = TPathWithScheme(),
        const TPathWithScheme& featureNamesPathWithScheme = TPathWithScheme(),
        const TDsvFormatOptions& format = TDsvFormatOptions()
    );
}
