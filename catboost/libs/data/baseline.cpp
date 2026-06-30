#include "baseline.h"

#include <catboost/private/libs/data_util/exists_checker.h>
#include <catboost/private/libs/labels/helpers.h>

#include <library/cpp/json/json_value.h>

#include <util/generic/algorithm.h>
#include <util/generic/cast.h>
#include <util/generic/is_in.h>
#include <util/generic/scope.h>
#include <util/string/split.h>


namespace {
    const TString CLASS_NAME_DELIMITER = ":Class=";
}

namespace NCB {
    TBaselineReaderFactory::TRegistrator<TDsvBaselineReader> DefBaselineReaderReg("");
    TBaselineReaderFactory::TRegistrator<TDsvBaselineReader> FileBaselineReaderReg("file");
    TBaselineReaderFactory::TRegistrator<TDsvBaselineReader> DsvBaselineReaderReg("dsv");


    TDsvBaselineReader::TDsvBaselineReader(TBaselineReaderArgs&& args)
        : Range_(std::move(args.Range))
    {
        CB_ENSURE(args.PathWithScheme.Inited(), "Path to baseline data is not inited");

        if (!args.ClassNames.empty()) {
            ClassNames_ = args.ClassNames;
        }

        Reader_ = GetProcessor<ILineDataReader, TLineDataReaderArgs>(
            args.PathWithScheme,
            TLineDataReaderArgs{args.PathWithScheme, TDsvFormatOptions{true, DELIMITER_}}
        );
        auto header = Reader_->GetHeader();
        CB_ENSURE(!header->empty(), "Empty baseline file header");
        ui32 columnIdx = 0;
        TVector<TString> tokens = StringSplitter(*header).Split('\t');
        BaselineSize_ = tokens.ysize();

        bool noClassName = false;

        for (const auto& token: tokens) {
            constexpr auto prefix = "RawFormulaVal"sv;
            if (token.StartsWith(prefix)) {
                if (token.size() > prefix.size()) {
                    CB_ENSURE(
                        token.substr(prefix.size(), CLASS_NAME_DELIMITER.size()) == CLASS_NAME_DELIMITER,
                        "Wrong baseline column name " <<  token <<  ", should be 'RawFormulaVal[:Class=<class_name>]'"
                    );
                    CB_ENSURE(
                        !noClassName,
                        "Inconsistent RawFormulaVal header specification - some columns contain class name and some "
                        "do not"
                    );

                    auto className = token.substr(prefix.size() + CLASS_NAME_DELIMITER.size());
                    if (!args.ClassNames.empty()) {
                        CB_ENSURE(
                            args.ClassNames[BaselineIndexes_.size()] == className,
                            "Unknown class name " << className
                        );
                    } else {
                        CB_ENSURE(
                            !IsIn(ClassNames_, className),
                            "Class name " << className << " is not unique in baseline header"
                        );
                        ClassNames_.push_back(className);
                    }
                } else {
                    CB_ENSURE_INTERNAL(
                        args.ClassNames.empty(),
                        "Dataset contain classes but baseline file header does not specify them"
                    );
                    CB_ENSURE(
                        ClassNames_.empty(),
                        "Inconsistent RawFormulaVal header specification - some columns contain class name and some "
                        "do not"
                    );
                    noClassName = true;
                }
                BaselineIndexes_.push_back(columnIdx);
            }
            ++columnIdx;
        }
        CB_ENSURE(
            !BaselineIndexes_.empty(),
            "Baseline file header should contain at least one 'RawFormulaVal[:Class=class_name]' value"
        );

        if (!args.ClassNames.empty()) {
            CB_ENSURE(
                (BaselineIndexes_.size() == 1 && (args.ClassNames.size() == 2)) ||
                (BaselineIndexes_.size() == args.ClassNames.size()),
                "Baseline file header for multiclass should contain one 'RawFormulaVal' value or several "
                "'RawFormulaVal:Class=class_name' values"
            );
        }
    }

    ui32 TDsvBaselineReader::GetBaselineCount() const {
        return SafeIntegerCast<ui32>(BaselineIndexes_.size());
    }

    TVector<TString> TDsvBaselineReader::GetClassNames() {
        return ClassNames_;
    }

    bool TDsvBaselineReader::Read(TObjectBaselineData* data, ui64* objectIdx) {
        TString line;

        while (true) {
            if (!Reader_->ReadLine(&line, objectIdx)) {
                return false;
            }
            if (Range_.Contains(*objectIdx)) {
                break;
            }
        }
        data->Baseline.yresize(BaselineIndexes_.size());

        ui32 baselineIdx = 0;
        ui32 columnIdx = 0;
        for (const TStringBuf token : StringSplitter(line).Split(DELIMITER_)) {
            Y_DEFER { ++columnIdx; };

            CB_ENSURE(
                columnIdx < BaselineSize_,
                "Too many columns in baseline file line " << LabeledOutput(*objectIdx)
            );

            if (baselineIdx >= BaselineIndexes_.size() || columnIdx != BaselineIndexes_[baselineIdx]) {
                continue;
            }

            CB_ENSURE(!token.empty(), "Empty token in baseline file line " << LabeledOutput(*objectIdx));

            float baseline;
            CB_ENSURE(
                TryFromString(token, baseline),
                "Failed to parse float " << LabeledOutput(token) << " in baseline file line "
                << LabeledOutput(*objectIdx)
            );
            data->Baseline[baselineIdx] = baseline;

            ++baselineIdx;
        }

        CB_ENSURE(columnIdx == BaselineSize_, "Not enough columns in baseline file line " << LabeledOutput(*objectIdx));
        *objectIdx -= Range_.Begin;

        return true;
    }

    void UpdateClassLabelsFromBaselineFile(
        const TPathWithScheme& baselineFilePath,
        TVector<NJson::TJsonValue>* classLabels
    ) {
        if (baselineFilePath.Inited()) {
            CB_ENSURE_INTERNAL(classLabels != nullptr, "ClassLabels has not been specified");
            THolder<IBaselineReader> reader = GetProcessor<IBaselineReader, TBaselineReaderArgs>(
                baselineFilePath, TBaselineReaderArgs{baselineFilePath, {}}
            );
            TVector<TString> classNamesFromBaselineFile = reader->GetClassNames();
            if (classLabels->empty()) {
                classLabels->assign(classNamesFromBaselineFile.begin(), classNamesFromBaselineFile.end());
            } else {
                CB_ENSURE(
                    NCB::ClassLabelsToStrings(*classLabels) == classNamesFromBaselineFile,
                    "Inconsistent class names in baseline file"
                );
            }
        }
    }
}
