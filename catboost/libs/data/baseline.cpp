#include "baseline.h"

#include <catboost/private/libs/data_util/exists_checker.h>

#include <util/generic/algorithm.h>
#include <util/generic/is_in.h>


namespace {
    const TString CLASS_NAME_DELIMITER = ":Class=";
}

static void GetClassNamesFromBaselineFile(const NCB::TPathWithScheme& baselineFilePath, TVector<TString>* classNames) {
    THolder<NCB::ILineDataReader> reader = GetLineDataReader(baselineFilePath);
    TString header;
    reader->ReadLine(&header);
    CB_ENSURE(!header.empty(), "Empty baseline file header");
    TVector<TString> tokens = StringSplitter(header).Split('\t');
    bool noClassName = false;
    for (const auto& token: tokens) {
        auto splitIdx = token.find(":");
        if (token.substr(0, splitIdx) == "RawFormulaVal") {
            if (splitIdx < token.size()) {
                CB_ENSURE(token.substr(splitIdx, CLASS_NAME_DELIMITER.size()) == CLASS_NAME_DELIMITER,
                          "Wrong baseline column name " <<  token <<  ", should be 'RawFormulaVal[:Class=<class_name>]'");
                CB_ENSURE(!noClassName, "Inconsistent RawFormulaVal header specification - some columns contain class name and some do not");

                auto className = token.substr(splitIdx + CLASS_NAME_DELIMITER.size());
                CB_ENSURE(!IsIn(*classNames, className), "Class name " << className << " is not unique in baseline header");
                classNames->push_back(className);
            } else {
                CB_ENSURE(classNames->empty(), "Inconsistent RawFormulaVal header specification - some columns contain class name and some do not");
                noClassName = true;
            }
        }
    }
}

namespace NCB {
    TBaselineReader::TBaselineReader(const TPathWithScheme& baselineFilePath, const TVector<TString>& classNames) {
        if (baselineFilePath.Inited()) {
            Reader_ = GetProcessor<ILineDataReader, TLineDataReaderArgs>(
                baselineFilePath, TLineDataReaderArgs{baselineFilePath, TDsvFormatOptions{true, DELIMITER_}});
            auto header = Reader_->GetHeader();
            CB_ENSURE(!header->empty(), "Empty baseline file header");
            ui32 columnIdx = 0;
            TVector<TString> tokens = StringSplitter(*header).Split('\t');
            BaselineSize_ = tokens.ysize();

            for (const auto& token: tokens) {
                auto splitIdx = token.find(CLASS_NAME_DELIMITER);
                if (token.substr(0, splitIdx) == "RawFormulaVal") {
                    if ((BaselineSize_ != 1) && !classNames.empty() && splitIdx < token.size()) {
                        TString className = token.substr(splitIdx + CLASS_NAME_DELIMITER.size());
                        CB_ENSURE(classNames[BaselineIndexes_.size()] == className, "Unknown class name " << className);
                    }
                    BaselineIndexes_.push_back(columnIdx);
                }
                ++columnIdx;
            }
            CB_ENSURE(!BaselineIndexes_.empty(), "Baseline file header should contain at least one 'RawFormulaVal[:Class=class_name]' value");
            CB_ENSURE((BaselineIndexes_.size() == 1 && (classNames.empty() || (classNames.size() == 2))) ||
                      (!classNames.empty() &&  BaselineIndexes_.size() == classNames.size()),
                      "Baseline file header for multiclass should contain one 'RawFormulaVal' value or several 'RawFormulaVal:Class=class_name' values");
            Inited_ = true;
        }
    }

    void UpdateClassNamesFromBaselineFile(
        const TPathWithScheme& baselineFilePath,
        TVector<TString>* classNames
    ) {
        if (baselineFilePath.Inited()) {
            CB_ENSURE_INTERNAL(classNames != nullptr, "ClassNames has not been specified");
            TVector<TString> classNamesFromBaselineFile;
            GetClassNamesFromBaselineFile(baselineFilePath, &classNamesFromBaselineFile);
            if (classNames->empty()) {
                (*classNames) = std::move(classNamesFromBaselineFile);
            } else {
                CB_ENSURE(*classNames == classNamesFromBaselineFile, "Inconsistent class names in baseline file");
            }
        }
    }
}
