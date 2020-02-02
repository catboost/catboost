#pragma once

#include <catboost/private/libs/data_util/path_with_scheme.h>
#include <catboost/private/libs/data_util/line_data_reader.h>
#include <catboost/libs/helpers/exception.h>

#include <util/generic/cast.h>
#include <util/generic/maybe.h>
#include <util/generic/ptr.h>
#include <util/generic/scope.h>
#include <util/generic/strbuf.h>
#include <util/generic/string.h>
#include <util/generic/vector.h>
#include <util/stream/labeled.h>
#include <util/string/cast.h>
#include <util/string/escape.h>
#include <util/string/split.h>
#include <util/system/types.h>


namespace NJson {
    class TJsonValue;
}


namespace NCB {
    class TBaselineReader {
    public:
        TBaselineReader() {}

        TBaselineReader(const TPathWithScheme& baselineFilePath, const TVector<TString>& classNames);

        bool Inited() const {
            return Inited_;
        }

        TMaybe<ui32> GetBaselineCount() const {
            if (Inited_) {
                return SafeIntegerCast<ui32>(BaselineIndexes_.size());
            } else {
                return Nothing();
            }
        }

        const TVector<ui32>& GetBaselineIndexes() const {
            return BaselineIndexes_;
        }

        bool ReadLine(TString* line) {
            return Reader_->ReadLine(line);
        }

        template <class TFunc>
        void Parse(TFunc addBaselineFunc, TStringBuf line, ui32 lineIdx) {
            ui32 baselineIdx = 0;
            ui32 columnIdx = 0;
            for (const TStringBuf token : StringSplitter(line).Split(DELIMITER_)) {
                Y_DEFER { ++columnIdx; };

                CB_ENSURE(columnIdx < BaselineSize_, "Too many columns in baseline file line " << LabeledOutput(lineIdx));

                if (baselineIdx >= BaselineIndexes_.size() || columnIdx != BaselineIndexes_[baselineIdx]) {
                    continue;
                }

                CB_ENSURE(!token.empty(), "Empty token in baseline file line " << LabeledOutput(lineIdx));

                float baseline;
                CB_ENSURE(TryFromString(token, baseline), "Failed to parse float " << LabeledOutput(token) << " in baseline file line " << LabeledOutput(lineIdx));
                addBaselineFunc(baselineIdx, baseline);

                ++baselineIdx;
            }

            CB_ENSURE(columnIdx == BaselineSize_, "Not enough columns in baseline file line " << LabeledOutput(lineIdx));
        }

    private:
        THolder<ILineDataReader> Reader_;
        TVector<ui32> BaselineIndexes_;
        ui32 BaselineSize_ = 0;
        bool Inited_ = false;
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
