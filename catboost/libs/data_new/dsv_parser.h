#pragma once

#include <util/generic/array_ref.h>
#include <util/generic/maybe.h>
#include <util/generic/string.h>
#include <util/system/types.h>

class TCatBoostException;
enum class EColumn;
struct TColumn;

namespace NCB {
    class IRawObjectsOrderDataVisitor;
    class TFeaturesLayout;
}

namespace NCB {
    class TDsvLineParser {
        public:
            TDsvLineParser(
                char delimiter,
                TConstArrayRef<TColumn> columnsDescription,
                TConstArrayRef<bool> featureIgnored,
                const TFeaturesLayout* featuresLayout,
                TArrayRef<float> numericFeaturesBuffer,
                TArrayRef<ui32> categoricalFeaturesBuffer,
                IRawObjectsOrderDataVisitor* visitor);

            enum class EErrorType : ui8 {
                Unknown                     = 0,
                EmptyToken                  = 1,
                TooManyColumns              = 2,
                NotEnoughColumns            = 3,
                FailedToParseNumericFeature = 4,
                FailedToParseFloat          = 5,
                FailedToParseUi64           = 6,
                ColumnTypeIsNotSupported    = 7
            };

            struct TErrorContext {
                EErrorType Type = EErrorType::Unknown;
                TString Token;
                TMaybe<ui32> ColumnIdx;
                TMaybe<ui32> FlatFeatureIdx;
                TMaybe<EColumn> ColumnType;  // Doesn't have proper default
            };

            TMaybe<TErrorContext> Parse(TStringBuf line, ui32 inBlockIdx);

            static TCatBoostException MakeException(const TErrorContext& context);

        private:
            TMaybe<TErrorContext> HandleToken(
                TStringBuf token,
                ui32 inBlockIdx,
                ui32 columnIdx,
                ui32* flatFeatureIdx,
                ui32* baselineIdx);

        private:
            char Delimiter_ = '\0';
            TConstArrayRef<TColumn> ColumnDescriptions_;
            TConstArrayRef<bool> FeatureIgnored_;
            const TFeaturesLayout* FeaturesLayout_ = nullptr;

            TArrayRef<float> NumericFeaturesBuffer_;
            TArrayRef<ui32> CategoricalFeaturesBuffer_;
            IRawObjectsOrderDataVisitor* Visitor_ = nullptr;
    };
}
