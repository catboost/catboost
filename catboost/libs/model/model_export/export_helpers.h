#pragma once

#include <catboost/libs/model/enums.h>
#include <catboost/libs/model/model.h>

#include <util/string/builder.h>
#include <util/generic/string.h>

namespace NCatboostModelExportHelpers {
    class TIndent {
        size_t IndentCount;
        const ui32 IndentSize = 4;
        TStringBuilder Spaces;
        public:
            TIndent() : IndentCount(0) {}

            explicit TIndent(size_t indentCount) : IndentCount(indentCount) {
                UpdateSize();
            }

            TIndent(const TIndent &other) : IndentCount(other.IndentCount) {
                UpdateSize();
            }

            TIndent(TIndent &&other) : IndentCount(other.IndentCount) {
                UpdateSize();
            };

            const TStringBuilder& GetString() const {
                return Spaces;
            }

            TIndent& operator++() {
                ++IndentCount;
                UpdateSize();
                return *this;
            }

            TIndent& operator--() {
                CB_ENSURE(IndentCount > 0, "Cannot unindent because indent count == 0");
                --IndentCount;
                UpdateSize();
                return *this;
            }

            TIndent operator++(int) {
                TIndent old(*this);
                ++(*this);
                return old;
            }
            TIndent operator--(int) {
                TIndent old(*this);
                --(*this);
                return old;
            }
        private:
            void UpdateSize() {
                if (IndentCount * IndentSize != Spaces.size()) {
                    Spaces.resize(IndentCount * IndentSize, ' ');
                }
            }
    };

    inline IOutputStream& operator<<(IOutputStream& stream, const TIndent& indent) noexcept {
                return stream << indent.GetString();
    }

    enum ESpaceAfterComma {
        AddSpaceAfterComma,
        NoSpaceAfterComma
    };

    class TSequenceCommaSeparator {
        /* Returns comma (Count - 1) times */
        public:
            explicit TSequenceCommaSeparator(size_t sequenceLength = 0, ESpaceAfterComma spaceAfterComma = NoSpaceAfterComma) : SequenceLength(sequenceLength), SpaceAfterComma(spaceAfterComma) {}
            explicit TSequenceCommaSeparator(ESpaceAfterComma spaceAfterComma) : SequenceLength(0), SpaceAfterComma(spaceAfterComma) {}
            void ResetCount(size_t sequenceLength) {
                SequenceLength = sequenceLength;
            }
            TString GetSeparator() {
                if (SequenceLength > 0) {
                    --SequenceLength;
                }
                if (SequenceLength) {
                    if (SpaceAfterComma == AddSpaceAfterComma) {
                        return ", ";
                    }
                    return ",";
                }
                return "";
            }
       private:
            size_t SequenceLength;
            ESpaceAfterComma SpaceAfterComma;

    };

    inline IOutputStream& operator<<(IOutputStream& stream, TSequenceCommaSeparator& commas) {
        return stream << commas.GetSeparator();
    }

    inline TStringBuilder& operator<<(TStringBuilder& stream, TSequenceCommaSeparator& commas) {
        return stream << commas.GetSeparator();
    };

    template <typename TElementAccessor>
    TString OutputArrayInitializer(TElementAccessor elementAccessor, size_t size) {
        TStringBuilder str;
        TSequenceCommaSeparator comma(size, AddSpaceAfterComma);
        for (size_t i = 0; i < size; ++i) {
            str << elementAccessor(i) << comma;
        }
        return str;
    }

    template <class T>
    TString OutputArrayInitializer(const T& values) {
        return OutputArrayInitializer([&values] (size_t i) { return values[i]; }, values.size());
    }

    int GetBinaryFeatureCount(const TFullModel& model);

    TString OutputBorderCounts(const TFullModel& model);

    TString OutputBorders(const TFullModel& model, bool addFloatingSuffix = false);

    TString OutputLeafValues(const TFullModel& model, TIndent indent, EModelType modelType);
}
