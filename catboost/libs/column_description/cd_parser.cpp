#include "cd_parser.h"

#include <catboost/libs/data_util/exists_checker.h>
#include <catboost/libs/data_util/line_data_reader.h>
#include <catboost/libs/helpers/exception.h>
#include <catboost/libs/logging/logging.h>

#include <util/generic/cast.h>
#include <util/generic/set.h>
#include <util/stream/file.h>
#include <util/stream/input.h>
#include <util/stream/labeled.h>
#include <util/string/split.h>
#include <util/system/fs.h>


using namespace NCB;

namespace {

    inline void CheckAllFeaturesPresent(const TVector<TColumn>& columns, const TSet<size_t>& parsedColumns) {
        for (size_t i = 0; i < columns.size(); ++i) {
            CB_ENSURE(parsedColumns.contains(i), "column not present in cd file: " << i);
        }
    }


    template <class TReadLineFunc>
    TVector<TColumn> ReadCDImpl(TReadLineFunc readLineFunc, const TCdParserDefaults& defaults) {
        size_t columnsCount = defaults.UseDefaultType ? SafeIntegerCast<size_t>(defaults.ColumnCount) : 0;

        TVector<TColumn> columns(columnsCount, TColumn{defaults.DefaultColumnType, TString()});
        TSet<size_t> parsedColumns;

        TString line;
        for (size_t lineNumber = 0; readLineFunc(&line); lineNumber++) {
            TVector<TString> tokens;
            try {
                try {
                    Split(line, "\t", tokens);
                } catch (const yexception& e) {
                    CATBOOST_DEBUG_LOG << "Got exception " << e.what() << " while parsing feature descriptions line " << line << Endl;
                    break;
                }
                if (tokens.empty()) {
                    continue;
                }
                CB_ENSURE(tokens.ysize() == 2 || tokens.ysize() == 3,
                    "Each line should have two or three columns. This line has " << tokens.size()
                );

                size_t index = 0;
                CB_ENSURE(
                    TryFromString(tokens[0], index),
                    "Invalid column index: \"" << tokens[0] << "\""
                );
                if (defaults.UseDefaultType) {
                    CB_ENSURE(
                        index < columnsCount,
                        "Invalid column index: " LabeledOutput(index, columnsCount));
                }
                CB_ENSURE(!parsedColumns.contains(index), "column specified twice in cd file: " << index);
                parsedColumns.insert(index);
                columns.resize(Max(columns.size(), index + 1));

                TStringBuf type = ToCanonicalColumnName(tokens[1]);

                CB_ENSURE(TryFromString<EColumn>(type, columns[index].Type), "unsupported column type " << type);
                if (tokens.ysize() == 3) {
                    columns[index].Id = tokens[2];
                }
            } catch (const TCatBoostException& e) {
                throw TCatBoostException() << "Incorrect CD file. Invalid line number #" << lineNumber
                    << ": " << e.what();
            }
        }
        if (!defaults.UseDefaultType) {
            CheckAllFeaturesPresent(columns, parsedColumns);
        }

        return columns;
    }

    class TCdFromFileProvider : public ICdProvider {
    public:
        TCdFromFileProvider(const NCB::TPathWithScheme& cdFilePath)
            : CdFilePath(cdFilePath) {}

        TVector<TColumn> GetColumnsDescription(ui32 columnsCount) const override;

        bool Inited() const override {
            return CdFilePath.Inited();
        }
    private:
        NCB::TPathWithScheme CdFilePath;
    };

    class TCdFromArrayProvider : public ICdProvider {
    public:
        TCdFromArrayProvider(const TVector<TColumn>& columnsDescription)
            : ColumnsDescription(columnsDescription) {}

        TVector<TColumn> GetColumnsDescription(ui32) const override {
            return ColumnsDescription;
        }

        bool Inited() const override {
            return ColumnsDescription.size() > 0;
        }
    private:
        TVector<TColumn> ColumnsDescription;
    };

}

THolder<ICdProvider> MakeCdProviderFromArray(const TVector<TColumn>& columnsDescription) {
    return MakeHolder<TCdFromArrayProvider>(columnsDescription);
}

THolder<ICdProvider> MakeCdProviderFromFile(const NCB::TPathWithScheme& path) {
    return MakeHolder<TCdFromFileProvider>(path);
}

TVector<TColumn> TCdFromFileProvider::GetColumnsDescription(ui32 columnsCount) const {
    TVector<TColumn> columnsDescription;
    if (CdFilePath.Inited()) {
        columnsDescription = ReadCD(CdFilePath, TCdParserDefaults(EColumn::Num, columnsCount));
    } else {
        columnsDescription.assign(columnsCount, TColumn{EColumn::Num, TString()});
        columnsDescription[0].Type = EColumn::Label;
    }
    return columnsDescription;
}

ICdProvider::~ICdProvider() = default;

TVector<TColumn> ReadCD(const TPathWithScheme& path, const TCdParserDefaults& defaults) {
    CB_ENSURE(CheckExists(path), "column description at [" << path << "] is not found");
    THolder<NCB::ILineDataReader> reader = NCB::GetLineDataReader(path);
    return ReadCDImpl([&reader](TString* l) -> bool { return reader->ReadLine(l); }, defaults);
}

TVector<TColumn> ReadCD(IInputStream* in, const TCdParserDefaults& defaults) {
    CB_ENSURE(in, "in pointer is `nullptr`");
    return ReadCDImpl([in](TString* l) -> bool { return in->ReadLine(*l); }, defaults);
}

TVector<TColumn> MakeDefaultColumnsDescription(size_t columnsCount) {
    TVector<TColumn> result(columnsCount, {EColumn::Num, TString()});
    result[0].Type = EColumn::Label;
    return result;
}

TVector<EColumn> GetColumnTypes(TConstArrayRef<TColumn> descriptions) {
    TVector<EColumn> types;
    types.reserve(descriptions.size());
    for (const auto& description : descriptions) {
        types.push_back(description.Type);
    }
    return types;
}
