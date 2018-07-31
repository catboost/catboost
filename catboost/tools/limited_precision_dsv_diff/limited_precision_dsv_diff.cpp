
#include <library/getopt/small/last_getopt.h>

#include <util/generic/hash.h>
#include <util/generic/ptr.h>
#include <util/generic/strbuf.h>
#include <util/generic/yexception.h>
#include <util/stream/file.h>
#include <util/stream/output.h>
#include <util/string/delim_string_iter.h>

#include <array>
#include <cmath>


int main(int argc, const char* argv[]) {
    TString delimiter = "\t";
    TString columnPrecisionSpec;

    auto opts = NLastGetopt::TOpts::Default();
    opts.AddHelpOption();

    opts.AddLongOption('d', "delimiter")
        .RequiredArgument("<string>")
        .StoreResult(&delimiter)
        .DefaultValue(delimiter)
        .Help("Use this delimiter");

    opts.AddLongOption("column-precision-spec")
        .RequiredArgument("<col_index:precision,col_index:precision...>")
        .StoreResult(&columnPrecisionSpec)
        .Help("set comparison precision for certain columns (col_index is zero-based)");

    opts.AddLongOption("have-header")
        .NoArgument()
        .Help("files have header (don't treat first line columns as numerical even with precision spec)");

    opts.SetFreeArgsNum(2);
    opts.SetFreeArgTitle(0, "<input-file1>", "Input dsv file");
    opts.SetFreeArgTitle(1, "<input-file2>", "Input dsv file");

    NLastGetopt::TOptsParseResult res(&opts, argc, argv);


    TVector<double> columnToPrecision;

    if (res.Has("column-precision-spec")) {
        for (TDelimStringIter it(columnPrecisionSpec, ","); it.Valid(); ++it) {
            TStringBuf columnString;
            TStringBuf precisionString;
            (*it).Split(':', columnString, precisionString);
            size_t columnNumber = FromString<size_t>(columnString);
            if (columnToPrecision.size() <= columnNumber) {
                columnToPrecision.resize(columnNumber + 1, 0.0);
            }
            columnToPrecision[columnNumber] = FromString<double>(precisionString);
        }
    }

    bool haveHeader = res.Has("have-header");

    auto inputFileNames = res.GetFreeArgs();

    THolder<TIFStream> input[2] = {
        MakeHolder<TIFStream>(inputFileNames[0]),
        MakeHolder<TIFStream>(inputFileNames[1])
    };

    bool isHeaderLine = haveHeader;

    TString line[2];

    for (size_t lineNumber = 0; ; ++lineNumber) {
        bool isEnd[2] = {false, false};
        for (auto i : {0,1}) {
            isEnd[i] = !input[i]->ReadLine(line[i]);
        }
        if (isEnd[0] != isEnd[1]) {
            if (isEnd[0]) {
                Cout << "file2 contains more lines than file1 (" << lineNumber << ")\n";
            } else {
                Cout << "file1 contains more lines than file2 (" << lineNumber << ")\n";
            }
            return 1;
        }
        if (isEnd[0]) {
            return 0;
        }

        TString diffMessagePrefix = isHeaderLine ? "Header. " : "";

        TDelimStringIter it0(line[0], delimiter);
        TDelimStringIter it1(line[1], delimiter);

        for (size_t columnNumber = 0; ; ++columnNumber, ++it0, ++it1) {
            if (it0.Valid() != it1.Valid()) {
                Cout << diffMessagePrefix << "Line " << lineNumber << ": ";
                if (it0.Valid()) {
                    Cout << "file1 contains more columns than file2 (" << columnNumber << ")\n";
                } else {
                    Cout << "file2 contains more columns than file1 (" << columnNumber << ")\n";
                }
                return 1;
            }
            if (!it0.Valid()) {
                break;
            }
            if (   !isHeaderLine
                && (columnNumber < columnToPrecision.size())
                && (columnToPrecision[columnNumber] != 0.0))
            {
                double precision = columnToPrecision[columnNumber];
                double diff = FromString<double>(*it0) - FromString<double>(*it1);
                if (std::abs(diff) > precision) {
                    Cout << diffMessagePrefix << "Line " << lineNumber
                         << ": file1 differs from file2 in column "
                         << columnNumber << " (precision=" << precision << ")\n";
                    return 1;
                }
            } else if (*it0 != *it1) {
                Cout << diffMessagePrefix << "Line " << lineNumber << ": file1 differs from file2 in column "
                     << columnNumber << Endl;
                return 1;
            }
        }
        if (isHeaderLine) {
            isHeaderLine = false;
        }
    }

    return 0;
}
