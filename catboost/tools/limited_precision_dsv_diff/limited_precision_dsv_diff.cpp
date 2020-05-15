#include <library/cpp/getopt/small/last_getopt.h>

#include <util/stream/file.h>
#include <util/stream/output.h>
#include <util/string/builder.h>
#include <library/cpp/deprecated/split/delim_string_iter.h>

#include <cmath>

static double CalcDiff(double number0, double number1) {
    const double delta = number0 - number1;
    const double maxAbsoluteValue = std::max(std::abs(number0), std::abs(number1));
    const double relativeDiff = maxAbsoluteValue == 0.0 ? 0.0 : std::abs(delta) / maxAbsoluteValue;
    const double minDiff = std::min(std::abs(delta), relativeDiff);
    return minDiff;
}

enum class EDiffType {
    LineCountDifferent,
    TokenCountDifferent,
    TokensLexicallyDifferent,
    NumericThresholdExceeded,
    NumericDiffGotNaN,
};

static void ReportDiff(
    EDiffType diffType,
    size_t line,
    size_t column,
    const TDelimStringIter& it0 = {},
    const TDelimStringIter& it1 = {},
    const TString& details = {}) {

    TStringBuilder out;
    out << "Line " << line;
    if (it0.Valid() || it1.Valid()) {
        out << ", column " << column;
    }
    out << ": ";
    switch (diffType) {
        case EDiffType::NumericDiffGotNaN:
            out << "numeric diff got not-a-number: "
                << *it0 << " vs " << *it1;
            break;
        case EDiffType::NumericThresholdExceeded:
            out << "numeric threshold exceeded: "
                << *it0 << " vs " << *it1;
            break;
        case EDiffType::TokensLexicallyDifferent:
            out << "tokens lexically different: "
                << *it0 << " vs " << *it1;
            break;
        case EDiffType::TokenCountDifferent:
            out << "token count different: "
                << (*it0 ? *it0 : "<not-a-token>") << " vs "
                << (*it1 ? *it1 : "<not-a-token>");
            break;
        case EDiffType::LineCountDifferent:
            out << "line count different";
            break;
    }
    if (details) {
        out << ", " << details;
    }
    out << Endl;
    Cout << out;
    return;
}

static bool AreFilesDifferent(
    TIFStream& input0,
    TIFStream& input1,
    bool haveHeader,
    const TString& delimiter,
    double diffLimit,
    bool stopEarly) {

    double maxDiffSoFar = 0;
    bool isDiffDetected = false;

    TString line0;
    TString line1;
    bool eof0 = !input0.ReadLine(line0);
    bool eof1 = !input1.ReadLine(line1);
    size_t lineNumber = 0;

    for (; !eof0 && !eof1; ++lineNumber, eof0 = !input0.ReadLine(line0), eof1 = !input1.ReadLine(line1)) {

        const bool isHeaderLine = haveHeader && lineNumber == 0;

        TDelimStringIter it0(line0, delimiter);
        TDelimStringIter it1(line1, delimiter);
        size_t columnNumber = 0;

        for (; it0.Valid() && it1.Valid(); ++columnNumber, ++it0, ++it1) {
            if (*it0 == *it1) {
                continue;
            }
            if (isHeaderLine) {
                ReportDiff(EDiffType::TokensLexicallyDifferent, lineNumber, columnNumber, it0, it1, "header line");
                return true;
            }

            double number0, number1;
            try {
                number0 = FromString<double>(*it0);
                number1 = FromString<double>(*it1);
            } catch (...) {
                ReportDiff(EDiffType::TokensLexicallyDifferent, lineNumber, columnNumber, it0, it1);
                return true;
            }

            const double diff = CalcDiff(number0, number1);

            if (isnan(number0) || isnan(number1) || isnan(diff)) {
                ReportDiff(EDiffType::NumericDiffGotNaN, lineNumber, columnNumber, it0, it1);
                return true;
            }

            if (diff >= maxDiffSoFar) {
                maxDiffSoFar = diff;
                if (diff >= diffLimit) {
                    ReportDiff(EDiffType::NumericThresholdExceeded, lineNumber, columnNumber, it0, it1,
                        TStringBuilder() << "err=" << diff << ", threshold=" << diffLimit);
                    isDiffDetected = true;
                    if (stopEarly) {
                        return true;
                    }
                }
            }
        }
        if (it0.Valid() != it1.Valid()) {
            ReportDiff(EDiffType::TokenCountDifferent, lineNumber, columnNumber, it0, it1);
            return true;
        }
    }
    if (eof0 != eof1) {
        ReportDiff(EDiffType::LineCountDifferent, lineNumber, 0, {}, {},
            TStringBuilder() << (eof0 ? "<EOF>" : "<line>")  << " vs " << (eof1 ? "<EOF>" : "<line>"));
        return true;
    }
    return isDiffDetected;
}

int main(int argc, const char* argv[]) {
    TString delimiter = "\t";

    auto opts = NLastGetopt::TOpts::Default();
    opts.AddHelpOption();
    opts.SetTitle("Compare files token-wise. Allow and report numerical differences incrementally (up to the specified threshold)");

    opts.AddLongOption('d', "delimiter")
        .RequiredArgument("<string>")
        .StoreResult(&delimiter)
        .DefaultValue(delimiter)
        .Help("Use this delimiter");

    opts.AddLongOption("have-header")
        .NoArgument()
        .Help("files have header (don't treat first line columns as numerical even with precision spec)");

    opts.AddLongOption("diff-limit")
        .RequiredArgument("THRESHOLD")
        .Help("tolerate token-wise err less than THRESHOLD (err = min(abs(diff), rel(diff))");

    opts.SetFreeArgsNum(2);
    opts.SetFreeArgTitle(0, "<input-file1>", "Input dsv file");
    opts.SetFreeArgTitle(1, "<input-file2>", "Input dsv file");

    NLastGetopt::TOptsParseResult res(&opts, argc, argv);

    const bool haveHeader = res.Has("have-header");
    const bool stopEarly = res.Has("diff-limit");
    const double diffLimit = res.GetOrElse<double>("diff-limit", 0.0);

    auto inputFileNames = res.GetFreeArgs();

    TIFStream input0(inputFileNames[0]);
    TIFStream input1(inputFileNames[1]);

    return AreFilesDifferent(input0, input1, haveHeader, delimiter, diffLimit, stopEarly);
}
