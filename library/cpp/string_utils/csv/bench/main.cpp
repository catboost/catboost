#include <library/cpp/string_utils/csv/csv.cpp>
#include <util/string/split.h>
#include <library/testing/benchmark/bench.h>

TString TestString = "2015-12-22T14:46:32.490Z,38.8218346,-122.8453369,2.09,0.89,md,12,127,0.008152,0.02,nc,nc72570596,2015-12-22T14:48:05.760Z,earthquake";
TString TripleQuotesString = "\"\"\"triple \"\" quotes\"\"\",\"\"\"triple \"\" quotes\"\"\"";
Y_CPU_BENCHMARK(CsvSplitter, iface) {
    for (size_t i = 0; i < iface.Iterations(); ++i) {
        auto splitter = NCsvFormat::CsvSplitter(TestString);
        do {
            TStringBuf buf = splitter.Consume();
            Y_UNUSED(buf);
        } while (splitter.Step());
    }
}
Y_CPU_BENCHMARK(CsvSplitterWithoutEscaping, iface) {
    for (size_t i = 0; i < iface.Iterations(); ++i) {
        auto splitter = NCsvFormat::CsvSplitter(TestString);
        do {
            TStringBuf buf = splitter.Consume();
            Y_UNUSED(buf);
        } while (splitter.Step());
    }
}
Y_CPU_BENCHMARK(ClassicSplitter, iface) {
    const TStringBuf buf(TestString);
    for (size_t i = 0; i < iface.Iterations(); ++i) {
        for (TStringBuf buf : StringSplitter((TString)TestString).Split(',')) {
            Y_UNUSED(buf);
        };
    }
}
Y_CPU_BENCHMARK(CsvSplitterTripleQuotes, iface) {
    for (size_t i = 0; i < iface.Iterations(); ++i) {
        auto splitter = NCsvFormat::CsvSplitter(TestString);
        do {
            TStringBuf buf = splitter.Consume();
            Y_UNUSED(buf);
        } while (splitter.Step());
    }
}
Y_CPU_BENCHMARK(ClassicSplitterTripleQuotes, iface) {
    const TStringBuf buf(TestString);
    for (size_t i = 0; i < iface.Iterations(); ++i) {
        for (TStringBuf buf : StringSplitter((TString)TripleQuotesString).Split(',')) {
            Y_UNUSED(buf);
        };
    }
}
