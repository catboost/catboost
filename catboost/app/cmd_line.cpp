#include "cmd_line.h"

#include <catboost/libs/helpers/exception.h>

#include <util/generic/strbuf.h>
#include <util/string/iterator.h>
#include <util/string/vector.h>
#include <util/system/info.h>



void TAnalyticalModeCommonParams::BindParserOpts(NLastGetopt::TOpts& parser) {
    parser.AddLongOption('m', "model-path", "path to model")
        .StoreResult(&ModelFileName)
        .DefaultValue("model.bin");
    parser.AddLongOption("input-path", "input path")
        .StoreResult(&InputPath)
        .DefaultValue("input.tsv");
    parser.AddLongOption("column-description", "path to columns descriptions")
        .AddLongName("cd")
        .StoreResult(&CdFile)
        .DefaultValue("");
    parser.AddLongOption('o', "output-path", "output result path")
        .StoreResult(&OutputPath)
        .DefaultValue("output.tsv");
    parser.AddLongOption('T', "thread-count", "worker thread count (default: core count)")
        .StoreResult(&ThreadCount);
    parser.AddLongOption("delimiter", "delimiter")
            .DefaultValue("\t")
            .Handler1T<TString>([&](const TString& oneChar) {
                CB_ENSURE(oneChar.size() == 1, "only single char delimiters supported");
                Delimiter = oneChar[0];
            });
    parser.AddLongOption("has-header", "has header flag")
            .NoArgument()
            .StoreValue(&HasHeader, true);
    parser.AddLongOption("class-names", "names for classes.")
            .RequiredArgument("comma separated list of names")
            .Handler1T<TString>([&](const TString& namesLine) {
                for (const auto& t : StringSplitter(namesLine).Split(',')) {
                    ClassNames.push_back(FromString<TString>(t.Token()));
                }
            });
}
