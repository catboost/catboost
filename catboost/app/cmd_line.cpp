#include "cmd_line.h"

#include "bind_options.h"

#include <catboost/libs/helpers/exception.h>

#include <util/generic/strbuf.h>
#include <util/string/iterator.h>
#include <util/string/vector.h>
#include <util/system/info.h>


using namespace NCB;


void TAnalyticalModeCommonParams::BindParserOpts(NLastGetopt::TOpts& parser) {
    BindDsvPoolFormatParams(&parser, &DsvPoolFormatParams);

    parser.AddLongOption('m', "model-path", "path to model")
        .StoreResult(&ModelFileName)
        .DefaultValue("model.bin");
    parser.AddLongOption("input-path", "input path")
        .DefaultValue("input.tsv")
        .Handler1T<TStringBuf>([&](const TStringBuf& str) {
            InputPath = TPathWithScheme(str, "dsv");
        });
    parser.AddLongOption('o', "output-path", "output result path")
        .StoreResult(&OutputPath)
        .DefaultValue("output.tsv");
    parser.AddLongOption('T', "thread-count", "worker thread count (default: core count)")
        .StoreResult(&ThreadCount);
    parser.AddLongOption("class-names", "names for classes.")
            .RequiredArgument("comma separated list of names")
            .Handler1T<TString>([&](const TString& namesLine) {
                for (const auto& t : StringSplitter(namesLine).Split(',')) {
                    ClassNames.push_back(FromString<TString>(t.Token()));
                }
            });
}
