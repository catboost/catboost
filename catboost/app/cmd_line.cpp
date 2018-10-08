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
    BindModelFileParams(&parser, &ModelFileName, &ModelFormat);
    parser.AddLongOption("input-path", "input path")
        .DefaultValue("input.tsv")
        .Handler1T<TStringBuf>([&](const TStringBuf& pathWithScheme) {
            InputPath = TPathWithScheme(pathWithScheme, "dsv");
        });
    parser.AddLongOption('o', "output-path", "output result path")
        .DefaultValue("output.tsv")
        .Handler1T<TStringBuf>([&](const TStringBuf& pathWithScheme) {
            OutputPath = TPathWithScheme(pathWithScheme, "dsv");
        });
    parser.AddLongOption('T', "thread-count", "worker thread count (default: core count)")
        .StoreResult(&ThreadCount);
}
