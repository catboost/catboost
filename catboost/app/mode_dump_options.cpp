#include "modes.h"

#include <catboost/libs/helpers/exception.h>

#include <library/cpp/getopt/small/modchooser.h>
#include <library/cpp/getopt/small/last_getopt.h>
#include <library/cpp/json/json_reader.h>
#include <library/cpp/json/json_value.h>

#include <util/generic/string.h>
#include <util/stream/file.h>
#include <util/stream/output.h>
#include <util/ysaveload.h>

static NJson::TJsonValue GetOptions(const TString& snapshotPath) {
    TFileInput snapthotFile(snapshotPath);
    TString label;
    TString optionsStr;
    ::LoadMany(&snapthotFile, label, optionsStr);
    NJson::TJsonValue options;
    CB_ENSURE(ReadJsonTree(optionsStr, &options), "Unable to parse options from snapshot");
    return options;
}

int mode_dump_options(int argc, const char* argv[]) {
    auto parser = NLastGetopt::TOpts();
    TString snapshotPath;
    parser
        .AddLongOption("input", "path to snapshot file")
        .RequiredArgument("STRING")
        .StoreResult(&snapshotPath);
    TString optionsPath;
    parser
        .AddLongOption("output", "path to options file; omit to output to stdout")
        .OptionalArgument("STRING")
        .StoreResult(&optionsPath);
    parser.SetFreeArgsMax(0);
    NLastGetopt::TOptsParseResult parserResult{&parser, argc, argv};
    const auto options = ToString(GetOptions(snapshotPath));
    if (optionsPath.empty()) {
        Cout << options;
    } else {
        TFileOutput optionsFile(optionsPath);
        optionsFile << options;
    }
    return 0;
}
