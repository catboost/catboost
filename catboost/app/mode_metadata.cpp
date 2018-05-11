#include "modes.h"
#include "model_metainfo_helpers.h"

#include <catboost/libs/model/model.h>

#include <library/getopt/small/last_getopt.h>
#include <library/getopt/small/modchooser.h>


struct TCommonMetaInfoParams {
    TString ModelPath;
    TFullModel Model;

    void BindParser(NLastGetopt::TOpts& parser) {
        parser.AddLongOption('m', "model-path", "path to model")
            .StoreResult(&ModelPath)
            .DefaultValue("model.cbm");
    }

    void LoadModel() {
        Model = ReadModel(ModelPath);
    }
};

int set_key(int argc, const char* argv[]) {
    TCommonMetaInfoParams params;
    TString key;
    TString value;
    TString outputModelPath;
    auto parser = NLastGetopt::TOpts();
    parser.AddHelpOption();
    params.BindParser(parser);
    parser.AddLongOption("key", "key name")
        .RequiredArgument("NAME")
        .StoreResult(&key);
    parser.AddLongOption("value", "value")
        .RequiredArgument("VALUE")
        .StoreResult(&value);
    parser.AddLongOption('o', "output-model-path")
        .OptionalArgument("PATH")
        .StoreResult(&outputModelPath);
    parser.SetFreeArgsMax(0);
    NLastGetopt::TOptsParseResult parserResult{&parser, argc, argv};
    params.LoadModel();
    params.Model.ModelInfo[key] = value;
    if (outputModelPath.Empty()) {
        ExportModel(params.Model, params.ModelPath);
    } else {
        ExportModel(params.Model, outputModelPath);
    }
    return 0;
}

int get_keys(int argc, const char* argv[]) {
    TCommonMetaInfoParams params;
    TVector<TString> keys;
    auto parser = NLastGetopt::TOpts();
    parser.AddHelpOption();
    params.BindParser(parser);
    parser.AddLongOption("key", "keys to dump")
        .AppendTo(&keys);
    parser.SetFreeArgDefaultTitle("KEY", "you can use freeargs to select keys");
    NLastGetopt::TOptsParseResult parserResult{&parser, argc, argv};
    params.LoadModel();
    for (const auto& key : parserResult.GetFreeArgs()) {
        keys.push_back(key);
    }
    CB_ENSURE(!keys.empty(), "Select at least one property to dump");
    for (const auto& key : keys) {
        Cout << key << "\t" << params.Model.ModelInfo[key] << Endl;
    }
    return 0;
}

int dump(int argc, const char* argv[]) {
    TCommonMetaInfoParams params;
    NCB::EInfoDumpFormat dumpFormat;
    auto parser = NLastGetopt::TOpts();
    parser.AddHelpOption();
    params.BindParser(parser);
    parser.AddLongOption("dump-format", "One of Plain, JSON")
        .DefaultValue("Plain")
        .StoreResult(&dumpFormat);
    parser.SetFreeArgsMax(0);
    NLastGetopt::TOptsParseResult parserResult{&parser, argc, argv};
    params.LoadModel();
    if (NCB::EInfoDumpFormat::Plain == dumpFormat) {
        for (const auto& keyValue : params.Model.ModelInfo) {
            Cout << keyValue.first << "\t" << keyValue.second << Endl;
        }
    } else if (NCB::EInfoDumpFormat::JSON == dumpFormat) {
        NJson::TJsonValue value;
        for (const auto& keyValue : params.Model.ModelInfo) {
            value[keyValue.first] = keyValue.second;
        }
        Cout << value.GetStringRobust() << Endl;
    }
    return 0;
}

int mode_metadata(int argc, const char* argv[]) {
    TModChooser modChooser;
    modChooser.AddMode("set", set_key, "set model property by name/value");
    modChooser.AddMode("get", get_keys, "get model property/properties by name");
    modChooser.AddMode("dump", dump, "dump model info fields");
    modChooser.DisableSvnRevisionOption();
    modChooser.Run(argc, argv);
    return 0;
}
