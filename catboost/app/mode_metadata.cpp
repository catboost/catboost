#include "modes.h"
#include "model_metainfo_helpers.h"

#include <catboost/libs/helpers/exception.h>
#include <catboost/libs/model/model.h>
#include <catboost/libs/options/analytical_mode_params.h>

#include <library/getopt/small/modchooser.h>
#include <library/getopt/small/last_getopt.h>
#include <library/json/json_value.h>

#include <util/stream/output.h>
#include <util/string/join.h>


struct TCommonMetaInfoParams {
    TString ModelPath;
    EModelType ModelFormat = EModelType::CatboostBinary;
    TFullModel Model;

    void BindParser(NLastGetopt::TOpts& parser) {
        NCB::BindModelFileParams(&parser, &ModelPath, &ModelFormat);
    }

    void LoadModel() {
        Model = ReadModel(ModelPath, ModelFormat);
    }
};

int set_key(int argc, const char* argv[]) {
    TCommonMetaInfoParams params;
    TString key;
    TString value;
    TString outputModelPath;
    EModelType outputModelFormat = EModelType::CatboostBinary;
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
            .Handler1T<TString>([&outputModelPath, &outputModelFormat](const TString& path) {
                outputModelPath = path;
                outputModelFormat = NCatboostOptions::DefineModelFormat(path);
            });
    parser.AddLongOption("output-model-format")
            .OptionalArgument("output model format")
            .Handler1T<TString>([&outputModelFormat](const TString& format) {
                outputModelFormat = FromString<EModelType>(format);
            });
    parser.SetFreeArgsMax(0);
    NLastGetopt::TOptsParseResult parserResult{&parser, argc, argv};
    params.LoadModel();
    params.Model.ModelInfo[key] = value;
    if (outputModelPath.empty()) {
        ExportModel(params.Model, params.ModelPath, outputModelFormat);
    } else {
        ExportModel(params.Model, outputModelPath, outputModelFormat);
    }
    return 0;
}

int get_keys(int argc, const char* argv[]) {
    TCommonMetaInfoParams params;
    NCB::EInfoDumpFormat dumpFormat;
    TVector<TString> keys;
    auto parser = NLastGetopt::TOpts();
    parser.AddHelpOption();
    params.BindParser(parser);
    parser.AddLongOption("key", "keys to dump")
        .AppendTo(&keys);
    parser.AddLongOption("dump-format", "One of Plain, JSON")
        .DefaultValue("Plain")
        .StoreResult(&dumpFormat);
    parser.SetFreeArgDefaultTitle("KEY", "you can use freeargs to select keys");
    NLastGetopt::TOptsParseResult parserResult{&parser, argc, argv};
    for (const auto& key : parserResult.GetFreeArgs()) {
        keys.push_back(key);
    }
    CB_ENSURE(!keys.empty(), "Select at least one property to dump");
    params.LoadModel();
    if (NCB::EInfoDumpFormat::Plain == dumpFormat) {
        for (const auto& key : keys) {
            Cout << key << "\t" << params.Model.ModelInfo[key] << Endl;
        }
    } else {
        NJson::TJsonValue value;
        for (const auto& key : keys) {
            value[key] = params.Model.ModelInfo[key];
        }
        Cout << value.GetStringRobust() << Endl;
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

int dump_feature_names(int argc, const char* argv[]) {
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
        Cout << "Feature names: " << JoinSeq("\t", GetModelUsedFeaturesNames(params.Model));
    } else if (NCB::EInfoDumpFormat::JSON == dumpFormat) {
        NJson::TJsonValue value;
        for (const auto& featureName : GetModelUsedFeaturesNames(params.Model)) {
            value["feature_names"].AppendValue(featureName);
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
    modChooser.AddMode("dump-feature-names", dump_feature_names, "dump feature names");
    modChooser.DisableSvnRevisionOption();
    modChooser.Run(argc, argv);
    return 0;
}
