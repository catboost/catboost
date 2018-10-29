#include "bind_options.h"
#include "modes.h"

#include <catboost/libs/algo/roc_curve.h>
#include <catboost/libs/helpers/binarize_target.h>
#include <catboost/libs/data/load_data.h>
#include <catboost/libs/data_util/line_data_reader.h>
#include <catboost/libs/model/model.h>
#include <catboost/libs/options/output_file_options.h>

#include <library/getopt/small/last_getopt.h>

#include <util/system/fs.h>
#include <util/system/info.h>
#include <util/string/iterator.h>


using namespace NCB;


struct TRocParams {
    TString OutputPath;
    TString EvalResultPath;
    float LabelBinarizationBorder;
    int ThreadCount;

    void BindParserOpts(NLastGetopt::TOpts& parser) {
        parser.AddLongOption('i', "eval-file", "eval result path")
            .StoreResult(&EvalResultPath)
            .RequiredArgument("PATH");
        parser.AddLongOption('o', "output-path", "output result path")
            .StoreResult(&OutputPath)
            .RequiredArgument("PATH")
            .DefaultValue("roc_data.tsv");
        parser.AddLongOption('b', "label-border", "label binarization border")
            .StoreResult(&LabelBinarizationBorder)
            .RequiredArgument("FLOAT")
            .DefaultValue(0.5);
        parser.AddLongOption('T', "thread-count", "worker thread count (default: core count)")
            .StoreResult(&ThreadCount)
            .RequiredArgument("INT")
            .DefaultValue(NSystemInfo::CachedNumberOfCpus());
    }
};

static void ParseEvalResult(const TString& evalResultPath, TVector<double>* approxes, TVector<float>* labels) {
    approxes->clear();
    labels->clear();
    THolder<ILineDataReader> reader = GetLineDataReader(TPathWithScheme(evalResultPath));

    TString header;
    reader->ReadLine(&header);
    TVector<TString> headerTokens = StringSplitter(header).Split('\t');
    THashMap<TString, ui32> columnHeaders;
    for (ui32 i = 0; i < headerTokens.size(); ++i) {
        columnHeaders[headerTokens[i]] = i;
    }
    CB_ENSURE(columnHeaders.has("Label") && columnHeaders.has("RawFormulaVal"), "Incorrect EvalResult format.");
    const ui32 approxColumnIndex = columnHeaders.at("RawFormulaVal");
    const ui32 labelColumnIndex = columnHeaders.at("Label");

    TString line;
    while (reader->ReadLine(&line)) {
        TVector<TString> tokens = StringSplitter(line).Split('\t');
        if (tokens.empty()) {
            continue;
        }
        CB_ENSURE(tokens.size() > Max(approxColumnIndex, labelColumnIndex));
        approxes->push_back(FromString<double>(tokens[approxColumnIndex]));
        labels->push_back(FromString<float>(tokens[labelColumnIndex]));
    }
}

int mode_roc(int argc, const char* argv[]) {
    TRocParams params;
    auto parser = NLastGetopt::TOpts();
    parser.AddHelpOption();
    params.BindParserOpts(parser);
    parser.SetFreeArgsNum(0);
    NLastGetopt::TOptsParseResult parserResult{&parser, argc, argv};

    TVector<TVector<double>> approxes(1);
    TVector<TVector<float>> labels(1);
    ParseEvalResult(params.EvalResultPath, &approxes[0], &labels[0]);
    PrepareTargetBinary(params.LabelBinarizationBorder, &labels[0]);

    TRocCurve rocCurve(approxes, labels, params.ThreadCount);
    rocCurve.OutputRocCurve(params.OutputPath);
    return 0;
}
