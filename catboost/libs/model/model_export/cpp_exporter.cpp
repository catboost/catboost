#include "cpp_exporter.h"

#include "export_helpers.h"

#include <catboost/libs/model/ctr_helpers.h>
#include <catboost/libs/model/static_ctr_provider.h>

#include <library/cpp/resource/resource.h>

#include <util/generic/map.h>
#include <util/generic/set.h>
#include <util/string/builder.h>
#include <util/string/cast.h>
#include <util/stream/input.h>

namespace NCB {
    using namespace NCatboostModelExportHelpers;

    /*
     * Tiny code for case when cat features not present
     */

    void TCatboostModelToCppConverter::WriteApplicator() {
        Out << "/* Model applicator */" << '\n';
        Out << "double ApplyCatboostModel(" << '\n';
        Out << "    const std::vector<float>& features" << '\n';
        Out << ") {" << '\n';
        Out << "    const struct CatboostModel& model = CatboostModelStatic;" << '\n';
        Out << '\n';
        Out << "    /* Binarise features */" << '\n';
        Out << "    std::vector<unsigned char> binaryFeatures(model.BinaryFeatureCount);" << '\n';
        Out << "    unsigned int binFeatureIndex = 0;" << '\n';
        Out << "    for (unsigned int i = 0; i < model.FloatFeatureCount; ++i) {" << '\n';
        Out << "        for(unsigned int j = 0; j < model.BorderCounts[i]; ++j) {" << '\n';
        Out << "            binaryFeatures[binFeatureIndex] = (unsigned char)(features[i] > model.Borders[binFeatureIndex]);" << '\n';
        Out << "            ++binFeatureIndex;" << '\n';
        Out << "        }" << '\n';
        Out << "    }" << '\n';
        Out << '\n';
        Out << "    /* Extract and sum values from trees */" << '\n';
        Out << "    double result = 0.0;" << '\n';
        Out << "    const unsigned int* treeSplitsPtr = model.TreeSplits;" << '\n';
        Out << "    const double* leafValuesForCurrentTreePtr = model.LeafValues;" << '\n';
        Out << "    for (unsigned int treeId = 0; treeId < model.TreeCount; ++treeId) {" << '\n';
        Out << "        const unsigned int currentTreeDepth = model.TreeDepth[treeId];" << '\n';
        Out << "        unsigned int index = 0;" << '\n';
        Out << "        for (unsigned int depth = 0; depth < currentTreeDepth; ++depth) {" << '\n';
        Out << "            index |= (binaryFeatures[treeSplitsPtr[depth]] << depth);" << '\n';
        Out << "        }" << '\n';
        Out << "        result += leafValuesForCurrentTreePtr[index];" << '\n';
        Out << "        treeSplitsPtr += currentTreeDepth;" << '\n';
        Out << "        leafValuesForCurrentTreePtr += (1 << currentTreeDepth);" << '\n';
        Out << "    }" << '\n';
        Out << "    return model.Scale * result + model.Bias;" << '\n';
        Out << "}" << '\n';

        // Also emit the API with catFeatures, for uniformity
        Out << '\n';
        Out << "double ApplyCatboostModel(" << '\n';
        Out << "    const std::vector<float>& floatFeatures," << '\n';
        Out << "    const std::vector<std::string>&" << '\n';
        Out << ") {" << '\n';
        Out << "    return ApplyCatboostModel(floatFeatures);" << '\n';
        Out << "}" << '\n';
    }

    void TCatboostModelToCppConverter::WriteModel(const TFullModel& model) {
        CB_ENSURE(!model.HasCategoricalFeatures(), "Export of model with categorical features to cpp is not yet supported.");
        CB_ENSURE(model.ModelTrees->GetDimensionsCount() == 1, "Export of MultiClassification model to cpp is not supported.");
        Out << "/* Model data */" << '\n';

        int binaryFeatureCount = GetBinaryFeatureCount(model);

        Out << "static const struct CatboostModel {" << '\n';
        Out << "    unsigned int FloatFeatureCount = " << model.GetNumFloatFeatures() << ";" << '\n';
        Out << "    unsigned int BinaryFeatureCount = " << binaryFeatureCount << ";" << '\n';
        Out << "    unsigned int TreeCount = " << model.ModelTrees->GetTreeSizes().size() << ";" << '\n';

        Out << "    unsigned int TreeDepth[" << model.ModelTrees->GetTreeSizes().size() << "] = {" << OutputArrayInitializer(model.ModelTrees->GetTreeSizes()) << "};" << '\n';
        Out << "    unsigned int TreeSplits[" << model.ModelTrees->GetTreeSplits().size() << "] = {" << OutputArrayInitializer(model.ModelTrees->GetTreeSplits()) << "};" << '\n';

        Out << "    unsigned int BorderCounts[" << model.ModelTrees->GetNumFloatFeatures() << "] = {" << OutputBorderCounts(model) << "};" << '\n';

        Out << "    float Borders[" << binaryFeatureCount << "] = {" << OutputBorders(model, true) << "};" << '\n';

        Out << '\n';
        Out << "    /* Aggregated array of leaf values for trees. Each tree is represented by a separate line: */" << '\n';
        Out << "    double LeafValues[" << model.ModelTrees->GetLeafValues().size() << "] = {" << OutputLeafValues(model, TIndent(1));
        Out << "    };" << '\n';
        Out << "    double Scale = " << model.GetScaleAndBias().Scale << ";" << '\n';
        Out << "    double Bias = " << model.GetScaleAndBias().Bias << ";" << '\n';
        Out << "} CatboostModelStatic;" << '\n';
        Out << '\n';
    }

    void TCatboostModelToCppConverter::WriteHeader(bool forCatFeatures) {
        if (forCatFeatures) {
           Out << "#include <cassert>" << '\n';
        }
        Out << "#include <string>" << '\n';
        Out << "#include <vector>" << '\n';
        if (forCatFeatures) {
            Out << "#include <unordered_map>" << '\n';
        }
        Out << '\n';
    }

    /*
     * Full model code with complete support of cat features
     */

    void TCatboostModelToCppConverter::WriteCTRStructs() {
        Out << NResource::Find("catboost_model_export_cpp_ctr_structs");
    };

    static inline TString WriteInitializerName(const TString& name) {
        TStringBuilder out;
        out << "." << name << " = ";
        return out;
    };

    static void WriteModelCTRs(IOutputStream& out, const TFullModel& model, TIndent indent) {
        TSequenceCommaSeparator comma;
        out << indent++ << "struct TCatboostCPPExportModelCtrs modelCtrs = {" << '\n';

        const TVector<TModelCtr>& neededCtrs = model.ModelTrees->GetUsedModelCtrs();
        if (neededCtrs.size() == 0) {
            out << --indent << "};" << '\n';
            return;
        }

        auto WN = WriteInitializerName;

        const TStaticCtrProvider* ctrProvider = dynamic_cast<TStaticCtrProvider*>(model.CtrProvider.Get());
        CB_ENSURE(ctrProvider, "Unsupported CTR provider");

        TVector<TCompressedModelCtr> compressedModelCtrs = CompressModelCtrs(neededCtrs);

        out << indent << WN("UsedModelCtrsCount") << model.ModelTrees->GetUsedModelCtrs().size() << "," << '\n';
        out << indent++ << WN("CompressedModelCtrs") << "{" << '\n';

        comma.ResetCount(compressedModelCtrs.size());
        for (const auto& compressedCtr : compressedModelCtrs) {
            TSequenceCommaSeparator commaInner;
            out << indent++ << "{" << '\n';

            out << indent++ << WN("Projection") << "{" << '\n';

            auto& proj = *compressedCtr.Projection;
            TVector<int> transposedCatFeatureIndexes;
            out << indent << WN("transposedCatFeatureIndexes") << "{";
            TSequenceCommaSeparator commaInnerWithSpace(proj.CatFeatures.size(), AddSpaceAfterComma);
            for (const auto feature : proj.CatFeatures) {
                out << feature << commaInnerWithSpace;
            }
            out << "}," << '\n';
            out << indent++ << WN("binarizedIndexes") << "{";
            commaInner.ResetCount(proj.BinFeatures.size() + proj.OneHotFeatures.size());
            for (const auto& feature : proj.BinFeatures) {
                const TBinFeatureIndexValue& featureValue = ctrProvider->GetFloatFeatureIndexes().at(feature);
                out << '\n' << indent << "{";
                out << WN("BinIndex") << featureValue.BinIndex << ", ";
                out << WN("CheckValueEqual") << featureValue.CheckValueEqual << ", ";
                out << WN("Value") << (int)featureValue.Value;
                out << "}" << commaInner;
            }
            for (const auto& feature : proj.OneHotFeatures) {
                const TBinFeatureIndexValue& featureValue = ctrProvider->GetOneHotFeatureIndexes().at(feature);
                out << '\n' << indent << "{";
                out << WN("BinIndex") << featureValue.BinIndex << ", ";
                out << WN("CheckValueEqual") << featureValue.CheckValueEqual << ", ";
                out << WN("Value") << (int)featureValue.Value;
                out << "}" << commaInner;
            }
            --indent;
            if (proj.BinFeatures.size() > 0 || proj.OneHotFeatures.size() > 0) {
                out << '\n' << indent;
            }
            out << "}," << '\n';

            out << --indent << "}," << '\n';
            out << indent++ << WN("ModelCtrs") << "{" << '\n';
            commaInner.ResetCount(compressedCtr.ModelCtrs.size());
            for (const auto& ctr : compressedCtr.ModelCtrs) {
                TSequenceCommaSeparator commaLocal(7, AddSpaceAfterComma);
                out << indent << "{";
                out << WN("BaseHash") << ctr->Base.GetHash() << "ull" << commaLocal;
                out << WN("BaseCtrType") << "ECatboostCPPExportModelCtrType::" << ctr->Base.CtrType << commaLocal;
                out << WN("TargetBorderIdx") << ctr->TargetBorderIdx << commaLocal;
                out << WN("PriorNum") << ctr->PriorNum << commaLocal;
                out << WN("PriorDenom") << ctr->PriorDenom << commaLocal;
                out << WN("Shift") << ctr->Shift << commaLocal;
                out << WN("Scale") << ctr->Scale << commaLocal;
                out << "}" << commaInner << '\n';
            }
            out << --indent << "}" << '\n';
            out << --indent << "}" << comma << '\n';
        }
        out << --indent << "}," << '\n';
        out << indent++ << WN("CtrData") << "{" << '\n';
        out << indent++ << WN("LearnCtrs") << "{" << '\n';
        comma.ResetCount(ctrProvider->CtrData.LearnCtrs.size());
        TMap<ui64, const TCtrValueTable*> orderedLearnCtrs;
        for (const auto& learnCtr : ctrProvider->CtrData.LearnCtrs) {
            orderedLearnCtrs.emplace(learnCtr.first.GetHash(), &learnCtr.second);
        }
        for (const auto& orderedLearnCtr : orderedLearnCtrs) {
            const auto& learnCtrValueTable = *orderedLearnCtr.second;
            TSequenceCommaSeparator commaInner(AddSpaceAfterComma);
            out << indent++ << "{" << '\n';
            out << indent << orderedLearnCtr.first << "ull," << '\n';
            out << indent++ << "{" << '\n';
            out << indent << WN("IndexHashViewer") << "{";
            const TConstArrayRef<NCatboost::TBucket> HashViewerBuckets = learnCtrValueTable.GetIndexHashViewer().GetBuckets();
            commaInner.ResetCount(HashViewerBuckets.size());
            for (const auto& bucket : HashViewerBuckets) {
                out << "{" << bucket.Hash << "ull, " << bucket.IndexValue << "}" << commaInner;
            }
            out << "}," << '\n';
            out << indent << WN("TargetClassesCount") << learnCtrValueTable.TargetClassesCount << "," << '\n';
            out << indent << WN("CounterDenominator") << learnCtrValueTable.CounterDenominator << "," << '\n';
            const TConstArrayRef<TCtrMeanHistory> ctrMeanHistories = learnCtrValueTable.GetTypedArrayRefForBlobData<TCtrMeanHistory>();
            out << indent << WN("CtrMeanHistory") << "{";
            commaInner.ResetCount(ctrMeanHistories.size());
            for (const auto& ctrMean : ctrMeanHistories) {
                out << "{";
                out << WN("Sum") << ctrMean.Sum << ", ";
                out << WN("Count") << ctrMean.Count;
                out << "}" << commaInner;
            }
            out << "}," << '\n';
            const TConstArrayRef<int> ctrTotal = learnCtrValueTable.GetTypedArrayRefForBlobData<int>();
            out << indent << WN("CtrTotal") << "{" << OutputArrayInitializer(ctrTotal) << "}" << '\n';
            out << --indent << "}" << '\n';
            out << --indent << "}" << comma << '\n';
        };
        out << --indent << "}" << '\n';
        out << --indent << "}" << '\n';
        out << --indent << "};" << '\n';
    };

    void TCatboostModelToCppConverter::WriteModelCatFeatures(const TFullModel& model, const THashMap<ui32, TString>* catFeaturesHashToString) {
        CB_ENSURE(model.ModelTrees->GetDimensionsCount() == 1, "Export of MultiClassification model to cpp is not supported.");

        WriteCTRStructs();
        Out << '\n';

        TIndent indent(0);
        TSequenceCommaSeparator comma;
        Out << "/* Model data */" << '\n';

        int binaryFeatureCount = model.ModelTrees->GetEffectiveBinaryFeaturesBucketsCount();

        Out << indent++ << "static const struct CatboostModel {" << '\n';
        Out << indent << "CatboostModel() {};" << '\n';
        Out << indent << "unsigned int FloatFeatureCount = " << model.GetNumFloatFeatures() << ";" << '\n';
        Out << indent << "unsigned int CatFeatureCount = " << model.GetNumCatFeatures() << ";" << '\n';
        Out << indent << "unsigned int BinaryFeatureCount = " << binaryFeatureCount << ";" << '\n';
        Out << indent << "unsigned int TreeCount = " << model.GetTreeCount() << ";" << '\n';

        Out << indent++ << "std::vector<std::vector<float>> FloatFeatureBorders = {" << '\n';
        comma.ResetCount(model.ModelTrees->GetFloatFeatures().size());
        for (const auto& floatFeature : model.ModelTrees->GetFloatFeatures()) {
            Out << indent << "{"
                << OutputArrayInitializer([&floatFeature](size_t i) { return FloatToString(floatFeature.Borders[i], PREC_NDIGITS, 9); }, floatFeature.Borders.size())
                << "}" << comma << '\n';
        }
        Out << --indent << "};" << '\n';

        Out << indent << "std::vector<unsigned int> TreeDepth = {" << OutputArrayInitializer(model.ModelTrees->GetTreeSizes()) << "};" << '\n';
        Out << indent << "std::vector<unsigned int> TreeSplits = {" << OutputArrayInitializer(model.ModelTrees->GetTreeSplits()) << "};" << '\n';

        const auto& bins = model.ModelTrees->GetRepackedBins();
        Out << indent << "std::vector<unsigned char> TreeSplitIdxs = {" << OutputArrayInitializer([&bins](size_t i) { return (int)bins[i].SplitIdx; }, bins.size()) << "};" << '\n';
        Out << indent << "std::vector<unsigned short> TreeSplitFeatureIndex = {" << OutputArrayInitializer([&bins](size_t i) { return (int)bins[i].FeatureIndex; }, bins.size()) << "};" << '\n';
        Out << indent << "std::vector<unsigned char> TreeSplitXorMask = {" << OutputArrayInitializer([&bins](size_t i) { return (int)bins[i].XorMask; }, bins.size()) << "};" << '\n';

        Out << indent << "std::vector<unsigned int> CatFeaturesIndex = {"
            << OutputArrayInitializer([&model](size_t i) { return model.ModelTrees->GetCatFeatures()[i].Position.Index; }, model.ModelTrees->GetCatFeatures().size()) << "};" << '\n';

        Out << indent << "std::vector<unsigned int> OneHotCatFeatureIndex = {"
            << OutputArrayInitializer([&model](size_t i) { return model.ModelTrees->GetOneHotFeatures()[i].CatFeatureIndex; }, model.ModelTrees->GetOneHotFeatures().size())
            << "};" << '\n';

        Out << indent++ << "std::vector<std::vector<int>> OneHotHashValues = {" << '\n';
        comma.ResetCount(model.ModelTrees->GetOneHotFeatures().size());
        for (const auto& oneHotFeature : model.ModelTrees->GetOneHotFeatures()) {
            Out << indent << "{"
                << OutputArrayInitializer([&oneHotFeature](size_t i) { return oneHotFeature.Values[i]; }, oneHotFeature.Values.size())
                << "}" << comma << '\n';
        }
        Out << --indent << "};" << '\n';

        Out << indent++ << "std::vector<std::vector<float>> CtrFeatureBorders = {" << '\n';
        comma.ResetCount(model.ModelTrees->GetCtrFeatures().size());
        for (const auto& ctrFeature : model.ModelTrees->GetCtrFeatures()) {
            Out << indent << "{"
                << OutputArrayInitializer([&ctrFeature](size_t i) { return FloatToString(ctrFeature.Borders[i], PREC_NDIGITS, 9) + "f"; }, ctrFeature.Borders.size())
                << "}" << comma << '\n';
        }
        Out << --indent << "};" << '\n';

        Out << '\n';
        Out << indent << "/* Aggregated array of leaf values for trees. Each tree is represented by a separate line: */" << '\n';
        Out << indent << "double LeafValues[" << model.ModelTrees->GetLeafValues().size() << "] = {" << OutputLeafValues(model, indent);
        Out << indent << "};" << '\n';
        Out << indent << "double Scale = " << model.GetScaleAndBias().Scale << ";" << '\n';
        Out << indent << "double Bias = " << model.GetScaleAndBias().Bias << ";" << '\n';

        WriteModelCTRs(Out, model, indent);

        Out << "} CatboostModelStatic;" << '\n';
        Out << '\n';

        indent--;
        Out << indent++ << "static std::unordered_map<std::string, int> CatFeatureHashes = {" << '\n';
        if (catFeaturesHashToString != nullptr) {
            TSet<int> ordered_keys;
            for (const auto& key_value: *catFeaturesHashToString) {
                ordered_keys.insert(key_value.first);
            }
            for (const auto& key_value: ordered_keys) {
                Out << indent << "{" << catFeaturesHashToString->at(key_value).Quote() << ", "  << key_value << "},\n";
            }
        }
        Out << --indent << "};" << '\n';
        Out << '\n';
    }

    void TCatboostModelToCppConverter::WriteApplicatorCatFeatures() {
        Out << NResource::Find("catboost_model_export_cpp_ctr_calcer");
        Out << '\n';
        Out << NResource::Find("catboost_model_export_cpp_model_applicator");
    }
}
