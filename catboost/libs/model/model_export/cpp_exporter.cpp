#include "cpp_exporter.h"

#include "export_helpers.h"

#include <library/resource/resource.h>

#include <util/string/builder.h>
#include <util/string/cast.h>
#include <util/stream/input.h>
#include <util/stream/file.h>

namespace NCatboost {
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
        Out << "    return result;" << '\n';
        Out << "}" << '\n';
    }

    void TCatboostModelToCppConverter::WriteModel(const TFullModel& model) {
        CB_ENSURE(!model.HasCategoricalFeatures(), "Export of model with categorical features to CPP is not yet supported.");
        CB_ENSURE(model.ObliviousTrees.ApproxDimension == 1, "Export of MultiClassification model to CPP is not supported.");
        Out << "/* Model data */" << '\n';

        int binaryFeatureCount = GetBinaryFeatureCount(model);

        Out << "static const struct CatboostModel {" << '\n';
        Out << "    unsigned int FloatFeatureCount = " << model.ObliviousTrees.FloatFeatures.size() << ";" << '\n';
        Out << "    unsigned int BinaryFeatureCount = " << binaryFeatureCount << ";" << '\n';
        Out << "    unsigned int TreeCount = " << model.ObliviousTrees.TreeSizes.size() << ";" << '\n';

        Out << "    unsigned int TreeDepth[" << model.ObliviousTrees.TreeSizes.size() << "] = {" << OutputArrayInitializer(model.ObliviousTrees.TreeSizes) << "};" << '\n';
        Out << "    unsigned int TreeSplits[" << model.ObliviousTrees.TreeSplits.size() << "] = {" << OutputArrayInitializer(model.ObliviousTrees.TreeSplits) << "};" << '\n';

        Out << "    unsigned int BorderCounts[" << model.ObliviousTrees.FloatFeatures.size() << "] = {" << OutputBorderCounts(model) << "};" << '\n';

        Out << "    float Borders[" << binaryFeatureCount << "] = {" << OutputBorders(model, true) << "};" << '\n';

        Out << '\n';
        Out << "    /* Aggregated array of leaf values for trees. Each tree is represented by a separate line: */" << '\n';
        Out << "    double LeafValues[" << model.ObliviousTrees.LeafValues.size() << "] = {" << OutputLeafValues(model, TIndent(1));
        Out << "    };" << '\n';
        Out << "} CatboostModelStatic;" << '\n';
        Out << '\n';
    }

    void TCatboostModelToCppConverter::WriteHeader() {
        Out << "#include <vector>" << '\n';
        Out << '\n';
    }

    /*
     * Full model code with complete support of cat features
     */

    void TCatboostModelToCppConverter::WriteHeaderCatFeatures() {
        Out << "#include <string>" << '\n';
        Out << '\n';
        Out << "#ifdef GOOOGLE_CITY_HASH // Required revision https://github.com/google/cityhash/tree/00b9287e8c1255b5922ef90e304d5287361b2c2a or earlier" << '\n';
        Out << "    #include \"city.h\"" << '\n';
        Out << "#else" << '\n';
        Out << "    #include <util/digest/city.h>" << '\n';
        Out << "#endif" << '\n';
        Out << '\n';
    }

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

        const TVector<TModelCtr>& neededCtrs = model.ObliviousTrees.GetUsedModelCtrs();
        if (neededCtrs.size() == 0) {
            out << --indent << "};" << '\n';
            return;
        }

        auto WN = WriteInitializerName;

        const TStaticCtrProvider* ctrProvider = dynamic_cast<TStaticCtrProvider*>(model.CtrProvider.Get());
        CB_ENSURE(ctrProvider, "Unsupported CTR provider");

        TVector<TCompressedModelCtr> compressedModelCtrs = CompressModelCtrs(neededCtrs);

        out << indent << WN("UsedModelCtrsCount") << model.ObliviousTrees.GetUsedModelCtrs().size() << "," << '\n';
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
                out << ctrProvider->GetCatFeatureIndex().at(feature) << commaInnerWithSpace;
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
        for (const auto& learnCtr : ctrProvider->CtrData.LearnCtrs) {
            TSequenceCommaSeparator commaInner(AddSpaceAfterComma);
            out << indent++ << "{" << '\n';
            out << indent << learnCtr.first.GetHash() << "ull," << '\n';
            out << indent++ << "{" << '\n';
            out << indent << WN("IndexHashViewer") << "{";
            const TConstArrayRef<TBucket> HashViewerBuckets = learnCtr.second.GetIndexHashViewer().GetBuckets();
            commaInner.ResetCount(HashViewerBuckets.size());
            for (const auto& bucket : HashViewerBuckets) {
                out << "{" << bucket.Hash << "ull, " << bucket.IndexValue << "}" << commaInner;
            }
            out << "}," << '\n';
            out << indent << WN("TargetClassesCount") << learnCtr.second.TargetClassesCount << "," << '\n';
            out << indent << WN("CounterDenominator") << learnCtr.second.CounterDenominator << "," << '\n';
            const TConstArrayRef<TCtrMeanHistory> ctrMeanHistories = learnCtr.second.GetTypedArrayRefForBlobData<TCtrMeanHistory>();
            out << indent << WN("CtrMeanHistory") << "{";
            commaInner.ResetCount(ctrMeanHistories.size());
            for (const auto& ctrMean : ctrMeanHistories) {
                out << "{";
                out << WN("Sum") << ctrMean.Sum << ", ";
                out << WN("Count") << ctrMean.Count;
                out << "}" << commaInner;
            }
            out << "}," << '\n';
            const TConstArrayRef<int> ctrTotal = learnCtr.second.GetTypedArrayRefForBlobData<int>();
            out << indent << WN("CtrTotal") << "{" << OutputArrayInitializer(ctrTotal) << "}" << '\n';
            out << --indent << "}" << '\n';
            out << --indent << "}" << comma << '\n';
        };
        out << --indent << "}" << '\n';
        out << --indent << "}" << '\n';
        out << --indent << "};" << '\n';
    };

    void TCatboostModelToCppConverter::WriteModelCatFeatures(const TFullModel& model) {
        CB_ENSURE(model.ObliviousTrees.ApproxDimension == 1, "MultiClassification model export to CPP is not supported.");

        WriteCTRStructs();
        Out << '\n';

        TIndent indent(0);
        TSequenceCommaSeparator comma;
        Out << "/* Model data */" << '\n';

        int binaryFeatureCount = model.ObliviousTrees.GetEffectiveBinaryFeaturesBucketsCount();

        Out << indent++ << "static const struct CatboostModel {" << '\n';
        Out << indent << "CatboostModel() {};" << '\n';
        Out << indent << "unsigned int FloatFeatureCount = " << model.ObliviousTrees.FloatFeatures.size() << ";" << '\n';
        Out << indent << "unsigned int CatFeatureCount = " << model.ObliviousTrees.CatFeatures.size() << ";" << '\n';
        Out << indent << "unsigned int BinaryFeatureCount = " << binaryFeatureCount << ";" << '\n';
        Out << indent << "unsigned int TreeCount = " << model.ObliviousTrees.TreeSizes.size() << ";" << '\n';

        Out << indent++ << "std::vector<std::vector<float>> FloatFeatureBorders = {" << '\n';
        comma.ResetCount(model.ObliviousTrees.FloatFeatures.size());
        for (const auto& floatFeature : model.ObliviousTrees.FloatFeatures) {
            Out << indent << "{"
                << OutputArrayInitializer([&floatFeature](size_t i) { return FloatToString(floatFeature.Borders[i], PREC_NDIGITS, 8); }, floatFeature.Borders.size())
                << "}" << comma << '\n';
        }
        Out << --indent << "};" << '\n';

        Out << indent << "std::vector<unsigned int> TreeDepth = {" << OutputArrayInitializer(model.ObliviousTrees.TreeSizes) << "};" << '\n';
        Out << indent << "std::vector<unsigned int> TreeSplits = {" << OutputArrayInitializer(model.ObliviousTrees.TreeSplits) << "};" << '\n';

        const auto& bins = model.ObliviousTrees.GetRepackedBins();
        Out << indent << "std::vector<unsigned char> TreeSplitIdxs = {" << OutputArrayInitializer([&bins](size_t i) { return (int)bins[i].SplitIdx; }, bins.size()) << "};" << '\n';
        Out << indent << "std::vector<unsigned short> TreeSplitFeatureIndex = {" << OutputArrayInitializer([&bins](size_t i) { return (int)bins[i].FeatureIndex; }, bins.size()) << "};" << '\n';
        Out << indent << "std::vector<unsigned char> TreeSplitXorMask = {" << OutputArrayInitializer([&bins](size_t i) { return (int)bins[i].XorMask; }, bins.size()) << "};" << '\n';

        Out << indent << "std::vector<unsigned int> CatFeaturesIndex = {"
            << OutputArrayInitializer([&model](size_t i) { return model.ObliviousTrees.CatFeatures[i].FeatureIndex; }, model.ObliviousTrees.CatFeatures.size()) << "};" << '\n';

        Out << indent << "std::vector<unsigned int> OneHotCatFeatureIndex = {"
            << OutputArrayInitializer([&model](size_t i) { return model.ObliviousTrees.OneHotFeatures[i].CatFeatureIndex; }, model.ObliviousTrees.OneHotFeatures.size())
            << "};" << '\n';

        Out << indent++ << "std::vector<std::vector<int>> OneHotHashValues = {" << '\n';
        comma.ResetCount(model.ObliviousTrees.OneHotFeatures.size());
        for (const auto& oneHotFeature : model.ObliviousTrees.OneHotFeatures) {
            Out << indent << "{"
                << OutputArrayInitializer([&oneHotFeature](size_t i) { return oneHotFeature.Values[i]; }, oneHotFeature.Values.size())
                << "}" << comma << '\n';
        }
        Out << --indent << "};" << '\n';

        Out << indent++ << "std::vector<std::vector<float>> CtrFeatureBorders = {" << '\n';
        comma.ResetCount(model.ObliviousTrees.CtrFeatures.size());
        for (const auto& ctrFeature : model.ObliviousTrees.CtrFeatures) {
            Out << indent << "{"
                << OutputArrayInitializer([&ctrFeature](size_t i) { return FloatToString(ctrFeature.Borders[i], PREC_NDIGITS, 8) + "f"; }, ctrFeature.Borders.size())
                << "}" << comma << '\n';
        }
        Out << --indent << "};" << '\n';

        Out << '\n';
        Out << indent << "/* Aggregated array of leaf values for trees. Each tree is represented by a separate line: */" << '\n';
        Out << indent << "double LeafValues[" << model.ObliviousTrees.LeafValues.size() << "] = {" << OutputLeafValues(model, indent);
        Out << indent << "};" << '\n';

        WriteModelCTRs(Out, model, indent);

        Out << "} CatboostModelStatic;" << '\n';
        Out << '\n';
    }

    void TCatboostModelToCppConverter::WriteApplicatorCatFeatures() {
        Out << NResource::Find("catboost_model_export_cpp_ctr_calcer");
        Out << '\n';
        Out << NResource::Find("catboost_model_export_cpp_model_applicator");
    }
}
