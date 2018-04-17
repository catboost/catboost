#include "cpp_exporter.h"

#include "export_helpers.h"

#include <util/string/builder.h>
#include <util/string/cast.h>
#include <util/stream/input.h>
#include <util/stream/file.h>

namespace NCatboost {
    using namespace NCatboostModelExportHelpers;

    void TCatboostModelToCppConverter::WriteApplicator() {
        Out << "/* Model applicator */" << Endl;
        Out << "double ApplyCatboostModel(" << Endl;
        Out << "    const std::vector<float>& features" << Endl;
        Out << ") {" << Endl;
        Out << "    const struct CatboostModel& model = CatboostModelStatic;" << Endl;
        Out << Endl;
        Out << "    /* Binarise features */" << Endl;
        Out << "    std::vector<unsigned char> binaryFeatures(model.BinaryFeatureCount);" << Endl;
        Out << "    unsigned int binFeatureIndex = 0;" << Endl;
        Out << "    for (unsigned int i = 0; i < model.FloatFeatureCount; ++i) {" << Endl;
        Out << "        for(unsigned int j = 0; j < model.BorderCounts[i]; ++j) {" << Endl;
        Out << "            binaryFeatures[binFeatureIndex] = (unsigned char)(features[i] > model.Borders[binFeatureIndex]);" << Endl;
        Out << "            ++binFeatureIndex;" << Endl;
        Out << "        }" << Endl;
        Out << "    }" << Endl;
        Out << "" << Endl;
        Out << "    /* Extract and sum values from trees */" << Endl;
        Out << "    double result = 0.0;" << Endl;
        Out << "    const unsigned int* treeSplitsPtr = model.TreeSplits;" << Endl;
        Out << "    const double* leafValuesForCurrentTreePtr = model.LeafValues;" << Endl;
        Out << "    for (unsigned int treeId = 0; treeId < model.TreeCount; ++treeId) {" << Endl;
        Out << "        const unsigned int currentTreeDepth = model.TreeDepth[treeId];" << Endl;
        Out << "        unsigned int index = 0;" << Endl;
        Out << "        for (unsigned int depth = 0; depth < currentTreeDepth; ++depth) {" << Endl;
        Out << "            index |= (binaryFeatures[treeSplitsPtr[depth]] << depth);" << Endl;
        Out << "        }" << Endl;
        Out << "        result += leafValuesForCurrentTreePtr[index];" << Endl;
        Out << "        treeSplitsPtr += currentTreeDepth;" << Endl;
        Out << "        leafValuesForCurrentTreePtr += (1 << currentTreeDepth);" << Endl;
        Out << "    }" << Endl;
        Out << "    return result;" << Endl;
        Out << "}" << Endl;
    }


    void TCatboostModelToCppConverter::WriteModel(const TFullModel& model) {
        CB_ENSURE(!model.HasCategoricalFeatures(), "Export of model with categorical features to CPP is not yet supported.");
        CB_ENSURE(model.ObliviousTrees.ApproxDimension == 1, "Export of MultiClassification model to CPP is not supported.");
        Out << "/* Model data */" << Endl;

        int binaryFeatureCount = GetBinaryFeatureCount(model);

        Out << "static const struct CatboostModel {" << Endl;
        Out << "    unsigned int TreeCount = " << model.ObliviousTrees.TreeSizes.size() << ";" << Endl;
        Out << "    unsigned int FloatFeatureCount = " << model.ObliviousTrees.FloatFeatures.size() << ";" << Endl;
        Out << "    unsigned int BinaryFeatureCount = " << binaryFeatureCount << ";" << Endl;

        Out << "    unsigned int BorderCounts[" << model.ObliviousTrees.FloatFeatures.size() << "] = {" << OutputBorderCounts(model) << "};" << Endl;

        Out << "    float Borders[" << binaryFeatureCount << "] = {" << OutputBorders(model) << "};" << Endl;

        Out << "    unsigned int TreeDepth[" << model.ObliviousTrees.TreeSizes.size() << "] = {" << OutputArrayInitializer(model.ObliviousTrees.TreeSizes) << "};" << Endl;
        Out << "    unsigned int TreeSplits[" << model.ObliviousTrees.TreeSplits.size() << "] = {" << OutputArrayInitializer(model.ObliviousTrees.TreeSplits) << "};" << Endl;

        int leafValuesCount = 0;
        for (const auto& treeLeaf : model.ObliviousTrees.LeafValues) {
            leafValuesCount += treeLeaf.size();
        }
        Out << Endl;
        Out << "    /* Aggregated array of leaf values for trees. Each tree is represented by a separate line: */" << Endl;
        Out << "    double LeafValues[" << leafValuesCount << "] = {" << OutputLeafValues(model, TIndent(1)) << "};" << Endl;
        Out << "} CatboostModelStatic;" << Endl;
        Out << Endl;
    }

    void TCatboostModelToCppConverter::WriteHeader() {
        Out << "#include <vector>" << Endl;
        Out << Endl;
    }
}
