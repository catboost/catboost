#include "python_exporter.h"

#include "export_helpers.h"

#include <util/string/builder.h>
#include <util/string/cast.h>
#include <util/stream/input.h>
#include <util/stream/file.h>

namespace NCatboost {
    using namespace NCatboostModelExportHelpers;

    void TCatboostModelToPythonConverter::WriteApplicator() {
        Out << "### Model applicator" << Endl;
        Out << "def apply_catboost_model(float_features):" << Endl;
        Out << "    model = CatboostModel" << Endl;
        Out << Endl;
        Out << "    # Binarise features" << Endl;
        Out << "    binary_features = [0] * model.binary_feature_count" << Endl;
        Out << "    binary_feature_index = 0" << Endl;

        Out << "    for i in range(model.float_feature_count):" << Endl;
        Out << "        for j in range(model.border_counts[i]):" << Endl;
        Out << "            binary_features[binary_feature_index] = 1 if (float_features[i] > model.borders[binary_feature_index]) else 0" << Endl;
        Out << "            binary_feature_index += 1" << Endl;
        Out << "" << Endl;
        Out << "    # Extract and sum values from trees" << Endl;
        Out << "    result = 0.0" << Endl;
        Out << "    tree_splits_index = 0" << Endl;
        Out << "    current_tree_leaf_values_index = 0" << Endl;
        Out << "    for tree_id in range(model.tree_count):" << Endl;
        Out << "        current_tree_depth = model.tree_depth[tree_id]" << Endl;
        Out << "        index = 0" << Endl;
        Out << "        for depth in range(current_tree_depth):" << Endl;
        Out << "            index |= (binary_features[model.tree_splits[tree_splits_index + depth]] << depth)" << Endl;
        Out << "        result += model.leaf_values[current_tree_leaf_values_index + index]" << Endl;
        Out << "        tree_splits_index += current_tree_depth" << Endl;
        Out << "        current_tree_leaf_values_index += (1 << current_tree_depth)" << Endl;
        Out << "    return result" << Endl;
    }

    void TCatboostModelToPythonConverter::WriteModel(const TFullModel& model) {
        CB_ENSURE(!model.HasCategoricalFeatures(), "Export of model with categorical features to Python is not yet supported.");
        CB_ENSURE(model.ObliviousTrees.ApproxDimension == 1, "Export of MultiClassification mdoel to Python is not supported.");
        Out << "### Model data" << Endl;

        Out << "class CatboostModel(object):" << Endl;
        Out << "        tree_count = " << model.ObliviousTrees.TreeSizes.size() << Endl;
        Out << "        float_feature_count = " << model.ObliviousTrees.FloatFeatures.size() << Endl;
        Out << "        binary_feature_count = " << GetBinaryFeatureCount(model) << Endl;

        Out << "        border_counts = [" << OutputBorderCounts(model) << "]" << Endl;

        Out << "        borders = [" << OutputBorders(model) << "]" << Endl;

        Out << "        tree_depth  = [" << OutputArrayInitializer(model.ObliviousTrees.TreeSizes) << "]" << Endl;
        Out << "        tree_splits = [" << OutputArrayInitializer(model.ObliviousTrees.TreeSplits) << "]" << Endl;

        Out << Endl;
        Out << "        # Aggregated array of leaf values for trees. Each tree is represented by a separate line:" << Endl;
        Out << "        leaf_values = [" << OutputLeafValues(model) << "]" << Endl;
        Out << Endl;
    }

}
