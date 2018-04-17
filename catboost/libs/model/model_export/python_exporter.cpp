#include "python_exporter.h"

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

    void TCatboostModelToPythonConverter::WriteApplicator() {
        Out << "### Model applicator" << Endl;
        Out << "def apply_catboost_model(float_features):" << Endl;
        Out << "    model = CatboostModel" << Endl;
        Out << Endl;
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
        CB_ENSURE(model.ObliviousTrees.ApproxDimension == 1, "Export of MultiClassification model to Python is not supported.");
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
        Out << "        leaf_values = [" << OutputLeafValues(model, TIndent(1)) << "]" << Endl;
        Out << Endl;
    }


    /*
     * Full model code with complete support of cat features
     */

    void TCatboostModelToPythonConverter::WriteHeaderCatFeatures() {
        Out << "from cityhash import CityHash64  # Available at https://github.com/Amper/cityhash" << Endl;
        Out << Endl;
    };


    void TCatboostModelToPythonConverter::WriteCTRStructs() {
        Out << NResource::Find("catboost_model_export_python_ctr_structs") << Endl;
    };

    struct TCompressedModelCtr {
        const TFeatureCombination* Projection;
        TVector<const TModelCtr*> ModelCtrs;
    };

    static void WriteModelCTRs(IOutputStream& out, const TFullModel& model, TIndent& indent) {
        const TVector<TModelCtr>& neededCtrs = model.ObliviousTrees.GetUsedModelCtrs();
        if (neededCtrs.empty()) {
            return;
        }

        TSequenceCommaSeparator comma;
        out << indent++ << "model_ctrs = catboost_model_ctrs_container(" << Endl;

        const TStaticCtrProvider* ctrProvider = dynamic_cast<TStaticCtrProvider*>(model.CtrProvider.Get());
        Y_VERIFY(ctrProvider, "Unsupported CTR provider");

        TVector<TCompressedModelCtr> compressedModelCtrs;
        compressedModelCtrs.emplace_back(TCompressedModelCtr{&neededCtrs[0].Base.Projection, {&neededCtrs[0]}});
        for (size_t i = 1; i < neededCtrs.size(); ++i) {
            Y_ASSERT(neededCtrs[i - 1] < neededCtrs[i]); // needed ctrs should be sorted
            if (*(compressedModelCtrs.back().Projection) != neededCtrs[i].Base.Projection) {
                compressedModelCtrs.emplace_back(TCompressedModelCtr{&neededCtrs[i].Base.Projection, {}});
            }
            compressedModelCtrs.back().ModelCtrs.push_back(&neededCtrs[i]);
        }

        out << indent << "used_model_ctrs_count = " << model.ObliviousTrees.GetUsedModelCtrs().size() << "," << Endl;
        out << indent++ << "compressed_model_ctrs = [" << Endl;

        comma.ResetCount(compressedModelCtrs.size());
        for (const auto& compressedCtr : compressedModelCtrs) {
            TSequenceCommaSeparator commaInner;

            out << indent++ << "catboost_compressed_model_ctr(" << Endl;

            out << indent++ << "projection = catboost_projection(" << Endl;

            const TFeatureCombination& proj = *compressedCtr.Projection;
            TVector<int> transposedCatFeatureIndexes;
            out << indent << "transposed_cat_feature_indexes = [";
            TSequenceCommaSeparator commaInnerWithSpace(proj.CatFeatures.size(), AddSpaceAfterComma);
            for (const auto feature : proj.CatFeatures) {
                out << ctrProvider->GetCatFeatureIndex().at(feature) << commaInnerWithSpace;
            }
            out << "]," << Endl;
            out << indent++ << "binarized_indexes = [";
            commaInner.ResetCount(proj.BinFeatures.size() + proj.OneHotFeatures.size());
            for (const auto& feature : proj.BinFeatures) {
                const TBinFeatureIndexValue& featureValue = ctrProvider->GetFloatFeatureIndexes().at(feature);
                out << Endl << indent << "catboost_bin_feature_index_value(";
                out << "bin_index = " << featureValue.BinIndex << ", ";
                out << "check_value_equal = " << featureValue.CheckValueEqual << ", ";
                out << "value = " << (int)featureValue.Value;
                out << ")" << commaInner;
            }
            for (const auto& feature : proj.OneHotFeatures) {
                const TBinFeatureIndexValue& featureValue = ctrProvider->GetOneHotFeatureIndexes().at(feature);
                out << Endl << indent << "catboost_bin_feature_index_value(";
                out << "bin_index = " << featureValue.BinIndex << ", ";
                out << "check_value_equal = " << featureValue.CheckValueEqual << ", ";
                out << "value = " << (int)featureValue.Value;
                out << ")" << commaInner;
            }
            --indent;
            if (proj.BinFeatures.size() > 0 || proj.OneHotFeatures.size() > 0) {
                out << Endl << indent;
            }
            out << "]" << Endl;

            out << --indent << ")," << Endl;
            out << indent++ << "model_ctrs = [" << Endl;
            commaInner.ResetCount(compressedCtr.ModelCtrs.size());
            for (const auto& ctr : compressedCtr.ModelCtrs) {
                out << indent << "catboost_model_ctr(";
                out << "base_hash = " << ctr->Base.GetHash() << ", ";
                out << "base_ctr_type = \"" << ctr->Base.CtrType << "\", ";
                out << "target_border_idx = " << ctr->TargetBorderIdx << ", ";
                out << "prior_num = " << ctr->PriorNum << ", ";
                out << "prior_denom = " << ctr->PriorDenom << ", ";
                out << "shift = " << ctr->Shift << ", ";
                out << "scale = " << ctr->Scale;
                out << ")" << commaInner << Endl;
            }
            out << --indent << "]" << Endl;
            out << --indent << ")" << comma << Endl;
        }
        out << --indent << "]," << Endl;
        out << indent++ << "ctr_data = catboost_ctr_data(" << Endl;
        out << indent++ << "learn_ctrs = {" << Endl;
        comma.ResetCount(ctrProvider->CtrData.LearnCtrs.size());
        for (const auto& learnCtr : ctrProvider->CtrData.LearnCtrs) {
            TSequenceCommaSeparator commaInner(AddSpaceAfterComma);
            out << indent << learnCtr.first.GetHash() << " :" << Endl;
            out << indent++ << "catboost_ctr_value_table(" << Endl;
            out << indent << "index_hash_viewer = {";
            const TConstArrayRef<TBucket> HashViewerBuckets = learnCtr.second.GetIndexHashViewer().GetBuckets();
            commaInner.ResetCount(HashViewerBuckets.size());
            for (const auto& bucket : HashViewerBuckets) {
                out << bucket.Hash << " : " << bucket.IndexValue << commaInner;
            }
            out << "}," << Endl;
            out << indent << "target_classes_count = " << learnCtr.second.TargetClassesCount << "," << Endl;
            out << indent << "counter_denominator = " << learnCtr.second.CounterDenominator << "," << Endl;
            const TConstArrayRef<TCtrMeanHistory> ctrMeanHistories = learnCtr.second.GetTypedArrayRefForBlobData<TCtrMeanHistory>();
            out << indent << "ctr_mean_history = [";
            commaInner.ResetCount(ctrMeanHistories.size());
            for (const auto& ctrMean : ctrMeanHistories) {
                out << "catboost_ctr_mean_history(";
                out << "sum = " << ctrMean.Sum << ", ";
                out << "count = " << ctrMean.Count;
                out << ")" << commaInner;
            }
            out << "]," << Endl;
            const TConstArrayRef<int> ctrTotal = learnCtr.second.GetTypedArrayRefForBlobData<int>();
            out << indent << "ctr_total = [" << OutputArrayInitializer(ctrTotal) << "]" << Endl;
            out << --indent << ")" << comma << Endl;
        };
        out << --indent << "}" << Endl;
        out << --indent << ")" << Endl;
        out << --indent << ")" << Endl;
    };


    void TCatboostModelToPythonConverter::WriteModelCatFeatures(const TFullModel& model) {
        CB_ENSURE(model.ObliviousTrees.ApproxDimension == 1, "Export of MultiClassification model to Python is not supported.");


        if (!model.ObliviousTrees.GetUsedModelCtrs().empty()) {
            WriteCTRStructs();
        }

        TIndent indent(0);
        TSequenceCommaSeparator comma;
        Out << indent << "##  Model data" << Endl;

        Out << indent++ << "class catboost_model(object):" << Endl;
        Out << indent << "float_features_count = " << model.ObliviousTrees.FloatFeatures.size() << Endl;
        Out << indent << "cat_features_count = " << model.ObliviousTrees.CatFeatures.size() << Endl;
        Out << indent << "binary_feature_count = " << model.ObliviousTrees.GetEffectiveBinaryFeaturesBucketsCount() << Endl;
        Out << indent << "tree_count = " << model.ObliviousTrees.TreeSizes.size() << Endl;

        Out << indent++ << "float_feature_borders = [" << Endl;
        comma.ResetCount(model.ObliviousTrees.FloatFeatures.size());
        for (const auto& floatFeature : model.ObliviousTrees.FloatFeatures) {
            Out << indent << "["
                << OutputArrayInitializer([&floatFeature](size_t i) { return FloatToString(floatFeature.Borders[i], PREC_NDIGITS, 8); }, floatFeature.Borders.size())
                << "]" << comma << Endl;
        }
        Out << --indent << "]" << Endl;

        Out << indent << "tree_depth = [" << OutputArrayInitializer(model.ObliviousTrees.TreeSizes) << "]" << Endl;

        const TVector<TRepackedBin>& bins = model.ObliviousTrees.GetRepackedBins();
        Out << indent << "tree_split_border = [" << OutputArrayInitializer([&bins](size_t i) { return (int)bins[i].SplitIdx; }, bins.size()) << "]" << Endl;
        Out << indent << "tree_split_feature_index = [" << OutputArrayInitializer([&bins](size_t i) { return (int)bins[i].FeatureIndex; }, bins.size()) << "]" << Endl;
        Out << indent << "tree_split_xor_mask = [" << OutputArrayInitializer([&bins](size_t i) { return (int)bins[i].XorMask; }, bins.size()) << "]" << Endl;

        Out << indent << "cat_features_index = ["
            << OutputArrayInitializer([&model](size_t i) { return model.ObliviousTrees.CatFeatures[i].FeatureIndex; }, model.ObliviousTrees.CatFeatures.size()) << "]" << Endl;

        Out << indent << "one_hot_cat_feature_index = ["
            << OutputArrayInitializer([&model](size_t i) { return model.ObliviousTrees.OneHotFeatures[i].CatFeatureIndex; }, model.ObliviousTrees.OneHotFeatures.size())
            << "]" << Endl;

        Out << indent++ << "one_hot_hash_values = [" << Endl;
        comma.ResetCount(model.ObliviousTrees.OneHotFeatures.size());
        for (const auto& oneHotFeature : model.ObliviousTrees.OneHotFeatures) {
            Out << indent << "["
                << OutputArrayInitializer([&oneHotFeature](size_t i) { return oneHotFeature.Values[i]; }, oneHotFeature.Values.size())
                << "]" << comma << Endl;
        }
        Out << --indent << "]" << Endl;

        Out << indent++ << "ctr_feature_borders = [" << Endl;
        comma.ResetCount(model.ObliviousTrees.CtrFeatures.size());
        for (const auto& ctrFeature : model.ObliviousTrees.CtrFeatures) {
            Out << indent << "["
                << OutputArrayInitializer([&ctrFeature](size_t i) { return FloatToString(ctrFeature.Borders[i], PREC_NDIGITS, 16); }, ctrFeature.Borders.size())
                << "]" << comma << Endl;
        }
        Out << --indent << "]" << Endl;

        int leafValuesCount = 0;
        for (const auto& treeLeaf : model.ObliviousTrees.LeafValues) {
            leafValuesCount += treeLeaf.size();
        }
        Out << Endl;
        Out << indent << "## Aggregated array of leaf values for trees. Each tree is represented by a separate line:" << Endl;
        Out << indent << "leaf_values = [" << OutputLeafValues(model, indent) << indent << "]" << Endl;

        if (!model.ObliviousTrees.GetUsedModelCtrs().empty()) {
            WriteModelCTRs(Out, model, indent);
            Out << Endl;
            Out << NResource::Find("catboost_model_export_python_ctr_calcer") << Endl;
        }
    };

    void TCatboostModelToPythonConverter::WriteApplicatorCatFeatures() {
        Out << NResource::Find("catboost_model_export_python_model_applicator") << Endl;
    };

}
