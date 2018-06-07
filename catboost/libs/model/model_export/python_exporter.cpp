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
        Out << "### Model applicator" << '\n';
        Out << "def apply_catboost_model(float_features):" << '\n';
        Out << "    model = CatboostModel" << '\n';
        Out << '\n';
        Out << "    binary_feature_index = 0" << '\n';
        Out << "    binary_features = [0] * model.binary_feature_count" << '\n';
        Out << "    for i in range(model.float_feature_count):" << '\n';
        Out << "        for j in range(model.border_counts[i]):" << '\n';
        Out << "            binary_features[binary_feature_index] = 1 if (float_features[i] > model.borders[binary_feature_index]) else 0" << '\n';
        Out << "            binary_feature_index += 1" << '\n';
        Out << '\n';
        Out << "    # Extract and sum values from trees" << '\n';
        Out << "    result = 0.0" << '\n';
        Out << "    tree_splits_index = 0" << '\n';
        Out << "    current_tree_leaf_values_index = 0" << '\n';
        Out << "    for tree_id in range(model.tree_count):" << '\n';
        Out << "        current_tree_depth = model.tree_depth[tree_id]" << '\n';
        Out << "        index = 0" << '\n';
        Out << "        for depth in range(current_tree_depth):" << '\n';
        Out << "            index |= (binary_features[model.tree_splits[tree_splits_index + depth]] << depth)" << '\n';
        Out << "        result += model.leaf_values[current_tree_leaf_values_index + index]" << '\n';
        Out << "        tree_splits_index += current_tree_depth" << '\n';
        Out << "        current_tree_leaf_values_index += (1 << current_tree_depth)" << '\n';
        Out << "    return result" << '\n';
    }

    void TCatboostModelToPythonConverter::WriteModel(const TFullModel& model) {
        CB_ENSURE(!model.HasCategoricalFeatures(), "Export of model with categorical features to Python is not yet supported.");
        CB_ENSURE(model.ObliviousTrees.ApproxDimension == 1, "Export of MultiClassification model to Python is not supported.");
        Out << "### Model data" << '\n';

        Out << "class CatboostModel(object):" << '\n';
        Out << "    tree_count = " << model.ObliviousTrees.TreeSizes.size() << '\n';
        Out << "    float_feature_count = " << model.ObliviousTrees.FloatFeatures.size() << '\n';
        Out << "    binary_feature_count = " << GetBinaryFeatureCount(model) << '\n';

        Out << "    border_counts = [" << OutputBorderCounts(model) << "]" << '\n';

        Out << "    borders = [" << OutputBorders(model) << "]" << '\n';

        Out << "    tree_depth  = [" << OutputArrayInitializer(model.ObliviousTrees.TreeSizes) << "]" << '\n';
        Out << "    tree_splits = [" << OutputArrayInitializer(model.ObliviousTrees.TreeSplits) << "]" << '\n';

        Out << '\n';
        Out << "    # Aggregated array of leaf values for trees. Each tree is represented by a separate line:" << '\n';
        Out << "    leaf_values = [" << OutputLeafValues(model, TIndent(1)) << "]" << '\n';
        Out << '\n';
    }


    /*
     * Full model code with complete support of cat features
     */

    void TCatboostModelToPythonConverter::WriteHeaderCatFeatures() {
        Out << "try:" << '\n';
        Out << "    from cityhash import CityHash64  # Available at https://github.com/Amper/cityhash #4f02fe0ba78d4a6d1735950a9c25809b11786a56" << '\n';
        Out << "except ImportError:" << '\n';
        Out << "    from cityhash import hash64 as CityHash64  # ${catboost_repo_root}/library/python/cityhash" << '\n';
        Out << '\n' << '\n';
    };


    void TCatboostModelToPythonConverter::WriteCTRStructs() {
        Out << NResource::Find("catboost_model_export_python_ctr_structs");
    };

    static void WriteModelCTRs(IOutputStream& out, const TFullModel& model, TIndent& indent) {
        const TVector<TModelCtr>& neededCtrs = model.ObliviousTrees.GetUsedModelCtrs();
        if (neededCtrs.empty()) {
            return;
        }

        TSequenceCommaSeparator comma;
        out << indent++ << "model_ctrs = catboost_model_ctrs_container(" << '\n';

        const TStaticCtrProvider* ctrProvider = dynamic_cast<TStaticCtrProvider*>(model.CtrProvider.Get());
        Y_VERIFY(ctrProvider, "Unsupported CTR provider");

        TVector<TCompressedModelCtr> compressedModelCtrs = CompressModelCtrs(neededCtrs);

        out << indent << "used_model_ctrs_count = " << model.ObliviousTrees.GetUsedModelCtrs().size() << "," << '\n';
        out << indent++ << "compressed_model_ctrs = [" << '\n';

        comma.ResetCount(compressedModelCtrs.size());
        for (const auto& compressedCtr : compressedModelCtrs) {
            TSequenceCommaSeparator commaInner;

            out << indent++ << "catboost_compressed_model_ctr(" << '\n';

            out << indent++ << "projection = catboost_projection(" << '\n';

            const TFeatureCombination& proj = *compressedCtr.Projection;
            TVector<int> transposedCatFeatureIndexes;
            out << indent << "transposed_cat_feature_indexes = [";
            TSequenceCommaSeparator commaInnerWithSpace(proj.CatFeatures.size(), AddSpaceAfterComma);
            for (const auto feature : proj.CatFeatures) {
                out << ctrProvider->GetCatFeatureIndex().at(feature) << commaInnerWithSpace;
            }
            out << "]," << '\n';
            out << indent++ << "binarized_indexes = [";
            commaInner.ResetCount(proj.BinFeatures.size() + proj.OneHotFeatures.size());
            for (const auto& feature : proj.BinFeatures) {
                const TBinFeatureIndexValue& featureValue = ctrProvider->GetFloatFeatureIndexes().at(feature);
                out << '\n' << indent << "catboost_bin_feature_index_value(";
                out << "bin_index = " << featureValue.BinIndex << ", ";
                out << "check_value_equal = " << featureValue.CheckValueEqual << ", ";
                out << "value = " << (int)featureValue.Value;
                out << ")" << commaInner;
            }
            for (const auto& feature : proj.OneHotFeatures) {
                const TBinFeatureIndexValue& featureValue = ctrProvider->GetOneHotFeatureIndexes().at(feature);
                out << '\n' << indent << "catboost_bin_feature_index_value(";
                out << "bin_index = " << featureValue.BinIndex << ", ";
                out << "check_value_equal = " << featureValue.CheckValueEqual << ", ";
                out << "value = " << (int)featureValue.Value;
                out << ")" << commaInner;
            }
            --indent;
            if (proj.BinFeatures.size() > 0 || proj.OneHotFeatures.size() > 0) {
                out << '\n' << indent;
            }
            out << "]" << '\n';

            out << --indent << ")," << '\n';
            out << indent++ << "model_ctrs = [" << '\n';
            commaInner.ResetCount(compressedCtr.ModelCtrs.size());
            for (const auto& ctr : compressedCtr.ModelCtrs) {
                TSequenceCommaSeparator commaLocal(7, AddSpaceAfterComma);
                out << indent << "catboost_model_ctr(";
                out << "base_hash = " << ctr->Base.GetHash() << commaLocal;
                out << "base_ctr_type = \"" << ctr->Base.CtrType << "\"" << commaLocal ;
                out << "target_border_idx = " << ctr->TargetBorderIdx << commaLocal;
                out << "prior_num = " << ctr->PriorNum << commaLocal;
                out << "prior_denom = " << ctr->PriorDenom << commaLocal;
                out << "shift = " << ctr->Shift << commaLocal;
                out << "scale = " << ctr->Scale << commaLocal;
                out << ")" << commaInner << '\n';
            }
            out << --indent << "]" << '\n';
            out << --indent << ")" << comma << '\n';
        }
        out << --indent << "]," << '\n';
        out << indent++ << "ctr_data = catboost_ctr_data(" << '\n';
        out << indent++ << "learn_ctrs = {" << '\n';
        comma.ResetCount(ctrProvider->CtrData.LearnCtrs.size());
        for (const auto& learnCtr : ctrProvider->CtrData.LearnCtrs) {
            TSequenceCommaSeparator commaInner(AddSpaceAfterComma);
            out << indent << learnCtr.first.GetHash() << " :" << '\n';
            out << indent++ << "catboost_ctr_value_table(" << '\n';
            out << indent << "index_hash_viewer = {";
            const TConstArrayRef<TBucket> HashViewerBuckets = learnCtr.second.GetIndexHashViewer().GetBuckets();
            commaInner.ResetCount(HashViewerBuckets.size());
            for (const auto& bucket : HashViewerBuckets) {
                out << bucket.Hash << " : " << bucket.IndexValue << commaInner;
            }
            out << "}," << '\n';
            out << indent << "target_classes_count = " << learnCtr.second.TargetClassesCount << "," << '\n';
            out << indent << "counter_denominator = " << learnCtr.second.CounterDenominator << "," << '\n';
            const TConstArrayRef<TCtrMeanHistory> ctrMeanHistories = learnCtr.second.GetTypedArrayRefForBlobData<TCtrMeanHistory>();
            out << indent << "ctr_mean_history = [";
            commaInner.ResetCount(ctrMeanHistories.size());
            for (const auto& ctrMean : ctrMeanHistories) {
                out << "catboost_ctr_mean_history(";
                out << "sum = " << ctrMean.Sum << ", ";
                out << "count = " << ctrMean.Count;
                out << ")" << commaInner;
            }
            out << "]," << '\n';
            const TConstArrayRef<int> ctrTotal = learnCtr.second.GetTypedArrayRefForBlobData<int>();
            out << indent << "ctr_total = [" << OutputArrayInitializer(ctrTotal) << "]" << '\n';
            out << --indent << ")" << comma << '\n';
        };
        out << --indent << "}" << '\n';
        out << --indent << ")" << '\n';
        out << --indent << ")" << '\n';
    };


    void TCatboostModelToPythonConverter::WriteModelCatFeatures(const TFullModel& model) {
        CB_ENSURE(model.ObliviousTrees.ApproxDimension == 1, "Export of MultiClassification model to Python is not supported.");

        if (!model.ObliviousTrees.GetUsedModelCtrs().empty()) {
            WriteCTRStructs();
        }

        TIndent indent(0);
        TSequenceCommaSeparator comma;
        Out << indent << "###  Model data" << '\n';

        Out << indent++ << "class catboost_model(object):" << '\n';
        Out << indent << "float_feature_count = " << model.ObliviousTrees.FloatFeatures.size() << '\n';
        Out << indent << "cat_feature_count = " << model.ObliviousTrees.CatFeatures.size() << '\n';
        Out << indent << "binary_feature_count = " << model.ObliviousTrees.GetEffectiveBinaryFeaturesBucketsCount() << '\n';
        Out << indent << "tree_count = " << model.ObliviousTrees.TreeSizes.size() << '\n';

        Out << indent++ << "float_feature_borders = [" << '\n';
        comma.ResetCount(model.ObliviousTrees.FloatFeatures.size());
        for (const auto& floatFeature : model.ObliviousTrees.FloatFeatures) {
            Out << indent << "["
                << OutputArrayInitializer([&floatFeature](size_t i) { return FloatToString(floatFeature.Borders[i], PREC_NDIGITS, 8); }, floatFeature.Borders.size())
                << "]" << comma << '\n';
        }
        Out << --indent << "]" << '\n';

        Out << indent << "tree_depth = [" << OutputArrayInitializer(model.ObliviousTrees.TreeSizes) << "]" << '\n';

        const TVector<TRepackedBin>& bins = model.ObliviousTrees.GetRepackedBins();
        Out << indent << "tree_split_border = [" << OutputArrayInitializer([&bins](size_t i) { return (int)bins[i].SplitIdx; }, bins.size()) << "]" << '\n';
        Out << indent << "tree_split_feature_index = [" << OutputArrayInitializer([&bins](size_t i) { return (int)bins[i].FeatureIndex; }, bins.size()) << "]" << '\n';
        Out << indent << "tree_split_xor_mask = [" << OutputArrayInitializer([&bins](size_t i) { return (int)bins[i].XorMask; }, bins.size()) << "]" << '\n';

        Out << indent << "cat_features_index = ["
            << OutputArrayInitializer([&model](size_t i) { return model.ObliviousTrees.CatFeatures[i].FeatureIndex; }, model.ObliviousTrees.CatFeatures.size()) << "]" << '\n';

        Out << indent << "one_hot_cat_feature_index = ["
            << OutputArrayInitializer([&model](size_t i) { return model.ObliviousTrees.OneHotFeatures[i].CatFeatureIndex; }, model.ObliviousTrees.OneHotFeatures.size())
            << "]" << '\n';

        Out << indent++ << "one_hot_hash_values = [" << '\n';
        comma.ResetCount(model.ObliviousTrees.OneHotFeatures.size());
        for (const auto& oneHotFeature : model.ObliviousTrees.OneHotFeatures) {
            Out << indent << "["
                << OutputArrayInitializer([&oneHotFeature](size_t i) { return oneHotFeature.Values[i]; }, oneHotFeature.Values.size())
                << "]" << comma << '\n';
        }
        Out << --indent << "]" << '\n';

        Out << indent++ << "ctr_feature_borders = [" << '\n';
        comma.ResetCount(model.ObliviousTrees.CtrFeatures.size());
        for (const auto& ctrFeature : model.ObliviousTrees.CtrFeatures) {
            Out << indent << "["
                << OutputArrayInitializer([&ctrFeature](size_t i) { return FloatToString(ctrFeature.Borders[i], PREC_NDIGITS, 16); }, ctrFeature.Borders.size())
                << "]" << comma << '\n';
        }
        Out << --indent << "]" << '\n';

        int leafValueCount = 0;
        for (const auto& treeSize : model.ObliviousTrees.TreeSizes) {
            leafValueCount += treeSize * model.ObliviousTrees.ApproxDimension;
        }
        Out << '\n';
        Out << indent << "## Aggregated array of leaf values for trees. Each tree is represented by a separate line:" << '\n';
        Out << indent << "leaf_values = [" << OutputLeafValues(model, indent) << indent << "]" << '\n';

        if (!model.ObliviousTrees.GetUsedModelCtrs().empty()) {
            WriteModelCTRs(Out, model, indent);
            Out << '\n' << '\n';
            Out << NResource::Find("catboost_model_export_python_ctr_calcer") << '\n';
        }
    };

    void TCatboostModelToPythonConverter::WriteApplicatorCatFeatures() {
        Out << NResource::Find("catboost_model_export_python_model_applicator") << '\n';
    };

}
