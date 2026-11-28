#include <catboost/libs/data/data_provider_builders.h>
#include <catboost/libs/fstr/shap_values.h>
#include <catboost/libs/model/model_build_helper.h>

#include <library/cpp/testing/unittest/registar.h>

using namespace NCB;

namespace {
    constexpr size_t FeatureCount = 3;

    THolder<TNonSymmetricTreeNode> Leaf(double value, double weight, size_t dimension) {
        auto node = MakeHolder<TNonSymmetricTreeNode>();
        TVector<double> values(dimension);
        for (size_t dim = 0; dim < dimension; ++dim) {
            values[dim] = (dim + 1) * value + dim;
        }
        node->Value = std::move(values);
        node->NodeWeight = weight;
        return node;
    }

    THolder<TNonSymmetricTreeNode> Split(
        int feature,
        float border,
        THolder<TNonSymmetricTreeNode> left,
        THolder<TNonSymmetricTreeNode> right
    ) {
        auto node = MakeHolder<TNonSymmetricTreeNode>();
        node->SplitCondition = TModelSplit(TFloatSplit(feature, border));
        node->NodeWeight = *left->NodeWeight + *right->NodeWeight;
        node->Left = std::move(left);
        node->Right = std::move(right);
        return node;
    }

    THolder<TNonSymmetricTreeNode> MixedTree(bool leafOnRight, bool repeatFeature, size_t dimension) {
        auto leaf = Leaf(2.0, 2.0, dimension);
        auto subtree = Split(repeatFeature ? 0 : 1, repeatFeature && leafOnRight ? -0.5f : 1.0f,
                             Leaf(4.0, 3.0, dimension), Leaf(10.0, 5.0, dimension));
        return leafOnRight
            ? Split(0, 0.0, std::move(subtree), std::move(leaf))
            : Split(0, 0.0, std::move(leaf), std::move(subtree));
    }

    // Conditional tree expectation with unknown features marginalized using leaf weights.
    // Enumerating all feature subsets gives an oracle independent of the TreeSHAP recursion.
    double ConditionalValue(
        const TNonSymmetricTreeNode& node,
        const TVector<float>& features,
        size_t knownFeatures,
        size_t dimension
    ) {
        if (!node.IsSplitNode()) {
            return std::get<TVector<double>>(node.Value)[dimension];
        }
        const auto& split = node.SplitCondition->FloatFeature;
        if (knownFeatures & (size_t(1) << split.FloatFeature)) {
            const auto& child = features[split.FloatFeature] > split.Split ? node.Right : node.Left;
            return ConditionalValue(*child, features, knownFeatures, dimension);
        }
        return (ConditionalValue(*node.Left, features, knownFeatures, dimension) * *node.Left->NodeWeight
              + ConditionalValue(*node.Right, features, knownFeatures, dimension) * *node.Right->NodeWeight)
             / *node.NodeWeight;
    }

    TVector<double> ReferenceShapValues(
        const TNonSymmetricTreeNode& tree,
        const TVector<float>& features,
        size_t dimension,
        ECalcTypeShapValues calcType
    ) {
        TVector<double> values(FeatureCount + 1, 0.0);
        values.back() = ConditionalValue(tree, features, 0, dimension);
        if (calcType == ECalcTypeShapValues::Regular) {
            const double subsetWeights[] = {1.0 / 3, 1.0 / 6, 1.0 / 3};
            for (size_t feature = 0; feature < FeatureCount; ++feature) {
                for (size_t subset = 0; subset < (size_t(1) << FeatureCount); ++subset) {
                    if (subset & (size_t(1) << feature)) {
                        continue;
                    }
                    size_t subsetSize = 0;
                    for (size_t other = 0; other < FeatureCount; ++other) {
                        subsetSize += bool(subset & (size_t(1) << other));
                    }
                    values[feature] += subsetWeights[subsetSize] * (
                        ConditionalValue(tree, features, subset | (size_t(1) << feature), dimension)
                        - ConditionalValue(tree, features, subset, dimension));
                }
            }
        } else {
            const auto* node = &tree;
            while (node->IsSplitNode()) {
                const auto& split = node->SplitCondition->FloatFeature;
                const auto* child = features[split.FloatFeature] > split.Split
                    ? node->Right.Get() : node->Left.Get();
                values[split.FloatFeature] += ConditionalValue(*child, features, 0, dimension)
                    - ConditionalValue(*node, features, 0, dimension);
                node = child;
            }
        }
        return values;
    }

    void CheckShapValues(ECalcTypeShapValues calcType, bool leafOnRight, bool repeatFeature, size_t dimension) {
        TVector<TVector<float>> features;
        for (float x0 : {-1.0f, -0.25f, 0.5f, 2.0f}) {
            for (float x1 : {-1.0f, 0.5f, 2.0f}) {
                for (float x2 : {-1.0f, 0.5f, 2.0f}) {
                    features.push_back({x0, x1, x2});
                }
            }
        }
        auto dataset = CreateDataProvider<IRawObjectsOrderDataVisitor>(
            [&](IRawObjectsOrderDataVisitor* visitor) {
                TDataMetaInfo metaInfo;
                metaInfo.FeaturesLayout = MakeIntrusive<TFeaturesLayout>(FeatureCount);
                visitor->Start(false, metaInfo, false, features.size(), EObjectsOrder::Undefined, {});
                visitor->StartNextBlock(features.size());
                for (size_t object = 0; object < features.size(); ++object) {
                    visitor->AddAllFloatFeatures(object, features[object]);
                }
                visitor->Finish();
            });

        TVector<THolder<TNonSymmetricTreeNode>> trees;
        trees.push_back(Leaf(3.0, 10.0, dimension));
        trees.push_back(MixedTree(leafOnRight, repeatFeature, dimension));
        trees.push_back(Split(2, 0.0,
            MixedTree(!leafOnRight, repeatFeature, dimension),
            Split(1, 0.0, Leaf(-2.0, 4.0, dimension), Leaf(7.0, 6.0, dimension))));

        TVector<TVector<TVector<double>>> expected(features.size(),
            TVector<TVector<double>>(dimension, TVector<double>(FeatureCount + 1, 0.0)));
        TNonSymmetricTreeModelBuilder builder(
            {TFloatFeature(false, 0, 0, {-0.5f, 0.0f, 1.0f}),
             TFloatFeature(false, 1, 1, {0.0f, 1.0f}),
             TFloatFeature(false, 2, 2, {0.0f})}, {}, {}, {}, dimension);
        for (auto& tree : trees) {
            for (size_t object = 0; object < features.size(); ++object) {
                for (size_t dim = 0; dim < dimension; ++dim) {
                    const auto shap = ReferenceShapValues(*tree, features[object], dim, calcType);
                    for (size_t feature = 0; feature <= FeatureCount; ++feature) {
                        expected[object][dim][feature] += shap[feature];
                    }
                }
            }
            builder.AddTree(std::move(tree));
        }
        TFullModel model;
        builder.Build(model.ModelTrees.GetMutable());
        model.ModelInfo["params"] = R"({"loss_function":{"type":"MultiRMSE","params":{}}})";
        model.UpdateDynamicData();
        NPar::TLocalExecutor executor;
        const auto actual = CalcShapValuesMulti(model, *dataset, nullptr, Nothing(), 0,
                                               EPreCalcShapValues::Auto, &executor, calcType);
        UNIT_ASSERT_VALUES_EQUAL(actual.size(), expected.size());
        for (size_t object = 0; object < features.size(); ++object) {
            UNIT_ASSERT_VALUES_EQUAL(actual[object].size(), dimension);
            for (size_t dim = 0; dim < dimension; ++dim) {
                UNIT_ASSERT_VALUES_EQUAL(actual[object][dim].size(), FeatureCount + 1);
                for (size_t feature = 0; feature <= FeatureCount; ++feature) {
                    UNIT_ASSERT_DOUBLES_EQUAL_C(actual[object][dim][feature], expected[object][dim][feature], 1e-10,
                        "object=" << object << ", dimension=" << dim << ", feature=" << feature
                        << ", leafOnRight=" << leafOnRight << ", repeatFeature=" << repeatFeature);
                }
            }
        }
    }
}

Y_UNIT_TEST_SUITE(TNonSymmetricShapValuesTest) {
    Y_UNIT_TEST(RegularMatchesEnumeratedShapleyValues) {
        for (bool leafOnRight : {false, true}) {
            for (bool repeatFeature : {false, true}) {
                for (size_t dimension : {1, 2}) {
                    CheckShapValues(ECalcTypeShapValues::Regular, leafOnRight, repeatFeature, dimension);
                }
            }
        }
    }

    Y_UNIT_TEST(ApproximateMatchesConditionalTreeValues) {
        for (bool leafOnRight : {false, true}) {
            for (bool repeatFeature : {false, true}) {
                for (size_t dimension : {1, 2}) {
                    CheckShapValues(ECalcTypeShapValues::Approximate, leafOnRight, repeatFeature, dimension);
                }
            }
        }
    }

    Y_UNIT_TEST(PreparationIncludesLeafStoredInRoot) {
        TNonSymmetricTreeModelBuilder builder({TFloatFeature(false, 0, 0, {0.0f}),
                                              TFloatFeature(false, 1, 1, {1.0f})}, {}, {}, {}, 1);
        builder.AddTree(Leaf(3.0, 10.0, 1));
        builder.AddTree(MixedTree(false, false, 1));
        builder.AddTree(MixedTree(true, false, 1));
        TFullModel model;
        builder.Build(model.ModelTrees.GetMutable());
        model.UpdateDynamicData();
        const auto prepared = PrepareTreesWithoutIndependent(model, -1, false, {},
            EPreCalcShapValues::Auto, false, ECalcTypeShapValues::Approximate);
        for (size_t tree = 0; tree < 3; ++tree) {
            const double mean = tree == 0 ? 3.0 : 6.6;
            UNIT_ASSERT_DOUBLES_EQUAL(prepared.SubtreeWeightsForAllTrees[tree][0][0], 10.0, 1e-12);
            UNIT_ASSERT_DOUBLES_EQUAL(prepared.SubtreeValuesForAllTrees[tree][0][0][0], mean, 1e-12);
            UNIT_ASSERT_DOUBLES_EQUAL(prepared.MeanValuesForAllTrees[tree][0], mean, 1e-12);
        }
    }
}
