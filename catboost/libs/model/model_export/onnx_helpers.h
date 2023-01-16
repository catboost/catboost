#pragma once

#include <catboost/libs/model/model.h>

#include <library/cpp/json/json_value.h>

#include <contrib/libs/onnx/onnx/onnx_pb.h>

#include <util/generic/maybe.h>
#include <util/generic/string.h>


struct TTreesAttributes;

namespace NCB {
    namespace NOnnx {
        void InitMetadata(
            const TFullModel& model,
            const NJson::TJsonValue& userParameters,
            onnx::ModelProto* onnxModel);

        void ConvertTreeToOnnxGraph(
            const TFullModel& model,
            const TMaybe<TString>& onnxGraphName, // "CatBoostModel" if not defined
            onnx::GraphProto* onnxGraph);

        struct TOnnxNode {
            enum class EType {
                Leaf, Inner
            };

            struct TTreeNodeBehaviorToOnnxTreeNodeMode {
                static const TString BRANCH_LEQ;
                static const TString BRANCH_LT;
                static const TString BRANCH_GTE;
                static const TString BRANCH_GT;
                static const TString BRANCH_EQ;
                static const TString BRANCH_NEQ;
                static const TString LEAF;
            };

            TOnnxNode() = default;
            int FalseNodeId = 0;
            int TrueNodeId = 0;
            EType Type;
            TMaybe<TModelSplit> SplitCondition;
            TVector<double> Values;
        };

        void ConvertOnnxToCatboostModel(const onnx::ModelProto& onnxModel, TFullModel* fullModel);
    }
}
