#pragma once

#include <catboost/libs/model/model.h>

#include <library/json/json_value.h>

#include <contrib/libs/onnx/proto/onnx_ml.pb.h>

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
