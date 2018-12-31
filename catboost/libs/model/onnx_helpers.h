#pragma once

#include "model.h"

#include <contrib/libs/onnx/proto/onnx_ml.pb.h>

#include <library/json/json_value.h>

#include <util/generic/maybe.h>
#include <util/generic/string.h>


namespace NCatboost {
    namespace NOnnx {
        void InitMetadata(
            const TFullModel& model,
            const NJson::TJsonValue& userParameters,
            onnx::ModelProto* onnxModel);

        void ConvertTreeToOnnxGraph(
            const TFullModel& model,
            const TMaybe<TString>& onnxGraphName, // "CatBoostModel" if not defined
            onnx::GraphProto* onnxGraph);
    }
}
