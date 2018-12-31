#pragma once

#include <contrib/libs/onnx/proto/onnx_ml.pb.h>

#include <util/generic/maybe.h>
#include <util/generic/strbuf.h>
#include <util/generic/string.h>


namespace NCB {

    TMaybe<onnx::ModelProto> TryLoadOnnxModel(TStringBuf filePath);

    // returns true if models are equal
    bool Compare(const onnx::ModelProto& model1, const onnx::ModelProto& model2, TString* diffString);

}
