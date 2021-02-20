#include "coreml_helpers.h"
#include "json_model_helpers.h"

#include <catboost/libs/model/model_import_interface.h>
#include <catboost/libs/model/model_export/onnx_helpers.h>

#include <contrib/libs/coreml/TreeEnsemble.pb.h>
#include <contrib/libs/coreml/Model.pb.h>

#include <util/generic/buffer.h>
#include <util/stream/buffer.h>
#include <util/stream/file.h>
#include <util/system/fs.h>

namespace NCB {
    class TCoreMLModelLoader : public IModelLoader {
    public:
        TFullModel ReadModel(IInputStream* modelStream) const override {
            TFullModel model;
            CoreML::Specification::Model coreMLModel;
            CB_ENSURE(coreMLModel.ParseFromString(modelStream->ReadAll()), "coreml model deserialization failed");
            NCB::NCoreML::ConvertCoreMLToCatboostModel(coreMLModel, &model);
            CheckModel(&model);
            return model;
        }
    };

    TModelLoaderFactory::TRegistrator<TCoreMLModelLoader> CoreMLModelLoaderRegistrator(EModelType::AppleCoreML);

    class TOnnxModelLoader : public IModelLoader {
    public:
        TFullModel ReadModel(IInputStream* modelStream) const override {
            TFullModel model;
            onnx::ModelProto onnxModel;
            CB_ENSURE(onnxModel.ParseFromString(modelStream->ReadAll()), "onnx model deserialization failed");
            NCB::NOnnx::ConvertOnnxToCatboostModel(onnxModel, &model);
            CheckModel(&model);
            return model;
        }
    };

    TModelLoaderFactory::TRegistrator<TOnnxModelLoader> OnnxModelLoaderRegistrator(EModelType::Onnx);
}
