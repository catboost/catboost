#include "decl.h"

#include <contrib/libs/onnx/onnx/common/constants.h>
#include <contrib/libs/onnx/onnx/onnx_pb.h>
#include <google/protobuf/util/message_differencer.h>

#include <util/stream/file.h>
#include <util/stream/output.h>

#include <util/generic/is_in.h>
#include <util/generic/maybe.h>
#include <util/generic/strbuf.h>
#include <util/generic/string.h>
#include <util/generic/xrange.h>
#include <util/system/compiler.h>

#include <vector>


namespace NCB {

    template <>
    TMaybe<onnx::ModelProto> TryLoadModel<onnx::ModelProto>(TStringBuf filePath) {
        TMaybe<onnx::ModelProto> model = MakeMaybe<onnx::ModelProto>();

        TIFStream in{TString(filePath)};
        if (!model->ParseFromArcadiaStream(&in)) {
            return Nothing();
        }

        if (!model->has_ir_version()) {
            return Nothing();
        }

        const auto& opset_import = model->opset_import();
        bool onnxMlDomainOpsetFound = false;
        for (auto i : xrange(opset_import.size())) {
            if (opset_import[i].domain() == onnx::AI_ONNX_ML_DOMAIN) {
                onnxMlDomainOpsetFound = true;
                break;
            }
        }
        if (!onnxMlDomainOpsetFound) {
            return Nothing();
        }

        if (!model->has_domain()) {
            return Nothing();
        }
        if (!model->has_graph()) {
            return Nothing();
        }

        return model;
    }

    class TOnnxModelIgnoreFields : public google::protobuf::util::MessageDifferencer::IgnoreCriteria {
    public:
        bool IsIgnored(
            const google::protobuf::Message& message1,
            const google::protobuf::Message& message2,
            const google::protobuf::FieldDescriptor* field,
            const std::vector<google::protobuf::util::MessageDifferencer::SpecificField>& parent_fields
        ) override {
            Y_UNUSED(parent_fields);

            if (dynamic_cast<const onnx::ModelProto*>(&message1) &&
                dynamic_cast<const onnx::ModelProto*>(&message2))
            {
                if (IsIn(
                        {"producer_name", "producer_version", "domain", "model_version", "doc_string"},
                        field->name()))
                {
                    return true;
                }
                return false;
            }

            {
                const auto* stringStringEntryProto
                    = dynamic_cast<const onnx::StringStringEntryProto*>(&message1);

                if (stringStringEntryProto && dynamic_cast<const onnx::StringStringEntryProto*>(&message2)){
                    if (stringStringEntryProto->key() == "cat_features") {
                        return false;
                    }
                    return true;
                }
            }

            if (dynamic_cast<const onnx::GraphProto*>(&message1) &&
                dynamic_cast<const onnx::GraphProto*>(&message2))
            {
                if (field->name() == "name") {
                    return true;
                }
                return false;
            }

            return false;
        }

    };


    template <>
    bool CompareModels<onnx::ModelProto>(const onnx::ModelProto& model1, const onnx::ModelProto& model2, double diffLimit, TString* diffString) {
        google::protobuf::util::DefaultFieldComparator fieldComparator;
        fieldComparator.set_treat_nan_as_equal(true);
        fieldComparator.set_float_comparison(google::protobuf::util::DefaultFieldComparator::APPROXIMATE);
        fieldComparator.SetDefaultFractionAndMargin(diffLimit, diffLimit);

        auto ignoreCriteria = MakeHolder<TOnnxModelIgnoreFields>();

        google::protobuf::util::MessageDifferencer messageDifferencer;
        messageDifferencer.ReportDifferencesToString(diffString);

        messageDifferencer.set_field_comparator(&fieldComparator);
        messageDifferencer.AddIgnoreCriteria(ignoreCriteria.Release());


        const google::protobuf::FieldDescriptor* metadata_props_field
            = onnx::ModelProto::descriptor()->FindFieldByName("metadata_props");
        const google::protobuf::FieldDescriptor* metadata_props_key
            = onnx::StringStringEntryProto::descriptor()->FindFieldByName("key");

        messageDifferencer.TreatAsMap(metadata_props_field, metadata_props_key);

        return messageDifferencer.Compare(model1, model2);
    }

}
