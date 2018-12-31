#include "onnx.h"

#include <contrib/libs/protobuf/util/message_differencer.h>

#include <util/stream/file.h>
#include <util/stream/output.h>

#include <util/generic/is_in.h>
#include <util/system/compiler.h>

#include <vector>


namespace NCB {

    TMaybe<onnx::ModelProto> TryLoadOnnxModel(TStringBuf filePath) {
        TMaybe<onnx::ModelProto> model = MakeMaybe<onnx::ModelProto>();

        TIFStream in{TString(filePath)};
        if (model->ParseFromIstream(&in)) {
            return model;
        }

        return Nothing();
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


    bool Compare(const onnx::ModelProto& model1, const onnx::ModelProto& model2, TString* diffString) {
        google::protobuf::util::DefaultFieldComparator fieldComparator;
        fieldComparator.set_treat_nan_as_equal(true);

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
