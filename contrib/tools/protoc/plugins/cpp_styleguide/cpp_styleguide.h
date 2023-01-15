#include <google/protobuf/compiler/code_generator.h>
#include <google/protobuf/compiler/plugin.h>
#include <google/protobuf/stubs/common.h>

namespace NProtobuf::NCompiler::NPlugins {

class TCppStyleGuideExtensionGenerator : public google::protobuf::compiler::CodeGenerator {
public:
    bool Generate(const google::protobuf::FileDescriptor* file,
        const TProtoStringType& parameter,
        google::protobuf::compiler::OutputDirectory* output_directory,
        TProtoStringType* error
    ) const override;

    uint64_t GetSupportedFeatures() const override {
        return FEATURE_PROTO3_OPTIONAL;
    }
};

} // namespace NProtobuf::NCompiler::NPlugins
