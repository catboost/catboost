#include <google/protobuf/compiler/code_generator.h>
#include <google/protobuf/compiler/plugin.h>
#include <google/protobuf/stubs/common.h>

namespace NProtobuf {
namespace NCompiler {
namespace NPlugins {

class TCppStyleGuideExtensionGenerator : public google::protobuf::compiler::CodeGenerator {
 public:
  TCppStyleGuideExtensionGenerator() {}
  ~TCppStyleGuideExtensionGenerator() {}

  virtual bool Generate(const google::protobuf::FileDescriptor* file,
      const TProtoStringType& parameter,
      google::protobuf::compiler::OutputDirectory* output_directory,
      TProtoStringType* error) const;
};


}
}
}
