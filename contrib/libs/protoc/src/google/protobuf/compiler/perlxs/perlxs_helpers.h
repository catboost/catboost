#ifndef GOOGLE_PROTOBUF_COMPILER_PERLXS_HELPERS_H__
#define GOOGLE_PROTOBUF_COMPILER_PERLXS_HELPERS_H__

#include <map>
#include "stubs/common.h"

namespace google {
namespace protobuf {
namespace compiler {
namespace perlxs {

void SetupDepthVars(std::map<string, string>& vars, int depth);

}  // namespace perlxs
}  // namespace compiler
}  // namespace protobuf
}  // namespace google

#endif  // GOOGLE_PROTOBUF_COMPILER_PERLXS_HELPERS_H__
