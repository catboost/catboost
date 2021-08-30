#include <sstream>
#include <vector>
#include <google/protobuf/compiler/perlxs/perlxs_helpers.h>
#include "google/protobuf/descriptor.pb.h"
#include <google/protobuf/io/printer.h>

namespace google {
namespace protobuf {

extern TProtoStringType StringReplace(const TProtoStringType& s, const TProtoStringType& oldsub,
			    const TProtoStringType& newsub, bool replace_all);

namespace compiler {
namespace perlxs {

void
SetupDepthVars(std::map<TProtoStringType, TProtoStringType>& vars, int depth)
{
  std::ostringstream ost_pdepth;
  std::ostringstream ost_depth;
  std::ostringstream ost_ndepth;

  ost_pdepth << depth;
  ost_depth  << depth + 1;
  ost_ndepth << depth + 2;

  vars["pdepth"] = ost_pdepth.str();
  vars["depth"]  = ost_depth.str();
  vars["ndepth"] = ost_ndepth.str();
}

}  // namespace perlxs
}  // namespace compiler
}  // namespace protobuf
}  // namespace google
