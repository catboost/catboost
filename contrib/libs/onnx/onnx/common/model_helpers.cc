// Copyright (c) Facebook Inc. and Microsoft Corporation.
// Licensed under the MIT license.

#include "onnx/common/model_helpers.h"
#include "onnx/checker.h"
#include "onnx/defs/schema.h"
#include "onnx/string_utils.h"

namespace ONNX_NAMESPACE {

Common::Status BuildNode(
    const TString& name,
    const TString& domain,
    const TString& doc_string,
    const TString& op_type,
    std::vector<TString> const& inputs,
    std::vector<TString> const& outputs,
    NodeProto* node) {
  if (node == NULL) {
    return Common::Status(
        Common::CHECKER,
        Common::INVALID_ARGUMENT,
        "node_proto should not be nullptr.");
  }
  node->set_name(name);
  node->set_domain(domain);
  node->set_doc_string(doc_string);
  node->set_op_type(op_type);
  for (auto& input : inputs) {
    node->add_input(input);
  }
  for (auto& output : outputs) {
    node->add_output(output);
  }

  return Common::Status::OK();
}
} // namespace ONNX_NAMESPACE