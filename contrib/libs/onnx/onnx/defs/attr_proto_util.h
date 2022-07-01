/*
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "onnx/onnx-operators_pb.h"

namespace ONNX_NAMESPACE {

AttributeProto MakeAttribute(const TString& attr_name, const float& value);
AttributeProto MakeAttribute(const TString& attr_name, const int64_t& value);
AttributeProto MakeAttribute(const TString& attr_name, const TString& value);
AttributeProto MakeAttribute(const TString& attr_name, const TensorProto& value);
AttributeProto MakeAttribute(const TString& attr_name, const GraphProto& value);
AttributeProto MakeAttribute(const TString& attr_name, const std::vector<float>& values);
AttributeProto MakeAttribute(const TString& attr_name, const std::vector<int64_t>& values);
AttributeProto MakeAttribute(const TString& attr_name, const std::vector<TString>& values);
AttributeProto MakeAttribute(const TString& attr_name, const std::vector<TensorProto>& values);
AttributeProto MakeAttribute(const TString& attr_name, const std::vector<GraphProto>& values);

// Make a "reference" attribute for a node in a function body.
// <attr_name> specifies the attribute name of both the function node and its
// function body node. They're using the same attribute name.
// <type> specifies the attribute type.
AttributeProto MakeRefAttribute(const TString& attr_name, AttributeProto_AttributeType type);

// Make a "reference" attribute for a node in a function body.
// <attr_name> specifies the attribute name of the function body node.
// <referred_attr_name> specifies the referred attribute name of the function
// node.
// <type> specifies the attribute type.
AttributeProto MakeRefAttribute(
    const TString& attr_name,
    const TString& referred_attr_name,
    AttributeProto_AttributeType type);

} // namespace ONNX_NAMESPACE
