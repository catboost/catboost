/*
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <util/generic/string.h>
#include <vector>
#include "onnx/common/status.h"
#include "onnx/onnx-operators_pb.h"

namespace ONNX_NAMESPACE {

// Helper function for register nodes in
// a FunctionProto. Attributes need to be
// registered separately.
Common::Status BuildNode(
    const TString& name,
    const TString& domain,
    const TString& doc_string,
    const TString& op_type,
    std::vector<TString> const& inputs,
    std::vector<TString> const& outputs,
    /*OUT*/ NodeProto* node);
} // namespace ONNX_NAMESPACE
