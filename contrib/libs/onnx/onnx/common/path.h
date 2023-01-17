/*
 * SPDX-License-Identifier: Apache-2.0
 */

// Copyright (c) ONNX Project Contributors.
// Licensed under the MIT license.

#pragma once

#include <util/generic/string.h>

namespace ONNX_NAMESPACE {

#ifdef _WIN32
const TString k_preferred_path_separator = "\\";
#else // POSIX
const TString k_preferred_path_separator = "/";
#endif

TString path_join(const TString& origin, const TString& append);

} // namespace ONNX_NAMESPACE
