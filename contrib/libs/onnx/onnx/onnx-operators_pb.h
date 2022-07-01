/*
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "onnx/onnx_pb.h"
#ifdef ONNX_ML
#include "onnx/onnx_operators_ml.pb.h"
#else
#error #include "onnx/onnx-operators.pb.h"
#endif
