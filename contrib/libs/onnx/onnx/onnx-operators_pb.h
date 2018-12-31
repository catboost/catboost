// Copyright (c) Facebook Inc. and Microsoft Corporation.
// Licensed under the MIT license.

#pragma once

#include "onnx/onnx_pb.h"
#ifdef ONNX_ML
#include <contrib/libs/onnx/proto/onnx_operators_ml.pb.h>
#else
//#include <contrib/libs/onnx/proto/onnx_operators.pb.h>
#error "Arcadia supports only ONNX_ML-enabled build"
#endif
