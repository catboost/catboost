#pragma once

#include "schema.h"

#include <catboost/idl/pool/proto/quantization_schema.pb.h>

#include <util/stream/fwd.h>
#include <util/generic/strbuf.h>
#include <util/system/types.h>


namespace NCB {
    enum class EQuantizationSchemaSerializationFormat : ui8 {
        Unknown = 0,
        Protobuf = 1,

        // NOTE:
        // - can't express empty set of borders
        // - can't express class labels
        Matrixnet = 2
    };

    TPoolQuantizationSchema LoadQuantizationSchema(
        EQuantizationSchemaSerializationFormat format,
        TStringBuf path);

    TPoolQuantizationSchema LoadQuantizationSchema(
        EQuantizationSchemaSerializationFormat format,
        IInputStream* input);

    void SaveQuantizationSchema(
        const TPoolQuantizationSchema& schema,
        EQuantizationSchemaSerializationFormat format,
        TStringBuf path);

    void SaveQuantizationSchema(
        const TPoolQuantizationSchema& schema,
        EQuantizationSchemaSerializationFormat format,
        IOutputStream* output);

    TPoolQuantizationSchema QuantizationSchemaFromProto(
        const NIdl::TPoolQuantizationSchema& schema);

    NIdl::TPoolQuantizationSchema QuantizationSchemaToProto(
        const TPoolQuantizationSchema& schema);
}
