#pragma once

#include <util/generic/fwd.h>
#include <util/stream/fwd.h>
#include <util/system/types.h>

namespace NCB {
    struct TPoolQuantizationSchema;

    namespace NIdl {
        class TPoolQuantizationSchema;
    }
}

namespace NCB {
    enum class EQuantizationsSchemaSerializationFormat : ui8 {
        Unknown = 0,
        Protobuf = 1,

        // NOTE: can't express empty set of borders
        Matrixnet = 2
    };

    TPoolQuantizationSchema LoadQuantizationSchema(
        EQuantizationsSchemaSerializationFormat format,
        TStringBuf path);

    TPoolQuantizationSchema LoadQuantizationSchema(
        EQuantizationsSchemaSerializationFormat format,
        IInputStream* input);

    void SaveQuantizationSchema(
        const TPoolQuantizationSchema& schema,
        EQuantizationsSchemaSerializationFormat format,
        TStringBuf path);

    void SaveQuantizationSchema(
        const TPoolQuantizationSchema& schema,
        EQuantizationsSchemaSerializationFormat format,
        IOutputStream* output);

    TPoolQuantizationSchema QuantizationSchemaFromProto(
        const NIdl::TPoolQuantizationSchema& schema);

    NIdl::TPoolQuantizationSchema QuantizationSchemaToProto(
        const TPoolQuantizationSchema& schema);
}
