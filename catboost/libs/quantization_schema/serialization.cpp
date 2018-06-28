#include "detail.h"
#include "schema.h"
#include "serialization.h"

#include <catboost/idl/pool/proto/quantization_schema.pb.h>
#include <catboost/libs/helpers/exception.h>
#include <catboost/libs/options/enums.h>

#include <util/generic/algorithm.h>
#include <util/generic/hash.h>
#include <util/generic/strbuf.h>
#include <util/generic/string.h>
#include <util/generic/vector.h>
#include <util/stream/file.h>
#include <util/stream/input.h>
#include <util/string/iterator.h>
#include <util/string/split.h>
#include <util/generic/maybe.h>

#include <utility>

using NCB::NQuantizationSchemaDetail::NanModeFromProto;
using NCB::NQuantizationSchemaDetail::NanModeToProto;

static NCB::TPoolQuantizationSchema LoadInMatrixnetFormat(IInputStream* const input) {
    THashMap<size_t, size_t> remapping;
    TVector<TVector<float>> borders;

    // `ENanMode` doesn't have `Unknown` member, so we have to invent our own
    TVector<TMaybe<ENanMode>> nanModes;

    size_t lineIndex = 0;
    TVector<TStringBuf> columns;
    for (TString line; input->ReadLine(line); ++lineIndex) {
        columns.clear();
        StringSplitter(line).SplitLimited('\t', 3).Collect(&columns);

        if (columns.size() < 2) {
            ythrow TCatboostException() << "only " << columns.size() << "columns at line " << lineIndex;
        }

        size_t index;
        if (!TryFromString(columns[0], index)) {
            ythrow TCatboostException() << "failed to parse feature index at line " << lineIndex;
        }

        float border;
        if (!TryFromString(columns[1], border)) {
            ythrow TCatboostException() << "failed to parse border value at line " << lineIndex;
        }

        ENanMode nanMode = ENanMode::Forbidden;
        if (columns.size() >= 3 && !TryFromString(columns[2], border)) {
            ythrow TCatboostException() << "failed to parse NaN mode value at line " << lineIndex;
        }

        const auto localIndex = remapping.emplace(index, remapping.size()).first->second;
        if (borders.size() <= localIndex) {
            borders.resize(localIndex + 1);
            nanModes.resize(localIndex + 1);
        }

        if (!nanModes[localIndex].Defined()) {
            if (borders[localIndex]) {
                // Some border were missing nan mode column
                ythrow TCatboostException() << "Inconsistent Nan modes for feature " << index << ' '
                    << "at line " << lineIndex;
            }

            nanModes[localIndex] = nanMode;
        } else if (*nanModes[localIndex] != nanMode) {
            ythrow TCatboostException()
                << "Inconsistent NaN modes for feature " << index << ' '
                << "at line " << lineIndex << ' '
                << *nanModes[localIndex] << " vs. " << nanMode;
        }

        borders[localIndex].push_back(border);
    }

    for (auto& featureBorders : borders) {
        // TODO(yazevnul): write warning if borders are not not unique
        // TODO(yazevnul): write warning if borders are not sorted
        SortUnique(featureBorders);
    }

    NCB::TPoolQuantizationSchema schema;
    schema.ColumnIndices.reserve(remapping.size());
    for (const auto& kv : remapping) {
        schema.ColumnIndices.push_back(kv.first);
    }

    Sort(schema.ColumnIndices);

    schema.Borders.resize(remapping.size());
    schema.NanModes.resize(remapping.size());
    for (size_t i = 0; i < remapping.size(); ++i) {
        // copy instead of moving and doing `shrink_to_fit` later
        schema.Borders[i] = borders[remapping[schema.ColumnIndices[i]]];
        schema.NanModes[i] = nanModes[i].Defined() ? *nanModes[i] : ENanMode::Forbidden;
    }

    return schema;
}

static NCB::TPoolQuantizationSchema LoadInProtobufFormat(IInputStream* const input) {
    NCB::NIdl::TPoolQuantizationSchema proto;
    const auto parsed = proto.ParseFromIstream(input);
    CB_ENSURE(parsed, "failed to parse serialization schema from stream");

    return NCB::QuantizationSchemaFromProto(proto);
}

NCB::TPoolQuantizationSchema NCB::LoadQuantizationSchema(
    const EQuantizationsSchemaSerializationFormat format,
    IInputStream* const input) {

    switch (format) {
        case EQuantizationsSchemaSerializationFormat::Protobuf:
            return ::LoadInProtobufFormat(input);
        case EQuantizationsSchemaSerializationFormat::Matrixnet:
            return ::LoadInMatrixnetFormat(input);
        case EQuantizationsSchemaSerializationFormat::Unknown:
            break;
    }

    ythrow TCatboostException() << "Unknown quantization schema serialization format : " << static_cast<int>(format);
}

NCB::TPoolQuantizationSchema NCB::LoadQuantizationSchema(
    const EQuantizationsSchemaSerializationFormat format,
    const TStringBuf path) {

    TFileInput input{TString(path)};  // {} because of the most vexing parse
    return NCB::LoadQuantizationSchema(format, &input);
}

void SaveInMatrixnetFormat(
    const NCB::TPoolQuantizationSchema& schema,
    IOutputStream* const output) {

    for (size_t i = 0; i < schema.ColumnIndices.size(); ++i) {
        for (size_t j = 0; j < schema.Borders[i].size(); ++j) {
            (*output)
                << schema.ColumnIndices[i] << '\t'
                << schema.Borders[i][j];

            if (schema.NanModes[i] != ENanMode::Forbidden) {
                (*output) << '\t' << schema.NanModes[i];
            }

            (*output) << '\n';
        }
    }
}

void SaveInProtobufFormat(
    const NCB::TPoolQuantizationSchema& schema,
    IOutputStream* const output) {

    const auto proto = NCB::QuantizationSchemaToProto(schema);

    // TODO(yazevnul): use deterministic serialization
    const auto serialized = proto.SerializeToOstream(output);
    CB_ENSURE(serialized, "failed to safe quantization schema to stream");
}

void NCB::SaveQuantizationSchema(
    const TPoolQuantizationSchema& schema,
    const EQuantizationsSchemaSerializationFormat format,
    IOutputStream* const output) {

    switch (format) {
        case EQuantizationsSchemaSerializationFormat::Protobuf:
            return ::SaveInProtobufFormat(schema, output);
        case EQuantizationsSchemaSerializationFormat::Matrixnet:
            return ::SaveInMatrixnetFormat(schema, output);
        case EQuantizationsSchemaSerializationFormat::Unknown:
            break;
    }

    ythrow TCatboostException() << "Unknown quantization schema serialization format : " << static_cast<int>(format);
}

void NCB::SaveQuantizationSchema(
    const TPoolQuantizationSchema& schema,
    const EQuantizationsSchemaSerializationFormat format,
    const TStringBuf path) {

    TFileOutput output{TString(path)};  // {} because of the most vexing parse
    return NCB::SaveQuantizationSchema(schema, format, &output);
}

NCB::TPoolQuantizationSchema NCB::QuantizationSchemaFromProto(
    const NIdl::TPoolQuantizationSchema& proto) {

    TPoolQuantizationSchema schema;
    schema.ColumnIndices.reserve(proto.GetColumnIndexToSchema().size());
    for (const auto& kv : proto.GetColumnIndexToSchema()) {
        schema.ColumnIndices.push_back(kv.first);
    }

    Sort(schema.ColumnIndices);

    schema.Borders.resize(schema.ColumnIndices.size());
    schema.NanModes.resize(schema.ColumnIndices.size());
    for (size_t i = 0; i < schema.ColumnIndices.size(); ++i) {
        const auto& featureSchema = proto.GetColumnIndexToSchema().at(schema.ColumnIndices[i]);
        schema.Borders[i].assign(
            featureSchema.GetBorders().begin(),
            featureSchema.GetBorders().end());
        schema.NanModes[i] = NanModeFromProto(featureSchema.GetNanMode());
    }

    return schema;
}

NCB::NIdl::TPoolQuantizationSchema NCB::QuantizationSchemaToProto(
    const TPoolQuantizationSchema& schema) {

    NIdl::TPoolQuantizationSchema proto;
    for (size_t i = 0; i < schema.ColumnIndices.size(); ++i) {
        NIdl::TFeatureQuantizationSchema featureSchema;
        featureSchema.MutableBorders()->Reserve(schema.Borders[i].size());
        for (const auto border : schema.Borders[i]) {
            featureSchema.AddBorders(border);
        }

        featureSchema.SetNanMode(NanModeToProto(schema.NanModes[i]));

        proto.MutableColumnIndexToSchema()->insert({
            static_cast<ui32>(schema.ColumnIndices[i]),
            std::move(featureSchema)});
    }

    return proto;
}
