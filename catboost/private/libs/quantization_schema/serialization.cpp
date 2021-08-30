#include "serialization.h"

#include "detail.h"

#include <catboost/libs/data/cat_feature_perfect_hash.h>
#include <catboost/libs/helpers/exception.h>
#include <catboost/private/libs/options/enums.h>

#include <util/generic/algorithm.h>
#include <util/generic/array_ref.h>
#include <util/generic/hash.h>
#include <util/generic/map.h>
#include <util/generic/maybe.h>
#include <util/generic/string.h>
#include <util/generic/vector.h>
#include <util/stream/file.h>
#include <util/stream/input.h>
#include <util/stream/labeled.h>
#include <util/string/escape.h>
#include <util/string/cast.h>
#include <util/string/split.h>

#include <google/protobuf/messagext.h>


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
        StringSplitter(line).Split('\t').Limit(3).Collect(&columns);

        if (columns.size() < 2) {
            ythrow TCatBoostException() << "only " << columns.size() << "columns at line " << lineIndex;
        }

        size_t index;
        if (!TryFromString(columns[0], index)) {
            ythrow TCatBoostException() << "failed to parse feature index; " << LabeledOutput(lineIndex, EscapeC(columns[0]));
        }

        float border;
        if (!TryFromString(columns[1], border)) {
            ythrow TCatBoostException() << "failed to parse border value; " << LabeledOutput(lineIndex, EscapeC(columns[1]));
        }

        ENanMode nanMode = ENanMode::Forbidden;
        if (columns.size() >= 3 && !TryFromString(columns[2], nanMode)) {
            ythrow TCatBoostException() << "failed to parse NaN mode value; " << LabeledOutput(lineIndex, EscapeC(columns[2]));
        }

        const auto localIndex = remapping.emplace(index, remapping.size()).first->second;
        if (borders.size() <= localIndex) {
            borders.resize(localIndex + 1);
            nanModes.resize(localIndex + 1);
        }

        if (!nanModes[localIndex].Defined()) {
            if (borders[localIndex]) {
                // Some border were missing nan mode column
                ythrow TCatBoostException() << "Inconsistent Nan modes for feature " << index << ' '
                    << "at line " << lineIndex;
            }

            nanModes[localIndex] = nanMode;
        } else if (*nanModes[localIndex] != nanMode) {
            ythrow TCatBoostException()
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
    schema.FloatFeatureIndices.reserve(remapping.size());
    for (const auto& kv : remapping) {
        schema.FloatFeatureIndices.push_back(kv.first);
    }

    Sort(schema.FloatFeatureIndices);

    schema.Borders.resize(remapping.size());
    schema.NanModes.resize(remapping.size());
    for (size_t i = 0; i < remapping.size(); ++i) {
        size_t localIndex = remapping[schema.FloatFeatureIndices[i]];

        // copy instead of moving and doing `shrink_to_fit` later
        schema.Borders[i] = borders[localIndex];
        schema.NanModes[i] = nanModes[localIndex].GetOrElse(ENanMode::Forbidden);
    }

    return schema;
}

static NCB::TPoolQuantizationSchema LoadInProtobufFormat(IInputStream* const input) {
    NCB::NIdl::TPoolQuantizationSchema proto;
    const auto parsed = proto.ParseFromArcadiaStream(input);
    CB_ENSURE(parsed, "failed to parse serialization schema from stream");

    return NCB::QuantizationSchemaFromProto(proto);
}

NCB::TPoolQuantizationSchema NCB::LoadQuantizationSchema(
    const EQuantizationSchemaSerializationFormat format,
    IInputStream* const input) {

    switch (format) {
        case EQuantizationSchemaSerializationFormat::Protobuf:
            return ::LoadInProtobufFormat(input);
        case EQuantizationSchemaSerializationFormat::Matrixnet:
            return ::LoadInMatrixnetFormat(input);
        case EQuantizationSchemaSerializationFormat::Unknown:
            break;
    }

    ythrow TCatBoostException() << "Unknown quantization schema serialization format : " << static_cast<int>(format);
}

NCB::TPoolQuantizationSchema NCB::LoadQuantizationSchema(
    const EQuantizationSchemaSerializationFormat format,
    const TStringBuf path) {

    TFileInput input{TString(path)};  // {} because of the most vexing parse
    return NCB::LoadQuantizationSchema(format, &input);
}

void SaveInMatrixnetFormat(
    const NCB::TPoolQuantizationSchema& schema,
    IOutputStream* const output) {

    for (size_t i = 0; i < schema.FloatFeatureIndices.size(); ++i) {
        for (size_t j = 0; j < schema.Borders[i].size(); ++j) {
            (*output)
                << schema.FloatFeatureIndices[i] << '\t'
                << FloatToString(schema.Borders[i][j], PREC_NDIGITS, 9);

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

    google::protobuf::io::TCopyingOutputStreamAdaptor outputAdaptor(output);
    google::protobuf::io::CodedOutputStream coder(&outputAdaptor);
    coder.SetSerializationDeterministic(true);
    CB_ENSURE(proto.SerializeToCodedStream(&coder), "failed to save quantization schema to stream");
}

void NCB::SaveQuantizationSchema(
    const TPoolQuantizationSchema& schema,
    const EQuantizationSchemaSerializationFormat format,
    IOutputStream* const output) {

    switch (format) {
        case EQuantizationSchemaSerializationFormat::Protobuf:
            return ::SaveInProtobufFormat(schema, output);
        case EQuantizationSchemaSerializationFormat::Matrixnet:
            return ::SaveInMatrixnetFormat(schema, output);
        case EQuantizationSchemaSerializationFormat::Unknown:
            break;
    }

    ythrow TCatBoostException() << "Unknown quantization schema serialization format : " << static_cast<int>(format);
}

void NCB::SaveQuantizationSchema(
    const TPoolQuantizationSchema& schema,
    const EQuantizationSchemaSerializationFormat format,
    const TStringBuf path) {

    TFileOutput output{TString(path)};  // {} because of the most vexing parse
    return NCB::SaveQuantizationSchema(schema, format, &output);
}


void BuildFeaturePerfectHash(
    const google::protobuf::Map<google::protobuf::uint32, NCB::NIdl::TValueWithCount>& protoPerfectHashes,
    TMap<ui32, NCB::TValueWithCount>* perfectHashes) {
    for (const auto& kv : protoPerfectHashes) {
        //init value with count
        NCB::TValueWithCount valueWithCount;
        valueWithCount.Value = kv.second.GetValue();
        valueWithCount.Count = kv.second.GetCount();
        perfectHashes->insert({kv.first, valueWithCount});
    }
}


void BuildProtoFeaturePerfectHash(
    const TMap<ui32, NCB::TValueWithCount>& perfectHashes,
    google::protobuf::Map<google::protobuf::uint32, NCB::NIdl::TValueWithCount>* protoPerfectHashes
    ) {
    for (const auto& kv : perfectHashes) {
        //init value with count
        NCB::NIdl::TValueWithCount valueWithCount;
        valueWithCount.SetValue(kv.second.Value);
        valueWithCount.SetCount(kv.second.Count);
        protoPerfectHashes->insert({kv.first, std::move(valueWithCount)});
    }
}


template <class TRepeatedField> // TRepeatedField in either RepeatedField or RepeatedPtrField
static void InitClassLabels(
    const TRepeatedField& protoClassLabels,
    TVector<NJson::TJsonValue>* dstClassLabels) {

    dstClassLabels->clear();
    dstClassLabels->reserve(protoClassLabels.size());

    for (int i = 0; i < protoClassLabels.size(); ++i) {
        dstClassLabels->emplace_back(protoClassLabels[i]);
    }
}


NCB::TPoolQuantizationSchema NCB::QuantizationSchemaFromProto(
    const NIdl::TPoolQuantizationSchema& proto) {

    TPoolQuantizationSchema schema;
    schema.FloatFeatureIndices.reserve(proto.GetFeatureIndexToSchema().size());
    for (const auto& kv : proto.GetFeatureIndexToSchema()) {
        schema.FloatFeatureIndices.push_back(kv.first);
    }

    Sort(schema.FloatFeatureIndices);

    schema.Borders.resize(schema.FloatFeatureIndices.size());
    schema.NanModes.resize(schema.FloatFeatureIndices.size());
    for (size_t i = 0; i < schema.FloatFeatureIndices.size(); ++i) {
        const auto& featureSchema = proto.GetFeatureIndexToSchema().at(schema.FloatFeatureIndices[i]);
        schema.Borders[i].assign(
            featureSchema.GetBorders().begin(),
            featureSchema.GetBorders().end());
        schema.NanModes[i] = NanModeFromProto(featureSchema.GetNanMode());
    }

    const auto& integerClassLabels = proto.GetIntegerClassLabels();
    const auto& floatClassLabels = proto.GetFloatClassLabels();
    const auto& stringClassLabels = proto.GetClassNames();

    CB_ENSURE_INTERNAL(
        !integerClassLabels.empty() + !floatClassLabels.empty() + !stringClassLabels.empty() <= 1,
        "More than one type of labels specified in TPoolQuantizationSchema"
    );

    if (!integerClassLabels.empty()) {
        InitClassLabels(integerClassLabels, &schema.ClassLabels);
    } else if (!floatClassLabels.empty()) {
        InitClassLabels(floatClassLabels, &schema.ClassLabels);
    } else if (!stringClassLabels.empty()) {
        InitClassLabels(stringClassLabels, &schema.ClassLabels);
    }

    // for categorical features

    schema.CatFeatureIndices.reserve(proto.GetCatFeatureIndexToSchema().size());
    for (const auto& kv : proto.GetCatFeatureIndexToSchema()) {
        schema.CatFeatureIndices.push_back(kv.first);
    }

    Sort(schema.CatFeatureIndices);

    schema.FeaturesPerfectHash.resize(schema.CatFeatureIndices.size());
    for (size_t i = 0; i < schema.CatFeatureIndices.size(); ++i) {
        const auto& featureSchema = proto.GetCatFeatureIndexToSchema().at(schema.CatFeatureIndices[i]);
        BuildFeaturePerfectHash(featureSchema.GetPerfectHashes(), &schema.FeaturesPerfectHash[i]);
    }


    return schema;
}


static void AddClassLabels(
    TConstArrayRef<NJson::TJsonValue> classLabels,
    NCB::NIdl::TPoolQuantizationSchema* protoSchema) {

    if (classLabels.empty()) {
        return;
    }

    switch (classLabels[0].GetType()) {
        case NJson::JSON_INTEGER:
            protoSchema->MutableIntegerClassLabels()->Reserve(classLabels.size());
            for (const auto& classLabel : classLabels) {
                protoSchema->AddIntegerClassLabels(classLabel.GetInteger());
            }
            break;
        case NJson::JSON_DOUBLE:
            protoSchema->MutableFloatClassLabels()->Reserve(classLabels.size());
            for (const auto& classLabel : classLabels) {
                protoSchema->AddFloatClassLabels(static_cast<float>(classLabel.GetDouble()));
            }
            break;
        case NJson::JSON_STRING:
            protoSchema->MutableClassNames()->Reserve(classLabels.size());
            for (const auto& classLabel : classLabels) {
                protoSchema->AddClassNames(classLabel.GetString());
            }
            break;
        default:
            CB_ENSURE_INTERNAL(false, "bad class label type: " << classLabels[0].GetType());
    }
}


NCB::NIdl::TPoolQuantizationSchema NCB::QuantizationSchemaToProto(
    const TPoolQuantizationSchema& schema) {

    NIdl::TPoolQuantizationSchema proto;
    for (size_t i = 0; i < schema.FloatFeatureIndices.size(); ++i) {
        NIdl::TFeatureQuantizationSchema floatFeatureSchema;
        floatFeatureSchema.MutableBorders()->Reserve(schema.Borders[i].size());
        for (const auto border : schema.Borders[i]) {
            floatFeatureSchema.AddBorders(border);
        }

        floatFeatureSchema.SetNanMode(NanModeToProto(schema.NanModes[i]));

        proto.MutableFeatureIndexToSchema()->insert({
            static_cast<ui32>(schema.FloatFeatureIndices[i]),
            std::move(floatFeatureSchema)});
    }

    AddClassLabels(schema.ClassLabels, &proto);

    for (size_t i = 0; i < schema.CatFeatureIndices.size(); ++i) {
        NIdl::TCatFeatureQuantizationSchema catFeatureSchema;
        BuildProtoFeaturePerfectHash(schema.FeaturesPerfectHash[i], catFeatureSchema.MutablePerfectHashes());
        proto.MutableCatFeatureIndexToSchema()->insert({
            static_cast<ui32>(schema.CatFeatureIndices[i]),
            std::move(catFeatureSchema)});
    }

    return proto;
}
