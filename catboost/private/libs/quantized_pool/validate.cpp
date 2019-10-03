#include <catboost/idl/pool/flat/quantized_chunk_t.fbs.h>
#include <catboost/libs/helpers/exception.h>
#include <catboost/private/libs/validate_fb/validate.h>

#include <util/system/types.h>

static void Validate(const NCB::NIdl::TQuantizedFeatureChunk& chunk) {
    CB_ENSURE(chunk.Quants());
}

template <>
void NCB::ValidateFlatBuffer<NCB::NIdl::TQuantizedFeatureChunk>(
    const void* const data,
    const size_t size) {

    CB_ENSURE(data, "got nullptr");
    const auto& schema = *flatbuffers::GetRoot<NCB::NIdl::TQuantizedFeatureChunk>(data);
    {
        flatbuffers::Verifier verifier(reinterpret_cast<const ui8*>(data), size);
        CB_ENSURE(schema.Verify(verifier), "not valid FlatBuffer");
    }
}
