#include "validate.h"

#include <catboost/idl/pool/flat/quantization_schema.fbs.h>

#include <catboost/libs/helpers/exception.h>

#include <util/generic/algorithm.h>
#include <util/system/types.h>

static void Validate(const NCB::NIdl::TFeatureQuantizationSchema& s) {
    if (s.Borders() && s.UniqueHashes()) {
        ythrow TCatboostException() << "feature quantization schema has both `Borders` and `UniqueHashes`";
    }

    if (const auto* const bs = s.Borders()) {
        // TODO(yazevnul): do we want to check for NaNs?

        CB_ENSURE(bs->size() > 0, "borders are empty");
        CB_ENSURE(IsSorted(bs->begin(), bs->end()), "borders are not sorted");
        for (size_t i = 0; i < bs->size() - 1; ++i) {
            if (bs->Get(i) == bs->Get(i + 1)) {
                ythrow TCatboostException() << "borders at " << i << "and at " << i + 1 << " are equal";
            }
        }
    }

    if (const auto* const hs = s.UniqueHashes()) {
        CB_ENSURE(hs->size() > 0, "unique hashes are empty");
        CB_ENSURE(IsSorted(hs->begin(), hs->end()), "unique hashes are not sorted");
        for (size_t i = 0; i < hs->size() - 1; ++i) {
            if (hs->Get(i) == hs->Get(i + 1)) {
                ythrow TCatboostException() << "uniqe hashes at " << i << "and at " << i + 1 << "are equal";
            }
        }
    }
}

template <>
void NCB::ValidateFlatBuffer<NCB::NIdl::TFeatureQuantizationSchema>(const void* data, const size_t size) {
    CB_ENSURE(data, "got nullptr");

    const auto& s = *flatbuffers::GetRoot<NCB::NIdl::TFeatureQuantizationSchema>(data);
    {
        flatbuffers::Verifier verifier(reinterpret_cast<const ui8*>(data), size);
        CB_ENSURE(s.Verify(verifier), "not valid FlatBuffer");
    }

    Validate(s);
}

template <>
void NCB::ValidateFlatBuffer<NCB::NIdl::TFeatureQuantizationSchemas>(const void* data, const size_t size) {
    CB_ENSURE(data, "got nullptr");

    const auto& ss = *flatbuffers::GetRoot<NCB::NIdl::TFeatureQuantizationSchemas>(data);
    {
        flatbuffers::Verifier verifier(reinterpret_cast<const ui8*>(data), size);
        CB_ENSURE(ss.Verify(verifier), "not valid FlatBuffer");
    }

    CB_ENSURE(ss.Schemas(), "no quantization schemas");
    for (const auto* const s : *ss.Schemas()) {
        Validate(*s);
    }
}
