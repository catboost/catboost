#include "pool.h"

#include <catboost/idl/pool/flat/quantized_chunk_t.fbs.h>
#include <catboost/libs/helpers/exception.h>
#include <util/system/unaligned_mem.h>

namespace NCB {
void TQuantizedPool::AddColumn(
    const size_t featureIndex,
    const size_t baselineIndex,
    const EColumn columnType,
    const size_t localIndex,
    NCB::IPoolBuilder* const builder) const {

    switch (columnType) {
        case EColumn::Num: {
            for (const auto& descriptor : Chunks[localIndex]) {
                CB_ENSURE(static_cast<size_t>(descriptor.Chunk->BitsPerDocument()) == sizeof(ui8) * 8);
                builder->AddBinarizedFloatFeaturePack(descriptor.DocumentOffset,
                    featureIndex,
                    *descriptor.Chunk->Quants());
            }
            break;
        }
        case EColumn::Label: {
            for (const auto& descriptor : Chunks[localIndex]) {
                CB_ENSURE(static_cast<size_t>(descriptor.Chunk->BitsPerDocument()) == sizeof(float) * 8);
                TUnalignedMemoryIterator<float> it(
                    descriptor.Chunk->Quants()->data(),
                    descriptor.Chunk->Quants()->size());
                for (ui32 i = descriptor.DocumentOffset; !it.AtEnd(); it.Next(), (void)++i) {
                    builder->AddTarget(i, it.Cur());
                }
            }
            break;
        }
        case EColumn::Baseline: {
            for (const auto& descriptor : Chunks[localIndex]) {
                CB_ENSURE(static_cast<size_t>(descriptor.Chunk->BitsPerDocument()) == sizeof(double) * 8);
                TUnalignedMemoryIterator<double> it(
                    descriptor.Chunk->Quants()->data(),
                    descriptor.Chunk->Quants()->size());
                for (ui32 i = descriptor.DocumentOffset; !it.AtEnd(); it.Next(), (void)++i) {
                    builder->AddBaseline(i, baselineIndex, it.Cur());
                }
            }
            break;
        }
        case EColumn::Weight:
        case EColumn::GroupWeight: {
            for (const auto& descriptor : Chunks[localIndex]) {
                CB_ENSURE(static_cast<size_t>(descriptor.Chunk->BitsPerDocument()) == sizeof(float) * 8);
                TUnalignedMemoryIterator<float> it(
                    descriptor.Chunk->Quants()->data(),
                    descriptor.Chunk->Quants()->size());
                for (ui32 i = descriptor.DocumentOffset; !it.AtEnd(); it.Next(), (void)++i) {
                    builder->AddWeight(i, it.Cur());
                }
            }
            break;
        }
        case EColumn::DocId: {
            break;
        }
        case EColumn::GroupId: {
            for (const auto& descriptor : Chunks[localIndex]) {
                CB_ENSURE(static_cast<size_t>(descriptor.Chunk->BitsPerDocument()) == sizeof(ui64) * 8);
                TUnalignedMemoryIterator<ui64> it(
                    descriptor.Chunk->Quants()->data(),
                    descriptor.Chunk->Quants()->size());
                for (ui32 i = descriptor.DocumentOffset; !it.AtEnd(); it.Next(), (void)++i) {
                    builder->AddQueryId(i, it.Cur());
                }
            }
            break;
        }
        case EColumn::SubgroupId: {
            for (const auto& descriptor : Chunks[localIndex]) {
                CB_ENSURE(static_cast<size_t>(descriptor.Chunk->BitsPerDocument()) == sizeof(ui32) * 8);
                TUnalignedMemoryIterator<ui32> it(
                    descriptor.Chunk->Quants()->data(),
                    descriptor.Chunk->Quants()->size());
                for (ui32 i = descriptor.DocumentOffset; !it.AtEnd(); it.Next(), (void)++i) {
                    builder->AddSubgroupId(i, it.Cur());
                }
            }
            break;
        }
        case EColumn::Categ:
            // TODO(yazevnul): categorical feature quantization on YT is still in progress
        case EColumn::Auxiliary:
            // Should not be present in quantized pool
        case EColumn::Timestamp:
            // not supported by quantized pools right now
        case EColumn::Sparse:
            // not supperted by CatBoost at all
        case EColumn::Prediction: {
            // can't be present in quantized pool
            ythrow TCatboostException() << "Unexpected column type " << columnType;
        }
    }
}
} // NCB
