#pragma once

#include "features_layout.h"

#include <catboost/private/libs/data_types/text.h>
#include <catboost/libs/helpers/array_subset.h>
#include <catboost/libs/helpers/checksum.h>
#include <catboost/libs/helpers/compression.h>
#include <catboost/libs/helpers/dynamic_iterator.h>
#include <catboost/libs/helpers/exception.h>
#include <catboost/libs/helpers/maybe_owning_array_holder.h>
#include <catboost/libs/helpers/polymorphic_type_containers.h>

#include <library/cpp/threading/local_executor/local_executor.h>

#include <util/system/types.h>
#include <util/generic/noncopyable.h>
#include <util/generic/string.h>
#include <util/generic/vector.h>
#include <util/generic/yexception.h>
#include <util/stream/buffer.h>
#include <util/stream/labeled.h>
#include <util/system/compiler.h>
#include <util/system/yassert.h>

#include <climits>
#include <cmath>
#include <type_traits>


namespace NCB {

    //feature values storage optimized for memory usage

    enum class EFeatureValuesType {
        Float,                      //32 bits per feature value
        QuantizedFloat,             //quantized with at most 8 bits (for GPU) or 16 bits (for CPU) per
                                    // feature value.
        HashedCategorical,          //values - 32 bit hashes of original strings
        PerfectHashedCategorical,   //after perfect hashing
        StringText,                 //unoptimized text feature
        TokenizedText,              //32 bits for each token in string
        Embedding,                  //array of float values
        BinaryPack,                 //aggregate of binary features
        ExclusiveFeatureBundle,     //aggregate of exclusive quantized features
        FeaturesGroup               //aggregate of several quantized float features
    };

    template<class TDst, class TSrc>
    THolder<TDst> DynamicHolderCast(THolder<TSrc>&& srcHolder, TStringBuf errorMessage) {
        CB_ENSURE_INTERNAL(
            dynamic_cast<TDst*>(srcHolder.Get()) != nullptr,
            errorMessage
        );
        return THolder<TDst>(dynamic_cast<TDst*>(srcHolder.Release()));
    }

    //feature values storage optimized for memory usage
    using TFeaturesArraySubsetIndexing = TArraySubsetIndexing<ui32>;
    using TFeaturesArraySubsetInvertedIndexing = TArraySubsetInvertedIndexing<ui32>;
    using TConstCompressedArraySubset = TArraySubset<const TCompressedArray, ui32>;

    template <class T>
    using TConstPtrArraySubset = TArraySubset<const T*, ui32>;

    struct TCloningParams {
        bool MakeConsecutive = false;
        const TFeaturesArraySubsetIndexing* SubsetIndexing = nullptr;
        // used for sparse columns
        TMaybe<const TFeaturesArraySubsetInvertedIndexing*> InvertedSubsetIndexing;
    };

    class IFeatureValuesHolder: TMoveOnly {
    public:
        virtual ~IFeatureValuesHolder() = default;

        IFeatureValuesHolder(EFeatureValuesType featureValuesType,
                             ui32 featureId,
                             ui32 size)
            : FeatureId(featureId)
            , Size(size)
            , FeatureValuesType(featureValuesType)
        {
        }

        IFeatureValuesHolder(IFeatureValuesHolder&& arg) noexcept = default;
        IFeatureValuesHolder& operator=(IFeatureValuesHolder&& arg) noexcept = default;

        EFeatureType GetFeatureType() const {
            switch (GetType()) {
                case EFeatureValuesType::Float:
                case EFeatureValuesType::QuantizedFloat:
                    return EFeatureType::Float;
                case EFeatureValuesType::HashedCategorical:
                case EFeatureValuesType::PerfectHashedCategorical:
                    return EFeatureType::Categorical;
                case EFeatureValuesType::StringText:
                case EFeatureValuesType::TokenizedText:
                    return EFeatureType::Text;
                case EFeatureValuesType::Embedding:
                    return EFeatureType::Embedding;
                case EFeatureValuesType::BinaryPack:
                case EFeatureValuesType::ExclusiveFeatureBundle:
                case EFeatureValuesType::FeaturesGroup:
                    CB_ENSURE_INTERNAL(false, "GetFeatureType called for Aggregate type");
            }
            CB_ENSURE(false, "This place should be inaccessible");
            return EFeatureType::Float; // to keep compiler happy
        }

        ui32 GetSize() const {
            return Size;
        }

        ui32 GetId() const {
            return FeatureId;
        }

        EFeatureValuesType GetType() const {
            return FeatureValuesType;
        }

        virtual const IFeatureValuesHolder* GetParent() const {
            return nullptr;
        }

        virtual ui32 CalcChecksum(NPar::ILocalExecutor* localExecutor) const = 0;
        virtual bool IsSparse() const = 0;

        virtual ui64 EstimateMemoryForCloning(
            const TCloningParams& cloningParams
        ) const = 0;

        virtual THolder<IFeatureValuesHolder> CloneWithNewSubsetIndexing(
            const TCloningParams& cloningParams,
            NPar::ILocalExecutor* localExecutor
        ) const = 0;

    private:
        ui32 FeatureId;
        ui32 Size;
        EFeatureValuesType FeatureValuesType;
    };

    /*******************************************************************************************************
     * Common interfaces
     */

    template <typename T, EFeatureValuesType ValuesType>
    struct ITypedFeatureValuesHolder : public IFeatureValuesHolder {
        using TValueType = T;
        ITypedFeatureValuesHolder(ui32 featureId, ui32 size)
            : IFeatureValuesHolder(ValuesType, featureId, size)
        {}
        virtual TMaybeOwningArrayHolder<T> ExtractValues(NPar::ILocalExecutor* localExecutor) const = 0;

        virtual IDynamicBlockIteratorPtr<T> GetBlockIterator(ui32 offset = 0) const = 0;

        ui32 CalcChecksum(NPar::ILocalExecutor* localExecutor) const override {
            const auto repackedHolder = ExtractValues(localExecutor);
            return UpdateCheckSum(0, *repackedHolder);
        }
    };

    template<class TTypedIteratorFunc>
    static void DispatchIteratorType(IDynamicBlockIteratorBase* blockIterator, TTypedIteratorFunc&& typedIteratorFunc) {
        if (auto ui8iter = dynamic_cast<IDynamicBlockIterator<ui8>*>(blockIterator)) {
            typedIteratorFunc(ui8iter);
        } else if (auto ui16iter = dynamic_cast<IDynamicBlockIterator<ui16>*>(blockIterator)) {
            typedIteratorFunc(ui16iter);
        } else if (auto ui32iter = dynamic_cast<IDynamicBlockIterator<ui32>*>(blockIterator)) {
            typedIteratorFunc(ui32iter);
        } else {
            CB_ENSURE(0, "Unexpected iterator basetype");
        }
    }

    template <typename T, EFeatureValuesType ValuesType, typename TBaseInterface = IFeatureValuesHolder>
    struct IQuantizedFeatureValuesHolder : public TBaseInterface {
    public:
        using TValueType = T;
    public:
        IQuantizedFeatureValuesHolder(ui32 featureId, ui32 size)
            : TBaseInterface(ValuesType, featureId, size)
        {}

        template <typename TExtractType>
        TVector<TExtractType> ExtractValues(NPar::ILocalExecutor* localExecutor, size_t copyBlockSize = 1024) const {
            TVector<TExtractType> result;
            result.yresize(this->GetSize());
            ParallelForEachBlock(
                localExecutor,
                [&result] (size_t blockStartIdx, auto vals) {
                    Copy(vals.begin(), vals.end(), result.begin() + blockStartIdx);
                },
                copyBlockSize
            );
            return result;
        }

        virtual IDynamicBlockIteratorBasePtr GetBlockIterator(ui32 offset = 0) const = 0;

        template<class TBlockCallable>
        static void ForEachBlockRange(IDynamicBlockIteratorBasePtr&& blockIterator, size_t blockStartIdx, size_t upperLimit, TBlockCallable&& blockCallable, size_t blockSize = 1024) {
            DispatchIteratorType(
                blockIterator.Get(),
                [blockStartIdx, upperLimit, blockCallable = std::move(blockCallable), blockSize] (auto typedIter) mutable {
                    while (auto block = typedIter->Next(Min(blockSize, upperLimit - blockStartIdx))) {
                        blockCallable(blockStartIdx, block);
                        blockStartIdx += block.size();
                        if (blockStartIdx >= upperLimit) {
                            break;
                        }
                    }
                }
            );
        }

        template<class TBlockCallable>
        void ForEachBlock(TBlockCallable&& blockCallable, size_t blockSize = 1024) const {
            ForEachBlockRange(this->GetBlockIterator(0), 0, this->GetSize(), std::move(blockCallable), blockSize);
        }

        template<class TBlockCallable>
        void ParallelForEachBlock(NPar::ILocalExecutor* localExecutor, TBlockCallable&& blockCallable, size_t blockSize = 1024) const {
            NPar::ILocalExecutor::TExecRangeParams blockParams(0, this->GetSize());
            blockParams.SetBlockCount(localExecutor->GetThreadCount() + 1);
            // round per-thread block size to iteration block size
            blockParams.SetBlockSize(Min<int>(this->GetSize(), CeilDiv<int>(blockParams.GetBlockSize(), blockSize) * blockSize));
            localExecutor->ExecRangeWithThrow(
                [blockCallable = std::move(blockCallable), blockParams, blockSize, this] (int blockId) {
                    const int blockFirstId = blockParams.FirstId + blockId * blockParams.GetBlockSize();
                    const int blockLastId = Min(blockParams.LastId, blockFirstId + blockParams.GetBlockSize());
                    ForEachBlockRange(this->GetBlockIterator(blockFirstId), blockFirstId, blockLastId, blockCallable, blockSize);
                }, 0, blockParams.GetBlockCount(), NPar::TLocalExecutor::WAIT_COMPLETE
            );
        }

        ui32 CalcChecksum(NPar::ILocalExecutor* localExecutor) const override {
            Y_UNUSED(localExecutor);
            ui32 checkSum = 0;
            auto blockFunc = [&checkSum] (size_t /*blockStartIdx*/, auto block) {
                checkSum = UpdateCheckSum(checkSum, block);
            };
            ForEachBlock(std::move(blockFunc));
            return checkSum;
        }
    };

    /*******************************************************************************************************
     * Raw data
     */

    template <class TBase>
    class TPolymorphicArrayValuesHolder: public TBase {
    public:
        using TValueType = typename TBase::TValueType;
    public:
        TPolymorphicArrayValuesHolder(ui32 featureId,
                                      ITypedArraySubsetPtr<TValueType>&& data)
            : TBase(featureId, data->GetSize())
            , Data(std::move(data))
        {}

        TPolymorphicArrayValuesHolder(ui32 featureId,
                                      TMaybeOwningConstArrayHolder<TValueType> srcData,
                                      const TFeaturesArraySubsetIndexing* subsetIndexing)
            : TBase(featureId, subsetIndexing->Size())
            , Data(MakeIntrusive<TTypeCastArraySubset<TValueType, TValueType>>(std::move(srcData), subsetIndexing))
        {}

        bool IsSparse() const override {
            return false;
        }

        ui64 EstimateMemoryForCloning(
            const TCloningParams& cloningParams
        ) const override {
            CB_ENSURE_INTERNAL(
                !cloningParams.MakeConsecutive,
                "Consecutive cloning of TPolymorphicArrayValuesHolder unimplemented"
            );
            return 0;
        }

        THolder<IFeatureValuesHolder> CloneWithNewSubsetIndexing(
            const TCloningParams& cloningParams,
            NPar::ILocalExecutor* localExecutor
        ) const override {
            Y_UNUSED(localExecutor);
            CB_ENSURE_INTERNAL(
                !cloningParams.MakeConsecutive,
                "Consecutive cloning of TPolymorphicArrayValuesHolder unimplemented"
            );
            return MakeHolder<TPolymorphicArrayValuesHolder>(
                this->GetId(),
                Data->CloneWithNewSubsetIndexing(cloningParams.SubsetIndexing)
            );
        }

        TMaybeOwningArrayHolder<TValueType> ExtractValues(NPar::ILocalExecutor* localExecutor) const override {
            TVector<TValueType> result;
            result.yresize(Data->GetSize());
            TArrayRef<TValueType> resultRef = result;

            Data->ParallelForEach(
                [=] (ui32 dstIdx, TValueType value) {
                    resultRef[dstIdx] = value;
                },
                localExecutor
            );
            return TMaybeOwningArrayHolder<TValueType>::CreateOwning(std::move(result));
        }

        IDynamicBlockIteratorPtr<TValueType> GetBlockIterator(ui32 offset = 0) const override {
            return Data->GetBlockIterator(offset);
        }

        ITypedArraySubsetPtr<TValueType> GetData() const {
            return Data;
        }

    private:
        ITypedArraySubsetPtr<TValueType> Data;
    };

    /*******************************************************************************************************
     * Quantized/prepared for quantization data
     */


    template <class TBase>
    class TCompressedValuesHolderImpl : public TBase {
    public:
        TCompressedValuesHolderImpl(ui32 featureId,
                                    TCompressedArray srcData,
                                    const TFeaturesArraySubsetIndexing* subsetIndexing)
            : TBase(featureId, subsetIndexing->Size())
            , SrcData(std::move(srcData))
            , SrcDataRawPtr(SrcData.GetRawPtr())
            , SubsetIndexing(subsetIndexing)
        {
            CB_ENSURE(SubsetIndexing, "subsetIndexing is empty");
        }

        bool IsSparse() const override {
            return false;
        }

        ui32 CalcChecksum(NPar::ILocalExecutor* localExecutor) const override {
            TConstCompressedArraySubset compressedDataSubset = GetCompressedData();

            auto consecutiveSubsetBegin = compressedDataSubset.GetSubsetIndexing()->GetConsecutiveSubsetBegin();
            const ui32 columnValuesBitWidth = GetBitsPerKey();
            if (consecutiveSubsetBegin.Defined()) {
                ui8 byteSize = columnValuesBitWidth / 8;
                return UpdateCheckSum(
                    0,
                    MakeArrayRef(
                        compressedDataSubset.GetSrc()->GetRawPtr() + *consecutiveSubsetBegin * byteSize,
                        compressedDataSubset.Size() * byteSize)
                );
            }
            return TBase::CalcChecksum(localExecutor);
        }

        ui64 EstimateMemoryForCloning(
            const TCloningParams& cloningParams
        ) const override {
            if (!cloningParams.MakeConsecutive) {
                return 0;
            } else {
                const ui32 objectCount = this->GetSize();
                const ui32 bitsPerKey = this->GetBitsPerKey();
                TIndexHelper<ui64> indexHelper(bitsPerKey);
                return indexHelper.CompressedSize(objectCount);
            }
        }

        THolder<IFeatureValuesHolder> CloneWithNewSubsetIndexing(
            const TCloningParams& cloningParams,
            NPar::ILocalExecutor* localExecutor
        ) const override {
            if (!cloningParams.MakeConsecutive) {
                return MakeHolder<TCompressedValuesHolderImpl>(
                    TBase::GetId(),
                    SrcData,
                    cloningParams.SubsetIndexing
                );
            } else {
                const ui32 objectCount = this->GetSize();
                const ui32 bitsPerKey = this->GetBitsPerKey();
                TIndexHelper<ui64> indexHelper(bitsPerKey);
                const ui32 dstStorageSize = indexHelper.CompressedSize(objectCount);

                TVector<ui64> storage;
                storage.yresize(dstStorageSize);

                if (bitsPerKey == 8) {
                    auto dstBuffer = (ui8*)(storage.data());

                    GetArrayData<ui8>().ParallelForEach(
                        [dstBuffer](ui32 idx, ui8 value) {
                            dstBuffer[idx] = value;
                        },
                        localExecutor
                    );
                } else if (bitsPerKey == 16) {
                    auto dstBuffer = (ui16*)(storage.data());

                    GetArrayData<ui16>().ParallelForEach(
                        [dstBuffer](ui32 idx, ui16 value) {
                            dstBuffer[idx] = value;
                        },
                        localExecutor
                    );
                } else {
                    auto dstBuffer = (ui32*)(storage.data());

                    GetArrayData<ui32>().ParallelForEach(
                        [dstBuffer](ui32 idx, ui32 value) {
                            dstBuffer[idx] = value;
                        },
                        localExecutor
                    );
                }

                return MakeHolder<TCompressedValuesHolderImpl>(
                    this->GetId(),
                    TCompressedArray(
                        objectCount,
                        bitsPerKey,
                        TMaybeOwningArrayHolder<ui64>::CreateOwning(std::move(storage))
                    ),
                    cloningParams.SubsetIndexing
                );
            }
        }

        TConstCompressedArraySubset GetCompressedData() const {
            return {&SrcData, SubsetIndexing};
        }

        template <class T2>
        TConstPtrArraySubset<T2> GetArrayData() const {
            SrcData.CheckIfCanBeInterpretedAsRawArray<T2>();
            return TConstPtrArraySubset<T2>((const T2**)&SrcDataRawPtr, SubsetIndexing);
        }

        IDynamicBlockIteratorBasePtr GetBlockIterator(ui32 offset = 0) const override {
            return SrcData.GetBlockIterator(offset, SubsetIndexing);
        }

        ui32 GetBitsPerKey() const {
            return SrcData.GetBitsPerKey();
        }

    private:
        TCompressedArray SrcData;
        void* SrcDataRawPtr;
        const TFeaturesArraySubsetIndexing* SubsetIndexing;
    };

    using TFloatValuesHolder = ITypedFeatureValuesHolder<float, EFeatureValuesType::Float>;
    using TFloatArrayValuesHolder = TPolymorphicArrayValuesHolder<TFloatValuesHolder>;

    using THashedCatValuesHolder = ITypedFeatureValuesHolder<ui32, EFeatureValuesType::HashedCategorical>;
    using THashedCatArrayValuesHolder = TPolymorphicArrayValuesHolder<THashedCatValuesHolder>;

    using TStringTextValuesHolder = ITypedFeatureValuesHolder<TString, EFeatureValuesType::StringText>;
    using TStringTextArrayValuesHolder = TPolymorphicArrayValuesHolder<TStringTextValuesHolder>;

    using TTokenizedTextValuesHolder = ITypedFeatureValuesHolder<TText, EFeatureValuesType::TokenizedText>;
    using TTokenizedTextArrayValuesHolder = TPolymorphicArrayValuesHolder<TTokenizedTextValuesHolder>;


    using TConstEmbedding = TMaybeOwningConstArrayHolder<float>;

    using TEmbeddingValuesHolder = ITypedFeatureValuesHolder<TConstEmbedding, EFeatureValuesType::Embedding>;
    using TEmbeddingArrayValuesHolder = TPolymorphicArrayValuesHolder<TEmbeddingValuesHolder>;


    using IQuantizedFloatValuesHolder = IQuantizedFeatureValuesHolder<ui8, EFeatureValuesType::QuantizedFloat>;
    using IQuantizedCatValuesHolder = IQuantizedFeatureValuesHolder<ui32, EFeatureValuesType::PerfectHashedCategorical>;

    using TQuantizedFloatValuesHolder = TCompressedValuesHolderImpl<IQuantizedFloatValuesHolder>;
    using TQuantizedCatValuesHolder = TCompressedValuesHolderImpl<IQuantizedCatValuesHolder>;

}
