#include "arrow.h"

#include "helpers.h"

#include <catboost/libs/cat_feature/cat_feature.h>
#include <catboost/libs/data/visitor.h>
#include <catboost/libs/helpers/maybe_owning_array_holder.h>
#include <catboost/libs/helpers/polymorphic_type_containers.h>
#include <catboost/libs/helpers/resource_holder.h>

#include <contrib/libs/apache/arrow_next/cpp/src/arrow/c/abi.h>

#include <library/cpp/float16/float16.h>
#include <library/cpp/threading/local_executor/local_executor.h>

#include <util/generic/algorithm.h>
#include <util/generic/hash.h>
#include <util/generic/maybe.h>
#include <util/generic/strbuf.h>
#include <util/generic/xrange.h>
#include <util/generic/yexception.h>
#include <util/generic/ylimits.h>
#include <util/generic/ymath.h>
#include <util/string/cast.h>
#include <util/system/compiler.h>
#include <util/system/unaligned_mem.h>

#include <climits>
#include <cstring>
#include <limits>
#include <type_traits>


namespace NCB {

    // subset of Arrow data types used in CatBoost
    enum class ESupportedArrowDataType : ui32 {
        BOOL,
        INT8,
        INT16,
        INT32,
        INT64,
        UINT8,
        UINT16,
        UINT32,
        UINT64,
        HALF_FLOAT,
        FLOAT,
        DOUBLE,
        STRING,
        STRING_VIEW
    };

    // save mapping from polars data type to data types used in CatBoost
    class TArrowDataTypeIdMapping {
    public:
        TArrowDataTypeIdMapping()
            : SimpleMapping_(
                {
                    {"b", ESupportedArrowDataType::BOOL},
                    {"c", ESupportedArrowDataType::INT8},
                    {"C", ESupportedArrowDataType::UINT8},
                    {"s", ESupportedArrowDataType::INT16},
                    {"S", ESupportedArrowDataType::UINT16},
                    {"i", ESupportedArrowDataType::INT32},
                    {"I", ESupportedArrowDataType::UINT32},
                    {"l", ESupportedArrowDataType::INT64},
                    {"L", ESupportedArrowDataType::UINT64},
                    {"e", ESupportedArrowDataType::HALF_FLOAT},
                    {"f", ESupportedArrowDataType::FLOAT},
                    {"g", ESupportedArrowDataType::DOUBLE},
                    {"u", ESupportedArrowDataType::STRING},
                    {"vu", ESupportedArrowDataType::STRING_VIEW}
                }
            )
        {
        }

        static TArrowDataTypeIdMapping& Instance() {
            static TArrowDataTypeIdMapping instance;
            return instance;
        }

        ESupportedArrowDataType Get(const ArrowSchema& schema) const {
            auto* res = SimpleMapping_.FindPtr(schema.format);
            Y_ENSURE(res, "Unsupported Arrow format: '" << schema.format << "'");
            return *res;
        }

    private:
        THashMap<TStringBuf, ESupportedArrowDataType> SimpleMapping_;
    };



    struct TArrowArrayHolder : public IResourceHolder {
    public:
        ArrowArray Data;

    public:
        explicit TArrowArrayHolder(ArrowArrayStream* stream) {
            Y_ENSURE(
                !stream->get_next(stream, &Data),
                "ArrowArrayStream::get_next failed: " << stream->get_last_error(stream)
            );
            Y_ENSURE(Data.release != nullptr, "ArrowArrayStream: no chunks");
        }

        ~TArrowArrayHolder() override {
            Data.release(&Data);
        }
    };

    using TArrowArrayHolderPtr = TIntrusivePtr<TArrowArrayHolder>;


    struct TArrowSchemaHolder : public IResourceHolder {
    public:
        ArrowSchema Data;

    public:
        explicit TArrowSchemaHolder(ArrowArrayStream* stream) {
            Y_ENSURE(
                !stream->get_schema(stream, &Data),
                "ArrowArrayStream::get_schema failed: " << stream->get_last_error(stream)
            );
            Y_ENSURE(
                Data.release != nullptr,
                "ArrowArrayStream::get_schema returned ArrowSchema with release == nullptr"
            );
        }

        ~TArrowSchemaHolder() override {
            Data.release(&Data);
        }
    };

    using TArrowSchemaHolderPtr = TIntrusivePtr<TArrowSchemaHolder>;


    // TArrayLike must:
    //  - be lightweight to copy
    //  - has operator[]
    //  - has GetSize()
    template <class TArrayLike>
    class TArrayLikeAsFloatBlockIterator final : public IDynamicBlockWithExactIterator<float> {
    public:
        TArrayLikeAsFloatBlockIterator(TArrayLike&& data)
            : Data_(std::move(data))
        {}

        TConstArrayRef<float> Next(size_t maxBlockSize = Max<size_t>()) override {
            return NextExact(Min(maxBlockSize, RemainingElements()));
        }

        TConstArrayRef<float> NextExact(size_t exactBlockSize) override {
            DstBuffer_.yresize(exactBlockSize);
            for (auto i : xrange(exactBlockSize)) {
                DstBuffer_[i] = static_cast<float>(Data_[Offset_ + i]);
            }
            Offset_ += exactBlockSize;
            return DstBuffer_;
        }
    private:
        size_t RemainingElements() const {
            return size_t(Data_.GetSize()) - Offset_;
        }

    private:
        const TArrayLike Data_;
        size_t Offset_;

        TVector<float> DstBuffer_;
    };


    // TArrayLike must:
    //  - be lightweight copyable
    //  - provide operator[] that returns a type convertible to float
    //  - provide operator== to compare with itself
    //  - serializable using IBinSaver
    //  - provide method `TArrayLike Slice(size_t offset, size_t size)`
    template <class TArrayLike>
    class TArrayLikeAsFloatSequence final : public ITypedSequence<float> {
    public:
        explicit TArrayLikeAsFloatSequence(TArrayLike&& data)
            : Data_(std::move(data))
        {}

        int operator&(IBinSaver& binSaver) override {
            binSaver.Add(0, &Data_);
            return 0;
        }

        bool EqualTo(const ITypedSequence<float>& rhs, bool strict = true) const override {
            if (strict) {
                if (const auto* rhsAsThisType
                        = dynamic_cast<const TArrayLikeAsFloatSequence*>(&rhs))
                {
                    return Data_ == rhsAsThisType->Data_;
                } else {
                    return false;
                }
            } else {
                return AreBlockedSequencesEqual<float, float>(
                    ITypedSequence<float>::GetBlockIterator(),
                    rhs.ITypedSequence<float>::GetBlockIterator()
                );
            }
        }

        ui32 GetSize() const override {
            return SafeIntegerCast<ui32>(Data_.GetSize());
        }

        IDynamicBlockWithExactIteratorPtr<float> GetBlockIterator(
            TIndexRange<ui32> indexRange
        ) const override {
            return MakeHolder<TArrayLikeAsFloatBlockIterator<TArrayLike>>(
                Data_.Slice(indexRange.Begin, indexRange.GetSize())
            );
        }

        TIntrusivePtr<ITypedArraySubset<float>> GetSubset(
            const TArraySubsetIndexing<ui32>* subsetIndexing
        ) const override {
            return MakeIntrusive<
                TTransformingArrayLikeSubset<float, TArrayLike, TStaticCast<float, float>>
            >(
                Data_,
                subsetIndexing
            );
        }

    private:
        TArrayLike Data_;
    };


    class TMaybeOwnedBitMap {
    public:
        // non-owning
        explicit TMaybeOwnedBitMap(
            const ui8* data,
            ui32 beginOffsetInData,
            ui32 size,
            TIntrusivePtr<IResourceHolder> resourceHolder
        )
            : Data_(
                TMaybeOwningConstArrayHolder<const ui8>::CreateOwning(
                    {
                        data + beginOffsetInData / CHAR_BIT,
                        CeilDiv<size_t>(beginOffsetInData % CHAR_BIT + size, CHAR_BIT)
                    },
                    std::move(resourceHolder)
                )
            )
            , BeginOffsetInData_(beginOffsetInData % CHAR_BIT)
            , Size_(size)
        {
        }

        int operator&(IBinSaver& binSaver) {
            binSaver.Add(0, &Data_);
            binSaver.Add(1, &BeginOffsetInData_);
            binSaver.Add(1, &Size_);
            return 0;
        }

        bool operator==(const TMaybeOwnedBitMap& rhs) const {
            if (Size_ != rhs.Size_) {
                return false;
            }
            // could be optimized for some cases but this operation is not performance-critical right now
            for (auto i : xrange(Size_)) {
                if ((*this)[i] != rhs[i]) {
                    return false;
                }
            }
            return true;
        }

        bool operator[](size_t idx) const {
            Y_ASSERT(idx < Size_);
            size_t offsetInData = BeginOffsetInData_ + idx;
            return (Data_[offsetInData / CHAR_BIT] >> (offsetInData % CHAR_BIT)) & 1;
        }

        ui32 GetSize() const {
            return Size_;
        }

        TMaybeOwnedBitMap Slice(size_t offset, size_t size) const {
            return TMaybeOwnedBitMap(
                Data_.data(),
                BeginOffsetInData_ + offset,
                size,
                Data_.GetResourceHolder()
            );
        }

    private:
        TMaybeOwningConstArrayHolder<const ui8> Data_;
        ui32 BeginOffsetInData_;
        ui32 Size_;
    };


    // TArrayLike must:
    //  - be lightweight copyable
    //  - provide operator[] that returns a type convertible to float
    //  - provide operator== to compare with itself
    //  - serializable using IBinSaver
    //  - provide method `TArrayLike Slice(size_t offset, size_t size)`
    template <class TArrayLike>
    class TArrayWithValidityHolder {
    public:
        explicit TArrayWithValidityHolder(
            TArrayLike values,
            TMaybeOwnedBitMap&& validities
        )
            : Values_(std::move(values))
            , Validities_(std::move(validities))
        {}

        int operator&(IBinSaver& binSaver) {
            binSaver.Add(0, &Values_);
            binSaver.Add(0, &Validities_);
            return 0;
        }

        float operator[](size_t idx) const {
            if (Validities_[idx]) {
                return Values_[idx];
            } else {
                return std::numeric_limits<float>::quiet_NaN();
            }
        }

        bool operator==(const TArrayWithValidityHolder& rhs) const {
            return (Values_ == rhs.Values_) && (Validities_ == rhs.Validities_);
        }

        size_t GetSize() const {
            return Validities_.GetSize();
        }

        TArrayWithValidityHolder Slice(size_t offset, size_t size) const {
            return TArrayWithValidityHolder(
                Values_.Slice(offset, size),
                Validities_.Slice(offset, size)
            );
        }

    private:
        TArrayLike Values_;
        TMaybeOwnedBitMap Validities_;
    };


    template <class F>
    void ProcessArrowArrayStream(
        PyObject* capsule,
        F&& callback
    ) {
        try {
            auto* stream = (ArrowArrayStream*)PyCapsule_GetPointer(capsule, "arrow_array_stream");
            if (!stream) {
                return;
            }

            auto array  = MakeIntrusive<TArrowArrayHolder>(stream);
            auto schema = MakeIntrusive<TArrowSchemaHolder>(stream);

            callback(std::move(schema), std::move(array));

        } catch (...) {
            ProcessException();
        }
    }

    void AsyncAddArrowNumColumn(
        ui32 flatFeatureIdx,
        PyObject* capsule,
        IRawFeaturesOrderDataVisitor* builderVisitor,
        TVector<std::future<void>>* result
    ) {
        auto callback = [=] (TArrowSchemaHolderPtr schema, TArrowArrayHolderPtr array) {
            auto process = [=] () {
                try {
                    CB_ENSURE(array->Data.n_buffers == 2, "Expected two buffers");

                    const ui8* __restrict validity = static_cast<const ui8*>(array->Data.buffers[0]);
                    const void* dataBuf = array->Data.buffers[1];
                    auto arrayOffset = array->Data.offset;
                    auto arraySize = array->Data.length;

                    ITypedSequencePtr<float> result;

                    auto typedProcessing = [&] (const auto* __restrict typedDataBuf) {
                        using TSrcType = std::remove_cvref_t<decltype(*typedDataBuf)>;
                        auto arrayDataHolder = TMaybeOwningConstArrayHolder<TSrcType>::CreateOwning(
                            {typedDataBuf + arrayOffset, typedDataBuf + arrayOffset + arraySize},
                            array
                        );

                        if (validity) {
                            using TInner = TArrayWithValidityHolder<TMaybeOwningConstArrayHolder<TSrcType>>;
                            result = MakeIntrusive<TArrayLikeAsFloatSequence<TInner>>(
                                TInner(
                                    std::move(arrayDataHolder),
                                    TMaybeOwnedBitMap(validity, arrayOffset, arraySize, array)
                                )
                            );
                        } else {
                            result = MakeTypeCastArrayHolder<float>(std::move(arrayDataHolder));
                        }
                    };

                    auto boolProcessing = [&] () {
                        auto data = TMaybeOwnedBitMap(
                            static_cast<const ui8*>(dataBuf),
                            arrayOffset,
                            arraySize,
                            array
                        );
                        if (validity) {
                            using TInner = TArrayWithValidityHolder<TMaybeOwnedBitMap>;
                            result = MakeIntrusive<TArrayLikeAsFloatSequence<TInner>>(
                                TInner(
                                    std::move(data),
                                    TMaybeOwnedBitMap(validity, arrayOffset, arraySize, array)
                                )
                            );
                        } else {
                            result = MakeIntrusive<TArrayLikeAsFloatSequence<TMaybeOwnedBitMap>>(std::move(data));
                        }
                    };

                    auto type = TArrowDataTypeIdMapping::Instance().Get(schema->Data);
                    switch (type) {
                        #define HANDLE_TYPE(enumType, cppType)                          \
                            case ESupportedArrowDataType::enumType:                     \
                                typedProcessing(static_cast<const cppType*>(dataBuf));  \
                                break;

                        HANDLE_TYPE(INT8, i8);
                        HANDLE_TYPE(INT16, i16);
                        HANDLE_TYPE(INT32, i32);
                        HANDLE_TYPE(INT64, i64);
                        HANDLE_TYPE(UINT8, ui8);
                        HANDLE_TYPE(UINT16, ui16);
                        HANDLE_TYPE(UINT32, ui32);
                        HANDLE_TYPE(UINT64, ui64);
                        HANDLE_TYPE(HALF_FLOAT, TFloat16);
                        HANDLE_TYPE(FLOAT, float);
                        HANDLE_TYPE(DOUBLE, double);

                        #undef HANDLE_TYPE

                        case NCB::ESupportedArrowDataType::BOOL:
                            boolProcessing();
                            break;
                        default:
                            ythrow TCatBoostException() << "Unsupported Arrow data type: " << schema->Data.format;
                    }

                    builderVisitor->AddFloatFeature(flatFeatureIdx, std::move(result));

                } catch (...) {
                    ythrow TCatBoostException() << "Error while processing column '" << schema->Data.name << "': "
                        << CurrentExceptionMessage();
                }
            };

            result->push_back(std::async(std::move(process)));
        };

        ProcessArrowArrayStream(
            capsule,
            std::move(callback)
        );
    }


    template <class TPerElement>
    void ProcessNonNullableColumn(
        const ui8* __restrict validity,
        size_t arrayOffset,
        size_t arraySize,
        TStringBuf columnType,
        TPerElement&& perElement
    ) {
        if (validity) {
            for (auto dstObjIdx : xrange(arraySize)) {
                const auto srcObjIdx = dstObjIdx + arrayOffset;
                const auto subIdx = srcObjIdx % CHAR_BIT;
                const bool isValid = (validity[srcObjIdx / CHAR_BIT] >> subIdx) & 1;
                CB_ENSURE(
                    isValid,
                    "contains null at index "
                    << dstObjIdx << " which is not supported for " << columnType << " columns"
                );
                perElement(dstObjIdx);
            }
        } else {
            for (auto dstObjIdx : xrange(arraySize)) {
                perElement(dstObjIdx);
            }
        }
    }


    struct TArrowStringView {
        ui32 Length;
        union {
            struct {
                char Prefix[4];
                ui32 BufferIndex;
                ui32 Offset;
            } LongString;
            char SmallString[12];
        };
    };

    static_assert(sizeof(TArrowStringView) == 16);


    template <bool Aligned>
    class TStringViewDataAccessor {
    public:
        explicit TStringViewDataAccessor(TArrowArrayHolderPtr array)
            : ArraySize_(array->Data.length)
            , ViewBuffer_(((const char*)array->Data.buffers[1]) + array->Data.offset * sizeof(TArrowStringView))
            , LongDataBuffers_((const char**)(array->Data.buffers + 2))
        {}

        TStringBuf operator()(size_t dstIdx) const {
            Y_ASSERT(dstIdx < ArraySize_);

            auto proc = [&] (const TArrowStringView& arrowStringView) -> TStringBuf {
                if (arrowStringView.Length <= 12) {
                    return TStringBuf(arrowStringView.SmallString, arrowStringView.Length);
                } else {
                    return TStringBuf(
                        LongDataBuffers_[arrowStringView.LongString.BufferIndex] + arrowStringView.LongString.Offset,
                        arrowStringView.Length
                    );
                }
            };

            if constexpr (Aligned) {
                return proc(static_cast<const TArrowStringView*>(ViewBuffer_)[dstIdx]);
            } else {
                return proc(
                    ReadUnaligned<TArrowStringView>(
                        static_cast<const char*>(ViewBuffer_) + sizeof(TArrowStringView) * dstIdx
                    )
                );
            }
        }

    private:
        size_t ArraySize_;
        const void* ViewBuffer_;    // non typed because of possible alignment issues
        const char** LongDataBuffers_;
    };


    template <class TPerElement>
    void ProcessNonNullableStringColumn(
        TArrowSchemaHolderPtr schema,
        TArrowArrayHolderPtr array,
        TStringBuf columnType,
        TPerElement&& perElement    // accepts (dstIdx, TStringBuf value) args
    ) {
        const ui8* __restrict validity = static_cast<const ui8*>(array->Data.buffers[0]);

        auto arraySize = array->Data.length;
        auto arrayOffset = array->Data.offset;

        auto processColumn = [&, perElement=std::move(perElement)] (auto&& accessor) {
            ProcessNonNullableColumn(
                validity,
                arrayOffset,
                arraySize,
                columnType,
                [=, perElement=std::move(perElement), accessor=std::move(accessor)] (size_t dstObjIdx) {
                    perElement(
                        dstObjIdx,
                        accessor(dstObjIdx)
                    );
                }
            );
        };

        switch (TArrowDataTypeIdMapping::Instance().Get(schema->Data)) {
            case ESupportedArrowDataType::STRING:
                {
                    CB_ENSURE(array->Data.n_buffers == 3, "Expected three buffers");

                    const char* __restrict data = static_cast<const char*>(array->Data.buffers[1]);
                    const i32* __restrict shiftedOffsets = static_cast<const i32*>(array->Data.buffers[2]) + arrayOffset;

                    processColumn(
                        [=] (size_t dstObjIdx) {
                            return TStringBuf(
                                data + shiftedOffsets[dstObjIdx],
                                data + shiftedOffsets[dstObjIdx + 1]
                            );
                        }
                    );
                }
                break;
            case ESupportedArrowDataType::STRING_VIEW:
                {
                    CB_ENSURE(array->Data.n_buffers >= 2, "Expected at least two buffers");

                    const void* viewBuffer = array->Data.buffers[1];
                    if (uintptr_t(viewBuffer) % sizeof(TArrowStringView) == 0) {
                        processColumn(TStringViewDataAccessor<true>(array));
                    } else {
                        processColumn(TStringViewDataAccessor<false>(array));
                    }
                }
                break;
            default:
                ythrow TCatBoostException() << "Unsupported Arrow data type: " << schema->Data.format;
        }
    }


    void AsyncAddArrowCategoricalColumnOfStrings(
        ui32 flatFeatureIdx,
        PyObject* capsule,
        IRawFeaturesOrderDataVisitor* builderVisitor,
        TVector<std::future<void>>* result
    ) {
        auto callback = [=] (TArrowSchemaHolderPtr schema, TArrowArrayHolderPtr array) {
            auto process = [=] () {
                try {
                    CB_ENSURE(
                        array->Data.null_count <= 0,
                        "Data with nulls is not supported for categorical columns"
                    );

                    TVector<ui32> result;
                    result.yresize(array->Data.length);

                    ProcessNonNullableStringColumn(
                        schema,
                        array,
                        "categorical",
                        [&, flatFeatureIdx](size_t dstObjIdx, TStringBuf value) {
                            result[dstObjIdx] = builderVisitor->GetCatFeatureValue(flatFeatureIdx, value);
                        }
                    );

                    builderVisitor->AddCatFeature(
                        flatFeatureIdx,
                        CreateOwningWithMaybeTypeCast<const ui32>(
                            TMaybeOwningArrayHolder<ui32>::CreateOwning(std::move(result))
                        )
                    );
                } catch (...) {
                    ythrow TCatBoostException() << "Error while processing column '" << schema->Data.name << "': "
                        << CurrentExceptionMessage();
                }
            };

            result->push_back(std::async(std::move(process)));
        };

        ProcessArrowArrayStream(
            capsule,
            std::move(callback)
        );
    }

    void AsyncAddArrowCategoricalColumnOfIntOrBoolean(
        ui32 flatFeatureIdx,
        PyObject* capsule,
        IRawFeaturesOrderDataVisitor* builderVisitor,
        TVector<std::future<void>>* result
    ) {
        auto callback = [=] (TArrowSchemaHolderPtr schema, TArrowArrayHolderPtr array) {
            auto process = [=] () {
                try {
                    CB_ENSURE(
                        array->Data.null_count <= 0,
                        "Data with nulls is not supported for categorical columns"
                    );

                    CB_ENSURE(array->Data.n_buffers == 2, "Expected two buffers");

                    const ui8* __restrict validity = static_cast<const ui8*>(array->Data.buffers[0]);
                    const void* dataBuf = array->Data.buffers[1];
                    auto arrayOffset = array->Data.offset;
                    auto arraySize = array->Data.length;

                    TVector<ui32> result;
                    result.yresize(arraySize);

                    auto processColumn = [&] (auto&& addElement) {
                        ProcessNonNullableColumn(
                            validity,
                            arrayOffset,
                            arraySize,
                            "categorical"_sb,
                            std::move(addElement)
                        );
                    };

                    auto typedProcessing = [&] (const auto* __restrict typedDataBuf) {
                        processColumn(
                            [=,&result] (size_t dstObjIdx) {
                                const auto srcObjIdx = dstObjIdx + arrayOffset;
                                result[dstObjIdx] = builderVisitor->GetCatFeatureValue(
                                    flatFeatureIdx,
                                    ToString(typedDataBuf[srcObjIdx])
                                );
                            }
                        );
                    };

                    auto boolProcessing = [&] () {
                        constexpr TStringBuf IdsAsString[2] = {"0"_sb, "1"_sb};
                        const ui32 Hashes[2] = {CalcCatFeatureHash("0"_sb), CalcCatFeatureHash("1"_sb)};
                        bool hasValues[2] = {false, false};

                        const ui8* __restrict data = static_cast<const ui8*>(dataBuf);

                        processColumn(
                            [=,&hasValues, &result] (size_t dstObjIdx) {
                                const auto srcObjIdx = dstObjIdx + arrayOffset;
                                const auto subIdx = srcObjIdx % CHAR_BIT;
                                const bool value = (data[srcObjIdx / CHAR_BIT] >> subIdx) & 1;
                                hasValues[value] = true;
                                result[dstObjIdx] = Hashes[value];
                            }
                        );

                        for (auto i : {0, 1}) {
                            if (hasValues[i]) {
                                builderVisitor->GetCatFeatureValue(flatFeatureIdx, IdsAsString[i]);
                            }
                        }
                    };

                    auto type = TArrowDataTypeIdMapping::Instance().Get(schema->Data);
                    switch (type) {
                        #define HANDLE_TYPE(enumType, cppType)                          \
                            case ESupportedArrowDataType::enumType:                     \
                                typedProcessing(static_cast<const cppType*>(dataBuf));  \
                                break;

                        HANDLE_TYPE(INT8, i8);
                        HANDLE_TYPE(INT16, i16);
                        HANDLE_TYPE(INT32, i32);
                        HANDLE_TYPE(INT64, i64);
                        HANDLE_TYPE(UINT8, ui8);
                        HANDLE_TYPE(UINT16, ui16);
                        HANDLE_TYPE(UINT32, ui32);
                        HANDLE_TYPE(UINT64, ui64);

                        #undef HANDLE_TYPE

                        case NCB::ESupportedArrowDataType::BOOL:
                            boolProcessing();
                            break;
                        default:
                            ythrow TCatBoostException() << "Unsupported Arrow data type: " << schema->Data.format;
                    }

                    builderVisitor->AddCatFeature(
                        flatFeatureIdx,
                        CreateOwningWithMaybeTypeCast<const ui32>(
                            TMaybeOwningArrayHolder<ui32>::CreateOwning(std::move(result))
                        )
                    );

                } catch (...) {
                    ythrow TCatBoostException() << "Error while processing column '" << schema->Data.name << "': "
                        << CurrentExceptionMessage();
                }
            };

            result->push_back(std::async(std::move(process)));
        };

        ProcessArrowArrayStream(
            capsule,
            std::move(callback)
        );
    }


    void AsyncAddArrowTextColumn(
        ui32 flatFeatureIdx,
        PyObject* capsule,
        IRawFeaturesOrderDataVisitor* builderVisitor,
        TVector<std::future<void>>* result
    ) {
        auto callback = [=] (TArrowSchemaHolderPtr schema, TArrowArrayHolderPtr array) {
            auto process = [=] () {
                try {
                    CB_ENSURE(
                        array->Data.null_count <= 0,
                        "Data with nulls is not supported for text columns"
                    );

                    TVector<TString> result(array->Data.length);

                    ProcessNonNullableStringColumn(
                        schema,
                        array,
                        "text",
                        [&, flatFeatureIdx](size_t dstObjIdx, TStringBuf value) {
                            result[dstObjIdx] = TString(value);
                        }
                    );

                    builderVisitor->AddTextFeature(flatFeatureIdx, result);
                } catch (...) {
                    ythrow TCatBoostException() << "Error while processing column '" << schema->Data.name << "': "
                        << CurrentExceptionMessage();
                }
            };

            result->push_back(std::async(std::move(process)));
        };

        ProcessArrowArrayStream(
            capsule,
            std::move(callback)
        );
    }

}
