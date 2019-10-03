#pragma once

#include "ctr_kernels.h"

#include <util/system/types.h>
#include <catboost/cuda/cuda_util/reorder_bins.h>
#include <catboost/cuda/cuda_util/helpers.h>
#include <catboost/cuda/cuda_util/fill.h>
#include <catboost/cuda/cuda_util/compression_helpers_gpu.h>
#include <catboost/cuda/cuda_util/algorithm.h>

#include <catboost/private/libs/ctr_description/ctr_config.h>
#include <catboost/libs/helpers/math_utils.h>

namespace NCatboostCuda {
    template <class TMapping>
    class TCtrBinBuilder {
    public:
        using TVisitor = std::function<void(const NCB::TCtrConfig&, const TCudaBuffer<float, TMapping>&, ui32 stream)>;

        explicit TCtrBinBuilder(ui32 stream = 0)
            : Stream(stream)
        {
        }

        TCtrBinBuilder& SetIndices(const TCudaBuffer<ui32, TMapping>& learn,
                                   const TCudaBuffer<ui32, TMapping>& test) {
            return SetIndices(learn, &test);
        }

        template <class TUi32>
        TCtrBinBuilder& SetIndices(const TCudaBuffer<TUi32, TMapping>& learn,
                                   const TCudaBuffer<TUi32, TMapping>* test = nullptr) {
            LearnSlice = learn.GetMapping().GetObjectsSlice();

            TSlice allIndices = LearnSlice;
            if (test) {
                allIndices.Right += test->GetMapping().GetObjectsSlice().Size();
                TestSlice = TSlice::Remove(allIndices, LearnSlice)[0];
            }

            Indices.Reset(learn.GetMapping().RepeatOnAllDevices(allIndices.Size()));
            Indices.SliceView(LearnSlice).Copy(learn, Stream);

            if (test && TestSlice.Size()) {
                auto testIndices = Indices.SliceView(TestSlice);
                testIndices.Copy(*test, Stream);
                AddVector(testIndices, static_cast<ui32>(LearnSlice.Size()), Stream);
            }

            Reset();
            return *this;
        }

        TCtrBinBuilder& SetIndices(TCudaBuffer<ui32, TMapping>&& learn,
                                   TSlice learnSlice) {
            Indices = std::move(learn);
            LearnSlice = learnSlice;
            auto rest = TSlice::Remove(Indices.GetObjectsSlice(), learnSlice);
            if (rest.size()) {
                CB_ENSURE(rest.size() == 1);
                TestSlice = rest[0];
            }
            Reset();
            return *this;
        }

        template <class TUi32>
        TCtrBinBuilder& SetIndices(const TCudaBuffer<TUi32, TMapping>& indices,
                                   TSlice learnSlice) {
            Indices.Reset(indices.GetMapping());
            Indices.Copy(indices, Stream);
            LearnSlice = learnSlice;
            auto rest = TSlice::Remove(Indices.GetObjectsSlice(), learnSlice);
            if (rest.size()) {
                CB_ENSURE(rest.size() == 1);
                TestSlice = rest[0];
            }
            Reset();
            return *this;
        }

        const TCudaBuffer<ui32, TMapping>& GetIndices() const {
            return Indices;
        };

        const TSlice& GetLearnSlice() const {
            return LearnSlice;
        }

        const TSlice& GetTestSlice() const {
            return TestSlice;
        }

        TCudaBuffer<ui32, TMapping> MoveIndices() {
            return std::move(Indices);
        };

        ui32 GetStream() const {
            return Stream;
        }

        template <class TUi64, NCudaLib::EPtrType PtrType>
        TCtrBinBuilder& AddCompressedBins(const TCudaBuffer<TUi64, TMapping, PtrType>& compressedLearn,
                                          ui32 uniqueValues) {
            CB_ENSURE(TestSlice.Size() == 0);
            AddLearnBins(compressedLearn, uniqueValues);
            ProceedNewBins(uniqueValues);
            return *this;
        };

        template <class TUi64, NCudaLib::EPtrType PtrType>
        TCtrBinBuilder& AddCompressedBinsWithCurrentBinsCache(const TCudaBuffer<ui32, TMapping>& currentBins,
                                                              const TCudaBuffer<TUi64, TMapping, PtrType>& compressedLearn,
                                                              ui32 uniqueValues) {
            CB_ENSURE(TestSlice.Size() == 0);
            if (PtrType == NCudaLib::EPtrType::CudaDevice) {
                ProceedCompressedBins(uniqueValues, compressedLearn, currentBins);
            } else {
                AddLearnBins(compressedLearn, uniqueValues);
                ProceedNewBins(uniqueValues, currentBins);
            }
            return *this;
        };

        template <NCudaLib::EPtrType Type>
        TCtrBinBuilder& AddCompressedBins(const TCudaBuffer<ui64, TMapping, Type>& compressedLearn,
                                          const TCudaBuffer<ui64, TMapping, Type>& compressedTest,
                                          ui32 uniqueValues) {
            AddLearnBins(compressedLearn, uniqueValues);
            AddTestBins(compressedTest, uniqueValues);
            ProceedNewBins(uniqueValues);
            return *this;
        };

        template <class TUi32>
        static void ComputeCurrentBins(const TCudaBuffer<TUi32, TMapping>& indices,
                                       TCudaBuffer<ui32, TMapping>& tmp,
                                       TCudaBuffer<ui32, TMapping>& dst,
                                       ui32 stream) {
            dst.Reset(indices.GetMapping());
            tmp.Reset(indices.GetMapping());
            ExtractMask(indices, dst, false, stream);
            ScanVector(dst, tmp, false, stream);
            ScatterWithMask(dst, tmp, indices, Mask, stream);
        }

        static constexpr ui32 GetMask() {
            return Mask;
        }

        //returns copy reference
        TCtrBinBuilder<TMapping>& CopyTo(TCtrBinBuilder<TMapping>& copy) const {
            copy.TestSlice = TestSlice;
            copy.LearnSlice = LearnSlice;

            copy.Indices.Reset(Indices.GetMapping());
            copy.Indices.Copy(Indices, Stream);

            copy.Bins.Reset(Indices.GetMapping());
            copy.DecompressedTempBins.Reset(Indices.GetMapping());
            copy.Tmp.Reset(Indices.GetMapping());
            copy.Stream = Stream;

            return copy;
        }

        //this function compute pure freq, not weighted one like binFreqCalcer. As a result, it much faster
        TCtrBinBuilder<TMapping>& VisitEqualUpToPriorFreqCtrs(const TVector<NCB::TCtrConfig>& ctrConfigs,
                                                              TVisitor& visitor) {
            //TODO(noxoomo): change tempFlags to ui8
            Tmp.Reset(Indices.GetMapping());
            Bins.Reset(Indices.GetMapping());
            ExtractMask(Indices, Tmp, false, Stream);
            ScanVector(Tmp, Bins, false, Stream);
            UpdatePartitionOffsets(Bins, Tmp, Stream);

            for (auto& ctrConfig : ctrConfigs) {
                CB_ENSURE(ctrConfig.Type == ECtrType::FeatureFreq);
                CB_ENSURE(ctrConfig.Prior.size() == 2);
                const float prior = GetNumeratorShift(ctrConfig);
                const float observationsPrior = GetDenumeratorShift(ctrConfig);
                auto dst = DecompressedTempBins.template ReinterpretCast<float>();
                ComputeNonWeightedBinFreqCtr(Indices, Bins, Tmp,
                                             prior, observationsPrior, dst,
                                             Stream);
                visitor(ctrConfig, dst, Stream);
            }
            return *this;
        }

    private:
        void Reset() {
            Bins.Reset(Indices.GetMapping());
            DecompressedTempBins.Reset(Indices.GetMapping());
            Tmp.Reset(Indices.GetMapping());
        }

        template <class TUi64, NCudaLib::EPtrType Type>
        TCtrBinBuilder& AddLearnBins(const TCudaBuffer<TUi64, TMapping, Type>& compressedLearn,
                                     ui32 uniqueValues) {
            auto learnBinsSlice = DecompressedTempBins.SliceView(LearnSlice);
            Decompress(compressedLearn, learnBinsSlice, uniqueValues, Stream);
            return *this;
        }

        template <class TUi64, NCudaLib::EPtrType Type>
        TCtrBinBuilder& AddTestBins(const TCudaBuffer<TUi64, TMapping, Type>& compressedTest,
                                    ui32 uniqueValues) {
            if (TestSlice.Size()) {
                auto testBinsSlice = DecompressedTempBins.SliceView(TestSlice);
                Decompress(compressedTest, testBinsSlice, uniqueValues, Stream);
            }
            return *this;
        }

        void ProceedNewBins(ui32 uniqueValues,
                            const TCudaBuffer<ui32, TMapping>& currentBins) {
            AssertTempBuffersInitialized();
            GatherWithMask(Bins, DecompressedTempBins, Indices, Mask, Stream);
            const ui32 newBits = NCB::IntLog2(uniqueValues);
            ReorderBins(Bins, Indices, 0, newBits, Tmp, DecompressedTempBins, Stream);
            UpdateBordersMask(Bins, currentBins, Indices, Stream);
        }

        template <NCudaLib::EPtrType Type>
        void ProceedCompressedBins(ui32 uniqueValues,
                                   const TCudaBuffer<ui64, TMapping, Type>& binsCompressed,
                                   const TCudaBuffer<ui32, TMapping>& currentBins) {
            AssertTempBuffersInitialized();
            const ui32 newBits = NCB::IntLog2(uniqueValues);
            GatherFromCompressed(binsCompressed, uniqueValues, Indices, Mask, Bins, Stream);
            ReorderBins(Bins, Indices, 0, newBits, Tmp, DecompressedTempBins, Stream);
            UpdateBordersMask(Bins, currentBins, Indices, Stream);
        }

        void ProceedNewBins(ui32 uniqueValues) {
            {
                auto& tmp = Bins;
                ComputeCurrentBins(Indices, tmp, CurrentBins, Stream);
            }
            ProceedNewBins(uniqueValues, CurrentBins);
        }

        void AssertTempBuffersInitialized() {
            Y_ASSERT(Indices.GetObjectsSlice().Size());
            Y_ASSERT(Indices.GetObjectsSlice() == Bins.GetObjectsSlice());
            Y_ASSERT(Indices.GetObjectsSlice() == DecompressedTempBins.GetObjectsSlice());
            Y_ASSERT(Tmp.GetObjectsSlice() == DecompressedTempBins.GetObjectsSlice());
        }

        TCudaBuffer<ui32, TMapping> Indices;
        TCudaBuffer<ui32, TMapping> Bins;
        TCudaBuffer<ui32, TMapping> DecompressedTempBins;
        TCudaBuffer<ui32, TMapping> CurrentBins;
        TCudaBuffer<ui32, TMapping> Tmp;

        TSlice LearnSlice;
        TSlice TestSlice;

        static constexpr ui32 Mask = 0x3FFFFFFF;
        ui32 Stream;

        friend TCtrBinBuilder<NCudaLib::TSingleMapping> CreateBinBuilderForSingleDevice(const TCtrBinBuilder<NCudaLib::TMirrorMapping>& mirrorBuilder,
                                                                                        ui32 deviceId,
                                                                                        ui32 streamId);
    };

    //stripe mapping can't be used for bin-tracking
    template <>
    class TCtrBinBuilder<NCudaLib::TStripeMapping>;

    extern template class TCtrBinBuilder<NCudaLib::TMirrorMapping>;
    extern template class TCtrBinBuilder<NCudaLib::TSingleMapping>;

    TCtrBinBuilder<NCudaLib::TSingleMapping> CreateBinBuilderForSingleDevice(const TCtrBinBuilder<NCudaLib::TMirrorMapping>& mirrorBuilder,
                                                                             ui32 deviceId,
                                                                             ui32 streamId = 0);

}
