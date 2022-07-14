#pragma once

#include <catboost/libs/helpers/exception.h>

#include <library/cpp/threading/future/future.h>
#include <library/cpp/threading/local_executor/local_executor.h>

#include <util/generic/vector.h>
#include <util/system/types.h>


namespace NCB {

    /*
     * make sure you call FinishAsyncProcessing before destroying anything that is used inside
     * TAsyncRowProcessor passed by reference in readFunc and processFunc
     * arguments in ReadBlock and ProcessBlock
     */
    template <class TData>
    class TAsyncRowProcessor {
    public:
        TAsyncRowProcessor(NPar::ILocalExecutor* localExecutor, size_t blockSize)
            : LocalExecutor(localExecutor)
            , BlockSize(blockSize)
            , FirstLineInReadBuffer(false)
            , LinesProcessed(0)
        {
            CB_ENSURE(BlockSize, "TAsyncRowProcessor: blockSize == 0");

            ReadBuffer.resize(blockSize);
            ParseBuffer.resize(blockSize);
        }

        ~TAsyncRowProcessor() {
            FinishAsyncProcessing();
        }

        // sometimes we need to separately process first data, but add it to usual processing as well
        void AddFirstLine(TData&& firstLine) {
            CB_ENSURE(!FirstLineInReadBuffer, "TAsyncRowProcessor: double call to AddFirstLine");
            ReadBuffer[0] = std::move(firstLine);
            FirstLineInReadBuffer = true;
        }

        /*
         * readFunc should be of type 'bool(TData* data)',
         *  fill the data and return true if data was read
         */
        template <class TReadDataFunc>
        void ReadBlockAsync(TReadDataFunc readFunc) {
            auto readLineBufferLambda = [this, readFunc = std::move(readFunc)](int) {
                for (size_t lineIdx = (FirstLineInReadBuffer ? 1 : 0); lineIdx < BlockSize; ++lineIdx) {
                    if (!readFunc(&(ReadBuffer[lineIdx]))) {
                        ReadBuffer.yresize(lineIdx);
                        break;
                    }
                }
                FirstLineInReadBuffer = false;
            };
            if (LocalExecutor->GetThreadCount() > 0) {
                auto readFuturesVector = LocalExecutor->ExecRangeWithFutures(
                    readLineBufferLambda,
                    0,
                    1,
                    NPar::TLocalExecutor::HIGH_PRIORITY
                );
                CB_ENSURE(readFuturesVector.size() == 1, "ExecRangeWithFutures returned unexpected number of futures");
                ReadFuture = std::move(readFuturesVector[0]);
            } else {
                readLineBufferLambda(0);
            }
        }

        /*
         * readFunc should be of type 'bool(TData* data)',
         *  fill the data and return true if data was read
         */
        template <class TReadDataFunc>
        bool ReadBlock(TReadDataFunc readFunc) {
            const bool haveReadFuture = ReadFuture.Initialized();
            if (haveReadFuture) { // ReadFuture is not used if there's only one thread
                ReadFuture.GetValueSync(); // will rethrow if there was an exception during read
            }
            ReadBuffer.swap(ParseBuffer);
            if (ParseBuffer.size() == BlockSize) { // more data could be available
                ReadBlockAsync(readFunc);
            } else {
                ReadBuffer.resize(0);
                if (haveReadFuture) {
                    ReadFuture = NThreading::TFuture<void>();
                }
            }
            return !!ParseBuffer;
        }

        // processFunc should accept 2 agrs: TData& and lineIdx
        template <class TProcessDataFunc>
        void ProcessBlock(TProcessDataFunc processFunc) {
            const int threadCount = LocalExecutor->GetThreadCount() + 1;

            NPar::ILocalExecutor::TExecRangeParams blockParams(0, ParseBuffer.ysize());
            blockParams.SetBlockCount(threadCount);
            LocalExecutor->ExecRangeWithThrow([this, blockParams, processFunc = std::move(processFunc)](int blockIdx) {
                const int blockOffset = blockIdx * blockParams.GetBlockSize();
                for (int i = blockOffset; i < Min(blockOffset + blockParams.GetBlockSize(), ParseBuffer.ysize()); ++i) {
                    processFunc(ParseBuffer[i], i);
                }
            }, 0, blockParams.GetBlockCount(), NPar::TLocalExecutor::WAIT_COMPLETE);
            LinesProcessed += ParseBuffer.ysize();
        }


        size_t GetParseBufferSize() const {
            return ParseBuffer.size();
        }

        size_t GetLinesProcessed() const {
            return LinesProcessed;
        }

        /*
         * make sure you call this before destroying anything that is used inside TAsyncRowProcessor
         * passed by reference in readFunc and processFunc
         * arguments in ReadBlock and ProcessBlock
         */
        void FinishAsyncProcessing() {
            // make sure that async reading that uses ReadBuffer has finished
            if (ReadFuture.Initialized()) { // ReadFuture is not used if there's only one thread
                ReadFuture.Wait();
                ReadFuture = NThreading::TFuture<void>();
            }
        }

    private:
        NPar::ILocalExecutor* LocalExecutor;
        size_t BlockSize;

        TVector<TData> ParseBuffer;

        bool FirstLineInReadBuffer; // if true, first line in ReadBuffer is already filled
        TVector<TData> ReadBuffer;
        NThreading::TFuture<void> ReadFuture;

        size_t LinesProcessed;
    };

}
