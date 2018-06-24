#pragma once

#include <catboost/libs/helpers/exception.h>

#include <library/threading/local_executor/local_executor.h>

#include <util/generic/vector.h>
#include <util/system/spinlock.h>
#include <util/system/types.h>


namespace NCB {

    template <class TData>
    class TAsyncRowProcessor {
    public:
        TAsyncRowProcessor(NPar::TLocalExecutor* localExecutor, size_t blockSize)
            : LocalExecutor(localExecutor)
            , BlockSize(blockSize)
            , FirstLineInReadBuffer(false)
            , LinesProcessed(0)
        {
            CB_ENSURE(BlockSize, "TAsyncRowProcessor: blockSize == 0");

            ReadBuffer.yresize(blockSize);
            ParseBuffer.yresize(blockSize);
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
                ReadBufferLock.Release();
            };
            ReadBufferLock.Acquire(); // ensure we hold the lock while the task is being launched
            if (LocalExecutor->GetThreadCount() > 0) {
                LocalExecutor->Exec(readLineBufferLambda, 0, NPar::TLocalExecutor::HIGH_PRIORITY);
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
            with_lock(ReadBufferLock) {
                ReadBuffer.swap(ParseBuffer);
            }
            if (ParseBuffer.size() == BlockSize) { // more data could be available
                ReadBlockAsync(readFunc);
            } else {
                ReadBuffer.resize(0);
            }
            return !!ParseBuffer;
        }

        // processFunc should accept 2 agrs: TData& and lineIdx
        template <class TProcessDataFunc>
        void ProcessBlock(TProcessDataFunc processFunc) {
            const int threadCount = LocalExecutor->GetThreadCount() + 1;

            NPar::TLocalExecutor::TExecRangeParams blockParams(0, ParseBuffer.ysize());
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
    private:
        NPar::TLocalExecutor* LocalExecutor;
        size_t BlockSize;

        TVector<TData> ParseBuffer;

        bool FirstLineInReadBuffer; // if true, first line in ReadBuffer is already filled
        TVector<TData> ReadBuffer;
        TAdaptiveLock ReadBufferLock;

        size_t LinesProcessed;
    };

}
