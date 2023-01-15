#include "stdafx.h"
#include "ib_mem.h"
#include "ib_low.h"
#include "cpu_affinity.h"

#if defined(_unix_)
#include <pthread.h>
#endif

namespace NNetliba_v12 {
    TIBMemSuperBlock::TIBMemSuperBlock(TIBMemPool* pool, size_t szLog)
        : Pool(pool)
        , SzLog(szLog)
        , UseCount(0)
    {
        size_t sz = GetSize();
        MemRegion = new TMemoryRegion(pool->GetIBContext(), sz);
        //printf("Alloc super block, size %" PRId64 "\n", sz);
    }

    TIBMemSuperBlock::~TIBMemSuperBlock() {
        Y_ASSERT(AtomicGet(UseCount) == 0);
    }

    char* TIBMemSuperBlock::GetData() {
        return MemRegion->GetData();
    }

    void TIBMemSuperBlock::DecRef() {
        if (AtomicAdd(UseCount, -1) == 0) {
            Pool->Return(this);
        }
    }

    TIBMemBlock::~TIBMemBlock() {
        if (Super.Get()) {
            Super->DecRef();
        } else {
            delete[] Data;
        }
    }

    //////////////////////////////////////////////////////////////////////////
    TIBMemPool::TIBMemPool(TPtrArg<TIBContext> ctx)
        : IBCtx(ctx)
        , AllocCacheSize(0)
        , CurrentOffset(IB_MEM_LARGE_BLOCK)
        , WorkThread(TThread::TParams(ThreadFunc, (void*)this).SetName("nl12_ib_mem"))
        , KeepRunning(true)
    {
        WorkThread.Start();
        HasStarted.Wait();
    }

    TIBMemPool::~TIBMemPool() {
        Y_ASSERT(WorkThread.Running());
        KeepRunning = false;
        HasWork.Signal();
        WorkThread.Join();
        {
            TJobItem* work = nullptr;
            while (Requests.Dequeue(&work)) {
                delete work;
            }
        }
    }

    TIntrusivePtr<TIBMemSuperBlock> TIBMemPool::AllocSuper(size_t szArg) {
        // assume CacheLock is taken
        size_t szLog = 12;
        while ((((size_t)1) << szLog) < szArg) {
            ++szLog;
        }
        TIntrusivePtr<TIBMemSuperBlock> super;
        {
            TVector<TIntrusivePtr<TIBMemSuperBlock>>& cc = AllocCache[szLog];
            if (!cc.empty()) {
                super = cc.back();
                cc.resize(cc.size() - 1);
                AllocCacheSize -= 1ll << super->SzLog;
            }
        }
        if (super.Get() == nullptr) {
            super = new TIBMemSuperBlock(this, szLog);
        }
        return super;
    }

    TIBMemBlock* TIBMemPool::Alloc(size_t sz) {
        TGuard<TMutex> gg(CacheLock);
        if (sz > IB_MEM_LARGE_BLOCK) {
            TIntrusivePtr<TIBMemSuperBlock> super = AllocSuper(sz);
            return new TIBMemBlock(super, super->GetData(), sz);
        } else {
            if (CurrentOffset + sz > IB_MEM_LARGE_BLOCK) {
                CurrentBlk.Assign(AllocSuper(IB_MEM_LARGE_BLOCK));
                CurrentOffset = 0;
            }
            CurrentOffset += sz;
            return new TIBMemBlock(CurrentBlk.Get(), CurrentBlk.Get()->GetData() + CurrentOffset - sz, sz);
        }
    }

    void TIBMemPool::Return(TPtrArg<TIBMemSuperBlock> blk) {
        TGuard<TMutex> gg(CacheLock);
        Y_ASSERT(AtomicGet(blk->UseCount) == 0);
        size_t sz = 1ull << blk->SzLog;
        if (sz + AllocCacheSize > IB_MEM_POOL_SIZE) {
            AllocCache.clear();
            AllocCacheSize = 0;
        }
        {
            TVector<TIntrusivePtr<TIBMemSuperBlock>>& cc = AllocCache[blk->SzLog];
            cc.push_back(blk.Get());
            AllocCacheSize += sz;
        }
    }

    void* TIBMemPool::ThreadFunc(void* param) {
        BindToSocket(0);
        SetHighestThreadPriority();
        TIBMemPool* pThis = (TIBMemPool*)param;
        pThis->HasStarted.Signal();

        while (pThis->KeepRunning) {
            TJobItem* work = nullptr;
            if (!pThis->Requests.Dequeue(&work)) {
                pThis->HasWork.Reset();
                if (!pThis->Requests.Dequeue(&work)) {
                    pThis->HasWork.Wait();
                }
            }
            if (work) {
                //printf("mem copy got work\n");
                int sz = work->Data->GetSize();
                work->Block = pThis->Alloc(sz);
                TBlockChainIterator bc(work->Data->GetChain());
                bc.Read(work->Block->GetData(), sz);
                TIntrusivePtr<TCopyResultStorage> dst = work->ResultStorage;
                work->ResultStorage = nullptr;
                dst->Results.Enqueue(work);
                //printf("mem copy completed\n");
            }
        }
        return nullptr;
    }

    //////////////////////////////////////////////////////////////////////////
    static TMutex IBMemMutex;
    static TIntrusivePtr<TIBMemPool> IBMemPool;
    static bool IBWasInitialized;

    TIntrusivePtr<TIBMemPool> GetIBMemPool() {
        TGuard<TMutex> gg(IBMemMutex);
        if (IBWasInitialized) {
            return IBMemPool;
        }
        IBWasInitialized = true;

        TIntrusivePtr<TIBPort> ibPort = GetIBDevice();
        if (ibPort.Get() == nullptr) {
            return nullptr;
        }
        IBMemPool = new TIBMemPool(ibPort->GetCtx());
        return IBMemPool;
    }
}
