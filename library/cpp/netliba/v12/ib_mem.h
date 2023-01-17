#pragma once

#include "block_chain.h"
#include <util/thread/lfqueue.h>
#include <util/system/thread.h>

namespace NNetliba_v12 {
    // registered memory blocks
    class TMemoryRegion;
    class TIBContext;

    class TIBMemPool;
    struct TIBMemSuperBlock: public TThrRefBase, TNonCopyable {
        TIntrusivePtr<TIBMemPool> Pool;
        size_t SzLog;
        TAtomic UseCount;
        TIntrusivePtr<TMemoryRegion> MemRegion;

        TIBMemSuperBlock(TIBMemPool* pool, size_t szLog);
        ~TIBMemSuperBlock() override;
        char* GetData();
        size_t GetSize() {
            return ((ui64)1) << SzLog;
        }
        void IncRef() {
            AtomicAdd(UseCount, 1);
        }
        void DecRef();
    };

    class TIBMemBlock: public TThrRefBase, TNonCopyable {
        TIntrusivePtr<TIBMemSuperBlock> Super;
        char* Data;
        size_t Size;

        ~TIBMemBlock() override;

    public:
        TIBMemBlock(TPtrArg<TIBMemSuperBlock> super, char* data, size_t sz)
            : Super(super)
            , Data(data)
            , Size(sz)
        {
            Super->IncRef();
        }
        TIBMemBlock(size_t sz)
            : Super(nullptr)
            , Size(sz)
        {
            // not really IB mem block, but useful IB code debug without IB
            Data = new char[sz];
        }
        char* GetData() {
            return Data;
        }
        ui64 GetAddr() {
            return reinterpret_cast<ui64>(Data) / sizeof(char);
        }
        size_t GetSize() {
            return Size;
        }
        TMemoryRegion* GetMemRegion() {
            return Super.Get() ? Super->MemRegion.Get() : nullptr;
        }
    };

    const size_t IB_MEM_LARGE_BLOCK_LN = 20;
    const size_t IB_MEM_LARGE_BLOCK = 1ul << IB_MEM_LARGE_BLOCK_LN;
    const size_t IB_MEM_POOL_SIZE = 1024 * 1024 * 1024;

    class TIBMemPool: public TThrRefBase, TNonCopyable {
    public:
        struct TCopyResultStorage;

    private:
        class TIBMemSuperBlockPtr {
            TIntrusivePtr<TIBMemSuperBlock> Blk;

        public:
            ~TIBMemSuperBlockPtr() {
                Detach();
            }
            void Assign(TIntrusivePtr<TIBMemSuperBlock> p) {
                Detach();
                Blk = p;
                if (p.Get()) {
                    AtomicAdd(p->UseCount, 1);
                }
            }
            void Detach() {
                if (Blk.Get()) {
                    Blk->DecRef();
                    Blk = nullptr;
                }
            }
            TIBMemSuperBlock* Get() {
                return Blk.Get();
            }
        };

        TIntrusivePtr<TIBContext> IBCtx;
        THashMap<size_t, TVector<TIntrusivePtr<TIBMemSuperBlock>>> AllocCache;
        size_t AllocCacheSize;
        TIBMemSuperBlockPtr CurrentBlk;
        int CurrentOffset;
        TMutex CacheLock;
        TThread WorkThread;
        TSystemEvent HasStarted;
        bool KeepRunning;

        struct TJobItem {
            TRopeDataPacket* Data;
            i64 MsgHandle;
            TIntrusivePtr<TThrRefBase> Context;
            TIntrusivePtr<TIBMemBlock> Block;
            TIntrusivePtr<TCopyResultStorage> ResultStorage;

            TJobItem(TRopeDataPacket* data, i64 msgHandle, TThrRefBase* context, TPtrArg<TCopyResultStorage> dst)
                : Data(data)
                , MsgHandle(msgHandle)
                , Context(context)
                , ResultStorage(dst)
            {
            }
        };

        TLockFreeQueue<TJobItem*> Requests;
        TSystemEvent HasWork;

        static void* ThreadFunc(void* param);

        void Return(TPtrArg<TIBMemSuperBlock> blk);
        TIntrusivePtr<TIBMemSuperBlock> AllocSuper(size_t sz);
        ~TIBMemPool() override;

    public:
        struct TCopyResultStorage: public TThrRefBase {
            TLockFreeStack<TJobItem*> Results;

            ~TCopyResultStorage() override {
                TJobItem* work;
                while (Results.Dequeue(&work)) {
                    delete work;
                }
            }
            template <class T>
            bool GetCopyResult(TIntrusivePtr<TIBMemBlock>* resBlock, i64* resMsgHandle, TIntrusivePtr<T>* context) {
                TJobItem* work;
                if (Results.Dequeue(&work)) {
                    *resBlock = work->Block;
                    *resMsgHandle = work->MsgHandle;
                    *context = static_cast<T*>(work->Context.Get()); // caller responsibility to make sure this makes sense
                    delete work;
                    return true;
                } else {
                    return false;
                }
            }
        };

    public:
        TIBMemPool(TPtrArg<TIBContext> ctx);
        TIBContext* GetIBContext() {
            return IBCtx.Get();
        }
        TIBMemBlock* Alloc(size_t sz);

        void CopyData(TRopeDataPacket* data, i64 msgHandle, TThrRefBase* context, TPtrArg<TCopyResultStorage> dst) {
            Requests.Enqueue(new TJobItem(data, msgHandle, context, dst));
            HasWork.Signal();
        }

        friend class TIBMemBlock;
        friend struct TIBMemSuperBlock;
    };

    extern TIntrusivePtr<TIBMemPool> GetIBMemPool();
}
