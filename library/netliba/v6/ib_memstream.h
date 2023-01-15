#pragma once

#include "ib_mem.h"
#include <library/cpp/binsaver/bin_saver.h>
#include <library/cpp/binsaver/buffered_io.h>

namespace NNetliba {
    class TIBMemStream: public IBinaryStream {
        struct TBlock {
            TIntrusivePtr<TIBMemBlock> Mem;
            i64 StartOffset;
            int BufSize, DataSize;

            TBlock()
                : StartOffset(0)
                , BufSize(0)
                , DataSize(0)
            {
            }
            TBlock(const TBlock& x) {
                Copy(x);
            }
            void operator=(const TBlock& x) {
                Copy(x);
            }
            void Copy(const TBlock& x) {
                if (x.BufSize > 0) {
                    Mem = GetIBMemPool()->Alloc(x.BufSize);
                    memcpy(Mem->GetData(), x.Mem->GetData(), x.DataSize);
                    StartOffset = x.StartOffset;
                    BufSize = x.BufSize;
                    DataSize = x.DataSize;
                } else {
                    Mem = nullptr;
                    StartOffset = 0;
                    BufSize = 0;
                    DataSize = 0;
                }
            }
        };

        TIntrusivePtr<TIBMemPool> MemPool;
        TVector<TBlock> Blocks;
        int CurBlock;
        int CurBlockOffset;

    public:
        struct TBlockDescr {
            ui64 Addr;
            int BufSize, DataSize;
            ui32 RemoteKey, LocalKey;
        };
        struct TBlockSizes {
            int BufSize, DataSize;
        };

    public:
        TIBMemStream()
            : MemPool(GetIBMemPool())
            , CurBlock(0)
            , CurBlockOffset(0)
        {
        }
        ~TIBMemStream() override {
        } // keep gcc happy

        bool IsValid() const override {
            return true;
        }
        bool IsFailed() const override {
            return false;
        }
        void Flush() {
        }

        i64 GetLength();
        i64 Seek(i64 pos);

        void GetBlocks(TVector<TBlockDescr>* res) const;
        void CreateBlocks(const TVector<TBlockSizes>& arr);

        void Clear();

    private:
        int WriteImpl(const void* userBuffer, int size) override;
        int ReadImpl(void* userBuffer, int size) override;
    };

    template <class T>
    inline void Serialize(bool bRead, TIBMemStream& ms, T& c) {
        IBinSaver bs(ms, bRead);
        bs.Add(1, &c);
    }

}
