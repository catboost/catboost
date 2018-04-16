#include "stdafx.h"
#include "ib_mem.h"
#include "ib_memstream.h"
#include "ib_low.h"

namespace NNetliba_v12 {
    int TIBMemStream::WriteImpl(const void* userBuffer, int sizeArg) {
        const char* srcData = (const char*)userBuffer;
        int size = sizeArg;
        for (;;) {
            if (size == 0)
                return sizeArg;
            if (CurBlock == Blocks.ysize()) {
                // add new block
                TBlock& blk = Blocks.emplace_back();
                blk.StartOffset = GetLength();
                int szLog = 17 + Min(Blocks.ysize() / 2, 13);
                blk.BufSize = 1 << szLog;
                blk.DataSize = 0;
                blk.Mem = MemPool->Alloc(blk.BufSize);
                Y_ASSERT(CurBlockOffset == 0);
            }
            TBlock& curBlk = Blocks[CurBlock];
            int leftSpace = curBlk.BufSize - CurBlockOffset;
            int copySize = Min(size, leftSpace);
            memcpy(curBlk.Mem->GetData() + CurBlockOffset, srcData, copySize);
            size -= copySize;
            CurBlockOffset += copySize;
            srcData += copySize;
            curBlk.DataSize = Max(curBlk.DataSize, CurBlockOffset);
            if (CurBlockOffset == curBlk.BufSize) {
                ++CurBlock;
                CurBlockOffset = 0;
            }
        }
    }

    int TIBMemStream::ReadImpl(void* userBuffer, int sizeArg) {
        char* dstData = (char*)userBuffer;
        int size = sizeArg;
        for (;;) {
            if (size == 0)
                return sizeArg;
            if (CurBlock == Blocks.ysize()) {
                //memset(dstData, 0, size);
                size = 0;
                continue;
            }
            TBlock& curBlk = Blocks[CurBlock];
            int leftSpace = curBlk.DataSize - CurBlockOffset;
            int copySize = Min(size, leftSpace);
            memcpy(dstData, curBlk.Mem->GetData() + CurBlockOffset, copySize);
            size -= copySize;
            CurBlockOffset += copySize;
            dstData += copySize;
            if (CurBlockOffset == curBlk.DataSize) {
                ++CurBlock;
                CurBlockOffset = 0;
            }
        }
    }

    i64 TIBMemStream::GetLength() {
        i64 res = 0;
        for (int i = 0; i < Blocks.ysize(); ++i) {
            res += Blocks[i].DataSize;
        }
        return res;
    }

    i64 TIBMemStream::Seek(i64 pos) {
        for (int resBlockId = 0; resBlockId < Blocks.ysize(); ++resBlockId) {
            const TBlock& blk = Blocks[resBlockId];
            if (pos < blk.StartOffset + blk.DataSize) {
                CurBlock = resBlockId;
                CurBlockOffset = pos - blk.StartOffset;
                return pos;
            }
        }
        CurBlock = Blocks.ysize();
        CurBlockOffset = 0;
        return GetLength();
    }

    void TIBMemStream::GetBlocks(TVector<TBlockDescr>* res) const {
        int blockCount = Blocks.ysize();
        res->resize(blockCount);
        for (int i = 0; i < blockCount; ++i) {
            const TBlock& blk = Blocks[i];
            TBlockDescr& dst = (*res)[i];
            dst.Addr = blk.Mem->GetAddr();
            dst.BufSize = blk.BufSize;
            dst.DataSize = blk.DataSize;
            TMemoryRegion* mem = blk.Mem->GetMemRegion();
            dst.LocalKey = mem->GetLKey();
            dst.RemoteKey = mem->GetRKey();
        }
    }

    void TIBMemStream::CreateBlocks(const TVector<TBlockSizes>& arr) {
        int blockCount = arr.ysize();
        Blocks.resize(blockCount);
        i64 offset = 0;
        for (int i = 0; i < blockCount; ++i) {
            const TBlockSizes& src = arr[i];
            TBlock& blk = Blocks[i];
            blk.BufSize = src.BufSize;
            blk.DataSize = src.DataSize;
            blk.Mem = MemPool->Alloc(blk.BufSize);
            blk.StartOffset = offset;
            offset += blk.DataSize;
        }
        CurBlock = 0;
        CurBlockOffset = 0;
    }

    void TIBMemStream::Clear() {
        Blocks.resize(0);
        CurBlock = 0;
        CurBlockOffset = 0;
    }
}
