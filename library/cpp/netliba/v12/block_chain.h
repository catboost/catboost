#pragma once

#include <util/system/shmat.h>
#include <util/system/defaults.h>
#include <util/generic/algorithm.h>
#include <util/generic/list.h>
#include <util/generic/ptr.h>
#include <util/generic/string.h>
#include <util/generic/vector.h>
#include <library/cpp/netliba/socket/allocator.h>
#include "udp_recv_packet.h"
#include "posix_shared_memory.h"
#include <util/generic/noncopyable.h>

namespace NNetliba_v12 {
    class TBlockChain {
    public:
        struct TBlock {
            const char* Data;
            int Offset, Size; // Offset in whole chain

            TBlock()
                : Data(nullptr)
                , Offset(0)
                , Size(0)
            {
            }
            TBlock(const char* data, int offset, int sz)
                : Data(data)
                , Offset(offset)
                , Size(sz)
            {
            }
        };

    private:
        TVector<TBlock, TCustomAllocator<TBlock>> Blocks;
        int Size;
        struct TBlockLess {
            bool operator()(const TBlock& b, int offset) const {
                return b.Offset < offset;
            }
        };

    public:
        TBlockChain()
            : Size(0)
        {
        }
        void AddBlock(const void* data, int sz) {
            Blocks.emplace_back((const char*)data, Size, sz);
            Size += sz;
        }
        int GetSize() const {
            return Size;
        }
        const TBlock& GetBlock(int i) const {
            return Blocks[i];
        }
        int GetBlockCount() const {
            return Blocks.ysize();
        }
        int GetBlockIdByOffset(int offset) const {
            const auto i = LowerBound(Blocks.cbegin(), Blocks.cend(), offset, TBlockLess());
            if (i == Blocks.cend())
                return Blocks.ysize() - 1;
            if (i->Offset == offset)
                return (int)(i - Blocks.cbegin());
            return (int)(i - Blocks.cbegin() - 1);
        }
    };

    //////////////////////////////////////////////////////////////////////////

    class TBlockChainIterator {
        const TBlockChain& Chain;
        int Pos, BlockPos, BlockId;
        bool Failed;

    public:
        TBlockChainIterator(const TBlockChain& chain)
            : Chain(chain)
            , Pos(0)
            , BlockPos(0)
            , BlockId(0)
            , Failed(false)
        {
        }
        void Read(void* dst, int sz) {
            char* dstBuf = (char*)dst;
            while (sz > 0) {
                if (BlockId >= Chain.GetBlockCount()) {
                    // JACKPOT!
                    fprintf(stderr, "reading beyond chain end: BlockId %d, Chain.GetBlockCount() %d, Pos %d, BlockPos %d\n",
                            BlockId, Chain.GetBlockCount(), Pos, BlockPos);
                    Y_ASSERT(0 && "reading beyond chain end");
                    memset(dstBuf, 0, sz);
                    Failed = true;
                    return;
                }
                const TBlockChain::TBlock& blk = Chain.GetBlock(BlockId);
                int copySize = Min(blk.Size - BlockPos, sz);
                memcpy(dstBuf, blk.Data + BlockPos, copySize);
                dstBuf += copySize;
                Pos += copySize;
                BlockPos += copySize;
                sz -= copySize;
                if (BlockPos == blk.Size) {
                    BlockPos = 0;
                    ++BlockId;
                }
            }
        }
        void Seek(int pos) {
            if (pos < 0 || pos > Chain.GetSize()) {
                Y_ASSERT(0);
                Pos = 0;
                BlockPos = 0;
                BlockId = 0;
                return;
            }
            BlockId = Chain.GetBlockIdByOffset(pos);
            const TBlockChain::TBlock& blk = Chain.GetBlock(BlockId);
            Pos = pos;
            BlockPos = Pos - blk.Offset;
        }
        int GetPos() const {
            return Pos;
        }
        int GetSize() const {
            return Chain.GetSize();
        }
        bool HasFailed() const {
            return Failed;
        }
        void Fail() {
            Failed = true;
        }
    };

    //////////////////////////////////////////////////////////////////////////
    using TDataVector = TVector<char>;

    class TRopeDataPacket: public TNonCopyable, public TWithCustomAllocator {
        TBlockChain Chain;
        TVector<char*, TCustomAllocator<char*>> Buf;
        char *Block, *BlockEnd;
        TList<TDataVector, TCustomAllocator<TDataVector>> DataVectors;
        TIntrusivePtr<TPosixSharedMemory> SharedData;
        TVector<TIntrusivePtr<TThrRefBase>, TCustomAllocator<TIntrusivePtr<TThrRefBase>>> AttachedStorage;
        TVector<TUdpRecvPacket*, TCustomAllocator<TUdpRecvPacket*>> PacketsStorage;
        char DefaultBuf[128]; // prevent allocs in most cases

        static constexpr int N_DEFAULT_BLOCK_SIZE = 1024;

        char* Alloc(int sz) {
            char* res = nullptr;
            if (BlockEnd - Block < sz) {
                int bufSize = Max((int)N_DEFAULT_BLOCK_SIZE, sz);
                char* newBlock = AllocBuf(bufSize);
                Block = newBlock;
                BlockEnd = Block + bufSize;
                Buf.push_back(newBlock);
            }
            res = Block;
            Block += sz;
            Y_ASSERT(Block <= BlockEnd);
            return res;
        }

        static char* AllocBuf(int sz) {
            return TCustomAllocator<char>().allocate(sz);
        }
        static void FreeBuf(char* buf) {
            return TCustomAllocator<char>().deallocate(buf, 0U);
        }

    public:
        TRopeDataPacket()
            : Block(DefaultBuf)
            , BlockEnd(DefaultBuf + Y_ARRAY_SIZE(DefaultBuf))
        {
        }
        ~TRopeDataPacket() {
            for (const auto& buf : Buf) {
                FreeBuf(buf);
            }
            for (const auto& ps : PacketsStorage) {
                delete ps;
            }
        }

        // buf - pointer to buffer which will be freed with FreeBuf()
        // data - pointer to data start within buf
        // sz - size of useful data
        //void AddBlock(char *buf, const char *data, int sz)
        //{
        //    Buf.push_back(buf);
        //    Chain.AddBlock(data, sz);
        //}
        void AddBlock(TUdpRecvPacket* buf, const char* data, int sz) {
            PacketsStorage.push_back(buf);
            Chain.AddBlock(data, sz);
        }
        void AddBlock(TThrRefBase* buf, const char* data, int sz) {
            AttachedStorage.push_back(buf);
            Chain.AddBlock(data, sz);
        }
        //
        void Write(const void* data, int sz) {
            char* buf = Alloc(sz);
            memcpy(buf, data, sz);
            Chain.AddBlock(buf, sz);
        }
        template <class T>
        void Write(const T& data) {
            Write(&data, sizeof(T));
        }
        //// caller guarantees that data will persist all *this lifetime
        //// int this case so we don`t have to copy data to locally held buffer
        //template<class T>
        //void WriteNoCopy(const T *data)
        //{
        //    Chain.AddBlock(data, sizeof(T));
        //}
        void WriteNoCopy(const char* buf, int sz) {
            Write(sz);
            Chain.AddBlock(buf, sz);
        }
        // write some array like TVector<>
        //template<class T>
        //void WriteArr(const T &sz)
        //{
        //    int n = (int)sz.size();
        //    Write(n);
        //    if (n > 0)
        //        Write(&sz[0], n * sizeof(sz[0]));
        //}
        void WriteStroka(const TString& sz) {
            int n = (int)sz.size();
            Write(n);
            if (n > 0)
                Write(sz.c_str(), n * sizeof(sz[0]));
        }
        // will take *data ownership, saves copy
        void WriteDestructive(TDataVector* data) {
            int n = data ? data->ysize() : 0;
            Write(n);
            if (n > 0) {
                auto& local = DataVectors.emplace_back(std::move(*data));
                Chain.AddBlock(&local[0], local.ysize());
            }
        }
        void AttachSharedData(TIntrusivePtr<TPosixSharedMemory> shm) {
            SharedData = shm;
        }
        TPosixSharedMemory* GetSharedData() const {
            return SharedData.Get();
        }
        const TBlockChain& GetChain() {
            return Chain;
        }
        int GetSize() {
            return Chain.GetSize();
        }
    };

    ///////////////////////////////////////////////////////////////////////////////

    template <class T>
    inline void ReadArr(TBlockChainIterator* res, T* dst) {
        int n;
        res->Read(&n, sizeof(n));
        if (n >= 0) {
            dst->resize(n);
            if (n > 0)
                res->Read(&(*dst)[0], n * sizeof((*dst)[0]));
        } else {
            dst->resize(0);
            res->Fail();
        }
    }

    template <>
    inline void ReadArr<TString>(TBlockChainIterator* res, TString* dst) {
        int n;
        res->Read(&n, sizeof(n));
        if (n >= 0) {
            dst->resize(n);
            if (n > 0)
                res->Read(dst->begin(), n * sizeof(TString::value_type));
        } else {
            dst->resize(0);
            res->Fail();
        }
    }

    // saves on zeroing *dst with yresize()
    template <class T, class A>
    static void ReadYArr(TBlockChainIterator* res, TVector<T, A>* dst) {
        int n;
        res->Read(&n, sizeof(n));
        if (n >= 0) {
            dst->yresize(n);
            if (n > 0)
                res->Read(&(*dst)[0], n * sizeof((*dst)[0]));
        } else {
            dst->yresize(0);
            res->Fail();
        }
    }

    template <class T>
    static void Read(TBlockChainIterator* res, T* dst) {
        res->Read(dst, sizeof(T));
    }

}
