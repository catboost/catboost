#pragma once

#include <util/generic/algorithm.h>
#include <util/generic/list.h>
#include <util/system/shmat.h>
#include <util/generic/noncopyable.h>

namespace NNetliba {
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
        typedef TVector<TBlock> TBlockVector;
        TBlockVector Blocks;
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
            Blocks.push_back(TBlock((const char*)data, Size, sz));
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
            TBlockVector::const_iterator i = LowerBound(Blocks.begin(), Blocks.end(), offset, TBlockLess());
            if (i == Blocks.end())
                return Blocks.ysize() - 1;
            if (i->Offset == offset)
                return (int)(i - Blocks.begin());
            return (int)(i - Blocks.begin() - 1);
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
    class TRopeDataPacket: public TNonCopyable {
        TBlockChain Chain;
        TVector<char*> Buf;
        char *Block, *BlockEnd;
        TList<TVector<char>> DataVectors;
        TIntrusivePtr<TSharedMemory> SharedData;
        TVector<TIntrusivePtr<TThrRefBase>> AttachedStorage;
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

    public:
        TRopeDataPacket()
            : Block(DefaultBuf)
            , BlockEnd(DefaultBuf + Y_ARRAY_SIZE(DefaultBuf))
        {
        }
        ~TRopeDataPacket() {
            for (size_t i = 0; i < Buf.size(); ++i)
                FreeBuf(Buf[i]);
        }
        static char* AllocBuf(int sz) {
            return new char[sz];
        }
        static void FreeBuf(char* buf) {
            delete[] buf;
        }

        // buf - pointer to buffer which will be freed with FreeBuf()
        // data - pointer to data start within buf
        // sz - size of useful data
        void AddBlock(char* buf, const char* data, int sz) {
            Buf.push_back(buf);
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
        void WriteDestructive(TVector<char>* data) {
            int n = data ? data->ysize() : 0;
            Write(n);
            if (n > 0) {
                TVector<char>& local = DataVectors.emplace_back(std::move(*data));
                Chain.AddBlock(&local[0], local.ysize());
            }
        }
        void AttachSharedData(TIntrusivePtr<TSharedMemory> shm) {
            SharedData = shm;
        }
        TSharedMemory* GetSharedData() const {
            return SharedData.Get();
        }
        const TBlockChain& GetChain() {
            return Chain;
        }
        int GetSize() {
            return Chain.GetSize();
        }
    };

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
    template <class T>
    static void ReadYArr(TBlockChainIterator* res, TVector<T>* dst) {
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

    ui32 CalcChecksum(const void* p, int size);
    ui32 CalcChecksum(const TBlockChain& chain);

    class TIncrementalChecksumCalcer {
        i64 TotalSum;
        int Offset;

    public:
        TIncrementalChecksumCalcer()
            : TotalSum(0)
            , Offset(0)
        {
        }
        void AddBlock(const void* p, int size);
        void AddBlockSum(ui32 sum, int size);
        ui32 CalcChecksum();

        static ui32 CalcBlockSum(const void* p, int size);
    };

    inline void AddChain(TIncrementalChecksumCalcer* ics, const TBlockChain& chain) {
        for (int k = 0; k < chain.GetBlockCount(); ++k) {
            const TBlockChain::TBlock& blk = chain.GetBlock(k);
            ics->AddBlock(blk.Data, blk.Size);
        }
    }
}
