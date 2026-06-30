
#include "pool.h"

TMemoryPool::IGrowPolicy* TMemoryPool::TLinearGrow::Instance() noexcept {
    return SingletonWithPriority<TLinearGrow, 0>();
}

TMemoryPool::IGrowPolicy* TMemoryPool::TExpGrow::Instance() noexcept {
    return SingletonWithPriority<TExpGrow, 0>();
}

void TMemoryPool::AddChunk(size_t hint) {
    const size_t dataLen = Max(BlockSize_, hint);
    size_t allocSize = dataLen + sizeof(TChunk);
    if (Options_.RoundUpToNextPowerOfTwo) {
        allocSize = FastClp2(allocSize);
    }
    TBlock nb = Alloc_->Allocate(allocSize);

    // Add previous chunk's stats
    if (Current_ != &Empty_) {
        MemoryAllocatedBeforeCurrent_ += Current_->Used();
        MemoryWasteBeforeCurrent_ += Current_->Left();
    }

    BlockSize_ = GrowPolicy_->Next(dataLen);
    Current_ = new (nb.Data) TChunk(nb.Len - sizeof(TChunk));
    Chunks_.PushBack(Current_);
}

size_t TMemoryPool::DoClear(bool keepfirst) noexcept {
    size_t chunksUsed = 0;
    while (!Chunks_.Empty()) {
        chunksUsed += 1;
        TChunk* c = Chunks_.PopBack();

        if (keepfirst && Chunks_.Empty()) {
            c->ResetChunk();
            Chunks_.PushBack(c);
            Current_ = c;
            BlockSize_ = c->BlockLength() - sizeof(TChunk);
            MemoryAllocatedBeforeCurrent_ = 0;
            MemoryWasteBeforeCurrent_ = 0;
            return chunksUsed;
        }

        TBlock b = {c, c->BlockLength()};

        c->~TChunk();
        Alloc_->Release(b);
    }

    Current_ = &Empty_;
    BlockSize_ = Origin_;
    MemoryAllocatedBeforeCurrent_ = 0;
    MemoryWasteBeforeCurrent_ = 0;
    return chunksUsed;
}
