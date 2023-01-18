#include "chunked_memory_pool.h"

namespace NYT {

////////////////////////////////////////////////////////////////////////////////

TAllocationHolder::TAllocationHolder(TMutableRef ref, TRefCountedTypeCookie cookie)
    : Ref_(ref)
#ifdef YT_ENABLE_REF_COUNTED_TRACKING
    , Cookie_(cookie)
#endif
{
#ifdef YT_ENABLE_REF_COUNTED_TRACKING
    if (Cookie_ != NullRefCountedTypeCookie) {
        TRefCountedTrackerFacade::AllocateTagInstance(Cookie_);
        TRefCountedTrackerFacade::AllocateSpace(Cookie_, Ref_.Size());
    }
#endif
}

TAllocationHolder::~TAllocationHolder()
{
#ifdef YT_ENABLE_REF_COUNTED_TRACKING
    if (Cookie_ != NullRefCountedTypeCookie) {
        TRefCountedTrackerFacade::FreeTagInstance(Cookie_);
        TRefCountedTrackerFacade::FreeSpace(Cookie_, Ref_.Size());
    }
#endif
}

////////////////////////////////////////////////////////////////////////////////

class TDefaultMemoryChunkProvider
    : public IMemoryChunkProvider
{
public:
    std::unique_ptr<TAllocationHolder> Allocate(size_t size, TRefCountedTypeCookie cookie) override
    {
        return std::unique_ptr<TAllocationHolder>(TAllocationHolder::Allocate<TAllocationHolder>(size, cookie));
    }
};

const IMemoryChunkProviderPtr& GetDefaultMemoryChunkProvider()
{
    static const IMemoryChunkProviderPtr Result = New<TDefaultMemoryChunkProvider>();
    return Result;
}

////////////////////////////////////////////////////////////////////////////////

TChunkedMemoryPool::TChunkedMemoryPool(
    TRefCountedTypeCookie tagCookie,
    IMemoryChunkProviderPtr chunkProvider,
    size_t startChunkSize)
    : TagCookie_(tagCookie)
    , ChunkProviderHolder_(std::move(chunkProvider))
    , ChunkProvider_(ChunkProviderHolder_.Get())
{
    Initialize(startChunkSize);
}

TChunkedMemoryPool::TChunkedMemoryPool(
    TRefCountedTypeCookie tagCookie,
    size_t startChunkSize)
    : TagCookie_(tagCookie)
    , ChunkProvider_(GetDefaultMemoryChunkProvider().Get())
{
    Initialize(startChunkSize);
}

void TChunkedMemoryPool::Initialize(size_t startChunkSize)
{
    NextSmallSize_ = startChunkSize;
    FreeZoneBegin_ = nullptr;
    FreeZoneEnd_ = nullptr;
}

void TChunkedMemoryPool::Purge()
{
    Chunks_.clear();
    OtherBlocks_.clear();
    Size_ = 0;
    Capacity_ = 0;
    NextChunkIndex_ = 0;
    FreeZoneBegin_ = nullptr;
    FreeZoneEnd_ = nullptr;
}

char* TChunkedMemoryPool::AllocateUnalignedSlow(size_t size)
{
    auto* large = AllocateSlowCore(size);
    if (large) {
        return large;
    }
    return AllocateUnaligned(size);
}

char* TChunkedMemoryPool::AllocateAlignedSlow(size_t size, int align)
{
    // NB: Do not rely on any particular alignment of chunks.
    auto* large = AllocateSlowCore(size + align);
    if (large) {
        return AlignUp(large, align);
    }
    return AllocateAligned(size, align);
}

char* TChunkedMemoryPool::AllocateSlowCore(size_t size)
{
    TMutableRef ref;
    if (size > RegularChunkSize) {
        auto block = ChunkProvider_->Allocate(size, TagCookie_);
        ref = block->GetRef();
        Size_ += size;
        Capacity_ += ref.Size();
        OtherBlocks_.push_back(std::move(block));
        return ref.Begin();
    }

    YT_VERIFY(NextChunkIndex_ <= std::ssize(Chunks_));

    if (NextSmallSize_ < RegularChunkSize) {
        auto block = ChunkProvider_->Allocate(std::max(NextSmallSize_, size), TagCookie_);
        ref = block->GetRef();
        Capacity_ += ref.Size();
        OtherBlocks_.push_back(std::move(block));
        NextSmallSize_ = 2 * ref.Size();
    } else if (NextChunkIndex_ == std::ssize(Chunks_)) {
        auto chunk = ChunkProvider_->Allocate(RegularChunkSize, TagCookie_);
        ref = chunk->GetRef();
        Capacity_ += ref.Size();
        Chunks_.push_back(std::move(chunk));
        ++NextChunkIndex_;
    } else {
        ref = Chunks_[NextChunkIndex_++]->GetRef();
    }

    FreeZoneBegin_ = ref.Begin();
    FreeZoneEnd_ = ref.End();

    return nullptr;
}

void TChunkedMemoryPool::Absorb(TChunkedMemoryPool&& other)
{
    YT_VERIFY(ChunkProvider_ == other.ChunkProvider_);

    OtherBlocks_.reserve(OtherBlocks_.size() + other.OtherBlocks_.size());
    for (auto& block : other.OtherBlocks_) {
        OtherBlocks_.push_back(std::move(block));
    }
    other.OtherBlocks_.clear();

    // Suppose that
    // - "A" is filled blocks of the current pool;
    // - "a" is free blocks of the current pool;
    // - "B" is filled blocks of the other pool;
    // - "b" is free blocks of the other pool.
    // Then, from the initial layouts "AA...Aaa...a" and "BB...Bbb...b" we obtain "BB..BAA..Aaa...abb...b".
    Chunks_.reserve(Chunks_.size() + other.Chunks_.size());
    size_t oldSize = Chunks_.size();
    for (auto& chunk : other.Chunks_) {
        Chunks_.push_back(std::move(chunk));
    }
    // Transform "AA...Aaa...aBB...B" => "BB...BAA...Aaa...a"
    std::rotate(Chunks_.begin(), Chunks_.begin() + oldSize, Chunks_.begin() + oldSize + other.NextChunkIndex_);
    if (NextChunkIndex_ == 0) {
        FreeZoneBegin_ = other.FreeZoneBegin_;
        FreeZoneEnd_ = other.FreeZoneEnd_;
    }
    NextChunkIndex_ += other.NextChunkIndex_;
    other.Chunks_.clear();
    other.FreeZoneBegin_ = nullptr;
    other.FreeZoneEnd_ = nullptr;
    other.NextChunkIndex_ = 0;

    Size_ += other.Size_;
    Capacity_ += other.Capacity_;
    other.Size_ = 0;
    other.Capacity_ = 0;
}

size_t TChunkedMemoryPool::GetSize() const
{
    return Size_;
}

size_t TChunkedMemoryPool::GetCapacity() const
{
    return Capacity_;
}

size_t TChunkedMemoryPool::GetCurrentChunkSpareSize() const
{
    return FreeZoneEnd_ - FreeZoneBegin_;
}

////////////////////////////////////////////////////////////////////////////////

} // namespace NYT
