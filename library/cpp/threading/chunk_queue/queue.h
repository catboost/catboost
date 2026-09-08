#pragma once

#include <util/generic/noncopyable.h>
#include <util/generic/ptr.h>
#include <util/generic/ylimits.h>
#include <util/system/datetime.h>
#include <util/system/guard.h>
#include <util/system/spinlock.h>
#include <util/system/yassert.h>

#include <atomic>
#include <type_traits>

namespace NThreading {
    ////////////////////////////////////////////////////////////////////////////////
    // Platform helpers

#if !defined(PLATFORM_CACHE_LINE)
    #define PLATFORM_CACHE_LINE 64
#endif

#if !defined(PLATFORM_PAGE_SIZE)
    #define PLATFORM_PAGE_SIZE (4 * 1024)
#endif

    template <typename T, size_t PadSize = PLATFORM_CACHE_LINE>
    struct alignas(PadSize) TPadded: public T {
        char Pad[PadSize - sizeof(T) % PadSize];

        TPadded() {
            static_assert(sizeof(*this) % PadSize == 0, "padding does not work");
            Y_UNUSED(Pad);
        }

        template <typename... Args>
        TPadded(Args&&... args)
            : T(std::forward<Args>(args)...)
        {
            static_assert(sizeof(*this) % PadSize == 0, "padding does not work");
            Y_UNUSED(Pad);
        }
    };

    ////////////////////////////////////////////////////////////////////////////////
    // One producer/one consumer chunked queue.

    template <typename T, size_t ChunkSize = PLATFORM_PAGE_SIZE>
    class TOneOneQueue: private TNonCopyable {
        // std::atomic_ref, or a minimal fallback for toolchains whose STL
        // lacks it (e.g. CUDA builds with old libc++)
#if defined(__cpp_lib_atomic_ref) && __cpp_lib_atomic_ref >= 201806L
        template <typename TT>
        using TAtomicRef = std::atomic_ref<TT>;
#else
        template <typename TT>
        class TAtomicRef {
            static_assert(std::is_trivially_copyable<TT>::value, "TAtomicRef requires a trivially copyable type");
            static_assert(static_cast<int>(std::memory_order_release) == __ATOMIC_RELEASE);
            static_assert(static_cast<int>(std::memory_order_acquire) == __ATOMIC_ACQUIRE);
            static_assert(static_cast<int>(std::memory_order_relaxed) == __ATOMIC_RELAXED);

        public:
            explicit TAtomicRef(TT& obj) noexcept
                : Obj_(&obj)
            {
            }

            TT load(std::memory_order order) const noexcept {
                return __atomic_load_n(Obj_, static_cast<int>(order));
            }

            void store(TT desired, std::memory_order order) noexcept {
                __atomic_store_n(Obj_, desired, static_cast<int>(order));
            }

        private:
            TT* Obj_;
        };
#endif

        struct TChunk;

        struct TChunkHeader {
            // Incremented by the producer (release) after writing an entry, read by
            // the consumer (acquire) — publishes the entry data written so far.
            // Plain field on purpose: only the producer ever writes it, and the
            // producer also reads it back (see PrepareWrite/CompleteWrite) — this
            // lets the compiler keep the write position in a register. All
            // cross-thread accesses go through std::atomic_ref.
            size_t Count = 0;
            // Set by the producer (release) when the chunk is exhausted, read by the
            // consumer (acquire) — publishes the next chunk and makes it safe for the
            // consumer to delete the exhausted one.
            TChunk* Next = nullptr;
        };

        struct TChunk: public TChunkHeader {
            // Offset of Entries inside TChunk: the header size rounded up to the
            // alignment of T, so that every slot is properly aligned.
            static constexpr size_t EntriesOffset = (sizeof(TChunkHeader) + alignof(T) - 1) / alignof(T) * alignof(T);
            static constexpr size_t MaxCount = ChunkSize > EntriesOffset ? (ChunkSize - EntriesOffset) / sizeof(T) : 0;
            static_assert(MaxCount > 0, "ChunkSize is too small to hold at least one element of T");

            alignas(T) char Entries[MaxCount * sizeof(T)];

            TChunk() {
                Y_UNUSED(Entries); // uninitialized
            }

            // No concurrent access: the chunk is destroyed after the producer
            // finished with it (or in the queue destructor). Called
            // unconditionally: for trivially destructible T the destructor
            // calls are no-ops and the whole loop is optimized away.
            void DestroyRangeFrom(size_t start) {
                const size_t end = this->Count;
                Y_ASSERT(start <= end);
                T* const endPtr = GetPtr(end);

                for (T* ptr = GetPtr(start); ptr != endPtr; ++ptr) {
                    ptr->~T();
                }
            }

            T* GetPtr(size_t i) {
                return reinterpret_cast<T*>(Entries) + i;
            }
        };

        struct TWriterState {
            TChunk* Chunk = nullptr;
        };

        struct TReaderState {
            TChunk* Chunk = nullptr;
            size_t Count = 0;
        };

    private:
        TPadded<TWriterState> Writer;
        TPadded<TReaderState> Reader;

    public:
        using TItem = T;

        TOneOneQueue() {
            Writer.Chunk = Reader.Chunk = new TChunk();
        }

        ~TOneOneQueue() {
            DestroyChunks(Reader.Chunk, Reader.Count);
        }

        template <typename TT>
        void Enqueue(TT&& value) {
            T* ptr = PrepareWrite();
            Y_ASSERT(ptr);
            new (ptr) T(std::forward<TT>(value));
            CompleteWrite();
        }

        bool Dequeue(T& value) {
            if (T* ptr = PrepareRead(); ptr) {
                value = std::move(*ptr);
                ptr->~T();
                CompleteRead();
                return true;
            }
            return false;
        }

        bool IsEmpty() {
            return !PrepareRead();
        }

    protected:
        T* PrepareWrite() {
            Y_ASSERT(Writer.Chunk);

            // Only the producer touches Count of the current chunk, so this
            // load cannot be stale. Keeping it a plain read lets the compiler
            // forward the store from CompleteWrite() into it and keep the
            // write position in a register instead of memory (an atomic load
            // here makes clang merge load+store in CompleteWrite() into a
            // memory RMW and spills the write position to the stack).
            if (Writer.Chunk->Count == TChunk::MaxCount) [[unlikely]] {
                TChunk* const next = new TChunk();
                // Release-publishes the new chunk to the consumer
                TAtomicRef<TChunk*>{Writer.Chunk->Next}.store(next, std::memory_order_release);
                Writer.Chunk = next;
            }
            return Writer.Chunk->GetPtr(Writer.Chunk->Count);
        }

        void CompleteWrite() {
            // Release-publishes the entry written by the preceding PrepareWrite().
            // A store suffices: Count of the current chunk is only ever written
            // by the (single) producer, so an atomic RMW (fetch_add) is not
            // needed — and would cost ~5x throughput (lock xadd on x86).
            // Note: a *relaxed* fetch_add would also be formally insufficient,
            // as it does not publish the entry data to the consumer.
            TChunk* chunk = Writer.Chunk;
            TAtomicRef<size_t> count{chunk->Count};
            count.store(chunk->Count + 1, std::memory_order_release);
        }

        T* PrepareRead() {
            TChunk* chunk = Reader.Chunk;
            Y_ASSERT(chunk);

            const size_t writerCount = TAtomicRef<size_t>{chunk->Count}.load(std::memory_order_acquire);
            if (Reader.Count != writerCount) {
                return chunk->GetPtr(Reader.Count);
            }

            if (writerCount == TChunk::MaxCount) {
                if (TChunk* next = TAtomicRef<TChunk*>{chunk->Next}.load(std::memory_order_acquire); next) {
                    delete chunk;
                    Reader.Chunk = next;
                    Reader.Count = 0;
                    // The next chunk may already contain published entries. It
                    // cannot be exhausted itself: the reader has consumed
                    // nothing from it yet.
                    if (TAtomicRef<size_t>{next->Count}.load(std::memory_order_acquire) != 0) {
                        return next->GetPtr(0);
                    }
                }
            }
            return nullptr;
        }

        void CompleteRead() {
            ++Reader.Count;
        }

        // Destroys the chunk chain starting at chunk, beginning with the
        // entries from offset `start` of the first chunk. Static and taking
        // only values (not the queue object) — see the destructor comment.
        static void DestroyChunks(TChunk* chunk, size_t start) {
            while (chunk) {
                chunk->DestroyRangeFrom(start);
                start = 0;
                TChunk* next = chunk->Next;
                delete chunk;
                chunk = next;
            }
        }
    };

    ////////////////////////////////////////////////////////////////////////////////
    // Multiple producers/single consumer partitioned queue.
    // Provides FIFO guaranties for each producer.

    template <typename T, size_t Concurrency = 4, size_t ChunkSize = PLATFORM_PAGE_SIZE>
    class TManyOneQueue: private TNonCopyable {
        struct TEntry {
            T Value;
            ui64 Tag;
        };

        struct TQueueType: public TOneOneQueue<TEntry, ChunkSize> {
            TPadded<TSpinLock> WriteLock;

            using TOneOneQueue<TEntry, ChunkSize>::PrepareWrite;
            using TOneOneQueue<TEntry, ChunkSize>::CompleteWrite;

            using TOneOneQueue<TEntry, ChunkSize>::PrepareRead;
            using TOneOneQueue<TEntry, ChunkSize>::CompleteRead;
        };

    private:
        TPadded<std::atomic<ui64>> WriteTag{1};
        TQueueType Queues[Concurrency];

    public:
        using TItem = T;

        template <typename TT>
        void Enqueue(TT&& value) {
            ui64 tag = NextTag();
            while (!TryEnqueue(std::forward<TT>(value), tag)) {
                SpinLockPause();
            }
        }

        bool Dequeue(T& value) {
            size_t index = 0;
            if (TEntry* entry = PrepareRead(index)) {
                T* valuePtr = &entry->Value;
                value = std::move(*valuePtr);
                valuePtr->~T();
                Queues[index].CompleteRead();
                return true;
            }
            return false;
        }

        bool IsEmpty() {
            for (size_t i = 0; i < Concurrency; ++i) {
                if (!Queues[i].IsEmpty()) {
                    return false;
                }
            }
            return true;
        }

    private:
        ui64 NextTag() {
            // TODO: can we avoid synchronization here? it costs 1.5x performance penalty
            // return GetCycleCount();
            return WriteTag.fetch_add(1, std::memory_order_relaxed);
        }

        template <typename TT>
        bool TryEnqueue(TT&& value, ui64 tag) {
            for (size_t i = 0; i < Concurrency; ++i) {
                TQueueType& queue = Queues[i];
                if (queue.WriteLock.IsLocked()) {
                    continue;
                }
                TTryGuard guard{queue.WriteLock};
                if (!guard) {
                    continue;
                }
                TEntry* entry = queue.PrepareWrite();
                Y_ASSERT(entry);
                new (&entry->Value) T(std::forward<TT>(value));
                entry->Tag = tag;
                queue.CompleteWrite();
                return true;
            }
            return false;
        }

        TEntry* PrepareRead(size_t& index) {
            TEntry* entry = nullptr;
            ui64 tag = Max();

            for (size_t i = 0; i < Concurrency; ++i) {
                TEntry* e = Queues[i].PrepareRead();
                if (e && e->Tag < tag) {
                    index = i;
                    entry = e;
                    tag = e->Tag;
                }
            }

            if (entry) {
                // need second pass to catch updates within already scanned range
                size_t candidate = index;
                for (size_t i = 0; i < candidate; ++i) {
                    TEntry* e = Queues[i].PrepareRead();
                    if (e && e->Tag < tag) {
                        index = i;
                        entry = e;
                        tag = e->Tag;
                    }
                }
            }

            return entry;
        }
    };

    ////////////////////////////////////////////////////////////////////////////////
    // Concurrent many-many queue with strong FIFO guaranties.
    // Writers will not block readers (and vice versa), but will block each other.

    template <typename T, size_t ChunkSize = PLATFORM_PAGE_SIZE, typename TLock = TAdaptiveLock>
    class TManyManyQueue: private TNonCopyable {
    private:
        TPadded<TLock> WriteLock;
        TPadded<TLock> ReadLock;

        TOneOneQueue<T, ChunkSize> Queue;

    public:
        using TItem = T;

        template <typename TT>
        void Enqueue(TT&& value) {
            with_lock (WriteLock) {
                Queue.Enqueue(std::forward<TT>(value));
            }
        }

        bool Dequeue(T& value) {
            with_lock (ReadLock) {
                return Queue.Dequeue(value);
            }
        }

        bool IsEmpty() {
            with_lock (ReadLock) {
                return Queue.IsEmpty();
            }
        }
    };

    ////////////////////////////////////////////////////////////////////////////////
    // Multiple producers/single consumer partitioned queue.
    // Because of random partitioning reordering possible - FIFO not guaranteed!

    template <typename T, size_t Concurrency = 4, size_t ChunkSize = PLATFORM_PAGE_SIZE>
    class TRelaxedManyOneQueue: private TNonCopyable {
        struct TQueueType: public TOneOneQueue<T, ChunkSize> {
            TPadded<TSpinLock> WriteLock;
        };

    private:
        union {
            size_t ReadPos = 0;
            char Pad[PLATFORM_CACHE_LINE];
        };

        TQueueType Queues[Concurrency];

    public:
        using TItem = T;

        template <typename TT>
        void Enqueue(TT&& value) {
            while (!TryEnqueue(std::forward<TT>(value))) {
                SpinLockPause();
            }
        }

        bool Dequeue(T& value) {
            for (size_t i = 0; i < Concurrency; ++i) {
                TQueueType& queue = Queues[ReadPos++ % Concurrency];
                if (queue.Dequeue(value)) {
                    return true;
                }
            }
            return false;
        }

        bool IsEmpty() {
            for (size_t i = 0; i < Concurrency; ++i) {
                if (!Queues[i].IsEmpty()) {
                    return false;
                }
            }
            return true;
        }

    private:
        template <typename TT>
        bool TryEnqueue(TT&& value) {
            size_t writePos = GetCycleCount();
            for (size_t i = 0; i < Concurrency; ++i) {
                TQueueType& queue = Queues[writePos++ % Concurrency];
                if (queue.WriteLock.IsLocked()) {
                    continue;
                }
                TTryGuard guard{queue.WriteLock};
                if (!guard) {
                    continue;
                }
                queue.Enqueue(std::forward<TT>(value));
                return true;
            }
            return false;
        }
    };

    ////////////////////////////////////////////////////////////////////////////////
    // Concurrent many-many partitioned queue.
    // Because of random partitioning reordering possible - FIFO not guaranteed!

    template <typename T, size_t Concurrency = 4, size_t ChunkSize = PLATFORM_PAGE_SIZE>
    class TRelaxedManyManyQueue: private TNonCopyable {
        struct TQueueType: public TOneOneQueue<T, ChunkSize> {
            TPadded<TSpinLock> WriteLock;
            TPadded<TSpinLock> ReadLock;
        };

    private:
        TQueueType Queues[Concurrency];

    public:
        using TItem = T;

        template <typename TT>
        void Enqueue(TT&& value) {
            while (!TryEnqueue(std::forward<TT>(value))) {
                SpinLockPause();
            }
        }

        bool Dequeue(T& value) {
            size_t readPos = GetCycleCount();
            for (size_t i = 0; i < Concurrency; ++i) {
                TQueueType& queue = Queues[readPos++ % Concurrency];
                if (queue.ReadLock.IsLocked()) {
                    continue;
                }
                TTryGuard guard{queue.ReadLock};
                if (!guard) {
                    continue;
                }
                if (queue.Dequeue(value)) {
                    return true;
                }
            }
            return false;
        }

        bool IsEmpty() {
            for (size_t i = 0; i < Concurrency; ++i) {
                TQueueType& queue = Queues[i];
                if (queue.ReadLock.IsLocked()) {
                    continue;
                }
                TTryGuard guard{queue.ReadLock};
                if (!guard) {
                    continue;
                }
                if (!queue.IsEmpty()) {
                    return false;
                }
            }
            return true;
        }

    private:
        template <typename TT>
        bool TryEnqueue(TT&& value) {
            size_t writePos = GetCycleCount();
            for (size_t i = 0; i < Concurrency; ++i) {
                TQueueType& queue = Queues[writePos++ % Concurrency];
                if (queue.WriteLock.IsLocked()) {
                    continue;
                }
                TTryGuard guard{queue.WriteLock};
                if (!guard) {
                    continue;
                }
                queue.Enqueue(std::forward<TT>(value));
                return true;
            }
            return false;
        }
    };

    ////////////////////////////////////////////////////////////////////////////////
    // Simple wrapper to deal with AutoPtrs

    template <typename T, typename TImpl>
    class TAutoQueueBase: private TNonCopyable {
    private:
        TImpl Impl;

    public:
        using TItem = TAutoPtr<T>;

        ~TAutoQueueBase() {
            TItem value;
            while (Dequeue(value)) {
                // do nothing
            }
        }

        void Enqueue(TItem value) {
            Impl.Enqueue(value.Get());
            Y_UNUSED(value.Release());
        }

        bool Dequeue(TItem& value) {
            T* ptr = nullptr;
            if (Impl.Dequeue(ptr)) {
                value.Reset(ptr);
                return true;
            }
            return false;
        }

        bool IsEmpty() {
            return Impl.IsEmpty();
        }
    };

    template <typename T, size_t ChunkSize = PLATFORM_PAGE_SIZE>
    using TAutoOneOneQueue = TAutoQueueBase<T, TOneOneQueue<T*, ChunkSize>>;

    template <typename T, size_t Concurrency = 4, size_t ChunkSize = PLATFORM_PAGE_SIZE>
    using TAutoManyOneQueue = TAutoQueueBase<T, TManyOneQueue<T*, Concurrency, ChunkSize>>;

    template <typename T, size_t ChunkSize = PLATFORM_PAGE_SIZE, typename TLock = TAdaptiveLock>
    using TAutoManyManyQueue = TAutoQueueBase<T, TManyManyQueue<T*, ChunkSize, TLock>>;

    template <typename T, size_t Concurrency = 4, size_t ChunkSize = PLATFORM_PAGE_SIZE>
    using TAutoRelaxedManyOneQueue = TAutoQueueBase<T, TRelaxedManyOneQueue<T*, Concurrency, ChunkSize>>;

    template <typename T, size_t Concurrency = 4, size_t ChunkSize = PLATFORM_PAGE_SIZE>
    using TAutoRelaxedManyManyQueue = TAutoQueueBase<T, TRelaxedManyManyQueue<T*, Concurrency, ChunkSize>>;
} // namespace NThreading
