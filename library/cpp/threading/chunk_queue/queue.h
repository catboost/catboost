#pragma once

#include <util/datetime/base.h>
#include <util/generic/noncopyable.h>
#include <util/generic/ptr.h>
#include <util/generic/typetraits.h>
#include <util/generic/vector.h>
#include <util/generic/ylimits.h>
#include <library/cpp/deprecated/atomic/atomic.h>
#include <util/system/guard.h>
#include <util/system/spinlock.h>
#include <util/system/yassert.h>

#include <type_traits>
#include <utility>

namespace NThreading {
////////////////////////////////////////////////////////////////////////////////
// Platform helpers

#if !defined(PLATFORM_CACHE_LINE)
#define PLATFORM_CACHE_LINE 64
#endif

#if !defined(PLATFORM_PAGE_SIZE)
#define PLATFORM_PAGE_SIZE 4 * 1024
#endif

    template <typename T, size_t PadSize = PLATFORM_CACHE_LINE>
    struct TPadded: public T {
        char Pad[PadSize - sizeof(T) % PadSize];

        TPadded() {
            static_assert(sizeof(*this) % PadSize == 0, "padding does not work");
            Y_UNUSED(Pad);
        }

        template<typename... Args>
        TPadded(Args&&... args)
            : T(std::forward<Args>(args)...)
        {
            static_assert(sizeof(*this) % PadSize == 0, "padding does not work");
            Y_UNUSED(Pad);
        }
    };

    ////////////////////////////////////////////////////////////////////////////////
    // Type helpers

    namespace NImpl {
        template <typename T>
        struct TPodTypeHelper {
            template <typename TT>
            static void Write(T* ptr, TT&& value) {
                *ptr = value;
            }

            static T Read(T* ptr) {
                return *ptr;
            }

            static void Destroy(T* ptr) {
                Y_UNUSED(ptr);
            }
        };

        template <typename T>
        struct TNonPodTypeHelper {
            template <typename TT>
            static void Write(T* ptr, TT&& value) {
                new (ptr) T(std::forward<TT>(value));
            }

            static T Read(T* ptr) {
                return std::move(*ptr);
            }

            static void Destroy(T* ptr) {
                (void)ptr; /* Make MSVC happy. */
                ptr->~T();
            }
        };

        template <typename T>
        using TTypeHelper = std::conditional_t<
            TTypeTraits<T>::IsPod,
            TPodTypeHelper<T>,
            TNonPodTypeHelper<T>>;

    }

    ////////////////////////////////////////////////////////////////////////////////
    // One producer/one consumer chunked queue.

    template <typename T, size_t ChunkSize = PLATFORM_PAGE_SIZE>
    class TOneOneQueue: private TNonCopyable {
        using TTypeHelper = NImpl::TTypeHelper<T>;

        struct TChunk;

        struct TChunkHeader {
            size_t Count = 0;
            TChunk* Next = nullptr;
        };

        struct TChunk: public TChunkHeader {
            static constexpr size_t MaxCount = (ChunkSize - sizeof(TChunkHeader)) / sizeof(T);

            char Entries[MaxCount * sizeof(T)];

            TChunk() {
                Y_UNUSED(Entries); // uninitialized
            }

            ~TChunk() {
                for (size_t i = 0; i < this->Count; ++i) {
                    TTypeHelper::Destroy(GetPtr(i));
                }
            }

            T* GetPtr(size_t i) {
                return (T*)Entries + i;
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
            DeleteChunks(Reader.Chunk);
        }

        template <typename TT>
        void Enqueue(TT&& value) {
            T* ptr = PrepareWrite();
            Y_ASSERT(ptr);
            TTypeHelper::Write(ptr, std::forward<TT>(value));
            CompleteWrite();
        }

        bool Dequeue(T& value) {
            if (T* ptr = PrepareRead()) {
                value = TTypeHelper::Read(ptr);
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
            TChunk* chunk = Writer.Chunk;
            Y_ASSERT(chunk && !chunk->Next);

            if (chunk->Count != TChunk::MaxCount) {
                return chunk->GetPtr(chunk->Count);
            }

            chunk = new TChunk();
            AtomicSet(Writer.Chunk->Next, chunk);
            Writer.Chunk = chunk;
            return chunk->GetPtr(0);
        }

        void CompleteWrite() {
            AtomicSet(Writer.Chunk->Count, Writer.Chunk->Count + 1);
        }

        T* PrepareRead() {
            TChunk* chunk = Reader.Chunk;
            Y_ASSERT(chunk);

            for (;;) {
                size_t writerCount = AtomicGet(chunk->Count);
                if (Reader.Count != writerCount) {
                    return chunk->GetPtr(Reader.Count);
                }

                if (writerCount != TChunk::MaxCount) {
                    return nullptr;
                }

                chunk = AtomicGet(chunk->Next);
                if (!chunk) {
                    return nullptr;
                }

                delete Reader.Chunk;
                Reader.Chunk = chunk;
                Reader.Count = 0;
            }
        }

        void CompleteRead() {
            ++Reader.Count;
        }

    private:
        static void DeleteChunks(TChunk* chunk) {
            while (chunk) {
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
        using TTypeHelper = NImpl::TTypeHelper<T>;

        struct TEntry {
            T Value;
            ui64 Tag;
        };

        struct TQueueType: public TOneOneQueue<TEntry, ChunkSize> {
            TAtomic WriteLock = 0;

            using TOneOneQueue<TEntry, ChunkSize>::PrepareWrite;
            using TOneOneQueue<TEntry, ChunkSize>::CompleteWrite;

            using TOneOneQueue<TEntry, ChunkSize>::PrepareRead;
            using TOneOneQueue<TEntry, ChunkSize>::CompleteRead;
        };

    private:
        union {
            TAtomic WriteTag = 0;
            char Pad[PLATFORM_CACHE_LINE];
        };

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
                value = TTypeHelper::Read(&entry->Value);
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
            return AtomicIncrement(WriteTag);
        }

        template <typename TT>
        bool TryEnqueue(TT&& value, ui64 tag) {
            for (size_t i = 0; i < Concurrency; ++i) {
                TQueueType& queue = Queues[i];
                if (AtomicTryAndTryLock(&queue.WriteLock)) {
                    TEntry* entry = queue.PrepareWrite();
                    Y_ASSERT(entry);
                    TTypeHelper::Write(&entry->Value, std::forward<TT>(value));
                    entry->Tag = tag;
                    queue.CompleteWrite();
                    AtomicUnlock(&queue.WriteLock);
                    return true;
                }
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
            TAtomic WriteLock = 0;
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
                if (AtomicTryAndTryLock(&queue.WriteLock)) {
                    queue.Enqueue(std::forward<TT>(value));
                    AtomicUnlock(&queue.WriteLock);
                    return true;
                }
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
            union {
                TAtomic WriteLock = 0;
                char Pad1[PLATFORM_CACHE_LINE];
            };
            union {
                TAtomic ReadLock = 0;
                char Pad2[PLATFORM_CACHE_LINE];
            };
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
                if (AtomicTryAndTryLock(&queue.ReadLock)) {
                    bool dequeued = queue.Dequeue(value);
                    AtomicUnlock(&queue.ReadLock);
                    if (dequeued) {
                        return true;
                    }
                }
            }
            return false;
        }

        bool IsEmpty() {
            for (size_t i = 0; i < Concurrency; ++i) {
                TQueueType& queue = Queues[i];
                if (AtomicTryAndTryLock(&queue.ReadLock)) {
                    bool empty = queue.IsEmpty();
                    AtomicUnlock(&queue.ReadLock);
                    if (!empty) {
                        return false;
                    }
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
                if (AtomicTryAndTryLock(&queue.WriteLock)) {
                    queue.Enqueue(std::forward<TT>(value));
                    AtomicUnlock(&queue.WriteLock);
                    return true;
                }
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
            TAutoPtr<T> value;
            while (Dequeue(value)) {
                // do nothing
            }
        }

        void Enqueue(TAutoPtr<T> value) {
            Impl.Enqueue(value.Get());
            Y_UNUSED(value.Release());
        }

        bool Dequeue(TAutoPtr<T>& value) {
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
}
