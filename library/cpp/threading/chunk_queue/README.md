# Chunk Queue Library

Non-blocking queues for inter-thread message passing.

All classes are declared in the [`queue.h`](queue.h) header in the `NThreading` namespace.

## Queue classes overview

| Class | Writers | Readers | Ordering (FIFO) | Contention |
|---|---|---|---|---|
| [`TOneOneQueue`](queue.h) | 1 | 1 | strict | none |
| [`TManyOneQueue`](queue.h) | N | 1 | per-writer; global best effort | writers |
| [`TManyManyQueue`](queue.h) | N | M | strict global | writers + readers |
| [`TRelaxedManyOneQueue`](queue.h) | N | 1 | none | writers |
| [`TRelaxedManyManyQueue`](queue.h) | N | M | none | writers + readers |

**Best effort** means the queue tries to preserve the global FIFO order across different writers, but does not guarantee it. Elements of the same writer are always consumed in the order they were enqueued (per-writer FIFO); the interleaving of elements from different writers is approximate and depends on timing and internal queue state.

All queues: non-copyable, non-blocking (`Dequeue` returns `false` when empty — waiting is up to the client), elements are stored by value.

## Queue classes

### [`TOneOneQueue<T, ChunkSize>`](queue.h) — SPSC

A strict FIFO queue for a single writer and a single reader. The fastest one. Contract:

- `Enqueue` — at most one writer at a time; the writer does not have to be the same thread, but concurrent `Enqueue` calls require external synchronization;
- `Dequeue`/`IsEmpty` — at most one reader at a time, same rules.

### [`TManyOneQueue<T, Concurrency, ChunkSize>`](queue.h) — MPSC with per-producer FIFO

More expensive than `TRelaxedManyOneQueue`: every element is stamped with a global sequence tag.

### [`TManyManyQueue<T, ChunkSize, TLock>`](queue.h) — MPMC with strict FIFO

Use it when element ordering matters.

### [`TRelaxedManyOneQueue<T, Concurrency, ChunkSize>`](queue.h) — MPSC without ordering guarantees

The cheapest MPSC variant: elements are spread across internal partitions, so FIFO is not guaranteed — even for elements of the same writer.

### [`TRelaxedManyManyQueue<T, Concurrency, ChunkSize>`](queue.h) — MPMC without ordering guarantees

The same as `TRelaxedManyOneQueue`, but with multiple readers allowed.

### Pointer queues: `TAutoOneOneQueue` and friends

[`TAutoQueueBase`](queue.h) is a wrapper for queues of owning pointers: `Enqueue(TAutoPtr<T>)` takes ownership, `Dequeue` returns a `TAutoPtr<T>`, and the queue destructor deletes all remaining elements. Aliases: `TAutoOneOneQueue`, `TAutoManyOneQueue`, `TAutoManyManyQueue`, `TAutoRelaxedManyOneQueue`, `TAutoRelaxedManyManyQueue`.

## Example

See usage examples: [`queue_ut.cpp`](queue_ut.cpp).

## Template parameters

- `T` — the element type (stored by value; for pointer ownership use the `TAuto*` aliases);
- `ChunkSize` — chunk size in bytes (default 4 KiB); a larger chunk means fewer allocations but more overhead for small queues;
- `Concurrency` — the number of internal partitions in the `TMany*`/`TRelaxed*` queues (default 4); scale it with the number of writers;
- `TLock` (`TManyManyQueue` only) — the lock type (default `TAdaptiveLock`).

## See also

- Tests: [`queue_ut.cpp`](queue_ut.cpp)
- Benchmarks: [`benchmark/queue_benchmark.cpp`](benchmark/queue_benchmark.cpp) (comparison with `TLockFreeQueue`)
