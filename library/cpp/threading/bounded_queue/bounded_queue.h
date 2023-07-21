#pragma once

#include <util/generic/yexception.h>

//https://www.1024cores.net/home/lock-free-algorithms/queues/bounded-mpmc-queue

namespace NThreading {
    template<typename T>
    class TBoundedQueue {
    public:
        explicit TBoundedQueue(size_t size)
            : Buffer_(new TCell[size])
            , Mask_(size - 1)
        {
            Y_ENSURE(size >= 2 && (size & (size - 1)) == 0);

            for (size_t i = 0; i < size; ++i) {
                Buffer_[i].Sequence.store(i, std::memory_order_relaxed);
            }
        }

        template <typename T_>
        [[nodiscard]] bool Enqueue(T_&& data) noexcept {
            TCell* cell;
            size_t pos = EnqueuePos_.load(std::memory_order_relaxed);

            for (;;) {
                cell = &Buffer_[pos & Mask_];
                size_t seq = cell->Sequence.load(std::memory_order_acquire);
                intptr_t dif = (intptr_t)seq - (intptr_t)pos;

                if (dif == 0) {
                    if (EnqueuePos_.compare_exchange_weak(pos, pos + 1, std::memory_order_relaxed)) {
                        break;
                    }
                } else if (dif < 0) {
                    return false;
                } else {
                    pos = EnqueuePos_.load(std::memory_order_relaxed);
                }
            }

            static_assert(noexcept(cell->Data = std::forward<T_>(data)));
            cell->Data = std::forward<T_>(data);
            cell->Sequence.store(pos + 1, std::memory_order_release);

            return true;
        }

        [[nodiscard]] bool Dequeue(T& data) noexcept {
            TCell* cell;
            size_t pos = DequeuePos_.load(std::memory_order_relaxed);

            for (;;) {
                cell = &Buffer_[pos & Mask_];
                size_t seq = cell->Sequence.load(std::memory_order_acquire);
                intptr_t dif = (intptr_t)seq - (intptr_t)(pos + 1);

                if (dif == 0) {
                    if (DequeuePos_.compare_exchange_weak(pos, pos + 1, std::memory_order_relaxed)) {
                        break;
                    }
                } else if (dif < 0) {
                    return false;
                } else {
                    pos = DequeuePos_.load(std::memory_order_relaxed);
                }
            }

            static_assert(noexcept(data = std::move(cell->Data)));
            data = std::move(cell->Data);

            cell->Sequence.store(pos + Mask_ + 1, std::memory_order_release);
            return true;
        }
    private:
        struct TCell {
            std::atomic<size_t> Sequence;
            T Data;
        };

        std::unique_ptr<TCell[]> Buffer_;
        const size_t Mask_;

        alignas(64) std::atomic<size_t> EnqueuePos_ = 0;
        alignas(64) std::atomic<size_t> DequeuePos_ = 0;
    };
}

