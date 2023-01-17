#pragma once

#include <cstddef>
#include <atomic>

/**
 * Simple thread-safe per-class counter that can be used to make sure you don't
 * have any leaks in your code, or for statistical purposes.
 *
 * Example usage:
 * \code
 * class TMyClass: public TObjectCounter<TMyClass> {
 *     // ...
 * };
 *
 * // In your code:
 * Cerr << "TMyClass instances in use: " << TMyClass::ObjectCount() << Endl;
 * \endcode
 */
template <class T>
class TObjectCounter {
public:
    inline TObjectCounter() noexcept {
        ++Count_;
    }

    inline TObjectCounter(const TObjectCounter& /*item*/) noexcept {
        ++Count_;
    }

    inline ~TObjectCounter() {
        --Count_;
    }

    static inline long ObjectCount() noexcept {
        return Count_.load();
    }

    /**
     * Resets object count. Mainly for tests, as you don't want to do this in
     * your code and then end up with negative counts.
     *
     * \returns                         Current object count.
     */
    static inline long ResetObjectCount() noexcept {
        return Count_.exchange(0);
    }

private:
    static std::atomic<intptr_t> Count_;
};

template <class T>
std::atomic<intptr_t> TObjectCounter<T>::Count_ = 0;
