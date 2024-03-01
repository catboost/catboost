#include "compact_queue.h"

namespace NYT {

////////////////////////////////////////////////////////////////////////////////

template <class T, size_t N>
void TCompactQueue<T, N>::Push(T value)
{
    if (Size_ == Queue_.size()) {
        auto oldSize = Queue_.size();
        Queue_.resize(2 * oldSize);

        if (FrontIndex_ + Size_ > oldSize) {
            std::move(
                Queue_.begin(),
                Queue_.begin() + FrontIndex_,
                Queue_.begin() + Size_);
        }
    }

    auto index = FrontIndex_ + Size_;
    if (index >= Queue_.size()) {
        index -= Queue_.size();
    }
    Queue_[index] = std::move(value);
    ++Size_;
}

template <class T, size_t N>
T TCompactQueue<T, N>::Pop()
{
    YT_VERIFY(!Empty());

    auto value = std::move(Queue_[FrontIndex_]);
    ++FrontIndex_;
    if (FrontIndex_ >= Queue_.size()) {
        FrontIndex_ -= Queue_.size();
    }
    --Size_;

    return value;
}

template <class T, size_t N>
const T& TCompactQueue<T, N>::Front() const
{
    return Queue_[FrontIndex_];
}

template <class T, size_t N>
size_t TCompactQueue<T, N>::Size() const
{
    return Size_;
}

template <class T, size_t N>
size_t TCompactQueue<T, N>::Capacity() const
{
    return Queue_.capacity();
}

template <class T, size_t N>
bool TCompactQueue<T, N>::Empty() const
{
    return Size_ == 0;
}

////////////////////////////////////////////////////////////////////////////////

} // namespace NYT
