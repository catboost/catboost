#pragma once

#include <util/generic/array_ref.h>
#include <util/generic/variant.h>
#include <util/generic/vector.h>
#include <util/generic/xrange.h>

#include <cstdlib>

namespace NCB {

    template <class TSize>
    struct TViewIndexBlock {
        TSize Offset;
        TSize Size;
    };

    template <class TSize>
    struct TBlockView {
        using TBlocks = TVector<TViewIndexBlock<TSize>>;

        TSize Size;
        TBlocks Blocks;

    public:
        TBlockView(size_t size, TBlocks&& blocks)
            : Size(size)
            , Blocks(std::move(blocks))
        {}

        explicit TBlockView(TBlocks&& blocks)
            : TBlockView(CalcSize(blocks), std::move(blocks))
        {}

    private:
        static TSize CalcSize(const TBlocks& blocks) {
            TSize size = 0;
            for (const auto& block : blocks) {
                size += block.Size;
            }
            return size;
        }
    };

    template <class TSize>
    using TIndexView = TVector<TSize>; // index in src data

    template <class TSize>
    struct TFullView {
        TSize Size;
    };

    template<class TSize>
    class TViewIndexing : public TVariant<TFullView<TSize>, TBlockView<TSize>, TIndexView<TSize>> {
        using TBase = TVariant<TFullView<TSize>, TBlockView<TSize>, TIndexView<TSize>>;

    public:
        template<class T>
        explicit TViewIndexing(T&& view)
            : TBase(std::move(view))
        {}

        TSize Size() const {
            switch (TBase::Index()) {
                case TBase::template TagOf<TFullView<TSize>>():
                    return Get<TFullView<TSize>>().Size;
                case TBase::template TagOf<TBlockView<TSize>>():
                    return Get<TBlockView<TSize>>().Size;
                case TBase::template TagOf<TIndexView<TSize>>():
                    return static_cast<TSize>(Get<TIndexView<TSize>>().size());
                default:
                    Y_VERIFY(false);
            }
        }

        // Had to redefine Get because automatic resolution does not work with current TVariant implementation
        template <class T>
        decltype(auto) Get() {
            return ::Get<T>((TBase&)*this);
        }

        template <class T>
        decltype(auto) Get() const {
            return ::Get<T>((const TBase&)*this);
        }
    };


    // TArrayLike must have O(1) random-access operator[].
    template<class TArrayLike, class TSize = size_t>
    struct TViewIter {
        TArrayLike Src;
        TViewIndexing<TSize>* ViewIndexing;

        TSize Size() const {
            return ViewIndexing->Size();
        }

        // F is a visitor function that will be repeatedly called with (index, element) arguments
        template<class F>
        void Iter(F&& f) {
            switch (ViewIndexing->Index()) {
                case TViewIndexing<TSize>::template TagOf<TFullView<TSize>>():
                    {
                        for (TSize index : xrange<TSize>(ViewIndexing->template Get<TFullView<TSize>>().Size)) {
                            f(index, Src[index]);
                        }
                    }
                    break;
                case TViewIndexing<TSize>::template TagOf<TBlockView<TSize>>():
                    {
                        TSize index = 0;
                        const auto& blocks = ViewIndexing->template Get<TBlockView<TSize>>().Blocks;
                        for (const auto& block : blocks) {
                            TSize srcEnd = block.Offset + block.Size;
                            for (TSize srcIndex = block.Offset; srcIndex != srcEnd; ++srcIndex, ++index) {
                                f(index, Src[srcIndex]);
                            }
                        }
                    }
                    break;
                case TViewIndexing<TSize>::template TagOf<TIndexView<TSize>>():
                    {
                        const auto& indexView = ViewIndexing->template Get<TIndexView<TSize>>();
                        for (TSize index : xrange<TSize>(indexView.size())) {
                            f(index, Src[indexView[index]]);
                        }
                    }
                    break;
                default:
                    Y_VERIFY(false);
            }
        }
    };
}

