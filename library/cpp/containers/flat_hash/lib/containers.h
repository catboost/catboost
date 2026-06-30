#pragma once

#include "concepts/container.h"
#include "value_markers.h"

#include <util/system/yassert.h>

#include <util/generic/vector.h>
#include <util/generic/typetraits.h>
#include <util/generic/utility.h>

#include <optional>

namespace NFlatHash {

/* FLAT CONTAINER */

template <class T, class Alloc = std::allocator<T>>
class TFlatContainer {
public:
    using value_type = T;
    using size_type = size_t;
    using difference_type = ptrdiff_t;
    using allocator_type = Alloc;
    using pointer = typename std::allocator_traits<allocator_type>::pointer;
    using const_pointer = typename std::allocator_traits<allocator_type>::const_pointer;

private:
    class TCage {
        enum ENodeStatus {
            NS_EMPTY,
            NS_TAKEN,
            NS_DELETED
        };

    public:
        TCage() noexcept = default;

        TCage(const TCage&) = default;
        TCage(TCage&&) = default;

        TCage& operator=(const TCage& rhs) {
            switch (rhs.Status_) {
            case NS_TAKEN:
                if constexpr (std::is_copy_assignable_v<value_type>) {
                    Value_ = rhs.Value_;
                } else {
                    Value_.emplace(rhs.Value());
                }
                break;
            case NS_EMPTY:
            case NS_DELETED:
                if (Value_.has_value()) {
                    Value_.reset();
                }
                break;
            default:
                Y_ABORT_UNLESS(false, "Not implemented");
            }
            Status_ = rhs.Status_;
            return *this;
        }
        // We never call it since all the TCage's are stored in vector
        TCage& operator=(TCage&& rhs) = delete;

        template <class... Args>
        void Emplace(Args&&... args) {
            Y_ASSERT(Status_ == NS_EMPTY);
            Value_.emplace(std::forward<Args>(args)...);
            Status_ = NS_TAKEN;
        }

        void Reset() noexcept {
            Y_ASSERT(Status_ == NS_TAKEN);
            Value_.reset();
            Status_ = NS_DELETED;
        }

        value_type& Value() {
            Y_ASSERT(Status_ == NS_TAKEN);
            return *Value_;
        }

        const value_type& Value() const {
            Y_ASSERT(Status_ == NS_TAKEN);
            return *Value_;
        }

        bool IsEmpty() const noexcept { return Status_ == NS_EMPTY; }
        bool IsTaken() const noexcept { return Status_ == NS_TAKEN; }
        bool IsDeleted() const noexcept { return Status_ == NS_DELETED; }

        ENodeStatus Status() const noexcept { return Status_; }

    private:
        std::optional<value_type> Value_;
        ENodeStatus Status_ = NS_EMPTY;
    };

public:
    explicit TFlatContainer(size_type initSize, const allocator_type& alloc = {})
        : Buckets_(initSize, alloc)
        , Taken_(0)
        , Empty_(initSize) {}

    TFlatContainer(const TFlatContainer&) = default;
    TFlatContainer(TFlatContainer&& rhs)
        : Buckets_(std::move(rhs.Buckets_))
        , Taken_(rhs.Taken_)
        , Empty_(rhs.Empty_)
    {
        rhs.Taken_ = 0;
        rhs.Empty_ = 0;
    }

    TFlatContainer& operator=(const TFlatContainer&) = default;
    TFlatContainer& operator=(TFlatContainer&&) = default;

    value_type& Node(size_type idx) { return Buckets_[idx].Value(); }
    const value_type& Node(size_type idx) const { return Buckets_[idx].Value(); }

    size_type Size() const noexcept { return Buckets_.size(); }
    size_type Taken() const noexcept { return Taken_; }
    size_type Empty() const noexcept { return Empty_; }

    template <class... Args>
    void InitNode(size_type idx, Args&&... args) {
        Buckets_[idx].Emplace(std::forward<Args>(args)...);
        ++Taken_;
        --Empty_;
    }

    void DeleteNode(size_type idx) noexcept {
        Buckets_[idx].Reset();
        --Taken_;
    }

    bool IsEmpty(size_type idx) const { return Buckets_[idx].IsEmpty(); }
    bool IsTaken(size_type idx) const { return Buckets_[idx].IsTaken(); }
    bool IsDeleted(size_type idx) const { return Buckets_[idx].IsDeleted(); }

    void Swap(TFlatContainer& rhs) noexcept {
        DoSwap(Buckets_, rhs.Buckets_);
        DoSwap(Taken_, rhs.Taken_);
        DoSwap(Empty_, rhs.Empty_);
    }

    TFlatContainer Clone(size_type newSize) const { return TFlatContainer(newSize, Buckets_.get_allocator()); }

private:
    TVector<TCage, allocator_type> Buckets_;
    size_type Taken_;
    size_type Empty_;
};

static_assert(NConcepts::ContainerV<TFlatContainer<int>>);
static_assert(NConcepts::RemovalContainerV<TFlatContainer<int>>);

/* DENSE CONTAINER */

template <class T, class EmptyMarker = NSet::TEqValueMarker<T>, class Alloc = std::allocator<T>>
class TDenseContainer {
    static_assert(NConcepts::ValueMarkerV<EmptyMarker>);

public:
    using value_type = T;
    using size_type = size_t;
    using difference_type = ptrdiff_t;
    using allocator_type = Alloc;
    using pointer = typename std::allocator_traits<allocator_type>::pointer;
    using const_pointer = typename std::allocator_traits<allocator_type>::const_pointer;

public:
    TDenseContainer(size_type initSize,  EmptyMarker emptyMarker = {}, const allocator_type& alloc = {})
        : Buckets_(initSize, emptyMarker.Create(), alloc)
        , Taken_(0)
        , EmptyMarker_(std::move(emptyMarker)) {}

    TDenseContainer(const TDenseContainer&) = default;
    TDenseContainer(TDenseContainer&&) = default;

    TDenseContainer& operator=(const TDenseContainer& rhs) {
        Taken_ = rhs.Taken_;
        EmptyMarker_ = rhs.EmptyMarker_;
        if constexpr (std::is_copy_assignable_v<value_type>) {
            Buckets_ = rhs.Buckets_;
        } else {
            auto tmp = rhs.Buckets_;
            Buckets_.swap(tmp);
        }
        return *this;
    }
    TDenseContainer& operator=(TDenseContainer&&) = default;

    value_type& Node(size_type idx) { return Buckets_[idx]; }
    const value_type& Node(size_type idx) const { return Buckets_[idx]; }

    size_type Size() const noexcept { return Buckets_.size(); }
    size_type Taken() const noexcept { return Taken_; }
    size_type Empty() const noexcept { return Size() - Taken(); }

    template <class... Args>
    void InitNode(size_type idx, Args&&... args) {
        Node(idx).~value_type();
        new (&Buckets_[idx]) value_type(std::forward<Args>(args)...);
        ++Taken_;
    }

    bool IsEmpty(size_type idx) const { return EmptyMarker_.Equals(Buckets_[idx]); }
    bool IsTaken(size_type idx) const { return !IsEmpty(idx); }

    void Swap(TDenseContainer& rhs)
    noexcept(noexcept(DoSwap(std::declval<EmptyMarker&>(), std::declval<EmptyMarker&>())))
    {
        DoSwap(Buckets_, rhs.Buckets_);
        DoSwap(EmptyMarker_, rhs.EmptyMarker_);
        DoSwap(Taken_, rhs.Taken_);
    }

    TDenseContainer Clone(size_type newSize) const { return { newSize, EmptyMarker_, GetAllocator() }; }

protected:
    allocator_type GetAllocator() const {
        return Buckets_.get_allocator();
    }

protected:
    TVector<value_type, allocator_type> Buckets_;
    size_type Taken_;
    EmptyMarker EmptyMarker_;
};

static_assert(NConcepts::ContainerV<TDenseContainer<int>>);
static_assert(!NConcepts::RemovalContainerV<TDenseContainer<int>>);

template <class T, class DeletedMarker = NSet::TEqValueMarker<T>,
          class EmptyMarker = NSet::TEqValueMarker<T>, class Alloc = std::allocator<T>>
class TRemovalDenseContainer : private TDenseContainer<T, EmptyMarker, Alloc> {
private:
    static_assert(NConcepts::ValueMarkerV<DeletedMarker>);

    using TBase = TDenseContainer<T, EmptyMarker>;

public:
    using typename TBase::value_type;
    using typename TBase::size_type;
    using typename TBase::difference_type;
    using typename TBase::allocator_type;
    using typename TBase::pointer;
    using typename TBase::const_pointer;

public:
    TRemovalDenseContainer(
        size_type initSize,
        DeletedMarker deletedMarker = {},
        EmptyMarker emptyMarker = {},
        const allocator_type& alloc = {})
        : TBase(initSize, std::move(emptyMarker), alloc)
        , DeletedMarker_(std::move(deletedMarker))
        , Empty_(initSize) {}

    TRemovalDenseContainer(const TRemovalDenseContainer&) = default;
    TRemovalDenseContainer(TRemovalDenseContainer&&) = default;

    TRemovalDenseContainer& operator=(const TRemovalDenseContainer&) = default;
    TRemovalDenseContainer& operator=(TRemovalDenseContainer&&) = default;

    using TBase::Node;
    using TBase::Size;
    using TBase::Taken;
    using TBase::InitNode;
    using TBase::IsEmpty;

    size_type Empty() const noexcept { return Empty_; }

    template <class... Args>
    void InitNode(size_type idx, Args&&... args) {
        TBase::InitNode(idx, std::forward<Args>(args)...);
        --Empty_;
    }

    void DeleteNode(size_type idx) {
        if constexpr (!std::is_trivially_destructible_v<value_type>) {
            TBase::Node(idx).~value_type();
        }
        new (&TBase::Node(idx)) value_type(DeletedMarker_.Create());
        --TBase::Taken_;
    }

    bool IsTaken(size_type idx) const { return !IsDeleted(idx) && TBase::IsTaken(idx); }
    bool IsDeleted(size_type idx) const { return DeletedMarker_.Equals(Node(idx)); }

    void Swap(TRemovalDenseContainer& rhs)
    noexcept(noexcept(std::declval<TBase>().Swap(std::declval<TBase&>())) &&
             noexcept(DoSwap(std::declval<DeletedMarker&>(), std::declval<DeletedMarker&>())))
    {
        TBase::Swap(rhs);
        DoSwap(DeletedMarker_, rhs.DeletedMarker_);
        DoSwap(Empty_, rhs.Empty_);
    }

    TRemovalDenseContainer Clone(size_type newSize) const {
        return { newSize, DeletedMarker_, TBase::EmptyMarker_, TBase::GetAllocator() };
    }

private:
    DeletedMarker DeletedMarker_;
    size_type Empty_;
};

static_assert(NConcepts::ContainerV<TRemovalDenseContainer<int>>);
static_assert(NConcepts::RemovalContainerV<TRemovalDenseContainer<int>>);

}  // namespace NFlatHash
