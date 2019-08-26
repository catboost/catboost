#pragma once

#include <util/generic/utility.h>
#include <util/generic/yexception.h>

using TRbTreeColorType = bool;

#define RBTreeRed false
#define RBTreeBlack true

struct TRbTreeNodeBase {
    using TColorType = TRbTreeColorType;
    using TBasePtr = TRbTreeNodeBase*;

    TColorType Color_;
    TBasePtr Parent_;
    TBasePtr Left_;
    TBasePtr Right_;
    size_t Children_;

    inline TRbTreeNodeBase() noexcept {
        ReInitNode();
    }

    inline void ReInitNode() noexcept {
        Color_ = RBTreeBlack;
        Parent_ = nullptr;
        Left_ = nullptr;
        Right_ = nullptr;
        Children_ = 1;
    }

    static TBasePtr MinimumNode(TBasePtr x) {
        while (x->Left_ != nullptr)
            x = x->Left_;

        return x;
    }

    static TBasePtr MaximumNode(TBasePtr x) {
        while (x->Right_ != nullptr)
            x = x->Right_;

        return x;
    }

    static TBasePtr ByIndex(TBasePtr x, size_t index) {
        if (x->Left_ != nullptr) {
            if (index < x->Left_->Children_)
                return ByIndex(x->Left_, index);
            index -= x->Left_->Children_;
        }
        if (0 == index)
            return x;
        if (!x->Right_)
            ythrow yexception() << "index not found";
        return ByIndex(x->Right_, index - 1);
    }
};

struct TRbTreeBaseIterator;

template <class TDummy>
class TRbGlobal {
public:
    using TBasePtr = TRbTreeNodeBase*;

    static void Rebalance(TBasePtr x, TBasePtr& root);
    static TBasePtr RebalanceForErase(TBasePtr z, TBasePtr& root, TBasePtr& leftmost, TBasePtr& rightmost);
    static void DecrementChildrenUntilRoot(TBasePtr x, TBasePtr root);
    static void RecalcChildren(TBasePtr x);

    static TBasePtr IncrementNode(TBasePtr);
    static TBasePtr DecrementNode(TBasePtr);
    static void RotateLeft(TBasePtr x, TBasePtr& root);
    static void RotateRight(TBasePtr x, TBasePtr& root);
};

using TRbGlobalInst = TRbGlobal<bool>;

struct TRbTreeBaseIterator {
    using TBasePtr = TRbTreeNodeBase*;
    TBasePtr Node_;

    inline TRbTreeBaseIterator(TBasePtr x = nullptr) noexcept
        : Node_(x)
    {
    }
};

template <class TValue, class TTraits>
struct TRbTreeIterator: public TRbTreeBaseIterator {
    using TReference = typename TTraits::TReference;
    using TPointer = typename TTraits::TPointer;
    using TSelf = TRbTreeIterator<TValue, TTraits>;
    using TBasePtr = TRbTreeNodeBase*;

    inline TRbTreeIterator() noexcept = default;

    template <class T1>
    inline TRbTreeIterator(const T1& x) noexcept
        : TRbTreeBaseIterator(x)
    {
    }

    inline TReference operator*() const noexcept {
        return *static_cast<TValue*>(Node_);
    }

    inline TPointer operator->() const noexcept {
        return static_cast<TValue*>(Node_);
    }

    inline TSelf& operator++() noexcept {
        Node_ = TRbGlobalInst::IncrementNode(Node_);
        return *this;
    }

    inline TSelf operator++(int) noexcept {
        TSelf tmp = *this;
        ++(*this);
        return tmp;
    }

    inline TSelf& operator--() noexcept {
        Node_ = TRbGlobalInst::DecrementNode(Node_);
        return *this;
    }

    inline TSelf operator--(int) noexcept {
        TSelf tmp = *this;
        --(*this);
        return tmp;
    }

    template <class T1>
    inline bool operator==(const T1& rhs) const noexcept {
        return Node_ == rhs.Node_;
    }

    template <class T1>
    inline bool operator!=(const T1& rhs) const noexcept {
        return Node_ != rhs.Node_;
    }
};

template <class TValue, class TCmp>
class TRbTree {
    struct TCmpAdaptor: public TCmp {
        inline TCmpAdaptor() noexcept = default;

        inline TCmpAdaptor(const TCmp& cmp) noexcept
            : TCmp(cmp)
        {
        }

        template <class T1, class T2>
        inline bool operator()(const T1& l, const T2& r) const {
            return TCmp::Compare(l, r);
        }
    };

    struct TNonConstTraits {
        using TReference = TValue&;
        using TPointer = TValue*;
    };

    struct TConstTraits {
        using TReference = const TValue&;
        using TPointer = const TValue*;
    };

    using TNodeBase = TRbTreeNodeBase;
    using TBasePtr = TRbTreeNodeBase*;
    using TColorType = TRbTreeColorType;

public:
    class TRealNode: public TNodeBase {
    public:
        inline TRealNode()
            : Tree_(nullptr)
        {
        }

        inline ~TRealNode() {
            UnLink();
        }

        inline void UnLink() noexcept {
            if (Tree_) {
                Tree_->EraseImpl(this);
                ReInitNode();
                Tree_ = nullptr;
            }
        }

        inline void SetRbTreeParent(TRbTree* parent) noexcept {
            Tree_ = parent;
        }

        inline TRbTree* ParentTree() const noexcept {
            return Tree_;
        }

    private:
        TRbTree* Tree_;
    };

    using TIterator = TRbTreeIterator<TValue, TNonConstTraits>;
    using TConstIterator = TRbTreeIterator<TValue, TConstTraits>;

    inline TRbTree() noexcept {
        Init();
    }

    inline TRbTree(const TCmp& cmp) noexcept
        : KeyCompare_(cmp)
    {
        Init();
    }

    inline void Init() noexcept {
        Data_.Color_ = RBTreeRed;
        Data_.Parent_ = nullptr;
        Data_.Left_ = &Data_;
        Data_.Right_ = &Data_;
        Data_.Children_ = 0;
    }

    struct TDestroy {
        inline void operator()(TValue& v) const noexcept {
            v.SetRbTreeParent(nullptr);
            v.ReInitNode();
        }
    };

    inline ~TRbTree() {
        ForEachNoOrder(TDestroy());
    }

    inline void Clear() noexcept {
        ForEachNoOrder(TDestroy());
        Init();
    }

    template <class F>
    inline void ForEachNoOrder(const F& f) {
        ForEachNoOrder(Root(), f);
    }

    template <class F>
    inline void ForEachNoOrder(TNodeBase* n, const F& f) {
        if (n && n != &Data_) {
            ForEachNoOrder(n->Left_, f);
            ForEachNoOrder(n->Right_, f);
            f(ValueNode(n));
        }
    }

    inline TIterator Begin() noexcept {
        return LeftMost();
    }

    inline TConstIterator Begin() const noexcept {
        return LeftMost();
    }

    inline TIterator End() noexcept {
        return &this->Data_;
    }

    inline TConstIterator End() const noexcept {
        return const_cast<TBasePtr>(&this->Data_);
    }

    inline bool Empty() const noexcept {
        return this->Begin() == this->End();
    }

    inline explicit operator bool() const noexcept {
        return !this->Empty();
    }

    inline TIterator Insert(TValue* val) {
        return Insert(*val);
    }

    inline TIterator Insert(TValue& val) {
        val.UnLink();

        TBasePtr y = &this->Data_;
        TBasePtr x = Root();

        while (x != nullptr) {
            ++(x->Children_);
            y = x;

            if (KeyCompare_(ValueNode(&val), ValueNode(x))) {
                x = LeftNode(x);
            } else {
                x = RightNode(x);
            }
        }

        return InsertImpl(y, &val, x);
    }

    template <class F>
    inline void ForEach(F& f) {
        TIterator it = Begin();

        while (it != End()) {
            f(*it++);
        }
    }

    inline void Erase(TValue& val) noexcept {
        val.UnLink();
    }

    inline void Erase(TValue* val) noexcept {
        Erase(*val);
    }

    inline void Erase(TIterator pos) noexcept {
        Erase(*pos);
    }

    inline void EraseImpl(TNodeBase* val) noexcept {
        TRbGlobalInst::RebalanceForErase(val, this->Data_.Parent_, this->Data_.Left_, this->Data_.Right_);
    }

    template <class T1>
    inline TValue* Find(const T1& k) const {
        TBasePtr y = nullptr;
        TBasePtr x = Root(); // Current node.

        while (x != nullptr)
            if (!KeyCompare_(ValueNode(x), k))
                y = x, x = LeftNode(x);
            else
                x = RightNode(x);

        if (y) {
            if (KeyCompare_(k, ValueNode(y))) {
                y = nullptr;
            }
        }

        return static_cast<TValue*>(y);
    }

    size_t GetIndex(TBasePtr x) const {
        size_t index = 0;

        if (x->Left_ != nullptr) {
            index += x->Left_->Children_;
        }

        while (x != nullptr && x->Parent_ != nullptr && x->Parent_ != const_cast<TBasePtr>(&this->Data_)) {
            if (x->Parent_->Right_ == x && x->Parent_->Left_ != nullptr) {
                index += x->Parent_->Left_->Children_;
            }
            if (x->Parent_->Right_ == x) {
                index += 1;
            }
            x = x->Parent_;
        }

        return index;
    }

    template <class T1>
    inline TBasePtr LowerBound(const T1& k) const {
        TBasePtr y = const_cast<TBasePtr>(&this->Data_); /* Last node which is not less than k. */
        TBasePtr x = Root();                             /* Current node. */

        while (x != nullptr)
            if (!KeyCompare_(ValueNode(x), k))
                y = x, x = LeftNode(x);
            else
                x = RightNode(x);

        return y;
    }

    template <class T1>
    inline TBasePtr UpperBound(const T1& k) const {
        TBasePtr y = const_cast<TBasePtr>(&this->Data_); /* Last node which is greater than k. */
        TBasePtr x = Root();                             /* Current node. */

        while (x != nullptr)
            if (KeyCompare_(k, ValueNode(x)))
                y = x, x = LeftNode(x);
            else
                x = RightNode(x);

        return y;
    }

    template <class T1>
    inline size_t LessCount(const T1& k) const {
        auto x = LowerBound(k);
        if (x == const_cast<TBasePtr>(&this->Data_)) {
            if (const auto root = Root()) {
                return root->Children_;
            } else {
                return 0;
            }
        } else {
            return GetIndex(x);
        }
    }

    template <class T1>
    inline size_t NotLessCount(const T1& k) const {
        return Root()->Children_ - LessCount<T1>(k);
    }

    template <class T1>
    inline size_t GreaterCount(const T1& k) const {
        auto x = UpperBound(k);
        if (x == const_cast<TBasePtr>(&this->Data_)) {
            return 0;
        } else {
            return Root()->Children_ - GetIndex(x);
        }
    }

    template <class T1>
    inline size_t NotGreaterCount(const T1& k) const {
        return Root()->Children_ - GreaterCount<T1>(k);
    }

    TValue* ByIndex(size_t index) {
        return static_cast<TValue*>(TRbTreeNodeBase::ByIndex(Root(), index));
    }

private:
    // CRP 7/10/00 inserted argument on_right, which is another hint (meant to
    // act like on_left and ignore a portion of the if conditions -- specify
    // on_right != nullptr to bypass comparison as false or on_left != nullptr to bypass
    // comparison as true)
    TIterator InsertImpl(TRbTreeNodeBase* parent, TRbTreeNodeBase* val, TRbTreeNodeBase* on_left = nullptr, TRbTreeNodeBase* on_right = nullptr) {
        ValueNode(val).SetRbTreeParent(this);
        TBasePtr new_node = val;

        if (parent == &this->Data_) {
            LeftNode(parent) = new_node;
            // also makes LeftMost() = new_node
            Root() = new_node;
            RightMost() = new_node;
        } else if (on_right == nullptr &&
                   // If on_right != nullptr, the remainder fails to false
                   (on_left != nullptr ||
                    // If on_left != nullptr, the remainder succeeds to true
                    KeyCompare_(ValueNode(val), ValueNode(parent))))
 {
            LeftNode(parent) = new_node;
            if (parent == LeftMost())
                // maintain LeftMost() pointing to min node
                LeftMost() = new_node;
        } else {
            RightNode(parent) = new_node;
            if (parent == RightMost())
                // maintain RightMost() pointing to max node
                RightMost() = new_node;
        }
        ParentNode(new_node) = parent;
        TRbGlobalInst::Rebalance(new_node, this->Data_.Parent_);
        return new_node;
    }

    TBasePtr Root() const {
        return this->Data_.Parent_;
    }

    TBasePtr LeftMost() const {
        return this->Data_.Left_;
    }

    TBasePtr RightMost() const {
        return this->Data_.Right_;
    }

    TBasePtr& Root() {
        return this->Data_.Parent_;
    }

    TBasePtr& LeftMost() {
        return this->Data_.Left_;
    }

    TBasePtr& RightMost() {
        return this->Data_.Right_;
    }

    static TBasePtr& LeftNode(TBasePtr x) {
        return x->Left_;
    }

    static TBasePtr& RightNode(TBasePtr x) {
        return x->Right_;
    }

    static TBasePtr& ParentNode(TBasePtr x) {
        return x->Parent_;
    }

    static TValue& ValueNode(TBasePtr x) {
        return *static_cast<TValue*>(x);
    }

    static TBasePtr MinimumNode(TBasePtr x) {
        return TRbTreeNodeBase::MinimumNode(x);
    }

    static TBasePtr MaximumNode(TBasePtr x) {
        return TRbTreeNodeBase::MaximumNode(x);
    }

private:
    TCmpAdaptor KeyCompare_;
    TNodeBase Data_;
};

template <class TValue, class TCmp>
class TRbTreeItem: public TRbTree<TValue, TCmp>::TRealNode {
};

template <class TDummy>
void TRbGlobal<TDummy>::RotateLeft(TRbTreeNodeBase* x, TRbTreeNodeBase*& root) {
    TRbTreeNodeBase* y = x->Right_;
    x->Right_ = y->Left_;
    if (y->Left_ != nullptr)
        y->Left_->Parent_ = x;
    y->Parent_ = x->Parent_;

    if (x == root)
        root = y;
    else if (x == x->Parent_->Left_)
        x->Parent_->Left_ = y;
    else
        x->Parent_->Right_ = y;
    y->Left_ = x;
    x->Parent_ = y;
    y->Children_ = x->Children_;
    x->Children_ = ((x->Left_) ? x->Left_->Children_ : 0) + ((x->Right_) ? x->Right_->Children_ : 0) + 1;
}

template <class TDummy>
void TRbGlobal<TDummy>::RotateRight(TRbTreeNodeBase* x, TRbTreeNodeBase*& root) {
    TRbTreeNodeBase* y = x->Left_;
    x->Left_ = y->Right_;
    if (y->Right_ != nullptr)
        y->Right_->Parent_ = x;
    y->Parent_ = x->Parent_;

    if (x == root)
        root = y;
    else if (x == x->Parent_->Right_)
        x->Parent_->Right_ = y;
    else
        x->Parent_->Left_ = y;
    y->Right_ = x;
    x->Parent_ = y;
    y->Children_ = x->Children_;
    x->Children_ = ((x->Left_) ? x->Left_->Children_ : 0) + ((x->Right_) ? x->Right_->Children_ : 0) + 1;
}

template <class TDummy>
void TRbGlobal<TDummy>::Rebalance(TRbTreeNodeBase* x, TRbTreeNodeBase*& root) {
    x->Color_ = RBTreeRed;
    while (x != root && x->Parent_->Color_ == RBTreeRed) {
        if (x->Parent_ == x->Parent_->Parent_->Left_) {
            TRbTreeNodeBase* y = x->Parent_->Parent_->Right_;
            if (y && y->Color_ == RBTreeRed) {
                x->Parent_->Color_ = RBTreeBlack;
                y->Color_ = RBTreeBlack;
                x->Parent_->Parent_->Color_ = RBTreeRed;
                x = x->Parent_->Parent_;
            } else {
                if (x == x->Parent_->Right_) {
                    x = x->Parent_;
                    RotateLeft(x, root);
                }
                x->Parent_->Color_ = RBTreeBlack;
                x->Parent_->Parent_->Color_ = RBTreeRed;
                RotateRight(x->Parent_->Parent_, root);
            }
        } else {
            TRbTreeNodeBase* y = x->Parent_->Parent_->Left_;
            if (y && y->Color_ == RBTreeRed) {
                x->Parent_->Color_ = RBTreeBlack;
                y->Color_ = RBTreeBlack;
                x->Parent_->Parent_->Color_ = RBTreeRed;
                x = x->Parent_->Parent_;
            } else {
                if (x == x->Parent_->Left_) {
                    x = x->Parent_;
                    RotateRight(x, root);
                }
                x->Parent_->Color_ = RBTreeBlack;
                x->Parent_->Parent_->Color_ = RBTreeRed;
                RotateLeft(x->Parent_->Parent_, root);
            }
        }
    }
    root->Color_ = RBTreeBlack;
}

template <class TDummy>
void TRbGlobal<TDummy>::RecalcChildren(TRbTreeNodeBase* x) {
    x->Children_ = ((x->Left_) ? x->Left_->Children_ : 0) + ((x->Right_) ? x->Right_->Children_ : 0) + 1;
}

template <class TDummy>
void TRbGlobal<TDummy>::DecrementChildrenUntilRoot(TRbTreeNodeBase* x, TRbTreeNodeBase* root) {
    auto* ptr = x;
    --ptr->Children_;
    while (ptr != root) {
        ptr = ptr->Parent_;
        --ptr->Children_;
    }
}

template <class TDummy>
TRbTreeNodeBase* TRbGlobal<TDummy>::RebalanceForErase(TRbTreeNodeBase* z,
                                                      TRbTreeNodeBase*& root,
                                                      TRbTreeNodeBase*& leftmost,
                                                      TRbTreeNodeBase*& rightmost) {
    TRbTreeNodeBase* y = z;
    TRbTreeNodeBase* x;
    TRbTreeNodeBase* x_parent;

    if (y->Left_ == nullptr) // z has at most one non-null child. y == z.
        x = y->Right_;       // x might be null.
    else {
        if (y->Right_ == nullptr)                        // z has exactly one non-null child. y == z.
            x = y->Left_;                                // x is not null.
        else {                                           // z has two non-null children.  Set y to
            y = TRbTreeNodeBase::MinimumNode(y->Right_); //   z's successor.  x might be null.
            x = y->Right_;
        }
    }

    if (y != z) {
        // relink y in place of z.  y is z's successor
        z->Left_->Parent_ = y;
        y->Left_ = z->Left_;
        if (y != z->Right_) {
            x_parent = y->Parent_;
            if (x)
                x->Parent_ = y->Parent_;
            y->Parent_->Left_ = x; // y must be a child of mLeft
            y->Right_ = z->Right_;
            z->Right_->Parent_ = y;
        } else
            x_parent = y;
        if (root == z)
            root = y;
        else if (z->Parent_->Left_ == z)
            z->Parent_->Left_ = y;
        else
            z->Parent_->Right_ = y;
        y->Parent_ = z->Parent_;
        DoSwap(y->Color_, z->Color_);

        RecalcChildren(y);
        if (x_parent != y) {
            --x_parent->Children_;
        }
        if (x_parent != root) {
            DecrementChildrenUntilRoot(x_parent->Parent_, root);
        }
        y = z;
        // y now points to node to be actually deleted
    } else {
        // y == z
        x_parent = y->Parent_;
        if (x)
            x->Parent_ = y->Parent_;
        if (root == z)
            root = x;
        else {
            if (z->Parent_->Left_ == z)
                z->Parent_->Left_ = x;
            else
                z->Parent_->Right_ = x;
            DecrementChildrenUntilRoot(z->Parent_, root); // we lost y
        }

        if (leftmost == z) {
            if (z->Right_ == nullptr) // z->mLeft must be null also
                leftmost = z->Parent_;
            // makes leftmost == _M_header if z == root
            else
                leftmost = TRbTreeNodeBase::MinimumNode(x);
        }
        if (rightmost == z) {
            if (z->Left_ == nullptr) // z->mRight must be null also
                rightmost = z->Parent_;
            // makes rightmost == _M_header if z == root
            else // x == z->mLeft
                rightmost = TRbTreeNodeBase::MaximumNode(x);
        }
    }

    if (y->Color_ != RBTreeRed) {
        while (x != root && (x == nullptr || x->Color_ == RBTreeBlack))
            if (x == x_parent->Left_) {
                TRbTreeNodeBase* w = x_parent->Right_;
                if (w->Color_ == RBTreeRed) {
                    w->Color_ = RBTreeBlack;
                    x_parent->Color_ = RBTreeRed;
                    RotateLeft(x_parent, root);
                    w = x_parent->Right_;
                }
                if ((w->Left_ == nullptr ||
                     w->Left_->Color_ == RBTreeBlack) &&
                    (w->Right_ == nullptr ||
                     w->Right_->Color_ == RBTreeBlack))
                {
                    w->Color_ = RBTreeRed;
                    x = x_parent;
                    x_parent = x_parent->Parent_;
                } else {
                    if (w->Right_ == nullptr || w->Right_->Color_ == RBTreeBlack) {
                        if (w->Left_)
                            w->Left_->Color_ = RBTreeBlack;
                        w->Color_ = RBTreeRed;
                        RotateRight(w, root);
                        w = x_parent->Right_;
                    }
                    w->Color_ = x_parent->Color_;
                    x_parent->Color_ = RBTreeBlack;
                    if (w->Right_)
                        w->Right_->Color_ = RBTreeBlack;
                    RotateLeft(x_parent, root);
                    break;
                }
            } else {
                // same as above, with mRight <-> mLeft.
                TRbTreeNodeBase* w = x_parent->Left_;
                if (w->Color_ == RBTreeRed) {
                    w->Color_ = RBTreeBlack;
                    x_parent->Color_ = RBTreeRed;
                    RotateRight(x_parent, root);
                    w = x_parent->Left_;
                }
                if ((w->Right_ == nullptr ||
                     w->Right_->Color_ == RBTreeBlack) &&
                    (w->Left_ == nullptr ||
                     w->Left_->Color_ == RBTreeBlack))
                {
                    w->Color_ = RBTreeRed;
                    x = x_parent;
                    x_parent = x_parent->Parent_;
                } else {
                    if (w->Left_ == nullptr || w->Left_->Color_ == RBTreeBlack) {
                        if (w->Right_)
                            w->Right_->Color_ = RBTreeBlack;
                        w->Color_ = RBTreeRed;
                        RotateLeft(w, root);
                        w = x_parent->Left_;
                    }
                    w->Color_ = x_parent->Color_;
                    x_parent->Color_ = RBTreeBlack;
                    if (w->Left_)
                        w->Left_->Color_ = RBTreeBlack;
                    RotateRight(x_parent, root);
                    break;
                }
            }
        if (x)
            x->Color_ = RBTreeBlack;
    }
    return y;
}

template <class TDummy>
TRbTreeNodeBase* TRbGlobal<TDummy>::DecrementNode(TRbTreeNodeBase* Node_) {
    if (Node_->Color_ == RBTreeRed && Node_->Parent_->Parent_ == Node_)
        Node_ = Node_->Right_;
    else if (Node_->Left_ != nullptr) {
        Node_ = TRbTreeNodeBase::MaximumNode(Node_->Left_);
    } else {
        TBasePtr y = Node_->Parent_;
        while (Node_ == y->Left_) {
            Node_ = y;
            y = y->Parent_;
        }
        Node_ = y;
    }
    return Node_;
}

template <class TDummy>
TRbTreeNodeBase* TRbGlobal<TDummy>::IncrementNode(TRbTreeNodeBase* Node_) {
    if (Node_->Right_ != nullptr) {
        Node_ = TRbTreeNodeBase::MinimumNode(Node_->Right_);
    } else {
        TBasePtr y = Node_->Parent_;
        while (Node_ == y->Right_) {
            Node_ = y;
            y = y->Parent_;
        }
        // check special case: This is necessary if mNode is the
        // _M_head and the tree contains only a single node y. In
        // that case parent, left and right all point to y!
        if (Node_->Right_ != y)
            Node_ = y;
    }
    return Node_;
}

#undef RBTreeRed
#undef RBTreeBlack
