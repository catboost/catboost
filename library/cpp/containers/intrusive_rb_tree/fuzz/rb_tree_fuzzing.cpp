#include <library/cpp/containers/intrusive_rb_tree/rb_tree.h>

#include <util/generic/deque.h>
#include <stdint.h>
#include <stddef.h>

struct TCmp {
    template <class T>
    static inline bool Compare(const T& l, const T& r) {
        return l.N < r.N;
    }

    template <class T>
    static inline bool Compare(const T& l, ui8 r) {
        return l.N < r;
    }

    template <class T>
    static inline bool Compare(ui8 l, const T& r) {
        return l < r.N;
    }
};

class TNode: public TRbTreeItem<TNode, TCmp> {
public:
    inline TNode(ui8 n) noexcept
        : N(n)
    {
    }

    ui8 N;
};

using TTree = TRbTree<TNode, TCmp>;

extern "C" int LLVMFuzzerTestOneInput(const ui8* data, size_t size) {
    TDeque<TNode> records;
    const ui8 half = 128u;
    TTree tree;
    for (size_t i = 0; i < size; ++i) {
        if (data[i] / half == 0) {
            records.emplace_back(data[i] % half);
            tree.Insert(&records.back());
        } else {
            auto* ptr = tree.Find(data[i] % half);
            if (ptr != nullptr) {
                tree.Erase(ptr);
            }
        }
        auto check = [](const TNode& node) {
            size_t childrens = 1;
            if (node.Left_) {
                Y_ENSURE(static_cast<const TNode*>(node.Left_)->N <= node.N);
                childrens += node.Left_->Children_;
            }
            if (node.Right_) {
                Y_ENSURE(node.N <= static_cast<const TNode*>(node.Right_)->N);
                childrens += node.Right_->Children_;
            }
            Y_ENSURE(childrens == node.Children_);
        };
        tree.ForEach(check);
    }
    return 0;
}
