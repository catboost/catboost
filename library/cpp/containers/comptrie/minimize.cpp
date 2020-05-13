#include "minimize.h"
#include "node.h"
#include "writeable_node.h"
#include "write_trie_backwards.h"
#include "comptrie_impl.h"

#include <util/generic/hash.h>
#include <util/generic/algorithm.h>

namespace NCompactTrie {
    // Minimize the trie. The result is equivalent to the original
    // trie, except that it takes less space (and has marginally lower
    // performance, because of eventual epsilon links).
    // The algorithm is as follows: starting from the largest pieces, we find
    // nodes that have identical continuations  (Daciuk's right language),
    // and repack the trie. Repacking is done in-place, so memory is less
    // of an issue; however, it may take considerable time.

    // IMPORTANT: never try to reminimize an already minimized trie or a trie with fast layout.
    // Because of non-local structure and epsilon links, it won't work
    // as you expect it to, and can destroy the trie in the making.

    namespace {
        using TOffsetList = TVector<size_t>;
        using TPieceIndex = THashMap<size_t, TOffsetList>;

        using TSizePair = std::pair<size_t, size_t>;
        using TSizePairVector = TVector<TSizePair>;
        using TSizePairVectorVector = TVector<TSizePairVector>;

        class TOffsetMap {
        protected:
            TSizePairVectorVector Data;

        public:
            TOffsetMap() {
                Data.reserve(0x10000);
            }

            void Add(size_t key, size_t value) {
                size_t hikey = key & 0xFFFF;

                if (Data.size() <= hikey)
                    Data.resize(hikey + 1);

                TSizePairVector& sublist = Data[hikey];

                for (auto& it : sublist) {
                    if (it.first == key) {
                        it.second = value;

                        return;
                    }
                }

                sublist.push_back(TSizePair(key, value));
            }

            bool Contains(size_t key) const {
                return (Get(key) != 0);
            }

            size_t Get(size_t key) const {
                size_t hikey = key & 0xFFFF;

                if (Data.size() <= hikey)
                    return 0;

                const TSizePairVector& sublist = Data[hikey];

                for (const auto& it : sublist) {
                    if (it.first == key)
                        return it.second;
                }

                return 0;
            }
        };

        class TOffsetDeltas {
        protected:
            TSizePairVector Data;

        public:
            void Add(size_t key, size_t value) {
                if (Data.empty()) {
                    if (key == value)
                        return; // no offset
                } else {
                    TSizePair last = Data.back();

                    if (key <= last.first) {
                        Cerr << "Trouble: elements to offset delta list added in wrong order" << Endl;

                        return;
                    }

                    if (last.first + value == last.second + key)
                        return; // same  offset
                }

                Data.push_back(TSizePair(key, value));
            }

            size_t Get(size_t key) const {
                if (Data.empty())
                    return key; // difference is zero;

                if (key < Data.front().first)
                    return key;

                // Binary search for the highest entry in the list that does not exceed the key
                size_t from = 0;
                size_t to = Data.size() - 1;

                while (from < to) {
                    size_t midpoint = (from + to + 1) / 2;

                    if (key < Data[midpoint].first)
                        to = midpoint - 1;
                    else
                        from = midpoint;
                }

                TSizePair entry = Data[from];

                return key - entry.first + entry.second;
            }
        };

        class TPieceComparer {
        private:
            const char* Data;
            const size_t Length;

        public:
            TPieceComparer(const char* buf, size_t len)
                : Data(buf)
                , Length(len)
            {
            }

            bool operator()(size_t p1, const size_t p2) {
                int res = memcmp(Data + p1, Data + p2, Length);

                if (res)
                    return (res > 0);

                return (p1 > p2); // the pieces are sorted in the reverse order of appearance
            }
        };

        struct TBranchPoint {
            TNode Node;
            int Selector;

        public:
            TBranchPoint()
                : Selector(0)
            {
            }

            TBranchPoint(const char* data, size_t offset, const ILeafSkipper& skipFunction)
                : Node(data, offset, skipFunction)
                , Selector(0)
            {
            }

            bool IsFinal() const {
                return Node.IsFinal();
            }

            // NextNode returns child nodes, starting from the last node: Right, then Left, then Forward
            size_t NextNode(const TOffsetMap& mergedNodes) {
                while (Selector < 3) {
                    size_t nextOffset = 0;

                    switch (++Selector) {
                        case 1:
                            if (Node.GetRightOffset())
                                nextOffset = Node.GetRightOffset();
                            break;

                        case 2:
                            if (Node.GetLeftOffset())
                                nextOffset = Node.GetLeftOffset();
                            break;

                        case 3:
                            if (Node.GetForwardOffset())
                                nextOffset = Node.GetForwardOffset();
                            break;

                        default:
                            break;
                    }

                    if (nextOffset && !mergedNodes.Contains(nextOffset))
                        return nextOffset;
                }
                return 0;
            }
        };

        class TMergingReverseNodeEnumerator: public TReverseNodeEnumerator {
        private:
            bool Fresh;
            TOpaqueTrie Trie;
            const TOffsetMap& MergeMap;
            TVector<TBranchPoint> Trace;
            TOffsetDeltas OffsetIndex;

        public:
            TMergingReverseNodeEnumerator(const TOpaqueTrie& trie, const TOffsetMap& mergers)
                : Fresh(true)
                , Trie(trie)
                , MergeMap(mergers)
            {
            }

            bool Move() override {
                if (Fresh) {
                    Trace.push_back(TBranchPoint(Trie.Data, 0, Trie.SkipFunction));
                    Fresh = false;
                } else {
                    if (Trace.empty())
                        return false;

                    Trace.pop_back();

                    if (Trace.empty())
                        return false;
                }

                size_t nextnode = Trace.back().NextNode(MergeMap);

                while (nextnode) {
                    Trace.push_back(TBranchPoint(Trie.Data, nextnode, Trie.SkipFunction));
                    nextnode = Trace.back().NextNode(MergeMap);
                }

                return (!Trace.empty());
            }

            const TNode& Get() const {
                return Trace.back().Node;
            }
            size_t GetLeafLength() const override {
                return Get().GetLeafLength();
            }

            // Returns recalculated offset from the end of the current node
            size_t PrepareOffset(size_t absoffset, size_t minilength) {
                if (!absoffset)
                    return NPOS;

                if (MergeMap.Contains(absoffset))
                    absoffset = MergeMap.Get(absoffset);
                return minilength - OffsetIndex.Get(Trie.Length - absoffset);
            }

            size_t RecreateNode(char* buffer, size_t resultLength) override {
                TWriteableNode newNode(Get(), Trie.Data);
                newNode.ForwardOffset = PrepareOffset(Get().GetForwardOffset(), resultLength);
                newNode.LeftOffset = PrepareOffset(Get().GetLeftOffset(), resultLength);
                newNode.RightOffset = PrepareOffset(Get().GetRightOffset(), resultLength);

                if (!buffer)
                    return newNode.Measure();

                const size_t len = newNode.Pack(buffer);
                OffsetIndex.Add(Trie.Length - Get().GetOffset(), resultLength + len);

                return len;
            }
        };

    }

    static void AddPiece(TPieceIndex& index, size_t offset, size_t len) {
        index[len].push_back(offset);
    }

    static TOffsetMap FindEquivalentSubtries(const TOpaqueTrie& trie, bool verbose, size_t minMergeSize) {
        // Tree nodes, arranged by span length.
        // When all nodes of a given size are considered, they pop off the queue.
        TPieceIndex subtries;
        TOffsetMap merger;
        // Start walking the trie from head.
        AddPiece(subtries, 0, trie.Length);

        size_t counter = 0;
        // Now consider all nodes with sizeable continuations
        for (size_t curlen = trie.Length; curlen >= minMergeSize && !subtries.empty(); curlen--) {
            TPieceIndex::iterator iit = subtries.find(curlen);

            if (iit == subtries.end())
                continue; // fast forward to the next available length value

            TOffsetList& batch = iit->second;
            TPieceComparer comparer(trie.Data, curlen);
            Sort(batch.begin(), batch.end(), comparer);

            TOffsetList::iterator it = batch.begin();
            while (it != batch.end()) {
                if (verbose)
                    ShowProgress(++counter);

                size_t offset = *it;

                // Fill the array with the subnodes of the element
                TNode node(trie.Data, offset, trie.SkipFunction);
                size_t end = offset + curlen;
                if (size_t rightOffset = node.GetRightOffset()) {
                    AddPiece(subtries, rightOffset, end - rightOffset);
                    end = rightOffset;
                }
                if (size_t leftOffset = node.GetLeftOffset()) {
                    AddPiece(subtries, leftOffset, end - leftOffset);
                    end = leftOffset;
                }
                if (size_t forwardOffset = node.GetForwardOffset()) {
                    AddPiece(subtries, forwardOffset, end - forwardOffset);
                }

                while (++it != batch.end()) {
                    // Find next different; until then, just add the offsets to the list of merged nodes.
                    size_t nextoffset = *it;

                    if (memcmp(trie.Data + offset, trie.Data + nextoffset, curlen))
                        break;

                    merger.Add(nextoffset, offset);
                }
            }

            subtries.erase(curlen);
        }
        if (verbose) {
            Cerr << counter << Endl;
        }
        return merger;
    }

    size_t RawCompactTrieMinimizeImpl(IOutputStream& os, TOpaqueTrie& trie, bool verbose, size_t minMergeSize, EMinimizeMode mode) {
        if (!trie.Data || !trie.Length) {
            return 0;
        }

        TOffsetMap merger = FindEquivalentSubtries(trie, verbose, minMergeSize);
        TMergingReverseNodeEnumerator enumerator(trie, merger);

        if (mode == MM_DEFAULT)
            return WriteTrieBackwards(os, enumerator, verbose);
        else
            return WriteTrieBackwardsNoAlloc(os, enumerator, trie, mode);
    }

}
