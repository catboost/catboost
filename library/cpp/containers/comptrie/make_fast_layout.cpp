#include "make_fast_layout.h"
#include "node.h"
#include "writeable_node.h"
#include "write_trie_backwards.h"
#include "comptrie_impl.h"

#include <util/generic/hash.h>
#include <util/generic/utility.h>

// Lay the trie in memory in such a way that there are less cache misses when jumping from root to leaf.
// The trie becomes about 2% larger, but the access became about 25% faster in our experiments.
// Can be called on minimized and non-minimized tries, in the first case in requires half a trie more memory.
// Calling it the second time on the same trie does nothing.
//
// The algorithm is based on van Emde Boas layout as described in the yandex data school lectures on external memory algoritms
// by Maxim Babenko and Ivan Puzyrevsky. The difference is that when we cut the tree into levels
// two nodes connected by a forward link are put into the same level (because they usually lie near each other in the original tree).
// The original paper (describing the layout in Section 2.1) is:
// Michael A. Bender, Erik D. Demaine, Martin Farach-Colton. Cache-Oblivious B-Trees // SIAM Journal on Computing, volume 35, number 2, 2005, pages 341–358.
// Available on the web: http://erikdemaine.org/papers/CacheObliviousBTrees_SICOMP/
// Or: Michael A. Bender, Erik D. Demaine, and Martin Farach-Colton. Cache-Oblivious B-Trees // Proceedings of the 41st Annual Symposium
// on Foundations of Computer Science (FOCS 2000), Redondo Beach, California, November 12–14, 2000, pages 399–409.
// Available on the web: http://erikdemaine.org/papers/FOCS2000b/
// (there is not much difference between these papers, actually).
//
namespace NCompactTrie {
    static size_t FindSupportingPowerOf2(size_t n) {
        size_t result = 1ull << (8 * sizeof(size_t) - 1);
        while (result > n) {
            result >>= 1;
        }
        return result;
    }

    namespace {
        class TTrieNodeSet {
        public:
            TTrieNodeSet() = default;

            explicit TTrieNodeSet(const TOpaqueTrie& trie)
                : Body(trie.Length / (8 * MinNodeSize) + 1, 0)
            {
            }

            bool Has(size_t offset) const {
                const size_t reducedOffset = ReducedOffset(offset);
                return OffsetCell(reducedOffset) & OffsetMask(reducedOffset);
            }

            void Add(size_t offset) {
                const size_t reducedOffset = ReducedOffset(offset);
                OffsetCell(reducedOffset) |= OffsetMask(reducedOffset);
            }

            void Remove(size_t offset) {
                const size_t reducedOffset = ReducedOffset(offset);
                OffsetCell(reducedOffset) &= ~OffsetMask(reducedOffset);
            }

            void Swap(TTrieNodeSet& other) {
                Body.swap(other.Body);
            }

        private:
            static const size_t MinNodeSize = 2;
            TVector<ui8> Body;

            static size_t ReducedOffset(size_t offset) {
                return offset / MinNodeSize;
            }
            static ui8 OffsetMask(size_t reducedOffset) {
                return 1 << (reducedOffset % 8);
            }
            ui8& OffsetCell(size_t reducedOffset) {
                return Body.at(reducedOffset / 8);
            }
            const ui8& OffsetCell(size_t reducedOffset) const {
                return Body.at(reducedOffset / 8);
            }
        };

        //---------------------------------------------------------------------

        class TTrieNodeCounts {
        public:
            TTrieNodeCounts() = default;

            explicit TTrieNodeCounts(const TOpaqueTrie& trie)
                : Body(trie.Length / MinNodeSize, 0)
                , IsTree(false)
            {
            }

            size_t Get(size_t offset) const {
                return IsTree ? 1 : Body.at(offset / MinNodeSize);
            }

            void Inc(size_t offset) {
                if (IsTree) {
                    return;
                }
                ui8& count = Body.at(offset / MinNodeSize);
                if (count != MaxCount) {
                    ++count;
                }
            }

            size_t Dec(size_t offset) {
                if (IsTree) {
                    return 0;
                }
                ui8& count = Body.at(offset / MinNodeSize);
                Y_ASSERT(count > 0);
                if (count != MaxCount) {
                    --count;
                }
                return count;
            }

            void Swap(TTrieNodeCounts& other) {
                Body.swap(other.Body);
                ::DoSwap(IsTree, other.IsTree);
            }

            void SetTreeMode() {
                IsTree = true;
                Body = TVector<ui8>();
            }

        private:
            static const size_t MinNodeSize = 2;
            static const ui8 MaxCount = 255;

            TVector<ui8> Body;
            bool IsTree = false;
        };

        //----------------------------------------------------------

        class TOffsetIndex {
        public:
            // In all methods:
            // Key --- offset from the beginning of the old trie.
            // Value --- offset from the end of the new trie.

            explicit TOffsetIndex(TTrieNodeCounts& counts) {
                ParentCounts.Swap(counts);
            }

            void Add(size_t key, size_t value) {
                Data[key] = value;
            }

            size_t Size() const {
                return Data.size();
            }

            size_t Get(size_t key) {
                auto pos = Data.find(key);
                if (pos == Data.end()) {
                    ythrow yexception() << "Bad node walking order: trying to get node offset too early or too many times!";
                }
                size_t result = pos->second;
                if (ParentCounts.Dec(key) == 0) {
                    // We don't need this offset any more.
                    Data.erase(pos);
                }
                return result;
            }

        private:
            TTrieNodeCounts ParentCounts;
            THashMap<size_t, size_t> Data;
        };

        //---------------------------------------------------------------------------------------

        class TTrieMeasurer {
        public:
            TTrieMeasurer(const TOpaqueTrie& trie, bool verbose);
            void Measure();

            size_t GetDepth() const {
                return Depth;
            }

            size_t GetNodeCount() const {
                return NodeCount;
            }

            size_t GetUnminimizedNodeCount() const {
                return UnminimizedNodeCount;
            }

            bool IsTree() const {
                return NodeCount == UnminimizedNodeCount;
            }

            TTrieNodeCounts& GetParentCounts() {
                return ParentCounts;
            }

            const TOpaqueTrie& GetTrie() const {
                return Trie;
            }

        private:
            const TOpaqueTrie& Trie;
            size_t Depth;
            TTrieNodeCounts ParentCounts;
            size_t NodeCount;
            size_t UnminimizedNodeCount;
            const bool Verbose;

            // returns depth, increments NodeCount.
            size_t MeasureSubtrie(size_t rootOffset, bool isNewPath);
        };

        TTrieMeasurer::TTrieMeasurer(const TOpaqueTrie& trie, bool verbose)
            : Trie(trie)
            , Depth(0)
            , ParentCounts(trie)
            , NodeCount(0)
            , UnminimizedNodeCount(0)
            , Verbose(verbose)
        {
            Y_ASSERT(Trie.Data);
        }

        void TTrieMeasurer::Measure() {
            if (Verbose) {
                Cerr << "Measuring the trie..." << Endl;
            }
            NodeCount = 0;
            UnminimizedNodeCount = 0;
            Depth = MeasureSubtrie(0, true);
            if (IsTree()) {
                ParentCounts.SetTreeMode();
            }
            if (Verbose) {
                Cerr << "Unminimized node count: " << UnminimizedNodeCount << Endl;
                Cerr << "Trie depth: " << Depth << Endl;
                Cerr << "Node count: " << NodeCount << Endl;
            }
        }

        // A chain of nodes linked by forward links
        // is considered one node with many left and right children
        // for depth measuring here and in
        // TVanEmdeBoasReverseNodeEnumerator::FindDescendants.
        size_t TTrieMeasurer::MeasureSubtrie(size_t rootOffset, bool isNewPath) {
            Y_ASSERT(rootOffset < Trie.Length);
            TNode node(Trie.Data, rootOffset, Trie.SkipFunction);
            size_t depth = 0;
            for (;;) {
                ++UnminimizedNodeCount;
                if (Verbose) {
                    ShowProgress(UnminimizedNodeCount);
                }
                if (isNewPath) {
                    if (ParentCounts.Get(node.GetOffset()) > 0) {
                        isNewPath = false;
                    } else {
                        ++NodeCount;
                    }
                    ParentCounts.Inc(node.GetOffset());
                }
                if (node.GetLeftOffset()) {
                    depth = Max(depth, 1 + MeasureSubtrie(node.GetLeftOffset(), isNewPath));
                }
                if (node.GetRightOffset()) {
                    depth = Max(depth, 1 + MeasureSubtrie(node.GetRightOffset(), isNewPath));
                }
                if (node.GetForwardOffset()) {
                    node = TNode(Trie.Data, node.GetForwardOffset(), Trie.SkipFunction);
                } else {
                    break;
                }
            }
            return depth;
        }

        //--------------------------------------------------------------------------------------

        using TLevelNodes = TVector<size_t>;

        struct TLevel {
            size_t Depth;
            TLevelNodes Nodes;

            explicit TLevel(size_t depth)
                : Depth(depth)
            {
            }
        };

        //----------------------------------------------------------------------------------------

        class TVanEmdeBoasReverseNodeEnumerator: public TReverseNodeEnumerator {
        public:
            TVanEmdeBoasReverseNodeEnumerator(TTrieMeasurer& measurer, bool verbose)
                : Fresh(true)
                , Trie(measurer.GetTrie())
                , Depth(measurer.GetDepth())
                , MaxIndexSize(0)
                , BackIndex(measurer.GetParentCounts())
                , ProcessedNodes(measurer.GetTrie())
                , Verbose(verbose)
            {
            }

            bool Move() override {
                if (!Fresh) {
                    CentralWord.pop_back();
                }
                if (CentralWord.empty()) {
                    return MoveCentralWordStart();
                }
                return true;
            }

            const TNode& Get() const {
                return CentralWord.back();
            }

            size_t GetLeafLength() const override {
                return Get().GetLeafLength();
            }

            // Returns recalculated offset from the end of the current node.
            size_t PrepareOffset(size_t absoffset, size_t resultLength) {
                if (!absoffset)
                    return NPOS;
                return resultLength - BackIndex.Get(absoffset);
            }

            size_t RecreateNode(char* buffer, size_t resultLength) override {
                TWriteableNode newNode(Get(), Trie.Data);
                newNode.ForwardOffset = PrepareOffset(Get().GetForwardOffset(), resultLength);
                newNode.LeftOffset = PrepareOffset(Get().GetLeftOffset(), resultLength);
                newNode.RightOffset = PrepareOffset(Get().GetRightOffset(), resultLength);

                const size_t len = newNode.Pack(buffer);
                ProcessedNodes.Add(Get().GetOffset());
                BackIndex.Add(Get().GetOffset(), resultLength + len);
                MaxIndexSize = Max(MaxIndexSize, BackIndex.Size());
                return len;
            }

        private:
            bool Fresh;
            TOpaqueTrie Trie;
            size_t Depth;
            size_t MaxIndexSize;

            TVector<TLevel> Trace;
            TOffsetIndex BackIndex;
            TVector<TNode> CentralWord;
            TTrieNodeSet ProcessedNodes;

            const bool Verbose;

        private:
            bool IsVisited(size_t offset) const {
                return ProcessedNodes.Has(offset);
            }

            void AddCascade(size_t step) {
                Y_ASSERT(!(step & (step - 1))); // Should be a power of 2.
                while (step > 0) {
                    size_t root = Trace.back().Nodes.back();
                    TLevel level(Trace.back().Depth + step);
                    Trace.push_back(level);
                    size_t actualStep = FindSupportingPowerOf2(FindDescendants(root, step, Trace.back().Nodes));
                    if (actualStep != step) {
                        // Retry with a smaller step.
                        Y_ASSERT(actualStep < step);
                        step = actualStep;
                        Trace.pop_back();
                    } else {
                        step /= 2;
                    }
                }
            }

            void FillCentralWord() {
                CentralWord.clear();
                CentralWord.push_back(TNode(Trie.Data, Trace.back().Nodes.back(), Trie.SkipFunction));
                // Do not check for epsilon links, as the traversal order now is different.
                while (CentralWord.back().GetForwardOffset() && !IsVisited(CentralWord.back().GetForwardOffset())) {
                    CentralWord.push_back(TNode(Trie.Data, CentralWord.back().GetForwardOffset(), Trie.SkipFunction));
                }
            }

            bool MoveCentralWordStart() {
                do {
                    if (Fresh) {
                        TLevel root(0);
                        Trace.push_back(root);
                        Trace.back().Nodes.push_back(0);
                        const size_t sectionDepth = FindSupportingPowerOf2(Depth);
                        AddCascade(sectionDepth);
                        Fresh = false;
                    } else {
                        Trace.back().Nodes.pop_back();
                        if (Trace.back().Nodes.empty() && Trace.size() == 1) {
                            if (Verbose) {
                                Cerr << "Max index size: " << MaxIndexSize << Endl;
                                Cerr << "Current index size: " << BackIndex.Size() << Endl;
                            }
                            // We just popped the root.
                            return false;
                        }
                        size_t lastStep = Trace.back().Depth - Trace[Trace.size() - 2].Depth;
                        if (Trace.back().Nodes.empty()) {
                            Trace.pop_back();
                        }
                        AddCascade(lastStep / 2);
                    }
                } while (IsVisited(Trace.back().Nodes.back()));
                FillCentralWord();
                return true;
            }

            // Returns the maximal depth it has reached while searching.
            // This is a method because it needs OffsetIndex to skip processed nodes.
            size_t FindDescendants(size_t rootOffset, size_t depth, TLevelNodes& result) const {
                if (depth == 0) {
                    result.push_back(rootOffset);
                    return 0;
                }
                size_t actualDepth = 0;
                TNode node(Trie.Data, rootOffset, Trie.SkipFunction);
                for (;;) {
                    if (node.GetLeftOffset() && !IsVisited(node.GetLeftOffset())) {
                        actualDepth = Max(actualDepth,
                                          1 + FindDescendants(node.GetLeftOffset(), depth - 1, result));
                    }
                    if (node.GetRightOffset() && !IsVisited(node.GetRightOffset())) {
                        actualDepth = Max(actualDepth,
                                          1 + FindDescendants(node.GetRightOffset(), depth - 1, result));
                    }
                    if (node.GetForwardOffset() && !IsVisited(node.GetForwardOffset())) {
                        node = TNode(Trie.Data, node.GetForwardOffset(), Trie.SkipFunction);
                    } else {
                        break;
                    }
                }
                return actualDepth;
            }
        };

    } // Anonymous namespace.

    //-----------------------------------------------------------------------------------

    size_t RawCompactTrieFastLayoutImpl(IOutputStream& os, const TOpaqueTrie& trie, bool verbose) {
        if (!trie.Data || !trie.Length) {
            return 0;
        }
        TTrieMeasurer mes(trie, verbose);
        mes.Measure();
        TVanEmdeBoasReverseNodeEnumerator enumerator(mes, verbose);
        return WriteTrieBackwards(os, enumerator, verbose);
    }

}
