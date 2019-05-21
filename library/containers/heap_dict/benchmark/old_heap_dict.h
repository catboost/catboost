#pragma once

#include <util/generic/vector.h>
#include <util/generic/hash.h>

namespace NDeprecated {
    template <class TKey,
              class TValue,
              class TPriority,
              class TCmp = TLess<TPriority>>
    class THeapDict {
    private:
        struct THeapItem {
            TKey Key;
            TValue Value;
            TPriority Priority;
        };

    public:
        struct TEntry {
            const TKey& Key;
            TValue& Value;
            const TPriority& Priority;

        private:
            friend class THeapDict;

            TEntry(const TKey& key, TValue& value, const TPriority& priority, ui64 position)
                : Key(key)
                , Value(value)
                , Priority(priority)
                , Position(position)
            {
            }

            ui64 Position;
        };

    public:
        THeapDict() = default;

        TEntry GetTop() {
            return GetEntryFromHeap(0);
        }

        TEntry operator[](const TKey& key) {
            auto it = PositionsInHeap.find(key);
            if (it == PositionsInHeap.end()) {
                PushUnique(key, TValue(), TPriority());
                it = PositionsInHeap.find(key);
            }
            return GetEntryFromHeap(it->second);
        }

        void Pop() {
            PopFromPosition(0);
        }

        void Pop(const TKey& key) {
            auto it = PositionsInHeap.find(key);
            if (it == PositionsInHeap.end()) {
                return;
            }
            PopFromPosition(it->second);
        }

        void Pop(const TEntry& entry) {
            PopFromPosition(entry.Position);
        }

        void Push(const TKey& key, const TValue& value, const TPriority& priority) {
            auto it = PositionsInHeap.find(key);
            if (it == PositionsInHeap.end()) {
                PushUnique(key, value, priority);
                return;
            }
            ui64 position = it->second;
            Heap[position].Value = value;
            SetPriority(position, priority);
        }

        void SetPriority(const TKey& key, const TPriority& priority) {
            auto it = PositionsInHeap.find(key);
            Y_VERIFY(it != PositionsInHeap.end());
            SetPriorityAtPosition(it->second, priority);
        }

        void SetPriority(const TEntry& entry, const TPriority& priority) {
            SetPriorityAtPosition(entry.Position, priority);
        }

        bool IsEmpty() const {
            return Heap.empty();
        }
        size_t GetSize() const {
            return Heap.size();
        }

    private:
        static ui64 GetChild(ui64 i, ui64 childId) {
            return 2 * i + childId + 1;
        }

        static ui64 GetParent(ui64 i) {
            return (i - 1) / 2;
        }

        void SwapInHeap(ui64 i, ui64 j) {
            DoSwap(Heap[i], Heap[j]);
            DoSwap(PositionsInHeap[Heap[i].Key], PositionsInHeap[Heap[j].Key]);
        }

        void SiftDown(ui64 i) {
            for (;;) {
                ui64 maxItem = i;
                for (ui64 childId = 0; childId < 2; ++childId) {
                    ui64 child = GetChild(i, childId);
                    if (child < Heap.size() && Cmp(Heap[maxItem].Priority, Heap[child].Priority)) {
                        maxItem = child;
                    }
                }
                if (maxItem == i) {
                    break;
                }
                SwapInHeap(maxItem, i);
                i = maxItem;
            }
        }

        void SiftUp(ui64 i) {
            for (; i > 0;) {
                ui64 parent = GetParent(i);
                if (Cmp(Heap[parent].Priority, Heap[i].Priority)) {
                    SwapInHeap(parent, i);
                    i = parent;
                } else {
                    break;
                }
            }
        }

        void Heapify(ui64 i) {
            SiftDown(i);
            SiftUp(i);
        }

        TEntry GetEntryFromHeap(ui64 i) {
            return {Heap[i].Key, Heap[i].Value, Heap[i].Priority, i};
        }

        void PopFromPosition(ui64 i) {
            if (Heap.size() == 1) {
                Y_VERIFY(i == 0);
                Heap.clear();
                PositionsInHeap.clear();
                return;
            }
            SwapInHeap(i, Heap.size() - 1);
            PositionsInHeap.erase(Heap.back().Key);
            Heap.pop_back();
            if (i < Heap.size()) {
                Heapify(i);
            } else {
                Y_VERIFY(i == Heap.size());
            }
        }

        void PushUnique(const TKey& key, const TValue& value, const TPriority& priority) {
            ui64 position = Heap.size();
            Heap.push_back({key, value, priority});
            PositionsInHeap[key] = position;
            SiftUp(position);
        }

        void SetPriorityAtPosition(ui64 i, const TPriority& priority) {
            Heap[i].Priority = priority;
            Heapify(i);
        }

    private:
        TCmp Cmp;
        THashMap<TKey, ui64> PositionsInHeap;
        TVector<THeapItem> Heap;
    };

}
