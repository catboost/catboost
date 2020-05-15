#include <library/cpp/containers/heap_dict/heap_dict.h>

#include <library/cpp/unittest/registar.h>

#include <util/random/fast.h>

Y_UNIT_TEST_SUITE(THeapDictTest) {
    Y_UNIT_TEST(TestBasicStuff) {
        THeapDict<TString, int> heapDict;
        using TItem = THeapDict<TString, int>::value_type;

        heapDict["default"];
        UNIT_ASSERT_EQUAL(heapDict.size(), 1);
        UNIT_ASSERT_EQUAL(heapDict["default"], 0);
        UNIT_ASSERT_EQUAL(heapDict.top().first, "default");
        UNIT_ASSERT_EQUAL(heapDict.top().second, 0);
        UNIT_ASSERT_EQUAL(heapDict.top(), TItem("default", 0));

        heapDict["sameKey"] = 5;
        heapDict.push("sameKey", 6);
        heapDict.push("sameKey", 4);
        UNIT_ASSERT_EQUAL(heapDict.size(), 2);
        UNIT_ASSERT_EQUAL(heapDict["sameKey"], 5);
        UNIT_ASSERT_EQUAL(heapDict.top(), TItem("sameKey", 5));

        heapDict.push("toBeNewTop", 4);
        heapDict.insert(TItem("andTheNextTop", 1));
        heapDict.push("dummy2", 2);
        heapDict.push("dummy3", 3);
        UNIT_ASSERT_EQUAL(heapDict.top(), TItem("sameKey", 5));

        size_t sizeBeforeErase = heapDict.size();
        heapDict.erase("sameKey");
        UNIT_ASSERT_EQUAL(heapDict.size(), sizeBeforeErase - 1);
        heapDict.erase("sameKey");
        UNIT_ASSERT_EQUAL(heapDict.size(), sizeBeforeErase - 1);

        UNIT_ASSERT_EQUAL(heapDict.top(), TItem("toBeNewTop", 4));
        heapDict["andTheNextTop"] = 6;
        UNIT_ASSERT_EQUAL(heapDict.top(), TItem("andTheNextTop", 6));
        heapDict["andCompletelyNewTop"] = 7;
        UNIT_ASSERT_EQUAL(heapDict.top(), TItem("andCompletelyNewTop", 7));
        heapDict["andCompletelyNewTop"] = 5;
        UNIT_ASSERT_EQUAL(heapDict.top(), TItem("andTheNextTop", 6));
        heapDict["andCompletelyNewTop"] = 7;
        UNIT_ASSERT_EQUAL(heapDict.top(), TItem("andCompletelyNewTop", 7));

        {
            THeapDict<TString, int>::iterator it = heapDict.find("unknown");
            UNIT_ASSERT_EQUAL(it, heapDict.end());
        }
        {
            THeapDict<TString, int>::iterator it = heapDict.find("andCompletelyNewTop");
            UNIT_ASSERT_EQUAL(it, heapDict.begin());
            UNIT_ASSERT_UNEQUAL(it, heapDict.end());
            it->second += 2;
            it->second += 1;
            UNIT_ASSERT_EQUAL(heapDict.top(), TItem("andCompletelyNewTop", 10));
            // After calling heapDict.top() [or any other data-accessing method of heapDict]
            // generally iterator "it" is no longer valid,
            // except for one case when the element, that was accessed via the iterator,
            // remained on the same position in heap.
            // Like in this case "it" is still pointing to heapDict.top()
            it->second -= 10;
            UNIT_ASSERT_EQUAL(heapDict.top(), TItem("andTheNextTop", 6));
            // Iterator "it" is no longer valid
            UNIT_ASSERT_EQUAL(heapDict["andCompletelyNewTop"], 0);
        }
        {
            THeapDict<TString, int>::iterator it = heapDict.find("andCompletelyNewTop");
            UNIT_ASSERT_UNEQUAL(it, heapDict.begin());
            UNIT_ASSERT_UNEQUAL(it, heapDict.end());
            it->second += 3;
            // Iterator remains valid, because no other actions on heap were performed.
            size_t sizeBeforeErase2 = heapDict.size();
            heapDict.erase(it);
            UNIT_ASSERT_EQUAL(heapDict.size(), sizeBeforeErase2 - 1);
            UNIT_ASSERT_EQUAL(heapDict.find("andCompletelyNewTop"), heapDict.end());
        }

        size_t sizeBeforePop = heapDict.size();
        heapDict.pop();
        UNIT_ASSERT_EQUAL(heapDict.size(), sizeBeforePop - 1);
        UNIT_ASSERT_EQUAL(heapDict.top(), TItem("toBeNewTop", 4));

        TVector<TItem> sortedItems = {
            {"toBeNewTop", 4},
            {"dummy3", 3},
            {"dummy2", 2},
            {"default", 0}};
        TVector<TItem> itemsFromHeap;
        for (; !heapDict.empty(); heapDict.pop()) {
            itemsFromHeap.push_back(heapDict.top());
        }
        UNIT_ASSERT_EQUAL(itemsFromHeap, sortedItems);
    }

    Y_UNIT_TEST(TestComplexKeyValues) {
        using TKey = std::pair<TString, int>;
        using TValue = std::pair<TString, ui64>;
        THeapDict<TKey, TValue, TGreater<TValue>> heapDict;

        auto getValue = [](const TString& keyL, int keyR) {
            return TValue(keyL + "!", static_cast<ui64>(keyR) * 10);
        };
        auto getKeyValue = [&getValue](const TString& keyL, int keyR) {
            return std::make_pair(TKey(keyL, keyR), getValue(keyL, keyR));
        };

        heapDict[TKey("abc", 1)] = getValue("abc", 1);
        heapDict[TKey("xyz", 0)] = getValue("xyz", 0);
        UNIT_ASSERT_EQUAL(heapDict.top(), getKeyValue("abc", 1));

        heapDict[TKey("aac", 3)] = getValue("aac", 3);
        UNIT_ASSERT_EQUAL(heapDict.top(), getKeyValue("aac", 3));

        heapDict[TKey("aac", 3)].first[1] = 'd';
        UNIT_ASSERT_EQUAL(heapDict.top(), getKeyValue("abc", 1));
        UNIT_ASSERT_EQUAL(heapDict[TKey("aac", 3)], TValue("adc!", 30));

        heapDict[TKey("aac", 3)].first[1] = 'a';
        UNIT_ASSERT_EQUAL(heapDict.top(), getKeyValue("aac", 3));

        heapDict[TKey("aac", 2)] = getValue("aac", 2);
        UNIT_ASSERT_EQUAL(heapDict.top(), getKeyValue("aac", 2));

        heapDict[TKey("aac", 2)].first += "?";
        UNIT_ASSERT_EQUAL(heapDict.top(), getKeyValue("aac", 3));
        UNIT_ASSERT_EQUAL(heapDict[TKey("aac", 2)], TValue("aac!?", 20));
    }

    Y_UNIT_TEST(TestConstIterator) {
        using TItem = THeapDict<TString, int>::value_type;

        TSet<TItem> items = {
            {"a", 1}, {"b", 2}, {"c", 3}, {"x", 1}, {"y", 2}};
        THeapDict<TString, int> heapDict;
        for (const auto& item : items) {
            heapDict.insert(item);
        }
        TSet<TItem> itemsFromHeap;
        for (THeapDict<TString, int>::const_iterator it = heapDict.cbegin(); it != heapDict.cend(); ++it) {
            itemsFromHeap.insert(*it);
        }
        UNIT_ASSERT_EQUAL(items, itemsFromHeap);
    }

    Y_UNIT_TEST(TestFunctionality) {
        const size_t datasetSize = 5000000;
        const size_t numKeys = 3000;
        const i64 maxCount = +30000000LL;
        const i64 minCount = -29000000LL;
        TFastRng<ui64> rng(0);

        THeapDict<ui64, i64> heapDict;

        TSet<std::pair<i64, ui64>> set;
        THashMap<ui64, i64> dict;

        for (size_t i = 0; i < datasetSize; ++i) {
            ui64 key = rng.GenRand() % numKeys;
            i64 count = static_cast<i64>(rng.GenRand() % (maxCount - minCount + 1)) + minCount;
            // update (set, hash_map) pair
            {
                auto& value = dict[key];
                set.erase(std::make_pair(value, key));
                value += count;
                if (value <= 0) {
                    dict.erase(key);
                } else {
                    set.insert(std::make_pair(value, key));
                }
            }
            Y_VERIFY(dict.size() == set.size());
            // update heapDict
            {
                if (i % 2 == 0) {
                    auto& value = heapDict[key];
                    value += count;
                    if (value <= 0) {
                        heapDict.erase(key);
                    }
                } else {
                    auto it = heapDict.find(key);
                    if (it == heapDict.end()) {
                        heapDict.push(key, 0);
                        it = heapDict.find(key);
                    }
                    it->second += count;
                    if (it->second <= 0) {
                        heapDict.erase(it);
                    }
                }
            }
            // compare (set, hash_map) with heapDict
            UNIT_ASSERT_EQUAL(heapDict.size(), set.size());
            if (!heapDict.empty()) {
                // These two asserts could break in a perfectly valid scenario
                // when we generated two keys with equal counts.
                // But since the range [minCount, maxCount] is big,
                // the possibility of that is really low.
                UNIT_ASSERT_EQUAL(heapDict.top().first, set.rbegin()->second);
                UNIT_ASSERT_EQUAL(heapDict.top().second, set.rbegin()->first);
            }
        }
        while (!heapDict.empty()) {
            UNIT_ASSERT_EQUAL(heapDict.top().first, set.rbegin()->second);
            UNIT_ASSERT_EQUAL(heapDict.top().second, set.rbegin()->first);
            set.erase(*set.rbegin());
            switch (heapDict.size() % 3) {
                case 0: {
                    heapDict.pop();
                    break;
                }
                case 1: {
                    heapDict.erase(heapDict.begin());
                    break;
                }
                case 2: {
                    heapDict.erase(heapDict.top().first);
                    break;
                }
            }
        }
        UNIT_ASSERT_EQUAL(heapDict.empty(), set.empty());
    }
}
