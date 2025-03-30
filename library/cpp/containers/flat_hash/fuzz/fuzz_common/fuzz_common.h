#pragma once

#include <util/generic/yexception.h>
#include <util/generic/vector.h>
#include <util/generic/xrange.h>

#include <util/random/random.h>

namespace NFlatHash::NFuzz {

#define FUZZ_ASSERT(cond) \
    Y_ENSURE_EX(cond, TWithBackTrace<yexception>() << Y_STRINGIZE(cond) << " assertion failed ")

#define FUZZ_ASSERT_THROW(cond, exc) \
    try { \
        cond; \
        FUZZ_ASSERT(false); \
    } catch (const exc&) { \
    } catch (...) { \
        FUZZ_ASSERT(false); \
    }

enum EActionType {
    AT_INSERT,
    AT_CLEAR,
    AT_REHASH,
    AT_ATOP,
    AT_AT,
    AT_ITERATORS,
    AT_ERASE,
    AT_FIND
};

template <class EtalonMap, class TesteeMap, class Key, class Value>
void MakeAction(EtalonMap& etalon, TesteeMap& testee, Key&& key, Value&& value, EActionType type) {
    switch (type) {
    case AT_INSERT: {
        auto itEt = etalon.insert({ key, value });
        if (itEt.second) {
            FUZZ_ASSERT(!testee.contains(key));
            auto size = testee.size();
            auto bucket_count = testee.bucket_count();

            auto itTs = testee.insert(std::make_pair(key, value));
            FUZZ_ASSERT(itTs.second);
            FUZZ_ASSERT(itTs.first->first == key);
            FUZZ_ASSERT(itTs.first->second == value);
            FUZZ_ASSERT(size + 1 == testee.size());
            FUZZ_ASSERT(bucket_count <= testee.bucket_count());
        } else {
            FUZZ_ASSERT(testee.contains(key));
            auto size = testee.size();
            auto bucket_count = testee.bucket_count();

            auto itTs = testee.insert(std::make_pair(key, value));
            FUZZ_ASSERT(!itTs.second);
            FUZZ_ASSERT(itTs.first->first == key);
            FUZZ_ASSERT(itTs.first->second == itEt.first->second);
            FUZZ_ASSERT(size == testee.size());
            FUZZ_ASSERT(bucket_count == testee.bucket_count());
        }
        break;
    }
    case AT_CLEAR: {
        auto bucket_count = testee.bucket_count();
        testee.clear();
        for (const auto& v : etalon) {
            FUZZ_ASSERT(!testee.contains(v.first));
        }
        FUZZ_ASSERT(testee.empty());
        FUZZ_ASSERT(testee.size() == 0);
        FUZZ_ASSERT(testee.bucket_count() == bucket_count);
        FUZZ_ASSERT(testee.load_factor() < std::numeric_limits<float>::epsilon());

        etalon.clear();
        break;
    }
    case AT_REHASH: {
        testee.rehash(key);
        FUZZ_ASSERT(testee.bucket_count() >= key);
        break;
    }
    case AT_ATOP: {
        if (etalon.contains(key)) {
            FUZZ_ASSERT(testee.contains(key));
            auto size = testee.size();
            auto bucket_count = testee.bucket_count();

            FUZZ_ASSERT(testee[key] == etalon[key]);

            FUZZ_ASSERT(size == testee.size());
            FUZZ_ASSERT(bucket_count == testee.bucket_count());
        } else {
            FUZZ_ASSERT(!testee.contains(key));
            auto size = testee.size();
            auto bucket_count = testee.bucket_count();

            FUZZ_ASSERT(testee[key] == etalon[key]);

            FUZZ_ASSERT(size + 1 == testee.size());
            FUZZ_ASSERT(bucket_count <= testee.bucket_count());
        }
        auto size = testee.size();
        auto bucket_count = testee.bucket_count();

        etalon[key] = value;
        testee[key] = value;
        FUZZ_ASSERT(testee[key] == etalon[key]);
        FUZZ_ASSERT(testee[key] == value);

        FUZZ_ASSERT(size == testee.size());
        FUZZ_ASSERT(bucket_count == testee.bucket_count());
        break;
    }
    case AT_AT: {
        auto size = testee.size();
        auto bucket_count = testee.bucket_count();
        if (etalon.contains(key)) {
            FUZZ_ASSERT(testee.contains(key));

            FUZZ_ASSERT(testee.at(key) == etalon.at(key));
            testee.at(key) = value;
            etalon.at(key) = value;
            FUZZ_ASSERT(testee.at(key) == etalon.at(key));
        } else {
            FUZZ_ASSERT(!testee.contains(key));
            FUZZ_ASSERT_THROW(testee.at(key) = value, std::out_of_range);
            FUZZ_ASSERT(!testee.contains(key));
        }
        FUZZ_ASSERT(size == testee.size());
        FUZZ_ASSERT(bucket_count == testee.bucket_count());
        break;
    }
    case AT_ITERATORS: {
        auto itBeginTs = testee.begin();
        auto itEndTs = testee.end();
        FUZZ_ASSERT((size_t)std::distance(itBeginTs, itEndTs) == testee.size());
        FUZZ_ASSERT(std::distance(itBeginTs, itEndTs) ==
                    std::distance(etalon.begin(), etalon.end()));
        FUZZ_ASSERT(std::distance(testee.cbegin(), testee.cend()) ==
                    std::distance(etalon.cbegin(), etalon.cend()));
        break;
    }
    case AT_ERASE: {
        if (etalon.contains(key)) {
            FUZZ_ASSERT(testee.contains(key));
            auto size = testee.size();
            auto bucket_count = testee.bucket_count();

            auto itTs = testee.find(key);
            FUZZ_ASSERT(itTs->first == key);
            FUZZ_ASSERT(itTs->second == etalon.at(key));

            testee.erase(itTs);
            FUZZ_ASSERT(size - 1 == testee.size());
            FUZZ_ASSERT(bucket_count == testee.bucket_count());
            etalon.erase(key);
        } else {
            FUZZ_ASSERT(!testee.contains(key));
        }
        break;
    }
    case AT_FIND: {
        auto itEt = etalon.find(key);
        if (itEt != etalon.end()) {
            FUZZ_ASSERT(testee.contains(key));

            auto itTs = testee.find(key);
            FUZZ_ASSERT(itTs != testee.end());
            FUZZ_ASSERT(itTs->first == key);
            FUZZ_ASSERT(itTs->second == itEt->second);

            itTs->second = value;
            itEt->second = value;
        } else {
            FUZZ_ASSERT(!testee.contains(key));

            auto itTs = testee.find(key);
            FUZZ_ASSERT(itTs == testee.end());
        }
        break;
    }
    };
}

template <class EtalonMap, class TesteeMap>
void CheckInvariants(const EtalonMap& etalon, const TesteeMap& testee) {
    using value_type = std::pair<typename TesteeMap::key_type,
          typename TesteeMap::mapped_type>;
    using size_type = typename TesteeMap::size_type;

    TVector<value_type> etalonVals{ etalon.begin(), etalon.end() };
    std::sort(etalonVals.begin(), etalonVals.end());
    TVector<value_type> testeeVals{ testee.begin(), testee.end() };
    std::sort(testeeVals.begin(), testeeVals.end());

    FUZZ_ASSERT(testeeVals == etalonVals);

    FUZZ_ASSERT(testee.size() == etalon.size());
    FUZZ_ASSERT(testee.empty() == etalon.empty());
    FUZZ_ASSERT(testee.load_factor() < 0.5f + std::numeric_limits<float>::epsilon());
    FUZZ_ASSERT(testee.bucket_count() > testee.size());

    size_type buckets = 0;
    for (auto b : xrange(testee.bucket_count())) {
        buckets += testee.bucket_size(b);
    }
    FUZZ_ASSERT(buckets == testee.size());

    for (const auto& v : etalon) {
        auto key = v.first;
        auto value = v.second;

        FUZZ_ASSERT(testee.contains(key));
        FUZZ_ASSERT(testee.count(key) == 1);

        auto it = testee.find(key);
        FUZZ_ASSERT(it->first == key);
        FUZZ_ASSERT(it->second == value);
    }
}

}  // namespace NFlatHash::NFuzz
