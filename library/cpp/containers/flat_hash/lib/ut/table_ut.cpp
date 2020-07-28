#include <library/cpp/containers/flat_hash/lib/containers.h>
#include <library/cpp/containers/flat_hash/lib/expanders.h>
#include <library/cpp/containers/flat_hash/lib/probings.h>
#include <library/cpp/containers/flat_hash/lib/size_fitters.h>
#include <library/cpp/containers/flat_hash/lib/table.h>

#include <library/cpp/testing/unittest/registar.h>

#include <util/generic/algorithm.h>
#include <util/random/random.h>
#include <util/random/shuffle.h>

using namespace NFlatHash;

namespace {
    template <class T>
    struct TJustType {
        using type = T;
    };

    template <class... Ts>
    struct TTypePack {};

    template <class F, class... Ts>
    constexpr void ForEachType(F&& f, TTypePack<Ts...>) {
        ApplyToMany(std::forward<F>(f), TJustType<Ts>{}...);
    }

/* Usage example:
 *
 * TForEachType<int, float, TString>::Apply([](auto t) {
 *     using T = GET_TYPE(t);
 * });
 * So T would be:
 *  int     on #0 iteration
 *  float   on #1 iteration
 *  TString on #2 iteration
 */
#define GET_TYPE(ti) typename decltype(ti)::type

    constexpr size_t INIT_SIZE = 32;
    constexpr size_t BIG_INIT_SIZE = 128;

    template <class T>
    struct TSimpleKeyGetter {
        static constexpr T& Apply(T& t) { return t; }
        static constexpr const T& Apply(const T& t) { return t; }
    };

    template <class T,
              class KeyEqual = std::equal_to<T>,
              class ValueEqual = std::equal_to<T>,
              class KeyGetter = TSimpleKeyGetter<T>,
              class F,
              class... Containers>
    void ForEachTable(F f, TTypePack<Containers...> cs) {
        ForEachType([&](auto p) {
            using TProbing = GET_TYPE(p);

            ForEachType([&](auto sf) {
                using TSizeFitter = GET_TYPE(sf);

                ForEachType([&](auto t) {
                    using TContainer = GET_TYPE(t);
                    static_assert(std::is_same_v<typename TContainer::value_type, T>);

                    using TTable = TTable<THash<T>,
                                          KeyEqual,
                                          TContainer,
                                          KeyGetter,
                                          TProbing,
                                          TSizeFitter,
                                          TSimpleExpander>;

                    f(TJustType<TTable>{});
                }, cs);
            }, TTypePack<TAndSizeFitter, TModSizeFitter>{});
        }, TTypePack<TLinearProbing, TQuadraticProbing, TDenseProbing>{});
    }

    using TAtomContainers = TTypePack<TFlatContainer<int>,
                                      TDenseContainer<int, NSet::TStaticValueMarker<-1>>>;
    using TContainers = TTypePack<TFlatContainer<int>,
                                  TDenseContainer<int, NSet::TStaticValueMarker<-1>>>;
    using TRemovalContainers = TTypePack<TFlatContainer<int>,
                                         TRemovalDenseContainer<int, NSet::TStaticValueMarker<-2>,
                                                                NSet::TStaticValueMarker<-1>>>;
}

Y_UNIT_TEST_SUITE(TCommonTableAtomsTest) {
    Y_UNIT_TEST(InitTest) {
        ForEachTable<int>([](auto t) {
            GET_TYPE(t) table{ INIT_SIZE };

            UNIT_ASSERT(table.empty());
            UNIT_ASSERT_EQUAL(table.size(), 0);
            UNIT_ASSERT_EQUAL(table.bucket_count(), INIT_SIZE);
            UNIT_ASSERT_EQUAL(table.bucket_size(RandomNumber<size_t>(INIT_SIZE)), 0);
        }, TAtomContainers{});
    }

    Y_UNIT_TEST(IteratorTest) {
        ForEachTable<int>([](auto t) {
            GET_TYPE(t) table{ INIT_SIZE };

            auto first = table.begin();
            auto last = table.end();
            UNIT_ASSERT_EQUAL(first, last);
            UNIT_ASSERT_EQUAL(std::distance(first, last), 0);

            auto cFirst = table.cbegin();
            auto cLast = table.cend();
            UNIT_ASSERT_EQUAL(cFirst, cLast);
            UNIT_ASSERT_EQUAL(std::distance(cFirst, cLast), 0);
        }, TAtomContainers{});
    }

    Y_UNIT_TEST(ContainsAndCountTest) {
        ForEachTable<int>([](auto t) {
            GET_TYPE(t) table{ INIT_SIZE };

            for (int i = 0; i < 100; ++i) {
                UNIT_ASSERT_EQUAL(table.count(i), 0);
                UNIT_ASSERT(!table.contains(i));
            }
        }, TAtomContainers{});
    }

    Y_UNIT_TEST(FindTest) {
        ForEachTable<int>([](auto t) {
            GET_TYPE(t) table{ INIT_SIZE };

            for (int i = 0; i < 100; ++i) {
                auto it = table.find(i);
                UNIT_ASSERT_EQUAL(it, table.end());
            }
        }, TAtomContainers{});
    }
}

Y_UNIT_TEST_SUITE(TCommonTableTest) {
    Y_UNIT_TEST(InsertTest) {
        ForEachTable<int>([](auto t) {
            GET_TYPE(t) table{ INIT_SIZE };

            UNIT_ASSERT(table.empty());
            UNIT_ASSERT_EQUAL(table.size(), 0);

            int toInsert = RandomNumber<size_t>(100);
            UNIT_ASSERT_EQUAL(table.count(toInsert), 0);
            UNIT_ASSERT(!table.contains(toInsert));

            auto p = table.insert(toInsert);
            UNIT_ASSERT_EQUAL(p.first, table.begin());
            UNIT_ASSERT(p.second);

            UNIT_ASSERT(!table.empty());
            UNIT_ASSERT_EQUAL(table.size(), 1);
            UNIT_ASSERT_EQUAL(table.count(toInsert), 1);
            UNIT_ASSERT(table.contains(toInsert));

            auto it = table.find(toInsert);
            UNIT_ASSERT_UNEQUAL(it, table.end());
            UNIT_ASSERT_EQUAL(it, table.begin());
            UNIT_ASSERT_EQUAL(*it, toInsert);

            auto p2 = table.insert(toInsert);
            UNIT_ASSERT_EQUAL(p.first, p2.first);
            UNIT_ASSERT(!p2.second);

            UNIT_ASSERT_EQUAL(table.size(), 1);
            UNIT_ASSERT_EQUAL(table.count(toInsert), 1);
            UNIT_ASSERT(table.contains(toInsert));
        }, TContainers{});
    }

    Y_UNIT_TEST(ClearTest) {
        ForEachTable<int>([](auto t) {
            GET_TYPE(t) table{ INIT_SIZE };

            TVector<int> toInsert(INIT_SIZE);
            Iota(toInsert.begin(), toInsert.end(), 0);
            ShuffleRange(toInsert);
            toInsert.resize(INIT_SIZE / 3);

            for (auto i : toInsert) {
                auto p = table.insert(i);
                UNIT_ASSERT_EQUAL(*p.first, i);
                UNIT_ASSERT(p.second);
            }
            UNIT_ASSERT_EQUAL(table.size(), toInsert.size());
            UNIT_ASSERT_EQUAL((size_t)std::distance(table.begin(), table.end()), toInsert.size());

            for (auto i : toInsert) {
                UNIT_ASSERT(table.contains(i));
                UNIT_ASSERT_EQUAL(table.count(i), 1);
            }

            auto bc = table.bucket_count();
            table.clear();
            UNIT_ASSERT(table.empty());
            UNIT_ASSERT_EQUAL(table.bucket_count(), bc);

            for (auto i : toInsert) {
                UNIT_ASSERT(!table.contains(i));
                UNIT_ASSERT_EQUAL(table.count(i), 0);
            }

            table.insert(toInsert.front());
            UNIT_ASSERT(!table.empty());
        }, TContainers{});
    }

    Y_UNIT_TEST(CopyMoveTest) {
        ForEachTable<int>([](auto t) {
            GET_TYPE(t) table{ INIT_SIZE };

            TVector<int> toInsert(INIT_SIZE);
            Iota(toInsert.begin(), toInsert.end(), 0);
            ShuffleRange(toInsert);
            toInsert.resize(INIT_SIZE / 3);

            for (auto i : toInsert) {
                auto p = table.insert(i);
                UNIT_ASSERT_EQUAL(*p.first, i);
                UNIT_ASSERT(p.second);
            }
            UNIT_ASSERT_EQUAL(table.size(), toInsert.size());
            UNIT_ASSERT_EQUAL((size_t)std::distance(table.begin(), table.end()), toInsert.size());

            for (auto i : toInsert) {
                UNIT_ASSERT(table.contains(i));
                UNIT_ASSERT_EQUAL(table.count(i), 1);
            }

            // Copy construction test
            auto table2 = table;
            UNIT_ASSERT_EQUAL(table2.size(), table.size());
            UNIT_ASSERT_EQUAL((size_t)std::distance(table2.begin(), table2.end()), table.size());
            for (auto i : table) {
                UNIT_ASSERT(table2.contains(i));
                UNIT_ASSERT_EQUAL(table2.count(i), 1);
            }

            table2.clear();
            UNIT_ASSERT(table2.empty());

            // Copy assignment test
            table2 = table;
            UNIT_ASSERT_EQUAL(table2.size(), table.size());
            UNIT_ASSERT_EQUAL((size_t)std::distance(table2.begin(), table2.end()), table.size());
            for (auto i : table) {
                UNIT_ASSERT(table2.contains(i));
                UNIT_ASSERT_EQUAL(table2.count(i), 1);
            }

            // Move construction test
            auto table3 = std::move(table2);
            UNIT_ASSERT(table2.empty());
            UNIT_ASSERT(table2.bucket_count() > 0);

            UNIT_ASSERT_EQUAL(table3.size(), table.size());
            UNIT_ASSERT_EQUAL((size_t)std::distance(table3.begin(), table3.end()), table.size());
            for (auto i : table) {
                UNIT_ASSERT(table3.contains(i));
                UNIT_ASSERT_EQUAL(table3.count(i), 1);
            }

            table2.insert(toInsert.front());
            UNIT_ASSERT(!table2.empty());
            UNIT_ASSERT_EQUAL(table2.size(), 1);
            UNIT_ASSERT_UNEQUAL(table2.bucket_count(), 0);

            // Move assignment test
            table2 = std::move(table3);
            UNIT_ASSERT(table3.empty());
            UNIT_ASSERT(table3.bucket_count() > 0);

            UNIT_ASSERT_EQUAL(table2.size(), table.size());
            UNIT_ASSERT_EQUAL((size_t)std::distance(table2.begin(), table2.end()), table.size());
            for (auto i : table) {
                UNIT_ASSERT(table2.contains(i));
                UNIT_ASSERT_EQUAL(table2.count(i), 1);
            }

            table3.insert(toInsert.front());
            UNIT_ASSERT(!table3.empty());
            UNIT_ASSERT_EQUAL(table3.size(), 1);
            UNIT_ASSERT_UNEQUAL(table3.bucket_count(), 0);
        }, TContainers{});
    }

    Y_UNIT_TEST(RehashTest) {
        ForEachTable<int>([](auto t) {
            GET_TYPE(t) table{ INIT_SIZE };

            TVector<int> toInsert(INIT_SIZE);
            Iota(toInsert.begin(), toInsert.end(), 0);
            ShuffleRange(toInsert);
            toInsert.resize(INIT_SIZE / 3);

            for (auto i : toInsert) {
                table.insert(i);
            }

            auto bc = table.bucket_count();
            table.rehash(bc * 2);
            UNIT_ASSERT(bc * 2 <= table.bucket_count());

            UNIT_ASSERT_EQUAL(table.size(), toInsert.size());
            UNIT_ASSERT_EQUAL((size_t)std::distance(table.begin(), table.end()), toInsert.size());
            for (auto i : toInsert) {
                UNIT_ASSERT(table.contains(i));
                UNIT_ASSERT_EQUAL(table.count(i), 1);
            }

            TVector<int> tmp(table.begin(), table.end());
            Sort(toInsert.begin(), toInsert.end());
            Sort(tmp.begin(), tmp.end());

            UNIT_ASSERT_VALUES_EQUAL(tmp, toInsert);

            table.rehash(0);
            UNIT_ASSERT_EQUAL(table.size(), toInsert.size());
            UNIT_ASSERT(table.bucket_count() > table.size());

            table.clear();
            UNIT_ASSERT(table.empty());
            table.rehash(INIT_SIZE);
            UNIT_ASSERT(table.bucket_count() >= INIT_SIZE);

            table.rehash(0);
            UNIT_ASSERT(table.bucket_count() > 0);
        }, TContainers{});
    }

    Y_UNIT_TEST(EraseTest) {
        ForEachTable<int>([](auto t) {
            GET_TYPE(t) table{ INIT_SIZE };

            int value = RandomNumber<ui32>();
            table.insert(value);
            UNIT_ASSERT_EQUAL(table.size(), 1);
            UNIT_ASSERT_EQUAL(table.count(value), 1);

            auto it = table.find(value);
            table.erase(it);

            UNIT_ASSERT_EQUAL(table.size(), 0);
            UNIT_ASSERT_EQUAL(table.count(value), 0);

            table.insert(value);
            UNIT_ASSERT_EQUAL(table.size(), 1);
            UNIT_ASSERT_EQUAL(table.count(value), 1);

            table.erase(value);

            UNIT_ASSERT_EQUAL(table.size(), 0);
            UNIT_ASSERT_EQUAL(table.count(value), 0);

            table.insert(value);
            UNIT_ASSERT_EQUAL(table.size(), 1);
            UNIT_ASSERT_EQUAL(table.count(value), 1);

            table.erase(table.find(value), table.end());

            UNIT_ASSERT_EQUAL(table.size(), 0);
            UNIT_ASSERT_EQUAL(table.count(value), 0);

            table.erase(value);

            UNIT_ASSERT_EQUAL(table.size(), 0);
            UNIT_ASSERT_EQUAL(table.count(value), 0);
        }, TRemovalContainers{});
    }

    Y_UNIT_TEST(EraseBigTest) {
        ForEachTable<int>([](auto t) {
            GET_TYPE(t) table{ BIG_INIT_SIZE };

            for (int i = 0; i < 1000; ++i) {
                for (int j = 0; j < static_cast<int>(BIG_INIT_SIZE); ++j) {
                    table.emplace(j);
                }
                for (int j = 0; j < static_cast<int>(BIG_INIT_SIZE); ++j) {
                    table.erase(j);
                }
            }
            UNIT_ASSERT(table.bucket_count() <= BIG_INIT_SIZE * 8);
        }, TRemovalContainers{});
    }

    Y_UNIT_TEST(ConstructWithSizeTest) {
        ForEachTable<int>([](auto t) {
            GET_TYPE(t) table{ 1000 };
            UNIT_ASSERT(table.bucket_count() >= 1000);

            int value = RandomNumber<ui32>();
            table.insert(value);
            UNIT_ASSERT_EQUAL(table.size(), 1);
            UNIT_ASSERT_EQUAL(table.count(value), 1);
            UNIT_ASSERT(table.bucket_count() >= 1000);

            table.rehash(10);
            UNIT_ASSERT_EQUAL(table.size(), 1);
            UNIT_ASSERT_EQUAL(table.count(value), 1);
            UNIT_ASSERT(table.bucket_count() < 1000);
            UNIT_ASSERT(table.bucket_count() >= 10);
        }, TContainers{});
    }
}
