#include <library/cpp/containers/flat_hash/lib/containers.h>

#include <library/cpp/testing/unittest/registar.h>

#include <util/generic/algorithm.h>
#include <util/random/random.h>
#include <util/random/shuffle.h>

using namespace NFlatHash;

namespace {
    constexpr size_t INIT_SIZE = 128;

    struct TDummy {
        static size_t Count;

        TDummy() { ++Count; }
        TDummy(const TDummy&) { ++Count; }
        ~TDummy() { --Count; }
    };
    size_t TDummy::Count = 0;

    struct TAlmostDummy {
        static size_t Count;

        TAlmostDummy(int j = 0) : Junk(j) { ++Count; }
        TAlmostDummy(const TAlmostDummy& d) : Junk(d.Junk) { ++Count; }
        ~TAlmostDummy() { --Count; }

        bool operator==(const TAlmostDummy& r) const { return Junk == r.Junk; }
        bool operator!=(const TAlmostDummy& r) const { return !operator==(r); }

        int Junk;
    };
    size_t TAlmostDummy::Count = 0;

    struct TNotSimple {
        enum class EType {
            Value,
            Empty,
            Deleted
        } Type_ = EType::Value;

        TString Junk = "something"; // to prevent triviality propagation
        int Value = 0;

        static int CtorCalls;
        static int DtorCalls;
        static int CopyCtorCalls;
        static int MoveCtorCalls;

        TNotSimple() {
            ++CtorCalls;
        }
        explicit TNotSimple(int value)
            : Value(value)
        {
            ++CtorCalls;
        }

        TNotSimple(const TNotSimple& rhs) {
            ++CopyCtorCalls;
            Value = rhs.Value;
            Type_ = rhs.Type_;
        }
        TNotSimple(TNotSimple&& rhs) {
            ++MoveCtorCalls;
            Value = rhs.Value;
            Type_ = rhs.Type_;
        }

        ~TNotSimple() {
            ++DtorCalls;
        }

        TNotSimple& operator=(const TNotSimple& rhs) {
            ++CopyCtorCalls;
            Value = rhs.Value;
            Type_ = rhs.Type_;
            return *this;
        }
        TNotSimple& operator=(TNotSimple&& rhs) {
            ++MoveCtorCalls;
            Value = rhs.Value;
            Type_ = rhs.Type_;
            return *this;
        }

        static TNotSimple Empty() {
            TNotSimple ret;
            ret.Type_ = EType::Empty;
            return ret;
        }

        static TNotSimple Deleted() {
            TNotSimple ret;
            ret.Type_ = EType::Deleted;
            return ret;
        }

        bool operator==(const TNotSimple& rhs) const noexcept {
            return Value == rhs.Value;
        }

        static void ResetStats() {
            CtorCalls = 0;
            DtorCalls = 0;
            CopyCtorCalls = 0;
            MoveCtorCalls = 0;
        }
    };

    int TNotSimple::CtorCalls = 0;
    int TNotSimple::DtorCalls = 0;
    int TNotSimple::CopyCtorCalls = 0;
    int TNotSimple::MoveCtorCalls = 0;

    struct TNotSimpleEmptyMarker {
        using value_type = TNotSimple;

        value_type Create() const {
            return TNotSimple::Empty();
        }

        bool Equals(const value_type& rhs) const {
            return rhs.Type_ == TNotSimple::EType::Empty;
        }
    };

    struct TNotSimpleDeletedMarker {
        using value_type = TNotSimple;

        value_type Create() const {
            return TNotSimple::Deleted();
        }

        bool Equals(const value_type& rhs) const {
            return rhs.Type_ == TNotSimple::EType::Deleted;
        }
    };

    template <class Container>
    void CheckContainersEqual(const Container& a, const Container& b) {
        UNIT_ASSERT_EQUAL(a.Size(), b.Size());
        UNIT_ASSERT_EQUAL(a.Taken(), b.Empty());
        for (typename Container::size_type i = 0; i < a.Size(); ++i) {
            if (a.IsTaken(i)) {
                UNIT_ASSERT(b.IsTaken(i));
                UNIT_ASSERT_EQUAL(a.Node(i), b.Node(i));
            }
        }
    }

    template <class Container, class... Args>
    void SmokingTest(typename Container::size_type size, Args&&... args) {
        using size_type = typename Container::size_type;
        using value_type = typename Container::value_type;

        Container cont(size, std::forward<Args>(args)...);
        UNIT_ASSERT_EQUAL(cont.Size(), size);
        UNIT_ASSERT_EQUAL(cont.Taken(), 0);

        for (size_type i = 0; i < cont.Size(); ++i) {
            UNIT_ASSERT(cont.IsEmpty(i));
            UNIT_ASSERT(!cont.IsTaken(i));
        }

        // Filling the container till half
        TVector<size_type> toInsert(cont.Size());
        Iota(toInsert.begin(), toInsert.end(), 0);
        Shuffle(toInsert.begin(), toInsert.end());
        toInsert.resize(toInsert.size() / 2);
        for (auto i : toInsert) {
            UNIT_ASSERT(cont.IsEmpty(i));
            UNIT_ASSERT(!cont.IsTaken(i));
            value_type value(RandomNumber<size_type>(cont.Size()));
            cont.InitNode(i, value);
            UNIT_ASSERT(!cont.IsEmpty(i));
            UNIT_ASSERT(cont.IsTaken(i));
            UNIT_ASSERT_EQUAL(cont.Node(i), value);
        }
        UNIT_ASSERT_EQUAL(cont.Taken(), toInsert.size());

        // Copy construction test
        auto cont2 = cont;
        CheckContainersEqual(cont, cont2);

        // Copy assignment test
        cont2 = cont2.Clone(0);
        UNIT_ASSERT_EQUAL(cont2.Size(), 0);
        UNIT_ASSERT_EQUAL(cont2.Taken(), 0);

        // Copy assignment test
        cont2 = cont;
        CheckContainersEqual(cont, cont2);

        // Move construction test
        auto cont3 = std::move(cont2);
        UNIT_ASSERT_EQUAL(cont2.Size(), 0);
        CheckContainersEqual(cont, cont3);

        // Move assignment test
        cont2 = std::move(cont3);
        UNIT_ASSERT_EQUAL(cont3.Size(), 0);
        CheckContainersEqual(cont, cont2);
    }
}

Y_UNIT_TEST_SUITE(TFlatContainerTest) {
    Y_UNIT_TEST(SmokingTest) {
        SmokingTest<TFlatContainer<int>>(INIT_SIZE);
    }

    Y_UNIT_TEST(SmokingTestNotSimpleType) {
        TNotSimple::ResetStats();
        SmokingTest<TFlatContainer<TNotSimple>>(INIT_SIZE);

        UNIT_ASSERT_EQUAL(TNotSimple::CtorCalls + TNotSimple::CopyCtorCalls + TNotSimple::MoveCtorCalls,
                          TNotSimple::DtorCalls);
        UNIT_ASSERT_EQUAL(TNotSimple::CtorCalls, INIT_SIZE / 2 /* created while filling */);
        UNIT_ASSERT_EQUAL(TNotSimple::DtorCalls, INIT_SIZE / 2 /* removed filling temporary */
                                                 + INIT_SIZE / 2 /* removed while cloning */
                                                 + INIT_SIZE /* 3 containers dtors */);
        UNIT_ASSERT_EQUAL(TNotSimple::CopyCtorCalls, INIT_SIZE / 2 /* 3 created while filling */
                                                     + INIT_SIZE / 2 /* created while copy constructing */
                                                     + INIT_SIZE / 2/* created while copy assigning */);
        UNIT_ASSERT_EQUAL(TNotSimple::MoveCtorCalls, 0);
    }

    Y_UNIT_TEST(DummyHalfSizeTest) {
        using TContainer = TFlatContainer<TDummy>;
        using size_type = typename TContainer::size_type;

        {
            TContainer cont(INIT_SIZE);
            UNIT_ASSERT_EQUAL(TDummy::Count, 0);

            TVector<size_type> toInsert(cont.Size());
            Iota(toInsert.begin(), toInsert.end(), 0);
            Shuffle(toInsert.begin(), toInsert.end());
            toInsert.resize(toInsert.size() / 2);
            for (auto i : toInsert) {
                UNIT_ASSERT(cont.IsEmpty(i));
                UNIT_ASSERT(!cont.IsTaken(i));
                cont.InitNode(i);
                UNIT_ASSERT_EQUAL(TDummy::Count, cont.Taken());
                UNIT_ASSERT(!cont.IsEmpty(i));
                UNIT_ASSERT(cont.IsTaken(i));
            }
            UNIT_ASSERT_EQUAL(cont.Taken(), cont.Size() / 2);
            UNIT_ASSERT_EQUAL(TDummy::Count, cont.Taken());
        }
        UNIT_ASSERT_EQUAL(TDummy::Count, 0);
    }

    Y_UNIT_TEST(DeleteTest) {
        using TContainer = TFlatContainer<TDummy>;
        using size_type = typename TContainer::size_type;

        TContainer cont(INIT_SIZE);
        auto idx = RandomNumber<size_type>(INIT_SIZE);
        UNIT_ASSERT(!cont.IsTaken(idx));
        UNIT_ASSERT(!cont.IsDeleted(idx));
        UNIT_ASSERT_EQUAL(TDummy::Count, 0);

        cont.InitNode(idx);
        UNIT_ASSERT_EQUAL(cont.Taken(), 1);
        UNIT_ASSERT(cont.IsTaken(idx));
        UNIT_ASSERT(!cont.IsDeleted(idx));
        UNIT_ASSERT_EQUAL(TDummy::Count, 1);

        cont.DeleteNode(idx);
        UNIT_ASSERT(!cont.IsTaken(idx));
        UNIT_ASSERT(cont.IsDeleted(idx));
        UNIT_ASSERT_EQUAL(TDummy::Count, 0);
    }
}

Y_UNIT_TEST_SUITE(TDenseContainerTest) {
    Y_UNIT_TEST(SmokingTest) {
        SmokingTest<TDenseContainer<int, NSet::TStaticValueMarker<-1>>>(INIT_SIZE);
    }

    Y_UNIT_TEST(NotSimpleTypeSmokingTest) {
        TNotSimple::ResetStats();
        SmokingTest<TDenseContainer<TNotSimple, TNotSimpleEmptyMarker>>(INIT_SIZE);

        UNIT_ASSERT_EQUAL(TNotSimple::CtorCalls + TNotSimple::CopyCtorCalls + TNotSimple::MoveCtorCalls,
                          TNotSimple::DtorCalls);
        UNIT_ASSERT_EQUAL(TNotSimple::CtorCalls, INIT_SIZE / 2 /* created while filling */
                                                 + 2 /* created by empty marker */);
        UNIT_ASSERT_EQUAL(TNotSimple::DtorCalls, 1 /* removed empty marker temporary */
                                                 + INIT_SIZE /* half removed while resetting in container,
                                                                half removed inserted temporary */
                                                 + INIT_SIZE /* removed while cloning */
                                                 + 1 /* removed empty marker temporary */
                                                 + INIT_SIZE * 2 /* 3 containers dtors */);
        UNIT_ASSERT_EQUAL(TNotSimple::CopyCtorCalls, INIT_SIZE /* created while constructing */
                                                     + INIT_SIZE / 2 /* 3 created while filling */
                                                     + INIT_SIZE /* created while copy constructing */
                                                     + INIT_SIZE /* created while copy assigning */);
        UNIT_ASSERT_EQUAL(TNotSimple::MoveCtorCalls, 0);
    }

    Y_UNIT_TEST(RemovalContainerSmokingTest) {
        SmokingTest<TRemovalDenseContainer<int, NSet::TStaticValueMarker<-1>,
                                           NSet::TStaticValueMarker<-2>>>(INIT_SIZE);
    }

    Y_UNIT_TEST(NotSimpleTypeRemovalContainerSmokingTest) {
        TNotSimple::ResetStats();
        SmokingTest<TRemovalDenseContainer<TNotSimple, TNotSimpleEmptyMarker,
                                           TNotSimpleDeletedMarker>>(INIT_SIZE);

        UNIT_ASSERT_EQUAL(TNotSimple::CtorCalls + TNotSimple::CopyCtorCalls + TNotSimple::MoveCtorCalls,
                          TNotSimple::DtorCalls);
        UNIT_ASSERT_EQUAL(TNotSimple::CtorCalls, INIT_SIZE / 2 /* created while filling */
                                                 + 2 /* created by empty marker */);
        UNIT_ASSERT_EQUAL(TNotSimple::DtorCalls, 1 /* removed empty marker temporary */
                                                 + INIT_SIZE /* half removed while resetting in container,
                                                                half removed inserted temporary */
                                                 + INIT_SIZE /* removed while cloning */
                                                 + 1 /* removed empty marker temporary */
                                                 + INIT_SIZE * 2 /* 3 containers dtors */);
        UNIT_ASSERT_EQUAL(TNotSimple::CopyCtorCalls, INIT_SIZE /* created while constructing */
                                                     + INIT_SIZE / 2 /* 3 created while filling */
                                                     + INIT_SIZE /* created while copy constructing */
                                                     + INIT_SIZE /* created while copy assigning */);
        UNIT_ASSERT_EQUAL(TNotSimple::MoveCtorCalls, 0);
    }

    Y_UNIT_TEST(DummyHalfSizeTest) {
        using TContainer = TDenseContainer<TAlmostDummy, NSet::TEqValueMarker<TAlmostDummy>>;
        using size_type = typename TContainer::size_type;

        {
            TContainer cont(INIT_SIZE, TAlmostDummy{-1});
            UNIT_ASSERT_EQUAL(TAlmostDummy::Count, cont.Size() + 1); // 1 for empty marker

            TVector<size_type> toInsert(cont.Size());
            Iota(toInsert.begin(), toInsert.end(), 0);
            Shuffle(toInsert.begin(), toInsert.end());
            toInsert.resize(toInsert.size() / 2);
            for (auto i : toInsert) {
                UNIT_ASSERT(cont.IsEmpty(i));
                UNIT_ASSERT(!cont.IsTaken(i));
                cont.InitNode(i, (int)RandomNumber<size_type>(cont.Size()));
                UNIT_ASSERT_EQUAL(TAlmostDummy::Count, cont.Size() + 1);
                UNIT_ASSERT(!cont.IsEmpty(i));
                UNIT_ASSERT(cont.IsTaken(i));
            }
            UNIT_ASSERT_EQUAL(cont.Taken(), toInsert.size());
            UNIT_ASSERT_EQUAL(TAlmostDummy::Count, cont.Size() + 1);
        }
        UNIT_ASSERT_EQUAL(TAlmostDummy::Count, 0);
    }

    Y_UNIT_TEST(DeleteTest) {
        using TContainer = TRemovalDenseContainer<TAlmostDummy, NSet::TEqValueMarker<TAlmostDummy>,
                                                  NSet::TEqValueMarker<TAlmostDummy>>;
        using size_type = typename TContainer::size_type;

        TContainer cont(INIT_SIZE, TAlmostDummy{ -2 }, TAlmostDummy{ -1 });
        auto idx = RandomNumber<size_type>(cont.Size());
        UNIT_ASSERT(!cont.IsTaken(idx));
        UNIT_ASSERT(!cont.IsDeleted(idx));
        UNIT_ASSERT_EQUAL(TAlmostDummy::Count, cont.Size() + 2); // 2 for markers

        cont.InitNode(idx, (int)RandomNumber<size_type>(cont.Size()));
        UNIT_ASSERT_EQUAL(cont.Taken(), 1);
        UNIT_ASSERT(cont.IsTaken(idx));
        UNIT_ASSERT(!cont.IsDeleted(idx));
        UNIT_ASSERT_EQUAL(TAlmostDummy::Count, cont.Size() + 2);

        cont.DeleteNode(idx);
        UNIT_ASSERT(!cont.IsTaken(idx));
        UNIT_ASSERT(cont.IsDeleted(idx));
        UNIT_ASSERT_EQUAL(TAlmostDummy::Count, cont.Size() + 2);
    }

    Y_UNIT_TEST(FancyInitsTest) {
        {
            using TContainer = TDenseContainer<int>;
            TContainer cont{ INIT_SIZE, -1 };
        }
        {
            using TContainer = TDenseContainer<int, NSet::TStaticValueMarker<-1>>;
            TContainer cont{ INIT_SIZE };
            static_assert(!std::is_constructible_v<TContainer, size_t, int>);
        }
        {
            using TContainer = TDenseContainer<int, NSet::TEqValueMarker<int>>;
            TContainer cont{ INIT_SIZE, -1 };
            TContainer cont2{ INIT_SIZE, NSet::TEqValueMarker<int>{ -1 } };
        }
        {
            using TContainer = TRemovalDenseContainer<int>;
            TContainer cont{ INIT_SIZE, -1, -2 };
            TContainer cont2{ INIT_SIZE, NSet::TEqValueMarker<int>{ -1 },
                              NSet::TEqValueMarker<int>{ -2 } };
        }
        {
            using TContainer = TRemovalDenseContainer<int, NSet::TStaticValueMarker<-1>,
                                                      NSet::TStaticValueMarker<-1>>;
            TContainer cont{ INIT_SIZE };
            static_assert(!std::is_constructible_v<TContainer, size_t, int>);
            static_assert(!std::is_constructible_v<TContainer, size_t, int, int>);
        }
    }
}
