#include "lazy_value.h"

#include <library/cpp/testing/unittest/registar.h>

Y_UNIT_TEST_SUITE(TLazyValueTestSuite) {
    Y_UNIT_TEST(TestLazyValue) {
        TLazyValue<int> value([]() {
            return 5;
        });
        UNIT_ASSERT(!value.WasLazilyInitialized());
        UNIT_ASSERT_EQUAL(*value, 5);
        UNIT_ASSERT(value.WasLazilyInitialized());
    }

    Y_UNIT_TEST(TestLazyValueInitialization) {
        TLazyValue<int> value1([]() { return 5; });

        TLazyValue<int> value2 = []() { return 5; };

        TLazyValue<int> notInitialized{};

        TLazyValue<int> copy1(value1);

        copy1 = value2;
    }

    Y_UNIT_TEST(TestLazyValueCopy) {
        TLazyValue<int> value([]() { return 5; });
        UNIT_ASSERT(!value.WasLazilyInitialized());

        TLazyValue<int> emptyCopy = value;
        UNIT_ASSERT(!emptyCopy.WasLazilyInitialized());

        UNIT_ASSERT_EQUAL(*emptyCopy, 5);
        UNIT_ASSERT(emptyCopy.WasLazilyInitialized());
        UNIT_ASSERT(!value.WasLazilyInitialized());

        UNIT_ASSERT_EQUAL(*value, 5);

        TLazyValue<int> notEmptyCopy = value;
        UNIT_ASSERT(notEmptyCopy.WasLazilyInitialized());
        UNIT_ASSERT_EQUAL(*notEmptyCopy, 5);
    }

    struct TCopyCounter {
        TCopyCounter(size_t& numCopies)
            : NumCopies(&numCopies)
        {
        }

        TCopyCounter() = default;

        TCopyCounter(const TCopyCounter& other)
            : NumCopies(other.NumCopies)
        {
            ++(*NumCopies);
        }

        TCopyCounter(TCopyCounter&&) = default;

        TCopyCounter& operator=(const TCopyCounter& other) {
            if (this == &other) {
                return *this;
            }
            NumCopies = other.NumCopies;
            ++(*NumCopies);
            return *this;
        }

        TCopyCounter& operator=(TCopyCounter&&) = default;

        size_t* NumCopies = nullptr;
    };

    Y_UNIT_TEST(TestLazyValueMoveValueInitialization) {
        size_t numCopies = 0;
        TCopyCounter counter{numCopies};
        TLazyValue<TCopyCounter> value{[v = std::move(counter)]() mutable { return std::move(v); }};
        value.InitDefault();
        UNIT_ASSERT_EQUAL(numCopies, 0);
    }

    Y_UNIT_TEST(TestLazyValueCopyValueInitialization) {
        size_t numCopies = 0;
        TCopyCounter counter{numCopies};
        TLazyValue<TCopyCounter> value{[&counter]() { return counter; }};
        UNIT_ASSERT_EQUAL(numCopies, 0);
        value.InitDefault();
        UNIT_ASSERT_EQUAL(numCopies, 1);
    }

    class TValueProvider {
    public:
        static size_t CountParseDataCalled;

        TValueProvider()
            : Data_([&] { return this->ParseData(); })
        {
        }

        const TString& GetData() const {
            return *Data_;
        }

    private:
        TLazyValue<TString> Data_;

        TString ParseData() {
            CountParseDataCalled++;
            return "hi";
        }
    };

    size_t TValueProvider::CountParseDataCalled = 0;

    Y_UNIT_TEST(TestValueProvider) {
        TValueProvider provider;

        UNIT_ASSERT(provider.GetData() == "hi");
    }

    Y_UNIT_TEST(TestValueProviderCopy) {
        TValueProvider provider;
        provider.GetData();
        const auto countParsed = TValueProvider::CountParseDataCalled;
        provider.GetData();
        UNIT_ASSERT_EQUAL(countParsed, TValueProvider::CountParseDataCalled);

        TValueProvider providerCopy;
        providerCopy = provider;
        providerCopy.GetData();
        UNIT_ASSERT_EQUAL(countParsed, TValueProvider::CountParseDataCalled);
    }

    Y_UNIT_TEST(TestEmptyProviderCopy) {
        TValueProvider provider;
        TValueProvider copy(provider);

        const auto countParsed = TValueProvider::CountParseDataCalled;
        provider.GetData();
        UNIT_ASSERT_EQUAL(countParsed + 1, TValueProvider::CountParseDataCalled);
        copy.GetData();
        UNIT_ASSERT_EQUAL(countParsed + 2, TValueProvider::CountParseDataCalled);
        const TValueProvider notEmptyCopy(copy);
        notEmptyCopy.GetData();
        UNIT_ASSERT_EQUAL(countParsed + 2, TValueProvider::CountParseDataCalled);
    }

    Y_UNIT_TEST(TestMakeLazy) {
        auto lv = MakeLazy([] {
            return 100500;
        });
        UNIT_ASSERT(!lv.WasLazilyInitialized());
        UNIT_ASSERT(lv.GetRef() == 100500);
        UNIT_ASSERT(lv.WasLazilyInitialized());
    }
} // Y_UNIT_TEST_SUITE(TLazyValueTestSuite)
