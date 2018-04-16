#include <library/object_factory/object_factory.h>
#include <library/unittest/registar.h>

#include <util/generic/string.h>
#include <util/generic/ptr.h>

using namespace NObjectFactory;

struct TArgument {
    TString Name;
    void* Discarded;
};

class ICommonInterface {
public:
    virtual ~ICommonInterface() {
    }

    virtual TString GetValue() const = 0;
};

class TDirectOrder: public ICommonInterface {
public:
    TDirectOrder(const TString& provider, float factor, TArgument& argument)
        : Provider(provider)
        , Factor(factor)
        , Argument(argument)
    {
    }

    TString GetValue() const override {
        return Provider + ToString(Factor) + Argument.Name;
    }

private:
    const TString Provider;
    const float Factor;
    const TArgument Argument;
};

class TInverseOrder: public ICommonInterface {
public:
    TInverseOrder(const TString& provider, float factor, TArgument& argument)
        : Provider(provider)
        , Factor(factor)
        , Argument(argument)
    {
    }

    TString GetValue() const override {
        return Argument.Name + ToString(Factor) + Provider;
    }

private:
    const TString Provider;
    const float Factor;
    const TArgument Argument;
};

struct TDirectOrderCreator: public IFactoryObjectCreator<ICommonInterface, const TString&, float, TArgument&> {
    ICommonInterface* Create(const TString& provider, float& factor, TArgument& argument) const override {
        ++CallsCounter;
        return new TDirectOrder(provider, factor, argument);
    }

    static int CallsCounter;
};
int TDirectOrderCreator::CallsCounter = 0;

using TTestFactory = TParametrizedObjectFactory<ICommonInterface, TString, const TString&, float, TArgument&>;

static TTestFactory::TRegistrator<TDirectOrder> Direct("direct", new TDirectOrderCreator);
static TTestFactory::TRegistrator<TInverseOrder> Inverse("inverse");

SIMPLE_UNIT_TEST_SUITE(TestObjectFactory) {
    SIMPLE_UNIT_TEST(TestParametrized) {
        TArgument directArg{"Name", nullptr};
        TArgument inverseArg{"Fake", nullptr};
        THolder<ICommonInterface> direct(TTestFactory::Construct("direct", "prov", 0.42, directArg));
        THolder<ICommonInterface> inverse(TTestFactory::Construct("inverse", "prov2", 1, inverseArg));

        UNIT_ASSERT(!!direct);
        UNIT_ASSERT(!!inverse);

        UNIT_ASSERT(direct->GetValue() == "prov0.42Name");
        UNIT_ASSERT(inverse->GetValue() == "Fake1prov2");

        UNIT_ASSERT_EQUAL(TDirectOrderCreator::CallsCounter, 1);
    }
}
