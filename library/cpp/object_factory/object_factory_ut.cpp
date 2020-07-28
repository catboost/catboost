#include <library/cpp/object_factory/object_factory.h>
#include <library/cpp/testing/unittest/registar.h>

#include <util/generic/noncopyable.h>
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
    ICommonInterface* Create(const TString& provider, float factor, TArgument& argument) const override {
        ++CallsCounter;
        return new TDirectOrder(provider, factor, argument);
    }

    static int CallsCounter;
};
int TDirectOrderCreator::CallsCounter = 0;

using TTestFactory = TParametrizedObjectFactory<ICommonInterface, TString, const TString&, float, TArgument&>;

static TTestFactory::TRegistrator<TDirectOrder> Direct("direct", new TDirectOrderCreator);
static TTestFactory::TRegistrator<TInverseOrder> Inverse("inverse");



class IMoveableOnlyInterface {
public:
    virtual ~IMoveableOnlyInterface() {
    }

    virtual TString GetValue() const = 0;
};

class TMoveableOnly: public IMoveableOnlyInterface, public TMoveOnly {
public:
    TMoveableOnly(TString&& value)
        : Value(value)
    {}

    TString GetValue() const override {
        return Value;
    }

private:
    const TString Value;
};


using TMoveableOnlyFactory = TParametrizedObjectFactory<IMoveableOnlyInterface, TString, TString&&>;

static TMoveableOnlyFactory::TRegistrator<TMoveableOnly> MoveableOnlyReg("move");



class TMoveableOnly2: public IMoveableOnlyInterface, public TMoveOnly {
public:
    TMoveableOnly2(THolder<TString>&& value)
        : Value(std::move(value))
    {}

    TString GetValue() const override {
        return *Value;
    }

private:
    const THolder<TString> Value;
};


using TMoveableOnly2Factory = TParametrizedObjectFactory<IMoveableOnlyInterface, TString, THolder<TString>&&>;

static TMoveableOnly2Factory::TRegistrator<TMoveableOnly2> MoveableOnly2Reg("move2");

class TDirectOrderDifferentSignature : public TDirectOrder {
public:
    TDirectOrderDifferentSignature(const TString& provider, TArgument& argument) :
        TDirectOrder(provider, 0.01f, argument)
    {
    }

};

struct TDirectOrderDSCreator: public IFactoryObjectCreator<ICommonInterface, const TString&, float, TArgument&> {
    ICommonInterface* Create(const TString& provider, float factor, TArgument& argument) const override {
        Y_UNUSED(factor);
        return new TDirectOrderDifferentSignature(provider, argument);
    }
};


static TTestFactory::TRegistrator<TDirectOrderDifferentSignature> DirectDs("direct_ds", new TDirectOrderDSCreator);

Y_UNIT_TEST_SUITE(TestObjectFactory) {
    Y_UNIT_TEST(TestParametrized) {
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

    Y_UNIT_TEST(TestMoveableOnly) {
        TString v = "value1";

        THolder<IMoveableOnlyInterface> moveableOnly(TMoveableOnlyFactory::Construct("move", std::move(v)));

        UNIT_ASSERT(!!moveableOnly);

        UNIT_ASSERT(moveableOnly->GetValue() == "value1");
    }

    Y_UNIT_TEST(TestMoveableOnly2) {
        THolder<TString> v = MakeHolder<TString>("value2");

        THolder<IMoveableOnlyInterface> moveableOnly2(TMoveableOnly2Factory::Construct("move2", std::move(v)));

        UNIT_ASSERT(!!moveableOnly2);

        UNIT_ASSERT(moveableOnly2->GetValue() == "value2");
    }

    Y_UNIT_TEST(TestDifferentSignature) {
        TArgument directArg{"Name", nullptr};
        THolder<ICommonInterface> directDs(TTestFactory::Construct("direct_ds", "prov", 0.42, directArg));

        UNIT_ASSERT(!!directDs);

        UNIT_ASSERT_EQUAL(directDs->GetValue(), "prov0.01Name");
    }
}
