#include <util/ysafeptr.h>

#include <library/cpp/testing/unittest/registar.h>

#include <util/generic/vector.h>
#include <util/digest/numeric.h>

namespace {

    class TTestObject: public IObjectBase {
        OBJECT_METHODS(TTestObject);

    public:
        TTestObject(int value = 20)
            : Value(value)
        {
        }

        int Value;
    };

    class TDerivedTestObject: public TTestObject {
    public:
        TDerivedTestObject(int value = 50)
            : TTestObject(value)
        {
        }
    };

    class TRecursive: public IObjectBase {
        OBJECT_METHODS(TRecursive);

    public:
        TRecursive(TVector<TObj<TRecursive>> own, TVector<TPtr<TRecursive>> ref)
            : Own(std::move(own))
            , Ref(std::move(ref))
        {
        }

        TRecursive(TObj<TRecursive> own, TVector<TPtr<TRecursive>> ref)
            : TRecursive(TVector<TObj<TRecursive>>{std::move(own)}, std::move(ref))
        {
        }

        TRecursive() = default;

        TVector<TObj<TRecursive>> Own;
        TVector<TPtr<TRecursive>> Ref;
    };

    std::pair<TVector<TObj<TRecursive>>, TVector<TPtr<TRecursive>>> MakeCycle(size_t length) {
        TVector<TObj<TRecursive>> objects;
        TVector<TPtr<TRecursive>> references;
        for (size_t i = 0; i < length; ++i) {
            TObj<TRecursive> p{new TRecursive{
                objects.empty() ? nullptr : objects.back(),
                references}};
            objects.push_back(p.Get());
            references.push_back(p.Get());
        }
        references.front()->Ref.push_back(objects.back().Get());
        return {std::move(objects), std::move(references)};
    }

    std::pair<TVector<TObj<TRecursive>>, TVector<TPtr<TRecursive>>> MakeGraph(size_t size) {
        TVector<TObj<TRecursive>> objects;
        TVector<TPtr<TRecursive>> references;
        for (size_t i = 0; i < size; ++i) {
            TObj<TRecursive> p{new TRecursive{}};
            if (i > 0) {
                for (size_t refsN = 0; refsN < 5; ++refsN) {
                    size_t random = IntHash(i) % references.size();
                    p->Ref.push_back(references[random]);
                }
                size_t parent = (i - 1) >> 1;
                objects[parent]->Own.push_back(p.Get());
                size_t random = IntHash(i) % objects.size();
                objects[random]->Own.push_back(p.Get());
            }
            objects.push_back(p.Get());
            references.push_back(p.Get());
        }
        return {std::move(objects), std::move(references)};
    }
} // namespace

Y_UNIT_TEST_SUITE(SafePtr) {
    Y_UNIT_TEST(Basic) {
        TObj<TTestObject> a = new TTestObject();
    }

    Y_UNIT_TEST(Clear) {
        TObj<TTestObject> a = new TTestObject(30);
        UNIT_ASSERT_VALUES_EQUAL(a->Value, 30);
        a->Clear();
        UNIT_ASSERT_VALUES_EQUAL(a->Value, 20);
    }

    Y_UNIT_TEST(ClearDynamicType) {
        TObj<TTestObject> a = new TDerivedTestObject(30);
        UNIT_ASSERT_VALUES_EQUAL(a->Value, 30);
        a->Clear();
        UNIT_ASSERT_VALUES_EQUAL(a->Value, 20);
    }

    Y_UNIT_TEST(RefCycle) {
        auto [objects, references] = MakeCycle(50);
        for (const auto& ref : references) {
            UNIT_ASSERT(IsValid(ref));
            UNIT_ASSERT_GE(ref->Ref.size(), 0);
        }
        objects.clear();
        for (const auto& ref : references) {
            UNIT_ASSERT(!IsValid(ref));
        }
    }

    Y_UNIT_TEST(RefGraph) {
        auto [objects, references] = MakeGraph(500);
        for (const auto& ref : references) {
            UNIT_ASSERT(IsValid(ref));
        }
        objects.clear();
        for (const auto& ref : references) {
            UNIT_ASSERT(!IsValid(ref));
        }
    }
}; // Y_UNIT_TEST_SUITE(SafePtr)
