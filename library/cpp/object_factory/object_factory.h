#pragma once

#include <util/system/guard.h>
#include <util/system/rwlock.h>
#include <util/generic/map.h>
#include <util/generic/set.h>
#include <util/generic/singleton.h>
#include <util/generic/yexception.h>

namespace NObjectFactory {
    template <class TProduct, class... TArgs>
    class IFactoryObjectCreator {
    public:
        virtual TProduct* Create(TArgs... args) const = 0;
        virtual ~IFactoryObjectCreator() {
        }
    };

    template <class TProduct>
    class IFactoryObjectCreator<TProduct, void> {
    public:
        virtual TProduct* Create(void) const = 0;
        virtual ~IFactoryObjectCreator() {
        }
    };

#define FACTORY_OBJECT_NAME(Name)              \
    static TString GetTypeName() {             \
        return #Name;                          \
    }                                          \
    virtual TString GetType() const override { \
        return #Name;                          \
    }

    template <class TBaseProduct, class TDerivedProduct, class... TArgs>
    class TFactoryObjectCreator: public IFactoryObjectCreator<TBaseProduct, TArgs...> {
        TDerivedProduct* Create(TArgs... args) const override {
            return new TDerivedProduct(std::forward<TArgs>(args)...);
        }
    };

    template <class TBaseProduct, class TDerivedProduct>
    class TFactoryObjectCreator<TBaseProduct, TDerivedProduct, void>: public IFactoryObjectCreator<TBaseProduct, void> {
        TDerivedProduct* Create() const override {
            return new TDerivedProduct();
        }
    };

    template <class P, class K, class... TArgs>
    class IObjectFactory {
    public:
        typedef P TProduct;
        typedef K TKey;

    public:
        template <class TDerivedProduct>
        void Register(const TKey& key, IFactoryObjectCreator<TProduct, TArgs...>* creator) {
            if (!creator)
                ythrow yexception() << "Please specify non-null creator for " << key;

            TWriteGuard guard(CreatorsLock);
            if (!Creators.insert(typename ICreators::value_type(key, creator)).second)
                ythrow yexception() << "Product with key " << key << " already registered";
        }

        template <class TDerivedProduct>
        void Register(const TKey& key) {
            Register<TDerivedProduct>(key, new TFactoryObjectCreator<TProduct, TDerivedProduct, TArgs...>);
        }

        void GetKeys(TSet<TKey>& keys) const {
            TReadGuard guard(CreatorsLock);
            keys.clear();
            for (typename ICreators::const_iterator i = Creators.begin(), e = Creators.end(); i != e; ++i) {
                keys.insert(i->first);
            }
        }

    protected:
        IFactoryObjectCreator<TProduct, TArgs...>* GetCreator(const TKey& key) const {
            TReadGuard guard(CreatorsLock);
            typename ICreators::const_iterator i = Creators.find(key);
            return i == Creators.end() ? nullptr : i->second.Get();
        }

        bool HasImpl(const TKey& key) const {
            TReadGuard guard(CreatorsLock);
            return Creators.find(key) != Creators.end();
        }

    private:
        typedef TSimpleSharedPtr<IFactoryObjectCreator<TProduct, TArgs...>> ICreatorPtr;
        typedef TMap<TKey, ICreatorPtr> ICreators;
        ICreators Creators;
        TRWMutex CreatorsLock;
    };

    template <class TProduct, class TKey>
    class TObjectFactory: public IObjectFactory<TProduct, TKey, void> {
    public:
        TProduct* Create(const TKey& key) const {
            IFactoryObjectCreator<TProduct, void>* creator = IObjectFactory<TProduct, TKey, void>::GetCreator(key);
            return creator == nullptr ? nullptr : creator->Create();
        }

        static TString KeysDebugString() {
            TSet<TString> keys;
            Singleton<TObjectFactory<TProduct, TKey>>()->GetKeys(keys);
            TString keysStr;
            for (auto&& k : keys) {
                keysStr += k + " ";
            }
            return keysStr;
        }

        static TProduct* Construct(const TKey& key, const TKey& defKey) {
            TProduct* result = Singleton<TObjectFactory<TProduct, TKey>>()->Create(key);
            if (!result && !!defKey) {
                result = Singleton<TObjectFactory<TProduct, TKey>>()->Create(defKey);
            }
            return result;
        }

        static TProduct* Construct(const TKey& key) {
            TProduct* result = Singleton<TObjectFactory<TProduct, TKey>>()->Create(key);
            return result;
        }

        static bool Has(const TKey& key) {
            return Singleton<TObjectFactory<TProduct, TKey>>()->HasImpl(key);
        }

        static void GetRegisteredKeys(TSet<TKey>& keys) {
            return Singleton<TObjectFactory<TProduct, TKey>>()->GetKeys(keys);
        }

        template <class Product>
        class TRegistrator {
        public:
            TRegistrator(const TKey& key, IFactoryObjectCreator<TProduct, void>* creator) {
                Singleton<TObjectFactory<TProduct, TKey>>()->template Register<Product>(key, creator);
            }

            TRegistrator(const TKey& key) {
                Singleton<TObjectFactory<TProduct, TKey>>()->template Register<Product>(key);
            }

            TRegistrator()
                : TRegistrator(Product::GetTypeName())
            {
            }
        };
    };

    template <class TProduct, class TKey, class... TArgs>
    class TParametrizedObjectFactory: public IObjectFactory<TProduct, TKey, TArgs...> {
    public:
        TProduct* Create(const TKey& key, TArgs... args) const {
            IFactoryObjectCreator<TProduct, TArgs...>* creator = IObjectFactory<TProduct, TKey, TArgs...>::GetCreator(key);
            return creator == nullptr ? nullptr : creator->Create(std::forward<TArgs>(args)...);
        }

        static bool Has(const TKey& key) {
            return Singleton<TParametrizedObjectFactory<TProduct, TKey, TArgs...>>()->HasImpl(key);
        }

        static TProduct* Construct(const TKey& key, TArgs... args) {
            return Singleton<TParametrizedObjectFactory<TProduct, TKey, TArgs...>>()->Create(key, std::forward<TArgs>(args)...);
        }

        static void GetRegisteredKeys(TSet<TKey>& keys) {
            return Singleton<TParametrizedObjectFactory<TProduct, TKey, TArgs...>>()->GetKeys(keys);
        }

        template <class Product>
        class TRegistrator {
        public:
            TRegistrator(const TKey& key, IFactoryObjectCreator<TProduct, TArgs...>* creator) {
                Singleton<TParametrizedObjectFactory<TProduct, TKey, TArgs...>>()->template Register<Product>(key, creator);
            }

            TRegistrator(const TKey& key) {
                Singleton<TParametrizedObjectFactory<TProduct, TKey, TArgs...>>()->template Register<Product>(key);
            }

            TRegistrator()
                : TRegistrator(Product::GetTypeName())
            {
            }

            TString GetName() const {
                return Product::GetTypeName();
            }
        };
    };

}
