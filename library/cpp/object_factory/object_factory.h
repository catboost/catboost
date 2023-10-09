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
        template <class T>
        IFactoryObjectCreator<TProduct, TArgs...>* GetCreator(const T& key) const {
            TReadGuard guard(CreatorsLock);
            typename ICreators::const_iterator i = Creators.find(key);
            return i == Creators.end() ? nullptr : i->second.Get();
        }

        template <class T>
        bool HasImpl(const T& key) const {
            TReadGuard guard(CreatorsLock);
            return Creators.find(key) != Creators.end();
        }

    private:
        typedef TSimpleSharedPtr<IFactoryObjectCreator<TProduct, TArgs...>> ICreatorPtr;
        typedef TMap<TKey, ICreatorPtr> ICreators;
        ICreators Creators;
        TRWMutex CreatorsLock;
    };

    template <class TProduct, class TKey, class... TArgs>
    class TParametrizedObjectFactory: public IObjectFactory<TProduct, TKey, TArgs...> {
    public:
        template <class T>
        TProduct* Create(const T& key, TArgs... args) const {
            IFactoryObjectCreator<TProduct, TArgs...>* creator = IObjectFactory<TProduct, TKey, TArgs...>::GetCreator(key);
            return creator == nullptr ? nullptr : creator->Create(std::forward<TArgs>(args)...);
        }

        template <class T>
        static bool Has(const T& key) {
            return Singleton<TParametrizedObjectFactory<TProduct, TKey, TArgs...>>()->HasImpl(key);
        }

        template <class T>
        static TProduct* Construct(const T& key, const TKey& defKey, TArgs... args) {
            TProduct* result = Singleton<TParametrizedObjectFactory<TProduct, TKey, TArgs...>>()->Create(key, std::forward<TArgs>(args)...);
            if (!result && !!defKey) {
                result = Singleton<TParametrizedObjectFactory<TProduct, TKey, TArgs...>>()->Create(defKey, std::forward<TArgs>(args)...);
            }
            return result;
        }

        template <class T>
        static TProduct* Construct(const T& key, TArgs... args) {
            return Singleton<TParametrizedObjectFactory<TProduct, TKey, TArgs...>>()->Create(key, std::forward<TArgs>(args)...);
        }

        template <class... Args>
        static THolder<TProduct> VerifiedConstruct(Args&&... args) {
            auto result = MakeHolder(std::forward<Args>(args)...);
            Y_ABORT_UNLESS(result, "Construct by factory failed");
            return result;
        }

        template<class... Args>
        static THolder<TProduct> MakeHolder(Args&&... args) {
            return THolder<TProduct>(Construct(std::forward<Args>(args)...));
        }

        static void GetRegisteredKeys(TSet<TKey>& keys) {
            return Singleton<TParametrizedObjectFactory<TProduct, TKey, TArgs...>>()->GetKeys(keys);
        }

        static TSet<TKey> GetRegisteredKeys() {
            TSet<TKey> keys;
            Singleton<TParametrizedObjectFactory<TProduct, TKey, TArgs...>>()->GetKeys(keys);
            return keys;
        }

        template <class TDerivedProduct>
        static TSet<TKey> GetRegisteredKeys() {
            TSet<TKey> registeredKeys(GetRegisteredKeys());
            TSet<TKey> fileredKeys;
            std::copy_if(registeredKeys.begin(), registeredKeys.end(), std::inserter(fileredKeys, fileredKeys.end()), [](const TKey& key) {
                THolder<TProduct> objectHolder(Construct(key));
                return !!dynamic_cast<const TDerivedProduct*>(objectHolder.Get());
            });
            return fileredKeys;
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

    template <class TProduct, class TKey, class... TArgs>
    using TObjectFactory = TParametrizedObjectFactory<TProduct, TKey, TArgs...>;
}
