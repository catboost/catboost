#pragma once

#include <library/cpp/pybind/method.h>
#include <library/cpp/pybind/typedesc.h>
#include <library/cpp/pybind/module.h>
#include <util/generic/hash.h>
#include <util/generic/hash_set.h>
#include <util/generic/string.h>
#include <util/generic/xrange.h>

#include <tuple>

namespace NPyBind {
#define DEFINE_CONVERTERS_IMPL(TClass)                            \
    PyObject* BuildPyObject(typename TClass::TBase&& base) {      \
        return TClass::BuildPyObject(std::move(base));            \
    }                                                             \
    PyObject* BuildPyObject(const typename TClass::TBase& base) { \
        return TClass::BuildPyObject(base);                       \
    }

#define DEFINE_CONVERTERS(function) DEFINE_CONVERTERS_IMPL(TFunctionResult<decltype(function)>)

#define DEFINE_TRANSFORMERS_IMPL(TClass)                                                                              \
    template <>                                                                                                       \
    bool ::NPyBind::FromPyObject<typename TClass::TBase*>(PyObject * obj, typename TClass::TBase * &res) {            \
        res = TClass::CastToObject(obj);                                                                              \
        return res != nullptr;                                                                                        \
    }                                                                                                                 \
    template <>                                                                                                       \
    bool ::NPyBind::FromPyObject<typename TClass::TBase const*>(PyObject * obj, typename TClass::TBase const*& res) { \
        res = TClass::CastToObject(obj);                                                                              \
        return res != nullptr;                                                                                        \
    }

#define DEFINE_TRANSFORMERS(function) DEFINE_TRANSFORMERS_IMPL(TFunctionResult<decltype(function)>)

    namespace Detail {
        struct IGetContextBase {
            virtual ~IGetContextBase() = default;
        };
    } //Detail
    struct TPyModuleDefinition {
        static void InitModule(const TString& name);
        static TPyModuleDefinition& GetModule();

        TString Name;
        NPyBind::TPyObjectPtr M;
        THashMap<TString, PyTypeObject*> ClassName2Type;
        THashMap<TString, Detail::IGetContextBase*> Class2ContextGetter;
    };

    namespace Detail {
        // Manages modules lifecycle
        // IMPORTANT!!! Don't use it in PyBind v1 environment, it will lead to inconsistent state of v1 module
        // UnnamedModule-> new unnamed module stub, this stub become current module. In this case you can add functions to it
        // InitModuleWithName -> convert unnamed module into named one, now you can switch to it in switch, this module remains current
        // SwitchToModule switches to the particular module in registry, this module becomes current.
        class TPyModuleRegistry {
        private:
            TPyModuleRegistry();
            TPyModuleRegistry(const TPyModuleRegistry&) = delete;
            TPyModuleRegistry& operator=(TPyModuleRegistry&) = delete;
        public:
            static TPyModuleRegistry& Get() {
                static TPyModuleRegistry registry;
                return registry;
            }
            TPyModuleDefinition& GetCurrentModule() {
                if (!CurrentModule) {
                    GetUnnamedModule();
                }
                return *CurrentModule;
            }

            TPyModuleDefinition& GetUnnamedModule() {
                if (!UnnamedModule) {
                    UnnamedModule = TPyModuleDefinition();
                    CurrentModule = const_cast<TPyModuleDefinition*>(UnnamedModule.Get());
                }
                return *UnnamedModule;
            }

            TPyModuleDefinition& InitModuleWithName(const TString& name) {
                if (!UnnamedModule) {
                    GetUnnamedModule();
                }
                Name2Def[name] = *UnnamedModule;
                UnnamedModule.Clear();
                CurrentModule = &Name2Def[name];
                return *CurrentModule;
            }

            TPyModuleDefinition& SwitchToModuleByName(const TString& name) {
                Y_ENSURE(Name2Def.contains(name));
                Y_ENSURE(UnnamedModule.Empty());
                CurrentModule = &Name2Def[name];
                return *CurrentModule;
            }

            const THashMap<TString, TPyModuleDefinition>& GetCurrentModules() const {
                return Name2Def;
            }
        private:
            TPyModuleDefinition* CurrentModule = nullptr;
            TMaybe<TPyModuleDefinition> UnnamedModule;//
            THashMap<TString, TPyModuleDefinition> Name2Def;
        };
    }//Detail

    inline void TPyModuleDefinition::InitModule(const TString& name) {
        Detail::TPyModuleRegistry::Get().GetUnnamedModule() = TPyModuleDefinition{name, TModuleHolder::Instance().InitModule(name), {}, {}};
        Detail::TPyModuleRegistry::Get().InitModuleWithName(name);
    }

    inline TPyModuleDefinition& TPyModuleDefinition::GetModule() {
        return Detail::TPyModuleRegistry::Get().GetCurrentModule();
    }

    namespace Detail {
        template <class TPythonType>
        struct TNameCtx {
            TString ClassShortName;
            static TNameCtx& GetNameCtx() {
                static TNameCtx result;
                return result;
            }
        };

        struct TParentData {
            PyTypeObject* ParentType;
            IGetContextBase* ParentContext;
        };

        TVector<PyTypeObject*> GetParentTypes(const TVector<TParentData>& parentsData);

        template <class TBase>
        struct TContextImpl {
            TVector<TParentData> ParentsData;
            TString ClassShortName;
            TString ClassFullName;
            TString ClassDescription;


            TVector<std::pair<TString, typename TPythonTypeAttributes<TBase>::TCallerPtr>> ListCallers;
            TVector<std::pair<TString, typename TPythonTypeAttributes<TBase>::TGetterPtr>> ListGetters;
            TVector<std::pair<TString, typename TPythonTypeAttributes<TBase>::TSetterPtr>> ListSetters;
        };

        template <class TObject>
        struct IGetContext: public IGetContextBase {
            virtual ~IGetContext() = default;
            virtual const TContextImpl<TObject>& GetContext() const = 0;
        };

        template <typename THolderClass, typename TBaseClass, bool ShouldEnable, typename=std::enable_if_t<!ShouldEnable || !std::is_default_constructible_v<TBaseClass>>>
        THolderClass* DoInitPureObject(const TVector<TString>&) {
            ythrow yexception() << "Can't create this object in pure mode from python";
        }

        template <typename THolderClass, typename TBaseClass, bool ShouldEnable, typename=std::enable_if_t<ShouldEnable && std::is_default_constructible_v<TBaseClass>>, typename=void>
        THolderClass* DoInitPureObject(const TVector<TString>&) {
            return new THolderClass(MakeHolder<TBaseClass>());
        }

        using TParentClassResolver = std::function<TString(const TString&, const THashSet<TString>&)>;

        TString DefaultParentResolver(const TString&, const THashSet<TString>& parentModules);

        template <typename TParentTypesTuple, int Ind>
        void InitializeBasesArray(TVector<TParentData>& parentsData, const TParentClassResolver& parentResolver) {
            auto shortName = Detail::TNameCtx<std::tuple_element_t<Ind, TParentTypesTuple>>::GetNameCtx().ClassShortName;

            THashMap<TString, TParentData> module2TParentData;
            THashSet<TString> parentModules;

            for (const auto& [_, module] : TPyModuleRegistry::Get().GetCurrentModules()) {
                auto it = module.ClassName2Type.find(shortName);
                if (it != module.ClassName2Type.end()) {
                    module2TParentData[module.Name].ParentType = it->second;
                    module2TParentData[module.Name].ParentContext = module.Class2ContextGetter.at(shortName);
                    parentModules.insert(module.Name);
                }
            }
            if (parentModules.empty()) {
                ythrow yexception() << "Can't find registrated PyClass for parent class";
            }
            if (parentModules.size() == 1) {
                parentsData[Ind] = module2TParentData.begin()->second;
                return;
            }
            TString resolvedModule = parentResolver(shortName, parentModules);
            parentsData[Ind] = module2TParentData[resolvedModule];
        }

        template <typename TParentTypesTuple>
        TVector<TParentData> GetParentsData(const TParentClassResolver& parentResolver) {
            constexpr int nTypes = std::tuple_size_v<TParentTypesTuple>;
            if constexpr (nTypes == 0) {
                return {};
            }

            TVector<TParentData> parentsData(nTypes);

            [&parentsData, &parentResolver] <std::size_t... Ind> (std::index_sequence<Ind...>) {
                (InitializeBasesArray<TParentTypesTuple, Ind>(parentsData, parentResolver), ...);
            }(std::make_index_sequence<nTypes>{});

            return parentsData;
        }

        template <bool InitEnabled>
        void UpdateClassNamesInModule(TPyModuleDefinition& M, const TString& name, PyTypeObject* pythonType);

        template <bool InitEnabled>
        void UpdateGetContextInModule(TPyModuleDefinition& M, const TString& name, IGetContextBase* base);
    }


    template <class... TParentPyClasses_>
    struct TPyParentClassTraits {
        using TParentPyClasses = std::tuple<TParentPyClasses_...>;
    };

    template <bool InitEnabled_, class... TParentPyClasses_>
    struct TPyClassConfigTraits: public TPyParentClassTraits<TParentPyClasses_...> {
        constexpr static bool InitEnabled = InitEnabled_;
        constexpr static bool RawInit = false;
    };

    template <class... TParentPyClasses_>
    struct TPyClassRawInitConfigTraits: public TPyParentClassTraits<TParentPyClasses_...> {
        constexpr static bool InitEnabled = true;
        constexpr static bool RawInit = true;
    };


    template <typename TBaseClass, typename TPyClassConfigTraits, typename... ConstructorArgs>
    class TPyClass  {
    public:
        using TBase = TBaseClass;
    private:
        using TThisClass = TPyClass<TBaseClass, TPyClassConfigTraits, ConstructorArgs...>;
        using TContext = Detail::TContextImpl<TBase>;
        struct THolder {
            ::THolder<TBase> Holder;
            THolder(::THolder<TBase>&& right)
                : Holder(std::move(right))
            {
            }
            THolder(TBase&& right)
                : Holder(MakeHolder<TBase>(std::move(right)))
            {
            }
        };

        class TSelectedTraits: public NPyBind::TPythonType<THolder, TBase, TSelectedTraits> {
        private:
            using TParent = NPyBind::TPythonType<THolder, TBase, TSelectedTraits>;
            friend TParent;

        public:
            TSelectedTraits()
                : TParent(TThisClass::GetContext().ClassFullName.data(), TThisClass::GetContext().ClassDescription.data(),
                          TThisClass::GetContext().ParentsData.empty() ? nullptr : TThisClass::GetContext().ParentsData[0].ParentType,
                          Detail::GetParentTypes(TThisClass::GetContext().ParentsData))
            {
                for (const auto& caller : TThisClass::GetContext().ListCallers) {
                    TParent::AddCaller(caller.first, caller.second);
                }

                for (const auto& getter : TThisClass::GetContext().ListGetters) {
                    TParent::AddGetter(getter.first, getter.second);
                }

                for (const auto& setter : TThisClass::GetContext().ListSetters) {
                    TParent::AddSetter(setter.first, setter.second);
                }
            }

            static TBase* GetObject(const THolder& holder) {
                return holder.Holder.Get();
            }

            static THolder* DoInitObject(PyObject* args, PyObject* kwargs) {
                if constexpr (TPyClassConfigTraits::InitEnabled) {
                    if constexpr (TPyClassConfigTraits::RawInit) {
                        static_assert(sizeof...(ConstructorArgs) == 0, "Do not pass construction args if use RawInit.");
                        return new THolder(::MakeHolder<TBase>(args, kwargs));
                    } else {
                        if (args && (!PyTuple_Check(args) || PyTuple_Size(args) != sizeof...(ConstructorArgs))) {
                            ythrow yexception() << "Method takes " << sizeof...(ConstructorArgs) << " arguments, " << PyTuple_Size(args) << " provided";
                        }
                        ::THolder<TBaseClass> basePtr{std::apply([](auto&&... unpackedArgs) {return new TBase(std::forward<decltype(unpackedArgs)>(unpackedArgs)...); }, GetArguments<ConstructorArgs...>(args))};
                        return new THolder(std::move(basePtr));
                    }
                } else {
                    ythrow yexception() << "Can't create this object from python";
                }
            }

            static THolder* DoInitPureObject(const TVector<TString>& properties) {
                return Detail::DoInitPureObject<THolder, TBase, TPyClassConfigTraits::InitEnabled>(properties);
            }

            static TBase* CastToObject(PyObject* obj) {
                return TParent::CastToObject(obj);
            }

            static PyTypeObject* GetType() {
                return TParent::GetPyTypePtr();
            }
        };

        class TContextHolder: public Detail::IGetContext<TBaseClass> {
        public:
            static TContextHolder& GetContextHolder() {
                static TContextHolder holder;
                return holder;
            }

            TContext& GetContext() {
                return Context;
            }
            const TContext& GetContext() const override {
                return Context;
            }
        private:
            TContext Context;
        };

        template <class TDerivedClass, class TSuperClass>
        class TCallerWrapper: public TBaseMethodCaller<TDerivedClass> {
        public:
            explicit TCallerWrapper(TSimpleSharedPtr<const TBaseMethodCaller<TSuperClass>> baseCaller)
                : BaseCaller(baseCaller) {
                Y_ENSURE(BaseCaller);
            }

            bool CallMethod(PyObject* owner, TDerivedClass* self, PyObject* args, PyObject* kwargs, PyObject*& res) const override {
                return BaseCaller->CallMethod(owner, static_cast<TSuperClass*>(self), args, kwargs, res);
            }

        private:
            TSimpleSharedPtr<const TBaseMethodCaller<TSuperClass>> BaseCaller;
        };

        template <class TDerivedClass, class TSuperClass>
        class TSetterWrapper: public TBaseAttrSetter<TDerivedClass> {
        public:
            explicit TSetterWrapper(TSimpleSharedPtr<TBaseAttrSetter<TSuperClass>> baseSetter)
                : BaseSetter(baseSetter) {
                Y_ENSURE(BaseSetter);
            }

            bool SetAttr(PyObject* owner, TDerivedClass& self, const TString& attr, PyObject* val) override {
                return BaseSetter->SetAttr(owner, static_cast<TSuperClass&>(self), attr, val);
            }

        private:
            TSimpleSharedPtr<TBaseAttrSetter<TSuperClass>> BaseSetter;
        };

        template <class TDerivedClass, class TSuperClass>
        class TGetterWrapper: public TBaseAttrGetter<TDerivedClass> {
        public:
            explicit TGetterWrapper(TSimpleSharedPtr<const TBaseAttrGetter<TSuperClass>> baseGetter)
                : BaseGetter(baseGetter) {
                Y_ENSURE(BaseGetter);
            }

            bool GetAttr(PyObject* owner, const TDerivedClass& self, const TString& attr, PyObject*& res) const override {
                return BaseGetter->GetAttr(owner, static_cast<const TSuperClass&>(self), attr, res);
            }

        private:
            TSimpleSharedPtr<const TBaseAttrGetter<TSuperClass>> BaseGetter;
        };

        template <class TSuperClass, int Ind, typename=std::enable_if_t<!std::is_same_v<TSuperClass, void>>>
        void ReloadAttrsFromBase() {
            auto callerBasePtr = GetContext().ParentsData[Ind].ParentContext;
            if (auto getContextPtr = dynamic_cast<const Detail::IGetContext<TSuperClass>*>(callerBasePtr)) {
                auto& ctx = getContextPtr->GetContext();
                auto getUniqueNames = [](const auto& collection) {
                    THashSet<TString> uniqueNames;
                    for (const auto& elem : collection) {
                        uniqueNames.insert(elem.first);
                    }
                    return uniqueNames;
                };

                auto uniqueCallerNames = getUniqueNames(GetContext().ListCallers);
                using TConcreteCallerWrapper = TCallerWrapper<TBaseClass, TSuperClass>;
                for (const auto& caller : ctx.ListCallers) {
                    if (uniqueCallerNames.contains(caller.first)) {
                        continue;
                    }
                    GetContext().ListCallers.push_back(std::make_pair(caller.first, MakeSimpleShared<TConcreteCallerWrapper>(caller.second)));
                }

                auto uniqueGettersNames = getUniqueNames(GetContext().ListGetters);
                using TConcreteGetterWrapper = TGetterWrapper<TBaseClass, TSuperClass>;
                for (const auto& getter : ctx.ListGetters) {
                    if (uniqueGettersNames.contains(getter.first)) {
                        continue;
                    }
                    GetContext().ListGetters.push_back(std::make_pair(getter.first, MakeSimpleShared<TConcreteGetterWrapper>(getter.second)));
                }

                auto uniqueSetterNames = getUniqueNames(GetContext().ListSetters);
                using TConcreteSetterWrapper = TSetterWrapper<TBaseClass, TSuperClass>;
                for (auto& setter : ctx.ListSetters) {
                    if (uniqueSetterNames.contains(setter.first)) {
                        continue;
                    }
                    GetContext().ListSetters.push_back(std::make_pair(setter.first, MakeSimpleShared<TConcreteSetterWrapper>(setter.second)));
                }
            }
        }

        template <class TSuperClass, int Ind, typename=std::enable_if_t<std::is_same_v<TSuperClass, void>>, typename=void>
        void ReloadAttrsFromBase() {
        }

        template<class TParentTypesTuple>
        void ReloadAttrsFromBases() {
            [&]<std::size_t... Ind> (std::index_sequence<Ind...>) {
                (ReloadAttrsFromBase<std::tuple_element_t<Ind, TParentTypesTuple>, Ind>(), ...);
            }(std::make_index_sequence<std::tuple_size_v<TParentTypesTuple>>{});
        }

        void CompleteImpl() {
            ReloadAttrsFromBases<typename TPyClassConfigTraits::TParentPyClasses>();
            TSelectedTraits::Instance().Register(M.M, GetContext().ClassShortName);
        }

        static TContext& GetContext() {
            return TContextHolder::GetContextHolder().GetContext();
        }


        friend struct Detail::TContextImpl<TBase>;//instead of context
        friend struct THolder;
        friend class TSelectedTraits;

        using TCallerFunc = std::function<bool(PyObject*, TBaseClass*, PyObject*, PyObject*, PyObject*&)>;
        class TFuncCallerWrapper: public TBaseMethodCaller<TBaseClass> {
        public:
            explicit TFuncCallerWrapper(TCallerFunc func)
                : Func(func) {
                Y_ENSURE(func);
            }

            bool CallMethod(PyObject* owner, TBaseClass* self, PyObject* args, PyObject* kwargs, PyObject*& res) const override {
                return Func(owner, self, args, kwargs, res);
            }
        private:
            mutable TCallerFunc Func;
        };
    public:
        TPyClass(const TString& name, const TString& descr = "", const Detail::TParentClassResolver& parentResolver = &Detail::DefaultParentResolver)
            : M(TPyModuleDefinition::GetModule())
        {
            Detail::UpdateClassNamesInModule<TPyClassConfigTraits::InitEnabled>(M, name, TSelectedTraits::GetType());
            Detail::UpdateGetContextInModule<TPyClassConfigTraits::InitEnabled>(M, name, &TContextHolder::GetContextHolder());

            GetContext().ClassFullName = TString::Join(M.Name, ".", name);
            GetContext().ClassShortName = name;
            GetContext().ClassDescription = descr;
            GetContext().ParentsData = Detail::GetParentsData<typename TPyClassConfigTraits::TParentPyClasses>(parentResolver);
            Detail::TNameCtx<TBaseClass>::GetNameCtx().ClassShortName = name;
        }

        template <typename TMemberFuction, typename = std::enable_if_t<std::is_member_function_pointer_v<TMemberFuction>>, typename=std::enable_if_t<!TIsPointerToConstMemberFunction<TMemberFuction>::value>>
        TThisClass& Def(const TString& name, TMemberFuction t) {
            GetContext().ListCallers.push_back(std::make_pair(name, CreateMethodCaller<TBase>(t)));
            return *this;
        }

        template <typename TMemberFuction, typename = std::enable_if_t<std::is_member_function_pointer_v<TMemberFuction>>, typename=std::enable_if_t<TIsPointerToConstMemberFunction<TMemberFuction>::value>, typename=void>
        TThisClass& Def(const TString& name, TMemberFuction t) {
            GetContext().ListCallers.push_back(std::make_pair(name, CreateConstMethodCaller<TBase>(t)));
            return *this;
        }

        template <typename TMemberObject, typename = std::enable_if_t<std::is_member_object_pointer_v<TMemberObject>>>
        TThisClass& Def(const TString& name, TMemberObject t)  {
            GetContext().ListGetters.push_back(std::make_pair(name, CreateAttrGetter<TBase>(t)));
            GetContext().ListSetters.push_back(std::make_pair(name, CreateAttrSetter<TBase>(t)));
            return *this;
        }

        template <typename TResultType, typename... Args>
        TThisClass& DefByFunc(const TString& name, std::function<TResultType(TBaseClass&, Args...)> func) {
            GetContext().ListCallers.push_back(std::make_pair(name, CreateFunctorCaller<TBase, TResultType, Args...>(func)));
            return *this;
        }

        TThisClass& DefByFunc(const TString& name, TCallerFunc origFunc) {
            GetContext().ListCallers.push_back(std::make_pair(name, MakeSimpleShared<TFuncCallerWrapper>(origFunc)));
            return *this;
        }

        template <typename TMemberObject>
        TThisClass& DefReadonly(const TString& name, TMemberObject t, std::enable_if_t<std::is_member_object_pointer<TMemberObject>::value>* = nullptr) {
            GetContext().ListGetters.push_back(std::make_pair(name, CreateAttrGetter<TBase>(t)));
            return *this;
        }


        template <typename TMethodGetter, typename TMethodSetter, typename=std::enable_if_t<std::is_member_function_pointer_v<TMethodGetter> && std::is_member_function_pointer_v<TMethodSetter>>>
        TThisClass& AsProperty(const TString& name, TMethodGetter getter, TMethodSetter setter) {
            GetContext().ListGetters.push_back(std::make_pair(name, CreateMethodAttrGetter<TBase>(getter)));
            GetContext().ListSetters.push_back(std::make_pair(name, CreateMethodAttrSetter<TBase>(setter)));
            return *this;
        }

        template <typename TMethodGetter, typename TMethodSetter, typename=std::enable_if_t<!std::is_member_function_pointer_v<TMethodGetter> && !std::is_member_function_pointer_v<TMethodSetter>>>
        TThisClass& AsPropertyByFunc(const TString& name, TMethodGetter getter, TMethodSetter setter) {
            GetContext().ListGetters.push_back(std::make_pair(name, CreateFunctorAttrGetter<TBase>(getter)));
            GetContext().ListSetters.push_back(std::make_pair(name, CreateFunctorAttrSetter<TBase>(setter)));
            return *this;
        }

        template <typename TMethodGetter, typename=std::enable_if_t<std::is_member_function_pointer_v<TMethodGetter>>>
        TThisClass& AsProperty(const TString& name, TMethodGetter getter) {
            GetContext().ListGetters.push_back(std::make_pair(name, CreateMethodAttrGetter<TBase>(getter)));
            return *this;
        }

        template <typename TMethodGetter>
        TThisClass& AsPropertyByFunc(const TString& name, TMethodGetter getter) {
            GetContext().ListGetters.push_back(std::make_pair(name, CreateFunctorAttrGetter<TBase>(getter)));
            return *this;
        }

        TThisClass& Complete() {
            if (!Completed) {
                CompleteImpl();
                Completed = true;
            }
            return *this;
        }

    public:
        static PyObject* BuildPyObject(TBase&& base) {
            return NPyBind::BuildPyObject(TSelectedTraits::Instance().CreatePyObject(new THolder(std::move(base))));
        }

        static PyObject* BuildPyObject(const TBase& base) {
            return NPyBind::BuildPyObject(TSelectedTraits::Instance().CreatePyObject(new THolder(TBase(base)))); // WARN - copy
        }

        static TBase* CastToObject(PyObject* obj) {
            return TSelectedTraits::CastToObject(obj);
        }

    private:
        TPyModuleDefinition& M;
        bool Completed = false;
    };

    template <typename TFunctionSignature, TFunctionSignature function>
    void DefImpl(const TString& name, const TString& descr = "") {
        NPyBind::TModuleHolder::Instance().AddModuleMethod<TModuleMethodCaller<TFunctionSignature, function>::Call>(name, descr);
    }

    template <typename TFunctionSignature, TFunctionSignature function>
    void DefRawImpl(const TString& name, const TString& descr = "") {
        NPyBind::TModuleHolder::Instance().AddModuleMethod<[](PyObject*, PyObject* args, PyObject* kwargs) -> PyObject* {
            return BuildPyObject(function(args, kwargs));
        }>(name, descr);
    }

#define DefFunc(NAME, FUNC) NPyBind::DefImpl<decltype(FUNC), FUNC>(NAME)
#define DefFuncDescr(NAME, FUNC, DESCR) NPyBind::DefImpl<decltype(FUNC), FUNC>(NAME, DESCR)

#define DefRawFunc(NAME, FUNC) NPyBind::DefRawImpl<decltype(FUNC), FUNC>(NAME)
#define DefRawFuncDescr(NAME, FUNC, DESCR) NPyBind::DefRawImpl<decltype(FUNC), FUNC>(NAME, DESCR)
};
