#include "v2.h"
namespace NPyBind {
    namespace Detail {

        TVector<PyTypeObject*> GetParentTypes(const TVector<TParentData>& parentsData) {
            TVector<PyTypeObject*> res;
            Transform(
                parentsData.begin(),
                parentsData.end(),
                back_inserter(res),
                [](const TParentData& el) { return el.ParentType; }
            );
            return res;
        }

        TString DefaultParentResolver(const TString&, const THashSet<TString>& parentModules) {
            return *parentModules.begin();
        }

        template <bool InitEnabled>
        void UpdateClassNamesInModule(TPyModuleDefinition& M, const TString& name, PyTypeObject* pythonType) {
            if (!InitEnabled) {
                return;
            }
            M.ClassName2Type[name] = pythonType;
        }

        template <bool InitEnabled>
        void UpdateGetContextInModule(TPyModuleDefinition& M, const TString& name, IGetContextBase* base) {
            if (!InitEnabled) {
                return;
            }
            M.Class2ContextGetter[name] = base;
        }

        TPyModuleRegistry::TPyModuleRegistry() {
#if PY_MAJOR_VERSION >= 3
            NPrivate::AddFinalizationCallBack([this]() {
                if (UnnamedModule) {
                    UnnamedModule.Clear();
                }
                Name2Def.clear();
            });
#endif
        }
        template void UpdateClassNamesInModule<false>(TPyModuleDefinition& M, const TString& name, PyTypeObject* pythonType);
        template void UpdateClassNamesInModule<true>(TPyModuleDefinition& M, const TString& name, PyTypeObject* pythonType);


        template void UpdateGetContextInModule<false>(TPyModuleDefinition& M, const TString& name, IGetContextBase* pythonType);
        template void UpdateGetContextInModule<true>(TPyModuleDefinition& M, const TString& name, IGetContextBase* pythonType);
    }//Detail
}//NPyBind
