#include "v2.h"
namespace NPyBind {
    namespace Detail {
        template <>
        PyTypeObject* GetParentType<void>(const TPyModuleDefinition&) {
            return nullptr;
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
