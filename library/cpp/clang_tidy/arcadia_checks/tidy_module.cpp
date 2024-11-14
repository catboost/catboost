#include <contrib/libs/clang16/tools/extra/clang-tidy/ClangTidy.h>
#include <contrib/libs/clang16/tools/extra/clang-tidy/ClangTidyModule.h>
#include <contrib/libs/clang16/tools/extra/clang-tidy/ClangTidyModuleRegistry.h>

#include "taxi_coroutine_unsafe_check.h"
#include "taxi_dangling_config_ref_check.h"

#include "ascii_compare_ignore_case_check.h"
#include "usage_restriction_checks.h"

using namespace clang::ast_matchers;

namespace clang::tidy::arcadia {
    class ArcadiaModule: public ClangTidyModule {
    public:
        void addCheckFactories(ClangTidyCheckFactories& CheckFactories) override {
            CheckFactories.registerCheck<misc::TaxiCoroutineUnsafeCheck>(
                "arcadia-taxi-coroutine-unsafe");
            CheckFactories.registerCheck<misc::TaxiDanglingConfigRefCheck>(
                "arcadia-taxi-dangling-config-ref");

            // https://st.yandex-team.ru/IGNIETFERRO-1863
            CheckFactories.registerCheck<TypeidNameRestrictionCheck>(
                "arcadia-typeid-name-restriction");
            CheckFactories.registerCheck<AsciiCompareIgnoreCaseCheck>("arcadia-ascii-compare-ignorecase");
        }
    };

    // Register the ArcadiaTidyModule using this statically initialized variable.
    static ClangTidyModuleRegistry::Add<ArcadiaModule>
        X("arcadia-module", "Adds Arcadia specific lint checks.");

}
