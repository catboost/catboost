#pragma once

#include <util/generic/singleton.h>
#include <util/generic/vector.h>

namespace NCB {
    template <class... TArgs>
    class TInitBase {
    public:
        using TInitFunc = void(*)(TArgs...);

    public:
        TInitBase() = default;

        void Add(TInitFunc func) {
            Funcs.push_back(func);
        }

        /* note: the absense of perfect forwarding is intentional,
         * the same arguments are used for all registered Funcs, they shouldn't be moved
         */
        static void Do(TArgs... args) {
            for (auto func : Singleton<TInitBase>()->Funcs) {
                func(args...);
            }
        }

        class TRegistrator {
        public:
            TRegistrator(TInitFunc func) {
                Singleton<TInitBase>()->Add(func);
            }
        };

        private:
            TVector<TInitFunc> Funcs;
    };

    using TCmdLineInit = TInitBase<int, const char**>; // params are argc, argv

    using TLibraryInit = TInitBase<>;

    void LibraryInit(); // for simplification of calls from libraries


    /*
     * Register functions for global initialization
     *
     *
     * 1) for cmdline app:
     *  in .cpp files in your library:
     *
     *  > NCB::TCmdLineInit::TRegistrator MyRegistrator(
     *  >     [](int argc, const char* argv[]) { Cout << "Last arg: " << argv[argc-1] << Endl; }
     *  > );
     *
     *
     *  It is necessary to add GLOBAL before src files with TRegistrator instances in ya.make
     *
     *  Call in main() like that:
     *
     *  > int main(int argc, const char* argv[]) {
     *  >     NCB::TCmdLineInit::Do(argc, argv);
     *  >     ...
     *
     * 2) for library, when time of initialization should be explicit
     *  (after all Registrators have been called)
     *   in .cpp files in your library:
     *
     *  > NCB::TLibraryInit::TRegistrator MyRegistrator(
     *  >     []() { Cout << "My Library init" << Endl; }
     *  > );
     *
     *
     *  It is necessary to add GLOBAL before src files with TRegistrator instances in ya.make
     *
     *  Call in library initialization func like that:
     *
     *  >
     *  >     NCB::LibraryInit();
     *  >     ...
     *
     *
     */

}
