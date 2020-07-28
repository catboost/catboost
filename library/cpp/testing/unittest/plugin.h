#pragma once

#include <util/generic/ptr.h>

namespace NUnitTest {
    namespace NPlugin {
        class IPlugin {
        public:
            virtual ~IPlugin() {
            }

            virtual void OnStartMain(int /*argc*/, char* /*argv*/ []) {
            }

            virtual void OnStopMain(int /*argc*/, char* /*argv*/ []) {
            }
        };

        void OnStartMain(int argc, char* argv[]);
        void OnStopMain(int argc, char* argv[]);

        class TPluginRegistrator {
        public:
            TPluginRegistrator(TSimpleSharedPtr<IPlugin> plugin);
        };

    }
}
