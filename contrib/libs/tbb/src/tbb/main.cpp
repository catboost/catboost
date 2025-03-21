/*
    Copyright (c) 2005-2024 Intel Corporation

    Licensed under the Apache License, Version 2.0 (the "License");
    you may not use this file except in compliance with the License.
    You may obtain a copy of the License at

        http://www.apache.org/licenses/LICENSE-2.0

    Unless required by applicable law or agreed to in writing, software
    distributed under the License is distributed on an "AS IS" BASIS,
    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    See the License for the specific language governing permissions and
    limitations under the License.
*/

#include "oneapi/tbb/detail/_config.h"

#include "main.h"
#include "governor.h"
#include "threading_control.h"
#include "environment.h"
#include "market.h"
#include "tcm_adaptor.h"
#include "misc.h"
#include "itt_notify.h"

namespace tbb {
namespace detail {
namespace r1 {

//------------------------------------------------------------------------
// Begin shared data layout.
// The following global data items are mostly read-only after initialization.
//------------------------------------------------------------------------

//------------------------------------------------------------------------
// governor data
basic_tls<thread_data*> governor::theTLS;
rml::tbb_factory governor::theRMLServerFactory;
bool governor::UsePrivateRML;
bool governor::is_rethrow_broken;

//------------------------------------------------------------------------
// threading_control data
threading_control* threading_control::g_threading_control;
threading_control::global_mutex_type threading_control::g_threading_control_mutex;

//------------------------------------------------------------------------
// context propagation data
context_state_propagation_mutex_type the_context_state_propagation_mutex;
std::atomic<uintptr_t> the_context_state_propagation_epoch{};

//------------------------------------------------------------------------
// One time initialization data

//! Counter of references to global shared resources such as TLS.
std::atomic<int> __TBB_InitOnce::count{};

std::atomic_flag __TBB_InitOnce::InitializationLock = ATOMIC_FLAG_INIT;

//! Flag that is set to true after one-time initializations are done.
std::atomic<bool> __TBB_InitOnce::InitializationDone{};

#if __TBB_USE_ITT_NOTIFY
//! Defined in profiling.cpp
extern bool ITT_Present;
void ITT_DoUnsafeOneTimeInitialization();
#endif

#if !(_WIN32||_WIN64) || __TBB_SOURCE_DIRECTLY_INCLUDED
static __TBB_InitOnce __TBB_InitOnceHiddenInstance;
#endif

//------------------------------------------------------------------------
// __TBB_InitOnce
//------------------------------------------------------------------------

void __TBB_InitOnce::add_ref() {
    if (++count == 1) {
        governor::acquire_resources();
        tcm_adaptor::initialize();
    }
}

void __TBB_InitOnce::remove_ref() {
    int k = --count;
    __TBB_ASSERT(k>=0,"removed __TBB_InitOnce ref that was not added?");
    if( k==0 ) {
        governor::release_resources();
        ITT_FINI_ITTLIB();
        ITT_RELEASE_RESOURCES();
    }
}

//------------------------------------------------------------------------
// One-time Initializations
//------------------------------------------------------------------------

//! Defined in cache_aligned_allocator.cpp
void initialize_cache_aligned_allocator();

//! Performs thread-safe lazy one-time general TBB initialization.
void DoOneTimeInitialization() {
    __TBB_InitOnce::lock();
    // No fence required for load of InitializationDone, because we are inside a critical section.
    if( !__TBB_InitOnce::InitializationDone ) {
        __TBB_InitOnce::add_ref();
        if( GetBoolEnvironmentVariable("TBB_VERSION") ) {
            PrintVersion();
            tcm_adaptor::print_version();
        }
        bool itt_present = false;
#if __TBB_USE_ITT_NOTIFY
        ITT_DoUnsafeOneTimeInitialization();
        itt_present = ITT_Present;
#endif /* __TBB_USE_ITT_NOTIFY */
        initialize_cache_aligned_allocator();
        governor::initialize_rml_factory();
        // Force processor groups support detection
        governor::default_num_threads();
        // Force OS regular page size detection
        governor::default_page_size();
        PrintExtraVersionInfo( "TOOLS SUPPORT", itt_present ? "enabled" : "disabled" );
        __TBB_InitOnce::InitializationDone = true;
    }
    __TBB_InitOnce::unlock();
}

#if (_WIN32||_WIN64) && !__TBB_SOURCE_DIRECTLY_INCLUDED
//! Windows "DllMain" that handles startup and shutdown of dynamic library.
extern "C" bool WINAPI DllMain( HANDLE /*hinstDLL*/, DWORD reason, LPVOID lpvReserved ) {
    switch( reason ) {
        case DLL_PROCESS_ATTACH:
            __TBB_InitOnce::add_ref();
            break;
        case DLL_PROCESS_DETACH:
            // Since THREAD_DETACH is not called for the main thread, call auto-termination
            // here as well - but not during process shutdown (due to risk of a deadlock).
            if ( lpvReserved == nullptr ) { // library unload
                governor::terminate_external_thread();
            }
            __TBB_InitOnce::remove_ref();
            // It is assumed that InitializationDone is not set after DLL_PROCESS_DETACH,
            // and thus no race on InitializationDone is possible.
            if ( __TBB_InitOnce::initialization_done() ) {
                // Remove reference that we added in DoOneTimeInitialization.
                __TBB_InitOnce::remove_ref();
            }
            break;
        case DLL_THREAD_DETACH:
            governor::terminate_external_thread();
            break;
    }
    return true;
}
#endif /* (_WIN32||_WIN64) && !__TBB_SOURCE_DIRECTLY_INCLUDED */

} // namespace r1
} // namespace detail
} // namespace tbb
