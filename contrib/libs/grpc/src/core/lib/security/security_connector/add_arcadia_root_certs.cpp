#include "add_arcadia_root_certs.h"
#include "grpc/support/alloc.h"

#include <library/cpp/resource/resource.h>

namespace grpc_core {
    grpc_slice AddArcadiaRootCerts(grpc_slice systemCerts) {
        TString cacert = NResource::Find("/builtin/cacert");
        size_t sumSize = cacert.size() + GRPC_SLICE_LENGTH(systemCerts);
        char* bundleString = static_cast<char*>(gpr_zalloc(sumSize + 1)); // With \0.
        memcpy(bundleString, cacert.data(), cacert.size());
        memcpy(bundleString + cacert.size(), GRPC_SLICE_START_PTR(systemCerts), GRPC_SLICE_LENGTH(systemCerts));
        grpc_slice_unref(systemCerts);
        return grpc_slice_new(bundleString, sumSize, gpr_free);
    }
}
