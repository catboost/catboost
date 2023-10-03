#pragma once

#include <grpc/slice.h>

namespace grpc_core {
    grpc_slice AddArcadiaRootCerts(grpc_slice systemCerts);
}
