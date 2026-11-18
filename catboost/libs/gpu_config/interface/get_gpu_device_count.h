#pragma once

namespace NCB {

    /*
     * Returns the number of usable devices for the configured GPU backend.
     * Darwin ARM64 builds with HAVE_METAL report the default Metal device.
     * Otherwise, builds without CUDA (including force_no_cuda) return zero.
     */
    int GetGpuDeviceCount();

    // Compile-time backend selection; availability is checked separately.
    bool IsMetalBackend();

}
