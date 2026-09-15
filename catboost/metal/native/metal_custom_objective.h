#pragma once

#include <cstring>
#include <stdexcept>
#include <string>

// Compile the user function into a session's Metal pipelines. The descriptor
// owns its source; no Python/CUDA callback runs during derivative evaluation.
// An empty prefix leaves every existing builtin shader unchanged.
inline std::string CBMCustomObjectivePrefix(const char* body) {
    if (!body) return {};
    const size_t length = strnlen(body, 65537);
    if (!length || length > 65536) {
        throw std::runtime_error("Metal custom objective source must contain 1..65536 bytes");
    }
    return std::string("#include <metal_stdlib>\nusing namespace metal;\n"
        "#define CBM_HAS_CUSTOM_OBJECTIVE 1\n"
        "inline float3 CBMUserObjectiveValueDerivatives(float approx, float target, float weight) {\n")
        + body + "\n}\n";
}
