#pragma once
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    uint64_t kernel_dispatches;
    double gpu_seconds;
    char device_name[256];
} CBMSortStats;

// Stable ascending unsigned-key sort, carrying arbitrary uint32 payloads.
// Null payload means original row IDs. Count zero permits null array pointers.
// Input/output aliases are supported; the two output arrays must not overlap.
// The implementation performs all ordering on Metal, including prefix scans.
int cbm_sort_u32(const uint32_t* keys, const uint32_t* payload, uint32_t count,
                 uint32_t* out_keys, uint32_t* out_payload, CBMSortStats* stats,
                 char* error, size_t error_capacity);

#ifdef __cplusplus
}
#endif

#ifdef __OBJC__
#import <Metal/Metal.h>
#include <memory>

// Reuse radix scratch across sequential encodes, including many candidate
// tiles within one command. The old convenience API allocates scratch for
// every call, which a retained command keeps until completion. Do not reuse
// one workspace concurrently across commands. After encoding into a command,
// finish that command before encoding into a different one.
class CBMSortU32Workspace {
public:
    CBMSortU32Workspace(id<MTLDevice> device, uint32_t capacity);
    ~CBMSortU32Workspace();
    CBMSortU32Workspace(const CBMSortU32Workspace&) = delete;
    CBMSortU32Workspace& operator=(const CBMSortU32Workspace&) = delete;
    uint64_t AllocatedBytes() const;
    void Encode(id<MTLCommandBuffer> command, id<MTLBuffer> keys, id<MTLBuffer> payload,
                uint32_t count, id<MTLBuffer> out_keys, id<MTLBuffer> out_payload,
                uint64_t* kernel_dispatches = nullptr, uint32_t low_key_bits = 32);
    // low_key_bits is a multiple of four in [4,32]. Smaller values stably sort
    // only that low-bit prefix, retaining the original full keys and payloads.
private:
    struct TState;
    std::unique_ptr<TState> State;
};

// Append a full stable sort to a caller-owned command buffer. Does not commit,
// wait, or access buffer contents on the CPU. All input/output buffers must
// belong to command's device; payload is required. The input/output pairs may
// alias, but out_keys and out_payload must be distinct buffers. Scratch buffers
// are retained by the command, which must use normal retained references.
// Throws std::runtime_error on invalid inputs or Metal allocation failure.
void CBMEncodeSortU32(id<MTLCommandBuffer> command, id<MTLBuffer> keys,
                      id<MTLBuffer> payload, uint32_t count,
                      id<MTLBuffer> out_keys, id<MTLBuffer> out_payload,
                      uint64_t* kernel_dispatches);
#endif
