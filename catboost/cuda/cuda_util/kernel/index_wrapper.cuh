#pragma once

namespace NKernel {

    class TIndexWrapper {
    public:

        __device__ TIndexWrapper(ui32 idx)
                : Idx(idx) {

        }

        __device__ void UpdateMask(bool isOnBorder) {
            Idx |= ((ui32) isOnBorder) << 31;
        }

        __device__ ui32 IsSegmentStart() const {
            return Idx >> 31;
        }

        __device__ ui32 Index() const {
            return Idx & 0x3FFFFFFF;
        }

        __device__ ui32 Value() const {
            return Idx;
        }

    private:
        ui32 Idx;
    };
}
