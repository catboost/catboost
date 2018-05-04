#include "pointwise_hist2.cuh"
#include "split_properties_helpers.cuh"
#include "compute_point_hist2_loop.cuh"
#include <cooperative_groups.h>
#include <catboost/cuda/cuda_lib/kernel/arch.cuh>
#include <catboost/cuda/cuda_util/kernel/instructions.cuh>
#include <catboost/cuda/cuda_util/kernel/kernel_helpers.cuh>
#include <catboost/cuda/cuda_lib/kernel/arch.cuh>

using namespace cooperative_groups;

namespace NKernel
{

    template<int OUTER_HIST_BITS_COUNT,
             int INNER_HIST_BITS_COUNT,
             int BLOCK_SIZE>
    struct TPointHist {
        float* __restrict__ Buffer;

        float mostRecentStat1[4];
        float mostRecentStat2[4];
        uchar mostRecentBin[4];

        __forceinline__ __device__ int SliceOffset() {

            const int maxBlocks = BLOCK_SIZE * 32 / (1024 << OUTER_HIST_BITS_COUNT);
            static_assert(OUTER_HIST_BITS_COUNT <= 2, "Error: assume 12 warps, so limited by 128-bin histogram per warp");
            static_assert(OUTER_HIST_BITS_COUNT > 0 && INNER_HIST_BITS_COUNT > 0, "This histogram is specialized for 255 bin count");

            const int warpId = (threadIdx.x / 32) % maxBlocks;
            const int warpOffset = (1024 << OUTER_HIST_BITS_COUNT) * warpId;
            const int blocks = 4 >> INNER_HIST_BITS_COUNT;
            const int innerHistStart = (threadIdx.x & ((blocks - 1) << (INNER_HIST_BITS_COUNT + 3)));
            return warpOffset + innerHistStart;
        }

        __forceinline__ __device__ TPointHist(float* buff) {

            const int HIST_SIZE = 32 * BLOCK_SIZE;

            #pragma unroll 8
            for (int i = threadIdx.x; i < HIST_SIZE; i += BLOCK_SIZE) {
                buff[i] = 0;
            }

            Buffer = buff + SliceOffset();

            __syncthreads();
            #pragma unroll
            for (int f = 0; f < 4; ++f) {
                mostRecentBin[f] = 0;
                mostRecentStat1[f] = 0;
                mostRecentStat2[f] = 0;
            }
        }

        __forceinline__ __device__ void Add(float val, float* dst) {
            atomicAdd(dst, val);
        }

        __forceinline__ __device__ void AddPoint(ui32 ci, const float t, const float w) {
            const bool flag = threadIdx.x & 1;

            const float stat1 = flag ? t : w;
            const float stat2 = flag ? w : t;

            #pragma unroll
            for (int i = 0; i < 4; i++) {
                const short f = ((i + threadIdx.x / 2) & 3);
                const uchar bin = bfe(ci, 24 - (f << 3), 8);

                if (bin != mostRecentBin[i]) {
                    const bool pass = (mostRecentBin[i] >> (5 + INNER_HIST_BITS_COUNT + OUTER_HIST_BITS_COUNT)) == 0;

                    if (pass) {
                        int offset = 2 * f;
                        const uchar mask = (1 << INNER_HIST_BITS_COUNT) - 1;
                        offset += 8 * (mostRecentBin[i] & mask);
                        offset += 32 * ((mostRecentBin[i] >> INNER_HIST_BITS_COUNT));

                        offset += flag;
                        Add(mostRecentStat1[i], Buffer + offset);
                        offset = flag ? offset - 1 : offset + 1;
                        Add(mostRecentStat2[i], Buffer + offset);
                    }

                    mostRecentBin[i] = bin;
                    mostRecentStat1[i] = 0;
                    mostRecentStat2[i] = 0;
                }

                {
                    mostRecentStat1[i] += stat1;
                    mostRecentStat2[i] += stat2;
                }
            }
        }

        __forceinline__ __device__ void AddPoint2(uint2 bin, const float2 t, const float2 w) {
            AddPoint(bin.x, t.x, w.x);
            AddPoint(bin.y, t.y, w.y);
        }

        __forceinline__ __device__ void AddPoint4(uint4 ci, const float4 t, const float4 w) {
            AddPoint(ci.x, t.x, w.x);
            AddPoint(ci.y, t.y, w.y);
            AddPoint(ci.z, t.z, w.z);
            AddPoint(ci.w, t.w, w.w);
        }
        //After reduce we store histograms by blocks: 256 floats (4 x 2 x 32)
        // for first 32 bins; than 256 floats for second 32 bins, etc
        __forceinline__ __device__ void Reduce() {
            {
                const bool flag = threadIdx.x & 1;
                #pragma unroll
                for (int i = 0; i < 4; i++) {
                    const short f = ((i + threadIdx.x / 2) & 3);
                    const bool pass = (mostRecentBin[i] >> (5 + INNER_HIST_BITS_COUNT + OUTER_HIST_BITS_COUNT)) == 0;
                    if (pass) {
                        int offset = 2 * f;
                        const uchar mask = (1 << INNER_HIST_BITS_COUNT) - 1;
                        offset += 8 * (mostRecentBin[i] & mask);
                        offset += 32 * ((mostRecentBin[i] >> INNER_HIST_BITS_COUNT));

                        Add(mostRecentStat1[i], Buffer + offset + flag);
                        Add(mostRecentStat2[i], Buffer + offset + !flag);
                    }
                }
            }


            Buffer -= SliceOffset();
            __syncthreads();

            {
                const int warpHistSize = 1024 << OUTER_HIST_BITS_COUNT;
                const int maxBlocks = BLOCK_SIZE * 32 / (1024 << OUTER_HIST_BITS_COUNT);

                for (int start = threadIdx.x; start < warpHistSize; start += BLOCK_SIZE) {
                    float sum = 0;
//                    12 iterations at 32-bin
                    #pragma unroll maxBlocks
                    for (int i = start; i < 32 * BLOCK_SIZE; i += warpHistSize) {
                        sum += Buffer[i];
                    }
                    Buffer[warpHistSize + start] = sum;
                }
            }

            __syncthreads();

            if (threadIdx.x < 256) {
                const int w = threadIdx.x & 1;

                float sum[4];

                const int maxFoldCount = (1 << (5 + INNER_HIST_BITS_COUNT + OUTER_HIST_BITS_COUNT));
                for (int fold = (threadIdx.x >> 1); fold < maxFoldCount; fold += 128) {

                    #pragma unroll
                    for (int f = 0; f < 4; ++f) {
                        sum[f] = 0;
                    }

                    const int innerHistCount = 4 >> INNER_HIST_BITS_COUNT;
                    const int lowBitMask = (1 << INNER_HIST_BITS_COUNT) - 1;
                    const float* __restrict__ src = Buffer
                                                    + (1024 << OUTER_HIST_BITS_COUNT)  //warpHistSize
                                                    + 8 * (fold & lowBitMask)
                                                    + 32 * (fold >> INNER_HIST_BITS_COUNT)
                                                    + w;

                    #pragma unroll
                    for (int inWarpHist = 0; inWarpHist < innerHistCount; ++inWarpHist) {
                        #pragma unroll
                        for (int f = 0; f < 4; ++f) {
                            sum[f] += src[2 * f + (inWarpHist << (3 + INNER_HIST_BITS_COUNT))];
                        }
                    }

                    #pragma unroll
                    for (int f = 0; f < 4; ++f) {
                        Buffer[2 * (maxFoldCount * f + fold) + w] = sum[f];
                    }
                }
            }
            __syncthreads();
        }
    };


    template<int BLOCK_SIZE>
    struct TPointHist<0, 0, BLOCK_SIZE> {
        float* __restrict__ Buffer;

        __forceinline__ __device__ int SliceOffset() {
            const int warpId = (threadIdx.x / 32);
            const int warpOffset = 1024 * warpId;
            const int blocks = 4;
            const int innerHistStart = (threadIdx.x & ((blocks - 1) << 3));
            return warpOffset + innerHistStart;
        }

        __forceinline__ __device__ TPointHist(float* buff) {

            const int HIST_SIZE = 32 * BLOCK_SIZE;

            #pragma unroll 8
            for (int i = threadIdx.x; i < HIST_SIZE; i += BLOCK_SIZE) {
                buff[i] = 0;
            }

            Buffer = buff + SliceOffset();
            __syncthreads();
        }

        __forceinline__ __device__ void Add(float val, float* dst) {
            dst[0] += val;
        }

        __forceinline__ __device__ void AddPoint(ui32 ci,
                                                 const float t,
                                                 const float w) {

            thread_block_tile<32> syncTile = tiled_partition<32>(this_thread_block());

            const bool flag = threadIdx.x & 1;

            const float stat1 = flag ? t : w;
            const float stat2 = flag ? w : t;

            #pragma unroll
            for (int i = 0; i < 4; i++) {
                const int f = ((2 * i + threadIdx.x) & 6);
                const int bin = (ci >> (24 - (f << 2))) & 255;
                const float pass = bin != 32 ? 1.0f : 0.0f;
                syncTile.sync();
                int offset = f + 32 * (bin & 31);
                const int offset1 = offset + flag;

                const float add1 = pass * stat1;
                Buffer[offset1] += add1;

                const int offset2 = offset + !flag;
                const float add2 = pass * stat2;

                syncTile.sync();
                Buffer[offset2] += add2;
            }
        }

        __forceinline__ __device__ void AddPoint2(uint2 ci,
                                                  const float2 t,
                                                  const float2 w) {

            thread_block_tile<32> syncTile = tiled_partition<32>(this_thread_block());

            const bool flag = threadIdx.x & 1;

            const float2 stat1 = flag ? t : w;
            const float2 stat2 = flag ? w : t;

            #pragma unroll
            for (int i = 0; i < 4; i++) {
                int f = ((2 * i + threadIdx.x) & 6);
                const int bin1 = (ci.x >> (24 - (f << 2))) & 255;
                const int bin2 = (ci.y >> (24 - (f << 2))) & 255;

                const float passx = bin1 != 32 ? 1.0f : 0.0f;
                const float passy = bin2 != 32 ? 1.0f : 0.0f;

                int offsetx = f + 32 * (bin1 & 31) + flag;
                int offsety = f + 32 * (bin2 & 31) + flag;

                syncTile.sync();
                Buffer[offsetx] += passx * stat1.x;
                Buffer[offsety] += passy * stat1.y;

                offsetx += flag ? -1 : 1;
                offsety += flag ? -1 : 1;

                syncTile.sync();

                Buffer[offsetx] += passx * stat2.x;
                Buffer[offsety] += passy * stat2.y;
            }
        }

        __forceinline__ __device__ void AddPoint4(uint4 ci, const float4 t, const float4 w) {
            thread_block_tile<32> syncTile = tiled_partition<32>(this_thread_block());

            const bool flag = threadIdx.x & 1;

            const float4 stat1 = flag ? t : w;
            const float4 stat2 = flag ? w : t;

            #pragma unroll
            for (int i = 0; i < 4; i++) {
                int f = ((2 * i + threadIdx.x) & 6);
                const ui32 shift = static_cast<ui32>(24 - (f << 2));
                f += flag;

                const int binx = (ci.x >> shift) & 255;
                const int biny = (ci.y >> shift) & 255;
                const int binz = (ci.z >> shift) & 255;
                const int binw = (ci.w >> shift) & 255;


                const float passx = binx != 32 ? 1.0f : 0.0f;
                const float passy = biny != 32 ? 1.0f : 0.0f;
                const float passz = binz != 32 ? 1.0f : 0.0f;
                const float passw = binw != 32 ? 1.0f : 0.0f;

                float* buffer = Buffer + f;


                int offsetx = (binx & 31) << 5;
                int offsety = (biny & 31) << 5;
                int offsetz = (binz & 31) << 5;
                int offsetw = (binw & 31) << 5;

                syncTile.sync();

                buffer[offsetx] += passx * stat1.x;
                buffer[offsety] += passy * stat1.y;
                buffer[offsetz] += passz * stat1.z;
                buffer[offsetw] += passw * stat1.w;

                offsetx += flag ? -1 : 1;
                offsety += flag ? -1 : 1;
                offsetz += flag ? -1 : 1;
                offsetw += flag ? -1 : 1;


                syncTile.sync();

                buffer[offsetx] += passx * stat2.x;
                buffer[offsety] += passy * stat2.y;
                buffer[offsetz] += passz * stat2.z;
                buffer[offsetw] += passw * stat2.w;
            }
        }

        //After reduce we store histograms by blocks: 256 floats (4 x 2 x 32)
        // for first 32 bins; than 256 floats for second 32 bins, etc
        __forceinline__ __device__ void Reduce() {
            Buffer -= SliceOffset();
            __syncthreads();

            {
                const int warpHistSize = 1024;

                for (int start = threadIdx.x; start < warpHistSize; start += BLOCK_SIZE) {
                    float sum = 0;
//                    12 iterations at 32-bin
                    #pragma unroll 12
                    for (int i = start; i < 32 * BLOCK_SIZE; i += warpHistSize) {
                        sum += Buffer[i];
                    }
                    Buffer[warpHistSize + start] = sum;
                }
            }

            __syncthreads();

            if (threadIdx.x < 256) {
                const int w = threadIdx.x & 1;
                const int f = threadIdx.x / 64;
                float sum = 0.0f;
                const int fold = (threadIdx.x >> 1) & 31;
                const int maxFoldCount = 32;

                if (fold < maxFoldCount) {
                    const int innerHistCount = 4;
                    const volatile float* __restrict__ src = Buffer
                                                    + 1024  //warpHistSize
                                                    + 32 * fold
                                                    + w;

                    #pragma unroll
                    for (int inWarpHist = 0; inWarpHist < innerHistCount; ++inWarpHist) {
                        sum += src[2 * f + (inWarpHist << 3)];
                    }

                    Buffer[2 * (maxFoldCount * f + fold) + w] = sum;
                }
            }
            __syncthreads();
        }
    };


    template<int BLOCK_SIZE>
    struct TPointHist<0, 1, BLOCK_SIZE> {
        float* __restrict__ Buffer;

        __forceinline__ __device__ int SliceOffset() {
            const int warpId = (threadIdx.x / 32);
            const int warpOffset = 1024 * warpId;
            const int blocks = 2;
            const int innerHistStart = (threadIdx.x & ((blocks - 1) << 4));
            return warpOffset + innerHistStart;
        }

        __forceinline__ __device__ TPointHist(float* buff) {

            const int HIST_SIZE = 32 * BLOCK_SIZE;

            #pragma unroll 8
            for (int i = threadIdx.x; i < HIST_SIZE; i += BLOCK_SIZE) {
                buff[i] = 0;
            }

            Buffer = buff + SliceOffset();
            __syncthreads();
        }


        __forceinline__ __device__ void AddPoint(ui32 ci,
                                                 const float t,
                                                 const float w) {

            thread_block_tile<32> syncTile = tiled_partition<32>(this_thread_block());

            const bool flag = threadIdx.x & 1;

            const float stat1 = flag ? t : w;
            const float stat2 = flag ? w : t;

            #pragma unroll
            for (int i = 0; i < 4; i++) {
                const short f = ((i + threadIdx.x / 2) & 3);
                const uchar bin = bfe(ci, 24 - (f << 3), 8);
                const bool pass = bin != 64;
                int offset = 2 * f;
                offset += 16 * (bin & 62) + 8 * (bin & 1);

                const bool writeFirstFlag = threadIdx.x & 8;

                const float val1 = pass * stat1;

                offset += flag;

                syncTile.sync();

                if (writeFirstFlag) {
                    Buffer[offset] += val1;
                }

                syncTile.sync();

                if (!writeFirstFlag) {
                    Buffer[offset] += val1;
                }


                const float val2 = pass * stat2;

                offset = flag ? offset - 1 : offset + 1;

                syncTile.sync();

                if (writeFirstFlag) {
                    Buffer[offset] += val2;
                }

                syncTile.sync();
                if (!writeFirstFlag) {
                    Buffer[offset] += val2;
                }
            }
        }

        __forceinline__ __device__ void AddPoint2(uint2 bin, const float2 t, const float2 w) {
            AddPoint(bin.x, t.x, w.x);
            AddPoint(bin.y, t.y, w.y);
        }

        __forceinline__ __device__ void AddPoint4(uint4 ci, const float4 t, const float4 w) {

            thread_block_tile<32> syncTile = tiled_partition<32>(this_thread_block());

            const bool flag = threadIdx.x & 1;

            const float4 stat1 = flag ? t : w;
            const float4 stat2 = flag ? w : t;

            #pragma unroll
            for (int i = 0; i < 4; i++) {
                int f = ((2 * i + threadIdx.x) & 6);
                const ui32 shift = static_cast<ui32>(24 - (f << 2));
                f += flag;

                const int binx = (ci.x >> shift) & 255;
                const int biny = (ci.y >> shift) & 255;
                const int binz = (ci.z >> shift) & 255;
                const int binw = (ci.w >> shift) & 255;

                const float passx = binx != 64;
                const float passy = biny != 64;
                const float passz = binz != 64;
                const float passw = binw != 64;

                float* buffer = Buffer + f;

                syncTile.sync();

                int offsetx = 16 * (binx & 62) + 8 * (binx & 1);
                int offsety = 16 * (biny & 62) + 8 * (biny & 1);
                int offsetz = 16 * (binz & 62) + 8 * (binz & 1);
                int offsetw = 16 * (binw & 62) + 8 * (binw & 1);

                const bool writeFirstFlag = threadIdx.x & 8;

                const float valx = passx * stat1.x;
                const float valy = passy * stat1.y;
                const float valz = passz * stat1.z;
                const float valw = passw * stat1.w;


                if (writeFirstFlag) {
                    buffer[offsetx] += valx;
                    buffer[offsety] += valy;
                    buffer[offsetz] += valz;
                    buffer[offsetw] += valw;
                }

                syncTile.sync();

                if (!writeFirstFlag) {
                    buffer[offsetx] += valx;
                    buffer[offsety] += valy;
                    buffer[offsetz] += valz;
                    buffer[offsetw] += valw;
                }


                const float val2x = passx * stat2.x;
                const float val2y = passy * stat2.y;
                const float val2z = passz * stat2.z;
                const float val2w = passw * stat2.w;

                syncTile.sync();

                offsetx += flag ? -1 : 1;
                offsety += flag ? -1 : 1;
                offsetz += flag ? -1 : 1;
                offsetw += flag ? -1 : 1;

                if (writeFirstFlag) {
                    buffer[offsetx] += val2x;
                    buffer[offsety] += val2y;
                    buffer[offsetz] += val2z;
                    buffer[offsetw] += val2w;
                }

                syncTile.sync();

                if (!writeFirstFlag) {
                    buffer[offsetx] += val2x;
                    buffer[offsety] += val2y;
                    buffer[offsetz] += val2z;
                    buffer[offsetw] += val2w;
                }
            }
        }

        //After reduce we store histograms by blocks: 256 floats (4 x 2 x 32)
        // for first 32 bins; than 256 floats for second 32 bins, etc
        __forceinline__ __device__ void Reduce() {

            Buffer -= SliceOffset();
            __syncthreads();

            {
                const int warpHistSize = 1024;

                for (int start = threadIdx.x; start < warpHistSize; start += BLOCK_SIZE) {
                    float sum = 0;
//                    12 iterations at 32-bin
                    #pragma unroll 12
                    for (int i = start; i < 32 * BLOCK_SIZE; i += warpHistSize) {
                        sum += Buffer[i];
                    }
                    Buffer[warpHistSize + start] = sum;
                }
            }

            __syncthreads();

            if (threadIdx.x < 256) {
                const int w = threadIdx.x & 1;
                const int f = threadIdx.x / 64;
                float sum0 = 0.0f;
                float sum1 = 0.0f;
                const int fold0 = (threadIdx.x >> 1) & 31;

                const int maxFoldCount = 64;

                {
                    const int innerHistCount = 2;
                    const volatile float* __restrict__ src = Buffer
                                                    + 1024  //warpHistSize
                                                    + 8 * (fold0 & 1)
                                                    + 32 * (fold0 >> 1)
                                                    + w;

                    #pragma unroll
                    for (int inWarpHist = 0; inWarpHist < innerHistCount; ++inWarpHist) {
                        sum0 += src[2 * f + (inWarpHist << 4)];
                        sum1 += src[2 * f + (inWarpHist << 4) + 512];
                    }

                    Buffer[2 * (maxFoldCount * f + fold0) + w] = sum0;
                    Buffer[2 * (maxFoldCount * f + fold0 + 32) + w] = sum1;
                }
            }
            __syncthreads();
        }
    };


    template<int BLOCK_SIZE>
    struct TPointHist<0, 2, BLOCK_SIZE> {
        float* __restrict__ Buffer;

        __forceinline__ __device__ int SliceOffset() {
            const int warpId = (threadIdx.x / 32);
            const int warpOffset = 1024 * warpId;
            return warpOffset;
        }

        __forceinline__ __device__ TPointHist(float* buff) {
            const int HIST_SIZE = 32 * BLOCK_SIZE;

            #pragma unroll 8
            for (int i = threadIdx.x; i < HIST_SIZE; i += BLOCK_SIZE) {
                buff[i] = 0;
            }

            Buffer = buff + SliceOffset();
            __syncthreads();
        }


        __forceinline__ __device__ void AddPoint(ui32 ci,
                                                 const float t,
                                                 const float w) {

            const bool flag = threadIdx.x & 1;

            const float stat1 = flag ? t : w;
            const float stat2 = flag ? w : t;

            #pragma unroll
            for (int i = 0; i < 4; i++) {
                const int f = ((2 * i + threadIdx.x) & 6);
                const int bin = (ci >> (24 - (f << 2))) & 255;
                const float pass = bin != 128;
                int offset = f;
                offset += 8 * (bin & 127);
//
                const int writeTime = (threadIdx.x >> 3) & 3;

                const float val1 = pass * stat1;
                offset += flag;

                #pragma unroll
                for (int k = 0; k < 4; ++k) {
                    if (k == writeTime) {
                        Buffer[offset] += val1;
                    }
                    __syncwarp();
                }

                const float val2 = pass * stat2;
                offset = flag ? offset - 1 : offset + 1;

                #pragma unroll
                for (int k = 0; k < 4; ++k) {
                    if (k == writeTime) {
                        Buffer[offset] += val2;
                    }
                    __syncwarp();
                }
            }
        }
        __forceinline__ __device__ void AddPoint2(uint2 bin, const float2 t, const float2 w) {
            AddPoint(bin.x, t.x, w.x);
            AddPoint(bin.y, t.y, w.y);
        }

        __forceinline__ __device__ void AddPoint4(uint4 ci, const float4 t, const float4 w) {
            AddPoint(ci.x, t.x, w.x);
            AddPoint(ci.y, t.y, w.y);
            AddPoint(ci.z, t.z, w.z);
            AddPoint(ci.w, t.w, w.w);
        }

        //After reduce we store histograms by blocks: 256 floats (4 x 2 x 32)
        // for first 32 bins; than 256 floats for second 32 bins, etc
        __forceinline__ __device__ void Reduce() {

            Buffer -= SliceOffset();
            __syncthreads();

            {
                const int warpHistSize = 1024;

                for (int start = threadIdx.x; start < warpHistSize; start += BLOCK_SIZE) {
                    float sum = 0;
//                    12 iterations at 32-bin
                    #pragma unroll 12
                    for (int i = start; i < 32 * BLOCK_SIZE; i += warpHistSize) {
                        sum += Buffer[i];
                    }
                    Buffer[warpHistSize + start] = sum;
                }
            }

            __syncthreads();

            if (threadIdx.x < 256) {
                const int w = threadIdx.x & 1;
                const int f = threadIdx.x / 64;
                const int fold0 = (threadIdx.x >> 1) & 31;

                const int maxFoldCount = 128;

                {
                    const volatile float* __restrict__ src = Buffer
                                                             + 1024  //warpHistSize
                                                             + 2 * f
                                                             + w;

                    #pragma unroll
                    for (int k = 0; k < 4; ++k) {
                        int fold = fold0 + 32 * k;
                        Buffer[2 * (maxFoldCount * f + fold) + w] = src[8 * fold];
                    }
                }
            }
            __syncthreads();
        }
    };




    template<int BLOCK_SIZE, int OUTER_HIST_BITS_COUNT, int INNER_HIST_BITS_COUNT, int BLOCKS_PER_FEATURE, bool USE_64_BIT_LOAD>
    __forceinline__ __device__ void ComputeSplitPropertiesPass(const TCFeature* __restrict__ feature, const ui32* __restrict__ cindex,
                                                               const float* __restrict__ target, const float* __restrict__ weight,
                                                               const ui32* __restrict__ indices,
                                                               const TDataPartition* __restrict__ partition, int fCount,
                                                               float* binSumsForPart,
                                                               float* smem) {

        using THist = TPointHist<OUTER_HIST_BITS_COUNT, INNER_HIST_BITS_COUNT, BLOCK_SIZE>;
        const int stripeSize = BLOCK_SIZE;
        const int histBlockCount = 1;



       if (USE_64_BIT_LOAD) {
           #if __CUDA_ARCH__ < 300
           const int OUTER_UNROLL = 2;
           #elif __CUDA_ARCH__ <= 350
           const int OUTER_UNROLL = 2;
           #elif __CUDA_ARCH__ < 700
           const int OUTER_UNROLL =  (INNER_HIST_BITS_COUNT + OUTER_HIST_BITS_COUNT) <= 2 ? 4 : 2;
           #else
           const int OUTER_UNROLL = 1;//(INNER_HIST_BITS_COUNT + OUTER_HIST_BITS_COUNT) <= 2 ? 4 : 2;
           #endif
           const int size = partition->Size;
           const int offset = partition->Offset;

           #if __CUDA_ARCH__ >= 700
           ComputeHistogram4 < stripeSize, OUTER_UNROLL,  histBlockCount, BLOCKS_PER_FEATURE, THist > (indices, offset, size,
                                                                                                       target,
                                                                                                       weight,
                                                                                                       cindex,
                                                                                                       smem);
           #else
           ComputeHistogram2 < stripeSize, OUTER_UNROLL,  histBlockCount, BLOCKS_PER_FEATURE, THist > (indices, offset, size,
                   target,
                   weight,
                   cindex,
                   smem);
           #endif
       } else {
           #if __CUDA_ARCH__ < 300
           const int INNER_UNROLL = (INNER_HIST_BITS_COUNT + OUTER_HIST_BITS_COUNT) == 0 ? 4 : 2;
           const int OUTER_UNROLL = 2;
           #elif __CUDA_ARCH__ <= 350
           const int INNER_UNROLL = (INNER_HIST_BITS_COUNT + OUTER_HIST_BITS_COUNT) == 0 ? 8 : 4;
           const int OUTER_UNROLL = 2;
           #else
           const int INNER_UNROLL = 1;
           const int OUTER_UNROLL = 2;
           #endif
           ComputeHistogram<stripeSize, OUTER_UNROLL, INNER_UNROLL, histBlockCount, BLOCKS_PER_FEATURE, THist>(indices,
                                                                                                               partition->Offset,
                                                                                                               partition->Size,
                                                                                                               target,
                                                                                                               weight,
                                                                                                               cindex,
                                                                                                               smem);
       }
        __syncthreads();

        const int maxFoldCount = (1 << (5 + INNER_HIST_BITS_COUNT + OUTER_HIST_BITS_COUNT));

        const int fid = (threadIdx.x / 64);
        const int w = threadIdx.x & 1;

        const int featureFolds =  fid < fCount ? feature[fid].Folds : 0;
        const int featureOffset = fid * maxFoldCount * 2 + w;

        for (int fold = (threadIdx.x / 2) & 31; fold < featureFolds; fold += 32) {

            if (fid < fCount) {
                const float val = smem[featureOffset + 2 * fold];

                if (abs(val) > 1e-20f) {
                    if (BLOCKS_PER_FEATURE > 1) {
                        atomicAdd(binSumsForPart + (feature[fid].FirstFoldIndex + fold) * 2 + w, val);
                    } else {
                        WriteThrough(binSumsForPart + (feature[fid].FirstFoldIndex + fold) * 2 + w, val);
                    }
                }
            }
        }
    }


#define DECLARE_PASS(O, I, M, USE_64_BIT_LOAD) \
    ComputeSplitPropertiesPass<BLOCK_SIZE, O, I, M, USE_64_BIT_LOAD>(feature, cindex, target, weight, indices, partition, fCount, binSums, &counters[0]);


    template<int BLOCK_SIZE, bool FULL_PASS, int M>
#if __CUDA_ARCH__ == 600
    __launch_bounds__(BLOCK_SIZE, 1)
#elif __CUDA_ARCH__ >= 520
    __launch_bounds__(BLOCK_SIZE, 2)
#else
    __launch_bounds__(BLOCK_SIZE, 1)
#endif
    __global__ void ComputeSplitPropertiesNBImpl(
            const TCFeature* __restrict__ feature, int fCount, const ui32* __restrict__ cindex,
            const float* __restrict__ target, const float* __restrict__ weight,
            const ui32* __restrict__ indices,
            const TDataPartition* __restrict__ partition,
            float* __restrict__ binSums,
            const int totalFeatureCount) {
        TPointwisePartOffsetsHelper helper(gridDim.z);
        helper.ShiftPartAndBinSumsPtr(partition, binSums, totalFeatureCount, FULL_PASS);


        feature += (blockIdx.x / M) * 4;
        cindex += feature->Offset;
        fCount = min(fCount - (blockIdx.x / M) * 4, 4);

        __shared__ float counters[32 * BLOCK_SIZE];
        const int maxBinCount = GetMaxBinCount(feature, fCount, (int*) &counters[0]);
        __syncthreads();

        //CatBoost always use direct loads on first pass of histograms calculation and for this step 64-bits loads are almost x2 faster
        #if __CUDA_ARCH__ > 350
        const bool use64BitLoad =  FULL_PASS;// float2 for target/indices/weights
        #else
        const bool use64BitLoad =  false;
        #endif

        if (partition->Size) {
            if (maxBinCount <= 32) {
                DECLARE_PASS(0, 0, M, use64BitLoad);
            } else if (maxBinCount <= 64) {
                DECLARE_PASS(0, 1, M, use64BitLoad);
            } else if (maxBinCount <= 128) {
                DECLARE_PASS(0, 2, M, use64BitLoad);
            } else {
                DECLARE_PASS(2, 1, M, use64BitLoad);
            }
        }
    }




    template<int BLOCK_SIZE,
            int BLOCKS_PER_FEATURE_COUNT>
    inline void RunComputeHist2NonBinaryKernel(const TCFeature* nbFeatures, int nbCount,
                                               const ui32* cindex,
                                               const float* target, const float* weight, const ui32* indices,
                                               const TDataPartition* partition,
                                               float* binSums, const int binFeatureCount,
                                               bool fullPass,
                                               TCudaStream stream,
                                               dim3 numBlocks)
    {

        if (fullPass) {
            ComputeSplitPropertiesNBImpl < BLOCK_SIZE, true, BLOCKS_PER_FEATURE_COUNT > << <numBlocks, BLOCK_SIZE, 0, stream>>>(
                    nbFeatures, nbCount, cindex, target, weight,
                            indices, partition, binSums, binFeatureCount
            );

        } else {
            ComputeSplitPropertiesNBImpl < BLOCK_SIZE, false, BLOCKS_PER_FEATURE_COUNT > << <numBlocks, BLOCK_SIZE, 0, stream>>>(
                    nbFeatures, nbCount, cindex, target, weight,
                            indices, partition, binSums, binFeatureCount
            );
        }

    }




    void ComputeHist2NonBinary(const TCFeature* nbFeatures, ui32 nbCount,
                               const ui32* cindex,
                               const float* target, const float* weight,
                               const ui32* indices, ui32 size,
                               const TDataPartition* partition, ui32 partCount, ui32 foldCount,
                               bool fullPass,
                               ui32 histLineSize,
                               float* binSums,
                               TCudaStream stream) {
        if (nbCount) {

            dim3 numBlocks;
            numBlocks.x = (nbCount + 3) / 4;
            const int histPartCount = (fullPass ? partCount : partCount / 2);
            numBlocks.y = histPartCount;
            numBlocks.z = foldCount;
            const int blockSize = 384;
            const ui32 multiplier = min(EstimateBlockPerFeatureMultiplier(numBlocks, size), 64);
            numBlocks.x *= multiplier;

            #define COMPUTE(k)\
             RunComputeHist2NonBinaryKernel<blockSize, k>(nbFeatures, nbCount, cindex,  target, weight,  indices, \
                                                          partition, binSums, histLineSize, fullPass, stream, numBlocks);
            if (multiplier == 1) {
                COMPUTE(1)
            } else if (multiplier == 2) {
                COMPUTE(2)
            } else if (multiplier == 4) {
                COMPUTE(4)
            } else if (multiplier == 8) {
                COMPUTE(8)
            } else if (multiplier == 16) {
                COMPUTE(16)
            } else if (multiplier == 32) {
                COMPUTE(32)
            } else if (multiplier == 64) {
                COMPUTE(64)
            } else {
                exit(1);
            }
            #undef COMPUTE
        }
    }
}
