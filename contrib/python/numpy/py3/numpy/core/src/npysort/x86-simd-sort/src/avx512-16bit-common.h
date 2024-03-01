/*******************************************************************
 * Copyright (C) 2022 Intel Corporation
 * SPDX-License-Identifier: BSD-3-Clause
 * Authors: Raghuveer Devulapalli <raghuveer.devulapalli@intel.com>
 * ****************************************************************/

#ifndef AVX512_16BIT_COMMON
#define AVX512_16BIT_COMMON

#include "avx512-common-qsort.h"

/*
 * Constants used in sorting 32 elements in a ZMM registers. Based on Bitonic
 * sorting network (see
 * https://en.wikipedia.org/wiki/Bitonic_sorter#/media/File:BitonicSort.svg)
 */
// ZMM register: 31,30,29,28,27,26,25,24,23,22,21,20,19,18,17,16,15,14,13,12,11,10,9,8,7,6,5,4,3,2,1,0
static const uint16_t network[6][32]
        = {{7,  6,  5,  4,  3,  2,  1,  0,  15, 14, 13, 12, 11, 10, 9,  8,
            23, 22, 21, 20, 19, 18, 17, 16, 31, 30, 29, 28, 27, 26, 25, 24},
           {15, 14, 13, 12, 11, 10, 9,  8,  7,  6,  5,  4,  3,  2,  1,  0,
            31, 30, 29, 28, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17, 16},
           {4,  5,  6,  7,  0,  1,  2,  3,  12, 13, 14, 15, 8,  9,  10, 11,
            20, 21, 22, 23, 16, 17, 18, 19, 28, 29, 30, 31, 24, 25, 26, 27},
           {31, 30, 29, 28, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17, 16,
            15, 14, 13, 12, 11, 10, 9,  8,  7,  6,  5,  4,  3,  2,  1,  0},
           {8,  9,  10, 11, 12, 13, 14, 15, 0,  1,  2,  3,  4,  5,  6,  7,
            24, 25, 26, 27, 28, 29, 30, 31, 16, 17, 18, 19, 20, 21, 22, 23},
           {16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31,
            0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15}};

/*
 * Assumes zmm is random and performs a full sorting network defined in
 * https://en.wikipedia.org/wiki/Bitonic_sorter#/media/File:BitonicSort.svg
 */
template <typename vtype, typename zmm_t = typename vtype::zmm_t>
X86_SIMD_SORT_INLINE zmm_t sort_zmm_16bit(zmm_t zmm)
{
    // Level 1
    zmm = cmp_merge<vtype>(
            zmm,
            vtype::template shuffle<SHUFFLE_MASK(2, 3, 0, 1)>(zmm),
            0xAAAAAAAA);
    // Level 2
    zmm = cmp_merge<vtype>(
            zmm,
            vtype::template shuffle<SHUFFLE_MASK(0, 1, 2, 3)>(zmm),
            0xCCCCCCCC);
    zmm = cmp_merge<vtype>(
            zmm,
            vtype::template shuffle<SHUFFLE_MASK(2, 3, 0, 1)>(zmm),
            0xAAAAAAAA);
    // Level 3
    zmm = cmp_merge<vtype>(
            zmm, vtype::permutexvar(vtype::get_network(1), zmm), 0xF0F0F0F0);
    zmm = cmp_merge<vtype>(
            zmm,
            vtype::template shuffle<SHUFFLE_MASK(1, 0, 3, 2)>(zmm),
            0xCCCCCCCC);
    zmm = cmp_merge<vtype>(
            zmm,
            vtype::template shuffle<SHUFFLE_MASK(2, 3, 0, 1)>(zmm),
            0xAAAAAAAA);
    // Level 4
    zmm = cmp_merge<vtype>(
            zmm, vtype::permutexvar(vtype::get_network(2), zmm), 0xFF00FF00);
    zmm = cmp_merge<vtype>(
            zmm, vtype::permutexvar(vtype::get_network(3), zmm), 0xF0F0F0F0);
    zmm = cmp_merge<vtype>(
            zmm,
            vtype::template shuffle<SHUFFLE_MASK(1, 0, 3, 2)>(zmm),
            0xCCCCCCCC);
    zmm = cmp_merge<vtype>(
            zmm,
            vtype::template shuffle<SHUFFLE_MASK(2, 3, 0, 1)>(zmm),
            0xAAAAAAAA);
    // Level 5
    zmm = cmp_merge<vtype>(
            zmm, vtype::permutexvar(vtype::get_network(4), zmm), 0xFFFF0000);
    zmm = cmp_merge<vtype>(
            zmm, vtype::permutexvar(vtype::get_network(5), zmm), 0xFF00FF00);
    zmm = cmp_merge<vtype>(
            zmm, vtype::permutexvar(vtype::get_network(3), zmm), 0xF0F0F0F0);
    zmm = cmp_merge<vtype>(
            zmm,
            vtype::template shuffle<SHUFFLE_MASK(1, 0, 3, 2)>(zmm),
            0xCCCCCCCC);
    zmm = cmp_merge<vtype>(
            zmm,
            vtype::template shuffle<SHUFFLE_MASK(2, 3, 0, 1)>(zmm),
            0xAAAAAAAA);
    return zmm;
}

// Assumes zmm is bitonic and performs a recursive half cleaner
template <typename vtype, typename zmm_t = typename vtype::zmm_t>
X86_SIMD_SORT_INLINE zmm_t bitonic_merge_zmm_16bit(zmm_t zmm)
{
    // 1) half_cleaner[32]: compare 1-17, 2-18, 3-19 etc ..
    zmm = cmp_merge<vtype>(
            zmm, vtype::permutexvar(vtype::get_network(6), zmm), 0xFFFF0000);
    // 2) half_cleaner[16]: compare 1-9, 2-10, 3-11 etc ..
    zmm = cmp_merge<vtype>(
            zmm, vtype::permutexvar(vtype::get_network(5), zmm), 0xFF00FF00);
    // 3) half_cleaner[8]
    zmm = cmp_merge<vtype>(
            zmm, vtype::permutexvar(vtype::get_network(3), zmm), 0xF0F0F0F0);
    // 3) half_cleaner[4]
    zmm = cmp_merge<vtype>(
            zmm,
            vtype::template shuffle<SHUFFLE_MASK(1, 0, 3, 2)>(zmm),
            0xCCCCCCCC);
    // 3) half_cleaner[2]
    zmm = cmp_merge<vtype>(
            zmm,
            vtype::template shuffle<SHUFFLE_MASK(2, 3, 0, 1)>(zmm),
            0xAAAAAAAA);
    return zmm;
}

// Assumes zmm1 and zmm2 are sorted and performs a recursive half cleaner
template <typename vtype, typename zmm_t = typename vtype::zmm_t>
X86_SIMD_SORT_INLINE void bitonic_merge_two_zmm_16bit(zmm_t &zmm1, zmm_t &zmm2)
{
    // 1) First step of a merging network: coex of zmm1 and zmm2 reversed
    zmm2 = vtype::permutexvar(vtype::get_network(4), zmm2);
    zmm_t zmm3 = vtype::min(zmm1, zmm2);
    zmm_t zmm4 = vtype::max(zmm1, zmm2);
    // 2) Recursive half cleaner for each
    zmm1 = bitonic_merge_zmm_16bit<vtype>(zmm3);
    zmm2 = bitonic_merge_zmm_16bit<vtype>(zmm4);
}

// Assumes [zmm0, zmm1] and [zmm2, zmm3] are sorted and performs a recursive
// half cleaner
template <typename vtype, typename zmm_t = typename vtype::zmm_t>
X86_SIMD_SORT_INLINE void bitonic_merge_four_zmm_16bit(zmm_t *zmm)
{
    zmm_t zmm2r = vtype::permutexvar(vtype::get_network(4), zmm[2]);
    zmm_t zmm3r = vtype::permutexvar(vtype::get_network(4), zmm[3]);
    zmm_t zmm_t1 = vtype::min(zmm[0], zmm3r);
    zmm_t zmm_t2 = vtype::min(zmm[1], zmm2r);
    zmm_t zmm_t3 = vtype::permutexvar(vtype::get_network(4),
                                      vtype::max(zmm[1], zmm2r));
    zmm_t zmm_t4 = vtype::permutexvar(vtype::get_network(4),
                                      vtype::max(zmm[0], zmm3r));
    zmm_t zmm0 = vtype::min(zmm_t1, zmm_t2);
    zmm_t zmm1 = vtype::max(zmm_t1, zmm_t2);
    zmm_t zmm2 = vtype::min(zmm_t3, zmm_t4);
    zmm_t zmm3 = vtype::max(zmm_t3, zmm_t4);
    zmm[0] = bitonic_merge_zmm_16bit<vtype>(zmm0);
    zmm[1] = bitonic_merge_zmm_16bit<vtype>(zmm1);
    zmm[2] = bitonic_merge_zmm_16bit<vtype>(zmm2);
    zmm[3] = bitonic_merge_zmm_16bit<vtype>(zmm3);
}

template <typename vtype, typename type_t>
X86_SIMD_SORT_INLINE void sort_32_16bit(type_t *arr, int32_t N)
{
    typename vtype::opmask_t load_mask = ((0x1ull << N) - 0x1ull) & 0xFFFFFFFF;
    typename vtype::zmm_t zmm
            = vtype::mask_loadu(vtype::zmm_max(), load_mask, arr);
    vtype::mask_storeu(arr, load_mask, sort_zmm_16bit<vtype>(zmm));
}

template <typename vtype, typename type_t>
X86_SIMD_SORT_INLINE void sort_64_16bit(type_t *arr, int32_t N)
{
    if (N <= 32) {
        sort_32_16bit<vtype>(arr, N);
        return;
    }
    using zmm_t = typename vtype::zmm_t;
    typename vtype::opmask_t load_mask
            = ((0x1ull << (N - 32)) - 0x1ull) & 0xFFFFFFFF;
    zmm_t zmm1 = vtype::loadu(arr);
    zmm_t zmm2 = vtype::mask_loadu(vtype::zmm_max(), load_mask, arr + 32);
    zmm1 = sort_zmm_16bit<vtype>(zmm1);
    zmm2 = sort_zmm_16bit<vtype>(zmm2);
    bitonic_merge_two_zmm_16bit<vtype>(zmm1, zmm2);
    vtype::storeu(arr, zmm1);
    vtype::mask_storeu(arr + 32, load_mask, zmm2);
}

template <typename vtype, typename type_t>
X86_SIMD_SORT_INLINE void sort_128_16bit(type_t *arr, int32_t N)
{
    if (N <= 64) {
        sort_64_16bit<vtype>(arr, N);
        return;
    }
    using zmm_t = typename vtype::zmm_t;
    using opmask_t = typename vtype::opmask_t;
    zmm_t zmm[4];
    zmm[0] = vtype::loadu(arr);
    zmm[1] = vtype::loadu(arr + 32);
    opmask_t load_mask1 = 0xFFFFFFFF, load_mask2 = 0xFFFFFFFF;
    if (N != 128) {
        uint64_t combined_mask = (0x1ull << (N - 64)) - 0x1ull;
        load_mask1 = combined_mask & 0xFFFFFFFF;
        load_mask2 = (combined_mask >> 32) & 0xFFFFFFFF;
    }
    zmm[2] = vtype::mask_loadu(vtype::zmm_max(), load_mask1, arr + 64);
    zmm[3] = vtype::mask_loadu(vtype::zmm_max(), load_mask2, arr + 96);
    zmm[0] = sort_zmm_16bit<vtype>(zmm[0]);
    zmm[1] = sort_zmm_16bit<vtype>(zmm[1]);
    zmm[2] = sort_zmm_16bit<vtype>(zmm[2]);
    zmm[3] = sort_zmm_16bit<vtype>(zmm[3]);
    bitonic_merge_two_zmm_16bit<vtype>(zmm[0], zmm[1]);
    bitonic_merge_two_zmm_16bit<vtype>(zmm[2], zmm[3]);
    bitonic_merge_four_zmm_16bit<vtype>(zmm);
    vtype::storeu(arr, zmm[0]);
    vtype::storeu(arr + 32, zmm[1]);
    vtype::mask_storeu(arr + 64, load_mask1, zmm[2]);
    vtype::mask_storeu(arr + 96, load_mask2, zmm[3]);
}

template <typename vtype, typename type_t>
X86_SIMD_SORT_INLINE type_t get_pivot_16bit(type_t *arr,
                                            const int64_t left,
                                            const int64_t right)
{
    // median of 32
    int64_t size = (right - left) / 32;
    type_t vec_arr[32] = {arr[left],
                          arr[left + size],
                          arr[left + 2 * size],
                          arr[left + 3 * size],
                          arr[left + 4 * size],
                          arr[left + 5 * size],
                          arr[left + 6 * size],
                          arr[left + 7 * size],
                          arr[left + 8 * size],
                          arr[left + 9 * size],
                          arr[left + 10 * size],
                          arr[left + 11 * size],
                          arr[left + 12 * size],
                          arr[left + 13 * size],
                          arr[left + 14 * size],
                          arr[left + 15 * size],
                          arr[left + 16 * size],
                          arr[left + 17 * size],
                          arr[left + 18 * size],
                          arr[left + 19 * size],
                          arr[left + 20 * size],
                          arr[left + 21 * size],
                          arr[left + 22 * size],
                          arr[left + 23 * size],
                          arr[left + 24 * size],
                          arr[left + 25 * size],
                          arr[left + 26 * size],
                          arr[left + 27 * size],
                          arr[left + 28 * size],
                          arr[left + 29 * size],
                          arr[left + 30 * size],
                          arr[left + 31 * size]};
    typename vtype::zmm_t rand_vec = vtype::loadu(vec_arr);
    typename vtype::zmm_t sort = sort_zmm_16bit<vtype>(rand_vec);
    return ((type_t *)&sort)[16];
}

template <typename vtype, typename type_t>
static void
qsort_16bit_(type_t *arr, int64_t left, int64_t right, int64_t max_iters)
{
    /*
     * Resort to std::sort if quicksort isnt making any progress
     */
    if (max_iters <= 0) {
        std::sort(arr + left, arr + right + 1, comparison_func<vtype>);
        return;
    }
    /*
     * Base case: use bitonic networks to sort arrays <= 128
     */
    if (right + 1 - left <= 128) {
        sort_128_16bit<vtype>(arr + left, (int32_t)(right + 1 - left));
        return;
    }

    type_t pivot = get_pivot_16bit<vtype>(arr, left, right);
    type_t smallest = vtype::type_max();
    type_t biggest = vtype::type_min();
    int64_t pivot_index = partition_avx512<vtype>(
            arr, left, right + 1, pivot, &smallest, &biggest);
    if (pivot != smallest)
        qsort_16bit_<vtype>(arr, left, pivot_index - 1, max_iters - 1);
    if (pivot != biggest)
        qsort_16bit_<vtype>(arr, pivot_index, right, max_iters - 1);
}

template <typename vtype, typename type_t>
static void qselect_16bit_(type_t *arr,
                           int64_t pos,
                           int64_t left,
                           int64_t right,
                           int64_t max_iters)
{
    /*
     * Resort to std::sort if quicksort isnt making any progress
     */
    if (max_iters <= 0) {
        std::sort(arr + left, arr + right + 1, comparison_func<vtype>);
        return;
    }
    /*
     * Base case: use bitonic networks to sort arrays <= 128
     */
    if (right + 1 - left <= 128) {
        sort_128_16bit<vtype>(arr + left, (int32_t)(right + 1 - left));
        return;
    }

    type_t pivot = get_pivot_16bit<vtype>(arr, left, right);
    type_t smallest = vtype::type_max();
    type_t biggest = vtype::type_min();
    int64_t pivot_index = partition_avx512<vtype>(
            arr, left, right + 1, pivot, &smallest, &biggest);
    if ((pivot != smallest) && (pos < pivot_index))
        qselect_16bit_<vtype>(arr, pos, left, pivot_index - 1, max_iters - 1);
    else if ((pivot != biggest) && (pos >= pivot_index))
        qselect_16bit_<vtype>(arr, pos, pivot_index, right, max_iters - 1);
}

#endif // AVX512_16BIT_COMMON
