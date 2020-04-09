#pragma once

#include "defaults.h"
/**
 * Fills a vector that indicates whether pages of the calling process's virtual memory are resident in RAM. Each byte
 * in the vector contains the status of a single page. The page size can be obtained via the NSystemInfo::GetPageSize()
 * function. Use the IsPageInCore function to interpret the page status byte.
 *
 * Can be overly pessimistic:
 * - Assumes nothing is in RAM on platforms other than Linux
 * - Recent Linux kernels (4.21 and some backports) may return zeroes if the process doesn't have writing permissions
 *   for the given file. See CVE-2019-5489.
 *
 * @param[in] addr      starting address of the memory range to be examined
 * @param[in] len       length (bytes) of the memory range to be examined
 * @param[out] vec      vector of bytes to store statuses of memory pages
 * @param[in] vecLen    length (bytes) of the vec, should be large enough to hold the requested pages count
 * @throws yexception   if there was a system error or if the vecLen is too small
 *
 * @note                this is only a snapshot, results may be stale by the time they're used
 * @see                 man 2 mincore
 */
void InCoreMemory(const void* addr, size_t len, unsigned char* vec, size_t vecLen);

/**
 * Takes as an argument an element of the vector previously filled by InCoreMemory.
 *
 * @param[in]           byte corresponding to the status of a single page
 *
 * @returns             true if this page was resident in memory at the time out the InCoreMemory execution
 */
inline bool IsPageInCore(unsigned char s) {
    /* From mincore(2): On return, the least significant bit of each byte will be set if the corresponding page is
     * currently resident in memory, and be clear otherwise.  (The settings of the other bits in each byte are
     * undefined; these bits are reserved for possible later use.)
     */
    return s & 1;
}
