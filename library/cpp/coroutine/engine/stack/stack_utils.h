#pragma once

#include "stack_common.h"


namespace NCoro::NStack {
    /*! Actual size of allocated memory can exceed size in pages, due to unaligned allocation.
     * @param sizeInPages : number of pages to allocate
     * @param rawPtr : pointer to unaligned memory. Should be passed to free() when is not used any more.
     * @param alignedPtr : pointer to beginning of first fully allocated page
     * @return : true on success
     */
    bool GetAlignedMemory(size_t sizeInPages, char*& rawPtr, char*& alignedPtr) noexcept;

    /*! Release mapped RSS memory.
     *  @param alignedPt : page-size aligned memory on which RSS memory should be freed
     *  @param numOfPages : number of pages to free from RSS memory
     */
    void ReleaseRss(char* alignedPtr, size_t numOfPages) noexcept;

    /*! Count pages with RSS memory
     * @param alignedPtr : pointer to page-aligned memory for which calculations would be done
     * @param numOfPages : number of pages to check
     * @return : number of pages with RSS memory
     */
    size_t CountMapped(char* alignedPtr, size_t numOfPages) noexcept;
}
