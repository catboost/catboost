#pragma once

class IAllocator;

/*
 * return anonymous memory based allocator
 */
IAllocator* MmapAllocator();
