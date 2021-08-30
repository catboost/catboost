#include "size_literals.h"

void CompileTestUnsigned() {
    static_assert(1_KB == 1024, "Wrong 1KB value");
    static_assert(3_KB == 3 * 1024, "Wrong 3KB value");
    static_assert(41_KB == 41 * 1024, "Wrong 41KB value");
    static_assert(1023_KB == 1023 * 1024, "Wrong 1023KB value");

    static_assert(1_MB == 1_KB * 1024, "Wrong 1MB value");
    static_assert(5_MB == 5_KB * 1024, "Wrong 5MB value");
    static_assert(71_MB == 71_KB * 1024, "Wrong 71MB value");
    static_assert(1023_MB == 1023_KB * 1024, "Wrong 1023MB value");

    static_assert(1_GB == 1_MB * 1024, "Wrong 1GB value");
    static_assert(7_GB == 7_MB * 1024, "Wrong 7GB value");
    static_assert(29_GB == 29_MB * 1024, "Wrong 29GB value");
    static_assert(1023_GB == 1023_MB * 1024, "Wrong 1023GB value");

    static_assert(1_TB == 1_GB * 1024, "Wrong 1TB value");
    static_assert(9_TB == 9_GB * 1024, "Wrong 9TB value");
    static_assert(57_TB == 57_GB * 1024, "Wrong 57TB value");
    static_assert(1023_TB == 1023_GB * 1024, "Wrong 1023TB value");

    static_assert(1_PB == 1_TB * 1024, "Wrong 1PB value");
    static_assert(9_PB == 9_TB * 1024, "Wrong 9PB value");
    static_assert(42_PB == 42_TB * 1024, "Wrong 42PB value");
    static_assert(1023_PB == 1023_TB * 1024, "Wrong 1023PB value");

    static_assert(1_EB == 1_PB * 1024, "Wrong 1EB value");
    static_assert(9_EB == 9_PB * 1024, "Wrong 9EB value");

    static_assert(9000000_TB == 9000000_GB * 1024, "Wrong 9000000TB value");
}

void CompileTestSigned() {
    static_assert(1_KBs == 1024, "Wrong 1KBs value");
    static_assert(3_KBs == 3 * 1024, "Wrong 3KBs value");
    static_assert(41_KBs == 41 * 1024, "Wrong 41KBs value");
    static_assert(1023_KBs == 1023 * 1024, "Wrong 1023KBs value");

    static_assert(1_MBs == 1_KBs * 1024, "Wrong 1MBs value");
    static_assert(5_MBs == 5_KBs * 1024, "Wrong 5MBs value");
    static_assert(71_MBs == 71_KBs * 1024, "Wrong 71MBs value");
    static_assert(1023_MBs == 1023_KBs * 1024, "Wrong 1023MBs value");

    static_assert(1_GBs == 1_MBs * 1024, "Wrong 1GBs value");
    static_assert(7_GBs == 7_MBs * 1024, "Wrong 7GBs value");
    static_assert(29_GBs == 29_MBs * 1024, "Wrong 29GBs value");
    static_assert(1023_GBs == 1023_MBs * 1024, "Wrong 1023GBs value");

    static_assert(1_TBs == 1_GBs * 1024, "Wrong 1TBs value");
    static_assert(9_TBs == 9_GBs * 1024, "Wrong 9TBs value");
    static_assert(57_TBs == 57_GBs * 1024, "Wrong 57TBs value");
    static_assert(1023_TBs == 1023_GBs * 1024, "Wrong 1023TBs value");

    static_assert(1_PBs == 1_TBs * 1024, "Wrong 1PBs value");
    static_assert(9_PBs == 9_TBs * 1024, "Wrong 9PBs value");
    static_assert(42_PBs == 42_TBs * 1024, "Wrong 42PBs value");
    static_assert(1023_PBs == 1023_TBs * 1024, "Wrong 1023PBs value");

    static_assert(1_EBs == 1_PBs * 1024, "Wrong 1EBs value");
    static_assert(7_EBs == 7_PBs * 1024, "Wrong 7EBs value");

    static_assert(8388607_TBs == 8388607_GBs * 1024, "Wrong 8388607TBs value"); // 2**23 - 1 TB

    // Should cause compilation error if uncommented
    //static_assert(8388608_TBs == 8388608_GBs * 1024, "Wrong 8388608TBs value");
}
