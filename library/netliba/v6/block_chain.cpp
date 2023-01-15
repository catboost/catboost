#include "stdafx.h"
#include "block_chain.h"

#include <util/system/unaligned_mem.h>

namespace NNetliba {
    ui32 CalcChecksum(const void* p, int size) {
        //return 0;
        //return CalcCrc32(p, size);
        i64 sum = 0;
        const unsigned char *pp = (const unsigned char*)p, *pend = pp + size;
        for (const unsigned char* pend4 = pend - 3; pp < pend4; pp += 4)
            sum += *(const ui32*)pp;

        ui32 left = 0, pos = 0;
        for (; pp < pend; ++pp) {
            pos += ((ui32)*pp) << left;
            left += 8;
        }

        sum += pos;
        sum = (sum & 0xffffffff) + (sum >> 32);
        sum += sum >> 32;
        return (ui32)~sum;
    }

    ui32 CalcChecksum(const TBlockChain& chain) {
        TIncrementalChecksumCalcer ics;
        AddChain(&ics, chain);
        return ics.CalcChecksum();
    }

    void TIncrementalChecksumCalcer::AddBlock(const void* p, int size) {
        ui32 sum = CalcBlockSum(p, size);
        AddBlockSum(sum, size);
    }

    void TIncrementalChecksumCalcer::AddBlockSum(ui32 sum, int size) {
        for (int k = 0; k < Offset; ++k)
            sum = (sum >> 24) + ((sum & 0xffffff) << 8);
        TotalSum += sum;

        Offset = (Offset + size) & 3;
    }

    ui32 TIncrementalChecksumCalcer::CalcBlockSum(const void* p, int size) {
        i64 sum = 0;
        const unsigned char *pp = (const unsigned char*)p, *pend = pp + size;
        for (const unsigned char* pend4 = pend - 3; pp < pend4; pp += 4)
            sum += ReadUnaligned<ui32>(pp);

        ui32 left = 0, pos = 0;
        for (; pp < pend; ++pp) {
            pos += ((ui32)*pp) << left;
            left += 8;
        }
        sum += pos;
        sum = (sum & 0xffffffff) + (sum >> 32);
        sum += sum >> 32;
        return (ui32)sum;
    }

    ui32 TIncrementalChecksumCalcer::CalcChecksum() {
        TotalSum = (TotalSum & 0xffffffff) + (TotalSum >> 32);
        TotalSum += TotalSum >> 32;
        return (ui32)~TotalSum;
    }

    //void TestChainChecksum()
    //{
    //    TVector<char> data;
    //    data.resize(10);
    //    for (int i = 0; i < data.ysize(); ++i)
    //        data[i] = rand();
    //    int crc1 = CalcChecksum(&data[0], data.size());
    //
    //    TBlockChain chain;
    //    TIncrementalChecksumCalcer incCS;
    //    for (int offset = 0; offset < data.ysize();) {
    //        int sz = Min(rand() % 10, data.ysize() - offset);
    //        chain.AddBlock(&data[offset], sz);
    //        incCS.AddBlock(&data[offset], sz);
    //        offset += sz;
    //    }
    //    int crc2 = CalcChecksum(chain);
    //    Y_ASSERT(crc1 == crc2);
    //    int crc3 = incCS.CalcChecksum();
    //    Y_ASSERT(crc1 == crc3);
    //}
}
