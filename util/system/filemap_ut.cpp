#include <library/cpp/testing/unittest/registar.h>

#ifdef _unix_
    #include <sys/resource.h>
#endif

#include "filemap.h"

#include <util/system/fs.h>

#include <cstring>
#include <cstdio>

Y_UNIT_TEST_SUITE(TFileMapTest) {
    static const char* FileName_("./mappped_file");

    void BasicTest(TMemoryMapCommon::EOpenMode mode) {
        char data[] = "abcdefgh";

        TFile file(FileName_, CreateAlways | WrOnly);
        file.Write(static_cast<void*>(data), sizeof(data));
        file.Close();

        {
            TFileMap mappedFile(FileName_, mode);
            mappedFile.Map(0, mappedFile.Length());
            UNIT_ASSERT(mappedFile.MappedSize() == sizeof(data) && mappedFile.Length() == sizeof(data));
            UNIT_ASSERT(mappedFile.IsOpen());
            for (size_t i = 0; i < sizeof(data); ++i) {
                UNIT_ASSERT(static_cast<char*>(mappedFile.Ptr())[i] == data[i]);
                static_cast<char*>(mappedFile.Ptr())[i] = data[i] + 1;
            }
            mappedFile.Flush();

            TFileMap::TMapResult mapResult = mappedFile.Map(2, 2);
            UNIT_ASSERT(mapResult.MappedSize() == 2);
            UNIT_ASSERT(mapResult.MappedData() == mappedFile.Ptr());
            UNIT_ASSERT(mappedFile.MappedSize() == 2);
            UNIT_ASSERT(static_cast<char*>(mappedFile.Ptr())[0] == 'd' && static_cast<char*>(mappedFile.Ptr())[1] == 'e');

            mappedFile.Unmap();
            UNIT_ASSERT(mappedFile.MappedSize() == 0);

            FILE* f = fopen(FileName_, "rb");
            TFileMap mappedFile2(f);
            mappedFile2.Map(0, mappedFile2.Length());
            UNIT_ASSERT(mappedFile2.MappedSize() == sizeof(data));
            UNIT_ASSERT(static_cast<char*>(mappedFile2.Ptr())[0] == data[0] + 1);
            fclose(f);
        }
        NFs::Remove(FileName_);
    }

    Y_UNIT_TEST(TestFileMap) {
        BasicTest(TMemoryMapCommon::oRdWr);
    }

    Y_UNIT_TEST(TestFileMapPopulate) {
        BasicTest(TMemoryMapCommon::oRdWr | TMemoryMapCommon::oPopulate);
    }

    Y_UNIT_TEST(TestFileRemap) {
        const char data1[] = "01234";
        const char data2[] = "abcdefg";
        const char data3[] = "COPY";
        const char dataFinal[] = "012abcdefg";
        const size_t data2Shift = 3;

        TFile file(FileName_, CreateAlways | WrOnly);
        file.Write(static_cast<const void*>(data1), sizeof(data1));
        file.Close();

        {
            TFileMap mappedFile(FileName_, TMemoryMapCommon::oRdWr);
            mappedFile.Map(0, mappedFile.Length());
            UNIT_ASSERT(mappedFile.MappedSize() == sizeof(data1) &&
                        mappedFile.Length() == sizeof(data1));

            mappedFile.ResizeAndRemap(data2Shift, sizeof(data2));
            memcpy(mappedFile.Ptr(), data2, sizeof(data2));
        }

        {
            TFileMap mappedFile(FileName_, TMemoryMapCommon::oCopyOnWr);
            mappedFile.Map(0, mappedFile.Length());
            UNIT_ASSERT(mappedFile.MappedSize() == sizeof(dataFinal) &&
                        mappedFile.Length() == sizeof(dataFinal));

            char* data = static_cast<char*>(mappedFile.Ptr());
            UNIT_ASSERT(data[0] == '0');
            UNIT_ASSERT(data[3] == 'a');
            memcpy(data, data3, sizeof(data3));
            UNIT_ASSERT(data[0] == 'C');
            UNIT_ASSERT(data[3] == 'Y');
        }

        TFile resFile(FileName_, RdOnly);
        UNIT_ASSERT(resFile.GetLength() == sizeof(dataFinal));
        char buf[sizeof(dataFinal)];
        resFile.Read(buf, sizeof(dataFinal));
        UNIT_ASSERT(0 == memcmp(buf, dataFinal, sizeof(dataFinal)));
        resFile.Close();

        NFs::Remove(FileName_);
    }

    Y_UNIT_TEST(TestFileMapDbgName) {
        // This test checks that dbgName passed to the TFileMap constructor is saved inside the object and appears
        // in subsequent error messages.
        const char* const dbgName = "THIS_IS_A_TEST";
        FILE* f = fopen(FileName_, "w+");
        UNIT_ASSERT(f);
        {
            TFileMap mappedFile(f, TFileMap::oRdWr, dbgName);
            bool gotException = false;
            try {
                // trying to map an empty file to force an exception and check the message
                mappedFile.Map(0, 1000);
            } catch (const yexception& e) {
                gotException = true;
                UNIT_ASSERT_STRING_CONTAINS(e.what(), dbgName);
            }
            UNIT_ASSERT(gotException);
        }
        fclose(f);
        NFs::Remove(FileName_);
    }

#if defined(_asan_enabled_) || defined(_msan_enabled_)
// setrlimit incompatible with asan runtime
#elif defined(_cygwin_)
// cygwin is not real unix :(
#else
    Y_UNIT_TEST(TestNotGreedy) {
        unsigned page[4096 / sizeof(unsigned)];

    #if defined(_unix_)
        // Temporary limit allowed virtual memory size to 1Gb
        struct rlimit rlim;

        if (getrlimit(RLIMIT_AS, &rlim)) {
            throw TSystemError() << "Cannot get rlimit for virtual memory";
        }

        rlim_t Limit = 1 * 1024 * 1024 * 1024;

        if (rlim.rlim_cur > Limit) {
            rlim.rlim_cur = Limit;

            if (setrlimit(RLIMIT_AS, &rlim)) {
                throw TSystemError() << "Cannot set rlimit for virtual memory to 1Gb";
            }
        }
    #endif
        // Make a 128M test file
        try {
            TFile file(FileName_, CreateAlways | WrOnly);

            for (unsigned pages = 128 * 1024 * 1024 / sizeof(page), i = 0; pages--; i++) {
                std::fill(page, page + sizeof(page) / sizeof(*page), i);
                file.Write(page, sizeof(page));
            }

            file.Close();

            // Make 16 maps of our file, which would require 16*128M = 2Gb and exceed our 1Gb limit
            TVector<THolder<TFileMap>> maps;

            for (int i = 0; i < 16; ++i) {
                maps.emplace_back(MakeHolder<TFileMap>(FileName_, TMemoryMapCommon::oRdOnly | TMemoryMapCommon::oNotGreedy));
                maps.back()->Map(i * sizeof(page), sizeof(page));
            }

            // Oh, good, we're not dead yet
            for (int i = 0; i < 16; ++i) {
                TFileMap& map = *maps[i];

                UNIT_ASSERT_EQUAL(map.Length(), 128 * 1024 * 1024);
                UNIT_ASSERT_EQUAL(map.MappedSize(), sizeof(page));

                const int* mappedPage = (const int*)map.Ptr();

                for (size_t j = 0; j < sizeof(page) / sizeof(*page); ++j) {
                    UNIT_ASSERT_EQUAL(mappedPage[j], i);
                }
            }

    #if defined(_unix_)
            // Restore limits and cleanup
            rlim.rlim_cur = rlim.rlim_max;

            if (setrlimit(RLIMIT_AS, &rlim)) {
                throw TSystemError() << "Cannot restore rlimit for virtual memory";
            }
    #endif
            maps.clear();
            NFs::Remove(FileName_);
        } catch (...) {
    // TODO: RAII'ize all this stuff
    #if defined(_unix_)
            rlim.rlim_cur = rlim.rlim_max;

            if (setrlimit(RLIMIT_AS, &rlim)) {
                throw TSystemError() << "Cannot restore rlimit for virtual memory";
            }
    #endif
            NFs::Remove(FileName_);

            throw;
        }
    }
#endif

    Y_UNIT_TEST(TestFileMappedArray) {
        {
            TFileMappedArray<ui32> mappedArray;
            ui32 data[] = {123, 456, 789, 10};
            size_t sz = sizeof(data) / sizeof(data[0]);

            TFile file(FileName_, CreateAlways | WrOnly);
            file.Write(static_cast<void*>(data), sizeof(data));
            file.Close();

            mappedArray.Init(FileName_);
            // actual test begin
            UNIT_ASSERT(mappedArray.Size() == sz);
            for (size_t i = 0; i < sz; ++i) {
                UNIT_ASSERT(mappedArray[i] == data[i]);
            }

            UNIT_ASSERT(mappedArray.GetAt(mappedArray.Size()) == 0);
            UNIT_ASSERT(*mappedArray.Begin() == data[0]);
            UNIT_ASSERT(size_t(mappedArray.End() - mappedArray.Begin()) == sz);
            UNIT_ASSERT(!mappedArray.Empty());
            // actual test end
            mappedArray.Term();

            // Init array via file mapping
            TFileMap fileMap(FileName_);
            fileMap.Map(0, fileMap.Length());
            mappedArray.Init(fileMap);

            // actual test begin
            UNIT_ASSERT(mappedArray.Size() == sz);
            for (size_t i = 0; i < sz; ++i) {
                UNIT_ASSERT(mappedArray[i] == data[i]);
            }

            UNIT_ASSERT(mappedArray.GetAt(mappedArray.Size()) == 0);
            UNIT_ASSERT(*mappedArray.Begin() == data[0]);
            UNIT_ASSERT(size_t(mappedArray.End() - mappedArray.Begin()) == sz);
            UNIT_ASSERT(!mappedArray.Empty());
            // actual test end

            file = TFile(FileName_, WrOnly);
            file.Seek(0, sEnd);
            file.Write("x", 1);
            file.Close();

            bool caught = false;
            try {
                mappedArray.Init(FileName_);
            } catch (const yexception&) {
                caught = true;
            }
            UNIT_ASSERT(caught);
        }
        NFs::Remove(FileName_);
    }

    Y_UNIT_TEST(TestMappedArray) {
        ui32 sz = 10;

        TMappedArray<ui32> mappedArray;

        ui32* ptr = mappedArray.Create(sz);
        UNIT_ASSERT(ptr != nullptr);
        UNIT_ASSERT(mappedArray.size() == sz);
        UNIT_ASSERT(mappedArray.begin() + sz == mappedArray.end());

        for (size_t i = 0; i < sz; ++i) {
            mappedArray[i] = (ui32)i;
        }
        for (size_t i = 0; i < sz; ++i) {
            UNIT_ASSERT(mappedArray[i] == i);
        }

        TMappedArray<ui32> mappedArray2(1000);
        mappedArray.swap(mappedArray2);
        UNIT_ASSERT(mappedArray.size() == 1000 && mappedArray2.size() == sz);
    }

    Y_UNIT_TEST(TestMemoryMap) {
        TFile file(FileName_, CreateAlways | WrOnly);
        file.Close();

        FILE* f = fopen(FileName_, "rb");
        UNIT_ASSERT(f != nullptr);
        try {
            TMemoryMap mappedMem(f);
            mappedMem.Map(mappedMem.Length() / 2, mappedMem.Length() + 100); // overflow
            UNIT_ASSERT(0);                                                  // should not go here
        } catch (yexception& exc) {
            TString text = exc.what(); // exception should contain failed file name
            UNIT_ASSERT(text.find(TMemoryMapCommon::UnknownFileName()) != TString::npos);
            fclose(f);
        }

        TFile fileForMap(FileName_, OpenExisting);
        try {
            TMemoryMap mappedMem(fileForMap);
            mappedMem.Map(mappedMem.Length() / 2, mappedMem.Length() + 100); // overflow
            UNIT_ASSERT(0);                                                  // should not go here
        } catch (yexception& exc) {
            TString text = exc.what(); // exception should contain failed file name
            UNIT_ASSERT(text.find(FileName_) != TString::npos);
        }
        NFs::Remove(FileName_);
    }

    Y_UNIT_TEST(TestMemoryMapIsWritable) {
        TFile file(FileName_, CreateAlways | WrOnly);
        file.Close();

        {
            TMemoryMap mappedMem(FileName_, TMemoryMap::oRdOnly);
            UNIT_ASSERT(!mappedMem.IsWritable());
        }
        {
            TMemoryMap mappedMem(FileName_, TMemoryMap::oRdWr);
            UNIT_ASSERT(mappedMem.IsWritable());
        }
        NFs::Remove(FileName_);
    }

    Y_UNIT_TEST(TestFileMapIsWritable) {
        TFile file(FileName_, CreateAlways | WrOnly);
        file.Close();
        {
            TMemoryMap mappedMem(FileName_, TMemoryMap::oRdOnly);
            TFileMap fileMap(mappedMem);
            UNIT_ASSERT(!fileMap.IsWritable());
        }
        {
            TMemoryMap mappedMem(FileName_, TMemoryMap::oRdWr);
            TFileMap fileMap(mappedMem);
            UNIT_ASSERT(fileMap.IsWritable());
        }
        {
            TFileMap fileMap(FileName_, TFileMap::oRdOnly);
            UNIT_ASSERT(!fileMap.IsWritable());
        }
        {
            TFileMap fileMap(FileName_, TFileMap::oRdWr);
            UNIT_ASSERT(fileMap.IsWritable());
        }
        NFs::Remove(FileName_);
    }
} // Y_UNIT_TEST_SUITE(TFileMapTest)
