#include "input.h"
#include "output.h"

#include <library/cpp/testing/unittest/registar.h>

#include <util/system/file.h>
#include <util/system/yassert.h>

#ifdef _win_
    #include <io.h>
#endif

class TMockStdIn {
public:
    TMockStdIn()
        : StdInCopy_(dup(0))
    {
    }
    ~TMockStdIn() {
        close(StdInCopy_);
    }

    template <typename FuncType>
    void ForInput(const TStringBuf text, const FuncType& func) {
        TFile tempFile(TFile::Temporary("input_ut"));
        tempFile.Write(text.data(), text.size());
        tempFile.FlushData();
        tempFile.Seek(0, sSet);

        TFileHandle tempFh(tempFile.GetHandle());
        tempFh.Duplicate2Posix(0);
        tempFh.Release();

        func();
        Cin.ReadAll();
        dup2(StdInCopy_, 0);
        clearerr(stdin);
    }

private:
    int StdInCopy_;
};

class TNoInput: public IInputStream {
public:
    TNoInput(ui64 size)
        : Size_(size)
    {
    }

protected:
    size_t DoRead(void*, size_t len) override {
        len = Min(static_cast<ui64>(len), Size_);
        Size_ -= len;
        return len;
    }

private:
    ui64 Size_;
};

class TNoOutput: public IOutputStream {
public:
    TNoOutput() = default;

protected:
    void DoWrite(const void*, size_t) override {
    }
};

class TSimpleStringInput: public IInputStream {
public:
    TSimpleStringInput(const TString& string)
        : String_(string)
    {
    }

protected:
    size_t DoRead(void* buf, size_t len) override {
        Y_ASSERT(len != 0);

        if (String_.empty()) {
            return 0;
        }

        *static_cast<char*>(buf) = String_[0];
        String_.remove(0, 1);
        return 1;
    }

private:
    TString String_;
};

Y_UNIT_TEST_SUITE(TInputTest) {
    Y_UNIT_TEST(BigTransfer) {
        ui64 size = 1024ull * 1024ull * 1024ull * 5;
        TNoInput input(size);
        TNoOutput output;

        ui64 transferred = TransferData(&input, &output);

        UNIT_ASSERT_VALUES_EQUAL(transferred, size);
    }

    Y_UNIT_TEST(TestReadTo) {
        /* This one tests default implementation of ReadTo. */

        TSimpleStringInput in("0123456789abc");

        TString t;
        UNIT_ASSERT_VALUES_EQUAL(in.ReadTo(t, '7'), 8);
        UNIT_ASSERT_VALUES_EQUAL(t, "0123456");
        UNIT_ASSERT_VALUES_EQUAL(in.ReadTo(t, 'z'), 5);
        UNIT_ASSERT_VALUES_EQUAL(t, "89abc");
        UNIT_ASSERT_VALUES_EQUAL(in.ReadTo(t, 'z'), 0);
        UNIT_ASSERT_VALUES_EQUAL(t, "89abc");
    }

    Y_UNIT_TEST(TestReadLine) {
        TSimpleStringInput in("1\n22\n333");

        TString t;
        UNIT_ASSERT_VALUES_EQUAL(in.ReadLine(t), 2);
        UNIT_ASSERT_VALUES_EQUAL(t, "1");
        UNIT_ASSERT_VALUES_EQUAL(in.ReadLine(t), 3);
        UNIT_ASSERT_VALUES_EQUAL(t, "22");
        UNIT_ASSERT_VALUES_EQUAL(in.ReadLine(t), 3);
        UNIT_ASSERT_VALUES_EQUAL(t, "333");
        UNIT_ASSERT_VALUES_EQUAL(in.ReadLine(t), 0);
        UNIT_ASSERT_VALUES_EQUAL(t, "333");
    }

    Y_UNIT_TEST(TestStdInReadTo) {
        std::pair<std::pair<TStringBuf, char>, TStringBuf> testPairs[] = {
            {{"", '\n'}, ""},
            {{"\n", '\n'}, ""},
            {{"\n\t", '\t'}, "\n"},
            {{"\t\n", '\n'}, "\t"},
            {{"a\tb\n", '\t'}, "a"}};

        TMockStdIn stdIn;

        for (const auto& testPair : testPairs) {
            const TStringBuf text = testPair.first.first;
            const char delim = testPair.first.second;
            const TStringBuf expectedValue = testPair.second;

            stdIn.ForInput(text,
                           [=] {
                               TString value;
                               Cin.ReadTo(value, delim);
                               UNIT_ASSERT_VALUES_EQUAL(value, expectedValue);
                           });
        }
    }
} // Y_UNIT_TEST_SUITE(TInputTest)
