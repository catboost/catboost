#include "shellcommand.h"

#include "compat.h"
#include "defaults.h"
#include "fs.h"
#include "sigset.h"
#include "spinlock.h"

#include <library/unittest/registar.h>

#include <util/folder/dirut.h>
#include <util/random/random.h>
#include <util/stream/str.h>
#include <util/stream/mem.h>
#include <util/string/strip.h>

#if defined(_win_)
#define NL "\r\n"
const char catCommand[] = "sort"; // not really cat but ok
const size_t textSize = 1;
#else
#define NL "\n"
const char catCommand[] = "/bin/cat";
const size_t textSize = 20000;
#endif

class TGuardedStringStream: public IInputStream, public IOutputStream {
public:
    TGuardedStringStream() {
        Stream.Reserve(100);
    }

    TString Str() const {
        with_lock (Lock) {
            return Stream.Str();
        }
        return TString(); // line for compiler
    }

protected:
    size_t DoRead(void* buf, size_t len) override {
        with_lock (Lock) {
            return Stream.Read(buf, len);
        }
        return 0; // line for compiler
    }

    void DoWrite(const void* buf, size_t len) override {
        with_lock (Lock) {
            return Stream.Write(buf, len);
        }
    }

private:
    TAdaptiveLock Lock;
    TStringStream Stream;
};

SIMPLE_UNIT_TEST_SUITE(TShellQuoteTest) {
    SIMPLE_UNIT_TEST(TestQuoteArg) {
        TString cmd;
        ShellQuoteArg(cmd, "/pr f/krev/prev.exe");
        ShellQuoteArgSp(cmd, "-DVal=\"W Quotes\"");
        ShellQuoteArgSp(cmd, "-DVal=W Space");
        ShellQuoteArgSp(cmd, "-DVal=Blah");
        UNIT_ASSERT_STRINGS_EQUAL(cmd, "\"/pr f/krev/prev.exe\" \"-DVal=\\\"W Quotes\\\"\" \"-DVal=W Space\" \"-DVal=Blah\"");
    }
}

SIMPLE_UNIT_TEST_SUITE(TShellCommandTest) {
    SIMPLE_UNIT_TEST(TestNoQuotes) {
        TShellCommandOptions options;
        options.SetQuoteArguments(false);
        TShellCommand cmd("echo hello");
        cmd.Run();
        UNIT_ASSERT_VALUES_EQUAL(cmd.GetError(), "");
        UNIT_ASSERT_VALUES_EQUAL(cmd.GetOutput(), "hello" NL);
        UNIT_ASSERT(TShellCommand::SHELL_FINISHED == cmd.GetStatus());
        UNIT_ASSERT(cmd.GetExitCode().Defined() && 0 == cmd.GetExitCode());
    }

    SIMPLE_UNIT_TEST(TestRun) {
        TShellCommand cmd("echo");
        cmd << "hello";
        cmd.Run();
        UNIT_ASSERT_VALUES_EQUAL(cmd.GetError(), "");
#if defined(_win_)
        UNIT_ASSERT_VALUES_EQUAL(cmd.GetOutput(), "\"hello\"\r\n");
#else
        UNIT_ASSERT_VALUES_EQUAL(cmd.GetOutput(), "hello\n");
#endif
        UNIT_ASSERT(TShellCommand::SHELL_FINISHED == cmd.GetStatus());
        UNIT_ASSERT(cmd.GetExitCode().Defined() && 0 == cmd.GetExitCode());
    }
    // running with no shell is not implemented for win
    // there should be no problem with it as long as SearchPath is on
    SIMPLE_UNIT_TEST(TestNoShell) {
#if defined(_win_)
        const char dir[] = "dir";
#else
        const char dir[] = "ls";
#endif

        TShellCommandOptions options;
        options.SetQuoteArguments(false);

        {
            options.SetUseShell(false);
            TShellCommand cmd(dir, options);
            cmd << "|"
                << "sort";

            cmd.Run();
            UNIT_ASSERT(TShellCommand::SHELL_ERROR == cmd.GetStatus());
            UNIT_ASSERT(cmd.GetExitCode().Defined() && 0 != cmd.GetExitCode());
        }
        {
            options.SetUseShell(true);
            TShellCommand cmd(dir, options);
            cmd << "|"
                << "sort";
            cmd.Run();
            UNIT_ASSERT(TShellCommand::SHELL_FINISHED == cmd.GetStatus());
            UNIT_ASSERT_VALUES_EQUAL(+cmd.GetError(), 0u);
            UNIT_ASSERT(cmd.GetExitCode().Defined() && 0 == cmd.GetExitCode());
        }
    }
    SIMPLE_UNIT_TEST(TestAsyncRun) {
        TShellCommandOptions options;
        options.SetAsync(true);
#if defined(_win_)
        // fails with weird error "Input redirection is not supported"
        // TShellCommand cmd("sleep", options);
        // cmd << "3";
        TShellCommand cmd("ping 1.1.1.1 -n 1 -w 2000", options);
#else
        TShellCommand cmd("sleep", options);
        cmd << "2";
#endif
        UNIT_ASSERT(TShellCommand::SHELL_NONE == cmd.GetStatus());
        cmd.Run();
        sleep(1);
        UNIT_ASSERT(TShellCommand::SHELL_RUNNING == cmd.GetStatus());
        cmd.Wait();
        UNIT_ASSERT(TShellCommand::SHELL_RUNNING != cmd.GetStatus());
        UNIT_ASSERT_VALUES_EQUAL(cmd.GetError(), "");
#if !defined(_win_)
        UNIT_ASSERT(TShellCommand::SHELL_FINISHED == cmd.GetStatus());
        UNIT_ASSERT_VALUES_EQUAL(+cmd.GetOutput(), 0u);
        UNIT_ASSERT(cmd.GetExitCode().Defined() && 0 == cmd.GetExitCode());
#endif
    }
    SIMPLE_UNIT_TEST(TestQuotes) {
        TShellCommandOptions options;
        TString input = TString("a\"a a");
        TString output;
        TStringOutput outputStream(output);
        options.SetOutputStream(&outputStream);
        TShellCommand cmd("echo", options);
        cmd << input;
        cmd.Run().Wait();
        output = StripString(output);
#if defined(_win_)
        UNIT_ASSERT_VALUES_EQUAL("\"a\\\"a a\"", output);
#else
        UNIT_ASSERT_VALUES_EQUAL(input, output);
#endif
        UNIT_ASSERT_VALUES_EQUAL(+cmd.GetError(), 0u);
    }
    SIMPLE_UNIT_TEST(TestRunNonexistent) {
        TShellCommand cmd("iwerognweiofnewio"); // some nonexistent command name
        cmd.Run().Wait();
        UNIT_ASSERT(TShellCommand::SHELL_ERROR == cmd.GetStatus());
        UNIT_ASSERT_VALUES_UNEQUAL(+cmd.GetError(), 0u);
        UNIT_ASSERT(cmd.GetExitCode().Defined() && 0 != cmd.GetExitCode());
    }
    SIMPLE_UNIT_TEST(TestExitCode) {
        TShellCommand cmd("grep qwerty qwerty"); // some nonexistent file name
        cmd.Run().Wait();
        UNIT_ASSERT(TShellCommand::SHELL_ERROR == cmd.GetStatus());
        UNIT_ASSERT_VALUES_UNEQUAL(+cmd.GetError(), 0u);
        UNIT_ASSERT(cmd.GetExitCode().Defined() && 2 == cmd.GetExitCode());
    }
    // 'type con' and 'copy con con' want real console, not stdin, use sort
    SIMPLE_UNIT_TEST(TestInput) {
        TShellCommandOptions options;
        TString input = (TString("a") * 2000).append(NL) * textSize;
        TStringInput inputStream(input);
        options.SetInputStream(&inputStream);
        TShellCommand cmd(catCommand, options);
        cmd.Run().Wait();
        UNIT_ASSERT_VALUES_EQUAL(input, cmd.GetOutput());
        UNIT_ASSERT_VALUES_EQUAL(+cmd.GetError(), 0u);
    }
    SIMPLE_UNIT_TEST(TestOutput) {
        TShellCommandOptions options;
        TString input = (TString("a") * 2000).append(NL) * textSize;
        TStringInput inputStream(input);
        options.SetInputStream(&inputStream);
        TString output;
        TStringOutput outputStream(output);
        options.SetOutputStream(&outputStream);
        TShellCommand cmd(catCommand, options);
        cmd.Run().Wait();
        UNIT_ASSERT_VALUES_EQUAL(input, output);
        UNIT_ASSERT_VALUES_EQUAL(+cmd.GetError(), 0u);
    }
    SIMPLE_UNIT_TEST(TestIO) {
        // descriptive test: use all options
        TShellCommandOptions options;
        options.SetAsync(true);
        options.SetQuoteArguments(false);
        options.SetLatency(10);
        options.SetClearSignalMask(true);
        options.SetCloseAllFdsOnExec(true);
        options.SetCloseInput(false);
        TGuardedStringStream write;
        options.SetInputStream(&write);
        TGuardedStringStream read;
        options.SetOutputStream(&read);
        options.SetUseShell(true);

        TShellCommand cmd("cat", options);
        cmd.Run();

        write << "alpha" << NL;
        while (read.Str() != "alpha" NL) {
            Sleep(TDuration::MilliSeconds(10));
        }

        write << "omega" << NL;
        while (read.Str() != "alpha" NL "omega" NL) {
            Sleep(TDuration::MilliSeconds(10));
        }

        write << "zeta" << NL;
        cmd.CloseInput();
        cmd.Wait();

        UNIT_ASSERT_VALUES_EQUAL(cmd.GetError(), "");
        UNIT_ASSERT(TShellCommand::SHELL_FINISHED == cmd.GetStatus());
        UNIT_ASSERT_VALUES_EQUAL(read.Str(), "alpha" NL "omega" NL "zeta" NL);
        UNIT_ASSERT(cmd.GetExitCode().Defined() && 0 == cmd.GetExitCode());
    }
    SIMPLE_UNIT_TEST(TestStreamClose) {
        struct TStream: public IOutputStream {
            size_t NumCloses = 0;
            void DoWrite(const void* buf, size_t len) override {
                Y_UNUSED(buf);
                Y_UNUSED(len);
            }
            void DoFinish() override {
                ++NumCloses;
            }
        } stream;

        auto options1 = TShellCommandOptions().SetCloseStreams(false).SetOutputStream(&stream).SetErrorStream(&stream);
        TShellCommand("echo hello", options1).Run().Wait();
        UNIT_ASSERT_VALUES_EQUAL(stream.NumCloses, 0);

        auto options = TShellCommandOptions().SetCloseStreams(true).SetOutputStream(&stream).SetErrorStream(&stream);
        TShellCommand("echo hello", options).Run().Wait();
        UNIT_ASSERT_VALUES_EQUAL(stream.NumCloses, 2);
    }
    SIMPLE_UNIT_TEST(TestInterruptSimple) {
        TShellCommandOptions options;
        options.SetAsync(true);
        options.SetCloseInput(false);
        TGuardedStringStream write;
        options.SetInputStream(&write); // set input stream that will be waited by cat
        TShellCommand cmd(catCommand, options);
        cmd.Run();
        sleep(1);
        UNIT_ASSERT(TShellCommand::SHELL_RUNNING == cmd.GetStatus());
        cmd.Terminate();
        cmd.Wait();
        UNIT_ASSERT(TShellCommand::SHELL_RUNNING != cmd.GetStatus());
    }
#if !defined(_win_)
    // this ut is unix-only, port to win using %TEMP%
    SIMPLE_UNIT_TEST(TestInterrupt) {
        TString tmpfile = TString("shellcommand_ut.interrupt.") + ToString(RandomNumber<ui32>());

        TShellCommandOptions options;
        options.SetAsync(true);
        options.SetQuoteArguments(false);
        {
            TShellCommand cmd("/bin/sleep", options);
            cmd << " 1300 & wait; /usr/bin/touch " << tmpfile;
            cmd.Run();
            sleep(1);
            UNIT_ASSERT(TShellCommand::SHELL_RUNNING == cmd.GetStatus());
            // Async mode requires Terminate() + Wait() to send kill to child proc!
            cmd.Terminate();
            cmd.Wait();
            UNIT_ASSERT(TShellCommand::SHELL_ERROR == cmd.GetStatus());
        }
        sleep(1);
        UNIT_ASSERT(!NFs::Exists(tmpfile));
    }
    // this ut is unix-only (win has no signal mask)
    SIMPLE_UNIT_TEST(TestSignalMask) {
        // block SIGTERM
        int rc;
        sigset_t newmask, oldmask;
        SigEmptySet(&newmask);
        SigAddSet(&newmask, SIGTERM);
        rc = SigProcMask(SIG_SETMASK, &newmask, &oldmask);
        UNIT_ASSERT(rc == 0);

        TString tmpfile = TString("shellcommand_ut.interrupt.") + ToString(RandomNumber<ui32>());

        TShellCommandOptions options;
        options.SetAsync(true);
        options.SetQuoteArguments(false);

        // child proc should not receive SIGTERM anymore
        {
            TShellCommand cmd("/bin/sleep", options);
            // touch file only if sleep not interrupted by SIGTERM
            cmd << " 10 & wait; [ $? == 0 ] || /usr/bin/touch " << tmpfile;
            cmd.Run();
            sleep(1);
            UNIT_ASSERT(TShellCommand::SHELL_RUNNING == cmd.GetStatus());
            cmd.Terminate();
            cmd.Wait();
            UNIT_ASSERT(TShellCommand::SHELL_ERROR == cmd.GetStatus() || TShellCommand::SHELL_FINISHED == cmd.GetStatus());
        }
        sleep(1);
        UNIT_ASSERT(!NFs::Exists(tmpfile));

        // child proc should receive SIGTERM
        options.SetClearSignalMask(true);
        {
            TShellCommand cmd("/bin/sleep", options);
            // touch file regardless -- it will be interrupted
            cmd << " 10 & wait; /usr/bin/touch " << tmpfile;
            cmd.Run();
            sleep(1);
            UNIT_ASSERT(TShellCommand::SHELL_RUNNING == cmd.GetStatus());
            cmd.Terminate();
            cmd.Wait();
            UNIT_ASSERT(TShellCommand::SHELL_ERROR == cmd.GetStatus());
        }
        sleep(1);
        UNIT_ASSERT(!NFs::Exists(tmpfile));

        // restore signal mask
        rc = SigProcMask(SIG_SETMASK, &oldmask, nullptr);
        UNIT_ASSERT(rc == 0);
    }
#endif
    SIMPLE_UNIT_TEST(TestInternalError) {
        TString input = (TString("a") * 2000).append("\n");
        TStringInput inputStream(input);
        TMemoryOutput outputStream(nullptr, 0);
        TShellCommandOptions options;
        options.SetInputStream(&inputStream);
        options.SetOutputStream(&outputStream);
        TShellCommand cmd(catCommand, options);
        cmd.Run().Wait();
        UNIT_ASSERT(TShellCommand::SHELL_INTERNAL_ERROR == cmd.GetStatus());
        UNIT_ASSERT_VALUES_UNEQUAL(+cmd.GetInternalError(), 0u);
    }
}
