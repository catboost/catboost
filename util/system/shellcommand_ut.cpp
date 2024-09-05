#include "shellcommand.h"

#include "compat.h"
#include "defaults.h"
#include "fs.h"
#include "sigset.h"
#include "spinlock.h"

#include <library/cpp/testing/unittest/env.h>
#include <library/cpp/testing/unittest/registar.h>

#include <util/random/random.h>
#include <util/stream/file.h>
#include <util/stream/str.h>
#include <util/stream/mem.h>
#include <util/string/strip.h>
#include <util/folder/tempdir.h>

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
        Stream_.Reserve(100);
    }

    TString Str() const {
        with_lock (Lock_) {
            return Stream_.Str();
        }
        return TString(); // line for compiler
    }

protected:
    size_t DoRead(void* buf, size_t len) override {
        with_lock (Lock_) {
            return Stream_.Read(buf, len);
        }
        return 0; // line for compiler
    }

    void DoWrite(const void* buf, size_t len) override {
        with_lock (Lock_) {
            return Stream_.Write(buf, len);
        }
    }

private:
    TAdaptiveLock Lock_;
    TStringStream Stream_;
};

Y_UNIT_TEST_SUITE(TShellQuoteTest) {
    Y_UNIT_TEST(TestQuoteArg) {
        TString cmd;
        ShellQuoteArg(cmd, "/pr f/krev/prev.exe");
        ShellQuoteArgSp(cmd, "-DVal=\"W Quotes\"");
        ShellQuoteArgSp(cmd, "-DVal=W Space");
        ShellQuoteArgSp(cmd, "-DVal=Blah");
        UNIT_ASSERT_STRINGS_EQUAL(cmd, "\"/pr f/krev/prev.exe\" \"-DVal=\\\"W Quotes\\\"\" \"-DVal=W Space\" \"-DVal=Blah\"");
    }
} // Y_UNIT_TEST_SUITE(TShellQuoteTest)

Y_UNIT_TEST_SUITE(TShellCommandTest) {
    Y_UNIT_TEST(TestNoQuotes) {
        TShellCommandOptions options;
        options.SetQuoteArguments(false);
        TShellCommand cmd("echo hello");
        cmd.Run();
        UNIT_ASSERT_VALUES_EQUAL(cmd.GetError(), "");
        UNIT_ASSERT_VALUES_EQUAL(cmd.GetOutput(), "hello" NL);
        UNIT_ASSERT(TShellCommand::SHELL_FINISHED == cmd.GetStatus());
        UNIT_ASSERT(cmd.GetExitCode().Defined() && 0 == cmd.GetExitCode());

        UNIT_ASSERT_VALUES_EQUAL(cmd.GetQuotedCommand(), "echo hello");
    }

    Y_UNIT_TEST(TestOnlyNecessaryQuotes) {
        TShellCommandOptions options;
        options.SetQuoteArguments(true);
        TShellCommand cmd("echo");
        cmd << "hey"
            << "hello&world";
        cmd.Run();
        UNIT_ASSERT_VALUES_EQUAL(cmd.GetError(), "");
        UNIT_ASSERT_VALUES_EQUAL(cmd.GetOutput(), "hey hello&world" NL);
        UNIT_ASSERT(TShellCommand::SHELL_FINISHED == cmd.GetStatus());
        UNIT_ASSERT(cmd.GetExitCode().Defined() && 0 == cmd.GetExitCode());
    }

    Y_UNIT_TEST(TestRun) {
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
    Y_UNIT_TEST(TestNoShell) {
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
            UNIT_ASSERT_VALUES_EQUAL(cmd.GetError().size(), 0u);
            UNIT_ASSERT(cmd.GetExitCode().Defined() && 0 == cmd.GetExitCode());
        }
    }
    Y_UNIT_TEST(TestAsyncRun) {
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
        UNIT_ASSERT_VALUES_EQUAL(cmd.GetOutput().size(), 0u);
        UNIT_ASSERT(cmd.GetExitCode().Defined() && 0 == cmd.GetExitCode());
#endif
    }
    Y_UNIT_TEST(TestQuotes) {
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
        UNIT_ASSERT_VALUES_EQUAL(cmd.GetError().size(), 0u);
    }
    Y_UNIT_TEST(TestRunNonexistent) {
        TShellCommand cmd("iwerognweiofnewio"); // some nonexistent command name
        cmd.Run().Wait();
        UNIT_ASSERT(TShellCommand::SHELL_ERROR == cmd.GetStatus());
        UNIT_ASSERT_VALUES_UNEQUAL(cmd.GetError().size(), 0u);
        UNIT_ASSERT(cmd.GetExitCode().Defined() && 0 != cmd.GetExitCode());
    }
    Y_UNIT_TEST(TestExitCode) {
        TShellCommand cmd("grep qwerty qwerty"); // some nonexistent file name
        cmd.Run().Wait();
        UNIT_ASSERT(TShellCommand::SHELL_ERROR == cmd.GetStatus());
        UNIT_ASSERT_VALUES_UNEQUAL(cmd.GetError().size(), 0u);
        UNIT_ASSERT(cmd.GetExitCode().Defined() && 2 == cmd.GetExitCode());
    }
    // 'type con' and 'copy con con' want real console, not stdin, use sort
    Y_UNIT_TEST(TestInput) {
        TShellCommandOptions options;
        TString input = (TString("a") * 2000).append(NL) * textSize;
        TStringInput inputStream(input);
        options.SetInputStream(&inputStream);
        TShellCommand cmd(catCommand, options);
        cmd.Run().Wait();
        UNIT_ASSERT_VALUES_EQUAL(input, cmd.GetOutput());
        UNIT_ASSERT_VALUES_EQUAL(cmd.GetError().size(), 0u);
    }
    Y_UNIT_TEST(TestOutput) {
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
        UNIT_ASSERT_VALUES_EQUAL(cmd.GetError().size(), 0u);
    }
    Y_UNIT_TEST(TestIO) {
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
    Y_UNIT_TEST(TestStreamClose) {
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
    Y_UNIT_TEST(TestInterruptSimple) {
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
    Y_UNIT_TEST(TestInterrupt) {
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
            UNIT_ASSERT(cmd.GetExitCode().Defined() && -15 == cmd.GetExitCode());
        }
        sleep(1);
        UNIT_ASSERT(!NFs::Exists(tmpfile));
    }
    // this ut is unix-only (win has no signal mask)
    Y_UNIT_TEST(TestSignalMask) {
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
#else
    // This ut is windows-only
    Y_UNIT_TEST(TestStdinProperlyConstructed) {
        TShellCommandOptions options;
        options.SetErrorStream(&Cerr);

        TShellCommand cmd(BinaryPath("util/system/ut/stdin_osfhandle/stdin_osfhandle"), options);
        cmd.Run().Wait();
        UNIT_ASSERT(TShellCommand::SHELL_FINISHED == cmd.GetStatus());
        UNIT_ASSERT(cmd.GetExitCode().Defined() && 0 == cmd.GetExitCode());
    }
#endif
    Y_UNIT_TEST(TestInternalError) {
        TString input = (TString("a") * 2000).append("\n");
        TStringInput inputStream(input);
        TMemoryOutput outputStream(nullptr, 0);
        TShellCommandOptions options;
        options.SetInputStream(&inputStream);
        options.SetOutputStream(&outputStream);
        TShellCommand cmd(catCommand, options);
        cmd.Run().Wait();
        UNIT_ASSERT(TShellCommand::SHELL_INTERNAL_ERROR == cmd.GetStatus());
        UNIT_ASSERT_VALUES_UNEQUAL(cmd.GetInternalError().size(), 0u);
    }
    Y_UNIT_TEST(TestHugeOutput) {
        TShellCommandOptions options;
        TGuardedStringStream stream;
        options.SetOutputStream(&stream);
        options.SetUseShell(true);

        TString input = TString(7000, 'a');
        TString command = TStringBuilder{} << "echo " << input;
        TShellCommand cmd(command, options);
        cmd.Run().Wait();

        UNIT_ASSERT_VALUES_EQUAL(stream.Str(), input + NL);
    }
    Y_UNIT_TEST(TestHugeError) {
        TShellCommandOptions options;
        TGuardedStringStream stream;
        options.SetErrorStream(&stream);
        options.SetUseShell(true);

        TString input = TString(7000, 'a');
        TString command = TStringBuilder{} << "echo " << input << ">&2";
        TShellCommand cmd(command, options);
        cmd.Run().Wait();

        UNIT_ASSERT_VALUES_EQUAL(stream.Str(), input + NL);
    }
    Y_UNIT_TEST(TestPipeInput) {
        TShellCommandOptions options;
        options.SetAsync(true);
        options.PipeInput();

        TShellCommand cmd(catCommand, options);
        cmd.Run();

        {
            TFile file(cmd.GetInputHandle().Release());
            TUnbufferedFileOutput fo(file);
            fo << "hello" << Endl;
        }

        cmd.Wait();
        UNIT_ASSERT_VALUES_EQUAL(cmd.GetOutput(), "hello" NL);
        UNIT_ASSERT_VALUES_EQUAL(cmd.GetError().size(), 0u);
    }
    Y_UNIT_TEST(TestPipeOutput) {
        TShellCommandOptions options;
        options.SetAsync(true);
        options.PipeOutput();
        constexpr TStringBuf firstMessage = "first message";
        constexpr TStringBuf secondMessage = "second message";
        const TString command = TStringBuilder() << "echo '" << firstMessage << "' && sleep 10 && echo '" << secondMessage << "'";
        TShellCommand cmd(command, options);
        cmd.Run();
        TUnbufferedFileInput cmdOutput(TFile(cmd.GetOutputHandle().Release()));
        TString firstLineOutput, secondLineOutput;
        {
            Sleep(TDuration::Seconds(5));
            firstLineOutput = cmdOutput.ReadLine();
            cmd.Wait();
            secondLineOutput = cmdOutput.ReadLine();
        }
        UNIT_ASSERT_VALUES_EQUAL(firstLineOutput, firstMessage);
        UNIT_ASSERT_VALUES_EQUAL(secondLineOutput, secondLineOutput);
    }
    Y_UNIT_TEST(TestOptionsConsistency) {
        TShellCommandOptions options;

        options.SetInheritOutput(false).SetInheritError(false);
        options.SetOutputStream(nullptr).SetErrorStream(nullptr);

        UNIT_ASSERT(options.OutputMode == TShellCommandOptions::HANDLE_STREAM);
        UNIT_ASSERT(options.ErrorMode == TShellCommandOptions::HANDLE_STREAM);
    }
    Y_UNIT_TEST(TestForkCallback) {
        TString tmpFile = TString("shellcommand_ut.test_for_callback.txt");
        TFsPath cwd(::NFs::CurrentWorkingDirectory());
        const TString tmpFilePath = cwd.Child(tmpFile);

        const TString text = "test output";
        auto afterForkCallback = [&tmpFilePath, &text]() -> void {
            TFixedBufferFileOutput out(tmpFilePath);
            out << text;
        };

        TShellCommandOptions options;
        options.SetFuncAfterFork(afterForkCallback);

        const TString command = "ls";
        TShellCommand cmd(command, options);
        cmd.Run();

        UNIT_ASSERT(NFs::Exists(tmpFilePath));

        TUnbufferedFileInput fileOutput(tmpFilePath);
        TString firstLine = fileOutput.ReadLine();

        UNIT_ASSERT_VALUES_EQUAL(firstLine, text);
    }
} // Y_UNIT_TEST_SUITE(TShellCommandTest)
