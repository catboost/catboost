#pragma once

#include <util/generic/noncopyable.h>
#include <util/generic/string.h>
#include <util/generic/list.h>
#include <util/generic/hash.h>
#include <util/generic/strbuf.h>
#include <util/generic/maybe.h>
#include <util/stream/input.h>
#include <util/stream/output.h>
#include "file.h"
#include "getpid.h"
#include "thread.h"
#include "mutex.h"
#include <sys/types.h>

#include <atomic>

class TShellCommandOptions {
    class TCopyableAtomicBool: public std::atomic<bool> {
    public:
        using std::atomic<bool>::atomic;
        TCopyableAtomicBool(const TCopyableAtomicBool& other)
            : std::atomic<bool>(other.load(std::memory_order_acquire))
        {
        }

        TCopyableAtomicBool& operator=(const TCopyableAtomicBool& other) {
            this->store(other.load(std::memory_order_acquire), std::memory_order_release);
            return *this;
        }
    };

public:
    struct TUserOptions {
        TString Name;
#if defined(_win_)
        TString Password;
#endif
#if defined(_unix_)
        /**
         * Run child process with the user supplementary groups.
         * If true, the user supplementary groups will be set in the child process upon exec().
         * If false, the supplementary groups of the parent process will be used.
         */
        bool UseUserGroups = false;
#endif
    };

    enum EHandleMode {
        HANDLE_INHERIT,
        HANDLE_PIPE,
        HANDLE_STREAM
    };

public:
    inline TShellCommandOptions() noexcept
        : ClearSignalMask(false)
        , CloseAllFdsOnExec(false)
        , AsyncMode(false)
        , PollDelayMs(DefaultSyncPollDelayMs)
        , UseShell(true)
        , QuoteArguments(true)
        , DetachSession(true)
        , CloseStreams(false)
        , ShouldCloseInput(true)
        , InputMode(HANDLE_INHERIT)
        , OutputMode(HANDLE_STREAM)
        , ErrorMode(HANDLE_STREAM)
        , InputStream(nullptr)
        , OutputStream(nullptr)
        , ErrorStream(nullptr)
        , Nice(0)
        , FuncAfterFork(std::function<void()>())
    {
    }

    inline TShellCommandOptions& SetNice(int value) noexcept {
        Nice = value;

        return *this;
    }

    /**
     * @brief clear signal mask from parent process. If true, child process
     * clears the signal mask inherited from the parent process; otherwise
     * child process retains the signal mask of the parent process.
     *
     * @param clearSignalMask true if child process should clear signal mask
     * @note in default child process inherits signal mask.
     * @return self
     */
    inline TShellCommandOptions& SetClearSignalMask(bool clearSignalMask) {
        ClearSignalMask = clearSignalMask;
        return *this;
    }

    /**
     * @brief set close-on-exec mode. If true, all file descriptors
     * from the parent process, except stdin, stdout, stderr, will be closed
     * in the child process upon exec().
     *
     * @param closeAllFdsOnExec true if close-on-exec mode is needed
     * @note in default close-on-exec mode is off.
     * @return self
     */
    inline TShellCommandOptions& SetCloseAllFdsOnExec(bool closeAllFdsOnExec) {
        CloseAllFdsOnExec = closeAllFdsOnExec;
        return *this;
    }

    /**
     * @brief set asynchronous mode. If true, task will be run
     * in separate thread, and control will be returned immediately
     *
     * @param async true if asynchonous mode is needed
     * @note in default async mode launcher will need 100% cpu for rapid process termination
     * @return self
     */
    inline TShellCommandOptions& SetAsync(bool async) {
        AsyncMode = async;
        if (AsyncMode)
            PollDelayMs = 0;
        return *this;
    }

    /**
     * @brief specify delay for process controlling loop
     * @param ms number of milliseconds to poll for
     * @note for synchronous process default of 1s should generally fit
     *       for async process default is no latency and that consumes 100% one cpu
     *       SetAsync(true) will reset this delay to 0, so call this method after
     * @return self
     */
    inline TShellCommandOptions& SetLatency(size_t ms) {
        PollDelayMs = ms;
        return *this;
    }

    /**
     * @brief set the stream, which is input fetched from
     *
     * @param stream Pointer to stream.
     * If stream is NULL or not set, input channel will be closed.
     *
     * @return self
     */
    inline TShellCommandOptions& SetInputStream(IInputStream* stream) {
        InputStream = stream;
        if (InputStream == nullptr) {
            InputMode = HANDLE_INHERIT;
        } else {
            InputMode = HANDLE_STREAM;
        }
        return *this;
    }

    /**
     * @brief set the stream, collecting the command output
     *
     * @param stream Pointer to stream.
     * If stream is NULL or not set, output will be collected to the
     * internal variable
     *
     * @return self
     */
    inline TShellCommandOptions& SetOutputStream(IOutputStream* stream) {
        OutputStream = stream;
        return *this;
    }

    /**
     * @brief set the stream, collecting the command error output
     *
     * @param stream Pointer to stream.
     * If stream is NULL or not set, errors will be collected to the
     * internal variable
     *
     * @return self
     */
    inline TShellCommandOptions& SetErrorStream(IOutputStream* stream) {
        ErrorStream = stream;
        return *this;
    }

    /**
     * @brief set if Finish() should be called on user-supplied streams
     * if process is run in async mode Finish will be called in process' thread
     * @param val if Finish() should be called
     * @return self
     */
    inline TShellCommandOptions& SetCloseStreams(bool val) {
        CloseStreams = val;
        return *this;
    }

    /**
     * @brief set if input stream should be closed after all data is read
     * call SetCloseInput(false) for interactive process
     * @param val if input stream should be closed
     * @return self
     */
    inline TShellCommandOptions& SetCloseInput(bool val) {
        ShouldCloseInput.store(val);
        return *this;
    }

    /**
     * @brief set if command should be interpreted by OS shell (/bin/sh or cmd.exe)
     * shell is enabled by default
     * call SetUseShell(false) for command to be sent to OS verbatim
     * @note shell operators > < | && || will not work if this option is off
     * @param useShell if command should be run in shell
     * @return self
     */
    inline TShellCommandOptions& SetUseShell(bool useShell) {
        UseShell = useShell;
        if (!useShell)
            QuoteArguments = false;
        return *this;
    }

    /**
     * @brief set if the arguments should be wrapped in quotes.
     * Please, note that this option makes no difference between
     * real arguments and shell syntax, so if you execute something
     * like \b TShellCommand("sleep") << "3" << "&&" << "ls", your
     * command will look like:
     *   sleep "3" "&&" "ls"
     * which will never end successfully.
     * By default, this option is turned on.
     *
     * @note arguments will only be quoted if shell is used
     * @param quote if the arguments should be quoted
     *
     * @return self
     */
    inline TShellCommandOptions& SetQuoteArguments(bool quote) {
        QuoteArguments = quote;
        return *this;
    }

    /**
     * @brief set to run command in new session
     * @note set this option to off to deliver parent's signals to command as well
     * @note currently ignored on windows
     * @param detach if command should be run in new session
     * @return self
     */
    inline TShellCommandOptions& SetDetachSession(bool detach) {
        DetachSession = detach;
        return *this;
    }

    /**
     * @brief specifies pure function to be called in the child process after fork, before calling execve
     * @note currently ignored on windows
     * @param function function to be called after fork
     * @return self
     */
    inline TShellCommandOptions& SetFuncAfterFork(const std::function<void()>& function) {
        FuncAfterFork = function;
        return *this;
    }

    /**
     * @brief create a pipe for child input
     * Write end of the pipe will be accessible via TShellCommand::GetInputHandle
     *
     * @return self
     */
    inline TShellCommandOptions& PipeInput() {
        InputMode = HANDLE_PIPE;
        InputStream = nullptr;
        return *this;
    }

    inline TShellCommandOptions& PipeOutput() {
        OutputMode = HANDLE_PIPE;
        OutputStream = nullptr;
        return *this;
    }

    inline TShellCommandOptions& PipeError() {
        ErrorMode = HANDLE_PIPE;
        ErrorStream = nullptr;
        return *this;
    }

    /**
     * @brief set if child should inherit output handle
     *
     * @param inherit if child should inherit output handle
     *
     * @return self
     */
    inline TShellCommandOptions& SetInheritOutput(bool inherit) {
        OutputMode = inherit ? HANDLE_INHERIT : HANDLE_STREAM;
        return *this;
    }

    /**
     * @brief set if child should inherit stderr handle
     *
     * @param inherit if child should inherit error output handle
     *
     * @return self
     */
    inline TShellCommandOptions& SetInheritError(bool inherit) {
        ErrorMode = inherit ? HANDLE_INHERIT : HANDLE_STREAM;
        return *this;
    }

public:
    static constexpr size_t DefaultSyncPollDelayMs = 1000;

public:
    bool ClearSignalMask = false;
    bool CloseAllFdsOnExec = false;
    bool AsyncMode = false;
    size_t PollDelayMs = 0;
    bool UseShell = false;
    bool QuoteArguments = false;
    bool DetachSession = false;
    bool CloseStreams = false;
    TCopyableAtomicBool ShouldCloseInput = false;
    EHandleMode InputMode = HANDLE_STREAM;
    EHandleMode OutputMode = HANDLE_STREAM;
    EHandleMode ErrorMode = HANDLE_STREAM;

    /// @todo more options
    // bool SearchPath // search exe name in $PATH
    // bool UnicodeConsole
    // bool EmulateConsole // provide isatty == true
    /// @todo command's stdin should be exposet as IOutputStream to support dialogue
    IInputStream* InputStream;
    IOutputStream* OutputStream;
    IOutputStream* ErrorStream;
    TUserOptions User;
    THashMap<TString, TString> Environment;
    int Nice = 0;

    std::function<void()> FuncAfterFork = {};
};

/**
 * @brief Execute command in shell and provide its results
 * @attention Not thread-safe
 */
class TShellCommand: public TNonCopyable {
private:
    TShellCommand();

public:
    enum ECommandStatus {
        SHELL_NONE,
        SHELL_RUNNING,
        SHELL_FINISHED,
        SHELL_INTERNAL_ERROR,
        SHELL_ERROR
    };

public:
    /**
     * @brief create the command with initial arguments list
     *
     * @param cmd binary name
     * @param args arguments list
     * @param options execution options
     * @todo store entire options structure
     */
    TShellCommand(const TStringBuf cmd, const TList<TString>& args, const TShellCommandOptions& options = TShellCommandOptions(),
                  const TString& workdir = TString());
    TShellCommand(const TStringBuf cmd, const TShellCommandOptions& options = TShellCommandOptions(), const TString& workdir = TString());
    ~TShellCommand();

public:
    /**
     * @brief append argument to the args list
     *
     * @param argument string argument
     *
     * @return self
     */
    TShellCommand& operator<<(const TStringBuf argument);

    /**
     * @brief return the collected output from the command.
     * If the output stream is set, empty string will be returned
     *
     * @return collected output
     */
    const TString& GetOutput() const;

    /**
     * @brief return the collected error output from the command.
     * If the error stream is set, empty string will be returned
     *
     * @return collected error output
     */
    const TString& GetError() const;

    /**
     * @brief return the internal error occured while watching
     * the command execution. Should be called if execution
     * status is SHELL_INTERNAL_ERROR
     *
     * @return error text
     */
    const TString& GetInternalError() const;

    /**
     * @brief get current status of command execution
     *
     * @return current status
     */
    ECommandStatus GetStatus() const;

    /**
     * @brief return exit code of finished process
     * The value is unspecified in case of internal errors or if the process is running
     *
     * @return exit code
     */
    TMaybe<int> GetExitCode() const;

    /**
     * @brief get id of underlying process
     * @note depends on os: pid_t on UNIX, HANDLE on win
     *
     * @return pid or handle
     */
    TProcessId GetPid() const;

    /**
     * @brief return the file handle that provides input to the child process
     *
     * @return input file handle
     */
    TFileHandle& GetInputHandle();

    /**
     * @brief return the file handle that provides output from the child process
     *
     * @return output file handle
     */
    TFileHandle& GetOutputHandle();

    /**
     * @brief return the file handle that provides error output from the child process
     *
     * @return error file handle
     */
    TFileHandle& GetErrorHandle();

    /**
     * @brief run the execution
     *
     * @return self
     */
    TShellCommand& Run();

    /**
     * @brief terminate the execution
     * @note if DetachSession is set, it terminates all procs in command's new process group
     *
     * @return self
     */
    TShellCommand& Terminate(int signal = SIGTERM);

    /**
     * @brief wait until the execution is finished
     *
     * @return self
     */
    TShellCommand& Wait();

    /**
     * @brief close process' stdin
     *
     * @return self
     */
    TShellCommand& CloseInput();

    /**
     * @brief Get quoted command (for debug/view purposes only!)
     **/
    TString GetQuotedCommand() const;

private:
    class TImpl;
    using TImplRef = TSimpleIntrusivePtr<TImpl>;
    TImplRef Impl;
};

/// Appends to dst: quoted arg
void ShellQuoteArg(TString& dst, TStringBuf arg);

/// Appends to dst: space, quoted arg
void ShellQuoteArgSp(TString& dst, TStringBuf arg);

/// Returns true if arg should be quoted
bool ArgNeedsQuotes(TStringBuf arg) noexcept;
