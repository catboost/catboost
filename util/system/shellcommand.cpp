#include "shellcommand.h"
#include "user.h"
#include "nice.h"
#include "sigset.h"

#include <util/folder/dirut.h>
#include <util/generic/algorithm.h>
#include <util/generic/buffer.h>
#include <util/generic/vector.h>
#include <util/generic/yexception.h>
#include <util/memory/tempbuf.h>
#include <util/network/socket.h>
#include <util/stream/pipe.h>
#include <util/stream/str.h>
#include <util/string/cast.h>
#include <util/system/info.h>

#include <errno.h>

#if defined(_unix_)
    #include <unistd.h>
    #include <fcntl.h>
    #include <grp.h>
    #include <sys/wait.h>

using TPid = pid_t;
using TWaitResult = pid_t;
using TExitStatus = int;
    #define WAIT_PROCEED 0

    #if defined(_darwin_)
using TGetGroupListGid = int;
    #else
using TGetGroupListGid = gid_t;
    #endif
#elif defined(_win_)
    #include <string>

    #include "winint.h"

using TPid = HANDLE;
using TWaitResult = DWORD;
using TExitStatus = DWORD;
    #define WAIT_PROCEED WAIT_TIMEOUT

    #pragma warning(disable : 4296) // 'wait_result >= WAIT_OBJECT_0' : expression is always tru
#else
    #error("unknown os, shell command is not implemented")
#endif

#define DBG(stmt) \
    {}
// #define DBG(stmt) stmt

namespace {
    constexpr static size_t DATA_BUFFER_SIZE = 128 * 1024;

#if defined(_unix_)
    void SetUserGroups(const passwd* pw) {
        int ngroups = 1;
        THolder<gid_t, TFree> groups = THolder<gid_t, TFree>(static_cast<gid_t*>(malloc(ngroups * sizeof(gid_t))));
        if (getgrouplist(pw->pw_name, pw->pw_gid, reinterpret_cast<TGetGroupListGid*>(groups.Get()), &ngroups) == -1) {
            groups.Reset(static_cast<gid_t*>(malloc(ngroups * sizeof(gid_t))));
            if (getgrouplist(pw->pw_name, pw->pw_gid, reinterpret_cast<TGetGroupListGid*>(groups.Get()), &ngroups) == -1) {
                ythrow TSystemError() << "getgrouplist failed: user " << pw->pw_name << " (" << pw->pw_uid << ")";
            }
        }
        if (setgroups(ngroups, groups.Get()) == -1) {
            ythrow TSystemError(errno) << "Unable to set groups for user " << pw->pw_name << Endl;
        }
    }

    void ImpersonateUser(const TShellCommandOptions::TUserOptions& userOpts) {
        if (GetUsername() == userOpts.Name) {
            return;
        }
        const passwd* newUser = getpwnam(userOpts.Name.c_str());
        if (!newUser) {
            ythrow TSystemError(errno) << "getpwnam failed";
        }
        if (userOpts.UseUserGroups) {
            SetUserGroups(newUser);
        }
        if (setuid(newUser->pw_uid)) {
            ythrow TSystemError(errno) << "setuid failed";
        }
    }
#elif defined(_win_)
    constexpr static size_t MAX_COMMAND_LINE = 32 * 1024;

    std::wstring GetWString(const char* astring) {
        if (!astring) {
            return std::wstring();
        }

        std::string str(astring);
        return std::wstring(str.begin(), str.end());
    }

    std::string GetAString(const wchar_t* wstring) {
        if (!wstring) {
            return std::string();
        }

        std::wstring str(wstring);
        return std::string(str.begin(), str.end());
    }
#endif
} // namespace

// temporary measure to avoid rewriting all poll calls on win TPipeHandle
#if defined(_win_)
using REALPIPEHANDLE = HANDLE;
    #define INVALID_REALPIPEHANDLE INVALID_HANDLE_VALUE

class TRealPipeHandle
    : public TNonCopyable {
public:
    inline TRealPipeHandle() noexcept
        : Fd_(INVALID_REALPIPEHANDLE)
    {
    }

    inline TRealPipeHandle(REALPIPEHANDLE fd) noexcept
        : Fd_(fd)
    {
    }

    inline ~TRealPipeHandle() {
        Close();
    }

    bool Close() noexcept {
        bool ok = true;
        if (Fd_ != INVALID_REALPIPEHANDLE) {
            ok = CloseHandle(Fd_);
        }
        Fd_ = INVALID_REALPIPEHANDLE;
        return ok;
    }

    inline REALPIPEHANDLE Release() noexcept {
        REALPIPEHANDLE ret = Fd_;
        Fd_ = INVALID_REALPIPEHANDLE;
        return ret;
    }

    inline void Swap(TRealPipeHandle& r) noexcept {
        DoSwap(Fd_, r.Fd_);
    }

    inline operator REALPIPEHANDLE() const noexcept {
        return Fd_;
    }

    inline bool IsOpen() const noexcept {
        return Fd_ != INVALID_REALPIPEHANDLE;
    }

    ssize_t Read(void* buffer, size_t byteCount) const noexcept {
        DWORD doneBytes;
        if (!ReadFile(Fd_, buffer, byteCount, &doneBytes, nullptr)) {
            return -1;
        }
        return doneBytes;
    }
    ssize_t Write(const void* buffer, size_t byteCount) const noexcept {
        DWORD doneBytes;
        if (!WriteFile(Fd_, buffer, byteCount, &doneBytes, nullptr)) {
            return -1;
        }
        return doneBytes;
    }

    static void Pipe(TRealPipeHandle& reader, TRealPipeHandle& writer, EOpenMode mode) {
        (void)mode;
        REALPIPEHANDLE fds[2];
        if (!CreatePipe(&fds[0], &fds[1], nullptr /* handles are not inherited */, 0)) {
            ythrow TFileError() << "failed to create a pipe";
        }
        TRealPipeHandle(fds[0]).Swap(reader);
        TRealPipeHandle(fds[1]).Swap(writer);
    }

private:
    REALPIPEHANDLE Fd_;
};

#else
using TRealPipeHandle = TPipeHandle;
using REALPIPEHANDLE = PIPEHANDLE;
    #define INVALID_REALPIPEHANDLE INVALID_PIPEHANDLE
#endif

class TShellCommand::TImpl
    : public TAtomicRefCount<TShellCommand::TImpl> {
private:
    TString Command;
    TList<TString> Arguments;
    TShellCommandOptions Options_;
    TString WorkDir;

    TShellCommandOptions::EHandleMode InputMode = TShellCommandOptions::HANDLE_STREAM;

    TPid Pid;
    std::atomic<size_t> ExecutionStatus; // TShellCommand::ECommandStatus
    TThread* WatchThread;
    bool TerminateFlag = false;

    TMaybe<int> ExitCode;
    TString CollectedOutput;
    TString CollectedError;
    TString InternalError;
    TMutex TerminateMutex;
    TFileHandle InputHandle;
    TFileHandle OutputHandle;
    TFileHandle ErrorHandle;

private:
    struct TProcessInfo {
        TImpl* Parent;
        TRealPipeHandle InputFd;
        TRealPipeHandle OutputFd;
        TRealPipeHandle ErrorFd;
        TProcessInfo(TImpl* parent, REALPIPEHANDLE inputFd, REALPIPEHANDLE outputFd, REALPIPEHANDLE errorFd)
            : Parent(parent)
            , InputFd(inputFd)
            , OutputFd(outputFd)
            , ErrorFd(errorFd)
        {
        }
    };

    struct TPipes {
        TRealPipeHandle OutputPipeFd[2];
        TRealPipeHandle ErrorPipeFd[2];
        TRealPipeHandle InputPipeFd[2];
        // pipes are closed by automatic dtor
        void PrepareParents() {
            if (OutputPipeFd[1].IsOpen()) {
                OutputPipeFd[1].Close();
            }
            if (ErrorPipeFd[1].IsOpen()) {
                ErrorPipeFd[1].Close();
            }
            if (InputPipeFd[1].IsOpen()) {
                InputPipeFd[0].Close();
            }
        }
        void ReleaseParents() {
            InputPipeFd[1].Release();
            OutputPipeFd[0].Release();
            ErrorPipeFd[0].Release();
        }
    };

    struct TPipePump {
        TRealPipeHandle* Pipe;
        IOutputStream* OutputStream;
        IInputStream* InputStream;
        std::atomic<bool>* ShouldClosePipe;
        TString InternalError;
    };

#if defined(_unix_)
    void OnFork(TPipes& pipes, sigset_t oldmask, char* const* argv, char* const* envp, const std::function<void()>& afterFork) const;
#else
    void StartProcess(TPipes& pipes);
#endif

public:
    inline TImpl(const TStringBuf cmd, const TList<TString>& args, const TShellCommandOptions& options, const TString& workdir)
        : Command(ToString(cmd))
        , Arguments(args)
        , Options_(options)
        , WorkDir(workdir)
        , InputMode(options.InputMode)
        , Pid(0)
        , ExecutionStatus(SHELL_NONE)
        , WatchThread(nullptr)
        , TerminateFlag(false)
    {
        if (Options_.InputStream) {
            // TODO change usages to call SetInputStream instead of directly assigning to InputStream
            InputMode = TShellCommandOptions::HANDLE_STREAM;
        }
    }

    inline ~TImpl() {
        if (WatchThread) {
            with_lock (TerminateMutex) {
                TerminateFlag = true;
            }

            delete WatchThread;
        }

#if defined(_win_)
        if (Pid) {
            CloseHandle(Pid);
        }
#endif
    }

    inline void AppendArgument(const TStringBuf argument) {
        if (ExecutionStatus.load(std::memory_order_acquire) == SHELL_RUNNING) {
            ythrow yexception() << "You cannot change command parameters while process is running";
        }
        Arguments.push_back(ToString(argument));
    }

    inline const TString& GetOutput() const {
        if (ExecutionStatus.load(std::memory_order_acquire) == SHELL_RUNNING) {
            ythrow yexception() << "You cannot retrieve output while process is running.";
        }
        return CollectedOutput;
    }

    inline const TString& GetError() const {
        if (ExecutionStatus.load(std::memory_order_acquire) == SHELL_RUNNING) {
            ythrow yexception() << "You cannot retrieve output while process is running.";
        }
        return CollectedError;
    }

    inline const TString& GetInternalError() const {
        if (ExecutionStatus.load(std::memory_order_acquire) != SHELL_INTERNAL_ERROR) {
            ythrow yexception() << "Internal error hasn't occured so can't be retrieved.";
        }
        return InternalError;
    }

    inline ECommandStatus GetStatus() const {
        return static_cast<ECommandStatus>(ExecutionStatus.load(std::memory_order_acquire));
    }

    inline TMaybe<int> GetExitCode() const {
        return ExitCode;
    }

    inline TProcessId GetPid() const {
#if defined(_win_)
        return GetProcessId(Pid);
#else
        return Pid;
#endif
    }

    inline TFileHandle& GetInputHandle() {
        return InputHandle;
    }

    inline TFileHandle& GetOutputHandle() {
        return OutputHandle;
    }

    inline TFileHandle& GetErrorHandle() {
        return ErrorHandle;
    }

    // start child process
    void Run();

    inline void Terminate(int signal) {
        if (!!Pid && (ExecutionStatus.load(std::memory_order_acquire) == SHELL_RUNNING)) {
#if defined(_unix_)
            bool ok = kill(Options_.DetachSession ? -1 * Pid : Pid, signal) == 0;
            if (!ok && (errno == ESRCH) && Options_.DetachSession) {
                // this could fail when called before child proc completes setsid().
                ok = kill(Pid, signal) == 0;
                kill(-Pid, signal); // between a failed kill(-Pid) and a successful kill(Pid) a grandchild could have been spawned
            }
#else
            Y_UNUSED(signal);
            bool ok = TerminateProcess(Pid, 1 /* exit code */);
#endif
            if (!ok) {
                ythrow TSystemError() << "cannot terminate " << Pid;
            }
        }
    }

    inline void Wait() {
        if (WatchThread) {
            WatchThread->Join();
        }
    }

    inline void CloseInput() {
        Options_.ShouldCloseInput.store(true);
    }

    inline static bool TerminateIsRequired(void* processInfo) {
        TProcessInfo* pi = reinterpret_cast<TProcessInfo*>(processInfo);
        if (!pi->Parent->TerminateFlag) {
            return false;
        }
        pi->InputFd.Close();
        pi->ErrorFd.Close();
        pi->OutputFd.Close();

        if (pi->Parent->Options_.CloseStreams) {
            if (pi->Parent->Options_.ErrorStream) {
                pi->Parent->Options_.ErrorStream->Finish();
            }
            if (pi->Parent->Options_.OutputStream) {
                pi->Parent->Options_.OutputStream->Finish();
            }
        }

        delete pi;
        return true;
    }

    // interchange io while process is alive
    inline static void Communicate(TProcessInfo* pi);

    inline static void* WatchProcess(void* data) {
        TProcessInfo* pi = reinterpret_cast<TProcessInfo*>(data);
        Communicate(pi);
        return nullptr;
    }

    inline static void* ReadStream(void* data) noexcept {
        TPipePump* pump = reinterpret_cast<TPipePump*>(data);
        try {
            int bytes = 0;
            TBuffer buffer(DATA_BUFFER_SIZE);

            while (true) {
                bytes = pump->Pipe->Read(buffer.Data(), buffer.Capacity());
                if (bytes > 0) {
                    pump->OutputStream->Write(buffer.Data(), bytes);
                } else {
                    break;
                }
            }
            if (pump->Pipe->IsOpen()) {
                pump->Pipe->Close();
            }
        } catch (...) {
            pump->InternalError = CurrentExceptionMessage();
        }
        return nullptr;
    }

    inline static void* WriteStream(void* data) noexcept {
        TPipePump* pump = reinterpret_cast<TPipePump*>(data);
        try {
            int bytes = 0;
            int bytesToWrite = 0;
            char* bufPos = nullptr;
            TBuffer buffer(DATA_BUFFER_SIZE);

            while (true) {
                if (!bytesToWrite) {
                    bytesToWrite = pump->InputStream->Read(buffer.Data(), buffer.Capacity());
                    if (bytesToWrite == 0) {
                        if (pump->ShouldClosePipe->load(std::memory_order_acquire)) {
                            break;
                        }
                        continue;
                    }
                    bufPos = buffer.Data();
                }

                bytes = pump->Pipe->Write(bufPos, bytesToWrite);
                if (bytes > 0) {
                    bytesToWrite -= bytes;
                    bufPos += bytes;
                } else {
                    break;
                }
            }
            if (pump->Pipe->IsOpen()) {
                pump->Pipe->Close();
            }
        } catch (...) {
            pump->InternalError = CurrentExceptionMessage();
        }
        return nullptr;
    }

    TString GetQuotedCommand() const;
};

#if defined(_win_)
void TShellCommand::TImpl::StartProcess(TShellCommand::TImpl::TPipes& pipes) {
    // Setup STARTUPINFO to redirect handles.
    STARTUPINFOW startup_info;
    ZeroMemory(&startup_info, sizeof(startup_info));
    startup_info.cb = sizeof(startup_info);
    startup_info.dwFlags = STARTF_USESTDHANDLES;

    if (Options_.OutputMode != TShellCommandOptions::HANDLE_INHERIT) {
        if (!SetHandleInformation(pipes.OutputPipeFd[1], HANDLE_FLAG_INHERIT, HANDLE_FLAG_INHERIT)) {
            ythrow TSystemError() << "cannot set handle info";
        }
    }
    if (Options_.ErrorMode != TShellCommandOptions::HANDLE_INHERIT) {
        if (!SetHandleInformation(pipes.ErrorPipeFd[1], HANDLE_FLAG_INHERIT, HANDLE_FLAG_INHERIT)) {
            ythrow TSystemError() << "cannot set handle info";
        }
    }
    if (InputMode != TShellCommandOptions::HANDLE_INHERIT) {
        if (!SetHandleInformation(pipes.InputPipeFd[0], HANDLE_FLAG_INHERIT, HANDLE_FLAG_INHERIT)) {
            ythrow TSystemError() << "cannot set handle info";
        }
    }

    // A sockets do not work as std streams for some reason
    if (Options_.OutputMode != TShellCommandOptions::HANDLE_INHERIT) {
        startup_info.hStdOutput = pipes.OutputPipeFd[1];
    } else {
        startup_info.hStdOutput = GetStdHandle(STD_OUTPUT_HANDLE);
    }
    if (Options_.ErrorMode != TShellCommandOptions::HANDLE_INHERIT) {
        startup_info.hStdError = pipes.ErrorPipeFd[1];
    } else {
        startup_info.hStdError = GetStdHandle(STD_ERROR_HANDLE);
    }
    if (InputMode != TShellCommandOptions::HANDLE_INHERIT) {
        startup_info.hStdInput = pipes.InputPipeFd[0];
    } else {
        // Don't leave hStdInput unfilled, otherwise any attempt to retrieve the operating-system file handle
        // that is associated with the specified file descriptor will led to errors.
        startup_info.hStdInput = GetStdHandle(STD_INPUT_HANDLE);
    }

    PROCESS_INFORMATION process_info;
    // TString cmd = "cmd /U" + TUtf16String can be used to read unicode messages from cmd
    // /A - ansi charset /Q - echo off, /C - command, /Q - special quotes
    TString qcmd = GetQuotedCommand();
    TString cmd = Options_.UseShell ? "cmd /A /Q /S /C \"" + qcmd + "\"" : qcmd;
    // winapi can modify command text, copy it

    Y_ENSURE_EX(cmd.size() < MAX_COMMAND_LINE, yexception() << "Command is too long (length=" << cmd.size() << ")");
    TTempArray<wchar_t> cmdcopy(MAX_COMMAND_LINE);
    Copy(cmd.data(), cmd.data() + cmd.size(), cmdcopy.Data());
    *(cmdcopy.Data() + cmd.size()) = 0;

    const wchar_t* cwd = NULL;
    std::wstring cwdBuff;
    if (WorkDir.size()) {
        cwdBuff = GetWString(WorkDir.data());
        cwd = cwdBuff.c_str();
    }

    void* lpEnvironment = nullptr;
    TString env;
    if (!Options_.Environment.empty()) {
        for (auto e = Options_.Environment.begin(); e != Options_.Environment.end(); ++e) {
            env += e->first + '=' + e->second + '\0';
        }
        env += '\0';
        lpEnvironment = const_cast<char*>(env.data());
    }

    // disable messagebox (may be in debug too)
    #ifndef NDEBUG
    SetErrorMode(GetErrorMode() | SEM_NOGPFAULTERRORBOX);
    #endif
    BOOL res = 0;
    if (Options_.User.Name.empty() || GetUsername() == Options_.User.Name) {
        res = CreateProcessW(
            nullptr, // image name
            cmdcopy.Data(),
            nullptr,       // process security attributes
            nullptr,       // thread security attributes
            TRUE,          // inherit handles - needed for IO, CloseAllFdsOnExec not respected
            0,             // obscure creation flags
            lpEnvironment, // environment
            cwd,           // current directory
            &startup_info,
            &process_info);
    } else {
        res = CreateProcessWithLogonW(
            GetWString(Options_.User.Name.data()).c_str(),
            nullptr, // domain (if this parameter is NULL, the user name must be specified in UPN format)
            GetWString(Options_.User.Password.data()).c_str(),
            0,    // logon flags
            NULL, // image name
            cmdcopy.Data(),
            0,             // obscure creation flags
            lpEnvironment, // environment
            cwd,           // current directory
            &startup_info,
            &process_info);
    }

    if (!res) {
        ExecutionStatus.store(SHELL_ERROR, std::memory_order_release);
        /// @todo: write to error stream if set
        TStringOutput out(CollectedError);
        out << "Process was not created: " << LastSystemErrorText() << " command text was: '" << GetAString(cmdcopy.Data()) << "'";
    }
    Pid = process_info.hProcess;
    CloseHandle(process_info.hThread);
    DBG(Cerr << "created process id " << Pid << " in dir: " << cwd << ", cmd: " << cmdcopy.Data() << Endl);
}
#endif

void ShellQuoteArg(TString& dst, TStringBuf argument) {
    dst.append("\"");
    TStringBuf l, r;
    while (argument.TrySplit('"', l, r)) {
        dst.append(l);
        dst.append("\\\"");
        argument = r;
    }
    dst.append(argument);
    dst.append("\"");
}

void ShellQuoteArgSp(TString& dst, TStringBuf argument) {
    dst.append(' ');
    ShellQuoteArg(dst, argument);
}

bool ArgNeedsQuotes(TStringBuf arg) noexcept {
    if (arg.empty()) {
        return true;
    }
    return arg.find_first_of(" \"\'\t&()*<>\\`^|") != TString::npos;
}

TString TShellCommand::TImpl::GetQuotedCommand() const {
    TString quoted = Command; /// @todo command itself should be quoted too
    for (const auto& argument : Arguments) {
        // Don't add unnecessary quotes. It's especially important for the windows with a 32k command line length limit.
        if (Options_.QuoteArguments && ArgNeedsQuotes(argument)) {
            ::ShellQuoteArgSp(quoted, argument);
        } else {
            quoted.append(" ").append(argument);
        }
    }
    return quoted;
}

#if defined(_unix_)
void TShellCommand::TImpl::OnFork(TPipes& pipes, sigset_t oldmask, char* const* argv, char* const* envp, const std::function<void()>& afterFork) const {
    try {
        if (Options_.DetachSession) {
            setsid();
        }

        // reset signal handlers from parent
        struct sigaction sa;
        sa.sa_handler = SIG_DFL;
        sa.sa_flags = 0;
        SigEmptySet(&sa.sa_mask);
        for (int i = 0; i < NSIG; ++i) {
            // some signals cannot be caught, so just ignore return value
            sigaction(i, &sa, nullptr);
        }
        if (Options_.ClearSignalMask) {
            SigEmptySet(&oldmask);
        }
        // clear / restore signal mask
        if (SigProcMask(SIG_SETMASK, &oldmask, nullptr) != 0) {
            ythrow TSystemError() << "Cannot " << (Options_.ClearSignalMask ? "clear" : "restore") << " signal mask in child";
        }

        TFileHandle sIn(0);
        TFileHandle sOut(1);
        TFileHandle sErr(2);
        if (InputMode != TShellCommandOptions::HANDLE_INHERIT) {
            pipes.InputPipeFd[1].Close();
            TFileHandle sInNew(pipes.InputPipeFd[0]);
            sIn.LinkTo(sInNew);
            sIn.Release();
            sInNew.Release();
        } else {
            // do not close fd 0 - next open will return it and confuse all readers
            /// @todo in case of real need - reopen /dev/null
        }
        if (Options_.OutputMode != TShellCommandOptions::HANDLE_INHERIT) {
            pipes.OutputPipeFd[0].Close();
            TFileHandle sOutNew(pipes.OutputPipeFd[1]);
            sOut.LinkTo(sOutNew);
            sOut.Release();
            sOutNew.Release();
        }
        if (Options_.ErrorMode != TShellCommandOptions::HANDLE_INHERIT) {
            pipes.ErrorPipeFd[0].Close();
            TFileHandle sErrNew(pipes.ErrorPipeFd[1]);
            sErr.LinkTo(sErrNew);
            sErr.Release();
            sErrNew.Release();
        }

        if (WorkDir.size()) {
            NFs::SetCurrentWorkingDirectory(WorkDir);
        }

        if (Options_.CloseAllFdsOnExec) {
            for (int fd = NSystemInfo::MaxOpenFiles(); fd > STDERR_FILENO; --fd) {
                fcntl(fd, F_SETFD, FD_CLOEXEC);
            }
        }

        if (!Options_.User.Name.empty()) {
            ImpersonateUser(Options_.User);
        }

        if (Options_.Nice) {
            // Don't verify Nice() call - it does not work properly with WSL https://github.com/Microsoft/WSL/issues/1838
            ::Nice(Options_.Nice);
        }
        if (afterFork) {
            afterFork();
        }

        if (envp == nullptr) {
            execvp(argv[0], argv);
        } else {
            execve(argv[0], argv, envp);
        }
        Cerr << "Process was not created: " << LastSystemErrorText() << Endl;
    } catch (const std::exception& error) {
        Cerr << "Process was not created: " << error.what() << Endl;
    } catch (...) {
        Cerr << "Process was not created: "
             << "unknown error" << Endl;
    }

    _exit(-1);
}
#endif

void TShellCommand::TImpl::Run() {
    Y_ENSURE(ExecutionStatus.load(std::memory_order_acquire) != SHELL_RUNNING, TStringBuf("Process is already running"));
    // Prepare I/O streams
    CollectedOutput.clear();
    CollectedError.clear();
    TPipes pipes;

    if (Options_.OutputMode != TShellCommandOptions::HANDLE_INHERIT) {
        TRealPipeHandle::Pipe(pipes.OutputPipeFd[0], pipes.OutputPipeFd[1], CloseOnExec);
    }
    if (Options_.ErrorMode != TShellCommandOptions::HANDLE_INHERIT) {
        TRealPipeHandle::Pipe(pipes.ErrorPipeFd[0], pipes.ErrorPipeFd[1], CloseOnExec);
    }
    if (InputMode != TShellCommandOptions::HANDLE_INHERIT) {
        TRealPipeHandle::Pipe(pipes.InputPipeFd[0], pipes.InputPipeFd[1], CloseOnExec);
    }

    ExecutionStatus.store(SHELL_RUNNING, std::memory_order_release);

#if defined(_unix_)
    // block all signals to avoid signal handler race after fork()
    sigset_t oldmask, newmask;
    SigFillSet(&newmask);
    if (SigProcMask(SIG_SETMASK, &newmask, &oldmask) != 0) {
        ythrow TSystemError() << "Cannot block all signals in parent";
    }

    /* arguments holders */
    TString shellArg;
    TVector<char*> qargv;
    /*
      Following "const_cast"s are safe:
      http://pubs.opengroup.org/onlinepubs/9699919799/functions/exec.html
    */
    if (Options_.UseShell) {
        shellArg = GetQuotedCommand();
        qargv.reserve(4);
        qargv.push_back(const_cast<char*>("/bin/sh"));
        qargv.push_back(const_cast<char*>("-c"));
        // two args for 'sh -c -- ',
        // one for program name, and one for NULL at the end
        qargv.push_back(const_cast<char*>(shellArg.data()));
    } else {
        qargv.reserve(Arguments.size() + 2);
        qargv.push_back(const_cast<char*>(Command.data()));
        for (auto& i : Arguments) {
            qargv.push_back(const_cast<char*>(i.data()));
        }
    }

    qargv.push_back(nullptr);

    TVector<TString> envHolder;
    TVector<char*> envp;
    if (!Options_.Environment.empty()) {
        for (auto& env : Options_.Environment) {
            envHolder.emplace_back(env.first + '=' + env.second);
            envp.push_back(const_cast<char*>(envHolder.back().data()));
        }
        envp.push_back(nullptr);
    }

    pid_t pid = fork();
    if (pid == -1) {
        ExecutionStatus.store(SHELL_ERROR, std::memory_order_release);
        /// @todo check if pipes are still open
        ythrow TSystemError() << "Cannot fork";
    } else if (pid == 0) { // child
        if (envp.size() != 0) {
            OnFork(pipes, oldmask, qargv.data(), envp.data(), Options_.FuncAfterFork);
        } else {
            OnFork(pipes, oldmask, qargv.data(), nullptr, Options_.FuncAfterFork);
        }
    } else { // parent
        // restore signal mask
        if (SigProcMask(SIG_SETMASK, &oldmask, nullptr) != 0) {
            ythrow TSystemError() << "Cannot restore signal mask in parent";
        }
    }
    Pid = pid;
#else
    StartProcess(pipes);
#endif
    pipes.PrepareParents();

    if (ExecutionStatus.load(std::memory_order_acquire) != SHELL_RUNNING) {
        return;
    }

    if (InputMode == TShellCommandOptions::HANDLE_PIPE) {
        TFileHandle inputHandle(pipes.InputPipeFd[1].Release());
        InputHandle.Swap(inputHandle);
    }

    if (Options_.OutputMode == TShellCommandOptions::HANDLE_PIPE) {
        TFileHandle outputHandle(pipes.OutputPipeFd[0].Release());
        OutputHandle.Swap(outputHandle);
    }

    if (Options_.ErrorMode == TShellCommandOptions::HANDLE_PIPE) {
        TFileHandle errorHandle(pipes.ErrorPipeFd[0].Release());
        ErrorHandle.Swap(errorHandle);
    }

    TProcessInfo* processInfo = new TProcessInfo(this,
                                                 pipes.InputPipeFd[1].Release(), pipes.OutputPipeFd[0].Release(), pipes.ErrorPipeFd[0].Release());
    if (Options_.AsyncMode) {
        WatchThread = new TThread(&TImpl::WatchProcess, processInfo);
        WatchThread->Start();
        /// @todo wait for child to start its process session (if options.Detach)
    } else {
        Communicate(processInfo);
    }

    pipes.ReleaseParents(); // not needed
}

void TShellCommand::TImpl::Communicate(TProcessInfo* pi) {
    THolder<IOutputStream> outputHolder;
    IOutputStream* output = pi->Parent->Options_.OutputStream;
    if (!output) {
        outputHolder.Reset(output = new TStringOutput(pi->Parent->CollectedOutput));
    }

    THolder<IOutputStream> errorHolder;
    IOutputStream* error = pi->Parent->Options_.ErrorStream;
    if (!error) {
        errorHolder.Reset(error = new TStringOutput(pi->Parent->CollectedError));
    }

    IInputStream*& input = pi->Parent->Options_.InputStream;

#if defined(_unix_)
    // not really needed, io is done via poll
    if (pi->OutputFd.IsOpen()) {
        SetNonBlock(pi->OutputFd);
    }
    if (pi->ErrorFd.IsOpen()) {
        SetNonBlock(pi->ErrorFd);
    }
    if (pi->InputFd.IsOpen()) {
        SetNonBlock(pi->InputFd);
    }
#endif

    try {
#if defined(_win_)
        TPipePump pumps[3] = {0};
        pumps[0] = {&pi->ErrorFd, error};
        pumps[1] = {&pi->OutputFd, output};

        TVector<THolder<TThread>> streamThreads;
        streamThreads.emplace_back(new TThread(&TImpl::ReadStream, &pumps[0]));
        streamThreads.emplace_back(new TThread(&TImpl::ReadStream, &pumps[1]));

        if (input) {
            pumps[2] = {&pi->InputFd, nullptr, input, &pi->Parent->Options_.ShouldCloseInput};
            streamThreads.emplace_back(new TThread(&TImpl::WriteStream, &pumps[2]));
        }

        for (auto& threadHolder : streamThreads) {
            threadHolder->Start();
        }
#else
        TBuffer buffer(DATA_BUFFER_SIZE);
        TBuffer inputBuffer(DATA_BUFFER_SIZE);
        int bytes;
        int bytesToWrite = 0;
        char* bufPos = nullptr;
#endif
        TWaitResult waitPidResult;
        TExitStatus status = 0;

        while (true) {
            {
                with_lock (pi->Parent->TerminateMutex) {
                    if (TerminateIsRequired(pi)) {
                        return;
                    }
                }

                waitPidResult =
#if defined(_unix_)
                    waitpid(pi->Parent->Pid, &status, WNOHANG);
#else
                    WaitForSingleObject(pi->Parent->Pid /* process_info.hProcess */, pi->Parent->Options_.PollDelayMs /* ms */);
                Y_UNUSED(status);
#endif
                // DBG(Cerr << "wait result: " << waitPidResult << Endl);
                if (waitPidResult != WAIT_PROCEED) {
                    break;
                }
            }
/// @todo factor out (poll + wfmo)
#if defined(_unix_)
            bool haveIn = false;
            bool haveOut = false;
            bool haveErr = false;

            if (!input && pi->InputFd.IsOpen()) {
                DBG(Cerr << "closing input stream..." << Endl);
                pi->InputFd.Close();
            }
            if (!output && pi->OutputFd.IsOpen()) {
                DBG(Cerr << "closing output stream..." << Endl);
                pi->OutputFd.Close();
            }
            if (!error && pi->ErrorFd.IsOpen()) {
                DBG(Cerr << "closing error stream..." << Endl);
                pi->ErrorFd.Close();
            }

            if (!input && !output && !error) {
                continue;
            }

            struct pollfd fds[] = {
                {REALPIPEHANDLE(pi->InputFd), POLLOUT, 0},
                {REALPIPEHANDLE(pi->OutputFd), POLLIN, 0},
                {REALPIPEHANDLE(pi->ErrorFd), POLLIN, 0}};
            int res;

            if (!input) {
                fds[0].events = 0;
            }
            if (!output) {
                fds[1].events = 0;
            }
            if (!error) {
                fds[2].events = 0;
            }

            res = PollD(fds, 3, TInstant::Now() + TDuration::MilliSeconds(pi->Parent->Options_.PollDelayMs));
            // DBG(Cerr << "poll result: " << res << Endl);
            if (-res == ETIMEDOUT || res == 0) {
                // DBG(Cerr << "poll again..." << Endl);
                continue;
            }
            if (res < 0) {
                ythrow yexception() << "poll failed: " << LastSystemErrorText();
            }

            if ((fds[1].revents & POLLIN) == POLLIN) {
                haveOut = true;
            } else if (fds[1].revents & (POLLERR | POLLHUP)) {
                output = nullptr;
            }

            if ((fds[2].revents & POLLIN) == POLLIN) {
                haveErr = true;
            } else if (fds[2].revents & (POLLERR | POLLHUP)) {
                error = nullptr;
            }

            if (input && ((fds[0].revents & POLLOUT) == POLLOUT)) {
                haveIn = true;
            }

            if (haveOut) {
                bytes = pi->OutputFd.Read(buffer.Data(), buffer.Capacity());
                DBG(Cerr << "transferred " << bytes << " bytes of output" << Endl);
                if (bytes > 0) {
                    output->Write(buffer.Data(), bytes);
                } else {
                    output = nullptr;
                }
            }
            if (haveErr) {
                bytes = pi->ErrorFd.Read(buffer.Data(), buffer.Capacity());
                DBG(Cerr << "transferred " << bytes << " bytes of error" << Endl);
                if (bytes > 0) {
                    error->Write(buffer.Data(), bytes);
                } else {
                    error = nullptr;
                }
            }

            if (haveIn) {
                if (!bytesToWrite) {
                    bytesToWrite = input->Read(inputBuffer.Data(), inputBuffer.Capacity());
                    if (bytesToWrite == 0) {
                        if (pi->Parent->Options_.ShouldCloseInput.load(std::memory_order_acquire)) {
                            input = nullptr;
                        }
                        continue;
                    }
                    bufPos = inputBuffer.Data();
                }

                bytes = pi->InputFd.Write(bufPos, bytesToWrite);
                if (bytes > 0) {
                    bytesToWrite -= bytes;
                    bufPos += bytes;
                } else {
                    input = nullptr;
                }

                DBG(Cerr << "transferred " << bytes << " bytes of input" << Endl);
            }
#endif
        }
        DBG(Cerr << "process finished" << Endl);

        // What's the reason of process exit.
        // We need to set exit code before waiting for input thread
        // Otherwise there is no way for input stream provider to discover
        // that process has exited and stream shouldn't wait for new data.
        bool cleanExit = false;
        TMaybe<int> processExitCode;
#if defined(_unix_)
        processExitCode = WEXITSTATUS(status);
        if (WIFEXITED(status) && processExitCode == 0) {
            cleanExit = true;
        } else if (WIFSIGNALED(status)) {
            processExitCode = -WTERMSIG(status);
        }
#else
        if (waitPidResult == WAIT_OBJECT_0) {
            DWORD exitCode = STILL_ACTIVE;
            if (!GetExitCodeProcess(pi->Parent->Pid, &exitCode)) {
                ythrow yexception() << "GetExitCodeProcess: " << LastSystemErrorText();
            }
            if (exitCode == 0) {
                cleanExit = true;
            }
            processExitCode = static_cast<int>(exitCode);
            DBG(Cerr << "exit code: " << exitCode << Endl);
        }
#endif
        pi->Parent->ExitCode = processExitCode;
        if (cleanExit) {
            pi->Parent->ExecutionStatus.store(SHELL_FINISHED, std::memory_order_release);
        } else {
            pi->Parent->ExecutionStatus.store(SHELL_ERROR, std::memory_order_release);
        }

#if defined(_win_)
        for (auto& threadHolder : streamThreads) {
            threadHolder->Join();
        }
        for (const auto pump : pumps) {
            if (!pump.InternalError.empty()) {
                throw yexception() << pump.InternalError;
            }
        }
#else
        // Now let's read remaining stdout/stderr
        while (output && (bytes = pi->OutputFd.Read(buffer.Data(), buffer.Capacity())) > 0) {
            DBG(Cerr << bytes << " more bytes of output: " << Endl);
            output->Write(buffer.Data(), bytes);
        }
        while (error && (bytes = pi->ErrorFd.Read(buffer.Data(), buffer.Capacity())) > 0) {
            DBG(Cerr << bytes << " more bytes of error" << Endl);
            error->Write(buffer.Data(), bytes);
        }
#endif
    } catch (const yexception& e) {
        // Some error in watch occured, set result to error
        pi->Parent->ExecutionStatus.store(SHELL_INTERNAL_ERROR, std::memory_order_release);
        pi->Parent->InternalError = e.what();
        if (input) {
            pi->InputFd.Close();
        }
        Cdbg << "shell command internal error: " << pi->Parent->InternalError << Endl;
    }
    // Now we can safely delete process info struct and other data
    pi->Parent->TerminateFlag = true;
    TerminateIsRequired(pi);
}

TShellCommand::TShellCommand(const TStringBuf cmd, const TList<TString>& args, const TShellCommandOptions& options,
                             const TString& workdir)
    : Impl(new TImpl(cmd, args, options, workdir))
{
}

TShellCommand::TShellCommand(const TStringBuf cmd, const TShellCommandOptions& options, const TString& workdir)
    : Impl(new TImpl(cmd, TList<TString>(), options, workdir))
{
}

TShellCommand::~TShellCommand() = default;

TShellCommand& TShellCommand::operator<<(const TStringBuf argument) {
    Impl->AppendArgument(argument);
    return *this;
}

const TString& TShellCommand::GetOutput() const {
    return Impl->GetOutput();
}

const TString& TShellCommand::GetError() const {
    return Impl->GetError();
}

const TString& TShellCommand::GetInternalError() const {
    return Impl->GetInternalError();
}

TShellCommand::ECommandStatus TShellCommand::GetStatus() const {
    return Impl->GetStatus();
}

TMaybe<int> TShellCommand::GetExitCode() const {
    return Impl->GetExitCode();
}

TProcessId TShellCommand::GetPid() const {
    return Impl->GetPid();
}

TFileHandle& TShellCommand::GetInputHandle() {
    return Impl->GetInputHandle();
}

TFileHandle& TShellCommand::GetOutputHandle() {
    return Impl->GetOutputHandle();
}

TFileHandle& TShellCommand::GetErrorHandle() {
    return Impl->GetErrorHandle();
}

TShellCommand& TShellCommand::Run() {
    Impl->Run();
    return *this;
}

TShellCommand& TShellCommand::Terminate(int signal) {
    Impl->Terminate(signal);
    return *this;
}

TShellCommand& TShellCommand::Wait() {
    Impl->Wait();
    return *this;
}

TShellCommand& TShellCommand::CloseInput() {
    Impl->CloseInput();
    return *this;
}

TString TShellCommand::GetQuotedCommand() const {
    return Impl->GetQuotedCommand();
}
