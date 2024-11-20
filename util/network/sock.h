#pragma once

#include <util/folder/path.h>
#include <util/system/defaults.h>
#include <util/string/cast.h>
#include <util/stream/output.h>
#include <util/system/sysstat.h>

#if defined(_win_) || defined(_cygwin_)
    #include <util/system/file.h>
#else
    #include <sys/un.h>
    #include <sys/stat.h>
#endif //_win_

#include "init.h"
#include "ip.h"
#include "socket.h"

constexpr ui16 DEF_LOCAL_SOCK_MODE = 00644;

// Base abstract class for socket address
struct ISockAddr {
    virtual ~ISockAddr() = default;
    // Max size of the address that we can store (arg of recvfrom)
    virtual socklen_t Size() const = 0;
    // Real length of the address (arg of sendto)
    virtual socklen_t Len() const = 0;
    // cast to sockaddr* to pass to any syscall
    virtual sockaddr* SockAddr() = 0;
    virtual const sockaddr* SockAddr() const = 0;
    // address in human readable form
    virtual TString ToString() const = 0;

protected:
    // below are the implemetation methods that can be called by T*Socket classes
    friend class TBaseSocket;
    friend class TDgramSocket;
    friend class TStreamSocket;

    virtual int ResolveAddr() const {
        // usually it's nothing to do here
        return 0;
    }
    virtual int Bind(SOCKET s, ui16 mode) const = 0;
};

#if defined(_win_) || defined(_cygwin_)
    #define YAF_LOCAL AF_INET
struct TSockAddrLocal: public ISockAddr {
    TSockAddrLocal() {
        Clear();
    }

    TSockAddrLocal(const char* path) {
        Set(path);
    }

    socklen_t Size() const {
        return sizeof(sockaddr_in);
    }

    socklen_t Len() const {
        return Size();
    }

    inline void Clear() noexcept {
        Zero(in);
        Zero(Path);
    }

    inline void Set(const char* path) noexcept {
        Clear();
        in.sin_family = AF_INET;
        in.sin_addr.s_addr = IpFromString("127.0.0.1");
        in.sin_port = 0;
        strlcpy(Path, path, PathSize);
    }

    inline void Set(TStringBuf path) noexcept {
        Clear();
        in.sin_family = AF_INET;
        in.sin_addr.s_addr = IpFromString("127.0.0.1");
        in.sin_port = 0;
        strlcpy(Path, path.data(), Min(PathSize, path.size() + 1));
    }

    sockaddr* SockAddr() {
        return (struct sockaddr*)(&in);
    }

    const sockaddr* SockAddr() const {
        return (const struct sockaddr*)(&in);
    }

    TString ToString() const {
        return TString(Path);
    }

    TFsPath ToPath() const {
        return TFsPath(Path);
    }

    int ResolveAddr() const {
        if (in.sin_port == 0) {
            int ret = 0;
            // 1. open file
            TFileHandle f(Path, OpenExisting | RdOnly);
            if (!f.IsOpen()) {
                return -errno;
            }

            // 2. read the port from file
            ret = f.Read(&in.sin_port, sizeof(in.sin_port));
            if (ret != sizeof(in.sin_port)) {
                return -(errno ? errno : EFAULT);
            }
        }

        return 0;
    }

    int Bind(SOCKET s, ui16 mode) const {
        Y_UNUSED(mode);
        int ret = 0;
        // 1. open file
        TFileHandle f(Path, CreateAlways | WrOnly);
        if (!f.IsOpen()) {
            return -errno;
        }

        // 2. find port and bind to it
        in.sin_port = 0;
        ret = bind(s, SockAddr(), Len());
        if (ret != 0) {
            return -WSAGetLastError();
        }

        int size = Size();
        ret = getsockname(s, (struct sockaddr*)(&in), &size);
        if (ret != 0) {
            return -WSAGetLastError();
        }

        // 3. write port to file
        ret = f.Write(&(in.sin_port), sizeof(in.sin_port));
        if (ret != sizeof(in.sin_port)) {
            return -errno;
        }

        return 0;
    }

    static constexpr size_t PathSize = 128;
    mutable struct sockaddr_in in;
    char Path[PathSize];
};
#else
    #define YAF_LOCAL AF_LOCAL
struct TSockAddrLocal: public sockaddr_un, public ISockAddr {
    TSockAddrLocal() {
        Clear();
    }

    TSockAddrLocal(TStringBuf path) {
        Set(path);
    }

    TSockAddrLocal(const char* path) {
        Set(path);
    }

    socklen_t Size() const override {
        return sizeof(sockaddr_un);
    }

    socklen_t Len() const override {
        return strlen(sun_path) + 2;
    }

    inline void Clear() noexcept {
        Zero(*(sockaddr_un*)this);
    }

    inline void Set(const char* path) noexcept {
        Clear();
        sun_family = AF_UNIX;
        strlcpy(sun_path, path, sizeof(sun_path));
    }

    inline void Set(TStringBuf path) noexcept {
        Clear();
        sun_family = AF_UNIX;
        strlcpy(sun_path, path.data(), Min(sizeof(sun_path), path.size() + 1));
    }

    sockaddr* SockAddr() override {
        return (struct sockaddr*)(struct sockaddr_un*)this;
    }

    const sockaddr* SockAddr() const override {
        return (const struct sockaddr*)(const struct sockaddr_un*)this;
    }

    TString ToString() const override {
        return TString(sun_path);
    }

    TFsPath ToPath() const {
        return TFsPath(sun_path);
    }

    int Bind(SOCKET s, ui16 mode) const override {
        (void)unlink(sun_path);

        int ret = bind(s, SockAddr(), Len());
        if (ret < 0) {
            return -errno;
        }

        ret = Chmod(sun_path, mode);
        if (ret < 0) {
            return -errno;
        }
        return 0;
    }
};
#endif // _win_

struct TSockAddrInet: public sockaddr_in, public ISockAddr {
    TSockAddrInet() {
        Clear();
    }

    TSockAddrInet(TIpHost ip, TIpPort port) {
        Set(ip, port);
    }

    TSockAddrInet(const char* ip, TIpPort port) {
        Set(IpFromString(ip), port);
    }

    socklen_t Size() const override {
        return sizeof(sockaddr_in);
    }

    socklen_t Len() const override {
        return Size();
    }

    inline void Clear() noexcept {
        Zero(*(sockaddr_in*)this);
    }

    inline void Set(TIpHost ip, TIpPort port) noexcept {
        Clear();
        sin_family = AF_INET;
        sin_addr.s_addr = ip;
        sin_port = HostToInet(port);
    }

    sockaddr* SockAddr() override {
        return (struct sockaddr*)(struct sockaddr_in*)this;
    }

    const sockaddr* SockAddr() const override {
        return (const struct sockaddr*)(const struct sockaddr_in*)this;
    }

    TString ToString() const override {
        return IpToString(sin_addr.s_addr) + ":" + ::ToString(InetToHost(sin_port));
    }

    int Bind(SOCKET s, ui16 mode) const override {
        Y_UNUSED(mode);
        int ret = bind(s, SockAddr(), Len());
        if (ret < 0) {
            return -errno;
        }

        socklen_t len = Len();
        if (getsockname(s, (struct sockaddr*)(SockAddr()), &len) < 0) {
            return -WSAGetLastError();
        }

        return 0;
    }

    TIpHost GetIp() const noexcept {
        return sin_addr.s_addr;
    }

    TIpPort GetPort() const noexcept {
        return InetToHost(sin_port);
    }

    void SetPort(TIpPort port) noexcept {
        sin_port = HostToInet(port);
    }
};

struct TSockAddrInet6: public sockaddr_in6, public ISockAddr {
    TSockAddrInet6() {
        Clear();
    }

    TSockAddrInet6(const char* ip6, const TIpPort port) {
        Set(ip6, port);
    }

    socklen_t Size() const override {
        return sizeof(sockaddr_in6);
    }

    socklen_t Len() const override {
        return Size();
    }

    inline void Clear() noexcept {
        Zero(*(sockaddr_in6*)this);
    }

    inline void Set(const char* ip6, const TIpPort port) noexcept {
        Clear();
        sin6_family = AF_INET6;
        inet_pton(AF_INET6, ip6, &sin6_addr);
        sin6_port = HostToInet(port);
    }

    sockaddr* SockAddr() override {
        return (struct sockaddr*)(struct sockaddr_in6*)this;
    }

    const sockaddr* SockAddr() const override {
        return (const struct sockaddr*)(const struct sockaddr_in6*)this;
    }

    TString ToString() const override {
        return "[" + GetIp() + "]:" + ::ToString(InetToHost(sin6_port));
    }

    int Bind(SOCKET s, ui16 mode) const override {
        Y_UNUSED(mode);
        int ret = bind(s, SockAddr(), Len());
        if (ret < 0) {
            return -errno;
        }
        socklen_t len = Len();
        if (getsockname(s, (struct sockaddr*)(SockAddr()), &len) < 0) {
            return -WSAGetLastError();
        }
        return 0;
    }

    TString GetIp() const noexcept {
        char ip6[INET6_ADDRSTRLEN];
        inet_ntop(AF_INET6, (void*)&sin6_addr, ip6, INET6_ADDRSTRLEN);
        return TString(ip6);
    }

    TIpPort GetPort() const noexcept {
        return InetToHost(sin6_port);
    }

    void SetPort(TIpPort port) noexcept {
        sin6_port = HostToInet(port);
    }
};

using TSockAddrLocalStream = TSockAddrLocal;
using TSockAddrLocalDgram = TSockAddrLocal;
using TSockAddrInetStream = TSockAddrInet;
using TSockAddrInetDgram = TSockAddrInet;
using TSockAddrInet6Stream = TSockAddrInet6;
using TSockAddrInet6Dgram = TSockAddrInet6;

class TBaseSocket: public TSocketHolder {
protected:
    TBaseSocket(SOCKET fd)
        : TSocketHolder(fd)
    {
    }

public:
    int Bind(const ISockAddr* addr, ui16 mode = DEF_LOCAL_SOCK_MODE) {
        return addr->Bind((SOCKET) * this, mode);
    }

    void CheckSock() {
        if ((SOCKET) * this == INVALID_SOCKET) {
            ythrow TSystemError() << "no socket";
        }
    }

    static ssize_t Check(ssize_t ret, const char* op = "") {
        if (ret < 0) {
            ythrow TSystemError(-(int)ret) << "socket operation " << op;
        }
        return ret;
    }
};

class TDgramSocket: public TBaseSocket {
protected:
    TDgramSocket(SOCKET fd)
        : TBaseSocket(fd)
    {
    }

public:
    ssize_t SendTo(const void* msg, size_t len, const ISockAddr* toAddr) {
        ssize_t ret = toAddr->ResolveAddr();
        if (ret < 0) {
            return -LastSystemError();
        }

        ret = sendto((SOCKET) * this, (const char*)msg, (int)len, 0, toAddr->SockAddr(), toAddr->Len());
        if (ret < 0) {
            return -LastSystemError();
        }

        return ret;
    }

    ssize_t RecvFrom(void* buf, size_t len, ISockAddr* fromAddr) {
        socklen_t fromSize = fromAddr->Size();
        const ssize_t ret = recvfrom((SOCKET) * this, (char*)buf, (int)len, 0, fromAddr->SockAddr(), &fromSize);
        if (ret < 0) {
            return -LastSystemError();
        }

        return ret;
    }
};

class TStreamSocket: public TBaseSocket {
protected:
    explicit TStreamSocket(SOCKET fd)
        : TBaseSocket(fd)
    {
    }

public:
    TStreamSocket()
        : TBaseSocket(INVALID_SOCKET)
    {
    }

    ssize_t Send(const void* msg, size_t len, int flags = 0) {
        const ssize_t ret = send((SOCKET) * this, (const char*)msg, (int)len, flags);
        if (ret < 0) {
            return -errno;
        }

        return ret;
    }

    ssize_t Recv(void* buf, size_t len, int flags = 0) {
        const ssize_t ret = recv((SOCKET) * this, (char*)buf, (int)len, flags);
        if (ret < 0) {
            return -errno;
        }

        return ret;
    }

    int Connect(const ISockAddr* addr) {
        int ret = addr->ResolveAddr();
        if (ret < 0) {
            return -errno;
        }

        ret = connect((SOCKET) * this, addr->SockAddr(), addr->Len());
        if (ret < 0) {
            return -errno;
        }

        return ret;
    }

    int Listen(int backlog) {
        int ret = listen((SOCKET) * this, backlog);
        if (ret < 0) {
            return -errno;
        }

        return ret;
    }

    int Accept(TStreamSocket* acceptedSock, ISockAddr* acceptedAddr = nullptr) {
        SOCKET s = INVALID_SOCKET;
        if (acceptedAddr) {
            socklen_t acceptedSize = acceptedAddr->Size();
            s = accept((SOCKET) * this, acceptedAddr->SockAddr(), &acceptedSize);
        } else {
            s = accept((SOCKET) * this, nullptr, nullptr);
        }

        if (s == INVALID_SOCKET) {
            return -errno;
        }

        TSocketHolder sock(s);
        acceptedSock->Swap(sock);
        return 0;
    }
};

class TLocalDgramSocket: public TDgramSocket {
public:
    TLocalDgramSocket(SOCKET fd)
        : TDgramSocket(fd)
    {
    }

    TLocalDgramSocket()
        : TDgramSocket(socket(YAF_LOCAL, SOCK_DGRAM, 0))
    {
    }
};

class TInetDgramSocket: public TDgramSocket {
public:
    TInetDgramSocket(SOCKET fd)
        : TDgramSocket(fd)
    {
    }

    TInetDgramSocket()
        : TDgramSocket(socket(AF_INET, SOCK_DGRAM, 0))
    {
    }
};

class TInet6DgramSocket: public TDgramSocket {
public:
    TInet6DgramSocket(SOCKET fd)
        : TDgramSocket(fd)
    {
    }

    TInet6DgramSocket()
        : TDgramSocket(socket(AF_INET6, SOCK_DGRAM, 0))
    {
    }
};

class TLocalStreamSocket: public TStreamSocket {
public:
    TLocalStreamSocket(SOCKET fd)
        : TStreamSocket(fd)
    {
    }

    TLocalStreamSocket()
        : TStreamSocket(socket(YAF_LOCAL, SOCK_STREAM, 0))
    {
    }
};

class TInetStreamSocket: public TStreamSocket {
public:
    TInetStreamSocket(SOCKET fd)
        : TStreamSocket(fd)
    {
    }

    TInetStreamSocket()
        : TStreamSocket(socket(AF_INET, SOCK_STREAM, 0))
    {
    }
};

class TInet6StreamSocket: public TStreamSocket {
public:
    TInet6StreamSocket(SOCKET fd)
        : TStreamSocket(fd)
    {
    }

    TInet6StreamSocket()
        : TStreamSocket(socket(AF_INET6, SOCK_STREAM, 0))
    {
    }
};

class TStreamSocketInput: public IInputStream {
public:
    TStreamSocketInput(TStreamSocket* socket)
        : Socket(socket)
    {
    }
    void SetSocket(TStreamSocket* socket) {
        Socket = socket;
    }

protected:
    TStreamSocket* Socket;

    size_t DoRead(void* buf, size_t len) override {
        Y_ABORT_UNLESS(Socket, "TStreamSocketInput: socket isn't set");
        const ssize_t ret = Socket->Recv(buf, len);

        if (ret >= 0) {
            return (size_t)ret;
        }

        ythrow TSystemError(-(int)ret) << "can not read from socket input stream";
    }
};

class TStreamSocketOutput: public IOutputStream {
public:
    TStreamSocketOutput(TStreamSocket* socket)
        : Socket(socket)
    {
    }
    void SetSocket(TStreamSocket* socket) {
        Socket = socket;
    }

    TStreamSocketOutput(TStreamSocketOutput&&) noexcept = default;
    TStreamSocketOutput& operator=(TStreamSocketOutput&&) noexcept = default;

protected:
    TStreamSocket* Socket;

    void DoWrite(const void* buf, size_t len) override {
        Y_ABORT_UNLESS(Socket, "TStreamSocketOutput: socket isn't set");

        const char* ptr = (const char*)buf;
        while (len) {
            const ssize_t ret = Socket->Send(ptr, len);

            if (ret < 0) {
                ythrow TSystemError(-(int)ret) << "can not write to socket output stream";
            }

            Y_ASSERT((size_t)ret <= len);
            len -= (size_t)ret;
            ptr += (size_t)ret;
        }
    }
};
