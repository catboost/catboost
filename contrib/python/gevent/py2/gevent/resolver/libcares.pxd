cdef extern from "ares.h":
    # These two are defined in <sys/socket.h> and <netdb.h>, respectively,
    # on POSIX. On Windows, they are in <winsock2.h>. "ares.h" winds up
    # indirectly including both of those.
    struct sockaddr:
        pass
    struct hostent:
        pass

    # Errors from getaddrinfo
    int EAI_ADDRFAMILY # The specified network host does not have
                       # any network addresses in the requested address family (Linux)

    int EAI_AGAIN     # temporary failure in name resolution
    int EAI_BADFLAGS  # invalid value for ai_flags
    int EAI_BADHINTS  # invalid value for hints (macOS only)
    int EAI_FAIL      # non-recoverable failure in name resolution
    int EAI_FAMILY    # ai_family not supported
    int EAI_MEMORY    # memory allocation failure
    int EAI_NODATA    # The specified network host exists, but does not have
                      # any network addresses defined. (Linux)
    int EAI_NONAME    # hostname or servname not provided, or not known
    int EAI_OVERFLOW  # argument buffer overflow (macOS only)
    int EAI_PROTOCOL  # resolved protocol is unknown (macOS only)
    int EAI_SERVICE   # servname not supported for ai_socktype
    int EAI_SOCKTYPE  # ai_socktype not supported
    int EAI_SYSTEM    # system error returned in errno (macOS and Linux)

    char* gai_strerror(int ecode)

    # Errors from gethostbyname and gethostbyaddr
    int HOST_NOT_FOUND
    int TRY_AGAIN
    int NO_RECOVERY
    int NO_DATA
    char* hstrerror(int err)

    struct ares_options:
        int flags
        void* sock_state_cb
        void* sock_state_cb_data
        int timeout
        int tries
        int ndots
        unsigned short udp_port
        unsigned short tcp_port
        char **domains
        int ndomains
        char* lookups

    int ARES_OPT_FLAGS
    int ARES_OPT_SOCK_STATE_CB
    int ARES_OPT_TIMEOUTMS
    int ARES_OPT_TRIES
    int ARES_OPT_NDOTS
    int ARES_OPT_TCP_PORT
    int ARES_OPT_UDP_PORT
    int ARES_OPT_SERVERS
    int ARES_OPT_DOMAINS
    int ARES_OPT_LOOKUPS

    int ARES_FLAG_USEVC
    int ARES_FLAG_PRIMARY
    int ARES_FLAG_IGNTC
    int ARES_FLAG_NORECURSE
    int ARES_FLAG_STAYOPEN
    int ARES_FLAG_NOSEARCH
    int ARES_FLAG_NOALIASES
    int ARES_FLAG_NOCHECKRESP

    int ARES_LIB_INIT_ALL
    int ARES_SOCKET_BAD

    int ARES_SUCCESS
    int ARES_EADDRGETNETWORKPARAMS
    int ARES_EBADFAMILY
    int ARES_EBADFLAGS
    int ARES_EBADHINTS
    int ARES_EBADNAME
    int ARES_EBADQUERY
    int ARES_EBADRESP
    int ARES_EBADSTR
    int ARES_ECANCELLED
    int ARES_ECONNREFUSED
    int ARES_EDESTRUCTION
    int ARES_EFILE
    int ARES_EFORMERR
    int ARES_ELOADIPHLPAPI
    int ARES_ENODATA
    int ARES_ENOMEM
    int ARES_ENONAME
    int ARES_ENOTFOUND
    int ARES_ENOTIMP
    int ARES_ENOTINITIALIZED
    int ARES_EOF
    int ARES_EREFUSED
    int ARES_ESERVFAIL
    int ARES_ESERVICE
    int ARES_ETIMEOUT

    int ARES_NI_NOFQDN
    int ARES_NI_NUMERICHOST
    int ARES_NI_NAMEREQD
    int ARES_NI_NUMERICSERV
    int ARES_NI_DGRAM
    int ARES_NI_TCP
    int ARES_NI_UDP
    int ARES_NI_SCTP
    int ARES_NI_DCCP
    int ARES_NI_NUMERICSCOPE
    int ARES_NI_LOOKUPHOST
    int ARES_NI_LOOKUPSERVICE

    ctypedef int ares_socklen_t

    int ares_library_init(int flags)
    void ares_library_cleanup()
    int ares_init_options(void *channelptr, ares_options *options, int)
    int ares_init(void *channelptr)
    void ares_destroy(void *channelptr)
    void ares_gethostbyname(void* channel, char *name, int family, void* callback, void *arg)
    void ares_gethostbyaddr(void* channel, void *addr, int addrlen, int family, void* callback, void *arg)
    void ares_process_fd(void* channel, int read_fd, int write_fd)
    char* ares_strerror(int code)
    void ares_cancel(void* channel)
    void ares_getnameinfo(void* channel, void* sa, int salen, int flags, void* callback, void *arg)

    # Added in 1.10
    int ares_inet_pton(int af, const char *src, void *dst)
    const char* ares_inet_ntop(int af, const void *src, char *dst, ares_socklen_t size);


    struct in_addr:
        pass

    struct ares_in6_addr:
        pass

    struct addr_union:
        in_addr addr4
        ares_in6_addr addr6

    struct ares_addr_node:
        ares_addr_node *next
        int family
        addr_union addr

    int ares_set_servers(void* channel, ares_addr_node *servers)

    # Added in 1.16
    int ARES_AI_NOSORT
    int ARES_AI_ENVHOSTS
    int ARES_AI_CANONNAME
    int ARES_AI_NUMERICSERV

    struct ares_addrinfo_hints:
        int ai_flags
        int ai_family
        int ai_socktype
        int ai_protocol

    struct ares_addrinfo_node:
        int ai_ttl
        int ai_flags
        int ai_family
        int ai_socktype
        int ai_protocol
        ares_socklen_t ai_addrlen
        sockaddr *ai_addr
        ares_addrinfo_node *ai_next

    struct ares_addrinfo_cname:
        int ttl
        char *alias
        char *name
        ares_addrinfo_cname *next

    struct ares_addrinfo:
        ares_addrinfo_cname *cnames
        ares_addrinfo_node  *nodes

    void ares_getaddrinfo(
        void* channel,
        const char *name,
        const char* service,
        const ares_addrinfo_hints *hints,
        #ares_addrinfo_callback callback,
        void* callback,
        void *arg)

    void ares_freeaddrinfo(ares_addrinfo *ai)
