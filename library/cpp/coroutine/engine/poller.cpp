#include "poller.h"
#include "sockmap.h"

#include <util/memory/smallobj.h>
#include <util/generic/intrlist.h>
#include <util/generic/singleton.h>
#include <util/system/env.h>
#include <util/string/cast.h>

namespace {
    using TChange = IPollerFace::TChange;
    using TEvent = IPollerFace::TEvent;
    using TEvents = IPollerFace::TEvents;

    template <class T>
    class TUnsafeBuf {
    public:
        TUnsafeBuf() noexcept
            : L_(0)
        {
        }

        T* operator~() const noexcept {
            return B_.Get();
        }

        size_t operator+() const noexcept {
            return L_;
        }

        void Reserve(size_t len) {
            len = FastClp2(len);

            if (len > L_) {
                B_.Reset(new T[len]);
                L_ = len;
            }
        }

    private:
        TArrayHolder<T> B_;
        size_t L_;
    };


    template <class T>
    class TVirtualize: public IPollerFace {
    public:
        TVirtualize(EContPoller pollerEngine)
            : PollerEngine_(pollerEngine)
        {
        }

        void Set(const TChange& c) override {
            P_.Set(c);
        }

        void Wait(TEvents& events, TInstant deadLine) override {
            P_.Wait(events, deadLine);
        }

        EContPoller PollEngine() const override {
            return PollerEngine_;
        }
    private:
        T P_;
        const EContPoller PollerEngine_;
    };


    template <class T>
    class TPoller {
        using TInternalEvent = typename T::TEvent;

    public:
        TPoller() {
            E_.Reserve(1);
        }

        void Set(const TChange& c) {
            P_.Set(c.Data, c.Fd, c.Flags);
        }

        void Reserve(size_t size) {
            E_.Reserve(size);
        }

        void Wait(TEvents& events, TInstant deadLine) {
            const size_t ret = P_.WaitD(~E_, +E_, deadLine);

            events.reserve(ret);

            for (size_t i = 0; i < ret; ++i) {
                const TInternalEvent* ie = ~E_ + i;

                const TEvent e = {
                    T::ExtractEvent(ie),
                    T::ExtractStatus(ie),
                    (ui16)T::ExtractFilter(ie),
                };

                events.push_back(e);
            }

            E_.Reserve(ret + 1);
        }

    private:
        T P_;
        TUnsafeBuf<TInternalEvent> E_;
    };


    template <class T>
    class TIndexedArray {
        struct TVal:
            public T,
            public TIntrusiveListItem<TVal>,
            public TObjectFromPool<TVal>
        {
            // NOTE Constructor must be user-defined (and not =default) here
            // because TVal objects are created in the UB-capable placement
            // TObjectFromPool::new operator that stores data in a memory
            // allocated for the object. Without user defined constructor
            // zero-initialization takes place in TVal() expression and the
            // data is overwritten.
            TVal() {
            }
        };

        typedef TIntrusiveList<TVal> TListType;

    public:
        typedef typename TListType::TIterator TIterator;
        typedef typename TListType::TConstIterator TConstIterator;

        TIndexedArray()
            : P_(TMemoryPool::TExpGrow::Instance(), TDefaultAllocator::Instance())
        {
        }

        TIterator Begin() noexcept {
            return I_.Begin();
        }

        TIterator End() noexcept {
            return I_.End();
        }

        TConstIterator Begin() const noexcept {
            return I_.Begin();
        }

        TConstIterator End() const noexcept {
            return I_.End();
        }

        T& operator[](size_t i) {
            return *Get(i);
        }

        T* Get(size_t i) {
            TValRef& v = V_.Get(i);

            if (Y_UNLIKELY(!v)) {
                v.Reset(new (&P_) TVal());
                I_.PushFront(v.Get());
            }

            Y_PREFETCH_WRITE(v.Get(), 1);

            return v.Get();
        }

        void Erase(size_t i) noexcept {
            V_.Get(i).Destroy();
        }

        size_t Size() const noexcept {
            return I_.Size();
        }

    private:
        using TValRef = THolder<TVal>;
        typename TVal::TPool P_;
        TSocketMap<TValRef> V_;
        TListType I_;
    };


    inline short PollFlags(ui16 flags) noexcept {
        short ret = 0;

        if (flags & CONT_POLL_READ) {
            ret |= POLLIN;
        }

        if (flags & CONT_POLL_WRITE) {
            ret |= POLLOUT;
        }

#if defined(_linux_)
        if (flags & CONT_POLL_RDHUP) {
            ret |= POLLRDHUP;
        }
#endif

        return ret;
    }


    class TPollPoller {
    public:
        size_t Size() const noexcept {
            return S_.Size();
        }

        template <class T>
        void Build(T& t) const {
            for (TFds::TConstIterator it = S_.Begin(); it != S_.End(); ++it) {
                t.Set(*it);
            }

            t.Reserve(Size());
        }

        void Set(const TChange& c) {
            if (c.Flags) {
                S_[c.Fd] = c;
            } else {
                S_.Erase(c.Fd);
            }
        }

        void Wait(TEvents& events, TInstant deadLine) {
            T_.clear();
            T_.reserve(Size());

            for (TFds::TConstIterator it = S_.Begin(); it != S_.End(); ++it) {
                const pollfd pfd = {
                    it->Fd,
                    PollFlags(it->Flags),
                    0,
                };

                T_.push_back(pfd);
            }

            const ssize_t ret = PollD(T_.data(), (nfds_t) T_.size(), deadLine);

            if (ret <= 0) {
                return;
            }

            events.reserve(T_.size());

            for (size_t i = 0; i < T_.size(); ++i) {
                const pollfd& pfd = T_[i];
                const short ev = pfd.revents;

                if (!ev) {
                    continue;
                }

                int status = 0;
                ui16 filter = 0;

                // We are perfectly fine with an EOF while reading a pipe or a unix socket
                if ((ev & POLLIN) || (ev & POLLHUP) && (pfd.events & POLLIN)) {
                    filter |= CONT_POLL_READ;
                }

                if (ev & POLLOUT) {
                    filter |= CONT_POLL_WRITE;
                }

#if defined(_linux_)
                if (ev & POLLRDHUP) {
                    filter |= CONT_POLL_RDHUP;
                }
#endif

                if (ev & POLLERR) {
                    status = EIO;
                } else if (ev & POLLHUP && pfd.events & POLLOUT) {
                    // Only write operations may cause EPIPE
                    status = EPIPE;
                } else if (ev & POLLNVAL) {
                    status = EINVAL;
                }

                if (status) {
                    filter = CONT_POLL_READ | CONT_POLL_WRITE | CONT_POLL_RDHUP;
                }

                const TEvent res = {
                    S_[pfd.fd].Data,
                    status,
                    filter,
                };

                events.push_back(res);
            }
        }

    private:
        typedef TIndexedArray<TChange> TFds;
        TFds S_;
        typedef TVector<pollfd> TPollVec;
        TPollVec T_;
    };

    struct TUserPoller: public TString {
        TUserPoller()
            : TString(GetEnv("USER_POLLER"))
        {
        }
    };
}

THolder<IPollerFace> IPollerFace::Default() {
    return Construct(*SingletonWithPriority<TUserPoller, 0>());
}

THolder<IPollerFace> IPollerFace::Construct(TStringBuf name) {
    return Construct(name ? FromString<EContPoller>(name) : EContPoller::Default);
}

THolder<IPollerFace> IPollerFace::Construct(EContPoller poller) {
    if (poller == EContPoller::Default) {
#if defined (HAVE_EPOLL_POLLER)
        poller = EContPoller::Epoll;
#elif defined(HAVE_KQUEUE_POLLER)
        poller = EContPoller::Kqueue;
#else
        poller = EContPoller::Select;
#endif
    }

    switch (poller) {
    case EContPoller::Select:
        return MakeHolder<TVirtualize<TPoller<TGenericPoller<TSelectPoller<TWithoutLocking>>>>>(poller);
    case EContPoller::Poll:
        return MakeHolder<TVirtualize<TPollPoller>>(poller);
    case EContPoller::Epoll:
#if defined(HAVE_EPOLL_POLLER)
        return MakeHolder<TVirtualize<TPoller<TGenericPoller<TEpollPoller<TWithoutLocking>>>>>(poller);
#else
        return nullptr;
#endif
    case EContPoller::Kqueue:
#if defined(HAVE_KQUEUE_POLLER)
        return MakeHolder<TVirtualize<TPoller<TGenericPoller<TKqueuePoller<TWithoutLocking>>>>>(poller);
#else
        return nullptr;
#endif
    default:
        Y_ABORT("bad poller type");
    }
}
