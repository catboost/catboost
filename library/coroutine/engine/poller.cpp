#include "poller.h"
#include "sockmap.h"

#include <util/memory/smallobj.h>
#include <util/generic/intrlist.h>
#include <util/generic/singleton.h>
#include <util/system/env.h>

namespace {
    typedef IPollerFace::TChange TChange;
    typedef IPollerFace::TEvent TEvent;
    typedef IPollerFace::TEvents TEvents;

    template <class T>
    class TUnsafeBuf {
    public:
        inline TUnsafeBuf() noexcept
            : L_(0)
        {
        }

        inline T* operator~() const noexcept {
            return B_.Get();
        }

        inline size_t operator+() const noexcept {
            return L_;
        }

        inline void Reserve(size_t len) {
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
        void Set(const TChange& c) override {
            P_.Set(c);
        }

        void Wait(TEvents& events, TInstant deadLine) override {
            P_.Wait(events, deadLine);
        }

    private:
        T P_;
    };

    template <class T>
    class TPoller {
        typedef typename T::TEvent TInternalEvent;

    public:
        inline TPoller() {
            E_.Reserve(1);
        }

        inline void Set(const TChange& c) {
            P_.Set(c.Data, c.Fd, c.Flags);
        }

        inline void Reserve(size_t size) {
            E_.Reserve(size);
        }

        inline void Wait(TEvents& events, TInstant deadLine) {
            const size_t ret = P_.WaitD(~E_, +E_, deadLine);

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
        struct TVal: public T, public TIntrusiveListItem<TVal>, public TObjectFromPool<TVal> {
            // NOTE Constructor must be user-defined (and not =default) here
            // because TVal objects are created in the UB-capable placement
            // TObjectFromPool::new operator that stores data in a memory
            // allocated for the object. Without user defined constructor
            // zero-initialization takes place in TVal() expression and the
            // data is overwritten.
            inline TVal() {
            }
        };

        typedef TIntrusiveList<TVal> TListType;

    public:
        typedef typename TListType::TIterator TIterator;
        typedef typename TListType::TConstIterator TConstIterator;

        inline TIndexedArray()
            : P_(TMemoryPool::TExpGrow::Instance(), TDefaultAllocator::Instance())
        {
        }

        inline TIterator Begin() noexcept {
            return I_.Begin();
        }

        inline TIterator End() noexcept {
            return I_.End();
        }

        inline TConstIterator Begin() const noexcept {
            return I_.Begin();
        }

        inline TConstIterator End() const noexcept {
            return I_.End();
        }

        inline T& operator[](size_t i) {
            return *Get(i);
        }

        inline T* Get(size_t i) {
            TValRef& v = V_.Get(i);

            if (Y_UNLIKELY(!v)) {
                v.Reset(new (&P_) TVal());
                I_.PushFront(v.Get());
            }

            Y_PREFETCH_WRITE(v.Get(), 1);

            return v.Get();
        }

        inline void Erase(size_t i) noexcept {
            V_.Get(i).Destroy();
        }

        inline size_t Size() const noexcept {
            return I_.Size();
        }

    private:
        typedef TAutoPtr<TVal> TValRef;
        typename TVal::TPool P_;
        TSocketMap<TValRef> V_;
        TListType I_;
    };

    static inline short PollFlags(ui16 flags) noexcept {
        short ret = 0;

        if (flags & CONT_POLL_READ) {
            ret |= POLLIN;
        }

        if (flags & CONT_POLL_WRITE) {
            ret |= POLLOUT;
        }

        return ret;
    }

    class TPollPoller {
    public:
        inline size_t Size() const noexcept {
            return S_.Size();
        }

        template <class T>
        inline void Build(T& t) const {
            for (TFds::TConstIterator it = S_.Begin(); it != S_.End(); ++it) {
                t.Set(*it);
            }

            t.Reserve(Size());
        }

        inline void Set(const TChange& c) {
            if (c.Flags) {
                S_[c.Fd] = c;
            } else {
                S_.Erase(c.Fd);
            }
        }

        inline void Wait(TEvents& events, TInstant deadLine) {
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

            const ssize_t ret = PollD(~T_, (nfds_t) + T_, deadLine);

            if (ret < 0) {
                return;
            }

            for (size_t i = 0; i < +T_; ++i) {
                const pollfd& pfd = T_[i];
                const short ev = pfd.revents;

                if (!ev) {
                    continue;
                }

                int status = 0;
                ui16 filter = 0;

                if (ev & POLLIN) {
                    filter |= CONT_POLL_READ;
                }

                if (ev & POLLOUT) {
                    filter |= CONT_POLL_WRITE;
                }

                if (ev & POLLERR) {
                    status = EIO;
                } else if (ev & POLLHUP && pfd.events & POLLOUT) {
                    // Only write operations may cause EPIPE
                    status = EPIPE;
                } else if (ev & POLLNVAL) {
                    status = EINVAL;
                }

                if (status) {
                    filter = CONT_POLL_READ | CONT_POLL_WRITE;
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

    class TCombinedPoller {
        typedef TPoller<TPollerImpl<TWithoutLocking>> TDefaultPoller;

    public:
        inline TCombinedPoller() {
            P_.Reset(new TPollPoller());
        }

        inline void Set(const TChange& c) {
            if (!P_) {
                D_->Set(c);
            } else {
                P_->Set(c);
            }
        }

        inline void Wait(TEvents& events, TInstant deadLine) {
            if (!P_) {
                D_->Wait(events, deadLine);
            } else {
                if (P_->Size() > 200) {
                    D_.Reset(new TDefaultPoller());
                    P_->Build(*D_);
                    P_.Destroy();
                    D_->Wait(events, deadLine);
                } else {
                    P_->Wait(events, deadLine);
                }
            }
        }

    private:
        TAutoPtr<TPollPoller> P_;
        TAutoPtr<TDefaultPoller> D_;
    };

    struct TUserPoller: public TString {
        inline TUserPoller()
            : TString(GetEnv("USER_POLLER"))
        {
        }
    };
}

TAutoPtr<IPollerFace> IPollerFace::Default() {
    return Construct(*Singleton<TUserPoller>());
}

TAutoPtr<IPollerFace> IPollerFace::Construct(const TStringBuf& name) {
    if (!name || name == AsStringBuf("default")) {
        return new TVirtualize<TCombinedPoller>();
    }

#if defined(HAVE_KQUEUE_POLLER)
    if (name == AsStringBuf("kqueue")) {
        return new TVirtualize<TPoller<TGenericPoller<TKqueuePoller<TWithoutLocking>>>>();
    }
#endif

#if defined(HAVE_EPOLL_POLLER)
    if (name == AsStringBuf("epoll")) {
        return new TVirtualize<TPoller<TGenericPoller<TEpollPoller<TWithoutLocking>>>>();
    }
#endif

#if defined(HAVE_SELECT_POLLER)
    if (name == AsStringBuf("select")) {
        return new TVirtualize<TPoller<TGenericPoller<TSelectPoller<TWithoutLocking>>>>();
    }
#endif

    if (name == AsStringBuf("poll")) {
        return new TVirtualize<TPollPoller>();
    }

    ythrow yexception() << "unsupported poller " << name;
}
