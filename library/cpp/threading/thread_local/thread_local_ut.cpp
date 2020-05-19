#include "thread_local.h"

#include <library/cpp/unittest/registar.h>

#include <util/generic/xrange.h>
#include <util/thread/pool.h>

namespace {

template<typename F>
void Repeat(F&& f, size_t reps) {
    for (size_t rep : xrange(reps)) {
        Y_UNUSED(rep);
        std::invoke(f);
    }
}

template<NThreading::EThreadLocalImpl Impl>
void SimpleUnitTest() {
    static constexpr ui32 threads = 32;
    static constexpr ui32 iters = 10000;

    THolder<IThreadPool> pool = CreateThreadPool(threads);
    NThreading::TThreadLocalValue<TThread::TId, Impl> threadLocal;

    TVector<std::pair<TThread::TId, TThread::TId>> ids;
    TMutex idsLock;
    for (ui32 i : xrange(threads)) {
        Y_UNUSED(i);
        pool->SafeAddFunc([&] {
            for (ui32 iter : xrange(iters)) {
                Y_UNUSED(iter);
                *threadLocal.Get() = TThread::CurrentThreadId();
                std::this_thread::yield();
            }

            with_lock(idsLock) {
                ids.emplace_back(*threadLocal.Get(), TThread::CurrentThreadId());
            }
        });
    }

    pool->Stop();
    for (auto&& [l, r] : ids) {
        UNIT_ASSERT_VALUES_EQUAL(l, r);
    }
}

class TTrace {
public:
    void Add(TString str) {
        with_lock (Lock_) {
            Trace_.emplace_back(std::move(str));
        }
    }

    const TVector<TString>& Get() const {
        return Trace_;
    }

    void Clear() {
        Trace_.clear();
    }

    static TTrace& Instance() {
        return *Singleton<TTrace>();
    }

private:
    TVector<TString> Trace_;
    TMutex Lock_;
};

class TNoisy {
public:
    TNoisy(const char* name = "TNoisy")
        : Name_{name} {
        TTrace::Instance().Add(Sprintf("%s::%s()", Name_, Name_));
    }

    ~TNoisy() {
        TTrace::Instance().Add(Sprintf("%s::~%s()", Name_, Name_));
    }
private:
    const char* Name_;
};

template<NThreading::EThreadLocalImpl Impl>
class TWrapper {
public:
    void DoWork() {
        ThreadLocal_.Get();
    }

private:
    TNoisy It{"TWrapper"};
    NThreading::TThreadLocalValue<TNoisy, Impl> ThreadLocal_;
};

template<NThreading::EThreadLocalImpl Impl>
void TraceUnitTest() {
    TTrace::Instance().Clear();

    for (ui32 i : xrange(2)) {
        Y_UNUSED(i);
        THolder<IThreadPool> pool = CreateThreadPool(3);
        TWrapper<Impl> wrapper;

        TAtomic sync; // Sorry
        AtomicSet(sync, 3);

        for (ui32 i : xrange(3)) {
            Y_UNUSED(i);
            pool->SafeAddFunc([&] {
                intptr_t cur = AtomicDecrement(sync);
                while (cur) {
                    std::this_thread::yield();
                    cur = AtomicGet(sync);
                }
                wrapper.DoWork();
            });
        }
        pool->Stop();
    }

    TVector<TString> expected;
    for (ui32 i : xrange(2)) {
        Y_UNUSED(i);
        expected.emplace_back("TWrapper::TWrapper()");
        for (ui32 i : xrange(3)) {
            Y_UNUSED(i);
            expected.emplace_back("TNoisy::TNoisy()");
        }
        for (ui32 i : xrange(3)) {
            Y_UNUSED(i);
            expected.emplace_back("TNoisy::~TNoisy()");
        }
        expected.emplace_back("TWrapper::~TWrapper()");
    }
    UNIT_ASSERT_VALUES_EQUAL(TTrace::Instance().Get(), expected);
}

struct TNonCopy {
    TNonCopy() = default;

    TNonCopy(const TNonCopy&) = delete;
    TNonCopy(TNonCopy&&) = default;

    TNonCopy& operator=(const TNonCopy&) = delete;
    TNonCopy& operator=(TNonCopy&&) = default;
};

template<NThreading::EThreadLocalImpl Impl>
void MoveOnlyUnitTest() {
    NThreading::TThreadLocalValue<TNonCopy, Impl> tls;
    Y_UNUSED(tls);
}

}

Y_UNIT_TEST_SUITE(HotSwapTest) {
    static constexpr NThreading::EThreadLocalImpl Impl = NThreading::EThreadLocalImpl::HotSwap;

    Y_UNIT_TEST(Simple) {
        Repeat(SimpleUnitTest<Impl>, 100);
    }

    Y_UNIT_TEST(Trace) {
        Repeat(TraceUnitTest<Impl>, 100);
    }

    Y_UNIT_TEST(MoveOnly) {
        MoveOnlyUnitTest<Impl>();
    }
}

Y_UNIT_TEST_SUITE(SkipListTest) {
    static constexpr NThreading::EThreadLocalImpl Impl = NThreading::EThreadLocalImpl::SkipList;

    Y_UNIT_TEST(Simple) {
        Repeat(SimpleUnitTest<Impl>, 100);
    }

    Y_UNIT_TEST(Trace) {
        Repeat(TraceUnitTest<Impl>, 100);
    }

    Y_UNIT_TEST(MoveOnly) {
        MoveOnlyUnitTest<Impl>();
    }
}

Y_UNIT_TEST_SUITE(ForwardListTest) {
    static constexpr NThreading::EThreadLocalImpl Impl = NThreading::EThreadLocalImpl::ForwardList;

    Y_UNIT_TEST(Simple) {
        Repeat(SimpleUnitTest<Impl>, 100);
    }

    Y_UNIT_TEST(Trace) {
        Repeat(TraceUnitTest<Impl>, 100);
    }

    Y_UNIT_TEST(MoveOnly) {
        MoveOnlyUnitTest<Impl>();
    }
}
