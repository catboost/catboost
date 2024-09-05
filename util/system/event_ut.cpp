#include "event.h"

#include <library/cpp/testing/unittest/registar.h>

#include <util/thread/pool.h>

#include <atomic>

namespace {
    struct TSharedData {
        std::atomic<size_t> Counter = 0;
        TManualEvent event;
        bool failed = false;
    };

    struct TThreadTask: public IObjectInQueue {
    public:
        TThreadTask(TSharedData& data, size_t id)
            : Data_(data)
            , Id_(id)
        {
        }

        void Process(void*) override {
            THolder<TThreadTask> This(this);

            if (Id_ == 0) {
                usleep(100);
                bool cond = Data_.Counter.load() == 0;
                if (!cond) {
                    Data_.failed = true;
                }
                Data_.event.Signal();
            } else {
                while (!Data_.event.WaitT(TDuration::Seconds(100))) {
                }
                Data_.Counter += Id_;
            }
        }

    private:
        TSharedData& Data_;
        size_t Id_;
    };

    class TSignalTask: public IObjectInQueue {
    private:
        TManualEvent& Ev_;

    public:
        TSignalTask(TManualEvent& ev)
            : Ev_(ev)
        {
        }

        void Process(void*) override {
            Ev_.Signal();
        }
    };

    class TOwnerTask: public IObjectInQueue {
    public:
        TManualEvent Barrier;
        THolder<TManualEvent> Ev;

    public:
        TOwnerTask()
            : Ev(new TManualEvent)
        {
        }

        void Process(void*) override {
            Ev->WaitI();
            Ev.Destroy();
        }
    };

} // namespace

Y_UNIT_TEST_SUITE(EventTest) {
    Y_UNIT_TEST(WaitAndSignalTest) {
        TSharedData data;
        TThreadPool queue;
        queue.Start(5);
        for (size_t i = 0; i < 5; ++i) {
            UNIT_ASSERT(queue.Add(new TThreadTask(data, i)));
        }
        queue.Stop();
        UNIT_ASSERT(data.Counter.load() == 10);
        UNIT_ASSERT(!data.failed);
    }

    Y_UNIT_TEST(ConcurrentSignalAndWaitTest) {
        // test for problem detected by thread-sanitizer (signal/wait race) SEARCH-2113
        const size_t limit = 200;
        TManualEvent event[limit];
        TThreadPool queue;
        queue.Start(limit);
        TVector<THolder<IObjectInQueue>> tasks;
        for (size_t i = 0; i < limit; ++i) {
            tasks.emplace_back(MakeHolder<TSignalTask>(event[i]));
            UNIT_ASSERT(queue.Add(tasks.back().Get()));
        }
        for (size_t i = limit; i != 0; --i) {
            UNIT_ASSERT(event[i - 1].WaitT(TDuration::Seconds(90)));
        }
        queue.Stop();
    }

    /** Test for a problem: http://nga.at.yandex-team.ru/5772 */
    Y_UNIT_TEST(DestructorBeforeSignalFinishTest) {
        return;
        TVector<THolder<IObjectInQueue>> tasks;
        for (size_t i = 0; i < 1000; ++i) {
            auto owner = MakeHolder<TOwnerTask>();
            tasks.emplace_back(MakeHolder<TSignalTask>(*owner->Ev));
            tasks.emplace_back(std::move(owner));
        }

        TThreadPool queue;
        queue.Start(4);
        for (auto& task : tasks) {
            UNIT_ASSERT(queue.Add(task.Get()));
        }
        queue.Stop();
    }
} // Y_UNIT_TEST_SUITE(EventTest)
