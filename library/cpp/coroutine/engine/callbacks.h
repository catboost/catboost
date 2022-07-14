#pragma once

class TCont;
class TContExecutor;

namespace NCoro {
    class IScheduleCallback {
    public:
        virtual void OnSchedule(TContExecutor&, TCont&) = 0;
        virtual void OnUnschedule(TContExecutor&) = 0;
    };

    class IEnterPollerCallback {
    public:
        virtual void OnEnterPoller() = 0;
        virtual void OnExitPoller() = 0;
    };
}
