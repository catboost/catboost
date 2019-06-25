#pragma once

class TCont;
class TContExecutor;

namespace NCoro {

    class IScheduleCallback {
    public:
        virtual ~IScheduleCallback() = default;

        virtual void OnSchedule(TContExecutor&, TCont&) {}

        virtual void OnUnschedule(TContExecutor&) {}
    };
}
