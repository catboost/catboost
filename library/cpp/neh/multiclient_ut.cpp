#include "multiclient.h"
#include "rpc.h"
#include "utils.h"

#include <library/cpp/testing/unittest/registar.h>

#include <util/stream/str.h>
#include <util/thread/factory.h>
#include <util/system/mutex.h>

using namespace NNeh;

Y_UNIT_TEST_SUITE(TNehMultiClient) {
    class TResponseDelayer: public IThreadFactory::IThreadAble {
        struct TTmResponse : TIntrusiveListItem<TTmResponse> {
            TTmResponse(TInstant time, const IRequestRef& request, TData& data)
                : Time(time)
                , Request(request)
            {
                Data.swap(data);
            }

            TInstant Time;
            IRequestRef Request;
            TData Data;
        };

    public:
        TResponseDelayer()
            : E_(TSystemEvent::rAuto)
            , Shutdown_(false)
        {
        }

        ~TResponseDelayer() override {
            Stop();
            if (!!Thr_) {
                Thr_->Join();
            }
            while (!R_.Empty()) {
                delete R_.PopFront();
            }
        }

        void Run() {
            Thr_ = SystemThreadFactory()->Run(this);
        }

        void Stop() {
            Shutdown_ = true;
            E_.Signal();
        }

        void SendResponseAt(TInstant& at, const IRequestRef& req, TData& data) {
            {
                TGuard<TMutex> g(M_);
                R_.PushBack(new TTmResponse(at, req, data));
            }
            E_.Signal();
        }

    private:
        void DoExecute() override {
            for (;;) {
                TInstant tm = TInstant::Max();
                TTmResponse* resp = nullptr;
                {
                    TGuard<TMutex> g(M_);
                    for (TIntrusiveList<TTmResponse>::TIterator it = R_.Begin(); it != R_.End(); ++it) {
                        if (it->Time < tm) {
                            tm = it->Time;
                            resp = &*it;
                        }
                    }
                }
                if (!E_.WaitD(tm)) {
                    resp->Request->SendReply(resp->Data);
                    TGuard<TMutex> g(M_);
                    delete resp;
                }
                if (Shutdown_) {
                    break;
                }
            }
        }

    private:
        TAutoPtr<IThreadFactory::IThread> Thr_;
        TMutex M_;
        TIntrusiveList<TTmResponse> R_;
        TSystemEvent E_;
        TAtomicBool Shutdown_;
    };

    class TServer {
    public:
        TServer(const TString& response)
            : R_(response)
        {
            D_.Run();
        }

        void ServeRequest(const IRequestRef& req) {
            TData res(R_.data(), R_.data() + R_.size());
            if (req->Data().StartsWith("delay ")) {
                TStringBuf delay = req->Data();
                delay.Skip(6);
                TInstant respTime = TDuration::MilliSeconds(FromString<size_t>(delay.Before('\n'))).ToDeadLine();
                D_.SendResponseAt(respTime, req, res);
            } else {
                req->SendReply(res);
            }
        }

    private:
        TString R_;
        TResponseDelayer D_;
    };

    class TResponsesDispatcher: public IThreadFactory::IThreadAble {
    public:
        TResponsesDispatcher(IMultiClient& mc)
            : MC_(mc)
        {
        }

        ~TResponsesDispatcher() override {
            Stop();
        }

        void Run() {
            Thr_ = SystemThreadFactory()->Run(this);
        }

        void Stop() {
            MC_.Interrupt();
            if (!!Thr_) {
                Thr_->Join();
            }
        }

    private:
        void DoExecute() override {
            try {
                size_t evNum = 0;
                IMultiClient::TEvent ev;
                while (MC_.Wait(ev)) {
                    Cdbg << "ev.Type = " << int(ev.Type) << Endl;
                    if (ev.Type == IMultiClient::TEvent::Response) {
                        TResponseRef resp = ev.Hndl->Get();
                        if (!!resp) {
                            Cdbg << "Request = " << resp->Request.Addr << ": " << resp->Request.Data << Endl;
                            if (resp->IsError()) {
                                Cdbg << "ErrorResponse = " << resp->GetErrorText() << Endl;
                            } else {
                                Cdbg << "Response = " << resp->Data << Endl;
                            }
                        }
                    } else {
                        Cdbg << "Timeout" << Endl;
                    }
                    Sleep(TDuration::MilliSeconds(5));
                    if (!ev.UserData) {
                        Error << "unexpected event";
                        return;
                    }
                    TStringBuf userData((const char*)ev.UserData);
                    if (userData.EndsWith('t')) {
                        if (ev.Type != IMultiClient::TEvent::Timeout) {
                            Error << "expect event timeout " << evNum << ", but have: " << userData;
                            return;
                        }
                        userData.Chop(1);
                    } else {
                        if (ev.Type != IMultiClient::TEvent::Response) {
                            Error << "expect event response " << evNum << ", but have: " << userData;
                            return;
                        }
                    }
                    size_t recEv = FromString<size_t>(userData);
                    if (recEv != evNum) {
                        Error << "expect event num " << evNum << ", but have: " << recEv;
                        return;
                    }
                    ++evNum;
                    ev.UserData = nullptr;
                }
                Cdbg << "Interrupted" << Endl;
                if (evNum != 5) {
                    Error << "receive not all events - expect next event: " << evNum;
                    return;
                }
            } catch (...) {
                Error << CurrentExceptionMessage();
            }
        }

    public:
        TStringStream Error;

    private:
        IMultiClient& MC_;
        TAutoPtr<IThreadFactory::IThread> Thr_;
    };

    Y_UNIT_TEST(TFewRequests) {
        TServer srv("test-response");
        IServicesRef svs = CreateLoop();
        const char* const url = "inproc://x:1/x";
        const char* const badUrl = "inproc://x:2/x";
        svs->Add(url, srv);
        svs->ForkLoop(2);

        {
            TMultiClientPtr mc = CreateMultiClient();
            TResponsesDispatcher disp(*mc);
            disp.Run();

            try {
                //request to nonregistered handler
                mc->Request(IMultiClient::TRequest(TMessage(badUrl, "test-data-0")));
                UNIT_ASSERT_C(false, "request to not existed inptoc service MUST cause exception");
            } catch (...) {
            }
            mc->Request(IMultiClient::TRequest(TMessage(url, "test-data-2"), TInstant::Max(), (void*)"0"));
            mc->Request(IMultiClient::TRequest(TMessage(url, "delay 10000\ntest-data-1"), TDuration::MilliSeconds(30).ToDeadLine(), (void*)"1t"));
            mc->Request(IMultiClient::TRequest(TMessage(url, "delay 50\ntest-data-3"), TDuration::MilliSeconds(5000).ToDeadLine(), (void*)"2"));
            mc->Request(IMultiClient::TRequest(TMessage(url, "delay 1000\ntest-data-4"), TInstant::Max(), (void*)"3"));
            mc->Request(IMultiClient::TRequest(TMessage(url, "delay 3000\ntest-data-5"), TDuration::MilliSeconds(2000).ToDeadLine(), (void*)"4t"));
            mc->Request(IMultiClient::TRequest(TMessage(url, "delay 10000\ntest-data-6")));

            Sleep(TDuration::MilliSeconds(4000));

            disp.Stop();
            if (!disp.Error.Empty()) {
                throw yexception() << disp.Error.Str();
            }
        }
    }

    Y_UNIT_TEST(TWaitDeadline) {
        TMultiClientPtr mc = CreateMultiClient();
        IMultiClient::TEvent event;
        UNIT_ASSERT(!mc->Wait(event, TDuration::MilliSeconds(1).ToDeadLine()));
        UNIT_ASSERT(!mc->Wait(event, TDuration::MilliSeconds(1).ToDeadLine())); // the second try (check Interrupt_ flag in cycle)
    }
}
