#include "location.h"

#include <util/generic/string.h>
#include <util/generic/vector.h>
#include <library/cpp/testing/unittest/registar.h>

Y_UNIT_TEST_SUITE(TParsedLocationTest) {
    struct TUrlTestCase {
        TString Scheme = "";
        TString Host = "";
        TString Port = "";
        TString Service = "";
        TString EndPoint = "";
        TString Url = "";

        TUrlTestCase(const TString& scheme, const TString& host, const TString& port, const TString& service)
            : Scheme(scheme)
            , Host(host)
            , Port(port)
            , Service(service != "/" ? service : "")
        {
            EndPoint = Host;
            if (!Port.empty()) {
                EndPoint = TStringBuilder() << EndPoint << ":" << Port;
            }

            Url = TStringBuilder() << Scheme << "://" << Host;
            if (!Port.empty()) {
                Url = TStringBuilder() << Url << ":" << Port;
            }
            if (service == "/") {
                Url = TStringBuilder() << Url << "/";
            } else if (!Service.empty()) {
                Url = TStringBuilder() << Url << "/" << Service;
            }
        }
    };

    Y_UNIT_TEST(TestEqual) {
        TVector<TString> schemes{"http", "http+unix"};
        TVector<TString> hosts{"[::1]", "[2a02:6b8:0:1410::5f6c:f3c2]", "yandex.ru", "[/tmp/unixsocket]"};
        TVector<TString> ports{"", "12345"};
        TVector<TString> services{"", "/", "service"};

        TVector<TUrlTestCase> testCases;
        for (const TString& scheme : schemes) {
            for (const TString& host : hosts) {
                for (const TString& port : ports) {
                    for (const TString& service : services) {
                        testCases.emplace_back(scheme, host, port, service);
                    }
                }
            }
        }

        for (const auto& testCase : testCases) {
            TString url = testCase.Url;
            NNeh::TParsedLocation parsed(url);

            UNIT_ASSERT_C(parsed.Scheme == testCase.Scheme, TStringBuilder() << parsed.Scheme << " != " << testCase.Scheme);
            UNIT_ASSERT_C(parsed.EndPoint == testCase.EndPoint, TStringBuilder() << parsed.EndPoint << " != " << testCase.EndPoint << " " << url);
            UNIT_ASSERT_C(parsed.Host == testCase.Host, TStringBuilder() << parsed.Host << " != " << testCase.Host << " " << url);
            UNIT_ASSERT_C(parsed.Port == testCase.Port, TStringBuilder() << parsed.Port << " != " << testCase.Port << " " << url);
            UNIT_ASSERT_C(parsed.Service == testCase.Service, TStringBuilder() << parsed.Service << " != " << testCase.Service << " " << url);
        }
    }
}
