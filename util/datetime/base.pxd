from libc.stdint cimport uint32_t
from libc.stdint cimport uint64_t
from libcpp cimport bool as bool_t
from posix.types cimport time_t

from util.generic.string cimport TString, TStringBuf


cdef extern from "<util/datetime/base.h>" nogil:

    cdef cppclass TTimeBase:
        TTimeBase()
        TTimeBase(uint64_t)

        uint64_t GetValue()
        double SecondsFloat()
        uint64_t MicroSeconds()
        uint64_t MilliSeconds()
        uint64_t Seconds()
        uint64_t Minutes()
        uint64_t Hours()
        uint64_t Days()
        uint64_t NanoSeconds()
        uint32_t MicroSecondsOfSecond()
        uint32_t MilliSecondsOfSecond()
        uint32_t NanoSecondsOfSecond()


    cdef cppclass TInstant(TTimeBase):
        TInstant()
        TInstant(uint64_t)

        @staticmethod
        TInstant Now() except +
        @staticmethod
        TInstant Max()
        @staticmethod
        TInstant Zero()
        @staticmethod
        TInstant MicroSeconds(uint64_t)
        @staticmethod
        TInstant MilliSeconds(uint64_t)
        @staticmethod
        TInstant Seconds(uint64_t)
        @staticmethod
        TInstant Minutes(uint64_t)
        @staticmethod
        TInstant Hours(uint64_t)
        @staticmethod
        TInstant Days(uint64_t)

        time_t TimeT()

        TString ToString() except +
        TString ToStringUpToSeconds() except +
        TString ToStringLocal() except +
        TString ToStringLocalUpToSeconds() except +
        TString FormatLocalTime(const char*)
        TString FormatGmTime(const char* format)

        @staticmethod
        TInstant ParseIso8601(const TStringBuf) except +
        @staticmethod
        TInstant ParseRfc822(const TStringBuf) except +
        @staticmethod
        TInstant ParseHttp(const TStringBuf) except +
        @staticmethod
        TInstant ParseX509Validity(const TStringBuf) except +

        @staticmethod
        bool_t TryParseIso8601(const TStringBuf, TInstant&) except +
        @staticmethod
        bool_t TryParseRfc822(const TStringBuf, TInstant&) except +
        @staticmethod
        bool_t TryParseHttp(const TStringBuf, TInstant&) except +
        @staticmethod
        bool_t TryParseX509(const TStringBuf, TInstant&) except +

        @staticmethod
        TInstant ParseIso8601Deprecated(const TStringBuf) except +
        @staticmethod
        TInstant ParseRfc822Deprecated(const TStringBuf) except +
        @staticmethod
        TInstant ParseHttpDeprecated(const TStringBuf) except +
        @staticmethod
        TInstant ParseX509ValidityDeprecated(const TStringBuf) except +

        @staticmethod
        bool_t TryParseIso8601Deprecated(const TStringBuf, TInstant&) except +
        @staticmethod
        bool_t TryParseRfc822Deprecated(const TStringBuf, TInstant&) except +
        @staticmethod
        bool_t TryParseHttpDeprecated(const TStringBuf, TInstant&) except +
        @staticmethod
        bool_t TryParseX509Deprecated(const TStringBuf, TInstant&) except +


    cdef cppclass TDuration(TTimeBase):
        TDuration()
        TDuration(uint64_t)

        @staticmethod
        TDuration MicroSeconds(uint64_t)

        TInstant ToDeadLine() except +
        TInstant ToDeadLine(TInstant) except +

        @staticmethod
        TDuration Max()
        @staticmethod
        TDuration Zero()
        @staticmethod
        TDuration Seconds(uint64_t)
        @staticmethod
        TDuration Minutes(uint64_t)
        @staticmethod
        TDuration Hours(uint64_t)
        @staticmethod
        TDuration Days(uint64_t)

        @staticmethod
        TDuration Parse(const TStringBuf)
        @staticmethod
        bool_t TryParse(const TStringBuf, TDuration&)

        TString ToString() except +
