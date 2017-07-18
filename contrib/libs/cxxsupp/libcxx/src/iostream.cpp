//===------------------------ iostream.cpp --------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "__config"
#include "__std_stream"
#include "string"
#include "new"

_LIBCPP_BEGIN_NAMESPACE_STD

#ifndef _LIBCPP_HAS_NO_STDIN
_ALIGNAS_TYPE (istream) char _cin [sizeof(istream)];
_ALIGNAS_TYPE (__stdinbuf<char> ) static char __cin [sizeof(__stdinbuf <char>)];
static mbstate_t mb_cin;
_ALIGNAS_TYPE (wistream) char _wcin [sizeof(wistream)];
_ALIGNAS_TYPE (__stdinbuf<wchar_t> ) static char __wcin [sizeof(__stdinbuf <wchar_t>)];
static mbstate_t mb_wcin;

_LIBCPP_FUNC_VIS istream& cin = *reinterpret_cast<istream*>(_cin);
_LIBCPP_FUNC_VIS wistream& wcin = *reinterpret_cast<wistream*>(_wcin);
#endif

#ifndef _LIBCPP_HAS_NO_STDOUT
_ALIGNAS_TYPE (ostream)  char _cout[sizeof(ostream)];
_ALIGNAS_TYPE (__stdoutbuf<char>) static char __cout[sizeof(__stdoutbuf<char>)];
static mbstate_t mb_cout;
_ALIGNAS_TYPE (wostream) char _wcout[sizeof(wostream)];
_ALIGNAS_TYPE (__stdoutbuf<wchar_t>) static char __wcout[sizeof(__stdoutbuf<wchar_t>)];
static mbstate_t mb_wcout;

_LIBCPP_FUNC_VIS ostream& cout = *reinterpret_cast<ostream*>(_cout);
_LIBCPP_FUNC_VIS wostream& wcout = *reinterpret_cast<wostream*>(_wcout);
#endif

_ALIGNAS_TYPE (ostream)  char _cerr[sizeof(ostream)];
_ALIGNAS_TYPE (__stdoutbuf<char>) static char __cerr[sizeof(__stdoutbuf<char>)];
static mbstate_t mb_cerr;
_ALIGNAS_TYPE (wostream) char _wcerr[sizeof(wostream)];
_ALIGNAS_TYPE (__stdoutbuf<wchar_t>) static char __wcerr[sizeof(__stdoutbuf<wchar_t>)];
static mbstate_t mb_wcerr;

_LIBCPP_FUNC_VIS ostream& cerr = *reinterpret_cast<ostream*>(_cerr);
_LIBCPP_FUNC_VIS wostream& wcerr = *reinterpret_cast<wostream*>(_wcerr);

_ALIGNAS_TYPE (ostream)  char _clog[sizeof(ostream)];
_ALIGNAS_TYPE (wostream) char _wclog[sizeof(wostream)];

_LIBCPP_FUNC_VIS ostream& clog = *reinterpret_cast<ostream*>(_clog);
_LIBCPP_FUNC_VIS wostream& wclog = *reinterpret_cast<wostream*>(_wclog);

ios_base::Init __start_std_streams;

ios_base::Init::Init()
{
#ifndef _LIBCPP_HAS_NO_STDIN
    istream* cin_ptr  = ::new(_cin)  istream(::new(__cin)  __stdinbuf <char>(stdin, &mb_cin));
    wistream* wcin_ptr  = ::new(_wcin)  wistream(::new(__wcin)  __stdinbuf <wchar_t>(stdin, &mb_wcin));
#endif
#ifndef _LIBCPP_HAS_NO_STDOUT
    ostream* cout_ptr = ::new(_cout) ostream(::new(__cout) __stdoutbuf<char>(stdout, &mb_cout));
    wostream* wcout_ptr = ::new(_wcout) wostream(::new(__wcout) __stdoutbuf<wchar_t>(stdout, &mb_wcout));
#endif
    ostream* cerr_ptr = ::new(_cerr) ostream(::new(__cerr) __stdoutbuf<char>(stderr, &mb_cerr));
                        ::new(_clog) ostream(cerr_ptr->rdbuf());
    wostream* wcerr_ptr = ::new(_wcerr) wostream(::new(__wcerr) __stdoutbuf<wchar_t>(stderr, &mb_wcerr));
                          ::new(_wclog) wostream(wcerr_ptr->rdbuf());

#if !defined(_LIBCPP_HAS_NO_STDIN) && !defined(_LIBCPP_HAS_NO_STDOUT)
    cin_ptr->tie(cout_ptr);
    wcin_ptr->tie(wcout_ptr);
#endif
    _VSTD::unitbuf(*cerr_ptr);
    _VSTD::unitbuf(*wcerr_ptr);
#ifndef _LIBCPP_HAS_NO_STDOUT
    cerr_ptr->tie(cout_ptr);
    wcerr_ptr->tie(wcout_ptr);
#endif
}

ios_base::Init::~Init()
{
#ifndef _LIBCPP_HAS_NO_STDOUT
    ostream* cout_ptr = reinterpret_cast<ostream*>(_cout);
    wostream* wcout_ptr = reinterpret_cast<wostream*>(_wcout);
    cout_ptr->flush();
    wcout_ptr->flush();
#endif

    ostream* clog_ptr = reinterpret_cast<ostream*>(_clog);
    wostream* wclog_ptr = reinterpret_cast<wostream*>(_wclog);
    clog_ptr->flush();
    wclog_ptr->flush();
}

_LIBCPP_END_NAMESPACE_STD
