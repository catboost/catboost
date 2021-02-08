from libcpp.string cimport string as _std_string

cdef extern from "<util/generic/strbuf.h>" nogil:

    cdef cppclass TStringBuf:
        TStringBuf() except +
        TStringBuf(const char*) except +
        TStringBuf(const char*, size_t) except +
        const char* data()
        char* Data()
        size_t size()
        size_t Size()


cdef extern from "<util/generic/string.h>" nogil:

    size_t npos "TString::npos"

    # Inheritance is bogus, but it's safe to assume TString is-a TStringBuf via implicit cast
    cdef cppclass TString(TStringBuf):
        TString() except +
        TString(TString&) except +
        TString(_std_string&) except +
        TString(TString&, size_t, size_t) except +
        TString(char*) except +
        TString(char*, size_t) except +
        TString(char*, size_t, size_t) except +
        # as a TString formed by a repetition of character c, n times.
        TString(size_t, char) except +
        TString(char*, char*) except +
        TString(TStringBuf&) except +
        TString(TStringBuf&, TStringBuf&) except +
        TString(TStringBuf&, TStringBuf&, TStringBuf&) except +

        const char* c_str()
        size_t max_size()
        size_t length()
        void resize(size_t) except +
        void resize(size_t, char c) except +
        size_t capacity()
        void reserve(size_t) except +
        void clear() except +
        bint empty()

        char& at(size_t)
        char& operator[](size_t)
        int compare(TStringBuf&)

        TString& append(TStringBuf&) except +
        TString& append(TStringBuf&, size_t, size_t) except +
        TString& append(char *) except +
        TString& append(char *, size_t) except +
        TString& append(size_t, char) except +

        void push_back(char c) except +

        TString& assign(TStringBuf&) except +
        TString& assign(TStringBuf&, size_t, size_t) except +
        TString& assign(char *) except +
        TString& assign(char *, size_t) except +

        TString& insert(size_t, TString&) except +
        TString& insert(size_t, TString&, size_t, size_t) except +
        TString& insert(size_t, char* s) except +
        TString& insert(size_t, char* s, size_t) except +
        TString& insert(size_t, size_t, char c) except +

        size_t copy(char *, size_t) except +
        size_t copy(char *, size_t, size_t) except +

        size_t find(TStringBuf&)
        size_t find(TStringBuf&, size_t pos)
        size_t find(char)
        size_t find(char, size_t pos)

        size_t rfind(TStringBuf&)
        size_t rfind(TStringBuf&, size_t pos)
        size_t rfind(char)
        size_t rfind(char, size_t pos)

        size_t find_first_of(char c)
        size_t find_first_of(char c, size_t pos)
        size_t find_first_of(TStringBuf& set)
        size_t find_first_of(TStringBuf& set, size_t pos)

        size_t find_first_not_of(char c)
        size_t find_first_not_of(char c, size_t pos)
        size_t find_first_not_of(TStringBuf& set)
        size_t find_first_not_of(TStringBuf& set, size_t pos)

        size_t find_last_of(char c)
        size_t find_last_of(char c, size_t pos)
        size_t find_last_of(TStringBuf& set)
        size_t find_last_of(TStringBuf& set, size_t pos)

        TString substr(size_t pos) except +
        TString substr(size_t pos, size_t n) except +

        TString operator+(TStringBuf& rhs) except +
        TString operator+(char* rhs) except +

        bint operator==(TStringBuf&)
        bint operator==(char*)

        bint operator!=(TStringBuf&)
        bint operator!=(char*)

        bint operator<(TStringBuf&)
        bint operator<(char*)

        bint operator>(TStringBuf&)
        bint operator>(char*)

        bint operator<=(TStringBuf&)
        bint operator<=(char*)

        bint operator>=(TStringBuf&)
        bint operator>=(char*)
