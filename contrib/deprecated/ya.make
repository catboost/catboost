

RECURSE(
    blackhole
    boost
    boost/libs/test
    folly
    glib
    glib/gio
    glibmm
    http-parser
    jsoncpp
    libahocorasick
    libcds
    libcouchbase
    libctpp
    libffi
    libintl
    libjsonxx
    libtorrent
    libtorrent/bindings/python
    libtorrent/test
    libxml++
    mimepp
    pygtrie
    pygtrie/tests
    sigc++
    udns
    uriparser
)


IF(OS_LINUX OR OS_DARWIN)
    RECURSE(
    
)
ENDIF()
