

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
    mongo-cxx-driver
    python
    sigc++
    udns
    uriparser
    wangle
)


IF(OS_LINUX OR OS_DARWIN)
    RECURSE(
    
)
ENDIF()
