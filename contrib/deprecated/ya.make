

RECURSE(
    blackhole
    boost
    boost/libs/test
    coreapi
    coreschema
    folly
    glib
    glib/gio
    glibmm
    itypes
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
    mongo-cxx-driver
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
