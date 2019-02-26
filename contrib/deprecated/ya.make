

RECURSE(
    boost
    boost/libs/test
    glib
    glib/gio
    glibmm
    jsoncpp
    libahocorasick
    libcouchbase
    libctpp
    libffi
    libintl
    libjsonxx
    libtorrent
    libtorrent/bindings/python
    libtorrent/test
    libxml++
    sigc++
    udns
    mimepp
    pygtrie
    pygtrie/tests
)


IF(OS_LINUX OR OS_DARWIN)
    RECURSE(
    
)
ENDIF()
