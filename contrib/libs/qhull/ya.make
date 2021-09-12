LIBRARY()

LICENSE(
    LicenseRef-scancode-other-permissive
    Qhull
)

LICENSE_TEXTS(.yandex_meta/licenses.list.txt)

VERSION(7.2.0)



SRCDIR(contrib/libs/qhull/qhull/src)

SRCS(
    geom2_r.c
    geom_r.c
    global_r.c
    io_r.c
    libqhull_r.c
    mem_r.c
    merge_r.c
    poly2_r.c
    poly_r.c
    qset_r.c
    random_r.c
    rboxlib_r.c
    stat_r.c
    usermem_r.c
    userprintf_rbox_r.c
    userprintf_r.c
    user_r.c
)

END()
