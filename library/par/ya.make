LIBRARY()



SRCS(
    distr_tree.cpp
    compression.cpp
    par.cpp
    par_context.cpp
    par_exec.cpp
    par_host.cpp
    par_jobreq.cpp
    par_master.cpp
    par_mr.cpp
    par_network.cpp
    par_remote.cpp
    par_util.cpp
    par_log.cpp
    par_wb.cpp
)

GENERATE_ENUM_SERIALIZATION(par_host_stats.h)

PEERDIR(
    library/cpp/binsaver
    library/blockcodecs
    library/chromium_trace
    library/cpp/containers/ring_buffer
    library/cpp/digest/crc32c
    library/cpp/logger/global
    library/neh
    library/netliba/v12
    library/cpp/threading/atomic
    library/cpp/threading/local_executor
)

END()
