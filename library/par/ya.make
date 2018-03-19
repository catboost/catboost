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
    library/binsaver
    library/blockcodecs
    library/chromium_trace
    library/containers/ring_buffer
    library/digest/crc32c
    library/logger/global
    library/neh
    library/netliba/v12
    library/threading/local_executor
)

END()
