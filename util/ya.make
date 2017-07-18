LIBRARY(yutil)



NO_UTIL()

PEERDIR(
    util/charset
)

# datetime
JOIN_SRCS(
    all_datetime.cpp
    datetime/base.cpp
    datetime/cputimer.cpp
    datetime/systime.cpp
    datetime/constants.cpp
)

SRCS(
    datetime/parser.rl6
)

IF (OS_WINDOWS)
    SRCS(
        datetime/strptime.cpp
    )
ENDIF()

# digest
JOIN_SRCS(
    all_digest.cpp
    digest/murmur.cpp
    digest/fnv.cpp
    digest/iterator.cpp
    digest/numeric.cpp
    digest/multi.cpp
)

SRCS(
    digest/city.cpp
)

JOIN_SRCS(
    all_util.cpp
    ysafeptr.cpp
    ysaveload.cpp
    str_stl.cpp
)

# folder
JOIN_SRCS(
    all_folder.cpp
    folder/fts.cpp
    folder/filelist.cpp
    folder/dirut.cpp
    folder/path.cpp
    folder/pathsplit.cpp
    folder/iterator.cpp
    folder/tempdir.cpp
)

IF (OS_WINDOWS)
    SRCS(
        folder/lstat_win.c
        folder/dirent_win.c
    )
ENDIF()

# generic
JOIN_SRCS(
    all_generic.cpp
    generic/adaptor.cpp
    generic/array_ref.cpp
    generic/array_size.cpp
    generic/buffer.cpp
    generic/chartraits.cpp
    generic/explicit_type.cpp
    generic/function.cpp
    generic/guid.cpp
    generic/hash.cpp
    generic/hash_primes.cpp
    generic/hide_ptr.cpp
    generic/mem_copy.cpp
    generic/ptr.cpp
    generic/singleton.cpp
    generic/strbuf.cpp
    generic/strfcpy.cpp
    generic/string.cpp
    generic/utility.cpp
    generic/va_args.cpp
    generic/xrange.cpp
    generic/yexception.cpp
    generic/ymath.cpp
    generic/algorithm.cpp
    generic/bitmap.cpp
    generic/bitops.cpp
    generic/bt_exception.cpp
    generic/cast.cpp
    generic/deque.cpp
    generic/fastqueue.cpp
    generic/flags.cpp
    generic/fwd.cpp
    generic/hash_set.cpp
    generic/intrlist.cpp
    generic/is_in.cpp
    generic/iterator.cpp
    generic/iterator_range.cpp
    generic/lazy_value.cpp
    generic/list.cpp
    generic/map.cpp
    generic/mapfindptr.cpp
    generic/maybe.cpp
    generic/noncopyable.cpp
    generic/object_counter.cpp
    generic/queue.cpp
    generic/ref.cpp
    generic/refcount.cpp
    generic/region.cpp
    generic/reinterpretcast.cpp
    generic/set.cpp
    generic/stack.cpp
    generic/stlfwd.cpp
    generic/store_policy.cpp
    generic/type_name.cpp
    generic/typelist.cpp
    generic/typetraits.cpp
    generic/vector.cpp
    generic/vector_ops.cpp
    generic/ylimits.cpp
)

# memory
JOIN_SRCS(
    all_memory.cpp
    memory/tempbuf.cpp
    memory/blob.cpp
    memory/mmapalloc.cpp
    memory/alloc.cpp
    memory/pool.cpp
    memory/addstorage.cpp
    memory/segmented_string_pool.cpp
    memory/segpool_alloc.cpp
    memory/smallobj.cpp
)

# network
JOIN_SRCS(
    all_network.cpp
    network/hostip.cpp
    network/init.cpp
    network/poller.cpp
    network/socket.cpp
    network/pair.cpp
    network/address.cpp
    network/endpoint.cpp
    network/interface.cpp
    network/nonblock.cpp
    network/iovec.cpp
    network/ip.cpp
    network/netloss.cpp
    network/pollerimpl.cpp
    network/sock.cpp
)

# random
JOIN_SRCS(
    all_random.cpp
    random/common_ops.cpp
    random/easy.cpp
    random/fast.cpp
    random/lcg_engine.cpp
    random/entropy.cpp
    random/mersenne.cpp
    random/mersenne32.cpp
    random/mersenne64.cpp
    random/normal.cpp
    random/shuffle.cpp
)

SRCS(
    random/random.cpp
)

# stream
PEERDIR(
    contrib/libs/zlib
)

JOIN_SRCS(
    all_stream.cpp
    stream/buffer.cpp
    stream/buffered.cpp
    stream/debug.cpp
    stream/direct_io.cpp
    stream/file.cpp
    stream/hex.cpp
    stream/input.cpp
    stream/length.cpp
    stream/mem.cpp
    stream/multi.cpp
    stream/null.cpp
    stream/output.cpp
    stream/pipe.cpp
    stream/str.cpp
    stream/tee.cpp
    stream/zerocopy.cpp
    stream/zlib.cpp
    stream/printf.cpp
    stream/format.cpp
    stream/tempbuf.cpp
    stream/walk.cpp
    stream/aligned.cpp
    stream/holder.cpp
    stream/labeled.cpp
    stream/tokenizer.cpp
    stream/trace.cpp
)

# string
PEERDIR(
    contrib/libs/double-conversion
)

JOIN_SRCS(
    all_string.cpp
    string/builder.cpp
    string/cgiparam.cpp
    string/delim_stroka_iter.cpp
    string/escape.cpp
    string/util.cpp
    string/vector.cpp
    string/split_iterator.cpp
    string/split.cpp
    string/url.cpp
    string/kmp.cpp
    string/quote.cpp
    string/ascii.cpp
    string/printf.cpp
    string/type.cpp
    string/strip.cpp
    string/pcdata.cpp
    string/hex.cpp
    string/cstriter.cpp
    string/iterator.cpp
    string/join.cpp
    string/scan.cpp
    string/strspn.cpp
    string/subst.cpp
)

SRCS(
    string/cast.cc
)

# system
IF (OS_DARWIN)
    CFLAGS(-Wno-deprecated-declarations)
ENDIF()

IF (ARCH_ARM)
    CFLAGS(-D_FORTIFY_SOURCE=0)
ENDIF()

JOIN_SRCS(
    all_system_1.cpp
    system/atexit.cpp
    system/backtrace.cpp
    system/compat.cpp
    system/compiler.cpp
    system/condvar.cpp
    system/context.cpp
    system/daemon.cpp
    system/datetime.cpp
    system/defaults.c
    system/demangle.cpp
    system/direct_io.cpp
    system/dynlib.cpp
    system/env.cpp
    system/err.cpp
    system/error.cpp
    system/event.cpp
    system/execpath.cpp
    system/fasttime.cpp
    system/file.cpp
    system/filemap.cpp
    system/flock.cpp
    system/file_lock.cpp
    system/fs.cpp
    system/fstat.cpp
    system/getpid.cpp
    system/hostname.cpp
    system/hp_timer.cpp
    system/info.cpp
)

JOIN_SRCS(
    all_system_2.cpp
    system/madvise.cpp
    system/mem_info.cpp
    system/mktemp.cpp
    system/mlock.cpp
    system/mutex.cpp
    system/nice.cpp
    system/pipe.cpp
    system/platform.cpp
    system/progname.cpp
    system/protect.cpp
    system/rusage.cpp
    system/rwlock.cpp
    system/sanitizers.cpp
    system/sem.cpp
    system/shmat.cpp
    system/spinlock.cpp
    system/sysstat.cpp
    system/sys_alloc.cpp
    system/tempfile.cpp
    system/thread.cpp
    system/tls.cpp
    system/types.cpp
    system/user.cpp
    system/yassert.cpp
    system/yield.cpp
    system/shellcommand.cpp
    system/src_location.cpp
    system/unaligned_mem.cpp
    system/align.cpp
    system/atomic.cpp
    system/byteorder.cpp
    system/fhandle.cpp
    system/guard.cpp
    system/maxlen.cpp
    system/sigset.cpp
    system/utime.cpp
    system/cpu_id.cpp
)

IF (OS_WINDOWS)
    SRCS(
        system/fs_win.cpp
        system/winint.cpp
    )
ELSEIF (OS_CYGWIN)
    # no asm context switching on cygwin
ELSE()
    IF (ARCH_X86_64 OR ARCH_I386)
        SRCS(
            system/context_x86.asm
        )
    ENDIF()

    IF (ARCH_AARCH64 OR ARCH_ARM64)
        SRCS(
            system/context_aarch64.S
        )
    ENDIF()
ENDIF()

IF (OS_LINUX)
    SRCS(
        system/valgrind.cpp
    )

    EXTRALIBS(
        -lrt
        -ldl
    )
ENDIF()

IF (MUSL)
    PEERDIR(
        contrib/libs/linuxvdso
    )
ELSE()
    SRCS(
        system/strlcpy.c
    )

    IF (OS_LINUX OR SUN OR CYGWIN OR OS_WINDOWS)
        SRCS(
            system/freeBSD_mktemp.cpp
        )
    ENDIF()
ENDIF()

# thread
JOIN_SRCS(
    all_thread.cpp
    thread/pool.cpp
    thread/queue.cpp
    thread/lfqueue.cpp
    thread/lfstack.cpp
    thread/singleton.cpp
)

END()
