LIBRARY(yutil)



NEED_CHECK()

NO_UTIL()

# stream
# string
PEERDIR(
    util/charset
    contrib/libs/zlib
    contrib/libs/double-conversion
)

IF (OS_ANDROID)
    PEERDIR(
        contrib/libs/android_ifaddrs
    )
    ADDINCL(
        contrib/libs/android_ifaddrs
    )
ENDIF()

# datetime
JOIN_SRCS(
    all_datetime.cpp
    datetime/base.cpp
    datetime/constants.cpp
    datetime/cputimer.cpp
    datetime/systime.cpp
    datetime/uptime.cpp
)

SRCS(
    datetime/parser.rl6
    digest/city.cpp
    random/random.cpp
    string/cast.cpp
)

IF (OS_WINDOWS)
    SRCS(
        datetime/strptime.cpp
    )
ENDIF()

# digest
JOIN_SRCS(
    all_digest.cpp
    digest/fnv.cpp
    digest/multi.cpp
    digest/murmur.cpp
    digest/numeric.cpp
    digest/sequence.cpp
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
    folder/dirut.cpp
    folder/filelist.cpp
    folder/fts.cpp
    folder/iterator.cpp
    folder/path.cpp
    folder/pathsplit.cpp
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
    generic/scope.cpp
    generic/adaptor.cpp
    generic/algorithm.cpp
    generic/array_ref.cpp
    generic/array_size.cpp
    generic/bitmap.cpp
    generic/bitops.cpp
    generic/bt_exception.cpp
    generic/buffer.cpp
    generic/cast.cpp
    generic/chartraits.cpp
    generic/deque.cpp
    generic/explicit_type.cpp
    generic/fastqueue.cpp
    generic/flags.cpp
    generic/function.cpp
    generic/fwd.cpp
    generic/guid.cpp
    generic/hash.cpp
    generic/hash_primes.cpp
    generic/hash_set.cpp
    generic/hide_ptr.cpp
    generic/intrlist.cpp
    generic/is_in.cpp
    generic/iterator.cpp
    generic/iterator_range.cpp
    generic/lazy_value.cpp
    generic/list.cpp
    generic/map.cpp
    generic/mapfindptr.cpp
    generic/maybe.cpp
    generic/mem_copy.cpp
    generic/noncopyable.cpp
    generic/object_counter.cpp
    generic/ptr.cpp
    generic/queue.cpp
    generic/refcount.cpp
    generic/serialized_enum.cpp
    generic/set.cpp
    generic/singleton.cpp
    generic/size_literals.cpp
    generic/stack.cpp
    generic/store_policy.cpp
    generic/strbuf.cpp
    generic/strfcpy.cpp
    generic/string.cpp
    generic/typelist.cpp
    generic/type_name.cpp
    generic/typetraits.cpp
    generic/utility.cpp
    generic/va_args.cpp
    generic/vector.cpp
    generic/xrange.cpp
    generic/yexception.cpp
    generic/ylimits.cpp
    generic/ymath.cpp
)

# memory
JOIN_SRCS(
    all_memory.cpp
    memory/addstorage.cpp
    memory/alloc.cpp
    memory/blob.cpp
    memory/mmapalloc.cpp
    memory/pool.cpp
    memory/segmented_string_pool.cpp
    memory/segpool_alloc.cpp
    memory/smallobj.cpp
    memory/tempbuf.cpp
)

# network
JOIN_SRCS(
    all_network.cpp
    network/address.cpp
    network/endpoint.cpp
    network/hostip.cpp
    network/init.cpp
    network/interface.cpp
    network/iovec.cpp
    network/ip.cpp
    network/netloss.cpp
    network/nonblock.cpp
    network/pair.cpp
    network/poller.cpp
    network/pollerimpl.cpp
    network/sock.cpp
    network/socket.cpp
)

# random
JOIN_SRCS(
    all_random.cpp
    random/common_ops.cpp
    random/easy.cpp
    random/entropy.cpp
    random/fast.cpp
    random/lcg_engine.cpp
    random/mersenne32.cpp
    random/mersenne64.cpp
    random/mersenne.cpp
    random/normal.cpp
    random/shuffle.cpp
    random/init_atfork.cpp
)

JOIN_SRCS(
    all_stream.cpp
    stream/aligned.cpp
    stream/buffer.cpp
    stream/buffered.cpp
    stream/debug.cpp
    stream/direct_io.cpp
    stream/file.cpp
    stream/format.cpp
    stream/fwd.cpp
    stream/hex.cpp
    stream/holder.cpp
    stream/input.cpp
    stream/labeled.cpp
    stream/length.cpp
    stream/mem.cpp
    stream/multi.cpp
    stream/null.cpp
    stream/output.cpp
    stream/pipe.cpp
    stream/printf.cpp
    stream/str.cpp
    stream/tee.cpp
    stream/tempbuf.cpp
    stream/tokenizer.cpp
    stream/trace.cpp
    stream/walk.cpp
    stream/zerocopy.cpp
    stream/zerocopy_output.cpp
    stream/zlib.cpp
)

JOIN_SRCS(
    all_string.cpp
    string/ascii.cpp
    string/builder.cpp
    string/cstriter.cpp
    string/escape.cpp
    string/hex.cpp
    string/join.cpp
    string/printf.cpp
    string/split.cpp
    string/strip.cpp
    string/strspn.cpp
    string/subst.cpp
    string/type.cpp
    string/util.cpp
    string/vector.cpp
)

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
    system/file_lock.cpp
    system/filemap.cpp
    system/flock.cpp
    system/fs.cpp
    system/fstat.cpp
    system/getpid.cpp
    system/hi_lo.cpp
    system/hostname.cpp
    system/hp_timer.cpp
    system/info.cpp
)

JOIN_SRCS(
    all_system_2.cpp
    system/align.cpp
    system/atomic.cpp
    system/byteorder.cpp
    system/cpu_id.cpp
    system/fhandle.cpp
    system/guard.cpp
    system/interrupt_signals.cpp
    system/madvise.cpp
    system/maxlen.cpp
    system/mem_info.cpp
    system/mincore.cpp
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
    system/shellcommand.cpp
    system/shmat.cpp
    system/sigset.cpp
    system/spinlock.cpp
    system/spin_wait.cpp
    system/src_location.cpp
    system/sys_alloc.cpp
    system/sysstat.cpp
    system/tempfile.cpp
    system/thread.cpp
    system/tls.cpp
    system/types.cpp
    system/unaligned_mem.cpp
    system/user.cpp
    system/utime.cpp
    system/yassert.cpp
    system/yield.cpp
)

IF (OS_WINDOWS)
    SRCS(
        system/fs_win.cpp
        system/winint.cpp
    )
ELSEIF (OS_CYGWIN OR OS_IOS)
    # no asm context switching on cygwin or iOS
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
            system/mktemp_system.cpp
        )
    ENDIF()
ENDIF()

# thread
JOIN_SRCS(
    all_thread.cpp
    thread/factory.cpp
    thread/fwd.cpp
    thread/lfqueue.cpp
    thread/lfstack.cpp
    thread/pool.cpp
    thread/singleton.cpp
)

END()
