#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import unicode_literals
"""
    Convert minidump machine-readable (-m) stackdump produced by minidump_stackwalk
    to format similar to gdb output (suitable for coredump_filter and cores aggregator)

    Author: mvel@
    Packaging (pypi.yandex-team.ru): alonger@
"""

import sys


def print_minidump_as_core(file_name):
    """
        Print core text to stdout
    """
    print(minidump_file_to_core(file_name))


def minidump_file_to_core(file_name):
    """
        Convert minidump `file_name` to gdb stacktrace
        :return stacktrace as string
    """
    return minidump_text_to_core(open(file_name).read())


def minidump_text_to_core(minidump_text):
    """
        Convert minidump text to gdb stacktrace
        :return stacktrace as string
    """
    core_text = ''
    threads = minidump_text_to_threads(minidump_text)
    for thread in threads:
        for line in thread:
            core_text += line + '\n'
    return core_text


def minidump_text_to_threads(minidump_text):
    """
        Convert minidump text to threads list
        :return list of threads
    """
    minidump_lines = minidump_text.splitlines()

    threads = []

    def process_thread(stack, active_thread_id):
        """
        0|0|libpthread-2.15.so|pthread_join|/build/buildd/eglibc-2.15/nptl/pthread_join.c|89|0x15
        0|1|libpthread-2.15.so||||0x9070
        0|2|srch-base-7332|TThread::Join|/place/sandbox-data/srcdir/arcadia_cache_3/util/system/thread.cpp|153|0x5
        0|3|srch-base-7332|THttpServer::Wait|/place/sandbox-data/srcdir/arcadia_cache_3/library/http/server/http.cpp|191|0x5

        Thread 583 (Thread 0x7ff317fa2700 (LWP 2125)):
        #0  0x00007ff369cf8d84 in pthread_cond_wait@@GLIBC_2.3.2 () from /lib/x86_64-linux-gnu/libpthread.so.0
        #1  0x0000000000fc6453 in TCondVar::WaitD(TMutex&, TInstant) () at /place/sandbox-data/srcdir/arcadia_cache_1/util/system/condvar.cpp:104
        #2  0x0000000000fd612b in TMtpQueue::TImpl::DoExecute() () at /place/sandbox-data/srcdir/arcadia_cache_1/util/system/condvar.h:33
        """
        if not stack:
            return

        output = []

        thread_id = stack[0][0]
        for frame in stack:
            frame_no = frame[1]
            func = frame[3] if frame[3] else '??'
            line = frame[4] if frame[4] else '??'
            line_no = ':' + frame[5] if frame[5] else ''
            output.append("#{}  0x0 in {} () from {}{}".format(frame_no, func, line, line_no))

        # generate fake thread id for gdb: active thread (that is coredumped)
        # should be 1st one, marked as "Thread 1"

        if thread_id == active_thread_id:
            gdb_thread_id = '1'
        else:
            gdb_thread_id = str(len(threads) + 1)

        output.insert(0, "Thread {} (Thread {})".format(gdb_thread_id, hex(int(thread_id))))
        output.append('')
        threads.append(output)

    fake_thread = None
    active_thread_id = '0'
    thread_id = ''
    stack = []

    for line in minidump_lines:
        line = line.strip()
        if line.startswith('OS|'):
            continue
        if line.startswith('CPU|'):
            continue
        if line.startswith('Crash|'):
            # detect active thread id
            # "Crash|SIGSEGV|0x452e|0"
            crash = line.split('|')
            signal = crash[1]
            fake_thread = [
                'Program terminated with signal ' + signal,
                '',
            ]
            active_thread_id = crash[3]
            continue
        if line.startswith('Module|'):
            continue

        if not line:
            continue

        frame = line.split('|')
        if frame[0] != thread_id:
            process_thread(stack, active_thread_id)
            stack = []
            thread_id = frame[0]
        stack.append(frame)

        """
        OS|Linux|0.0.0 Linux 3.10.69-25 #28 SMP Fri Feb 20 15:46:36 MSK 2015 x86_64
        CPU|amd64|family 6 model 45 stepping 7|32
        Crash|SIGSEGV|0x452e|0
        Module|srch-base-7332||srch-base-7332|4EB3424E24B86D22FABA35D0F8D672770|0x00400000|0x1c7c6fff|1

        0|0|libpthread-2.15.so|pthread_join|/build/buildd/eglibc-2.15/nptl/pthread_join.c|89|0x15
        0|1|libpthread-2.15.so||||0x9070
        0|2|srch-base-7332|TThread::Join|/place/sandbox-data/srcdir/arcadia_cache_3/util/system/thread.cpp|153|0x5
        0|3|srch-base-7332|THttpServer::Wait|/place/sandbox-data/srcdir/arcadia_cache_3/library/http/server/http.cpp|191|0x5
        0|4|srch-base-7332|THttpService::Run|/place/sandbox-data/srcdir/arcadia_cache_3/search/daemons/httpsearch/httpsearch.cpp|278|0x5
        0|5|srch-base-7332|RunPureMain|/place/sandbox-data/srcdir/arcadia_cache_3/search/daemons/httpsearch/httpsearch.cpp|347|0x12
        0|6|libc-2.15.so|__libc_start_main|/build/buildd/eglibc-2.15/csu/libc-start.c|226|0x17
        0|7|srch-base-7332||||0x69e320
        0|8|srch-base-7332|_init|||0x1130
        1|0|libpthread-2.15.so||||0xbd84

        """
    process_thread(stack, active_thread_id)
    threads.append(fake_thread)
    threads.reverse()
    return threads


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: minidump2core.py <minidump.txt>")
        sys.exit(1)

    print_minidump_as_core(
        file_name=sys.argv[1],
    )
