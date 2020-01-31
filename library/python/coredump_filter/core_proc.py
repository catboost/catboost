#!/usr/bin/env python
# coding: utf-8

from __future__ import print_function

import os
import sys
import re
import datetime
import six


MY_PATH = os.path.dirname(os.path.abspath(__file__))
CORE_PROC_VERSION = "0.1.006"

ARCADIA_ROOT_SIGN = '$S/'


class SourceRoot(object):
    patterns = [
        '/util/system/',
        '/util/generic/',
        '/library/',
        '/search/',
    ]

    root = None
    # Distbuild produces the following source paths, so we will count slashes
    root_slash_count = None

    @staticmethod
    def detect(source):
        if SourceRoot.root is not None:
            return

        for pattern in SourceRoot.patterns:
            pos = source.find(pattern)

            if pos > 0:
                root = source[0:pos + 1]
                SourceRoot.root = root
                SourceRoot.root_slash_count = root.count('/') - 1
                break

    @staticmethod
    def crop(source):
        # when traceback contains only ??, source root cannot be detected

        # first of all, cut by slash count
        if SourceRoot.root_slash_count is not None:
            for i in six.moves.range(0, SourceRoot.root_slash_count):
                slash_pos = source.find('/')
                if slash_pos >= 0:
                    source = source[slash_pos + 1:]
            return ARCADIA_ROOT_SIGN + source

        if SourceRoot.root is None:
            return ''

        return source.replace(SourceRoot.root, ARCADIA_ROOT_SIGN)


def highlight_func(s):
    return s \
        .replace('=', '<span class="symbol">=</span>') \
        .replace('(', '<span class="symbol">(</span>') \
        .replace(')', '<span class="symbol">)</span>')


class Frame:
    def __init__(self, frame_no=None, addr='', func='', source=''):
        self.frame_no = frame_no
        self.addr = addr
        self.func = func
        self.source_no = ''
        self.func_name = ''

        SourceRoot.detect(source)

        self.source = source

        m = re.match(r'(.*):(\d+)', source)
        if m:
            self.source = m.group(1)
            self.source_no = m.group(2)

        m = re.match(r'(.*) \(.*\)', self.func)
        if m:
            self.func_name = m.group(1)

    def __str__(self):
        return "{0:3}\t{1:20}\t{2:50}\t{3}".format(
            self.frame_no,
            self.addr,
            self.func,
            self.source,
        )

    def fingerprint(self):
        return self.func_name

    def cropped_source(self):
        return SourceRoot.crop(self.source)

    def html(self):
        source = self.cropped_source()
        source_fmt = ''
        if self.source_no:
            source_fmt = ' +<span class="source-no">{}</span>'.format(self.source_no)

        # '<span class="addr">{1}</span>'  # unused
        return (
            '<span class="frame">{frame}</span>'
            '<span class="func">{func}</span> '
            '<span class="source">{source}</span>{source_fmt}'.format(
                frame=self.frame_no,
                addr=self.addr,  # not used
                func=highlight_func(self.func.replace('&', '&amp;').replace('<', '&lt;')),
                source=source,
                source_fmt=source_fmt,
            )
        )


class Stack:
    # priority classes
    LOW_IMPORTANT = 25
    DEFAULT_IMPORTANT = 50
    SUSPICIOUS_IMPORTANT = 75
    MAX_IMPORTANT = 100

    fingerprint_blacklist = [
        # bottom frames
        'raise',
        'abort',
        '__gnu_cxx::__verbose_terminate_handler',
        '_cxxabiv1::__terminate',
        'std::terminate',
        '__cxxabiv1::__cxa_throw',
        # top frames
        'start_thread',
        'clone',
        '??',
    ]

    suspicious_functions = [
        'CheckedDelete',
        'NPrivate::Panic',
        'abort',
        'close_all_fds',
        '__cxa_throw',
    ]

    low_important_functions_eq = [
        'poll ()',
        'recvfrom ()',
        'pthread_join ()',
    ]

    low_important_functions_match = [
        'TCommonSockOps::SendV',
        'WaitD (',
        'SleepT (',
        'Join (',
        'epoll_wait',
        'nanosleep',
        'pthread_cond_wait',
        'pthread_cond_timedwait',
    ]

    def __init__(self, lines, thread_ptr=0, thread_id=None, stream=None):
        self.lines = lines
        self.thread_ptr = thread_ptr
        self.thread_id = thread_id
        self.frames = []
        self.important = Stack.DEFAULT_IMPORTANT
        if thread_id == 1:
            self.important = Stack.MAX_IMPORTANT
        self.fingerprint_hash = None
        self.stream = stream

    def low_important(self):
        return self.important <= Stack.LOW_IMPORTANT

    def check_importance(self, frame):
        # raised priority cannot be lowered
        if self.important > self.DEFAULT_IMPORTANT:
            return

        # detect suspicious stacks
        for name in Stack.suspicious_functions:
            if name in frame.func:
                self.important = Stack.SUSPICIOUS_IMPORTANT
                return

        for name in Stack.low_important_functions_eq:
            if name == frame.func:
                self.important = Stack.LOW_IMPORTANT

        for name in Stack.low_important_functions_match:
            if name in frame.func:
                self.important = Stack.LOW_IMPORTANT

    def push_frame(self, frame):
        self.check_importance(frame)
        # ignore duplicated frames
        if len(self.frames) and self.frames[-1].frame_no == frame.frame_no:
            return
        self.frames.append(frame)

    def parse(self):
        """
        Parse one stack
        """
        for line in self.lines:

            # #6  0x0000000001d9203e in NAsio::TIOService::TImpl::Run (this=0x137b1ec00) at /place/
            # sandbox-data/srcdir/arcadia_cache/library/neh/asio/io_service_impl.cpp:77
            m = re.match(r'#(\d+)[ \t]+(0x[0-9a-f]+) in (.*) at (/.*)', line)
            if m:
                self.push_frame(Frame(
                    frame_no=m.group(1),
                    addr=m.group(2),
                    func=m.group(3),
                    source=m.group(4)
                ))
                continue

            # #5 TCondVar::WaitD (this=this@entry=0x10196b2b8, mutex=..., deadLine=..., deadLine@entry=...)
            # at /place/sandbox-data/srcdir/arcadia_cache/util/system/condvar.cpp:150
            m = re.match(r'#(\d+)[ \t]+(.*) at (/.*)', line)
            if m:
                self.push_frame(Frame(
                    frame_no=m.group(1),
                    func=m.group(2),
                    source=m.group(3)
                ))
                continue

            # #0  0x00007faf8eb31d84 in pthread_cond_wait@@GLIBC_2.3.2 ()
            # from /lib/x86_64-linux-gnu/libpthread.so.0
            m = re.match(r'#(\d+)[ \t]+(0x[0-9a-f]+) in (.*) from (.*)', line)
            if m:
                self.push_frame(Frame(
                    frame_no=m.group(1),
                    addr=m.group(2),
                    func=m.group(3),
                    source=m.group(4)
                ))
                continue

            m = re.match(r'#(\d+)[ \t]+(0x[0-9a-f]+) in (.*)', line)
            if m:
                self.push_frame(Frame(
                    frame_no=m.group(1),
                    addr=m.group(2),
                    func=m.group(3)
                ))
                continue

            self.bad_frame(line)

    def bad_frame(self, line):
        # print("BAD:", line)
        pass

    def debug(self):
        if self.low_important():
            return

        for f in self.frames:
            self.stream.write(f + '\n')
        self.stream.write('----------------------------- DEBUG END\n')

    def html(self, same_hash=False, same_count=1):
        pre_class = "important-" + str(self.important)
        if same_hash:
            pre_class += " same-hash"

        self.stream.write('<pre class="{0}">'.format(pre_class))
        if not same_hash:
            self.stream.write('<a name="stack{0}"></a>'.format(self.hash()))

        self.stream.write(
            '<span class="hash"><a href="#stack{0}">#{0}</a>, {1} stack(s) with same hash</span>\n'.format(
                self.hash(), same_count,
            )
        )

        for f in self.frames:
            self.stream.write(f.html() + '\n')
        self.stream.write('</pre>\n')

    def fingerprint(self):
        """
        Stack fingerprint: concatenation of non-common stack frames
        """
        # if self.low_important():
        #    return ""

        stack_fp = ""

        for f in self.frames:
            fp = f.fingerprint()
            if len(fp) == 0:
                continue
            if fp in Stack.fingerprint_blacklist:
                continue

            stack_fp += fp + "\n"

        stack_fp = stack_fp.strip()
        return stack_fp

    def hash(self):
        """
        Entire stack hash for merging same stacks
        """
        if self.fingerprint_hash is None:
            self.fingerprint_hash = hash(self.fingerprint())
        return self.fingerprint_hash


def get_jquery_path():
    if getattr(sys, 'is_standalone_binary', False):
        return '/contrib/libs/jquery_data/jquery.min.js'

    cur_dir_path = MY_PATH + '/jquery-1.7.1.min.js'
    if os.path.exists(cur_dir_path):
        return cur_dir_path

    same_repo_path = MY_PATH + '/../../contrib/libs/jquery/1.7.1/jquery-1.7.1.min.js'
    if os.path.exists(same_repo_path):
        return same_repo_path

    # finally, try to download it
    os.system("wget 'http://yandex.st/jquery/1.7.1/jquery.min.js' -O " + cur_dir_path)
    if os.path.exists(cur_dir_path):
        return cur_dir_path

    raise Exception("Cannot find jquery. ")


def _file_contents(file_name):
    if getattr(sys, 'is_standalone_binary', False):
        import __res

        for prefix in ['/coredump_filter_data/', '']:
            res_name = prefix + file_name
            data = __res.find(res_name)
            if data is not None:
                return data
        raise IOError("Failed to find resource: " + file_name)
    else:
        if not os.path.exists(file_name):
            file_name = os.path.join(MY_PATH, file_name)
        with open(file_name) as f:
            return f.read()


def html_prolog(stream):
    prolog = _file_contents('prolog.html')

    stream.write(prolog.format(
        style=_file_contents('styles.css'),
        jquery_js=_file_contents(get_jquery_path()),
        coredump_js=_file_contents('core_proc.js'),
        version=CORE_PROC_VERSION,
        timestamp=datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    ))


def html_epilog(stream):
    stream.write(_file_contents('epilog.html'))


def filter_stackdump(file_name, use_fingerprint=False, sandbox_failed_task_id=None, stream=None):
    stack_lines = []
    main_info = []
    stacks = []
    stack_detected = False
    thread_id = None
    stream = stream or sys.stdout

    sandbox_task_id = None

    with open(file_name) as f:
        for line in f:
            line = line.strip()
            if len(line) == 0:
                continue

            if '[New LWP' in line:
                continue

            # see test2
            if '[New Thread' in line:
                continue

            if '[Thread debugging' in line:
                continue

            if 'Using host libthread_db library' in line:
                continue

            if line.startswith('warning:'):
                continue

            if '[New process' in line:
                continue

            if 'Core was generated' in line:
                m = re.match('.*/[0-9a-f]/[0-9a-f]/([0-9]+)/.*', line)
                if m:
                    sandbox_task_id = int(m.group(1))

            # [Switching to thread 55 (Thread 0x7f100a94c700 (LWP 21034))]
            # Thread 584 (Thread 0x7ff363c03700 (LWP 2124)):

            # see test2 and test3
            tm = re.match('.*[Tt]hread (\d+) .*', line)
            if tm:
                stack_detected = True
                # TODO: thread_ptr
                if len(stack_lines) > 0:
                    stack = Stack(lines=stack_lines, thread_id=thread_id, stream=stream)
                    stacks.append(stack)

                stack_lines = []
                thread_id = int(tm.group(1))
                continue

            if stack_detected:
                stack_lines.append(line)
            else:
                main_info.append(line)

    # parse last stack
    stack = Stack(lines=stack_lines, thread_id=thread_id, stream=stream)
    stacks.append(stack)

    for stack in stacks:
        stack.parse()
        # stack.debug()

    if use_fingerprint:
        for stack in stacks:
            stream.write(stack.fingerprint() + '\n')
            stream.write('--------------------------------------\n')
    else:
        html_prolog(stream)

        if sandbox_task_id is not None:
            stream.write(
                '<div style="padding-top: 6px; font-size: 18px; font-weight: bold;">Coredumped binary build task: ' +
                '<a href="https://sandbox.yandex-team.ru/sandbox/tasks/view?task_id={0}">{0}</a></div>\n'.format(
                    sandbox_task_id)
            )

        if sandbox_failed_task_id is not None:
            stream.write(
                '<div style="padding-top: 6px; font-size: 18px; font-weight: bold;">Sandbox failed task: ' +
                '<a href="https://sandbox.yandex-team.ru/sandbox/tasks/view?task_id={0}">{0}</a></div>\n'.format(
                    sandbox_failed_task_id)
            )

        pre_class = ""
        stream.write('<pre class="{0}">\n'.format(pre_class))
        for line in main_info:
            stream.write(line.replace('&', '&amp;').replace('<', '&lt;') + '\n')
        stream.write('</pre>\n')

        sorted_stacks = sorted(stacks, key=lambda x: (x.important, x.fingerprint()), reverse=True)
        prev_hash = None
        all_hash_stacks = []
        cur_hash_stacks = []
        for stack in sorted_stacks:
            same_hash = stack.hash() == prev_hash
            if not same_hash:
                if len(cur_hash_stacks) > 0:
                    all_hash_stacks.append(cur_hash_stacks)
                cur_hash_stacks = [
                    stack
                ]
            else:
                cur_hash_stacks.append(stack)
            prev_hash = stack.hash()
        # push last
        if len(cur_hash_stacks) > 0:
            all_hash_stacks.append(cur_hash_stacks)

        for cur_hash_stacks in all_hash_stacks:
            same_hash = False
            for stack in cur_hash_stacks:
                stack.html(same_hash=same_hash, same_count=len(cur_hash_stacks))
                same_hash = True

        html_epilog(stream)


if __name__ == '__main__':
    if len(sys.argv) < 2:
        sys.stderr.write(
            """Traceback filter "Tri Korochki"
https://wiki.yandex-team.ru/development/poisk/arcadia/devtools/coredumpfilter/
Usage:
    core_proc.py <traceback.txt> [-f|--fingerprint]
    core_proc.py -v|--version
"""
        )
        sys.exit(1)

    if sys.argv[1] == '--version' or sys.argv[1] == '-v':
        if os.system("svn info 2>/dev/null | grep '^Revision'") != 0:
            print(CORE_PROC_VERSION)
        sys.exit(0)

    sandbox_failed_task_id = None

    use_fingerprint = False
    if len(sys.argv) >= 3:
        if sys.argv[2] == '-f' or sys.argv[2] == '--fingerprint':
            use_fingerprint = True
        sandbox_failed_task_id = sys.argv[2]

    filter_stackdump(
        file_name=sys.argv[1],
        use_fingerprint=use_fingerprint,
        sandbox_failed_task_id=sandbox_failed_task_id,
    )
