#!/usr/bin/env python
# coding: utf-8

from __future__ import print_function
import os
import sys
import re
import datetime
from json import JSONEncoder


ARCADIA_ROOT_LINK = "https://a.yandex-team.ru/arc/trunk/arcadia/"

ARCADIA_ROOT_DIRS = [
    "/lib/x86_64-linux-gnu/",
    "/aapi/",
    "/addappter/",
    "/adfox/",
    "/admins/",
    "/ads/",
    "/adv/",
    "/advq/",
    "/afisha/",
    "/afro/",
    "/alet/",
    "/alice/",
    "/analytics/",
    "/antiadblock/",
    "/antirobot/",
    "/apphost/",
    "/april/",
    "/arc/",
    "/arcanum/",
    "/augur/",
    "/aurora/",
    "/autocheck/",
    "/balancer/",
    "/bass/",
    "/billing/",
    "/bindings/",
    "/browser/",
    "/build/",
    "/bunker/",
    "/caas/",
    "/canvas/",
    "/captcha/",
    "/catboost/",
    "/certs/",
    "/ci/",
    "/clickhouse/",
    "/client_analytics/",
    "/cloud/",
    "/cmicot/",
    "/cmnt/",
    "/comdep_analytics/",
    "/commerce/",
    "/contrib/",
    "/crm/",
    "/crowdsourcing/",
    "/crypta/",
    "/cv/",
    "/datacloud/",
    "/datalens/",
    "/data-ui/",
    "/devtools/",
    "/dict/",
    "/direct/",
    "/disk/",
    "/distribution/",
    "/distribution_interface/",
    "/district/",
    "/dj/",
    "/docs/",
    "/douber/",
    "/drive/",
    "/edadeal/",
    "/education/",
    "/entity/",
    "/ether/",
    "/extdata/",
    "/extsearch/",
    "/FactExtract/",
    "/fintech/",
    "/frontend/",
    "/fuzzing/",
    "/games/",
    "/gencfg/",
    "/geobase/",
    "/geoproduct/",
    "/geosuggest/",
    "/geotargeting/",
    "/glycine/",
    "/groups/",
    "/haas/",
    "/health/",
    "/helpdesk/",
    "/hitman/",
    "/home/",
    "/htf/",
    "/hw_watcher/",
    "/hypercube/",
    "/iaas/",
    "/iceberg/",
    "/infra/",
    "/intranet/",
    "/inventori/",
    "/ipreg/",
    "/irt/",
    "/it-office/",
    "/jdk/",
    "/juggler/",
    "/junk/",
    "/jupytercloud/",
    "/kernel/",
    "/keyboard/",
    "/kikimr/",
    "/kinopoisk/",
    "/kinopoisk-ott/",
    "/laas/",
    "/lbs/",
    "/library/",
    "/load/",
    "/locdoc/",
    "/logbroker/",
    "/logfeller/",
    "/mail/",
    "/mapreduce/",
    "/maps/",
    "/maps_adv/",
    "/market/",
    "/mb/",
    "/mds/",
    "/media/",
    "/media-billing/",
    "/media-crm/",
    "/mediapers/",
    "/mediaplanner/",
    "/mediastat/",
    "/media-stories/",
    "/metrika/",
    "/milab/",
    "/ml/",
    "/mlp/",
    "/mlportal/",
    "/mobile/",
    "/modadvert/",
    "/ms/",
    "/mssngr/",
    "/music/",
    "/musickit/",
    "/netsys/",
    "/nginx/",
    "/nirvana/",
    "/noc/",
    "/ofd/",
    "/offline_data/",
    "/opensource/",
    "/orgvisits/",
    "/ott/",
    "/packages/",
    "/partner/",
    "/passport/",
    "/payplatform/",
    "/paysys/",
    "/plus/",
    "/portal/",
    "/portalytics/",
    "/pythia/",
    "/quality/",
    "/quasar/",
    "/razladki/",
    "/regulargeo/",
    "/release_machine/",
    "/rem/",
    "/repo/",
    "/rnd_toolbox/",
    "/robot/",
    "/rtc/",
    "/rtline/",
    "/rtmapreduce/",
    "/rt-research/",
    "/saas/",
    "/samogon/",
    "/samsara/",
    "/sandbox/",
    "/scarab/",
    "/sdc/",
    "/search/",
    "/security/",
    "/semantic-web/",
    "/serp/",
    "/sitesearch/",
    "/skynet/",
    "/smart_devices/",
    "/smarttv/",
    "/smm/",
    "/solomon/",
    "/specsearches/",
    "/speechkit/",
    "/sport/",
    "/sprav/",
    "/statbox/",
    "/strm/",
    "/suburban-trains/",
    "/sup/",
    "/switch/",
    "/talents/",
    "/tasklet/",
    "/taxi/",
    "/taxi_efficiency/",
    "/testenv/",
    "/testpalm/",
    "/testpers/",
    "/toloka/",
    "/toolbox/",
    "/tools/",
    "/tracker/",
    "/traffic/",
    "/transfer_manager/",
    "/travel/",
    "/trust/",
    "/urfu/",
    "/util/",
    "/vcs/",
    "/velocity/",
    "/vendor/",
    "/vh/",
    "/voicetech/",
    "/weather/",
    "/web/",
    "/wmconsole/",
    "/xmlsearch/",
    "/yabs/",
    "/yadoc/",
    "/yandex_io/",
    "/yaphone/",
    "/ydf/",
    "/ydo/",
    "/yp/",
    "/yql/",
    "/ysite/",
    "/yt/",
    "/yweb/",
    "/zen/",
    "/zapravki/",
    "/zen/",
    "/zootopia/",
    "/zora/",
]

MY_PATH = os.path.dirname(os.path.abspath(__file__))
CORE_PROC_VERSION = "0.1.006"

ARCADIA_ROOT_SIGN = '$S/'
SIGNAL_FLAG = 'Program terminated with signal'


# #6  0x0000000001d9203e in NAsio::TIOService::TImpl::Run (this=0x137b1ec00) at /place/
# sandbox-data/srcdir/arcadia_cache/library/neh/asio/io_service_impl.cpp:77
regexp_1 = re.compile(r'#(?P<frame_no>\d+)[ \t]+(?P<addr>0x[0-9a-f]+) in (?P<func>.*) at (?P<source>/.*)')

# #5 TCondVar::WaitD (this=this@entry=0x10196b2b8, mutex=..., deadLine=..., deadLine@entry=...)
# at /place/sandbox-data/srcdir/arcadia_cache/util/system/condvar.cpp:150
regexp_2 = re.compile(r'#(?P<frame_no>\d+)[ \t]+(?P<func>.*) at (?P<source>/.*)')

# #0  0x00007faf8eb31d84 in pthread_cond_wait@@GLIBC_2.3.2 ()
# from /lib/x86_64-linux-gnu/libpthread.so.0
regexp_3 = re.compile(r'#(?P<frame_no>\d+)[ \t]+(?P<addr>0x[0-9a-f]+) in (?P<func>.*) from (?P<source>.*)')

# mvel doesn't provide example :-)
# #10 0x0000000000000000 in ?? ()
regexp_4 = re.compile(r'#(?P<frame_no>\d+)[ \t]+(?P<addr>0x[0-9a-f]+) in (?P<func>.*)')

regexp_all = [regexp_1, regexp_2, regexp_3, regexp_4]


class EncoderStack(JSONEncoder):
    def default(self, o):
        tmp = o.__dict__
        tmp["frames"] = map(lambda x: x.__dict__, tmp["frames"])
        return tmp


class SourceRoot(object):
    root = None
    # Distbuild produces the following source paths, so we will count slashes

    @staticmethod
    def detect(source):
        if not source:
            # For example, regexp_4
            return

        for root_dir in ARCADIA_ROOT_DIRS:
            pos = source.find(root_dir)
            if pos > 0:
                if (SourceRoot.root and len(SourceRoot.root) > pos) or (not SourceRoot.root):
                    SourceRoot.root = source[:pos + 1]

    @staticmethod
    def crop(source):
        if not source:
            return ""
        # when traceback contains only ??, source root cannot be detected

        if SourceRoot.root is not None:
            return source.replace(SourceRoot.root, ARCADIA_ROOT_SIGN, 1)

        return ARCADIA_ROOT_SIGN + source.lstrip("/")


def highlight_func(s):
    return s \
        .replace('=', '<span class="symbol">=</span>') \
        .replace('(', '<span class="symbol">(</span>') \
        .replace(')', '<span class="symbol">)</span>')


class Frame:
    def __init__(self, frame_no=None, addr='', func='', source='', source_no='', func_name=''):
        self.frame_no = frame_no
        self.addr = addr
        self.func = func
        self.source_no = source_no
        self.func_name = func_name
        SourceRoot.root = None
        SourceRoot.detect(source)

        self.source = SourceRoot.crop(source)
        if not source_no:
            m = re.match(r'(.*):(\d+)', source)
            if m:
                self.source = SourceRoot.crop(m.group(1))
                self.source_no = m.group(2)
        if not func_name:
            m = re.match(r'(.*) \(.*\)', self.func)
            if m:
                self.func_name = m.group(1)

    def __str__(self):
        return "{0:3}\t{1:30}\t{2}".format(
            self.frame_no,
            self.func,
            self.source,
        )

    def fingerprint(self):
        return self.func_name

    def cropped_source(self):
        return self.source

    def find_source(self):
        """
        Returns link to arcadia if source is path in arcadia, else just string with path
        :return: pair (source, source_fmt)
        """
        source_fmt = ''
        source = ''
        link = ''
        dirs = self.source.split('/')
        if len(dirs) > 1 and '/{dir}/'.format(dir=dirs[1]) in ARCADIA_ROOT_DIRS:
            link = self.source.replace(ARCADIA_ROOT_SIGN, ARCADIA_ROOT_LINK)
        else:
            source = self.source
        if self.source_no:
            source_fmt = ' +<span class="source-no">{}</span>'.format(self.source_no)
            if link:
                link += "?#L{line}".format(line=self.source_no)

        if link:
            source = '<a href="{link}">{source}</a>'.format(
                link=link,
                source=self.source,
            )
        return source, source_fmt

    def raw(self):
        return u'{frame} {func} {source}'.format(frame=self.frame_no, func=self.func, source=self.source)

    def html(self):
        source, source_fmt = self.find_source()
        return (
            u'<span class="frame">{frame}</span>'
            u'<span class="func">{func}</span> '
            u'<span class="source">{source}</span>{source_fmt}\n'.format(
                frame=self.frame_no,
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

    def __init__(
        self,
        lines,
        thread_ptr=0,
        thread_id=None,
        frames=None,
        important=None,
        stack_fp=None,
        fingerprint_hash=None,
        stream=None,
    ):
        self.lines = lines
        self.thread_ptr = thread_ptr
        self.thread_id = thread_id
        if frames and type(frames[0]) is dict:
            frames = map(lambda x: Frame(**x), frames)
        self.frames = frames or []
        self.important = important or Stack.DEFAULT_IMPORTANT
        if thread_id == 1:
            self.important = Stack.MAX_IMPORTANT
        self.fingerprint_hash = fingerprint_hash
        self.stack_fp = stack_fp
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

            for regexp in regexp_all:
                m = regexp.match(line)
                if m:
                    self.push_frame(Frame(**m.groupdict()))
            self.bad_frame(line)

    def bad_frame(self, line):
        # print("BAD:", line)
        pass

    def debug(self, return_result=False):
        if self.low_important():
            return

        res = ""
        for f in self.frames:
            res += f + "\n"
        res += "----------------------------- DEBUG END\n"
        if return_result:
            return res
        self.stream.write(res)

    def raw(self):
        return "\n".join([frame.raw() for frame in self.frames])

    def html(self, same_hash=False, same_count=1, return_result=False):
        ans = ""
        pre_class = "important-" + str(self.important)
        if same_hash:
            pre_class += " same-hash"

        ans += '<pre class="{0}">'.format(pre_class)
        if not same_hash:
            ans += '<a name="stack{0}"></a>'.format(self.hash())

        ans += '<span class="hash"><a href="#stack{0}">#{0}</a>, {1} stack(s) with same hash</span>\n'.format(
            self.hash(), same_count,
        )

        for f in self.frames:
            ans += f.html()
        ans += '</pre>\n'

        if return_result:
            return ans
        self.stream.write(ans)

    def fingerprint(self, max_num=None):
        """
        Stack fingerprint: concatenation of non-common stack frames
        """

        stack_fp = ""
        len_frames = min((max_num or len(self.frames)), len(self.frames))
        for f in self.frames[: len_frames]:
            fp = f.fingerprint()
            if len(fp) == 0:
                continue
            if fp in Stack.fingerprint_blacklist:
                continue

            stack_fp += fp + "\n"
        stack_fp = stack_fp.strip()
        return stack_fp

    def simple_html(self, num_frames=None):
        if not num_frames:
            num_frames = len(self.frames)
        pre_class = "important-0"
        ans = '<pre class="{0}">'.format(pre_class)
        for i in range(min(len(self.frames), num_frames)):
            ans += self.frames[i].html()
        ans += '</pre>\n'
        return ans

    def __str__(self):
        return "\n".join(map(str, self.frames))

    def hash(self, max_num=None):
        """
        Entire stack hash for merging same stacks
        """
        if self.fingerprint_hash is None:
            self.fingerprint_hash = hash(self.fingerprint(max_num))
        return self.fingerprint_hash


python_frame_regex = re.compile(r'File \"(?P<source>.*)\", line (?P<source_no>\d+), in (?P<func_name>.*)')


def parse_python_traceback(trace):
    trace = trace.replace("/home/zomb-sandbox/client/", "/")
    trace = trace.replace("/home/zomb-sandbox/tasks/", "/sandbox/")
    trace = trace.split("\n")
    exception = trace[-1]  # Will use it later
    trace = trace[1: -1]
    pairs = zip(trace[::2], trace[1::2])
    stack = Stack(lines=[])
    for frame_no, (path, row) in enumerate(pairs):
        m = python_frame_regex.match(path.strip())
        if m:
            frame_args = m.groupdict()
            if not frame_args["source"].startswith("/"):
                frame_args["source"] = "/" + frame_args["source"]
            frame_args["frame_no"] = str(frame_no)
            frame_args["func"] = row.strip()
            stack.push_frame(Frame(**frame_args))
    return [[stack]], [[stack.raw()]], 6


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


def filter_stackdump(
    file_name=None, use_fingerprint=False, sandbox_failed_task_id=None, stream=None, file_lines=None, use_stream=True,
):
    stack_lines = []
    main_info = []
    stacks = []
    stack_detected = False
    thread_id = None
    stream = stream or sys.stdout
    signal = 'signal not found'

    sandbox_task_id = None

    if file_lines is None:
        file_lines = []
        with open(file_name) as f:
            for line in f:
                file_lines.append(line)

    for line in file_lines:
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

        if SIGNAL_FLAG in line:
            signal = line[line.find(SIGNAL_FLAG) + len(SIGNAL_FLAG):].split(',')[0]

        # [Switching to thread 55 (Thread 0x7f100a94c700 (LWP 21034))]
        # Thread 584 (Thread 0x7ff363c03700 (LWP 2124)):

        # see test2 and test3
        tm = re.match(r'.*[Tt]hread (\d+) .*', line)
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

    if use_stream:
        if use_fingerprint:
            for stack in stacks:
                stream.write(stack.fingerprint() + '\n')
                stream.write('--------------------------------------\n')
            return
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
            cur_hash_stacks = [stack, ]
        else:
            cur_hash_stacks.append(stack)
        prev_hash = stack.hash()
    # push last
    if len(cur_hash_stacks) > 0:
        all_hash_stacks.append(cur_hash_stacks)

    if use_stream:
        for cur_hash_stacks in all_hash_stacks:
            same_hash = False
            for stack in cur_hash_stacks:
                stack.html(same_hash=same_hash, same_count=len(cur_hash_stacks))
                same_hash = True

        html_epilog(stream)
    else:
        raw_hash_stacks = [[stack.raw() for stack in common_hash_stacks] for common_hash_stacks in all_hash_stacks]
        return all_hash_stacks, raw_hash_stacks, signal


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
