import os, sys

# Explicitly enable local imports
# Don't forget to add imported scripts to inputs of the calling command!
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import process_command_files as pcf


class ProcessWholeArchiveOption:
    def __init__(self, arch, peers=None, libs=None):
        self.arch = arch.upper()
        self.peers = {x: 0 for x in peers} if peers else None
        self.libs = {x: 0 for x in libs} if libs else None
        self.start_wa_marker = '--start-wa'
        self.end_wa_marker = '--end-wa'

    def _match_peer_lib(self, arg, ext):
        key = None
        if arg.endswith(ext):
            key = os.path.dirname(arg)
        return key if key and self.peers and key in self.peers else None

    def _match_lib(self, arg):
        return arg if self.libs and arg in self.libs else None

    def _process_arg(self, arg, ext='.a'):
        peer_key = self._match_peer_lib(arg, ext)
        lib_key = self._match_lib(arg)
        if peer_key:
            self.peers[peer_key] += 1
        if lib_key:
            self.libs[lib_key] += 1
        return peer_key if peer_key else lib_key

    def _check_peers(self):
        if self.peers:
            for key, value in self.peers.items():
                assert value > 0, '"{}" specified in WHOLE_ARCHIVE() macro is not used on link command'.format(key)

    def _construct_cmd_apple(self, args):
        force_load_flag = '-Wl,-force_load,'
        is_inside_wa_markers = False

        cmd = []
        for arg in args:
            if arg.startswith(force_load_flag):
                cmd.append(arg)
            elif arg == self.start_wa_marker:
                is_inside_wa_markers = True
            elif arg == self.end_wa_marker:
                is_inside_wa_markers = False
            elif is_inside_wa_markers:
                cmd.append(force_load_flag + arg)
            else:
                key = self._process_arg(arg)
                cmd.append(force_load_flag + arg if key else arg)

        self._check_peers()

        return cmd

    def _construct_cmd_win(self, args):
        whole_archive_prefix = '/WHOLEARCHIVE:'
        is_inside_wa_markers = False

        def add_prefix(arg, need_check_peers_and_libs):
            key = self._process_arg(arg, '.lib') if need_check_peers_and_libs else arg
            return whole_archive_prefix + arg if key else arg

        def add_whole_archive_prefix(arg, need_check_peers_and_libs):
            if not pcf.is_cmdfile_arg(arg):
                return add_prefix(arg, need_check_peers_and_libs)

            cmd_file_path = pcf.cmdfile_path(arg)
            cf_args = pcf.read_from_command_file(cmd_file_path)
            with open(cmd_file_path, 'w') as afile:
                for cf_arg in cf_args:
                    afile.write(add_prefix(cf_arg, need_check_peers_and_libs) + "\n")
            return arg

        cmd = []
        for arg in args:
            if arg == self.start_wa_marker:
                is_inside_wa_markers = True
            elif arg == self.end_wa_marker:
                is_inside_wa_markers = False
            elif is_inside_wa_markers:
                cmd.append(add_whole_archive_prefix(arg, False))
                continue
            elif self.peers or self.libs:
                cmd.append(add_whole_archive_prefix(arg, True))
            else:
                cmd.append(arg)

        self._check_peers()

        return cmd

    def _construct_cmd_linux(self, args):
        whole_archive_flag = '-Wl,--whole-archive'
        no_whole_archive_flag = '-Wl,--no-whole-archive'

        def replace_markers(arg):
            if arg == self.start_wa_marker:
                return whole_archive_flag
            if arg == self.end_wa_marker:
                return no_whole_archive_flag
            return arg

        args = [replace_markers(arg) for arg in args]

        cmd = []
        is_inside_whole_archive = False
        is_whole_archive = False
        # We are trying not to create excessive sequences of consecutive flags
        # -Wl,--no-whole-archive  -Wl,--whole-archive ('externally' specified
        # flags -Wl,--[no-]whole-archive are not taken for consideration in this
        # optimization intentionally)
        for arg in args:
            if arg == whole_archive_flag:
                is_inside_whole_archive = True
                is_whole_archive = False
            elif arg == no_whole_archive_flag:
                is_inside_whole_archive = False
                is_whole_archive = False
            else:
                key = self._process_arg(arg)
                if not is_inside_whole_archive:
                    if key:
                        if not is_whole_archive:
                            cmd.append(whole_archive_flag)
                            is_whole_archive = True
                    elif is_whole_archive:
                        cmd.append(no_whole_archive_flag)
                        is_whole_archive = False

            cmd.append(arg)

        if is_whole_archive:
            cmd.append(no_whole_archive_flag)

        # There can be an empty sequence of archive files between
        # -Wl, --whole-archive and -Wl, --no-whole-archive flags.
        # As a result an unknown option error may occur, therefore to
        # prevent this case we need to remove both flags from cmd.
        # These flags affects only on subsequent archive files.
        if len(cmd) == 2:
            return []

        self._check_peers()

        return cmd

    def construct_cmd(self, args):
        if self.arch in ('DARWIN', 'IOS', 'IOSSIM'):
            return self._construct_cmd_apple(args)

        if self.arch == 'WINDOWS':
            return self._construct_cmd_win(args)

        return self._construct_cmd_linux(args)


def get_whole_archive_peers_and_libs(args):
    remaining_args = []
    peers = []
    libs = []
    peers_flag = '--whole-archive-peers'
    libs_flag = '--whole-archive-libs'

    next_is_peer = False
    next_is_lib = False
    for arg in args:
        if arg == peers_flag:
            next_is_peer = True
        elif arg == libs_flag:
            next_is_lib = True
        elif next_is_peer:
            peers.append(arg)
            next_is_peer = False
        elif next_is_lib:
            libs.append(arg)
            next_is_lib = False
        else:
            remaining_args.append(arg)
    return remaining_args, peers, libs
