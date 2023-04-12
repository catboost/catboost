import argparse
import tempfile
import shlex
import subprocess


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split-factor", type=int, default=0)
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    return parser.parse_args()


def list_tests(binary):
    with tempfile.NamedTemporaryFile() as tmpfile:
        cmd = [binary, "--list-verbose", "--list-path", tmpfile.name]
        subprocess.check_call(cmd)

        with open(tmpfile.name) as afile:
            lines = afile.read().strip().split("\n")
            lines = [x.strip() for x in lines]
            return [x for x in lines if x]


def get_shard_tests(args):
    test_names = list_tests(args.command[0])
    test_names = sorted(test_names)

    chunk_size = len(test_names) // args.split_factor
    not_used = len(test_names) % args.split_factor
    shift = chunk_size + (args.shard < not_used)
    start = chunk_size * args.shard + min(args.shard, not_used)
    end = start + shift
    return [] if end > len(test_names) else test_names[start:end]


def get_shard_cmd_args(args):
    return ["+{}".format(x) for x in get_shard_tests(args)]


def main():
    args = parse_args()

    if args.split_factor:
        shard_cmd = get_shard_cmd_args(args)
        if shard_cmd:
            cmd = args.command + shard_cmd
        else:
            print("No tests for {} shard".format(args.shard))
            return 0
    else:
        cmd = args.command

    rc = subprocess.call(cmd)
    if rc:
        print("Some tests failed. To reproduce run: {}".format(shlex.join(cmd)))
    return rc


if __name__ == "__main__":
    exit(main())
