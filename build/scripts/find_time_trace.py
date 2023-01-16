import os
import sys

# /scripts/find_time_trace.py <object_file> <destination>
# clang generates `-ftime-trace` output file path based on main output file path


def main():
    assert len(sys.argv) == 3
    obj_path = sys.argv[1]
    trace_path = sys.argv[2]
    orig_trace_path = obj_path.rpartition('.o')[0] + '.json'
    os.rename(orig_trace_path, trace_path)


if __name__ == '__main__':
    main()
