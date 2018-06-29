import os
import sys

# /script/move.py <src-1> <tgt-1> <src-2> <tgt-2> ... <src-n> <tgt-n>
# renames src-1 to tgt-1, src-2 to tgt-2, ..., src-n to tgt-n.


def main():
    assert len(sys.argv) % 2 == 1
    for index in range(1, len(sys.argv), 2):
        os.rename(sys.argv[index], sys.argv[index + 1])


if __name__ == '__main__':
    main()
