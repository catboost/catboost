#!/usr/bin/env python3
# - * - encoding: UTF-8 - * -

from argparse import ArgumentParser
import random
import sys
import math


def parse_args():
    parser = ArgumentParser(description="")
    parser.add_argument('--range', type=int)
    parser.add_argument('--enum', nargs=2, action="append", metavar=("NAME", "SIZE"))
    parser.add_argument('--namespace', type=str)
    args = parser.parse_args()
    return args


def gen_enum(name, n):
    rg = random.Random(n)
    h1 = list(range(n))
    h2 = list(range(n))
    rg.shuffle(h1)
    rg.shuffle(h2)

    print("enum class %s {" % name)
    for  k, v in zip(h1, h2):
        print("    V%x = 0x%04x," % (k, v))
    print("};")
    print()


def main():
    args = parse_args()

    print("#pragma once\n\n")

    gr = {}
    for name, size in args.enum or []:
        assert name not in gr
        gr[name] = int(size)
    if args.range:
        step = max(int(math.sqrt(args.range)), 1)
        for s in range(args.range, -1, -step):
            gr["EDenseEnum%04d" % s] = s

    if args.namespace:
        print(f"namespace {args.namespace} {{")

    for name, size in sorted(gr.items(), key=lambda kv: -kv[1]):
        gen_enum(name, size)

    if args.namespace:
        print(f"}} // namespace {args.namespace}")

    return 0


if __name__ == '__main__':
    sys.exit(main())
