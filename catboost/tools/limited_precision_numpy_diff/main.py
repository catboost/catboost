import argparse
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--rtol', default=1.e-5)
    parser.add_argument('--atol', default=1.e-8)
    parser.add_argument('npyArrayPath1')
    parser.add_argument('npyArrayPath2')
    return parser.parse_args()


def main():
    args = parse_args()

    array1 = np.load(args.npyArrayPath1)
    array2 = np.load(args.npyArrayPath2)

    assert np.allclose(array1, array2, args.rtol, args.atol)


if __name__ == "__main__":
    main()
