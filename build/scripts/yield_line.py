import sys


if __name__ == '__main__':
    pos = sys.argv.index('--')

    with open(sys.argv[pos + 1], 'a') as f:
        f.write(' '.join(sys.argv[pos + 2 :]) + '\n')
