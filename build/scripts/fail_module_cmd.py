import sys


if __name__ == '__main__':
    assert len(sys.argv) == 2, 'Unexpected number of arguments...'
    sys.stderr.write(
        'Error: module command for target [[bad]]{}[[rst]] was not executed due to build graph configuration errors...\n'.format(
            sys.argv[1]
        )
    )
    sys.exit(1)
