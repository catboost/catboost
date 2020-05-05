import sys


if __name__ == "__main__":

    file_path = sys.argv[1]
    with open(file_path, "a") as f:
        for text in sys.argv[2:]:
            print >>f, text
