import sys


if __name__ == "__main__":

    file_path = sys.argv[1]
    text = sys.argv[2]
    with open(file_path, "a") as f:
        print >>f, text
