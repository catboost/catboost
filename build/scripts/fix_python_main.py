import sys


with open(sys.argv[1], 'r') as f:
    data = f.read()

for s in ('if __name__ == \'__main__\'', 'if __name__ == "__main__"'):
    data = data.replace(s, 'def real_main_func()')

with open(sys.argv[2], 'w') as f:
    f.write(data)
