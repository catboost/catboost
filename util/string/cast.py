print 'static const ui8 SAFE_LENS[4][15] = {'

def nb(n, b):
    if n == 0:
        return [0]

    digits = []

    while n:
        digits.append(int(n % b))
        n /= b

    return digits[::-1]


for p in (1, 2, 4, 8):
    def it1():
        for base in range(2, 17):
            m = 2 ** (8 * p) - 1

            yield len(nb(m, base)) - 1

    print '     {0, 0, ' + ', '.join(str(x) for x in it1()) + '},'

print '};'
