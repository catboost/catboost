foo: int = 4
(foo): int = 3
((foo)): int = 3
foo.bar: int
foo[3]: int


def glob():
    global x
    y: foo = x


def c():
    a = 3

    def d():
        class X():
            nonlocal a


def x():
    a = 3

    def y():
        nonlocal a


def x():
    def y():
        nonlocal a

    a = 3


def x():
    a = 3

    def y():
        class z():
            nonlocal a


a = *args, *args
error[(*args, *args)] = 3
*args, *args
