import sys
import threading
import six

from six.moves import queue


def par_apply(seq, func, thr_num, join_polling=None):
    if thr_num < 2:
        for x in seq:
            yield func(x)

        return

    in_q = queue.Queue()
    out_q = queue.Queue()

    def enumerate_blocks():
        n = 0

        for b in seq:
            yield n, [b]
            n += 1

        yield n, None

    def iter_out():
        n = 0
        d = {}

        while True:
            if n in d:
                r = d[n]
                del d[n]
                n += 1

                yield r
            else:
                res = out_q.get()

                d[res[0]] = res

    out_iter = iter_out()

    def wait_block():
        for x in out_iter:
            return x

    def iter_compressed():
        p = 0

        for n, b in enumerate_blocks():
            in_q.put((n, b))

            while n > p + (thr_num * 2):
                p, b, c = wait_block()

                if not b:
                    return

                yield p, c

        while True:
            p, b, c = wait_block()

            if not b:
                return

            yield p, c

    def proc():
        while True:
            data = in_q.get()

            if data is None:
                return

            n, b = data

            if b:
                try:
                    res = (func(b[0]), None)
                except Exception:
                    res = (None, sys.exc_info())
            else:
                res = (None, None)

            out_q.put((n, b, res))

    thrs = [threading.Thread(target=proc) for i in range(0, thr_num)]

    for t in thrs:
        t.start()

    try:
        for p, c in iter_compressed():
            res, err = c

            if err:
                six.reraise(*err)

            yield res
    finally:
        for t in thrs:
            in_q.put(None)

        for t in thrs:
            if join_polling is not None:
                while True:
                    t.join(join_polling)
                    if not t.is_alive():
                        break
            else:
                t.join()
