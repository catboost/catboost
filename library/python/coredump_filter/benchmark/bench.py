#!/usr/bin/env python

import logging
import threading
import random
import Queue
import requests
import time

QUEUE = Queue.Queue()
REQUEST_COUNT = 10
THREAD_COUNT = 10

log = logging.getLogger("bench")
log.setLevel(logging.INFO)
log_handler = logging.StreamHandler()
log_handler.setLevel(logging.INFO)
log_formatter = logging.Formatter("%(asctime)s|%(levelname)s: %(message)s")
log_handler.setFormatter(log_formatter)
log.addHandler(log_handler)


def worker():
    while True:
        try:
            req_time = time.time()
            post_body = QUEUE.get()
            #print "Posing a core..."
            req = requests.post(
                'http://arachnid09.search.yandex.net:37771/corecomes',
                data=post_body,
                #timeout=1,
            )
            log.info("ANSWERED in %d seconds with '%s'", time.time() - req_time, req.text[0:30].strip())
        except:
            pass
        finally:
            QUEUE.task_done()


post_body = open('core.bench.txt').read()
# run threads
for i in range(0, THREAD_COUNT):
    log.info("Run thread %d", i)
    thread = threading.Thread(target=worker)
    thread.daemon = True
    thread.start()

start_time = time.time()

for i in xrange(0, REQUEST_COUNT):
    post_unique = post_body.replace('@@UNIQUE@', 'unique' + str(random.randint(0, 100000)))
    log.info("Put %d", i)
    QUEUE.put(post_unique)

log.info("Waiting for finish")
QUEUE.join()
time_delta = time.time() - start_time

print "RPS: ", REQUEST_COUNT / float(time_delta)
