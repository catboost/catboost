import os
import sys
import time

import webdav.client

client = webdav.client.Client({
    "webdav_hostname": "https://webdav.yandex.ru",
    "webdav_login":    os.environ["WEBDAV_LOGIN"],
    "webdav_password": os.environ["WEBDAV_PASSWORD"]
})

base_dir = "webdav_test"

work_dir = os.path.join(
    base_dir, 
    os.environ.get("TRAVIS_BUILD_NUMBER", "")
)

if not client.check(work_dir):
    client.mkdir(work_dir)

for path in sys.argv[1:]:
    client.upload(
        os.path.join(work_dir, os.path.basename(path)),
        path
    )
