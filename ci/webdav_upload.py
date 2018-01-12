import os
import sys
import glob

import webdav.client

client = webdav.client.Client({
    "webdav_hostname": "https://webdav.yandex.ru",
    "webdav_login":    os.environ["WEBDAV_LOGIN"],
    "webdav_password": os.environ["WEBDAV_PASSWORD"]
})

if os.name == "nt":
    client.default_options["SSL_VERIFYPEER"] = 0

base_dir = "webdav_test"

if "APPVEYOR_BUILD_NUMBER" in os.environ:
    work_dir = base_dir + "/" + "appveyor_" + os.environ["APPVEYOR_BUILD_NUMBER"]
elif "TRAVIS_BUILD_NUMBER" in os.environ:
    work_dir = base_dir + "/" + "travis_" + os.environ["TRAVIS_BUILD_NUMBER"]
else:
    work_dir = base_dir + "/" + "unknown"

if not client.check(work_dir):
    client.mkdir(work_dir)

for path in sys.argv[1:]:
    if "*" in path:
        paths = glob.glob(path)
    else:
        paths = [path]
    for p in paths:
        client.upload(
            work_dir + "/" + os.path.basename(p),
            p
        )
