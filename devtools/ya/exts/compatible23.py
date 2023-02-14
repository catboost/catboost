import base64

import six

if six.PY2:
    base64.encodebytes = base64.encodestring
    base64.decodebytes = base64.decodestring
