from __future__ import absolute_import

import re
import sys
import copy
import logging

from . import tools
from datetime import date, datetime

import enum
import six

logger = logging.getLogger(__name__)
MDS_URI_PREFIX = 'https://storage.yandex-team.ru/get-devtools/'


def _do_apply(func, value, apply_to_keys, value_path):
    if value_path is None:
        value_path = []

    if isinstance(value, list) or isinstance(value, tuple):
        res = []
        for ind, item in enumerate(value):
            path = copy.copy(value_path)
            path.append(ind)
            res.append(_do_apply(func, item, apply_to_keys, path))
    elif isinstance(value, dict):
        if is_external(value):
            # this is a special serialized object pointing to some external place
            res = func(value, value_path)
        else:
            res = {}
            for key, val in sorted(value.items(), key=lambda dict_item: dict_item[0]):
                path = copy.copy(value_path)
                path.append(key)
                res[_do_apply(func, key, apply_to_keys, path) if apply_to_keys else key] = _do_apply(func, val, apply_to_keys, path)
    else:
        res = func(value, value_path)
    return res


def apply(func, value, apply_to_keys=False):
    """
    Applies func to every possible member of value
    :param value: could be either a primitive object or a complex one (list, dicts)
    :param func: func to be applied
    :return:
    """
    return _do_apply(func, value, apply_to_keys, None)


def is_coroutine(val):
    if sys.version_info[0] < 3:
        return False
    else:
        import asyncio
        return asyncio.iscoroutinefunction(val) or asyncio.iscoroutine(val)


def serialize(value):
    """
    Serialize value to json-convertible object
    Ensures that all components of value can be serialized to json
    :param value: object to be serialized
    """
    def _serialize(val, _):
        if val is None:
            return val
        if isinstance(val, six.string_types) or isinstance(val, bytes):
            return tools.to_utf8(val)
        if isinstance(val, enum.Enum):
            return str(val)
        if isinstance(val, six.integer_types) or type(val) in [float, bool]:
            return val
        if is_external(val):
            return dict(val)
        if isinstance(val, (date, datetime)):
            return repr(val)
        if is_coroutine(val):
            return None
        raise ValueError("Cannot serialize value '{}' of type {}".format(val, type(val)))
    return apply(_serialize, value, apply_to_keys=True)


def is_external(value):
    return isinstance(value, dict) and "uri" in value.keys()


class ExternalSchema(object):
    File = "file"
    SandboxResource = "sbr"
    Delayed = "delayed"
    HTTP = "http"


class CanonicalObject(dict):
    def __iter__(self):
        raise TypeError("Iterating canonical object is not implemented")


def canonical_path(path):
    return path.replace('\\', '/')


class ExternalDataInfo(object):

    def __init__(self, data):
        assert is_external(data)
        self._data = data

    def __str__(self):
        type_str = "File" if self.is_file else "Sandbox resource"
        return "{}({})".format(type_str, self.path)

    def __repr__(self):
        return str(self)

    @property
    def uri(self):
        return self._data["uri"]

    @property
    def checksum(self):
        return self._data.get("checksum")

    @property
    def is_file(self):
        return self.uri.startswith(ExternalSchema.File)

    @property
    def is_sandbox_resource(self):
        return self.uri.startswith(ExternalSchema.SandboxResource)

    @property
    def is_delayed(self):
        return self.uri.startswith(ExternalSchema.Delayed)

    @property
    def is_http(self):
        return self.uri.startswith(ExternalSchema.HTTP)

    @property
    def path(self):
        if self.uri.count("://") != 1:
            logger.error("Invalid external data uri: '%s'", self.uri)
            return self.uri
        _, path = self.uri.split("://")
        return path

    def get_mds_key(self):
        assert self.is_http
        m = re.match(re.escape(MDS_URI_PREFIX) + r'(.*?)($|#)', self.uri)
        if m:
            return m.group(1)
        raise AssertionError("Failed to extract mds key properly from '{}'".format(self.uri))

    @property
    def size(self):
        return self._data.get("size")

    def serialize(self):
        return self._data

    @classmethod
    def _serialize(cls, schema, path, checksum=None, attrs=None):
        res = CanonicalObject({"uri": "{}://{}".format(schema, path)})
        if checksum:
            res["checksum"] = checksum
        if attrs:
            res.update(attrs)
        return res

    @classmethod
    def serialize_file(cls, path, checksum=None, diff_tool=None, local=False, diff_file_name=None, diff_tool_timeout=None, size=None):
        attrs = {}
        if diff_tool:
            attrs["diff_tool"] = diff_tool
        if local:
            attrs["local"] = local
        if diff_file_name:
            attrs["diff_file_name"] = diff_file_name
        if diff_tool_timeout:
            attrs["diff_tool_timeout"] = diff_tool_timeout
        if size is not None:
            attrs["size"] = size
        path = canonical_path(path)
        return cls._serialize(ExternalSchema.File, path, checksum, attrs=attrs)

    @classmethod
    def serialize_resource(cls, id, checksum=None):
        return cls._serialize(ExternalSchema.SandboxResource, id, checksum)

    @classmethod
    def serialize_delayed(cls, upload_id, checksum):
        return cls._serialize(ExternalSchema.Delayed, upload_id, checksum)

    def get(self, key, default=None):
        return self._data.get(key, default)
