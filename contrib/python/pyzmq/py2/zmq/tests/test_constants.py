# Copyright (C) PyZMQ Developers
# Distributed under the terms of the Modified BSD License.

import json
from unittest import TestCase

import pytest

import zmq

from zmq.utils import constant_names
from zmq.sugar import constants as sugar_constants
from zmq.backend import constants as backend_constants

all_set = set(constant_names.all_names)

class TestConstants(TestCase):
    
    def _duplicate_test(self, namelist, listname):
        """test that a given list has no duplicates"""
        dupes = {}
        for name in set(namelist):
            cnt = namelist.count(name)
            if cnt > 1:
                dupes[name] = cnt
        if dupes:
            self.fail("The following names occur more than once in %s: %s" % (listname, json.dumps(dupes, indent=2)))
    
    def test_duplicate_all(self):
        return self._duplicate_test(constant_names.all_names, "all_names")
    
    def _change_key(self, change, version):
        """return changed-in key"""
        return "%s-in %d.%d.%d" % tuple([change] + list(version))

    def test_duplicate_changed(self):
        all_changed = []
        for change in ("new", "removed"):
            d = getattr(constant_names, change + "_in")
            for version, namelist in d.items():
                all_changed.extend(namelist)
                self._duplicate_test(namelist, self._change_key(change, version))
        
        self._duplicate_test(all_changed, "all-changed")
    
    def test_changed_in_all(self):
        missing = {}
        for change in ("new", "removed"):
            d = getattr(constant_names, change + "_in")
            for version, namelist in d.items():
                key = self._change_key(change, version)
                for name in namelist:
                    if name not in all_set:
                        if key not in missing:
                            missing[key] = []
                        missing[key].append(name)
        
        if missing:
            self.fail(
                "The following names are missing in `all_names`: %s" % json.dumps(missing, indent=2)
            )
    
    def test_no_negative_constants(self):
        for name in sugar_constants.__all__:
            self.assertNotEqual(getattr(zmq, name), sugar_constants._UNDEFINED)
    
    def test_undefined_constants(self):
        all_aliases = []
        for alias_group in sugar_constants.aliases:
            all_aliases.extend(alias_group)
        
        for name in all_set.difference(all_aliases):
            raw = getattr(backend_constants, name)
            if raw == sugar_constants._UNDEFINED:
                self.assertRaises(AttributeError, getattr, zmq, name)
            else:
                self.assertEqual(getattr(zmq, name), raw)
    
    def test_new(self):
        zmq_version = zmq.zmq_version_info()
        for version, new_names in constant_names.new_in.items():
            should_have = zmq_version >= version
            for name in new_names:
                try:
                    value = getattr(zmq, name)
                except AttributeError:
                    if should_have:
                        self.fail("AttributeError: zmq.%s" % name)
                else:
                    if not should_have:
                        self.fail("Shouldn't have: zmq.%s=%s" % (name, value))

    @pytest.mark.skipif(not zmq.DRAFT_API, reason="Only test draft API if built with draft API")
    def test_draft(self):
        zmq_version = zmq.zmq_version_info()
        for version, new_names in constant_names.draft_in.items():
            should_have = zmq_version >= version
            for name in new_names:
                try:
                    value = getattr(zmq, name)
                except AttributeError:
                    if should_have:
                        self.fail("AttributeError: zmq.%s" % name)
                else:
                    if not should_have:
                        self.fail("Shouldn't have: zmq.%s=%s" % (name, value))

    def test_removed(self):
        zmq_version = zmq.zmq_version_info()
        for version, new_names in constant_names.removed_in.items():
            should_have = zmq_version < version
            for name in new_names:
                try:
                    value = getattr(zmq, name)
                except AttributeError:
                    if should_have:
                        self.fail("AttributeError: zmq.%s" % name)
                else:
                    if not should_have:
                        self.fail("Shouldn't have: zmq.%s=%s" % (name, value))

