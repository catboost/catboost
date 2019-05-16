"""Tests for scandir.walk(), copied from CPython's tests for os.walk()."""

import os
import shutil
import sys
import unittest

import scandir

import yatest.common

walk_func = scandir.walk

IS_PY3 = sys.version_info >= (3, 0)


class TestWalk(unittest.TestCase):
    testfn = os.path.join(os.path.dirname(__file__), 'temp')

    def test_traversal(self):
        self.testfn = yatest.common.test_output_path('temp')
        # Build:
        #     TESTFN/
        #       TEST1/              a file kid and two directory kids
        #         tmp1
        #         SUB1/             a file kid and a directory kid
        #           tmp2
        #           SUB11/          no kids
        #         SUB2/             a file kid and a dirsymlink kid
        #           tmp3
        #           link/           a symlink to TESTFN.2
        #       TEST2/
        #         tmp4              a lone file
        walk_path = os.path.join(self.testfn, "TEST1")
        sub1_path = os.path.join(walk_path, "SUB1")
        sub11_path = os.path.join(sub1_path, "SUB11")
        sub2_path = os.path.join(walk_path, "SUB2")
        tmp1_path = os.path.join(walk_path, "tmp1")
        tmp2_path = os.path.join(sub1_path, "tmp2")
        tmp3_path = os.path.join(sub2_path, "tmp3")
        link_path = os.path.join(sub2_path, "link")
        t2_path = os.path.join(self.testfn, "TEST2")
        tmp4_path = os.path.join(self.testfn, "TEST2", "tmp4")

        # Create stuff.
        os.makedirs(sub11_path)
        os.makedirs(sub2_path)
        os.makedirs(t2_path)
        for path in tmp1_path, tmp2_path, tmp3_path, tmp4_path:
            f = open(path, "w")
            f.write("I'm " + path + " and proud of it.  Blame test_os.\n")
            f.close()
        has_symlink = hasattr(os, "symlink")
        if has_symlink:
            try:
                if IS_PY3:
                    os.symlink(os.path.abspath(t2_path), link_path, target_is_directory=True)
                else:
                    os.symlink(os.path.abspath(t2_path), link_path)
                sub2_tree = (sub2_path, ["link"], ["tmp3"])
            except NotImplementedError:
                sub2_tree = (sub2_path, [], ["tmp3"])
        else:
            sub2_tree = (sub2_path, [], ["tmp3"])

        # Walk top-down.
        all = list(walk_func(walk_path))
        self.assertEqual(len(all), 4)
        # We can't know which order SUB1 and SUB2 will appear in.
        # Not flipped:  TESTFN, SUB1, SUB11, SUB2
        #     flipped:  TESTFN, SUB2, SUB1, SUB11
        flipped = all[0][1][0] != "SUB1"
        all[0][1].sort()
        self.assertEqual(all[0], (walk_path, ["SUB1", "SUB2"], ["tmp1"]))
        self.assertEqual(all[1 + flipped], (sub1_path, ["SUB11"], ["tmp2"]))
        self.assertEqual(all[2 + flipped], (sub11_path, [], []))
        self.assertEqual(all[3 - 2 * flipped], sub2_tree)

        # Prune the search.
        all = []
        for root, dirs, files in walk_func(walk_path):
            all.append((root, dirs, files))
            # Don't descend into SUB1.
            if 'SUB1' in dirs:
                # Note that this also mutates the dirs we appended to all!
                dirs.remove('SUB1')
        self.assertEqual(len(all), 2)
        self.assertEqual(all[0], (walk_path, ["SUB2"], ["tmp1"]))
        self.assertEqual(all[1], sub2_tree)

        # Walk bottom-up.
        all = list(walk_func(walk_path, topdown=False))
        self.assertEqual(len(all), 4)
        # We can't know which order SUB1 and SUB2 will appear in.
        # Not flipped:  SUB11, SUB1, SUB2, TESTFN
        #     flipped:  SUB2, SUB11, SUB1, TESTFN
        flipped = all[3][1][0] != "SUB1"
        all[3][1].sort()
        self.assertEqual(all[3], (walk_path, ["SUB1", "SUB2"], ["tmp1"]))
        self.assertEqual(all[flipped], (sub11_path, [], []))
        self.assertEqual(all[flipped + 1], (sub1_path, ["SUB11"], ["tmp2"]))
        self.assertEqual(all[2 - 2 * flipped], sub2_tree)

        if has_symlink:
            # Walk, following symlinks.
            for root, dirs, files in walk_func(walk_path, followlinks=True):
                if root == link_path:
                    self.assertEqual(dirs, [])
                    self.assertEqual(files, ["tmp4"])
                    break
            else:
                self.fail("Didn't follow symlink with followlinks=True")

        # Test creating a directory and adding it to dirnames
        sub3_path = os.path.join(walk_path, "SUB3")
        all = []
        for root, dirs, files in walk_func(walk_path):
            all.append((root, dirs, files))
            if 'SUB1' in dirs:
                os.makedirs(sub3_path)
                dirs.append('SUB3')
        all.sort()
        self.assertEqual(os.path.split(all[-1][0])[1], 'SUB3')

    def tearDown(self):
        # Tear everything down.  This is a decent use for bottom-up on
        # Windows, which doesn't have a recursive delete command.  The
        # (not so) subtlety is that rmdir will fail unless the dir's
        # kids are removed first, so bottom up is essential.
        for root, dirs, files in os.walk(self.testfn, topdown=False):
            for name in files:
                os.remove(os.path.join(root, name))
            for name in dirs:
                dirname = os.path.join(root, name)
                if not os.path.islink(dirname):
                    os.rmdir(dirname)
                else:
                    os.remove(dirname)
        os.rmdir(self.testfn)


class TestWalkSymlink(unittest.TestCase):
    temp_dir = os.path.join(os.path.dirname(__file__), 'temp')

    def setUp(self):
        self.temp_dir = yatest.common.test_output_path('temp')
        os.mkdir(self.temp_dir)
        self.dir_name = os.path.join(self.temp_dir, 'dir')
        os.mkdir(self.dir_name)
        open(os.path.join(self.dir_name, 'subfile'), 'w').close()
        self.file_name = os.path.join(self.temp_dir, 'file')
        open(self.file_name, 'w').close()

    def tearDown(self):
        shutil.rmtree(self.temp_dir)

    def test_symlink_to_file(self):
        if not hasattr(os, 'symlink'):
            return

        try:
            os.symlink(self.file_name, os.path.join(self.temp_dir,
                                                    'link_to_file'))
        except NotImplementedError:
            # Windows versions before Vista don't support symbolic links
            return

        output = sorted(walk_func(self.temp_dir))
        dirs = sorted(output[0][1])
        files = sorted(output[0][2])
        self.assertEqual(dirs, ['dir'])
        self.assertEqual(files, ['file', 'link_to_file'])

        self.assertEqual(len(output), 2)
        self.assertEqual(output[1][1], [])
        self.assertEqual(output[1][2], ['subfile'])

    def test_symlink_to_directory(self):
        if not hasattr(os, 'symlink'):
            return

        link_name = os.path.join(self.temp_dir, 'link_to_dir')
        try:
            if IS_PY3:
                os.symlink(self.dir_name, link_name, target_is_directory=True)
            else:
                os.symlink(self.dir_name, link_name)
        except NotImplementedError:
            # Windows versions before Vista don't support symbolic links
            return

        output = sorted(walk_func(self.temp_dir))
        dirs = sorted(output[0][1])
        files = sorted(output[0][2])
        self.assertEqual(dirs, ['dir', 'link_to_dir'])
        self.assertEqual(files, ['file'])

        self.assertEqual(len(output), 2)
        self.assertEqual(output[1][1], [])
        self.assertEqual(output[1][2], ['subfile'])

        output = sorted(walk_func(self.temp_dir, followlinks=True))
        dirs = sorted(output[0][1])
        files = sorted(output[0][2])
        self.assertEqual(dirs, ['dir', 'link_to_dir'])
        self.assertEqual(files, ['file'])

        self.assertEqual(len(output), 3)
        self.assertEqual(output[1][1], [])
        self.assertEqual(output[1][2], ['subfile'])
        self.assertEqual(os.path.basename(output[2][0]), 'link_to_dir')
        self.assertEqual(output[2][1], [])
        self.assertEqual(output[2][2], ['subfile'])
