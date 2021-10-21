import os
import json

from six import iteritems

from . import constants


class PackageJsonWorkspaceError(RuntimeError):
    pass


class PackageJson(object):
    DEP_KEY = "dependencies"
    DEV_DEP_KEY = "devDependencies"
    PEER_DEP_KEY = "peerDependencies"
    OPT_DEP_KEY = "optionalDependencies"
    DEP_KEYS = (DEP_KEY, DEV_DEP_KEY, PEER_DEP_KEY, OPT_DEP_KEY)

    WORKSPACE_SCHEMA = "workspace:"

    @classmethod
    def load(cls, path):
        """
        :param path: package.json path
        :type path: str
        :rtype: PackageJson
        """
        pj = cls(path)
        pj.read()

        return pj

    def __init__(self, path):
        if not os.path.isabs(path):
            raise TypeError("Absolute path required, given: {}".format(path))

        self.path = path
        self.data = None

    def read(self):
        with open(self.path) as f:
            self.data = json.load(f)

    def get_name(self):
        return self.data.get("name")

    def get_workspace_dep_paths(self):
        """
        :return: Workspace dependencies.
        :rtype: list of (str, str)
        """
        dep_paths = []
        schema = self.WORKSPACE_SCHEMA
        schema_len = len(schema)

        for deps in map(lambda x: self.data.get(x), self.DEP_KEYS):
            if not deps:
                continue

            for name, spec in iteritems(deps):
                if not spec.startswith(schema):
                    continue

                spec_path = spec[schema_len:]
                if not (spec_path.startswith(".") or spec_path.startswith("..")):
                    raise PackageJsonWorkspaceError(
                        "Expected relative path specifier for workspace dependency, but got '{}' for {} in {}".format(spec, name, self.path))

                dep_paths.append((name, spec_path))

        return dep_paths

    def get_workspace_deps(self):
        """
        :rtype: list of PackageJson
        """
        ws_deps = []
        pj_dir = os.path.dirname(self.path)

        for (name, rel_path) in self.get_workspace_dep_paths():
            dep_path = os.path.normpath(os.path.join(pj_dir, rel_path))
            dep_pj = PackageJson.load(os.path.join(dep_path, constants.PACKAGE_JSON_FILENAME))

            if name != dep_pj.get_name():
                raise PackageJsonWorkspaceError(
                    "Workspace dependency name mismatch, found '{}' instead of '{}' in {}".format(name, dep_pj.get_name(), self.path))

            ws_deps.append(dep_pj)

        return ws_deps

    def get_workspace_map(self):
        """
        :return: Absolute paths of workspace dependencies (including transitive) mapped to package.json and depth.
        :rtype: dict of (PackageJson, int)
        """
        ws_deps = {}
        # list of (pj, depth)
        pj_queue = [(self, 0)]

        while len(pj_queue):
            (pj, depth) = pj_queue.pop()
            pj_dir = os.path.dirname(pj.path)
            if pj_dir in ws_deps:
                continue

            ws_deps[pj_dir] = (pj, depth)

            for dep_pj in pj.get_workspace_deps():
                pj_queue.append((dep_pj, depth + 1))

        return ws_deps
