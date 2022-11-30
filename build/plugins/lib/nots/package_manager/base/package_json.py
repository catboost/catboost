import os
import json

from six import iteritems

from .utils import build_pj_path


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

    def write(self, path=None):
        """
        :param path: path to store package.json, defaults to original path
        :type path: str
        """
        if path is None:
            path = self.path

        with open(path, "w") as f:
            json.dump(self.data, f, indent=4)

    def get_name(self):
        return self.data["name"]

    def get_version(self):
        return self.data["version"]

    def get_description(self):
        return self.data.get("description")

    def get_nodejs_version(self):
        return self.data.get("engines", {}).get("node")

    def dependencies_iter(self):
        for key in self.DEP_KEYS:
            deps = self.data.get(key)
            if not deps:
                continue

            for name, spec in iteritems(deps):
                yield (name, spec)

    def get_workspace_dep_spec_paths(self):
        """
        Returns names and paths from specifiers of the defined workspace dependencies.
        :rtype: list of (str, str)
        """
        spec_paths = []
        schema = self.WORKSPACE_SCHEMA
        schema_len = len(schema)

        for name, spec in self.dependencies_iter():
            if not spec.startswith(schema):
                continue

            spec_path = spec[schema_len:]
            if not (spec_path.startswith(".") or spec_path.startswith("..")):
                raise PackageJsonWorkspaceError(
                    "Expected relative path specifier for workspace dependency, but got '{}' for {} in {}".format(spec, name, self.path))

            spec_paths.append((name, spec_path))

        return spec_paths

    def get_workspace_dep_paths(self, base_path=None):
        """
        Returns paths of the defined workspace dependencies.
        :param base_path: base path to resolve relative dep paths
        :type base_path: str
        :rtype: list of str
        """
        if base_path is None:
            base_path = os.path.dirname(self.path)

        return [os.path.normpath(os.path.join(base_path, p)) for _, p in self.get_workspace_dep_spec_paths()]

    def get_workspace_deps(self):
        """
        :rtype: list of PackageJson
        """
        ws_deps = []
        pj_dir = os.path.dirname(self.path)

        for name, rel_path in self.get_workspace_dep_spec_paths():
            dep_path = os.path.normpath(os.path.join(pj_dir, rel_path))
            dep_pj = PackageJson.load(build_pj_path(dep_path))

            if name != dep_pj.get_name():
                raise PackageJsonWorkspaceError(
                    "Workspace dependency name mismatch, found '{}' instead of '{}' in {}".format(name, dep_pj.get_name(), self.path))

            ws_deps.append(dep_pj)

        return ws_deps

    def get_workspace_map(self, ignore_self=False):
        """
        Returns absolute paths of the workspace dependencies (including transitive) mapped to package.json and depth.
        :param ignore_self: whether path of the current module will be excluded
        :type ignore_self: bool
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

            if not ignore_self or pj != self:
                ws_deps[pj_dir] = (pj, depth)

            for dep_pj in pj.get_workspace_deps():
                pj_queue.append((dep_pj, depth + 1))

        return ws_deps
