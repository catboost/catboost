import os
import json

from .ts_errors import TsError, TsValidationError


class TsConfig(object):
    @classmethod
    def load(cls, path):
        """
        :param path: tsconfig.json path
        :type path: str
        :rtype: TsConfig
        """
        tsconfig = cls(path)
        tsconfig.read()

        return tsconfig

    def __init__(self, path):
        if not os.path.isabs(path):
            raise TypeError("Absolute path required, given: {}".format(path))

        self.path = path
        self.data = {}

    def read(self):
        try:
            with open(self.path) as f:
                self.data = json.load(f)
        except Exception as e:
            raise TsError("Failed to read tsconfig {}: {}".format(self.path, e))

    def get_or_create_compiler_options(self):
        """
        Returns ref to the "compilerOptions" dict.
        :rtype: dict
        """
        opts = self.data.get("compilerOptions")
        if opts is None:
            opts = {}
            self.data["compilerOptions"] = opts

        return opts

    def compiler_option(self, name, default=None):
        """
        :param name: option key
        :type name: str
        :param default: default value
        :type default: mixed
        :rtype: mixed
        """
        return self.get_or_create_compiler_options().get(name, default)

    def inject_plugin(self, plugin):
        """
        :param plugin: plugin dict (ts-patch compatible, see https://github.com/nonara/ts-patch)
        :type plugin: dict of str
        """
        opts = self.get_or_create_compiler_options()
        if not opts.get("plugins"):
            opts["plugins"] = []
        opts["plugins"].append(plugin)

    def validate(self):
        """
        Checks whether the config is compatible with current toolchain.
        """
        opts = self.get_or_create_compiler_options()
        errors = []
        root_dir = opts.get("rootDir")
        out_dir = opts.get("outDir")
        config_dir = os.path.dirname(self.path)
        is_mod_subdir = lambda p: not os.path.isabs(p) and os.path.normpath(os.path.join(config_dir, p)).startswith(config_dir)

        if root_dir is None:
            errors.append("'rootDir' option is required")
        elif not is_mod_subdir(root_dir):
            errors.append("'rootDir' should be a subdirectory of the module")

        if out_dir is None:
            errors.append("'outDir' option is required")
        elif not is_mod_subdir(out_dir):
            errors.append("'outDir' should be a subdirectory of the module")

        if opts.get("outFile") is not None:
            errors.append("'outFile' option is not supported")

        if opts.get("preserveSymlinks"):
            errors.append("'preserveSymlinks' option is not supported due to pnpm limitations")

        if opts.get("rootDirs") is not None:
            errors.append("'rootDirs' option is not supported, relative imports should have single root")

        if self.data.get("files") is not None:
            errors.append("'files' option is not supported, use 'include'")

        if self.data.get("references") is not None:
            errors.append("composite builds are not supported, use peerdirs in ya.make instead of 'references' option")

        if len(errors):
            raise TsValidationError(self.path, errors)

    def transform_paths(self, build_path, sources_path):
        """
        Updates config with correct abs paths.
        All source files/dirs will be mapped to `sources_path`, output files/dirs will be mapped to `build_path`.
        :param build_path: module's build root
        :type build_path: str
        :param sources_path: module's source root
        :type sources_path: str
        """
        opts = self.get_or_create_compiler_options()

        sources_path_rel = lambda x: os.path.normpath(os.path.join(sources_path, x))
        build_path_rel = lambda x: os.path.normpath(os.path.join(build_path, x))

        root_dir = opts["rootDir"]
        out_dir = opts["outDir"]

        opts["rootDir"] = sources_path_rel(root_dir)
        opts["outDir"] = build_path_rel(out_dir)

        if opts.get("typeRoots"):
            opts["typeRoots"] = list(map(sources_path_rel, opts["typeRoots"])) + list(map(build_path_rel, opts["typeRoots"]))

        if opts.get("paths") is None:
            opts["paths"] = {}

        # See: https://st.yandex-team.ru/FBP-47#62b4750775525b18f08205c7
        opts["paths"]["*"] = ["*", "./@types/*"]

        opts["baseUrl"] = "./node_modules"

        self.data["include"] = list(map(sources_path_rel, self.data.get("include", [])))
        self.data["exclude"] = list(map(sources_path_rel, self.data.get("exclude", [])))

        if opts.get("sourceMap"):
            opts["sourceRoot"] = os.path.relpath(root_dir, out_dir)

        opts["skipLibCheck"] = True

    def write(self, path=None):
        """
        :param path: tsconfig path, defaults to original path
        :type path: str
        """
        if path is None:
            path = self.path

        with open(path, "w") as f:
            json.dump(self.data, f)
