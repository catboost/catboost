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

    def write(self, path=None):
        """
        :param path: tsconfig path, defaults to original path
        :type path: str
        """
        if path is None:
            path = self.path

        with open(path, "w") as f:
            json.dump(self.data, f)
