"""Tools for managing kernel specs"""

# Copyright (c) Jupyter Development Team.
# Distributed under the terms of the Modified BSD License.

import errno
import io
import json
import os
import re
import shutil
import warnings

pjoin = os.path.join

from ipython_genutils.py3compat import PY3
from traitlets import (
    HasTraits, List, Unicode, Dict, Set, Bool, Type, CaselessStrEnum
)
from traitlets.config import LoggingConfigurable

from jupyter_core.paths import jupyter_data_dir, jupyter_path, SYSTEM_JUPYTER_PATH


NATIVE_KERNEL_NAME = 'python3' if PY3 else 'python2'


class KernelSpec(HasTraits):
    argv = List()
    display_name = Unicode()
    language = Unicode()
    env = Dict()
    resource_dir = Unicode()
    interrupt_mode = CaselessStrEnum(
        ['message', 'signal'], default_value='signal'
    )
    metadata = Dict()

    @classmethod
    def from_resource_dir(cls, resource_dir):
        """Create a KernelSpec object by reading kernel.json

        Pass the path to the *directory* containing kernel.json.
        """
        kernel_file = pjoin(resource_dir, 'kernel.json')
        with io.open(kernel_file, 'r', encoding='utf-8') as f:
            kernel_dict = json.load(f)
        return cls(resource_dir=resource_dir, **kernel_dict)

    def to_dict(self):
        d = dict(argv=self.argv,
                 env=self.env,
                 display_name=self.display_name,
                 language=self.language,
                 interrupt_mode=self.interrupt_mode,
                 metadata=self.metadata,
                )

        return d

    def to_json(self):
        """Serialise this kernelspec to a JSON object.

        Returns a string.
        """
        return json.dumps(self.to_dict())


_kernel_name_pat = re.compile(r'^[a-z0-9._\-]+$', re.IGNORECASE)

def _is_valid_kernel_name(name):
    """Check that a kernel name is valid."""
    # quote is not unicode-safe on Python 2
    return _kernel_name_pat.match(name)


_kernel_name_description = "Kernel names can only contain ASCII letters and numbers and these separators:" \
 " - . _ (hyphen, period, and underscore)."


def _is_kernel_dir(path):
    """Is ``path`` a kernel directory?"""
    return os.path.isdir(path) and os.path.isfile(pjoin(path, 'kernel.json'))


def _list_kernels_in(dir):
    """Return a mapping of kernel names to resource directories from dir.

    If dir is None or does not exist, returns an empty dict.
    """
    if dir is None or not os.path.isdir(dir):
        return {}
    kernels = {}
    for f in os.listdir(dir):
        path = pjoin(dir, f)
        if not _is_kernel_dir(path):
            continue
        key = f.lower()
        if not _is_valid_kernel_name(key):
            warnings.warn("Invalid kernelspec directory name (%s): %s"
                % (_kernel_name_description, path), stacklevel=3,
            )
        kernels[key] = path
    return kernels


class NoSuchKernel(KeyError):
    def __init__(self, name):
        self.name = name

    def __str__(self):
        return "No such kernel named {}".format(self.name)


class KernelSpecManager(LoggingConfigurable):

    kernel_spec_class = Type(KernelSpec, config=True,
        help="""The kernel spec class.  This is configurable to allow
        subclassing of the KernelSpecManager for customized behavior.
        """
    )

    ensure_native_kernel = Bool(True, config=True,
        help="""If there is no Python kernelspec registered and the IPython
        kernel is available, ensure it is added to the spec list.
        """
    )

    data_dir = Unicode()
    def _data_dir_default(self):
        return jupyter_data_dir()

    user_kernel_dir = Unicode()
    def _user_kernel_dir_default(self):
        return pjoin(self.data_dir, 'kernels')

    whitelist = Set(config=True,
        help="""Whitelist of allowed kernel names.

        By default, all installed kernels are allowed.
        """
    )
    kernel_dirs = List(
        help="List of kernel directories to search. Later ones take priority over earlier."
    )
    def _kernel_dirs_default(self):
        dirs = jupyter_path('kernels')
        # At some point, we should stop adding .ipython/kernels to the path,
        # but the cost to keeping it is very small.
        try:
            from IPython.paths import get_ipython_dir
        except ImportError:
            try:
                from IPython.utils.path import get_ipython_dir
            except ImportError:
                # no IPython, no ipython dir
                get_ipython_dir = None
        if get_ipython_dir is not None:
            dirs.append(os.path.join(get_ipython_dir(), 'kernels'))
        return dirs

    def find_kernel_specs(self):
        """Returns a dict mapping kernel names to resource directories."""
        d = {}
        for kernel_dir in self.kernel_dirs:
            kernels = _list_kernels_in(kernel_dir)
            for kname, spec in kernels.items():
                if kname not in d:
                    self.log.debug("Found kernel %s in %s", kname, kernel_dir)
                    d[kname] = spec

        if self.ensure_native_kernel and NATIVE_KERNEL_NAME not in d:
            try:
                from ipykernel.kernelspec import RESOURCES
                self.log.debug("Native kernel (%s) available from %s",
                               NATIVE_KERNEL_NAME, RESOURCES)
                d[NATIVE_KERNEL_NAME] = RESOURCES
            except ImportError:
                self.log.warning("Native kernel (%s) is not available", NATIVE_KERNEL_NAME)

        if self.whitelist:
            # filter if there's a whitelist
            d = {name:spec for name,spec in d.items() if name in self.whitelist}
        return d
        # TODO: Caching?

    def _get_kernel_spec_by_name(self, kernel_name, resource_dir):
        """ Returns a :class:`KernelSpec` instance for a given kernel_name
        and resource_dir.
        """
        if kernel_name == NATIVE_KERNEL_NAME:
            try:
                from ipykernel.kernelspec import RESOURCES, get_kernel_dict
            except ImportError:
                # It should be impossible to reach this, but let's play it safe
                pass
            else:
                if resource_dir == RESOURCES:
                    return self.kernel_spec_class(resource_dir=resource_dir, **get_kernel_dict())

        return self.kernel_spec_class.from_resource_dir(resource_dir)

    def _find_spec_directory(self, kernel_name):
        """Find the resource directory of a named kernel spec"""
        for kernel_dir in self.kernel_dirs:
            try:
                files = os.listdir(kernel_dir)
            except OSError as e:
                if e.errno in (errno.ENOTDIR, errno.ENOENT):
                    continue
                raise
            for f in files:
                path = pjoin(kernel_dir, f)
                if f.lower() == kernel_name and _is_kernel_dir(path):
                    return path

        if kernel_name == NATIVE_KERNEL_NAME:
            try:
                from ipykernel.kernelspec import RESOURCES
            except ImportError:
                pass
            else:
                return RESOURCES

    def get_kernel_spec(self, kernel_name):
        """Returns a :class:`KernelSpec` instance for the given kernel_name.

        Raises :exc:`NoSuchKernel` if the given kernel name is not found.
        """
        if not _is_valid_kernel_name(kernel_name):
            self.log.warning("Kernelspec name %r is invalid: %s", kernel_name,
                             _kernel_name_description)

        resource_dir = self._find_spec_directory(kernel_name.lower())
        if resource_dir is None:
            raise NoSuchKernel(kernel_name)

        return self._get_kernel_spec_by_name(kernel_name, resource_dir)

    def get_all_specs(self):
        """Returns a dict mapping kernel names to kernelspecs.

        Returns a dict of the form::

            {
              'kernel_name': {
                'resource_dir': '/path/to/kernel_name',
                'spec': {"the spec itself": ...}
              },
              ...
            }
        """
        d = self.find_kernel_specs()
        res = {}
        for kname, resource_dir in d.items():
            try:
                if self.__class__ is KernelSpecManager:
                    spec = self._get_kernel_spec_by_name(kname, resource_dir)
                else:
                    # avoid calling private methods in subclasses,
                    # which may have overridden find_kernel_specs
                    # and get_kernel_spec, but not the newer get_all_specs
                    spec = self.get_kernel_spec(kname)

                res[kname] = {
                    "resource_dir": resource_dir,
                    "spec": spec.to_dict()
                }
            except Exception:
                self.log.warning("Error loading kernelspec %r", kname, exc_info=True)
        return res

    def remove_kernel_spec(self, name):
        """Remove a kernel spec directory by name.

        Returns the path that was deleted.
        """
        save_native = self.ensure_native_kernel
        try:
            self.ensure_native_kernel = False
            specs = self.find_kernel_specs()
        finally:
            self.ensure_native_kernel = save_native
        spec_dir = specs[name]
        self.log.debug("Removing %s", spec_dir)
        if os.path.islink(spec_dir):
            os.remove(spec_dir)
        else:
            shutil.rmtree(spec_dir)
        return spec_dir

    def _get_destination_dir(self, kernel_name, user=False, prefix=None):
        if user:
            return os.path.join(self.user_kernel_dir, kernel_name)
        elif prefix:
            return os.path.join(os.path.abspath(prefix), 'share', 'jupyter', 'kernels', kernel_name)
        else:
            return os.path.join(SYSTEM_JUPYTER_PATH[0], 'kernels', kernel_name)


    def install_kernel_spec(self, source_dir, kernel_name=None, user=False,
                            replace=None, prefix=None):
        """Install a kernel spec by copying its directory.

        If ``kernel_name`` is not given, the basename of ``source_dir`` will
        be used.

        If ``user`` is False, it will attempt to install into the systemwide
        kernel registry. If the process does not have appropriate permissions,
        an :exc:`OSError` will be raised.

        If ``prefix`` is given, the kernelspec will be installed to
        PREFIX/share/jupyter/kernels/KERNEL_NAME. This can be sys.prefix
        for installation inside virtual or conda envs.
        """
        source_dir = source_dir.rstrip('/\\')
        if not kernel_name:
            kernel_name = os.path.basename(source_dir)
        kernel_name = kernel_name.lower()
        if not _is_valid_kernel_name(kernel_name):
            raise ValueError("Invalid kernel name %r.  %s" % (kernel_name, _kernel_name_description))

        if user and prefix:
            raise ValueError("Can't specify both user and prefix. Please choose one or the other.")

        if replace is not None:
            warnings.warn(
                "replace is ignored. Installing a kernelspec always replaces an existing installation",
                DeprecationWarning,
                stacklevel=2,
            )

        destination = self._get_destination_dir(kernel_name, user=user, prefix=prefix)
        self.log.debug('Installing kernelspec in %s', destination)

        kernel_dir = os.path.dirname(destination)
        if kernel_dir not in self.kernel_dirs:
            self.log.warning("Installing to %s, which is not in %s. The kernelspec may not be found.",
                kernel_dir, self.kernel_dirs,
            )

        if os.path.isdir(destination):
            self.log.info('Removing existing kernelspec in %s', destination)
            shutil.rmtree(destination)

        shutil.copytree(source_dir, destination)
        self.log.info('Installed kernelspec %s in %s', kernel_name, destination)
        return destination

    def install_native_kernel_spec(self, user=False):
        """DEPRECATED: Use ipykernel.kenelspec.install"""
        warnings.warn("install_native_kernel_spec is deprecated."
            " Use ipykernel.kernelspec import install.", stacklevel=2)
        from ipykernel.kernelspec import install
        install(self, user=user)


def find_kernel_specs():
    """Returns a dict mapping kernel names to resource directories."""
    return KernelSpecManager().find_kernel_specs()

def get_kernel_spec(kernel_name):
    """Returns a :class:`KernelSpec` instance for the given kernel_name.

    Raises KeyError if the given kernel name is not found.
    """
    return KernelSpecManager().get_kernel_spec(kernel_name)

def install_kernel_spec(source_dir, kernel_name=None, user=False, replace=False,
                        prefix=None):
    return KernelSpecManager().install_kernel_spec(source_dir, kernel_name,
                                                    user, replace, prefix)

install_kernel_spec.__doc__ = KernelSpecManager.install_kernel_spec.__doc__

def install_native_kernel_spec(user=False):
    return KernelSpecManager().install_native_kernel_spec(user=user)

install_native_kernel_spec.__doc__ = KernelSpecManager.install_native_kernel_spec.__doc__
