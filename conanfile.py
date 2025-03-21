from conan import ConanFile

import os

from conan.tools.files import copy
from conan.tools.cmake import CMakeToolchain, CMakeDeps, cmake_layout
from conan.tools.env import Environment


class App(ConanFile):

    settings = "os", "compiler", "build_type", "arch"

    default_options = {}

    def requirements(self):
        self.requires("openssl/3.0.15")

    def build_requirements(self):
        self.tool_requires("ragel/6.10")
        self.tool_requires("swig/4.0.2")
        self.tool_requires("yasm/1.3.0")

    def generate(self):
        CMakeDeps(self).generate()
        CMakeToolchain(self).generate()

        for dep in self.dependencies.values():
            for bindir in dep.cpp_info.bindirs:
                copy(self, pattern="*swig*", src=bindir, dst=self.build_folder + "../../../.././bin")
                # SWIG recipe under Conan2 does not set SWIG_LIB, do it manually here or else build fails
                if os.path.exists(os.path.join(bindir, "swig")):
                    env = Environment()
                    if not env.vars(self).get("SWIG_LIB"):
                        env.define("SWIG_LIB", os.path.join(bindir, "swiglib"))
            for bindir in dep.cpp_info.bindirs:
                copy(self, pattern="*yasm*", src=bindir, dst=self.build_folder + "../../../.././bin")
            for bindir in dep.cpp_info.bindirs:
                copy(self, pattern="ragel*", src=bindir, dst=self.build_folder + "../../../.././bin")
            for bindir in dep.cpp_info.bindirs:
                copy(self, pattern="ytasm*", src=bindir, dst=self.build_folder + "../../../.././bin")

    def layout(self):
        cmake_layout(self)
