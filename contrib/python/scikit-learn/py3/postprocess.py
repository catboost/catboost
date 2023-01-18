#!/usr/bin/env python3

import os

from Cython import Tempita


joiner = "\n    "
cython_sources_c = []
cython_sources_cpp = []


for (dir, dirnames, filenames) in os.walk("sklearn"):
    for filename in filenames:
        if not filename.endswith(".tp"):
            continue

        path_template = os.path.join(dir, filename)
        with open(path_template, "rt") as file_template:
            template = file_template.read()

        path_cython = path_template[:-3]
        with open(path_cython, "wt") as file_cython:
            file_cython.write(Tempita.sub(template))

        if path_cython.endswith(".pyx"):
            if path_cython == "sklearn/metrics/_dist_metrics.pyx":
                cython_sources_c.append(path_cython)
            else:
                cython_sources_cpp.append(path_cython)


print("Add the following SRCS to CYTHON_C section")
print(
f"""
    CYTHON_C
    {joiner.join(sorted(cython_sources_c))}

    CYTHON_CPP
    {joiner.join(sorted(cython_sources_cpp))}
""")
