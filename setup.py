#!/usr/bin/env python
from ez_setup import use_setuptools
use_setuptools()

from setuptools import setup, find_packages, Extension

from Cython.Distutils import build_ext
from Cython.Build import cythonize

from os import path
import subprocess

#Cython extensions can be built in the standard fashion via distribute
#but files including numpy headers must have numpy include path specified.
import numpy

#Versioning logic
def get_git_version():
    """Get current version tag via call to git describe.

    Version tags are of the format:
        v<major>.<minor>.<patch>
    """

    try:
        version_string = subprocess.check_output(["git", "describe", "--match", "v*"])
        return version_string.strip().lstrip("v")
    except:
        return "0.0.0"

#Cython build logic
cython_subdirs = [
    "fragment_profiling",
]

cython_modules= cythonize([path.join(d, "*.pyx") for d in cython_subdirs])

for r in cython_modules:
    r.extra_compile_args=["-fopenmp"]
    r.extra_link_args=["-fopenmp"]

def pkgconfig(*packages, **kw):
    flag_map = {'-I': 'include_dirs', '-L': 'library_dirs', '-l': 'libraries'}
    for token in subprocess.check_output(["pkg-config", "--libs", "--cflags"] + list(packages)).strip().split(" "):
        kw.setdefault(flag_map.get(token[:2]), []).append(token[2:])
    return kw

include_dirs = [numpy.get_include(), path.abspath(".")]

setup(
        name="fragment_profiling",
        description="Modules for structure fragment profiling.",
        author="Alex Ford",
        author_email="fordas@uw.edu",
        url="https://github.com/fordas/fragment_profiling/",
        version=get_git_version(),
        provides=["fragment_profiling"],
        packages=find_packages(),
        install_requires=["interface_fragment_matching", "numpy", "matplotlib"],
        setup_requires=["Cython>=0.18"],
        tests_require=["nose", "nose-html", "coverage", "nose-progressive"],
        test_suite = "nose.collector",
        cmdclass = {'build_ext': build_ext},
        #Extra options to cythonize are *not* the same 'Extension'
        #See Cython.Compiler.Main.CompilationOptions for details
        #
        #In particular, include_dirs must be specified in setup
        #   as opposed to Extension if globs are used.
        include_dirs=include_dirs,
        ext_modules = cython_modules
    )
