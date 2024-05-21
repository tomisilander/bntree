#!/usr/bin/env python
from setuptools import setup
from Cython.Build import cythonize

setup(
    ext_modules = cythonize("cycounts.pyx", annotate=True,
                            compiler_directives={'boundscheck' : False,
                                                 'wraparound' : False,
                                                 'initializedcheck' : False,
                                                 'language_level' : "3"})
)
