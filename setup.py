#!/usr/bin/env python

from distutils.core import setup, Extension
import numpy as np

ext_modules = [ Extension('memtools.ckernel', sources = ['memtools/ckernel.cpp']) ]

setup(
        name="memtools",
        version='1.11',
        description='Python module for the extraction of memory kernels from time series',
        author='Jan Daldrop',
        author_email='daldrop@zedat.fu-berlin.de',
        include_dirs = [np.get_include()],
        ext_modules = ext_modules,
        install_requires=['numpy','pandas', 'scipy'],
        packages=["memtools"]
      )
