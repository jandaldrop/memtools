#!/usr/bin/env python

from distutils.core import setup, Extension
import numpy as np

ext_modules = [ Extension('ckernel', sources = ['ckernel.cpp']) ]

setup(
        name = 'ckernel',
        version = '0.2',
        include_dirs = [np.get_include()], #Add Include path of numpy
        ext_modules = ext_modules
      )
