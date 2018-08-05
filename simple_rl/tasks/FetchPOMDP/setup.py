from distutils.core import setup
from Cython.Build import cythonize

setup(name='cython code',
      ext_modules=cythonize("cstuff.pyx"))