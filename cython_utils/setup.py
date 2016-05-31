from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext


ext_modules = [
    Extension("token_based_jac_cython", ["token_based_jac_cython.pyx"]),
    ]

setup(
  name = 'token based jaccard sim cython',
  cmdclass = {'build_ext': build_ext},
  ext_modules = ext_modules
)