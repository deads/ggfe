from distutils.core import setup, Extension
import sys, os, os.path, string

if sys.platform != 'darwin':
    extra_link_args = ['-s']

ext_modules = []
ext_modules.append(Extension('_ggfe_image_wrap',
                             ['ggfe/ggfe_image_wrap.cpp', 'ggfe/viola_jones.cpp'],
                             extra_link_args = extra_link_args,
                             include_dirs=['../convert-xy/'],
                             libraries=['m', 'cvd']))

setup(name='ggfe',
      version='0.8.0',
      description='Grammar-Guided Feature Extraction',
      author='Damian Eads',
      packages=['ggfe'],
      scripts=[],
      ext_modules=ext_modules,
     )
