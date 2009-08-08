from distutils.core import setup, Extension
import sys, os, os.path, string

if sys.platform != 'darwin':
    extra_link_args = ['-s']

setup(name='etse',
      version='0.8.0',
      description='Grammar-Guided Feature Extraction',
      author='Damian Eads',
      packages=['ggfe'],
      scripts=[],
     )
