#!/usr/bin/env python
import sys

if 'develop' in sys.argv:
    # use setuptools for develop, but nothing else
    from setuptools import setup
else:
    from distutils.core import setup

with open('README.rst') as file:
    long_description = file.read()

#with open('CHANGES') as file:
#    long_description += file.read()


__version__ = '0.1'


setup(name='h2co_modeling',
      version=__version__,
      description='H2CO modeling codes',
      long_description=long_description,
      author='Adam Ginsburg',
      author_email='adam.g.ginsburg@gmail.com',
      data_files=[],
      url='',
      packages=['h2co_modeling'],
     )
