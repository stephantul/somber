# -*- coding: utf-8 -*-
"""Setup file."""
from setuptools import setup
from setuptools import find_packages
from Cython.Build import cythonize
import numpy as np

setup(name='somber',
      version='2.0.2',
      description='Self-Organizing Maps in Numpy',
      author='Stéphan Tulkens',
      author_email='stephan.tulkens@uantwerpen.be',
      url='https://github.com/stephantul/somber',
      license='MIT',
      packages=find_packages(exclude=['examples']),
      install_requires=['numpy>=1.11.0'],
      classifiers=[
          'Intended Audience :: Developers',
          'Programming Language :: Python :: 3'],
      keywords='self-organizing maps machine learning unsupervised',
      ext_modules=cythonize("somber/distance/distance.pyx"),
      include_dirs=[np.get_include()],
      zip_safe=True)
