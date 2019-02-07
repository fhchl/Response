#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Note: To use the 'upload' functionality of this file, you must:
#   $ pip install twine

import io
import os

from setuptools import setup

# Package meta-data.
NAME = "response"
DESCRIPTION = "Your handy frequency and impulse response processing object"
URL = "https://github.com/fhchl/Response"
EMAIL = "franz.heuchel@gmail.com"
AUTHOR = "Franz M. Heuchel"
REQUIRES_PYTHON = ">=3.6.0"
VERSION = "0.1.1"

# What packages are required for this module to be executed?
REQUIRED = ["numpy", "scipy", "matplotlib>=2.2.0"]

# What packages are optional?
EXTRAS = {"dev": ["pytest", "python-semantic-release"]}

here = os.path.abspath(os.path.dirname(__file__))

# Import the README and use it as the long-description.
# Note: this will only work if 'README.md' is present in your MANIFEST.in file!
try:
    with io.open(os.path.join(here, "README.md"), encoding="utf-8") as f:
        long_description = "\n" + f.read()
except FileNotFoundError:
    long_description = DESCRIPTION

setup(
    name=NAME,
    version=VERSION,
    description=DESCRIPTION,
    long_description=long_description,
    long_description_content_type="text/markdown",
    author=AUTHOR,
    author_email=EMAIL,
    python_requires=REQUIRES_PYTHON,
    url=URL,
    py_modules=["response"],
    install_requires=REQUIRED,
    include_package_data=True,
    license="MIT",
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Topic :: Multimedia :: Sound/Audio",
        "Topic :: Utilities",
    ],
)
