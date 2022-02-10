#!/usr/bin/env python

import os

import setuptools

__version__ = "0.1"

# Find the absolute path
here = os.path.abspath(os.path.dirname(__file__))

# Get the long description from the README file
# with open(os.path.join(here, "README.md")) as f:
#     long_description = f.read()

short_description = "Catalyst Scoring"


setuptools.setup(
    name="catalyst",
    version=__version__,
    description=short_description,
    python_requires=">=3.9",
)
