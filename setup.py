#!/usr/bin/env python3
#
# setup.py
#
"""
Setup script for stumpy
"""

from setuptools import setup, find_packages
from importlib.machinery import SourceFileLoader

metadata = SourceFileLoader("stumpy.metadata", "stumpy/__meta__.py").load_module()

tar_url = 'https://github.com/akubera/stumpy/archive/v%s.tar.gz' % (metadata.version)

setup(
    name=metadata.package,
    version=metadata.version,
    author=metadata.author,
    license=metadata.license,
    url=metadata.url,
    download_url=tar_url,
    author_email=metadata.author_email,
)
