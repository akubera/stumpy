#!/usr/bin/env python3
#
# setup.py
#
"""
A package to connect ROOT files to numpy projects.
"""

from os import path
from glob import glob
from setuptools import setup, find_packages
from importlib.machinery import SourceFileLoader


NAME = 'stumpy'

REQUIRES = [
]

OPTIONAL_REQUIRES = {
}

TESTS_REQUIRE = [
    'pytest',
    'pytest-asyncio',
]

SETUP_REQUIRES = [
    'pytest-runner',
]

PACKAGES = find_packages(
    exclude=["*.tests", "*.tests.*", "tests.*", "tests"]
)

NAMESPACES = [
]

CLASSIFIERS = [
    "Development Status :: 1 - Planning",
    "Operating System :: OS Independent",
    "License :: OSI Approved :: BSD License",
    "Programming Language :: Python",
    "Programming Language :: Python :: 3",
    "Programming Language :: Python :: 3.5",
    "Programming Language :: Python :: 3.6",
    "Topic :: Internet :: WWW/HTTP",
    "Topic :: Scientific/Engineering :: Physics",
    "Topic :: Scientific/Engineering :: Visualization",
    "Natural Language :: English",
]

metafile = path.join("stumpy", "__meta__.py")
metadata = SourceFileLoader("stumpy.metadata", metafile).load_module()

tar_url = 'https://github.com/akubera/stumpy/archive/v%s.tar.gz' % (metadata.version) # noqa

setup(
    name=NAME,
    version=metadata.version,
    author=metadata.author,
    license=metadata.license,
    url=metadata.url,
    download_url=tar_url,
    author_email=metadata.author_email,
    description=__doc__.strip(),
    classifiers=CLASSIFIERS,
    install_requires=REQUIRES,
    extras_require=OPTIONAL_REQUIRES,
    tests_require=TESTS_REQUIRE,
    setup_requires=SETUP_REQUIRES,
    packages=PACKAGES,
    namespace_packages=NAMESPACES,
    platforms='all',
    scripts=glob('scripts/*')
)
