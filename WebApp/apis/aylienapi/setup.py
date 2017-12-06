# Copyright 2015 Aylien, Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Setup script for AYLIEN Text API Python client.

Also installs included versions of third party libraries, if those libraries
are not already installed.
"""
from __future__ import print_function

import sys

if sys.version_info < (2, 6):
  print('aylien-apiclient requires python version >= 2.6',
      file=sys.stderr)
  sys.exit(1)

from setuptools import setup

packages = [
    'aylienapiclient'
]

install_requires = [
    'httplib2>=0.9'
]

tests_require = [
    'nose',
    'vcrpy==1.7.3',
]

import aylienapiclient
version = aylienapiclient.__version__

setup(
    name="aylien-apiclient",
    version=version,
    description="AYLIEN Text API Client Library for Python",
    long_description=open('README.rst').read(),
    author="Aylien, Inc.",
    url="https://github.com/AYLIEN/aylien_textapi_python",
    install_requires=install_requires,
    tests_require=tests_require,
    packages=packages,
    package_data={},
    license="Apache 2.0",
    keywords="aylien text api client",
    classifiers=[
      'Development Status :: 3 - Alpha',
      'Intended Audience :: Developers',
      'License :: OSI Approved :: Apache Software License',
      'Operating System :: POSIX',
      'Topic :: Internet :: WWW/HTTP',
    ],
)
