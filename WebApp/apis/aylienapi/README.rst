About
=====
This is the Python client library for AYLIEN's APIs. If you haven't already done so, you will need to `sign up`_.

Installation
============
To install, simply use ``pip``:

.. code-block:: bash

  $ pip install --upgrade aylien-apiclient

See the `Developers Guide`_ for additional documentation.

Example
=======
.. code-block:: python

  from aylienapiclient import textapi
  c = textapi.Client("YourApplicationID", "YourApplicationKey")
  s = c.Sentiment({'text': 'John is a very good football player!'})
  
Third Party Libraries and Dependencies
======================================
The following libraries will be installed when you install the client library:

- httplib2

For development you will also need the following libraries:

- httpretty
- unittest2
- nose

.. _documentation: http://httpd.apache.org
.. _sign up: https://developer.aylien.com/signup
.. _Developers Guide: https://developer.aylien.com/docs
