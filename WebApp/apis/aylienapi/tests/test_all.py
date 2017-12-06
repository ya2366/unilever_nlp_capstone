from __future__ import print_function

import vcr
import os
import sys
import inspect

from nose.tools import eq_
from nose.tools import ok_
from nose.tools import raises

from aylienapiclient import http
from aylienapiclient import textapi
from aylienapiclient.errors import HttpError
from aylienapiclient.errors import MissingParameterError
from aylienapiclient.errors import MissingCredentialsError

try:
  from urllib.request import urlopen
except ImportError:
  from urllib2 import urlopen

APP_ID = os.environ['TEXTAPI_APP_ID']
APP_KEY = os.environ['TEXTAPI_APP_KEY']

aylien_vcr = vcr.VCR(
    cassette_library_dir='tests/fixtures/cassettes',
    path_transformer=vcr.VCR.ensure_suffix('.yaml'),
    filter_headers=['x-aylien-textapi-application-id', 'x-aylien-textapi-application-key'],
)

endpoints = ['Extract', 'Classify', 'Concepts', 'Entities', 'Hashtags',
  'Language', 'Related', 'Sentiment', 'Summarize', 'Microformats', 'ImageTags',
  'UnsupervisedClassify', 'Combined']

generic_counter = 0

def build_path_for_generic_generator(function):
  return function.__name__ + "_" + endpoints[generic_counter].lower()

def test_generic_generator():
  for e in endpoints:
    if e not in ['Microformats', 'ImageTags']:
      yield check_invalid_auth, e, "random"
    else:
      yield check_invalid_auth, e, "http://example.com/"

@raises(HttpError)
@aylien_vcr.use_cassette(func_path_generator=build_path_for_generic_generator)
def check_invalid_auth(endpoint, input):
  global generic_counter
  generic_counter += 1
  client = textapi.Client("app_id", "app_key")
  method = getattr(client, endpoint)
  method(input)

@raises(MissingCredentialsError)
def test_raises_missing_credentials_error():
  client = textapi.Client("", "")

def test_empty_params_generator():
  for e in endpoints:
    yield check_empty_params, e

@raises(MissingParameterError)
def check_empty_params(endpoint):
  client = textapi.Client("app_id", "app_key")
  method = getattr(client, endpoint)
  method({})

@aylien_vcr.use_cassette()
def test_sentiment():
  client = textapi.Client(APP_ID, APP_KEY)
  sentiment = client.Sentiment({'text': 'John is a very good football player!'})
  for prop in ['polarity', 'subjectivity', 'polarity_confidence', 'subjectivity_confidence']:
    ok_(prop in sentiment)
  rate_limits = client.RateLimits()
  for prop in ['limit', 'remaining', 'reset']:
    ok_(prop in rate_limits)

@aylien_vcr.use_cassette()
def test_extract():
  client = textapi.Client(APP_ID, APP_KEY)
  extract = client.Extract({'url': 'http://techcrunch.com/2014/02/27/aylien-launches-text-analysis-api-to-help-developers-extract-meaning-from-documents/'})
  for prop in ['author', 'image', 'article', 'videos', 'title', 'feeds']:
    ok_(prop in extract)

@aylien_vcr.use_cassette()
def test_classify():
  client = textapi.Client(APP_ID, APP_KEY)
  classify = client.Classify({'url': 'http://www.bbc.com/sport/0/football/25912393'})
  for prop in ['text', 'language', 'categories']:
    ok_(prop in classify)
  ok_(isinstance(classify['categories'], list))

@aylien_vcr.use_cassette()
def test_classify_by_taxonomu():
  client = textapi.Client(APP_ID, APP_KEY)
  classify = client.ClassifyByTaxonomy({'text': "John is a very good football player", 'taxonomy': "iab-qag"})
  for prop in ['text', 'language', 'categories']:
    ok_(prop in classify)
  ok_(isinstance(classify['categories'], list))

@aylien_vcr.use_cassette()
def test_concepts():
  client = textapi.Client(APP_ID, APP_KEY)
  concepts = client.Concepts({'url': 'http://www.bbc.co.uk/news/business-25821345'})
  for prop in ['text', 'language', 'concepts']:
    ok_(prop in concepts)
  ok_(isinstance(concepts['concepts'], dict))

@aylien_vcr.use_cassette()
def test_entities():
  client = textapi.Client(APP_ID, APP_KEY)
  entities = client.Entities({'url': 'http://www.businessinsider.com/carl-icahn-open-letter-to-apple-2014-1'})
  for prop in ['text', 'language', 'entities']:
    ok_(prop in entities)
  ok_(isinstance(entities['entities'], dict))

@aylien_vcr.use_cassette()
def test_hashtags():
  client = textapi.Client(APP_ID, APP_KEY)
  hashtags = client.Hashtags({'url': 'http://www.bbc.com/sport/0/football/25912393'})
  for prop in ['text', 'language', 'hashtags']:
    ok_(prop in hashtags)
  ok_(isinstance(hashtags['hashtags'], list))

@aylien_vcr.use_cassette()
def test_language():
  client = textapi.Client(APP_ID, APP_KEY)
  language = client.Language("Hello there! How's it going?")
  for prop in ['text', 'lang', 'confidence']:
    ok_(prop in language)

@aylien_vcr.use_cassette()
def test_related():
  client = textapi.Client(APP_ID, APP_KEY)
  related = client.Related('android')
  for prop in ['phrase', 'related']:
    ok_(prop in related)
  ok_(isinstance(related['related'], list))

@aylien_vcr.use_cassette()
def test_summarize():
  client = textapi.Client(APP_ID, APP_KEY)
  summary = client.Summarize('http://techcrunch.com/2014/02/27/aylien-launches-text-analysis-api-to-help-developers-extract-meaning-from-documents/')
  for prop in ['text', 'sentences']:
    ok_(prop in summary)
  ok_(isinstance(summary['sentences'], list))

@aylien_vcr.use_cassette()
def test_microformats():
  client = textapi.Client(APP_ID, APP_KEY)
  microformats = client.Microformats('http://codepen.io/anon/pen/ZYaKbz.html')
  ok_('hCards' in microformats)
  ok_(isinstance(microformats['hCards'], list))

@aylien_vcr.use_cassette()
def test_imagetags():
  client = textapi.Client(APP_ID, APP_KEY)
  image_tags = client.ImageTags('http://gluegadget.com/liffey/IMG_1209.png')
  for prop in ['image', 'tags']:
    ok_(prop in image_tags)

@aylien_vcr.use_cassette()
def test_unsupervised_classification():
  client = textapi.Client(APP_ID, APP_KEY)
  classification = client.UnsupervisedClassify({
    'url': 'http://techcrunch.com/2015/07/16/microsoft-will-never-give-up-on-mobile',
    'class': ['technology', 'sports'],
  })
  for prop in ['classes', 'text']:
    ok_(prop in classification)
  ok_(isinstance(classification['classes'], list))

def test_combined():
  with aylien_vcr.use_cassette('tests/fixtures/cassettes/test_combined.yaml') as cass:
    client = textapi.Client(APP_ID, APP_KEY)
    request_endpoints = ['entities', 'concepts', 'classify']
    combined = client.Combined({
      'url': 'http://www.bbc.com/news/technology-33764155',
      'endpoint': request_endpoints
    })
    for prop in ['results', 'text']:
      ok_(prop in combined)
    request = cass.requests[0]
    eq_(request.method, 'POST')
    for e in request_endpoints:
      ok_("endpoint=" + e in str(request.body))
    endpoints = set(map(lambda e: e['endpoint'], combined['results']))
    ok_(endpoints == set(request_endpoints))
