from aylienapiclient import textapi
c = textapi.Client("4df6473c", "f827888e31b6b52f85a6061eb3f18ad1")
s = c.Sentiment({'text': 'John is a very good football player!'})
print s