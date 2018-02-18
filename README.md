# Tweet_normalizer
The following program is normalization system which helps to change raw English tweets into (partially)
normalized tweets, suitable for further NLP processing

## Requirements
- Python 2.7
- Chainer 1.7 (chainer)
- NLTK 3.0
- context2vec

## To run:
sh normalize_tweets.sh -f true
arguments are:

• -c: path to context2vec model (optional, hardcoded in normalize_tweets.sh)

• -i: path to the input raw tweets (optional, hardcoded in normalize_tweets.sh)

• -o: path to the output file (optional)

• -f: use  Damereau-Levenshtein fast computainon (optional, dafault=true)

```
$ sh normalize_tweets.sh -c context2vec.ukwac.model.package/context2vec.ukwac.model.params -i CorpusBataclan_en.1M.raw.tx -f true
```
