# Tweet_normalizer
The following program is a normalization system which helps to convert raw English tweets into (partially)
normalized tweets, suitable for further NLP processing

## Requirements
- Python 2.7
- Chainer 1.7 (chainer)
- NLTK 3.0
- context2vec
- If not installed the NLTK Brown corpus will be downloaded when running the script

## To run:
sh normalize_tweets.sh -f true
arguments are:

• -c: path for context2vec model 

• -i: path for the input raw tweets 

• -o: path for the output file (optional)

• -f: use  Damereau-Levenshtein fast computainon (optional, dafault=true)

```
$ sh normalize_tweets.sh -c context2vec.ukwac.model.package/context2vec.ukwac.model.params -i CorpusBataclan_en.1M.raw.tx -f true
```
