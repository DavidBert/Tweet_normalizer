
# coding: utf-8
import numpy as np
import sys
import traceback
import re
import nltk
from context2vec.common.model_reader import ModelReader

regex_str = [
    r'<[^>]+>', # HTML tags
    r'(?:@[\w_]+)', # @-mentions
    r"(?:\#+[\w_]+[\w\'_\-]*[\w_]+)", # hash-tags
    r'(?:(?:\d+,?)+(?:\.?\d+)?)', # numbers
    r"(?:[a-z][a-z'\-_]+[a-z])", # words with - and '
    r'(?:[\w_]+)', # other words
    r'(?:\S)' # anything else
]
tokens_re = re.compile(r'('+'|'.join(regex_str)+')', re.VERBOSE | re.IGNORECASE)

def remove_emoji(tweet):
    regx = re.compile(u"[^\U00000000-\U0000d7ff\U0000e000-\U0000ffff]", flags=re.UNICODE)
    return regx.sub(u'', unicode(tweet, 'utf-8')).encode('utf8')

def tokenize(tweet):
    return tokens_re.findall(tweet)

def remove_tweet_RT(tweet):
    cleanned = re.sub('RT @', '@', tweet, count=1)
    return re.sub(r'@.*?:','',cleanned, count=1)

def remove_hex(tweet):
    return re.sub(r'[^\x00-\x7f]',r'', tweet) 

def remove_url(tweet):
    return re.sub(r'http\S+', '', tweet)

def preprocess(tweet):
    tweet = remove_url(tweet)
    tweet = remove_tweet_RT(tweet)
    tweet = remove_emoji(tweet)
    tweet = remove_hex(tweet)
    tokens = tokenize(tweet)
    return tokens

def get_misspelled_words(tokens):
    misspelled_words = []
    for i, word in enumerate(tokens):
        if (not word.lower() in dico) & (not word in dico):
            misspelled_words.append((i, word))
    return misspelled_words                   

def levenshtein(word1, word2):
    l1 = len(word1)
    l2 = len(word2)
    dist = [[0 for x in range(l2 + 1)] for x in range(l1 + 1)]

    for i in range(l1 + 1):
        for j in range(l2 + 1):

            if i == 0:
                dist[i][j] = j    

            elif j == 0:
                dist[i][j] = i  

            elif word1[i - 1] == word2[j -1 ]:
                dist[i][j] = dist[i - 1][j - 1]

            else:
                dist[i][j] = 1 + min(dist[i][j - 1],        
                                   dist[i - 1][j],        
                                   dist[i - 1][j - 1])    

    return dist[l1][l2]

def best_proposition(tokens, target_pos, propositions):

    best_sim = None
    best_answer = None
    context_v = None
    for proposition in propositions:
        if len(tokens) <= 1:
            raise Exception("Can't find context for target word.")
        if proposition in word2index:

            target_v = w[word2index[proposition]]
            context_v = model.context2vec(tokens, target_pos) 
            context_v = context_v / np.sqrt((context_v * context_v).sum())
            sim =  (np.dot(target_v.tolist(), context_v.tolist()) + 1) / 2    
        else:
            sim = 0

        if best_sim is None or sim > best_sim:
            best_sim = sim
            best_answer = proposition

    return best_answer

def find_closest(word):
    res = []
    if len(word) < 2:
        return res
    for other in dico:
        if levenshtein(word, other) < 3:
            res.append(other.encode('utf8'))
    return res

def damereau_levenshtein(word):
    letters    = 'abcdefghijklmnopqrstuvwxyz'
    splits     = [(word[:i], word[i:])    for i in range(len(word) + 1)]
    deletes    = [L + R[1:]               for L, R in splits if R]
    transposes = [L + R[1] + R[0] + R[2:] for L, R in splits if len(R)>1]
    replaces   = [L + c + R[1:]           for L, R in splits if R for c in letters]
    inserts    = [L + c + R               for L, R in splits for c in letters]
    words = set(deletes + transposes + replaces + inserts)

    return list(w for w in words if w in dico)

def correct_tokens(tokens, fast=True):
    misspelled_words = get_misspelled_words(tokens)
    if len(misspelled_words) > 0:
        for (target_pos, word) in misspelled_words:
            if (len(word) == 1) & (not word.isalpha()):
                pass
            else:
                propositions = []
                if fast:
                    propositions = find_closest(word)
                else:
                    propositions = list(damereau_levenshtein(word))
                if (len(propositions) > 0) & (len(tokens) > 1):  
                    correction = best_proposition(tokens, target_pos, propositions + [word])
                    tokens[target_pos] = correction
    return tokens

def correct_tweet(tweet, fast=True):
    tokens = preprocess(tweet)
    return ' '.join(correct_tokens(tokens, fast))

def read_content(path):

    with open(path) as f:
        content = f.readlines()
    content = [x.strip() for x in content]
    return content

def correct_and_save_tweets(dico, c2v_model_path, input_path, output_path, fast=True):
    
    model_reader = ModelReader(c2v_model_path)
    w = model_reader.w
    word2index = model_reader.word2index
    index2word = model_reader.index2word
    model = model_reader.model
    
    tweets = read_content(input_path)
    tweets = tweets[:10]
    
    f = open(output_path,'w')
    for tweet in tweets:
        f.write(correct_tweet(tweet, fast) + '\n')
    f.close()
        

if __name__ == '__main__':

    # Load corpus to creat dictionnary
    print("Loading brown corpus...")
    nltk.download('brown');
    from nltk.corpus import brown
    dico = set(brown.words())
    print("Brown corpus loaded!")

    # Load context2vec Model
    print("Loading context2vec Model...")
    c2v_model_path = sys.argv[1]
    model_reader = ModelReader(c2v_model_path);
    w = model_reader.w
    word2index = model_reader.word2index
    index2word = model_reader.index2word
    model = model_reader.model
    print("Context2vec Model loaded!")
    
    # Load tweets
    print("Loading tweets...")
    input_path = sys.argv[2]
    tweets = read_content(input_path)
    tweets = tweets[:1000]
    print("Tweets loaded!")
    
    #Use the fast method to compute propositions before language model
    fast = sys.argv[4].lower() == 'true'

    # Compute and save normalized tweets
    print("Computing normalized tweets...")
    output_path = sys.argv[3]
    f = open(output_path,'w')
    for tweet in tweets:
        f.write(correct_tweet(tweet, fast) + '\n')
    f.close()
    print("Normalized tweets computed!")
    
    
    
    
    

