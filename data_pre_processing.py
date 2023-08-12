import nltk
from nltk.stem import PorterStemmer
stemmer=PorterStemmer()
import json
import pickle
import numpy as np
words=[]
classes=[]
pattern_word_tags_list=[]
ignore_words=["Eu","!","?",",","."]
trainDataFile=open("intents.json").read()
intents=json.loads(trainDataFile)
def getStemWords(words,ignore_words):
    stem_words=[]
    for i in words:
        if i not in ignore_words:
            w=stemmer.stem(i.lower())
            stem_words.append(w)
    return stem_words
def create_bot_corpus(words, classes, pattern_word_tags_list, ignore_words):

    for intent in intents['intents']:

        # Adicione todos os padrões e tags a uma lista
        for pattern in intent['patterns']:            
            pattern_word = nltk.word_tokenize(pattern)            
            words.extend(pattern_word)                        
            pattern_word_tags_list.append((pattern_word, intent['tag']))
              
        # Adicione todas as tags à lista classes
        if intent['tag'] not in classes:
            classes.append(intent['tag'])
            
    stem_words = getStemWords(words, ignore_words) 
    stem_words = sorted(list(set(stem_words)))
    classes = sorted(list(set(classes)))

    return stem_words, classes, pattern_word_tags_list

