import torch
from torch.autograd import Variable
import numpy as np
import torch.functional as func
import torch.nn.functional as nnfunc
import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
import graphviz
from unidecode import unidecode
import nltk
from nltk.corpus import stopwords, wordnet
from nltk.tokenize import word_tokenize
from nltk.probability import FreqDist
from nltk.stem import WordNetLemmatizer
from re import sub as regexSubstitute

def tagConversion(tag):
    tagsetMapping = {
        'J': wordnet.ADJ,
        'V': wordnet.VERB,
        'N': wordnet.NOUN,
        'R': wordnet.ADV
    }

    start = tag[0].upper()
    tag = tagsetMapping.get(start)
    if tag == None: tag = wordnet.NOUN

    return tag

def freqArray(pairs):
    uniquePairs, counts = np.unique(pairs, axis = 0, return_counts = True)
    counts = counts.reshape(-1, 1)
    pairFreqs = np.concatenate((uniquePairs, counts), axis = 1)
    pairFreqs = np.array(sorted(pairFreqs, key=lambda x: -x[2]))

    return pairFreqs

def conMatrix(pairFreqs):
    dim = len(pairFreqs)
    matrix = np.zeros((dim, dim))
    for i in range(dim):
        matrix[pairFreqs[i][0], pairFreqs[i][1]] = pairFreqs[i][2]

    return matrix

nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('universal_tagset')
nltk.download('stopwords')

filePath = 'test_text.txt'
with open(filePath, 'r') as file:
    corpus = file.read()

corpus = unidecode(corpus.lower())
corpus = regexSubstitute(r'[^a-z\s]', '', corpus)
corpus = regexSubstitute(r'\s+', ' ', corpus)

stopWords = set(stopwords.words('english'))
stopWords.update(['us','whose'])
tokenCorpus = word_tokenize(corpus - stopWords)

patternTags = nltk.pos_tag(tokenCorpus)

lemmatizer = WordNetLemmatizer()
lemmas = []
vocab = set()
for token, tag in patternTags:
    lemma = lemmatizer.lemmatize(token, pos = tagConversion(tag))
    lemmas.append(lemma)
    vocab.add(lemma)

frequency = FreqDist(tokenCorpus)

wordIndex = {}
indexWord = {}
for index, word in enumerate(vocab):
    wordIndex[word] = index
    indexWord[index] = word

window = 2
indexPairs = []
for centerPos in range(len(lemmas)):
    for w in range(-window, window + 1):
        contextPos = centerPos + w
        if contextPos in range(len(lemmas)) and centerPos != contextPos:
            contextIndex = wordIndex[lemmas[contextPos]]
            centerIndex = wordIndex[lemmas[centerPos]]
            indexPairs.append((centerIndex, contextIndex))

freqPairs = freqArray(indexPairs)
np.savetxt('pairFrequency.csv',  freqPairs, delimiter=',', fmt='%d')

matrix = conMatrix(freqPairs)

dataframe = pd.DataFrame(freqPairs)
dataframe.columns = ['A', 'B', 'Weight']
dataframe['wordA'] = dataframe['A'].map(indexWord)
dataframe['wordB'] = dataframe['B'].map(indexWord)

dot = graphviz.Digraph()
graph = nx.Graph()

# Create the data frame 
for _, row in dataframe.iterrows():
    a = row['A']
    b = row['B']
    w = row['Weight']
    graph.add_edge(a, b, weight = w)

    dot.node(row['A'], row['wordA'])
    dot.node(row['B'], row['wordB'])
    dot.edge(row['A'], row['B'], weight = w)

# Obtener las posiciones de los nodos para el gráfico
positions = nx.spring_layout(graph)

# Obtener los pesos de las aristas
weights = nx.get_edge_attributes(graph, 'weight')