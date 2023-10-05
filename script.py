# import torch
# from torch.autograd import Variable
# import torch.functional as func
# import torch.nn.functional as nnfunc
import numpy as np
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

filePath = 'pepeloni.txt'
with open(filePath, 'r', encoding="utf-8") as file:
    corpus = file.read()

corpus = unidecode(corpus.lower())
corpus = regexSubstitute(r'[^a-z\s]', '', corpus)
corpus = regexSubstitute(r'\s+', ' ', corpus).split()

stopWords = set(stopwords.words('english'))
stopWords.update(['us','whose'])
tokenCorpus = word_tokenize(str.join(" ",[x for x in corpus if x not in stopWords]))

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

dot = graphviz.Graph('Word Graph', engine='neato')
graph = nx.Graph()

# Create the data frame 
for _, row in dataframe.iterrows():
    a = row['A']
    b = row['B']
    w = row['Weight']
    graph.add_edge(a, b, weight = w)

    dot.node(str(row['A']), row['wordA'])
    dot.node(str(row['B']), row['wordB'])
    dot.edge(str(row['A']), str(row['B']), weight= str(w))
dot.render(filename="wordgraph.gv")
#dot.attr(attrs={"overlap": "scale", "splines": "true"})
#dot.view()

# Obtener las posiciones de los nodos para el gráfico
positions = nx.spring_layout(graph)

# Obtener los pesos de las aristas
weights = nx.get_edge_attributes(graph, 'weight')

for node in graph.nodes:
    shortest_path = nx.shortest_path(graph, source=node, weight='weight')
    degree_centrality = nx.degree_centrality(graph)
    betweenness_centrality = nx.betweenness_centrality(graph, weight='weight')
    closeness_centrality = nx.closeness_centrality(graph, distance='weight')
    clustering_coefficient = nx.clustering(graph, weight='weight')

df_metrics = pd.DataFrame(indexWord.items(), columns=['Node', 'word'])
df_metrics['frequency_word'] = df_metrics['word'].map(frequency)
df_metrics['shortest_path'] = df_metrics['Node'].map(shortest_path)
df_metrics['degree_centrality'] = df_metrics['Node'].map(degree_centrality)
df_metrics['betweenness_centrality'] = df_metrics['Node'].map(betweenness_centrality)
df_metrics['closeness_centrality'] = df_metrics['Node'].map(closeness_centrality)
df_metrics['clustering_coefficient'] = df_metrics['Node'].map(clustering_coefficient)
df_metrics['eigenvector_centrality'] = df_metrics['Node'].map(nx.eigenvector_centrality(graph, weight='weight'))
df_metrics = df_metrics.sort_values("eigenvector_centrality", ascending=False)

def class_quartiles(df,name):
    name1 = 'Class_' + name + 'low'
    name2 = 'Class_' + name + 'medium'
    name3 = 'Class_' + name + 'high'
    quartiles = np.percentile(df[name], [25, 50, 75])
    df[name1] = np.where(df[name] >= quartiles[2], 1, 0)
    df[name2] = np.where((df[name] > quartiles[0]) & (df[name] < quartiles[2]), 1, 0)
    df[name3] = np.where((df[name] <= quartiles[0]) | (df[name] >= quartiles[2]), 1, 0)
    return df

df_metrics=class_quartiles(df_metrics,'degree_centrality')
df_metrics=class_quartiles(df_metrics,'betweenness_centrality')
df_metrics=class_quartiles(df_metrics,'closeness_centrality')
df_metrics=class_quartiles(df_metrics,'clustering_coefficient')
df_metrics=class_quartiles(df_metrics,'eigenvector_centrality')

## save the metrics in a csv file
df_metrics.to_csv('metrics.csv', index=False)

## create the dataframe only with the word and all columns beging Class

df_metrics_class=df_metrics.filter(regex='Class', axis=1)

# df_metric aggregate the metrics of each word in THE FIRST COLUMN
df_metrics_class['word']=df_metrics['word']

columnas = df_metrics_class.columns.tolist()
columnas = [columnas[-1]] + columnas[:-1]
df_metrics_class = df_metrics_class[columnas]

# save the dataframe in a csv file
df_metrics_class.to_csv('metrics_class.csv', index=False)