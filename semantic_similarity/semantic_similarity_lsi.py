#### This code finds the semantic similarity between the the given query and the documents (list of sentences) and gives the output in descending order
from __future__ import print_function
from gensim import corpora,models,similarities
from collections import defaultdict
from nltk.corpus import stopwords
from collections import Counter

documents = ['car insurance',
		 'car insurance coverage',
		 'auto insurance',
		 'best insurance',
		 'how much is the car insurance',
		 'best auto coverage',
		 'auto policy',
		 'car policy insurance']


stop_words = ([set(stopwords.words('english'))])
texts = [[ word.lower() for word in document.split()
	   if word.lower() not in stop_words]
	   for document in documents]
print("\n texts => ",texts)

freq = defaultdict(int)
print("\n freq => ",freq)
for text in texts:
	print("\n text => ",text)
	for token in text: 
		print("\n token => ",token)
		freq[token] += 1
print("\n freq 2 => ",freq)

# freq2 = dict(Counter(text))
# print ("\n freq 3 => ",freq2)

texts = [[token for token in text if freq[token] > 1] # we can set the minimum frequency
	for text in texts]
print("\n texts with token > 1 => ",texts)

dictionary = corpora.Dictionary(texts)
print("\n dictionary => ",dictionary, dir(dictionary))

# doc2bow counts the number of occurences of each distinct word,
# converts the word to its integer word id and returns the result
# as a sparse vector

corpus = [dictionary.doc2bow(text) for text in texts] #### this is nothing but Document-Term-Matrix
print("\n corpus => ",corpus) 

#### processing user query

query = "i want to buy a best car insurance."
vec_bow = dictionary.doc2bow(query.lower().split())
print("\n vec_bow => ",vec_bow)

################################################# LSI( Latent Semantic Indexing ) #################################
#### link => http://stackoverflow.com/questions/31821821/semantic-similarity-between-phrases-using-gensim

lsi = models.LsiModel(corpus,id2word=dictionary,num_topics=10)

vec_lsi = lsi[vec_bow]
print("\nvec_lsi => ",vec_lsi)
index = similarities.MatrixSimilarity(lsi[corpus])
print("\n index => ",index)

sims = index[vec_lsi]

sims = sorted(enumerate(sims),key=lambda item: -item[1])
print("\n sims => ",sims)

