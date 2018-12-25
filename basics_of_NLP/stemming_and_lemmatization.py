# 1. lemmatizer and Stemmer
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.stem import PorterStemmer
import spacy

#### spacy object
spacy_obj = spacy.load('en_md')

#### stemmers
nltk_porter_stemmer = PorterStemmer()

#### lemmatizers
nltk_lem = WordNetLemmatizer()


sentences = ["this is a lovely place","that is a playable delivery","this is amazing life!","I am learning Data Science","My knowledge is getting multiplied"]

for sent in sentences:
	print "\n original = ",sent
	print "\n stemming operations :: \n"
	print "\n stemming = ",nltk_porter_stemmer.stem(sent),"\n"

	print "\n lemmatization operations :: "
	print "\n lemmatized = ",nltk_lem.lemmatize(sent),"\n"
	for tok in spacy_obj(unicode(sent)):
		print "\n spacy lemmatization::"
		print tok, tok.lemma_
