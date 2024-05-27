import nltk
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from nltk.corpus import wordnet as wn

import spacy

# Descargar recursos necesarios de NLTK
nltk.download('punkt')
nltk.download('wordnet')

# Inicializar lematizador de spaCy y stemming de NLTK
nlp = spacy.load('es_core_news_sm')
stemmer = PorterStemmer()

# Ejemplo de texto
text = "¡Hola! Este es un ejemplo de preprocesamiento de texto. ¿Cómo estás hoy?"

# Tokenización
tokens = word_tokenize(text)

# Lematización con spaCy
lemmas_spacy = [token.lemma_ for token in nlp(text)]

# Stemming con NLTK
stems_nltk = [stemmer.stem(token) for token in tokens]

# Imprimir resultados
print("Original Tokens:", tokens)
print("Lemmatization (spaCy):", lemmas_spacy)
print("Stemming (NLTK):", stems_nltk)