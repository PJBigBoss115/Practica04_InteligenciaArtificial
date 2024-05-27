import nltk
from nltk import ngrams
from collections import Counter
import matplotlib.pyplot as plt
import spacy
import re
import pandas as pd

# Descargar recursos necesarios de NLTK
nltk.download('punkt')
nltk.download('stopwords')

# Inicializar el modelo de spaCy para español
nlp = spacy.load('es_core_news_sm')

# Función para preprocesar el texto
def preprocess_text(text):
    # Convertir a minúsculas
    text = text.lower()
    # Eliminar caracteres especiales y números
    text = re.sub(r'[^a-záéíóúñü\s]', '', text)
    text = re.sub(r'\d+', '', text)
    return text

# Función para extraer bigramas y trigramas
def extract_ngrams(text, n):
    tokens = nltk.word_tokenize(text)
    n_grams = ngrams(tokens, n)
    return [' '.join(grams) for grams in n_grams]

# Función para visualizar los n-gramas más frecuentes
def plot_ngrams(ngrams, n, title):
    ngram_counts = Counter(ngrams)
    common_ngrams = ngram_counts.most_common(n)
    ngram_df = pd.DataFrame(common_ngrams, columns=['Ngram', 'Count'])
    plt.figure(figsize=(10, 6))
    plt.bar(ngram_df['Ngram'], ngram_df['Count'])
    plt.title(title)
    plt.xticks(rotation=45, ha='right')
    plt.show()

# Cargar y preprocesar el documento de texto
with open('documento.txt', 'r', encoding='utf-8') as file:
    text = file.read()

preprocessed_text = preprocess_text(text)

# Extraer y visualizar los 20 bigramas más frecuentes
bigrams = extract_ngrams(preprocessed_text, 2)
plot_ngrams(bigrams, 20, 'Top 20 Bigrams')

# Extraer y visualizar los 20 trigramas más frecuentes
trigrams = extract_ngrams(preprocessed_text, 3)
plot_ngrams(trigrams, 20, 'Top 20 Trigrams')