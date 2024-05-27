import spacy

# Cargar el modelo de spaCy para español
nlp = spacy.load('es_core_news_sm')

# Ejemplo de texto
text = "¡Hola! Este es un ejemplo de preprocesamiento de texto. ¿Cómo estás hoy?"

# Procesar el texto con spaCy
doc = nlp(text)

# Extraer y mostrar tokens y sus etiquetas PoS
for token in doc:
    print(f'Token: {token.text}, PoS: {token.pos_}, Lemma: {token.lemma_}')

# También puedes extraer solo los tokens y sus etiquetas PoS en una lista
tokens_pos = [(token.text, token.pos_, token.lemma_) for token in doc]

print(tokens_pos)