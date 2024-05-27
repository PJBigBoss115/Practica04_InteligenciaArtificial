import nltk
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# Descargar recursos necesarios de NLTK
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Ejemplo de texto
text = "¡Hola! Este es un ejemplo de preprocesamiento de texto. ¿Cómo estás hoy?"

# 1. Eliminación de caracteres especiales
text = re.sub(r'[^\w\s]', '', text)

# 2. Tokenización
tokens = word_tokenize(text)

# 3. Conversión a minúsculas
tokens = [token.lower() for token in tokens]

# 4. Eliminación de stopwords
stop_words = set(stopwords.words('spanish'))
tokens = [token for token in tokens if token not in stop_words]

# 5. Lematización
lemmatizer = WordNetLemmatizer()
tokens = [lemmatizer.lemmatize(token) for token in tokens]

# Resultado del preprocesamiento
preprocessed_text = ' '.join(tokens)
print(preprocessed_text)