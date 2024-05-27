import tensorflow as tf
import numpy as np
import pandas as pd
import plotly.express as px

# Generar datos de ejemplo usando TensorFlow
# Definir una función simple para generar datos
def generate_data(num_points):
    x = np.linspace(0, 10, num_points)
    y = 2 * x + 1 + np.random.normal(scale=2, size=num_points)
    return x, y

# Generar datos
num_points = 100
x, y = generate_data(num_points)

# Crear un DataFrame con los datos generados
df = pd.DataFrame({
    'X': x,
    'Y': y
})

# Gráfico de líneas
fig_line = px.line(df, x='X', y='Y', title='Gráfico de Líneas')
fig_line.show()

# Gráfico de dispersión
fig_scatter = px.scatter(df, x='X', y='Y', title='Gráfico de Dispersión')
fig_scatter.show()

# Generar categorías para un gráfico de barras
categories = ['A', 'B', 'C', 'D']
values = np.random.randint(5, 20, size=len(categories))
df_bar = pd.DataFrame({
    'Category': categories,
    'Values': values
})

# Gráfico de barras
fig_bar = px.bar(df_bar, x='Category', y='Values', title='Gráfico de Barras')
fig_bar.show()