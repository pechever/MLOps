# model.py
import pandas as pd
from sklearn.linear_model import LinearRegression
import pickle

# Generar datos de ejemplo
data = {'feature': [1, 2, 3, 4, 5], 'target': [2, 4, 5, 4, 5]}
df = pd.DataFrame(data)

# Entrenar el modelo
model = LinearRegression()
model.fit(df[['feature']], df['target'])

# Guardar el modelo entrenado
with open('modelo_entrenado.pkl', 'wb') as file:
    pickle.dump(model, file)

print("Modelo entrenado y guardado como modelo_entrenado.pkl")