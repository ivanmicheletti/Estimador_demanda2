pip install catboost lightgbm

pip install streamlit

#Importación de librería de manipulación y analisis de datos
import pandas as pd
import streamlit as st
import pickle


#importar las librerias necesarias
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import pickle
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor
#pip install catboost lightgbm
import xgboost as xgb
from catboost import CatBoostRegressor
from sklearn.metrics import r2_score

#Lectura del archivo Excel
df=pd.read_csv('https://raw.githubusercontent.com/ivanmicheletti/Estimador_demanda2/refs/heads/main/BD%20-%20Hoja%201.csv')
df

df.isna().sum()

df.columns

df2=df[['Día y Hora', 'Temp amb','MVA totales']]

df2.isna().sum()

df2.isna().sum().sum()

df2.info()

df3 = df2.dropna()

df3.isna().sum().sum()

df3.columns

#Cambio de nombre de las columnas
df3.columns=['tiempo', 'temp_amb', 'MVA_totales']

df3.head()

# Convertir la columna 'fecha' a tipo datetime
df3['tiempo'] = pd.to_datetime(df3['tiempo'])

# Extraer características de la columna 'tiempo'
df3['año'] = df3['tiempo'].dt.year
df3['mes'] = df3['tiempo'].dt.month
df3['día'] = df3['tiempo'].dt.day
df3['hora'] = df3['tiempo'].dt.hour
df3['día_semana'] = df3['tiempo'].dt.dayofweek  # Lunes=0, Domingo=6

# Eliminar la columna 'tiempo' original (si ya no es útil)
df3 = df3.drop(['tiempo','día'], axis=1)

df3.head()

df3.dtypes

# Convert problematic columns to numeric
for column in df3.select_dtypes(include=['object']).columns:
    df3[column] = df3[column].str.replace(',', '.').astype(float)

df3.dtypes

X=df3.drop(['MVA_totales'],axis=1)

y=df3.MVA_totales

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

print('Dimensiones en train \n-X:{}'.format(X_train.shape, y_train.shape))

print('Dimensiones en test \n-X:{}'.format(X_test.shape, y_test.shape))

# Crear y entrenar el modelo
lin_reg_model = LinearRegression()
lin_reg_model.fit(X_train, y_train)

xgboost_model = XGBRegressor()
xgboost_model.fit(X_train,y_train)

catboost_model = CatBoostRegressor(
    learning_rate=0.06,  # Tasa de aprendizaje
    n_estimators=2000,   # Número de árboles
    verbose=100          # Frecuencia de logueo en la consola (opcional, puedes ajustarlo)
)

catboost_model.fit(X_train, y_train)

# Evaluar el modelo
y_pred = lin_reg_model.predict(X_test)
y_pred2= xgboost_model.predict(X_test)
y_pred3 = catboost_model.predict(X_test)

#Calculo la métrica para validar el modelo:

r2_lineal = r2_score(y_test, y_pred)
r2_xgb = r2_score(y_test, y_pred2)
r2_cat =r2_score(y_test, y_pred3)

print(f"R2 Score: {r2_lineal}")
print(f"R2 Score: {r2_xgb}")
print(f"R2 Score: {r2_cat}")

with open('lin_reg.pkl', 'wb') as li:
    pickle.dump(lin_reg_model, li)

with open('xgboost.pkl', 'wb') as xg:
    pickle.dump(xgboost_model, xg)

with open('catboost.pkl', 'wb') as cat:
    pickle.dump(catboost_model, cat)