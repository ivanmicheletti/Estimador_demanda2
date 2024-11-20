#importar librerias
import streamlit as st
import pickle
import pandas as pd

#Extrar los archivos pickle
with open('lin_reg.pkl', 'rb') as li:
    lin_reg = pickle.load(li)


with open('xgboost.pkl', 'rb') as xg:
     xgb = pickle.load(xg)


with open('catboost.pkl', 'rb') as cat:
     catb = pickle.load(cat)

def main():
    #titulo
    st.title('Estimador de demanda')
    #titulo de sidebar
    st.sidebar.header('Parametros de entrada')

    #funcion para poner los parametrso en el sidebar
    def user_input_parameters():
        año = st.sidebar.slider('Año', 2024, 2026, 2024)
        mes = st.sidebar.slider('Mes', 1, 12, 6)
        día_semana = st.sidebar.slider('Día de la semana (0:Lunes, 6:Domingo)', 0, 6, 3)
        hora = st.sidebar.slider('Hora', 0, 24, 12)
        temp_amb = st.sidebar.slider('Temperatura ambiente', -10, 50, 25)
        data = {'temp_amb': temp_amb,
                'año': año,
                'mes': mes,
                'hora': hora,
                'día_semana': día_semana,
                                
                }
        features = pd.DataFrame(data, index=[0])
        return features

    df = user_input_parameters()

    #escoger el modelo preferido
    option = ['Linear Regression', 'xcboots','catboots']
    model = st.sidebar.selectbox('Que modelo le gustaría usar?', option)

    st.subheader('User Input Parameters')
    st.subheader(model)
    st.write(df)

    if st.button('RUN'):
        if model == 'Linear Regression':
            st.success(lin_reg.predict(df))
        elif model == 'xgboost':
            st.success(xgb.predict(df))
        else:
            st.success(catb.predict(df))


if __name__ == '__main__':
    main()