#Importação das bibliotecas
import gradio as gr
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

#Carregar dados
X = pd.read_excel('new_X.xlsx')
y = pd.read_csv('y.csv', sep=';', header=0)
y = np.array(y['Vunit'].values).reshape(-1, 1)

#Normalizar dados
scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()
input_scaler = scaler_new_X.fit(X)
output_scaler = scaler_y.fit(y)
X_norm = new_X_scaler.transform(X)
y_norm = output_scaler.transform(y)
new_X = X_norm
new_y = np.ravel(y_norm)

#Dividir dados em base de treino e de teste
train_dataset, test_dataset, train_labels, test_labels = train_test_split(new_X, new_y, test_size=0.2, random_state=42)

#Transformar dados em DMatrix
dtrain = xgb.DMatrix(train_dataset, train_labels)
dtest = xgb.DMatrix(test_dataset, test_labels)

#Carregar o modelo
loaded_model = xgb.Booster()
loaded_model.load_model("boxes_2021_FINAL.model")

#Fazer as inferências
def predict_box(Atotal, Residencial, Coberta, Idade, DF, B, C, D, E, X, Y):
    df = pd.DataFrame.from_dict({'Atotal': np.log([Atotal]),
                                 'Residencial': [Residencial],
                                 'Coberta': [Coberta],
                                 'Idade': [Idade],
                                 'DF': [DF],
                                 'B': [B],
                                 'C': [C],
                                 'D': [D],
                                 'E': [E],
                                 'X': [X], 
                                 'Y': [Y]})
    df = input_scaler.transform(df)
    df = xgb.DMatrix(df)
    pred = loaded_model.predict(df)
    pred = output_scaler.inverse_transform(np.array(pred).reshape(-1, 1))
    pred = np.exp(pred).tolist()     
    return f"""Valor do m²: R${round(pred[0][0])} | Valor Total do box: R${round(pred[0][0]*Atotal)}"""

#Definir os campos de inserção de dados
Atotal = gr.inputs.Number(default = 20., label="Área Total")
Residencial = gr.inputs.Number(default = 1, label="Residencial")
Coberta = gr.inputs.Number(default = 1, label="Coberta")
Idade = gr.inputs.Number(default = 5, label="Idade")
DF = gr.inputs.Number(default = 1, label="Divisão Fiscal")
B = gr.inputs.Number(default = 0, label="Padrão Construtivo B")
C = gr.inputs.Number(default = 1, label="Padrão Construtivo C")
D = gr.inputs.Number(default = 0, label="Padrão Construtivo D")
E = gr.inputs.Number(default = 0, label="Padrão Construtivo E")
X = gr.inputs.Number(default =274800., label="Latitude")
Y = gr.inputs.Number(default =1.662188e+06, label="Latitude")

#Criar interface do aplicativo
gr.Interface(predict_box, [Atotal, Residencial, Coberta, Idade, DF, B, C, D, E, X, Y], "label", live=False).launch(share=True)
