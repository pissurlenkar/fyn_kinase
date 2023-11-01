# -*- coding: utf-8 -*-
"""
Created on Tue Oct 31 15:02:29 2023

@author: WINDOWS
"""
import pandas as pd
import pickle
import streamlit as st
from streamlit_option_menu import option_menu
import joblib
import pubchempy as pcp
import numpy as np                
import pandas as pd               
import matplotlib.pyplot as plt   
import seaborn as sns
from rdkit.Chem import Draw, Descriptors, rdqueries
from rdkit import RDLogger  
RDLogger.DisableLog('rdApp.*')  
from rdkit import Chem
import random
import requests
import io


st.title('FYN KINASE SCREENING')

smiles_input = st.text_area("Enter your structure!")

data_entries = []
if smiles_input:
    entries = smiles_input.split('\n')
    
    data_entries.extend(entries)

for smiles in data_entries:
    st.write(smiles)
    
df = pd.read_csv('Fyn_kinase.csv')
Predict_Result1 = ''
results1 = []

if st.button('Result') and data_entries: 
    #longle = df.loc[(df['Smiles'] == smiles_input)]
    df1 = pd.DataFrame({'Smiles':data_entries})
    for i in range(len(df1)):
             df1['mol'] = df1['Smiles'].apply(lambda x: Chem.MolFromSmiles(x)) 
             df1['mol'] = df1['mol'].apply(lambda x: Chem.AddHs(x))
    #
             from mol2vec.features import mol2alt_sentence, mol2sentence, MolSentence, DfVec, sentences2vec
             from gensim.models import word2vec
             w2vec_model = word2vec.Word2Vec.load('model_300dim.pkl')
             df1['sentence'] = df1.apply(lambda x: MolSentence(mol2alt_sentence(x['mol'], 1)), axis=1)
             df1['mol2vec'] = [DfVec(x) for x in sentences2vec(df1['sentence'], w2vec_model, unseen='UNK')]
    
    # Create dataframe 
             X1 = np.array([x.vec for x in df1['mol2vec']])  
             X = pd.concat((pd.DataFrame(X1), df1.drop(['mol2vec', 
                                               'mol',
                                              'sentence', 'Smiles'], axis=1)), axis=1)
# Load pretrained model
             model = joblib.load('Desktop/model_fyn.pkl')
              
             y_prediction = model.predict(X.iloc[[i]].values)
             probs1 = np.round(model.predict_proba(X.iloc[[i]].values)[:, 1] * 100, 2)
             probs0 = np.round(model.predict_proba(X.iloc[[i]].values)[:, 0] * 100, 2)
             #if y_prediction[0] == 1: 
              #   Predict_Result1 = f'Kết quả dự đoán cho hàng {i + 1}: Incompatible. Xác suất: {probs1[0]}%'
             #else: 
              #   Predict_Result1 = f'Kết quả dự đoán cho hàng {i + 1}: Compatible. Xác suất: {probs0[0]}%'
             if y_prediction[0] == 1:
                 result = ('Active', f'{probs1[0]}%')
             else:
                 result = ('Inactive', f'{probs0[0]}%')
             results1.append(result)
longle = results1
for i, (prediction, probability) in enumerate(longle):
    st.success(f'Kết quả dự đoán cho hàng {i + 1}: {prediction}. Xác suất: {probability}')
#Hiển thị dataframe
df3 = pd.DataFrame(longle)
st.dataframe(df3)
# Xuất file
def convert_df(df):
    return df.to_csv().encode('utf-8')  
csv = convert_df(pd.DataFrame(longle, columns=['Prediction', 'Probability']))

st.download_button(label="Download results of prediction as CSV",
                    data=csv, file_name='Results.csv',
                    mime='text/csv',
                )


