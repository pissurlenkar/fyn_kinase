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
import subprocess
from xgboost import XGBClassifier

# Calculate Application Domain
df = pd.read_csv('Fyn_kinase.csv')
y = df['Activity'].values
X_model = df.iloc[:, 0:100]
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_model, y, test_size=0.3, random_state = 42)
from imblearn.over_sampling import SVMSMOTE
svmsmote = SVMSMOTE(random_state=42)
X_train, y_train = svmsmote.fit_resample(X_train, y_train)
from sklearn.neighbors import NearestNeighbors
k = 5
knn_model = NearestNeighbors(n_neighbors=k)
knn_model.fit(X_train)
distances, indices = knn_model.kneighbors(X_train)
Dk = np.mean(distances)
Sk = np.std(distances)
threshold = Dk + 0.5 * Sk


Predict_Result1 = ''
activity = []
probability = []
AD = []

def convert_df(df):
    return df.to_csv().encode('utf-8')  

st.title('FYN KINASE SCREENING')

# Create a radio button to choose the mode
selected_mode = st.selectbox("Select Mode of Screening", ["Single Mode", "Batch Mode"])

# Depending on the user's choice, display different content
if selected_mode == "Single Mode":
    smiles_input = st.text_input("Enter your structure!")
    if st.button('Result'):
        df1 = pd.DataFrame({'Smiles': smiles_input},index=[0])
        df1['mol'] = df1['Smiles'].apply(lambda x: Chem.MolFromSmiles(x)) 
        df1['mol'] = df1['mol'].apply(lambda x: Chem.AddHs(x))
        # Calculate mol2vec descriptors
        from mol2vec.features import mol2alt_sentence, mol2sentence, MolSentence, DfVec, sentences2vec
        from gensim.models import word2vec
        w2vec_model = word2vec.Word2Vec.load('model_300dim.pkl')
        df1['sentence'] = df1.apply(lambda x: MolSentence(mol2alt_sentence(x['mol'], 1)), axis=1)
        df1['mol2vec'] = [DfVec(x) for x in sentences2vec(df1['sentence'], w2vec_model, unseen='UNK')]
        # Create dataframe 
        X1 = np.array([x.vec for x in df1['mol2vec']])  
        X = pd.concat((pd.DataFrame(X1), df1.drop(['mol2vec', 'mol', 'sentence', 'Smiles'], axis=1)), axis=1)
        #Application of Domain
        distances_screen,_ = knn_model.kneighbors(X)
        Di = np.mean(distances_screen)
        if Di > threshold:
            result = 'Your compound is out of our application domain'
        else:
            # Load pretrained model
            model = joblib.load('model_fyn.pkl')
            y_prediction = model.predict(X.values)
            probs1 = np.round(model.predict_proba(X.values)[:, 1] * 100, 2)
            probs0 = np.round(model.predict_proba(X.values)[:, 0] * 100, 2)
            if y_prediction[0] == 1:
                result = f'Your compound is active with probality of {probs1[0]}%'
            else:
                result = f'Your compound is inactive with probality of {probs0[0]}%'
        st.success(result)
else:
    smiles_input = st.text_area('Enter your structures!')
    data_entries = []
    if smiles_input:
        entries = smiles_input.split('\n')        
        data_entries.extend(entries)
    if st.button('Result'):
        df1 = pd.DataFrame({'Smiles': data_entries})
        for i in range(len(df1)):
            df1['mol'] = df1['Smiles'].apply(lambda x: Chem.MolFromSmiles(x)) 
            df1['mol'] = df1['mol'].apply(lambda x: Chem.AddHs(x))
            from mol2vec.features import mol2alt_sentence, mol2sentence, MolSentence, DfVec, sentences2vec
            from gensim.models import word2vec
            w2vec_model = word2vec.Word2Vec.load('model_300dim.pkl')
            df1['sentence'] = df1.apply(lambda x: MolSentence(mol2alt_sentence(x['mol'], 1)), axis=1)
            df1['mol2vec'] = [DfVec(x) for x in sentences2vec(df1['sentence'], w2vec_model, unseen='UNK')]
        
        # Creat dataframe 
            X1 = np.array([x.vec for x in df1['mol2vec']])  
            X = pd.concat((pd.DataFrame(X1), df1.drop(['mol2vec', 
                                                   'mol',
                                                  'sentence', 'Smiles'], axis=1)), axis=1)
            
            distances_screen,_ = knn_model.kneighbors(X.iloc[[i]])
            Di = np.mean(distances_screen)
            
            if Di > threshold:
                act = 'Not Determine'
                probs = 'Not Determine'
                note = 'Outside of Application Domain'
            else:
                # Load pretrained model
                model = joblib.load('model_fyn.pkl')                 
                y_prediction = model.predict(X.iloc[[i]].values)
                probs1 = np.round(model.predict_proba(X.iloc[[i]].values)[:, 1] * 100, 2)
                probs0 = np.round(model.predict_proba(X.iloc[[i]].values)[:, 0] * 100, 2)                
                if y_prediction[0] == 1:
                    act = 'Active'
                    probs = probs1[0]
                    note = ''
                else:
                    act = 'Inactive'
                    probs = probs0[0]
                    note = ''

            probability.append(probs)
            activity.append(act)
            AD.append(note)

        df3 = pd.DataFrame({
            'Compound': data_entries,
            'Predicted Activity': activity,
            'Probability (%)': probability,
            'Note': AD
            })
        st.dataframe(df3)
        
        st.download_button(
            label="Download results as CSV file",
            data=df3.to_csv(index=False),
            file_name='Results.csv',
            mime='text/csv'
            )