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
import os
import numpy as np                
import pandas as pd               
import matplotlib.pyplot as plt   
import seaborn as sns
import rdkit
from rdkit.Chem import Draw, Descriptors, rdqueries
from rdkit import RDLogger  
RDLogger.DisableLog('rdApp.*')  
from rdkit import Chem
import random
import requests
import io
import subprocess
from xgboost import XGBClassifier
from PIL import Image
import meeko
from meeko import MoleculePreparation
import subprocess
import re
#from streamlit_ketcher import st_ketcher

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

def read_file(file):
    content = file.read()
    return content
    
#st.title('FYN KINASE SCREENING')
st.markdown("<h1 style='text-align: center; font-size: 2.5em'>FynPred v1.0</h1>", unsafe_allow_html=True)
st.markdown("<h1 style='text-align: center; font-size: 2.5em'>FYN KINASE INHIBITORS SCREENING</h1>", unsafe_allow_html=True)

#st.markdown("<h2 style='text-align: left; font-size: 1.5em; font-style: italic;'>Convert 2D structures to SMILES</h2>", unsafe_allow_html=True)
#smile_code = st_ketcher()
#st.markdown(f"Smile code: ``{smile_code}``")

selected_mode = st.sidebar.selectbox("Select Mode", ["QSAR", "Molecular docking"])

if selected_mode == "QSAR":
    st.sidebar.markdown(
        "<div style='text-align: justify;'>"
        "In our investigation, the classification of a compound as active or inactive for Fyn kinase was conducted using the XGBoost model. A compound was considered active if the anticipated IC<sub>50</sub> was below 1 &micro;M, and inactive if the predicted IC<sub>50</sub> equaled or exceeded 1 &micro;M."
        "</div>",
        unsafe_allow_html=True)
    st.markdown("<h2 style='text-align: left; font-size: 2em; font-style: italic;'>QSAR Model</h2>", unsafe_allow_html=True)
    # Create a radio button to choose the mode
    selected_mode = st.selectbox("Select Mode of Screening", ["Single Mode", "Batch Mode"])

    # Depending on the user's choice, display different content
    if selected_mode == "Single Mode":
        smiles_input = st.text_input("Enter your SMILES string!")
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
                result = 'Your compound is out of our application domain. Please consider using molecular docking for screening this compound!'
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
        smiles_input = st.text_area('Enter your SMILES strings! (One SMILES per line)')
        data_entries = []
        if smiles_input:
            entries = smiles_input.split('\n')        
            data_entries.extend(entries)
        if st.button('Result'):
            df1 = pd.DataFrame({'Smiles': data_entries})
            #structure_images = []
            #img_bytes_list = []
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
                file_name='QSAR results.csv',
                mime='text/csv'
                )

elif selected_mode == "Molecular docking":
    st.sidebar.markdown(
        "<div style='text-align: justify;'>"
        "In our investigation, molecular docking between the ligand and Fyn kinase was carried out using Autodock Vina. For screening purposes, we present only the energy value corresponding to the conformation that exhibits the strongest interaction with the protein."
        "<br><br>"
        "It's worth noting that molecular docking results are notably slower than QSAR. Therefore, we recommend employing QSAR initially for screening potential compounds and subsequently using molecular docking to screen these selected compounds." 
        #"<br><br>"
        #"For reference, the binding energy for the positive control Staurosporine (STU-902), a well-established Fyn kinase inhibitor, is -12.2 kcal/mol."
        "</div>",
        unsafe_allow_html=True)
    st.markdown("<h2 style='text-align: left; font-size: 2em; font-style: italic;'>Molecular docking</h2>", unsafe_allow_html=True)
    selected_mode = st.selectbox("Select Mode of Screening", ["Single Mode", "Batch Mode"])
    if selected_mode == "Single Mode":
        smiles_input = st.text_input("Enter your SMILES string!")
        if st.button('Result'):
            #st.write('Keep calm!')
            lig = Chem.MolFromSmiles(smiles_input)
            if lig is None:
                st.success('Your SMILES is invalid! Please check again!')
            else:
                protonated_lig = Chem.AddHs(lig)
                rdkit.Chem.AllChem.EmbedMolecule(protonated_lig)
                meeko_prep = meeko.MoleculePreparation()
                meeko_prep.prepare(protonated_lig)
                meeko_prep.write_pdbqt_file("ligand_1.pdbqt")
    
                #with open("ligand_1.pdbqt", 'r') as file:
                #    pdbqt_contents = read_file(file)
                #st.code(pdbqt_contents, language='text')
    
                process = subprocess.Popen(['bash', 'run_vina.sh'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    
                # Wait for the process to finish
                stdout, stderr = process.communicate()
                #st.code(stdout.decode(), language='text')

                with open('ligand_1_out.pdbqt', 'r') as file:
                    output_lines = file.readlines()
                # Find the line that starts with "MODEL 1"
                model_1_line_index = next(index for index, line in enumerate(output_lines) if line.startswith('MODEL 1'))
                # Extract the affinity value from the line following "MODEL 1"
                result_line = output_lines[model_1_line_index + 1]
                result_affinity = float(result_line.split()[3])
                st.success(f"The binding energy of your ligand and Fyn kinase is {result_affinity} kcal/mol")
                
                current_dir = os.getcwd()
                file_path_inp = f"{current_dir}/ligand_1.pdbqt"
                file_path_out = f"{current_dir}/ligand_1_out.pdbqt"
                os.remove(file_path_out)
                os.remove(file_path_inp)
                
    else:
        smiles_input = st.text_area('Enter your SMILES strings! (One SMILES per line)')
        data_entries = []
        affinity = []
        if smiles_input:
            entries = smiles_input.split('\n')        
            data_entries.extend(entries)
        if st.button('Result'):
            df1 = pd.DataFrame({'Smiles': data_entries})
            #for i in range(len(df1)):
            for i, row in df1.iterrows():
                #lig = Chem.MolFromSmiles(df1['Smiles'])
                lig = Chem.MolFromSmiles(row['Smiles'])
                protonated_lig = Chem.AddHs(lig)
                rdkit.Chem.AllChem.EmbedMolecule(protonated_lig)
                meeko_prep = meeko.MoleculePreparation()
                meeko_prep.prepare(protonated_lig)
                meeko_prep.write_pdbqt_file(f"ligand_{i+1}.pdbqt")
                process = subprocess.Popen(['bash', 'run_vina.sh'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)

                stdout, stderr = process.communicate()
                #st.code(stdout.decode(), language='text')

                with open(f'ligand_{i+1}_out.pdbqt', 'r') as file:
                    output_lines = file.readlines()
                # Find the line that starts with "MODEL 1"
                model_1_line_index = next(index for index, line in enumerate(output_lines) if line.startswith('MODEL 1'))
                # Extract the affinity value from the line following "MODEL 1"
                result_line = output_lines[model_1_line_index + 1]
                result_affinity = float(result_line.split()[3])
                affinity.append(result_affinity)
                
                current_dir = os.getcwd()
                file_path_inp = f"{current_dir}/ligand_{i+1}.pdbqt"
                file_path_out = f"{current_dir}/ligand_{i+1}_out.pdbqt"
                os.remove(file_path_inp)
                os.remove(file_path_out)
            
            df3 = pd.DataFrame({
                'Compound': data_entries,
                'Binding energy (kcal/mol)': affinity,
                })
            st.dataframe(df3)
            
            st.download_button(
                label="Download results as CSV file",
                data=df3.to_csv(index=False),
                file_name='Molecular docking results.csv',
                mime='text/csv'
                )

st.markdown(
    """
    <div style="position: fixed; bottom: 8px; width: 100%; text-align: left; padding-left: 5cm;">
        Created by Nguyen-Van Phuong, et al. (2023)
    </div>
    """,
    unsafe_allow_html=True
)
