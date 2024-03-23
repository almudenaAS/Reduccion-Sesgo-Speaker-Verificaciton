import sys
import os
import numpy as np
import pandas as pd
import csv
from collections import OrderedDict
import matplotlib.pyplot as plt
import random
from sklearn.model_selection import train_test_split 

def Vox2Fair_train(meta, fout, Lan):
    df_meta = pd.read_csv(meta)
    with open(fout, 'w', newline='') as f_out:
        for _, row in df_meta.iterrows():
            id, audio = row.audio.split('/')
            linea = id + ' ' + Lan+'/'+row.audio
            f_out.write(linea)
            f_out.write('\n')

def Vox2Fair_test(meta, fout, Lan):
    df_meta = pd.read_csv(meta)
    with open(fout, 'w', newline='') as f_out:
        for _, row in df_meta.iterrows():
            linea = str(row.label) + ' ' + Lan+'/'+row.audio_1 + ' ' + Lan+'/'+row.audio_2
            f_out.write(linea)
            f_out.write('\n')  


# id_user,hash_user,language_l1,language_l2,gender,age,accent,n_sensitive,n_sample
meta_fem = './Esp/female_test.csv'
accents=[]
meta_fem = pd.read_csv(meta_fem) 
for _, row in meta_fem.iterrows():
    if str(row.accent) not in accents: 
        accents.append(str(row.accent))

meta_man = './Esp/male_test.csv'
meta_man = pd.read_csv(meta_man) 
for _, row in meta_man.iterrows():
    if str(row.accent) not in accents: 
        accents.append(str(row.accent))


for acento in accents:
    file = open('./Esp/list_'+acento+'.txt', 'w')
    for _, row in meta_fem.iterrows():
        if row.accent == acento:
            for index in range(5):
                if index < row.n_sample:
                    n=0
                    while True:
                        n+=1
                        num = '0' * n + str(index)
                        if len(num)==5:
                            break
                    aux=0
                    while aux <=9:
                        random_id = np.random.choice(meta_fem.id_user)
                        if random_id != row.id_user:
                            linea = '0' + ' ' + row.language_l1+'/'+row.id_user + '/' + 'audio'+ num + '.wav' + ' ' + row.language_l1+'/'+random_id + '/' + 'audio'+ num + '.wav' + '\n'
                            file.write(linea)
                            aux+=1

                    linea = '1' + ' ' + row.language_l1+'/'+row.id_user + '/' + 'audio'+ num + '.wav' + ' ' + row.language_l1+'/'+row.id_user + '/' + 'audio'+ num + '.wav' + '\n'
                    file.write(linea)
                else: 
                    break
    
    for _, row in meta_man.iterrows():
        if row.accent == acento:
            for index in range(5):
                if index < row.n_sample:
                    n=0
                    while True:
                        n+=1
                        num = '0' * n + str(index)
                        if len(num)==5:
                            break
                    aux=0
                    while aux <=9:
                        random_id = np.random.choice(meta_man.id_user)
                        if random_id != row.id_user:
                            linea = '0' + ' ' + row.language_l1+'/'+row.id_user + '/' + 'audio'+ num + '.wav' + ' ' + row.language_l1+'/'+random_id + '/' + 'audio'+ num + '.wav' + '\n'
                            file.write(linea)
                            aux+=1

                    linea = '1' + ' ' + row.language_l1+'/'+row.id_user + '/' + 'audio'+ num + '.wav' + ' ' + row.language_l1+'/'+row.id_user + '/' + 'audio'+ num + '.wav' + '\n'
                    file.write(linea)
                else: 
                    break