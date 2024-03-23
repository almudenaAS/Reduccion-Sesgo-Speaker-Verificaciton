import datetime
import os
import pandas as pd
import random
import argparse

from generate_test_train import *
from funciones_FairVoice import *
from funciones_noFairVoice import * 

def cambinacion_idiomas(train_Eps, train_Eng, path, name):
    fout = open(path+name, 'w')
    
    with open(train_Eps, 'r') as archivo:
        for linea in archivo:
            fout.write(linea)
    
    with open(train_Eng, 'r') as archivo:
        for linea in archivo:
            fout.write(linea)


def cvsFiles_generator():
    file_path='metadata.csv'
    lan = 'English'
    num_of_spk = 486
    min_samples = 5
    split_function(path=file_path,language=lan,num_of_spk=num_of_spk,num_sample=min_samples)

    lan = 'Spanish'
    num_of_spk = 486
    min_samples = 5
    split_function(path=file_path,language=lan,num_of_spk=num_of_spk,num_sample=min_samples)

def trn_metafile(path, lista, name):
    metafile = open(path+'/'+name, 'w')
    file_in = open(lista, 'r')
    Lines = file_in.readlines()
    
    for linea in Lines:
        #Spanish/id00485/audio00000.wav [1] Train
        #Spanish/id00159/audio00000.wav [1] Test
        linea = linea.split(' ')
        
        audio, tipo = linea[1].split('.')
        meta = audio.split('/')
        if len(meta)>2:
            meta = meta[0] + '_'+ meta[1] + '_'+ meta[2] +'\n'
            metafile.write(meta)


def dev_table(path, lista, name, embeddingsName):
    #voxceleb1, embeddings_voxceleb1.npz, veri_test.lst, durations
    table_data=[{'FairVoice', embeddingsName, lista, str(6.0)}]
    np.savez(path+'/'+name, table_data)

    
def main1():
    trn_metafile('./EspEng_List/Balance', './EspEng_List/Balance/list_train.txt', 'metafile_train.txt')
    dev_table('./EspEng_List/Balance', 'list_train.txt', 'dev_table_train', 'embeddingsTrain.npz')
    trn_metafile('./EspEng_List/Balance', './EspEng_List/Balance/list_test.txt', 'metafile_test.txt')
    dev_table('./EspEng_List/Balance', 'list_test.txt', 'dev_table_testn', 'embeddingsTest.npz')


    trn_metafile('./EspEng_List/Balance', './EspEng_List/Balance/list_male.txt', 'metafile_male.txt')
    trn_metafile('./EspEng_List/Balance', './EspEng_List/Balance/list_female.txt', 'metafile_female.txt')
    trn_metafile('./EspEng_List/Balance', './EspEng_List/Balance/list_senior.txt', 'metafile_senior.txt')
    trn_metafile('./EspEng_List/Balance', './EspEng_List/Balance/list_junior.txt', 'metafile_junior.txt')

    trn_metafile('./EspEng_List/Balance', './EspEng_List/Balance/list_juniorFemale.txt', 'metafile_juniorFemale.txt')
    trn_metafile('./EspEng_List/Balance', './EspEng_List/Balance/list_seniorFemale.txt', 'metafile_seniorFemale.txt')
    trn_metafile('./EspEng_List/Balance', './EspEng_List/Balance/list_juniorMale.txt', 'metafile_juniorMale.txt')
    trn_metafile('./EspEng_List/Balance', './EspEng_List/Balance/list_seniorMale.txt', 'metafile_seniorMale.txt')


    trn_metafile('./EspEng_List/noBalance', './EspEng_List/noBalance/list_train.txt', 'metafile_train.txt')
    dev_table('./EspEng_List/noBalance', 'list_train.txt', 'dev_table_train', 'embeddingsTrain.npz')
    trn_metafile('./EspEng_List/noBalance', './EspEng_List/noBalance/list_test.txt', 'metafile_test.txt')
    dev_table('./EspEng_List/noBalance', 'list_test.txt', 'dev_table_testn', 'embeddingsTest.npz')


main1()
def main():
    
    # cvsFiles_generator()
    print('FairVoice')

    cambinacion_idiomas('./EspEng_List/Balance/Esp/test_Esp.txt', './EspEng_List/Balance/Eng/test_Eng.txt', './EspEng_List/Balance/', 'list_test.txt')
    cambinacion_idiomas('./EspEng_List/Balance/Esp/train_Esp.txt', './EspEng_List/Balance/Eng/train_Eng.txt', './EspEng_List/Balance/', 'list_train.txt')
    
    cambinacion_idiomas('./EspEng_List/Balance/Esp/list_female.txt', './EspEng_List/Balance/Eng/list_female.txt', './EspEng_List/Balance/', 'list_female.txt')
    cambinacion_idiomas('./EspEng_List/Balance/Esp/list_male.txt', './EspEng_List/Balance/Eng/list_male.txt', './EspEng_List/Balance/', 'list_male.txt')
    cambinacion_idiomas('./EspEng_List/Balance/Esp/list_senior.txt', './EspEng_List/Balance/Eng/list_senior.txt', './EspEng_List/Balance/', 'list_senior.txt')
    cambinacion_idiomas('./EspEng_List/Balance/Esp/list_junior.txt', './EspEng_List/Balance/Eng/list_junior.txt', './EspEng_List/Balance/', 'list_junior.txt')

    cambinacion_idiomas('./EspEng_List/Balance/Esp/list_seniorFemale.txt', './EspEng_List/Balance/Eng/list_seniorFemale.txt', './EspEng_List/Balance/', 'list_seniorFemale.txt')
    cambinacion_idiomas('./EspEng_List/Balance/Esp/list_seniorMale.txt', './EspEng_List/Balance/Eng/list_seniorMale.txt', './EspEng_List/Balance/', 'list_seniorMale.txt')
    cambinacion_idiomas('./EspEng_List/Balance/Esp/list_juniorFemale.txt', './EspEng_List/Balance/Eng/list_juniorFemale.txt', './EspEng_List/Balance/', 'list_juniorFemale.txt')
    cambinacion_idiomas('./EspEng_List/Balance/Esp/list_juniorMale.txt', './EspEng_List/Balance/Eng/list_juniorMale.txt', './EspEng_List/Balance/', 'list_juniorMale.txt')


    print('No FairVoice')
    cambinacion_idiomas('./EspEng_List/noBalance/Esp/list_test.txt', './EspEng_List/noBalance/Eng/list_test.txt', './EspEng_List/noBalance/', 'list_test.txt')
    cambinacion_idiomas('./EspEng_List/noBalance/Esp/list_train.txt', './EspEng_List/noBalance/Eng/list_train.txt', './EspEng_List/noBalance/', 'list_train.txt')
    
    cambinacion_idiomas('./EspEng_List/noBalance/Esp/list_female.txt', './EspEng_List/noBalance/Eng/list_female.txt', './EspEng_List/noBalance/', 'list_female.txt')
    cambinacion_idiomas('./EspEng_List/noBalance/Esp/list_male.txt', './EspEng_List/noBalance/Eng/list_male.txt', './EspEng_List/noBalance/', 'list_male.txt')
    cambinacion_idiomas('./EspEng_List/noBalance/Esp/list_senior.txt', './EspEng_List/noBalance/Eng/list_senior.txt', './EspEng_List/noBalance/', 'list_senior.txt')
    cambinacion_idiomas('./EspEng_List/noBalance/Esp/list_junior.txt', './EspEng_List/noBalance/Eng/list_junior.txt', './EspEng_List/noBalance/', 'list_junior.txt')

    cambinacion_idiomas('./EspEng_List/noBalance/Esp/list_senior_female.txt', './EspEng_List/noBalance/Eng/list_senior_female.txt', './EspEng_List/noBalance/', 'list_senior_female.txt')
    cambinacion_idiomas('./EspEng_List/noBalance/Esp/list_senior_male.txt', './EspEng_List/noBalance/Eng/list_senior_male.txt', './EspEng_List/noBalance/', 'list_senior_male.txt')
    cambinacion_idiomas('./EspEng_List/noBalance/Esp/list_junior_female.txt', './EspEng_List/noBalance/Eng/list_junior_female.txt', './EspEng_List/noBalance/', 'list_junior_female.txt')
    cambinacion_idiomas('./EspEng_List/noBalance/Esp/list_junior_male.txt', './EspEng_List/noBalance/Eng/list_junior_male.txt', './EspEng_List/noBalance/', 'list_junior_male.txt')



