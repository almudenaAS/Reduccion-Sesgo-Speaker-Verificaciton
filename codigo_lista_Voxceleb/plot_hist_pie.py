import sys
import os
import csv
from collections import OrderedDict
from collections import Counter

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

    
def plot_pie_gender_test(male_list, female_list):
    mylabels = ["Male", "Female"]
    file_m = open(male_list, 'r') 
    file_f = open(female_list, 'r') 
    
    id_m=[]
    for linea in file_m:
        linea = linea.strip()
        linea = linea.split(' ')
        id, src, wav = linea[1].split('/')
        if id not in id_m:
            id_m.append(id)
            
    id_f=[]
    for linea in file_f:
        linea = linea.strip()
        linea = linea.split(' ')
        id, src, wav = linea[1].split('/')
        if id not in id_f:
            id_f.append(id)
    
    y = [len(id_m), len(id_f)]
    plt.figure()
    plt.pie(y, labels = mylabels, autopct='%1.1f%%')
    plt.title('Gender Test List - Voxceleb')
       
def plot_pie_accent_test(CA_list, IN_list, IR_list, NO_list, UK_list, USA_list, OT_list):
    mylabels = ["Canadian", "Indian", "Irish", "Norwegian", "British", "American", "Others"]
    y=[]
    
    file= open(CA_list, 'r')
    id_vect=[]
    for linea in file:
        linea = linea.strip()
        linea = linea.split(' ')
        id, src, wav = linea[1].split('/')
        if id not in id_vect:
            id_vect.append(id)
    id_vect = np.array(id_vect)
    y.append(len(id_vect))
    
    file= open(IN_list, 'r')
    id_vect=[]
    for linea in file:
        linea = linea.strip()
        linea = linea.split(' ')
        id, src, wav = linea[1].split('/')
        if id not in id_vect:
            id_vect.append(id)
    id_vect = np.array(id_vect)
    y.append(len(id_vect))
    
    file= open(IR_list, 'r')
    id_vect=[]
    for linea in file:
        linea = linea.strip()
        linea = linea.split(' ')
        id, src, wav = linea[1].split('/')
        if id not in id_vect:
            id_vect.append(id)
    id_vect = np.array(id_vect)
    y.append(len(id_vect))
    
    file= open(NO_list, 'r')
    id_vect=[]
    for linea in file:
        linea = linea.strip()
        linea = linea.split(' ')
        id, src, wav = linea[1].split('/')
        if id not in id_vect:
            id_vect.append(id)
    id_vect = np.array(id_vect)
    y.append(len(id_vect))
    
    file= open(UK_list, 'r')
    id_vect=[]
    for linea in file:
        linea = linea.strip()
        linea = linea.split(' ')
        id, src, wav = linea[1].split('/')
        if id not in id_vect:
            id_vect.append(id)
    id_vect = np.array(id_vect)
    y.append(len(id_vect))
    
    file= open(USA_list, 'r')
    id_vect=[]
    for linea in file:
        linea = linea.strip()
        linea = linea.split(' ')
        id, src, wav = linea[1].split('/')
        if id not in id_vect:
            id_vect.append(id)
    id_vect = np.array(id_vect)
    y.append(len(id_vect))
    
    file= open(OT_list, 'r')
    id_vect=[]
    for linea in file:
        linea = linea.strip()
        linea = linea.split(' ')
        id, src, wav = linea[1].split('/')
        if id not in id_vect:
            id_vect.append(id)
    id_vect = np.array(id_vect)
    y.append(len(id_vect))
     
    plt.figure()
    plt.pie(y, labels = mylabels, autopct='%1.1f%%')
    plt.title('Accent Test List - Voxceleb')
    
def plot_pie_gender_train(meta, train):
    mylabels = ["Male", "Female"]
    with open(meta) as data:
        id_m=[]
        id_f=[]
        n=0
        for fila in csv.reader(data):
            for index, elemento in enumerate(fila):
                if n>0:
                    elemntos = elemento.split(';')
                    VoxCeleb2ID, VGGFace2ID, Gender, Set  = elemntos
                    if Set == 'dev ': 
                        if Gender == 'm ':
                            id, _ = VoxCeleb2ID.split(' ')
                            id_m.append(id)
                        else:
                            id, _ = VoxCeleb2ID.split(' ')
                            id_f.append(id)
                n=+1
    
    id_male=[]; id_female = []
    train = open(train, 'r')
    for linea in train: 
        linea = linea.strip()
        id, wav = linea.split(' ')
        if id in id_m:
            if id not in id_male:
                id_male.append(id)
        elif id in id_f:
            if id not in id_female:
                id_female.append(id)
    
    id_male = np.array(id_male)
    id_female = np.array(id_female)
    
    y = [len(id_male), len(id_female)]   
    plt.figure()
    plt.pie(y, labels = mylabels, autopct='%1.1f%%')
    plt.title('Gender Train List - Histogram')


def main():
    male_list = ('./veri/veri_list_male.txt')
    female_list = ('./veri/veri_list_female.txt')
    meta = ('./metadata-Voxceleb2/vox2_meta.csv')
    train = ('./Train_list.txt')
    plot_pie_gender_test(male_list, female_list)
    
    CA_list = ('./veri/veri_accent_Canada.txt')
    IN_list = ('./veri/veri_accent_India.txt')
    IR_list = ('./veri/veri_accent_Ireland.txt')
    NO_list = ('./veri/veri_accent_Norway.txt')
    UK_list = ('./veri/veri_accent_UK.txt')
    USA_list = ('./veri/veri_accent_USA.txt')
    OT_list = ('./veri/veri_accent_others.txt')
    
    plot_pie_accent_test(CA_list, IN_list, IR_list, NO_list, UK_list, USA_list, OT_list)
    
    plt.show() 


def Voxceleb_distributed(metadata):
    with open(metadata) as data:
        gender_list = []
        n=0
        for fila in csv.reader(data):
            for index, elemento in enumerate(fila):
                if n>0:
                    elemntos = elemento.split('\t')
                    id, vg, gender, nat, src = elemntos
                    gender_list.append(gender) 
                n=+1

    ocurrencias = dict((i, gender_list.count(i)) for i in gender_list)
    
    etiquetas = list(ocurrencias.keys()) #etiquetas
    numeros = list(ocurrencias.values()) #numeros
    
    etiquetas=['Male', 'Female']
    
    plt.figure()
    plt.pie(numeros, autopct='%1.1f%%')
    plt.legend(labels = etiquetas, loc='upper right')
    plt.title('Gender Distributed - Voxceleb')
    
    with open(metadata) as data:
        nacionalidades = []
        n=0
        for fila in csv.reader(data):
            for index, elemento in enumerate(fila):
                if n>0:
                    elemntos = elemento.split('\t')
                    id, vg, gender, nat, src = elemntos
                    nacionalidades.append(nat) 
                n=+1

    ocurrencias = dict((i, nacionalidades.count(i)) for i in nacionalidades)
    lista_tuplas = list(ocurrencias.items())
    lista_tuplas_ordenada = sorted(lista_tuplas, key=lambda x: x[1], reverse=True)
    diccionario_ordenado = dict(lista_tuplas_ordenada)
    
    etiquetas = list(diccionario_ordenado.keys()) #etiquetas
    numeros = list(diccionario_ordenado.values()) #numeros
    
    plt.figure()
    plt.pie(numeros)
    plt.legend(labels = etiquetas[0:5], loc='best')
    plt.title('Accent Distributed - Voxceleb')
    

  
def test():
    meta = ('./metadata-Voxceleb1/vox1_meta.csv')
    Voxceleb_distributed(meta)
    
    plt.show()
   
main()        
   

    
    
    