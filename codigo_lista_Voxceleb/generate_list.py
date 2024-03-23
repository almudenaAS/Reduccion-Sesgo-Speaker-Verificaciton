import sys
import os
import numpy as np
import pandas as pd
import csv
from collections import OrderedDict


def calculate_nationalities(metadata): 
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
    ordenado_por_valor = OrderedDict(sorted(ocurrencias.items(), key=lambda x: x[1], reverse=True))
    claves_valores = ordenado_por_valor.items()
    
    mayor_list=[]; minor_list=[]
    for clave, valor in claves_valores:
        if valor > 15:
            mayor_list.append(clave)
        else:
            minor_list.append(clave)
            
    return mayor_list, minor_list

def id_gender(metadata):
    male_list=[]; female_list=[]
    with open(metadata) as data:
        n=0 
        for fila in csv.reader(data):
            for index, elemento in enumerate(fila):
                if n>0: 
                    elemntos = elemento.split('\t')
                    id, vg, gender, nat, src = elemntos
                    if gender == 'm':
                        male_list.append(id)
                    else:
                        female_list.append(id)
                        
                n+=1
    return male_list, female_list

def listID_accent(metadata, mayor_list, minor_list):
    minor_ID=[]; mayor_ID=[]
    with open(metadata) as data:
        n=0
        for fila in csv.reader(data):
            for index, elemento in enumerate(fila):
                if n>0:
                    elemntos = elemento.split('\t')
                    id, vg, gender, nat, src = elemntos
                    if src == 'test':
                        if nat in minor_list:
                            minor_ID.append([id, nat])
                        elif nat in mayor_list:
                            mayor_ID.append([id, nat])
                n+=1
    return mayor_ID, minor_ID     

def split_gender_list(inFile, outDir, mlist, flist):
    archivo_in = open(inFile, 'r')
    archivo_out_male = open(outDir +  'veri_list_male.txt', 'a')
    archivo_out_female = open(outDir +  'veri_list_female.txt', 'a')
    
    for linea in archivo_in:
        linea = linea.strip()
        linea = linea.split(' ')
        #linea[0] = resultado linea[1]= know locutor #linea[2] = unkown locutor
        #resultado: 0 si no son, 1 si son 
        id, src, wav = linea[1].split('/')
        if id in mlist: 
            archivo_out_male.write(linea[0] + ' '+ linea[1] + ' '+ linea[2] +'\n')
        elif id in flist:
            archivo_out_female.write(linea[0] + ' '+ linea[1] + ' '+ linea[2] +'\n')

def split_accent_list(inFile, outDir, mayorList, minorList):
    archivo_in = open(inFile, 'r')
    for linea in archivo_in:
        linea = linea.strip()
        linea = linea.split(' ')
        id, src, wav = linea[1].split('/')
        
        for idSpek1, Land1 in mayorList:
            archivo_out1 = open(outDir + 'veri_accent_' + Land1 +'.txt', 'a')
            if id == idSpek1:
                archivo_out1.write(linea[0] + ' '+ linea[1] + ' '+ linea[2] +'\n')
        
        for idSpek2, Land2 in minorList:
            archivo_out2 = open(outDir + 'veri_accent_others.txt', 'a')
            if id == idSpek2:
                archivo_out2.write(linea[0] + ' '+ linea[1] + ' '+ linea[2] +'\n')
    

def accent_list(inFile, outDir, mayorList, minorList):
    archivo_out2 = open(outDir + 'veri_accent_others.txt', 'a')
    archivo_in = open(inFile, 'r') 
    for idSpek1, Land1 in mayorList:
        archivo_out1 = open(outDir + 'veri_accent_' + Land1 +'.txt', 'a')
        for linea in archivo_in:
            linea = linea.strip()
            linea = linea.split(' ')
            id, src, wav = linea[1].split('/')
            if id == idSpek1:
                archivo_out1.write(linea[0] + ' '+ linea[1] + ' '+ linea[2] +'\n')
    
    for idSpek2, Land2 in minorList:
        for linea in archivo_in:
            linea = linea.strip()
            linea = linea.split(' ')
            id, src, wav = linea[1].split('/')
            if id == idSpek2:
                archivo_out2.write(linea[0] + ' '+ linea[1] + ' '+ linea[2] +'\n')

def test():
    print('Strat Program.............')
    meta = ('./metadata-Voxceleb1/vox1_meta.csv')
    in_veri = ('./list_test_hard.txt')
    out_veri = ('./hard/')
    
    mayor_list, minor_list = calculate_nationalities(meta)
    male_list, female_list = id_gender(meta)
    mayor_ID, minor_ID = listID_accent(meta, mayor_list, minor_list)
    
    print('Start Creating List')
    split_gender_list(in_veri, out_veri, male_list, female_list)
    print('End Gender')
    split_accent_list(in_veri, out_veri, mayor_ID, minor_ID)
    print('End Aceent')
    
    print('Listas Sesgadas DONE!')

test()







