from random import *
import os

def convert_to_uppercase(array):
    return [item.upper() for item in array]

def add_value_to_end(array, value):
    new_array = array + [value]
    return new_array

u_pwd=input("Enter password: ")
pwd = ['z', 'x', 'c', 'v', 'b', 'n', 'm', 
'a', 's', 'd', 'f', 'g', 'h', 'j', 'k','l', 'ñ',
'q', 'w', 'e', 'r', 't', 'y', 'u', 'i', 'o', 'p']

uppercase_array = convert_to_uppercase(pwd)
for value in uppercase_array:
    pwd = add_value_to_end(pwd, value)

npwd = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '0']
for value in npwd:
    pwd = add_value_to_end(pwd, value)

sigpwd = ['@', '#', '$', '%', '&', '/', '(', ')', '=', '?', 
        '¿', '¡', '!', '*', '.', ';',',', ':', '.', '-', '_', 
        '+', '<', '>', '{', '}', 'Ç']
for value in sigpwd:
    pwd = add_value_to_end(pwd, value)


pw=''
while(pw!=u_pwd):
    pw=""
    for letter in range(len(u_pwd)):
        guees_pwd = pwd[randint(0, len(pwd))]
        pw=str(guees_pwd)+str(pw)
        print(pw)
        print("Cracking pass ... please wait")
        
