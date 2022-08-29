#!/usr/bin/env python
# coding: utf-8

from pdf2image import convert_from_path
import os

# [!] PATH A MODIFICAR SEGÚN INSTALACIÓN DE POPPLER
poppler_path = 'C:/Users/asr_l/miniconda3/envs/dealing_with_docs/Lib/site-packages/poppler-0.68.0/bin'

def create_folder(name):
    '''
    Crea una carpeta con el nombre <name> si no existe ya
    '''
    
    if os.path.isdir(name) == False:
        os.mkdir(name)

def pdf_to_image(input_folder_name,poppler_path, output_folder_path):

    '''
    Esta función toma un pdf y dado el nombre de la carpeta de input
    genera imágenes para cada una de las páginas en una carpeta
    
    pdf_pages_<input_folder_name>
    
    El nombre de las imágenes creada seguirá la estructura:

    <nombre_del_tomo>.pdf_page_<numero_de_pagina>.jpg
    '''

    tomo = os.listdir('./input_volume/'+input_folder_name+'/')[0]
    print(tomo)
    out_path_pdf = output_folder_path +'pdf_pages_'+input_folder_name+'/'
    create_folder(out_path_pdf)

    pages = convert_from_path('./input_volume/' + input_folder_name +'/'+tomo, 200,poppler_path=poppler_path)
    
    for i in range(len(pages)):
        path_page_img = output_folder_path +'pdf_pages_'+ input_folder_name +'/' + tomo + '_page_'+str(i)+'.jpg'
        print('Saving page image: ', path_page_img)
        pages[i].save(path_page_img, 'JPEG')     




