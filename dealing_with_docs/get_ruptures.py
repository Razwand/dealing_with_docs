#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
import os
from os import listdir
from os.path import isfile, join

from fpdf import FPDF

def get_rupture(df, col, name_col_changes, name_col_rupture):
    '''
    Esta función realiza una transformación de la tabla original de forma
    ---------------------
    COLUMN_1
    ---------------------
    0
    1
    1
    0
    0

    Donde cada uno marca la aparación de la característica tratada en dicha columna.

    Se pasa a transformar una tabla que marca cambios de 1 a 0 o de 0 a 1 en la secuencia
    que marca la columna. Así la columna reflejada arriba como ejemplo quedaría:

    ---------------------
    COLUMN_1_CHANGES
    ---------------------
    0
    1
    0
    1
    0
    '''

    df.groupby((df[col].shift() != df[col]).cumsum())
    group_list = []
    for k, v in df.groupby((df[col].shift() != df[col]).cumsum()):
        for element in v['Page_Num']:
               group_list.append(k)
    df[name_col_changes] = group_list
    df[name_col_rupture] = np.where(df[name_col_changes] == df[name_col_changes].shift(), 0, 1)
    
    df.drop(columns=name_col_changes)
    
    return(df)
 
def create_folder(name):
    '''
    Crea una carpeta con el nombre <name> si no existe ya
    '''  
    if os.path.isdir(name) == False:
        os.mkdir(name)

def build_rupture_column(df): 
    '''
    Se crea la columna de ruptura en base a la tabla obtenida de 
    características marcadas (df), se filtran fax/informes y páginas vacías y
    se consideran los escudos detectados en función de si el texto extraído de éstos
    es el esperado.
    '''        
   
    # Filtrado Vacíos, Faxes, Informes
    df =  df[(df['Empty_Page']==0)]
        
    columns_to_keep = ['Page_Num','SIZE','BACKGROUND','SHIELD']
    df = df[columns_to_keep]
        
    df_rupture_esc_esp = get_rupture(df, 'SHIELD', 'CHANGE_SHIELD', 'RUPTURE_SHIELD')[['Page_Num','RUPTURE_SHIELD']]    
    df_rupture_size = get_rupture(df, 'SIZE', 'CHANGE_SIZE', 'RUPTURE_SIZE')[['Page_Num','RUPTURE_SIZE']]
    df_rupture_background = get_rupture(df, 'BACKGROUND', 'CHANGE_BACK', 'RUPTURE_BACK')[['Page_Num','RUPTURE_BACK']]

    df_final = df_rupture_size.merge(df_rupture_background,how ='left').merge(df_rupture_esc_esp,how ='left')

    df_final = df_final.set_index('Page_Num')
    df_final['Document'] = 1
    df_final['Document'][(df_final['RUPTURE_SIZE'] == 0) & (df_final['RUPTURE_BACK'] == 0)  &  (df_final['RUPTURE_SHIELD'] == 0)] = 0

    
    return(df_final)

def build_df(great_list,df_names,sample):

    '''
    Esta función agrupa las páginas donde no hay cambios
    según la columna genérica de ruptura en pdfs que representarán
    los subdocumentos extraídos de la ruptura
    
    '''

    path = './input_pages/pdf_pages_'+sample
    doc = 0
    create_folder('output')
    for element in great_list:
        pdf = FPDF()
        for page in element:
            name_page = df_names[df_names['Numbers']==page]['Original_File'].values[0]
            complete_path = path +'/'+ name_page
            pdf.add_page()
            pdf.image(complete_path,0,10,210,297)
        pdf.output("./output/doc"+str(doc)+"_pags_"+str(element[0])+"-"+str(element[-1])+".pdf", "F")
        doc = doc + 1

def get_pdfs(sample,df):

    '''
    Función general encargada de tomar la tabla que marca las características
    por página del tomo (df) y genera pdfs de subdocumentos según las rupturas marcadas.
    '''
    df =  build_rupture_column(df) 

    great_list = []
    list_sub_doc = []
    for ele in df.index:
        if ele != df.index[0]: # Si no es el elemento de la primera fila
            if df['Document'].loc[ele] == 1:
                great_list.append(list_sub_doc)
                list_sub_doc = []
                list_sub_doc.append(ele)
            else:
                list_sub_doc.append(ele)
            if ele == df.index[df.shape[0]-1]:# si es el último elemento
                great_list.append(list_sub_doc)

        else: # Si es el elemento de la primera fila
            list_sub_doc.append(ele)
        list_flat = [x for xs in great_list for x in xs]

    path = './input_pages/pdf_pages_'+sample

    image_name_list = [f for f in listdir(path) if isfile(join(path, f))]

    pags_escudos = [x.split('.')[-2] for x in image_name_list]
    pags_escudos = [x.split('_')[-1] for x in pags_escudos]

    df_names = pd.DataFrame()
    df_names['Original_File'] = image_name_list
    df_names['Numbers'] = pags_escudos
    df_names['Numbers']  = df_names['Numbers'].astype(int)
    df_names[df_names['Numbers'].isin(list_flat)]
    
    build_df(great_list,df_names,sample)






