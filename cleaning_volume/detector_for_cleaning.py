#!/usr/bin/env python
# coding: utf-8

from split_pdf import *

from os import listdir
from os.path import join, isfile
import sys

import time

import pandas as pd
import numpy as np

from Levenshtein import distance as lev
import re

import cv2
import imutils

from fpdf import FPDF
import pytesseract
import warnings
warnings.simplefilter("ignore")


##############################################
#  GENERAL TOOLS                             #
##############################################

def build_strings(sample, type_string):
    '''
    This function generates a  string needed (input path/output file).
    - sample: string that contains the volume directory name 
    - type_string: input/output, depending on the directory/file we are pointing at
    '''
    if type_string == 'input':
        return('./input_pages/pdf_pages_' + sample + '/')
    elif type_string == 'output':
        return('./output/Final_Result_'+sample+'.csv')

def build_df_total(path):

    '''
    This functions builds a table with all the features that will be extracted.
    Two columns will be created:

    - ID identifying each one of the volume pages
    - Numeric column in order to be able to order them based on the page number.

    The file name storing each page will have the following structure:
    
    <name_of_pdf_volume>.pdf_page_<page_number>.jpg.
    '''

    set_of_pages = [f for f in listdir(path) if isfile(join(path, f))]
    
    df = pd.DataFrame()
    df['Page_Volume'] = set_of_pages
    all_pages_num = [x.split('.')[-2].split('_')[-1] for x in set_of_pages]
    
    df['Page_Volume'] = set_of_pages
    df['Page_Num'] = all_pages_num
    
    return(df)

def build_df(list_pages,sample,name_doc):

    '''
    This function selects the filtered pages and return a volume version with only this pages and 
    save it in the output folder.

    The name of the file will be:
     <sample>_filtered.pdf
    '''
    
    path = './input_pages/pdf_pages_'+sample +'/'
    pdf = FPDF()
    for element in list_pages:
        pdf.add_page()
        pdf.image(path + element,0,10,210,297)
    pdf.output('./output/'+ sample + '_' + name_doc + ".pdf", "F")

##############################################
#  FEATURES                                  #
##############################################

# Empty Pages Detector

def take_contours_empty(img):

    '''
    Image contours are detected. <img> corresponds to a given page.
    Some image processing is performed before detecting contours.
    '''
    
    img = img[100:, 100:]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur_bilfil = cv2.bilateralFilter(gray, 15, 80, 80 )
    edged = imutils.auto_canny(blur_bilfil)
    contours = cv2.findContours(image=edged.copy(), mode=cv2.RETR_LIST, method=cv2.CHAIN_APPROX_SIMPLE)
    contours = contours[0] if len(contours) == 2 else contours[1]
    
    return(contours)

def extract_empty_pages(img,contour_limit):

    '''
    Having an image (<img>) corresponding to a single page, if the contours detected are less than 100
    the page is considered to be empty (value 1 will be returned). Else, it is considered to be not-empty (value 0 will be returned).

    '''
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    cnts = take_contours_empty(img)
    
    if len(cnts)<contour_limit: 
        return(1)
    else:
        return(0)

def detect_empty_pages(img):

    '''
    Function that initialize parameters that will be used when detecting the shield
    '''

    contour_limit = 100

    return(extract_empty_pages(img,contour_limit))

# Shields Detector

def extract_shield(img,lim_inf_siz_pag, lim_sup_siz_pag, per_new_h,per_new_w, area_limit_sup):

    '''
    Function that returns 1 if a shield is detected and 0 otherwise.
    
    '''
    # Flag for shield detection
    shield  = 0

    # Filtering images that are not A4
    # imshape[1] = width, imshape[0] = height
    keep_page = img.shape[1]>lim_inf_siz_pag and img.shape[1]<lim_sup_siz_pag 
    
    # Reducing image to left sup corner
    new_h = int(np.round(img.shape[0]*per_new_h)) 
    im_reduced= img[0:new_h, 0:]
    new_w = int(np.round(im_reduced.shape[1]*per_new_w))  
    im = im_reduced[0:,0:new_w]
     
    # BGR to RGB transformation, greyscale  and binary (simplifying pixels over 150 to 255)
    rgb = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY) 
    binary = cv2.threshold(gray, 150, 255,cv2.THRESH_BINARY_INV)[1]  

    # Components
    connectivity = 4 
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary,connectivity , cv2.CV_32S)

    # Width, height of the original image (left sup corner)
    max_alto = binary.shape[0]

    if keep_page == True:
        # For each one of the components... 
        for i in range(1, num_labels):

            # Height, Width and Area
            w = stats[i, cv2.CC_STAT_WIDTH]
            h = stats[i, cv2.CC_STAT_HEIGHT]
            area = stats[i, cv2.CC_STAT_AREA]

            # Filtering Components
            keep_width = w > 80  
            keep_height = h > 90 and h < max_alto 
            keep_area = area < area_limit_sup  
            keep_squared = h-w < 100  
            keep_not_banners = h>=w 

            if keep_width and keep_height and keep_not_banners and keep_area and keep_squared:
                shield = 1

    return(shield)

def detect_shield(img):

    '''
    Function that initialize parameters that will be used when detecting the shield
    '''

    lim_inf_siz_pag = 1500
    lim_sup_siz_pag= 2000
    per_new_h = 0.2
    per_new_w = 0.2
    area_limit_sup = 40000

    return(extract_shield(img,lim_inf_siz_pag, lim_sup_siz_pag, per_new_h,per_new_w, area_limit_sup))


# Main Functions

def extract_page_info(path_img, obj):

    '''
    This function reads an image (path_img) that corresponds to a page and given the search object (obj) 
    a specific extraction function will be called.
    obj:
    - <emp_page>: empty pages search
    - <shield>: shield search
    
    '''
    im_orig = cv2.imread(path_img)
    
    if obj == 'emp_page':
        return(detect_empty_pages(im_orig.copy()))
    elif obj == 'shield':
        return(detect_shield(im_orig.copy()))

def deal_with_dfs(path,df_volume):
    
    '''
    This functions starts from the dataframe storing all pages from the volume and stores
    each of the features detected.
    When the new columns are created they are also filtered in terms of what's needed.
    In this case:
    - Empty pages are discarded
    - Shield pages are discarded
    '''

    df_volume['Empty_Page'] = df_volume['Page_Volume'].apply(lambda x: extract_page_info(path + x, 'emp_page'))
    df_volume = df_volume[(df_volume['Empty_Page'] == 0)]
                                                   
    df_volume['SHIELD'] = df_volume['Page_Volume'].apply(lambda x: extract_page_info(path + x, 'shield'))
    df_volume = df_volume[df_volume['SHIELD'] == 0] 

    df_volume['Page_Num'] = df_volume['Page_Num'].astype(int)
    df_volume = df_volume.sort_values(by=['Page_Num'])
 
    return(df_volume)

def detector_flow(sample):

    '''
    Main function with the following steps:
    1-  Creates output folder if it doesn't exist
    2 - Build needed paths
    3 - Initialized dataframe with all volume pages
    4 - Build columns with extracted features of each one of the pages
    5 - A new filtered pdf is created (no FAX, shields or empty pages)
    6 - A .csv file is saved with a list of resulting pages after the cleaning filter
    '''
    
    time0 = time.time()
    create_folder('output')
    path = build_strings(sample, 'input')

    df_volume = build_df_total(path)
    df_volume = deal_with_dfs(path,df_volume)

    list_pages = df_volume['Page_Volume']
    build_df(list_pages,sample,'filtered')

    print('------------------------------------------------------') 
    print('\U0000270C Filtered Dataframe shape with size {}, time {}'.format(df_volume.shape[0],time.time()-time0))
    print('------------------------------------------------------')

    df_volume['Page_Volume'].to_csv(build_strings(sample, 'output'))

def initialize_pdf_mode(argv):
    
        main_folder = './input_volume/'
        if os.path.isdir('input_volume') and len(os.listdir(main_folder) ) != 0 and len(os.listdir(main_folder+ argv[1]) ) != 0:
           return(True)
        else:
            print('Please, include the pdf folder inside the directory /input_volume')
            return(False)

def initialize_img_mode(argv):

        pages_folder = './input_pages/'
        if os.path.isdir('input_pages') and len(os.listdir(pages_folder) ) != 0 and len(os.listdir(pages_folder+ 'pdf_pages_' + argv[1]) ) != 0:
            return(True)
        else:
            print('Please, include the pages folder inside the directory /input_pages')
            return(False)

def check_args(argv):
    '''
    Function checking input arguments.
    '''
    if len(argv) != 3 or argv[2] not in ['PDF', 'IMG']:
         print('\U0001F4A5 Incorrect number of arguments. Arguments should be: name of the directory to be treated and execution mode (PDF or IMG)')     
    else:
        return(True)

if __name__ == "__main__":

    '''
    Input Arguments:
    - CASE IMG: <sample_name> (the pages folder will be named pdf_pages_<sample_name>) and mode "IMG"
    - CASO PDF: <sample_name> (the folder that contains the pdf volume inside the folder ./input) and mode "PDF"
    '''

    if check_args(sys.argv):
        sample = sys.argv[1]
        mode = sys.argv[2]

        if mode == 'PDF'and initialize_pdf_mode(sys.argv):
            print('Correct Input!')
            print('------------------------------------------------------')
            print('\U0001F4AB You are about to process {} volume in mode {}'.format(sample,mode))
            print('------------------------------------------------------')
            create_folder('input_pages')
            pdf_to_image(sample,poppler_path,'./input_pages/')

            detector_flow(sample)

        elif mode == 'IMG' and initialize_img_mode(sys.argv):
            print('Correct Input!')
            print('------------------------------------------------------')
            print('\U0001F4AB You are about to process {} volume in mode {}'.format(sample,mode))
            print('------------------------------------------------------')
            
            detector_flow(sample)

        else:
            print('\U0001F61E	Processing will not be performed.')
    else:
        print('WRONG INPUT')
        print('\U0001F61E	Processing will not be performed.')




