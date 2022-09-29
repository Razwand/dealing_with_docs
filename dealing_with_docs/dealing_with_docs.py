#!/usr/bin/env python
# coding: utf-8

from split_pdf import *
from get_ruptures import *

from os import listdir
from os.path import join, isfile
import sys

import time

import pandas as pd
import numpy as np
from collections import Counter

import cv2
import imutils

from fpdf import FPDF
import warnings
warnings.simplefilter("ignore")


##############################################
#  GENERAL TOOLS                             #
##############################################

def build_strings(sample, type_string,action):
    '''
    This function generates a  string needed (input path/output file).
    - sample: string that contains the volume directory name 
    - type_string: input/output, depending on the directory/file we are pointing at
    '''
    if type_string == 'input':
        return('./input_pages/pdf_pages_' + sample + '/')
    elif type_string == 'output':
        return('./output/Final_Result_'+sample+'_'+action+'.csv')

def build_df_total(path):

    '''
    These functions builds a table with all the features that will be extracted.
    Two columns will be created:

    - ID identifying each one of the volume pages
    - Numeric column in order to be able to order them based on the page number.

    The file name where each page will be stored have the following structure:
    
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
    This function selects the filtered pages and returns a volume version with only these pages and 
    saves it in the output folder.

    The name of the file will be:
     <sample>_filtered.pdf
    '''
    
    path = './input_pages/pdf_pages_'+sample +'/'
    pdf = FPDF()
    for element in list_pages:
        pdf.add_page()
        pdf.image(path + element,0,10,210,297)
    pdf.output('./output/'+ sample + '_' + name_doc + ".pdf", "F")

def alturas_anchuras_docs_tomo(df, path):
    '''
    A function that detects which height/width is common for pages in the volume
    '''
    pages = df['Page_Volume']
    anchuras = []
    alturas = []
    for page in pages:
        im = cv2.imread(path + page)
    
        anchuras.append(im.shape[1])
        alturas.append(im.shape[0])
        
    max_alt = most_frequent(alturas)
    max_anch = most_frequent(anchuras)
        
    return(max_alt,max_anch)

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
    Function that initiates parameters that will be used when detecting the shield
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
    Function that initiates parameters that will be used when detecting the shield
    '''

    lim_inf_siz_pag = 1500
    lim_sup_siz_pag= 2000
    per_new_h = 0.2
    per_new_w = 0.2
    area_limit_sup = 40000

    return(extract_shield(img,lim_inf_siz_pag, lim_sup_siz_pag, per_new_h,per_new_w, area_limit_sup))

# Detect size change
    
def extract_size_change(img,  max_width, max_height):
    '''
    This function will return 1 if a change in size is detected, 0 otherwise.
    The common size for pages in the volume is taken from max_width and max_height
    setting an error margin.

    This margin is greater in terms of height given the nature of scanned documents 
    (that sometimes have a normal size but the scanning procedure has made the image to be extended vertically)

    '''
    anchura = img.shape[1]
    altura = img.shape[0]

    if altura in range(max_height - 1000, max_height + 1000) and anchura in range(max_width - 50 , max_width + 50):
        return(0)
    else:
        return(1)

# Detect colour change
def most_frequent(llist):
    '''
    A function that takes the most frequent value in the list
    '''
    occurence_count = Counter(llist)
    return occurence_count.most_common(1)[0][0]
def count(h,w,img,manual_count):
    '''
    A function that goes through the img and counts the presence of R,G,B values
    per pixel
    '''
    for y in range(0, h):
        for x in range(0, w):
            RGB = (img[x, y, 2], img[x, y, 1], img[x, y, 0])
            if RGB in manual_count:
                manual_count[RGB] += 1
            else:
                manual_count[RGB] = 1
    return(manual_count)

def extract_background_color(img):
    '''
    This function will return 1 if a change in background is detected, 0 otherwise.
    '''  
    new_h = int(np.round(img.shape[0]*0.1))  
    img= img[0:new_h, 0:]

    new_w = int(np.round(img.shape[1]*0.9)) 
    img = img[0:,new_w:]

    manual_count = {}
    w, h, channels = img.shape
    manual_count = count(h,w,img,manual_count)
    number_counter = Counter(manual_count).most_common(1)

    if number_counter[0][0]!= (255, 255, 255):
        return(1)
    else:
        return(0)


# Main Functions

def extract_page_info(path_img, obj, max_width=None, max_height=None):

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
    elif obj == 'back':
        return(extract_background_color(im_orig.copy()))
    elif obj == 'size':
        return(extract_size_change(im_orig.copy(),  max_width, max_height))


def deal_with_dfs(path,df_volume, action, max_height = None,max_width=None):
    
    '''
    This function starts with the dataframe storing all pages from the volume and stores
    each of the features detected.
    When the new columns are created they are also filtered in terms of what's needed.
    In this case:
    - Empty pages are discarded
    - Shield pages are discarded
    '''

    df_volume['Empty_Page'] = df_volume['Page_Volume'].apply(lambda x: extract_page_info(path + x, 'emp_page'))
    df_volume = df_volume[(df_volume['Empty_Page'] == 0)]
                                                
    df_volume['SHIELD'] = df_volume['Page_Volume'].apply(lambda x: extract_page_info(path + x, 'shield'))

    if action == 'clean_keep':
            df_volume = df_volume[df_volume['SHIELD'] == 1] 
    elif action == 'clean_discard':
            df_volume = df_volume[df_volume['SHIELD'] == 0] 
    else:
        df_volume['BACKGROUND'] = df_volume['Page_Volume'].apply(lambda x: extract_page_info(path + x, 'back'))
        df_volume['SIZE'] = df_volume['Page_Volume'].apply(lambda x: extract_page_info(path + x, 'size', max_height, max_width))


    df_volume['Page_Num'] = df_volume['Page_Num'].astype(int)
    df_volume = df_volume.sort_values(by=['Page_Num'])
    
 
    return(df_volume)

def detector_flow(sample, action):

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
    path = build_strings(sample, 'input',action)
    df_volume = build_df_total(path)


    if action == 'clean_keep' or action == 'clean_discard':
        df_volume = deal_with_dfs(path,df_volume,action)
        list_pages = df_volume['Page_Volume']
        build_df(list_pages,sample,'filtered')

        print('------------------------------------------------------') 
        print('\U0000270C Filtered Dataframe shape with size {}, time {}'.format(df_volume.shape[0],time.time()-time0))
        print('------------------------------------------------------')

        df_volume['Page_Volume'].to_csv(build_strings(sample, 'output',action))

    elif action == 'split':
        max_height,max_width = alturas_anchuras_docs_tomo(df_volume, path)
        df_volume = deal_with_dfs(path,df_volume, action,max_height,max_width)
        get_pdfs(sample,df_volume)
        
        print('------------------------------------------------------') 
        print('\U0000270C Subdocuments have been created, time {}'.format(time.time()-time0))
        print('------------------------------------------------------')


def initialize_pdf_mode(argv):

        'This function checks the existence of a given folder and if it contains the data needed (case PDF)'
    
        main_folder = './input_volume/'
        if os.path.isdir('input_volume') and len(os.listdir(main_folder) ) != 0 and len(os.listdir(main_folder+ argv[1]) ) != 0:
           return(True)
        else:
            print('Please, include the pdf folder inside the directory /input_volume')
            return(False)

def initialize_img_mode(argv):
        'This function checks the existence of a given folder and if it contains the data needed (case IMG)'


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
    if len(argv) != 4 or argv[2] not in ['PDF', 'IMG'] or argv[3] not in ['split','clean_keep','clean_discard']:
         print('\U0001F914  Incorrect number of arguments. Arguments should be: name of the directory to be treated and execution mode (PDF or IMG) and action')     
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
        action = sys.argv[3]

        if mode == 'PDF'and initialize_pdf_mode(sys.argv):
            print('Correct Input!')
            print('------------------------------------------------------')
            print('\U0001F643 You are about to process {} volume in mode {}'.format(sample,mode))
            print('------------------------------------------------------')
            create_folder('input_pages')
            pdf_to_image(sample,poppler_path,'./input_pages/')

            detector_flow(sample,action)

        elif mode == 'IMG' and initialize_img_mode(sys.argv):
            print('Correct Input!')
            print('------------------------------------------------------')
            print('\U0001F643 You are about to process {} volume in mode {}'.format(sample,mode))
            print('------------------------------------------------------')
            
            detector_flow(sample,action)

        else:
            print('\U0001F914	Processing will not be performed.')
    else:
        print('WRONG INPUT')
        print('\U0001F914	Processing will not be performed.')




