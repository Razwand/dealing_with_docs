
from os import listdir
import pandas as pd
from os.path import join
import os.path
from os.path import isfile
import cv2
import numpy as np
from collections import Counter


def count(h,w,img,manual_count):
        for y in range(0, h):
            for x in range(0, w):
                RGB = (img[x, y, 2], img[x, y, 1], img[x, y, 0])
                if RGB in manual_count:
                    manual_count[RGB] += 1
                else:
                    manual_count[RGB] = 1
        return(manual_count)

def extract_background_color(img_name, path):
    img_path = path + img_name
    img = cv2.imread(img_path, 1)
    
    new_h = int(np.round(img.shape[0]*0.1))  ### PARAM
    img= img[0:new_h, 0:]

    new_w = int(np.round(img.shape[1]*0.9))  ### PARAM
    img = img[0:,new_w:]

    manual_count = {}
    w, h, channels = img.shape


    manual_count = count(h,w,img,manual_count)
    number_counter = Counter(manual_count).most_common(1)


    return(number_counter[0][0])

def build_input(path):

    df = pd.DataFrame()
    df['Pages'] = [f for f in listdir(path) if isfile(join(path, f))]
    return(df)

def check_background_pages(path):

    df = build_input(path)
    df['Colour'] = df['Pages'].apply(lambda x: extract_background_color(x,path))
    df['is_it_white'] = df['Colour'].apply(lambda x: 1 if x == (255,255,255) else 0)
    return(df)

df = check_background_pages('./data/')
