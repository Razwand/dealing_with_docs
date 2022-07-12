
from os import listdir
from tkinter import W
import pandas as pd
from os.path import join
import os.path
from os.path import isfile
import cv2
import numpy as np
from collections import Counter


class Page:
    def __init__(self, path, name):
        self.path = path
        self.name = name


    def extract_background_color(self):
        img_path = self.path + self.name
        img = cv2.imread(img_path, 1)
        
        new_h = int(np.round(img.shape[0]*0.1))  ### PARAM
        img= img[0:new_h, 0:]

        new_w = int(np.round(img.shape[1]*0.9))  ### PARAM
        img = img[0:,new_w:]

        manual_count = {}
        w, h, channels = img.shape

        manual_count = self.count(h,w,img,manual_count)
        number_counter = Counter(manual_count).most_common(1)

        return(number_counter[0][0])

    @staticmethod
    def count(h,w,img,manual_count):
        for y in range(0, h):
            for x in range(0, w):
                RGB = (img[x, y, 2], img[x, y, 1], img[x, y, 0])
                if RGB in manual_count:
                    manual_count[RGB] += 1
                else:
                    manual_count[RGB] = 1
        return(manual_count)

class Doc:
    def __init__(self, path, name):
        self.path = path
        self.name = name

    def build_input(self):

        df = pd.DataFrame()
        df['Pages'] = [f for f in listdir(self.path) if isfile(join(self.path, f))]
        return(df)


doc1 = Doc('./data/','first')
backgrounddf = doc1.build_input()
backgrounddf['Colour'] = backgrounddf['Pages'].apply(lambda x: Page('./data/', x).extract_background_color() )

print(backgrounddf)
