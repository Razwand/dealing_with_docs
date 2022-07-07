#!/usr/bin/env python
# coding: utf-8

# In[1]:


from PyPDF2 import PdfFileWriter, PdfFileReader, PdfFileMerger
from pdf2image import convert_from_path
import sys


# In[2]:


poppler_path = 'C:/Users\/asr_l/miniconda3/pkgs/poppler-22.04.0-h24fffdf_1/Library/bin'


# In[3]:


def pdf_to_image(path,poppler_path,execution):
    
    pages = convert_from_path(path, 200,poppler_path=poppler_path)
    
    name = path.split('/')[-1]
    
    for i in range(len(pages)):
        path_page_img = './pdf_pages'+ execution +'/' + name + '_page_'+str(i)+'.jpg'
        print('Saving page image: ', path_page_img)
        pages[i].save(path_page_img, 'JPEG')           

if __name__ == "__main__":
    
    pdf_to_image(sys.argv[1], poppler_path, sys.argv[2])

