#!/usr/bin/env python
# coding: utf-8


def create_folder(name):

    '''
    Creates a folder with name <name> if it doesn't exist
    '''
    
    if os.path.isdir(name) == False:
        os.mkdir(name)

def pdf_to_image(input_folder_name,poppler_path, output_folder_path):

    '''
    This fuction takes a pdf and, given de input folder name,
    generate images for each one of the pdf pages.
    
    pdf_pages_<input_folder_name>
    
    The name of the images created will have the following structure:

    <name_of_the_volume>.pdf_page_<page_number>.jpg
    '''

    volume = os.listdir('./input_volume/'+input_folder_name+'/')[0]
    out_path_pdf = output_folder_path +'pdf_pages_'+input_folder_name+'/'
    create_folder(out_path_pdf)

    pages = convert_from_path('./input_volume/' + input_folder_name +'/'+volume, 200,poppler_path=poppler_path)
    
    for i in range(len(pages)):
        path_page_img = output_folder_path +'pdf_pages_'+ input_folder_name +'/' + volume + '_page_'+str(i)+'.jpg'
        print('⏳ Saving page image: ', path_page_img)
        pages[i].save(path_page_img, 'JPEG')     




