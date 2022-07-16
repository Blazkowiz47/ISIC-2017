import logging
import os
import random as rd
import cv2
from tqdm import tqdm
from glob import glob
import numpy as np

def encrypt(input_dir:str, output_dir:str, isMask:bool = False):
    key = rd.randint(0,255)
    if not os.path.exists(input_dir):
        logging.debug(
                    "Input Directory doesnot exist")
        raise
    if not os.path.exists(output_dir):
            try:
                os.makedirs(output_dir)
            except:
                logging.debug(
                    "Output Directory doesnot exist and cannot be created")
                raise
    images = glob(os.path.join(input_dir+'\\*'))
    images = sorted([x for x in images if x.endswith('jpg')])
    for img in tqdm(images):
        image = cv2.imread(img)
        for index, values in enumerate(image):
            image[index] = values ^ key
        cv2.imwrite(output_dir+'\\' +img.split('\\')[-1],image)
 
def __encrypt(image, key, isMask:bool):
    
    pass


def generate_secret_key(height:int= 192, width:int=256):
    imarray = np.random.randint(256, size=(height, width,3),dtype=np.uint8)
    cv2.imwrite('secret_key_image.png',imarray)
    mask_array = np.zeros((height,width),dtype=np.uint8)
    for h in range(height):
        for w in range(width):
            mask_array[h][w] = np.bitwise_xor(np.bitwise_xor(imarray[h][w][0] , imarray[h][w][1]) , imarray[h][w][2]) 
    # im = Image.fromarray(imarray.astype('uint8')).convert()
    cv2.imwrite('secret_key_client_mask.png', mask_array)
