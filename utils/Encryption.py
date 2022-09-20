import logging
import os
import json
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
from glob import glob
import utils as ut


def encrypt(image, all_pob_numbers_list: list[int] = None):
    
    result = np.zeros(image.shape)
    image = image.clip(0,251)
    for idx , x in np.ndenumerate(image):
        result[idx] = all_pob_numbers_list[x]
    
    # Normalising images
    mi, mx = result.min(), result.max()
    result = (result - mi) * 255 / (mx-mi) 
    return result.astype('uint8') 


def decrypt(image, all_pob_numbers_list: list[int] = None):
    result = np.zeros(image.shape, dtype=np.int32)
    for idx , x in np.ndenumerate(image):
        try:
            result[idx] = all_pob_numbers_list.index(x)
        except:
            result[idx] = 255
    return result 


def scramble(image,key):
    assert(image.shape[0]*image.shape[1] == len(key['rp']))
    assert(image.shape[0]*image.shape[1] == len(key['gp']))
    assert(image.shape[0]*image.shape[1] == len(key['bp']))
    if len(image.shape) == 3 and image.shape[2] == 3:
        r, g , b = image[:,:,0],image[:,:,1],image[:,:,2]
        r , g , b = r.flatten() , g.flatten() , b.flatten()
        pr,pg,pb = [],[],[]
        for  vr,vg,vb in zip(key['rp'],key['gp'],key['bp']):
            pr.append(r[vr-1])
            pg.append(g[vg-1])
            pb.append(b[vb-1])
        pr,pg,pb = np.reshape(pr,(image.shape[0],image.shape[1])),np.reshape(pg,(image.shape[0],image.shape[1])),np.reshape(pb,(image.shape[0],image.shape[1]))
        return np.dstack((pr,pg,pb))
        
    else:
        mask = image[:,:]
        mask = mask.flatten()
        pmask = []
        for  v in key['mp']:
            pmask.append(mask[v-1])
        return np.reshape(pmask,(image.shape[0],image.shape[1]))

# changed every key to rp for now
def unscramble(image,key):
    assert(image.shape[0]*image.shape[1] == len(key['rp']))
    if len(image.shape) == 3 and image.shape[2] == 3:
        r , g , b = image[:,:,0],image[:,:,1],image[:,:,2]
        r , g , b = r.flatten() , g.flatten() , b.flatten()
        pr,pg,pb = [0]*len(r) ,[0]*len(g),[0]*len(b)
        for i, (vr,vg,vb) in enumerate(zip(key['rp'],key['gp'],key['bp'])):
            pr[vr-1] , pg[vg-1] , pb[vb-1] = r[i] , g[i] , b[i]
        pr,pg,pb = np.reshape(pr,(image.shape[0],image.shape[1])),np.reshape(pg,(image.shape[0],image.shape[1])),np.reshape(pb,(image.shape[0],image.shape[1]))
        return np.dstack((pr,pg,pb))
        
    else:
        mask = image[:,:]
        mask = mask.flatten()
        pmask = [0]*len(mask)
        for i, v in enumerate(key['mp']):
            pmask[v-1] = mask[i]
        return np.reshape(pmask,(image.shape[0],image.shape[1]))


def generate_pob_values(n:int = 10,r:int = 5) -> list:
    ''' 
        Consider n = 10
        and r = 5
    '''
    pob_values = []
    B = ''
    done = False

    for i in range(n):
        B += '1' if i <= r-1 else '0'
    
    B = list(B)

    while not done:
        pob_values.append(int(''.join(B),2))
        no_of_zeros , i , j = 0, 0 , 1
        while B[j] == '1' or B[i] == '0':
            if B[i] == '0':
                no_of_zeros += 1
            if j == n-1:
                done = True
                break
            i = j
            j += 1
        B[j] = '1'
        j = i - no_of_zeros
        while i >= j:
            B[i] = '0'
            i -= 1
        while i >= 0:
            B[i] = '1'
            i -= 1
    pob_values = sorted(pob_values)
    return pob_values


def calC(n:int , r:int) -> int:
    r = r if n-r < r else n-r
    t = 1
    for tr in range(r+1,n+1):
        t *= tr
    for tr in range(1,n-r+1):
        t = t // tr
    return t


def generate_permutation(length: int, R:int = 3.999) ->list:
    key = np.arange(length) + 1
    key = np.random.choice(key,length,False)
    return key.tolist()    


def generate_and_save_key(length:int, file_path: str, singular: bool = True):
    # rd, gd, bd are for rgb color channels in image
    rp = generate_permutation(length)
    if not singular:
        gp = generate_permutation(length)
        bp = generate_permutation(length)
        # md is for the segmentation mask 
        mp = generate_permutation(length)
    else:
        gp, bp, mp = rp, rp, rp

    with open(file_path, 'w+') as fp:
        json.dump({'rp': rp, 'gp': gp, 'bp': bp, 'mp': mp},fp,indent=4 )


def preprocess(image, out_shape = (256,256)):
    return cv2.resize(image, out_shape, interpolation = cv2.INTER_AREA)


def encrypt_dir(input_dir:str, output_dir:str, metadata_path: str = None,format:str = 'jpg') -> None:
    pob_values =  generate_pob_values()
    ut.loop_over_dataset(input_dir,output_dir, lambda image : encrypt(image,pob_values), metadata_path,format)    


def preprocess_dir(input_dir:str, output_dir:str,format:str = 'jpg',out_shape:tuple=(256,256), metadata_path:str = None) -> None:
    ut.loop_over_dataset(input_dir,output_dir, lambda image : preprocess(image,out_shape),metadata_path,format)


def get_pob_value(value: int,n: int = 10, r:int = 5 ):
    j , temp = n , value
    B = '0' * n
    B = list(B)
    for k in reversed(range(1,r+1)):
        while True:
            j -= 1
            p = calC(j,k)
            if temp >= p:
                temp -= p
                assert(j < n)
                assert(j >= 0)
                B[j] = '1'
                break
            if j < 0:
                break
    B = reversed(B)
    return int(''.join(B),2)


def add_noise(image, max_value:int = 5):
    image = image + np.random.choice( range(-max_value-1,max_value+1,1) , size=image.shape)
    image = (image - image.min()) / (image.max() - image.min())
    image = image * 255
    image = image.astype(np.uint8) 
    return image 