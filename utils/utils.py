import logging
import os
import json
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
from glob import glob


def loop_over_dataset(input_dir:str, output_dir:str, action, metadata_path: str = None,format:str = 'jpg'):
    if not os.path.exists(input_dir):
        logging.debug("Input Directory doesnot exist")
        raise
    if not os.path.exists(output_dir):
            try:
                os.makedirs(output_dir)
            except:
                logging.debug(
                    "Output Directory doesnot exist and cannot be created")
                raise
    
    if not metadata_path:
        image_names = get_images_list(input_dir,format)
    else:
        image_names = pd.read_csv(metadata_path)
        image_names = image_names['image_id'].to_list()
    for img in tqdm(image_names):

        if format == 'png':
            image = cv2.imread(f'{input_dir}\\{img}.{format}',  cv2.IMREAD_GRAYSCALE)
        else:
            image = cv2.imread(f'{input_dir}\\{img}.{format}')

        image = np.asarray(image)
        image = action(image)

        cv2.imwrite(f'{output_dir}\\{img}.{format}',image)


def get_images_list(input_dir:str,format:str = 'jpg'):
    images = glob(os.path.join(input_dir+'\\*'))
    images = sorted([x for x in images if x.endswith(format)])
    return [image.split('\\')[-1].split('.')[0] for image in images]

