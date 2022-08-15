import os
import tensorflow as tf
from glob import glob


class CustomDataset:

    def __init__(self, dir: str, out_shape = (256,256), image_channels = 3, mask_channels = 1, input_dir = 'DataE', mask_dir = 'GroundTruth_1E', batch_size= 10):
        self.out_shape = out_shape
        self.dir = dir
        self.image_channels = image_channels
        self.input_dir = input_dir
        self.mask_dir = mask_dir
        self.batch_size = batch_size
        self.mask_channels = mask_channels

    # loads data
    def load_data(self,dir):
        images = glob(os.path.join(dir , self.input_dir+'\\*'))
        images = sorted([x for x in images if x.endswith('jpg')])
        masks = sorted(glob(os.path.join(dir , self.mask_dir+'\\*')))
        return images , masks

    # Decodes the image and resizes it into 
    def decode_img(self,img,channels):
        img = tf.image.decode_jpeg(img, channels=channels)
        img = tf.image.convert_image_dtype(img, tf.float32)
        img = tf.image.resize(img, [self.out_shape[0] , self.out_shape[1]])
        return img
    # Normalises image pixels from 0-1
    def process_path(self,filepath, maskpath):
        img = tf.io.read_file(filepath)
        img = self.decode_img(img, self.image_channels)
        img = tf.reshape( img, (1,*self.out_shape,self.image_channels))
        img = img / 255.0
        mask = tf.io.read_file(maskpath)
        mask = self.decode_img(mask, self.mask_channels)
        mask = tf.reshape(mask , (1,*self.out_shape,self.mask_channels))
        mask = mask / 255.0
        return img, mask


    def tf_dataset(self,x,y):
        dataset = tf.data.Dataset.from_tensor_slices((x,y))
        dataset = dataset.shuffle(buffer_size=1000)
        dataset = dataset.map(self.process_path)
        # dataset = dataset.batch(batch_size=self.batch_size)
        # dataset = dataset.prefetch(2)
        return dataset

    # Call this to get the data
    def get_Dataset(self):
        images , masks = self.load_data(self.dir)
        return  self.tf_dataset(images , masks)
        
