import os
import tensorflow as tf
from glob import glob


class CustomDataset:

    def __init__(self, dir: str, out_shape = (2016, 3024), image_channels = 3, input_dir = 'Data', mask_dir = 'GroundTruth_1', batch_size= 10):
        self.out_shape = out_shape
        self.dir = dir
        self.image_channels = image_channels
        self.input_dir = input_dir
        self.mask_dir = mask_dir
        self.batch_size = batch_size


    def load_data(self,dir):
        images = glob(os.path.join(dir , self.input_dir+'\\*'))
        images = sorted([x for x in images if x.endswith('jpg')])
        masks = sorted(glob(os.path.join(dir , self.mask_dir+'\\*')))
        return images , masks


    def decode_img(self,img,channels):
        img = tf.image.decode_jpeg(img, channels=channels)
        img = tf.image.convert_image_dtype(img, tf.float32)
        return tf.image.resize(img, [self.out_shape[0] , self.out_shape[1]])


    def process_path(self,filepath, maskpath):
        img = tf.io.read_file(filepath)
        img = self.decode_img(img, self.image_channels)
        img = img / 255.0
        mask = tf.io.read_file(maskpath)
        mask = self.decode_img(mask, 1)
        mask = mask / 255.0
        return img, mask


    def tf_dataset(self,x,y):
        dataset = tf.data.Dataset.from_tensor_slices((x,y))
        dataset = dataset.shuffle(buffer_size=1000)
        dataset = dataset.map(self.process_path)
        dataset = dataset.batch(batch_size=self.batch_size)
        dataset = dataset.prefetch(2)
        return dataset

        
    def get_Dataset(self):
        images , masks = self.load_data(self.dir)
        return  self.tf_dataset(images , masks)
        
