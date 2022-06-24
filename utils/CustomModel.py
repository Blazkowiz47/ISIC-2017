import tensorflow as tf
from tensorflow.keras import layers, losses , Sequential
from tensorflow.keras.models import Model

class CustomModel(Model):

    def __init__(self, input_shape, batch_size= 10):
        super(CustomModel, self).__init__()
        self.batch_size = batch_size
        self.input_layer = layers.InputLayer(input_shape=input_shape, batch_size=self.batch_size)
        self.c1 = layers.Conv2D(8, (5,5), activation='relu', padding='same')
        self.c2 = layers.Conv2D(16, (3,3), activation='relu', padding='same')
        self.p1 = layers.MaxPool2D()
        self.c3 = layers.Conv2D(32, (4,4), activation='relu', padding='same')
        self.p2 = layers.MaxPool2D()
        self.c4 = layers.Conv2D(64, (4,4), activation='relu', padding='same')
        self.p3 = layers.MaxPool2D()
        self.c5 = layers.Conv2D(64, (5,5), activation='relu', padding='same')
        self.d1 = layers.Conv2DTranspose(64, kernel_size=(5,5),activation='relu', padding='same')
        self.u1 = layers.UpSampling2D()
        self.d2 = layers.Conv2DTranspose(32, kernel_size=(4,4),  activation='relu', padding='same')
        self.u2 = layers.UpSampling2D()
        self.d3 = layers.Conv2DTranspose(16, kernel_size=(4,4), activation='relu', padding='same')
        self.u3 = layers.UpSampling2D()
        self.d4 = layers.Conv2DTranspose(8, kernel_size=(3,3), activation='relu', padding='same')
        self.output_layer = layers.Conv2D(1, kernel_size=(5,5), activation='sigmoid', padding='same')
        
    def call(self,x):
        x = self.input_layer(x)
        x = self.c1(x)
        x = self.c2(x)
        x = self.p1(x)
        x = self.c3(x)
        x = self.p2(x)
        x = self.c4(x)
        x = self.p3(x)
        x = self.c5(x)
        x = self.d1(x)
        x = self.u1(x)
        x = self.d2(x)
        x = self.u2(x)
        x = self.d3(x)
        x = self.u3(x)
        x = self.d4(x)
        x = self.output_layer(x) 
        return x