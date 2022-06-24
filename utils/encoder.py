import tensorflow as tf
from tensorflow.keras import layers


class CustomEncoder(layers.Layer):
    def __init__(self,input_shape,batch_size = 10):
        super().__init__(input_shape)
        self.batch_size = batch_size
        # self.input_shape = input_shape
        # self.input_layer = layers.InputLayer(input_shape=input_shape, batch_size=self.batch_size)
        self.c1 = layers.Conv2D(8, (5,5),name='c_1', activation='relu', padding='same', input_shape = input_shape)
        self.c2 = layers.Conv2D(16, (3,3),name='c_2', activation='relu', padding='same')
        self.p1 = layers.MaxPool2D(name='p_1')
        self.c3 = layers.Conv2D(32, (4,4),name='c_3', activation='relu', padding='same')
        self.p2 = layers.MaxPool2D(name='p_2')
        self.c4 = layers.Conv2D(64, (4,4),name='c_4', activation='relu', padding='same')
        self.p3 = layers.MaxPool2D(name='p_3')
        self.c5 = layers.Conv2D(64, (5,5),name='c_5', activation='relu', padding='same')
    
    def call(self, input):
        # x = self.input_layer(input)
        x = self.c1(input)
        x = self.c2(x)
        x = self.p1(x)
        x = self.c3(x)
        x = self.p2(x)
        x = self.c4(x)
        x = self.p3(x)
        return self.c5(x)
