import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Model
import utils.Evaluation as ev

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

class JaccardIndex(tf.keras.metrics.Metric):
    def __init__(self, name='Jaccard Index',threshold=0.5,**kwargs) -> None:
        super(JaccardIndex, self).__init__(name=name,**kwargs)
        self.ji = self.add_weight(name='JI', initializer='zeros',dtype='float64')
        self.count = self.add_weight(name='count', initializer='zeros',dtype='float64')
        self.threshold = threshold
        

    def update_state(self,y_true, y_pred, sample_weight=None):
        conf_mat = ev.get_confusion_matrix(y_true,y_pred,self.threshold)
        self.ji.assign(self.count*self.ji)
        self.ji.assign_add(ev.get_jaccard_index(conf_mat))
        self.count.assign_add(tf.constant(1.,dtype=tf.float64))
        self.ji.assign(self.ji/self.count)

    def result(self):        
        return self.ji


class DiceIndex(tf.keras.metrics.Metric):
    def __init__(self, name='Dice Index',threshold=0.5,**kwargs) -> None:
        super(DiceIndex, self).__init__(name=name, **kwargs)
        self.ji = self.add_weight(name='DI', initializer='zeros',dtype='float64')
        self.count = self.add_weight(name='count', initializer='zeros',dtype='float64')
        self.threshold = threshold
    
    def update_state(self,y_true, y_pred, sample_weight=None):
        conf_mat = ev.get_confusion_matrix(y_true,y_pred,self.threshold)
        self.ji.assign(self.count*self.ji)
        self.ji.assign_add(ev.get_dice_coefficient(conf_mat))
        self.count.assign_add(tf.constant(1.,dtype=tf.float64))
        self.ji.assign(self.ji/self.count)
        

    def result(self):
        return self.ji 


class Sensitvity(tf.keras.metrics.Metric):
    def __init__(self, name='Sensitivity',threshold=0.5,**kwargs) -> None:
        super(Sensitvity, self).__init__(name=name, **kwargs)
        self.ji = self.add_weight(name='SN', initializer='zeros',dtype='float64')
        self.count = self.add_weight(name='count', initializer='zeros',dtype='float64')
        self.threshold = threshold

    def update_state(self,y_true, y_pred, sample_weight=None):
        conf_mat = ev.get_confusion_matrix(y_true,y_pred,self.threshold)
        self.ji.assign(self.count*self.ji)
        self.ji.assign_add(ev.get_sensitivity(conf_mat))
        self.count.assign_add(tf.constant(1.,dtype=tf.float64))
        self.ji.assign(self.ji/self.count)
        

    def result(self):
        return self.ji 


class Specificity(tf.keras.metrics.Metric):
    def __init__(self, name='Specificity',threshold=0.5,**kwargs) -> None:
        super(Specificity, self).__init__(name=name, **kwargs)
        self.ji = self.add_weight(name='SP', initializer='zeros',dtype='float64')
        self.count = self.add_weight(name='count', initializer='zeros',dtype='float64')
        self.threshold = threshold
        
    def update_state(self,y_true, y_pred, sample_weight=None):
        conf_mat = ev.get_confusion_matrix(y_true,y_pred,self.threshold)
        self.ji.assign(self.count*self.ji)
        self.ji.assign_add(ev.get_specificity(conf_mat))
        self.count.assign_add(tf.constant(1.,dtype=tf.float64))
        self.ji.assign(self.ji/self.count)
        
    def result(self):
        return self.ji


class Accuracy(tf.keras.metrics.Metric):
    def __init__(self, name='Accuracy',threshold=0.5,**kwargs) -> None:
        super(Accuracy, self).__init__(name=name, **kwargs)
        self.ji = self.add_weight(name='AC', initializer='zeros',dtype='float64')
        self.count = self.add_weight(name='count', initializer='zeros',dtype='float64')
        self.threshold = threshold

    def update_state(self,y_true, y_pred, sample_weight=None):
        conf_mat = ev.get_confusion_matrix(y_true,y_pred,self.threshold)
        self.ji.assign(self.count*self.ji)
        self.ji.assign_add(ev.get_accuracy(conf_mat))
        self.count.assign_add(tf.constant(1.,dtype=tf.float64))
        self.ji.assign(self.ji/self.count)

    def result(self):
        return self.ji 
