import tensorflow as tf
import tensorflow.keras as keras
import tensorflow_addons as tfa
#from keras.layers.core import MaxoutDense

        

class Normalization(keras.layers.Layer):
    
    def __init__(self, name=None):
        super(Normalization, self).__init__(name=name)
        
        
    def build(self, input_shape):
        self.w = self.add_weight(shape=(input_shape[-1],input_shape[-1]),
                            initializer=tf.zeros,
                            trainable=True, name="norm_w")
        self.b = self.add_weight(shape=(input_shape[-1],),
                            initializer=tf.zeros,
                            trainable=True, name="norm_b")
    
    def call(self, inputs):
        out = inputs+self.b
        return  tf.matmul(inputs, self.w)
        
    def get_config(self):
        return super(Normalization, self).get_config()
        
    def compute_output_shape(self, input_shape):
        return input_shape
        
   
