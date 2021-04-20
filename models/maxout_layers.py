import tensorflow as tf
import tensorflow.keras as keras
import tensorflow_addons as tfa
#from keras.layers.core import MaxoutDense

class MaxoutDense1D(keras.layers.Layer):
    
    def __init__(self, output_units, units=2):
        super(MaxoutDense1D, self).__init__()
        self.units=units
        self.output_units = output_units
    def build(self, input_shape):
        self.sub_layers = self.units*[keras.layers.Dense(self.output_units)]
        for i in range(self.units):
            self.sub_layers[i].build(input_shape)
    
    def call(self, inputs):
        Z = [self.sub_layers[i](inputs) for i in range(self.units)]
        return tf.math.reduce_max(Z, 0 )
        

class Maxout1D(keras.layers.Layer):
    
    def __init__(self, output_units, units=2):
        super(Maxout1D, self).__init__()
        self.units=units
        self.output_units = output_units
        
    def build(self, input_shape):
        self.w = self.add_weight(shape=(input_shape[-1], self.output_units*self.units),
                                 initializer='random_normal',
                                 trainable=True, name="maxout_w")
        self.b = self.add_weight(shape=(self.output_units*self.units,),
                                initializer = 'random_normal',
                                trainable=True, name="maxout_b")
    
    def call(self, inputs):
        Z = tf.matmul(inputs, self.w)+ self.b
        m = tf.nn.max_pool1d(tf.reshape(Z, [-1,self.output_units*self.units, 1]),
                             ksize = self.units,
                            strides=self.units,
                            padding='SAME')
        return  tf.reshape(m, [-1,self.output_units])
    
    def get_config(self):
        #config = super(Maxout1D, self).get_config()
        #config.update({"units": self.units,
         #       "output_units": self.output_units})
        return {"units": self.units,
              "output_units": self.output_units}


