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
    
    def __init__(self, output_units, units=2, name=None):
        super(Maxout1D, self).__init__(name=name)
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
        Z = tf.reshape((tf.matmul(inputs, self.w)+ self.b), [-1, self.units, self.output_units])
        
        return tf.reshape(tfa.layers.Maxout(1,-2)(Z), [-1,Z.shape[-1]])
    
    def get_config(self):
        #config = super(Maxout1D, self).get_config()
        #return  config.update({"units": self.units,
        #            "output_units": self.output_units})
        return {"units": self.units,
              "output_units": self.output_units}


