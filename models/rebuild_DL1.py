import uproot
import json
from io import StringIO

import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
import tensorflow_addons as tfa
#from keras.layers.core import MaxoutDense

from models.maxout_layers import Maxout1D
from models.normalization_layer import Normalization


def get_net_struct(obj_path):
    '''
    Directly read ROOT objects specified in obj_path.
    obj_path -- file.root:Tdiectory/subdirectory/..../obj
    
    In this fuction we only need TString obj from the file.
    '''
    with uproot.open(obj_path) as net_config:
    #convert string to dictionary
        assert (type(net_config) is uproot.models.TObjString.Model_TObjString )
        _struct = json.load(StringIO(net_config))
    return _struct

        
def get_maxout_weights(NN_layer):
    maxout_unit=0
    maxout_h_unit=len(NN_layer['sublayers'][maxout_unit]['bias'])
    in_features = len(NN_layer['sublayers'][maxout_unit]['weights'])//maxout_h_unit
    maxout_weights=[]
    maxout_biases = []
    units = len(NN_layer['sublayers'])
    
    for maxout_unit in range(units):
        maxout_weights.append(
                                np.array(NN_layer['sublayers'][maxout_unit]['weights']
                              ).reshape( maxout_h_unit, in_features).transpose() )
        maxout_biases.append(
                                np.array(NN_layer['sublayers'][maxout_unit]['bias'])
                            )
    
    return (in_features, maxout_h_unit, units,
            np.stack(maxout_weights, axis=2).reshape(in_features,maxout_h_unit*units),
            np.stack(maxout_biases, axis=1).flatten() )
   

def get_dense_weights(NN_layer):
    h_unit=len(NN_layer["bias"])
    in_features = len(NN_layer['weights'])//h_unit
    weight = np.array(NN_layer['weights']).reshape( h_unit, in_features).transpose()
    return (in_features, h_unit, weight, np.array(NN_layer["bias"]) )

def get_BN_weights(NN_layer):
    h_unit=len(NN_layer["bias"])
    return (np.diag(np.array(NN_layer['weights'])),
            np.array(NN_layer["bias"]) )


def pars_layers(layers):
    N_layers = len(layers)
    layersDic = {}
    tf_layers = []
    N_features = -1
    for i, layer in enumerate(layers):
        arch = layer["architecture"]
        if arch == 'maxout':
            layer_name="maxout%s"%i
            
            # return Nfeatures, hiden nodes, maxout units, weights, bias
            v, h,unit, w, b = get_maxout_weights(layer)
            if N_features<1:  N_features = v
                
            layersDic[layer_name] = [w, b]
            tf_layers.append( Maxout1D(h, unit, name=layer_name) )
            tf_layers.append( keras.layers.Activation(
                                                        activation=layer["activation"],
                                                        name="activ%s"%i
                                                        )
                            )
            
        elif arch == 'normalization':
            layer_name="BN%s"%i
            layersDic[layer_name] = [*get_BN_weights(layer)]
            tf_layers.append( Normalization(name=layer_name) )
            
        elif arch == 'dense':
            layer_name="dense%s"%i
            #Ninputs, hiden nodes, weights, bias
            v, h, w, b = get_dense_weights(layer)
            if N_features<1: N_features = v
            layersDic[layer_name]=[w, b ]
            activation="relu" if layer["activation"]=='rectified' else layer["activation"]
            
            tf_layers.append( keras.layers.Dense(h, activation=activation,
                              kernel_initializer='glorot_uniform', name=layer_name)
                            )
        else:
            raise Exception('Unkown layer %s'%arch )
            
    return N_features, tf_layers, layersDic

#create NN from input layers
#each layer has unique name
def get_DL1(N_features, dl1_layers, lr=0.005, drops=None):
    
    In = tf.keras.layers.Input(shape=(N_features,), name="input")
    x = In
    drop_index=0
    for layer in dl1_layers[:-1]:
        if drops:
            if 'BN' in layer.name:
                x = keras.layers.Dropout( drops[drop_index],
                                          name="drop%s"%drop_index )(x, training=True)
                drop_index=drop_index+1
        x = layer(x)
        
    predictions = dl1_layers[-1](x)
    
    model = keras.models.Model(inputs=In, outputs=predictions)
    model_optimizer = keras.optimizers.Adam(lr=lr)
    model.compile(
        loss=tf.keras.losses.CategoricalCrossentropy(from_logits=False),
        optimizer=model_optimizer,
        metrics=['accuracy']
    )
    return model

def set_dl1_weights(model, weights):
    for name in weights.keys():
        layer = model.get_layer( name=name)
        layer.set_weights(weights[name])
