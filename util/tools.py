import matplotlib
import matplotlib.pyplot as plt

import numpy as np


def DL1_score(pb, pc, pl):
    '''
    DL1 btagging scor:
        log(pb/(0.08*pc + 0.92*pl) )
    arg:
        pb - softmax probability of b-jet class
        pc - softmax probability of c-jet class
        pl - softmax probability of light-jet class
        
    return DL1 score for a batch:
        a numpy array, array size is equal to batch size.
    '''
    return np.log(pb/(0.08*pc + 0.92*pl))
    
    
def plot_prob_score(pb, pc, pl):
    '''
    plot DL1 score in to bins, 100 bins.
    arg:
        pb - softmax probability of b-jet class
        pc - softmax probability of c-jet class
        pl - softmax probability of light-jet class
        
    return: no return value.
    '''
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    _n, bins, _ = ax1.hist(pb, bins=50, range=(0,1), alpha=0.5, label="p_b")
    _ = ax1.hist(pc, bins=bins, alpha=0.5, label="p_c")
    _ = ax1.hist(pl, bins=bins, alpha=0.5, label="p_l")
    ax1.legend(loc="upper right")
    
    _ =ax2.hist(DL1_score(pb, pc, pl), 100, alpha=0.6)


def plot_prob_score_from_model(event,label, model):
    '''
    Plot DL1 socre for a single jet. Single input jet is
    Evaluated 10K time with drouput enabled in the test.
    arg:
        event - Single input (for one jet).
        label - true lable. just for printout.
        model - trained model. Dropout must be enabled in
                the test.
    '''
    
    Single_Pred_prob = model(np.array(10000*[event]))
    pb = Single_Pred_prob[:,2].numpy()
    pc = Single_Pred_prob[:,1].numpy()
    pl = Single_Pred_prob[:,0].numpy()
    print("true label: ", label)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    _n, bins, _ = ax1.hist(pb, bins=50, range=(0,1),  alpha=0.5, label="p_b")
    _ = ax1.hist(pc, bins=bins,  alpha=0.5, label="p_c")
    _ = ax1.hist(pl, bins=bins,  alpha=0.5, label="p_l")
    ax1.legend(loc="upper right")
    
    _ =ax2.hist(DL1_socre(pb, pc, pl), 100,  alpha=0.6)
    

def get_mean_score(X_test, model, N_foward=100, batch_size=None, median=False):
    '''
    Calculate mean score for a jet using MC dropout. Mean score is
    a mean of 10K DL1 score ontained by 10K evaluation on a sinle input.
    arg:
        X_test - two dementional array, first demention is the batch size.
                 Last demention is features.
        model - Already trained model, with dropout enabled in tests.
        N_foward - number of evaluation for a single input.
        
    return: A numpy array of [mean, std].
            For each jet mean and stadard deviation (std) of 10K DL1 score are calculated.
            Array lenth eqaul to the batch size.
    '''
    
    def _scores(batch_size=batch_size):
        _predict_prob =None
        if batch_size:
            _predict_prob = model.predict(X_test, batch_size=batch_size)
        else:
            _predict_prob = model(X_test, training=False).numpy()
        return DL1_score(_predict_prob[:,2], _predict_prob[:,1], _predict_prob[:,0])
        
    _results = np.array([_scores() for i in range(N_foward)])
            
    #def _mean_score(inputs):
    #    result_prob = model(np.array(N_foward*[inputs]), training=False).numpy()
    #    result = DL1_score(result_prob[:,2], result_prob[:,1], result_prob[:,0])
    #    return result.mean(), result.std()
    np_stat = np.mean
    if median:
        np_stat = np.median

    return np.stack([np_stat(_results, axis=0),np.std(_results, axis=0)],axis=1)
    




def get_mode_from_root(File_path="BTagCalibRUN2-08-40.root:DL1/AntiKt4EMTopo/net_configuration"):
    #filename.root:Tdirectory/directory/obj
    #File_path="BTagCalibRUN2-08-40.root:DL1/AntiKt4EMTopo/net_configuration"
    import models.rebuild_DL1 as DL1

    DL1_struct = DL1.get_net_struct(File_path)
    DL1_weights = DL1_struct['layers']

    #DL1_layers = [ 72, 57, 60, 48, 36,24, 12, 6]
    DL1_dropouts = [0.1, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2]

    #findout input size, and weight matrix for each layer.
    #create corresponding tensorflow layers and store them in a list.
    features, dl1_layers, dl1_weights = DL1.pars_layers(DL1_struct['layers'])

    _model = DL1.get_DL1(features , dl1_layers, drops=None )
    _model_dropout = DL1.get_DL1(features , dl1_layers, drops=DL1_dropouts )
    #model.summary()
    DL1.set_dl1_weights(model=_model, weights=dl1_weights)
    DL1.set_dl1_weights(model=_model_dropout, weights=dl1_weights)
    return _model, _model_dropout

def load_trained(model_weight):
    import models.binbin_model as binbin_model
    test_model = binbin_model.DL1_model(InputShape=44, training=False)
    test_model_Dropout = binbin_model.DL1_model(InputShape=44, training=True)
    test_model.load_weights(model_weight)
    test_model_Dropout.load_weights(model_weight)
    return test_model, test_model_Dropout

def data_category(data, label,category='b'):
    #1 is b, 2 is c, 4 is light
    label_binary = {'b':2, 'c':1, 'l':0}
    label_index = label_binary[category]
    label_filter = (label[:,label_index]==1)
    print(label_filter)
    return data[label_filter], label[label_filter]
    

#get test dataset
def get_dataset(path):
    hf = h5py.File(path, 'r')
    X_all, Y_all = hf['X_test'][:], hf['Y_test'][:]
    hf.close()
    X, Y = data_category(X_all, Y_all,category='b')
    del X_all, Y_all
    return X, Y
