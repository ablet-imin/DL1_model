
import numpy as np
from util import DL1_score, get_mean_score
import pandas as pd

import gc

def momentum_space(X, shift = -159402.65625, scale = 4.1431180575664244e-06):
    '''
    momentum_space(X)
    convert normalized momentum to normal momentum space.
    inputs:
        X - normalized pT. X can be numpy array with argbitrary
            batch size.
        shift - shift center of X.
        scale - scale values of X.
        
    return:
        Momentum in GeV unit.
    '''
    shift = shift
    scaling = scale
    Momentum_space = X/scaling - shift
    return Momentum_space/1000. #GeV unit

def efficiency_hist(test_data, model, Nbins=20, batch_size=None, wp_cut=1.21):
    '''
    efficiency_hist(test_data, model, Nbins=20)
    calculate tagged efficiancy.
    arg:
        test_data - collection of jet features as inputs.
        model - trained model, dropout enabled for test.
        Nbins - number of bins for binning.
        
    return: numpy array.
        Each element of the array is a number of jets in that bin
    '''
    _jetPt, _bins = np.histogram(momentum_space(test_data[:,1]), Nbins)
    _re = None
    if batch_size:
        _re = model.predict(test_data, batch_size=batch_size)
    else:
        _re = model(test_data, training=False).numpy()
    _score = DL1_score(_re[:,2],_re[:,1], _re[:,0])
    #btagging
    _Htagged_jetPt, _ = np.histogram( momentum_space(test_data[(_score>wp_cut)][:,1]), bins=_bins)
    del _score, _re
    gc.collect()
    return _Htagged_jetPt/_jetPt #efficiency
    
    
def efficiecy_mean_std(test_data, model, N_forward=30, Nbins=20, batch_size=None, wp_cut=1.21):
    '''
    mean and std of efficiencies. Efficiency histogram is calculated K times using
    a MC dropout enabled model.
    efficiecy_mean_std(test_data, model, N_forward)
    arg:
       test_data - collection of jet features as inputs.
       model - trained model, dropout enabled for test.
       N_forward - number evaluation with CM dropout.
       Nbins - number of bins for binning.
       
    return: mean, std
            mean - numpy array, content of each bin.
            std - numpy array, std of each bin.
    '''
    _hist_effs = []
    for i in range(0, N_forward):
        _hist_effs.append(efficiency_hist(test_data, model, Nbins=Nbins, batch_size=batch_size, wp_cut=wp_cut) )
        gc.collect()
    return np.mean(_hist_effs, axis=0).flatten(), np.std(_hist_effs, axis=0).flatten()

def efficiecy_from_mean(test_data, model, N_forward=30, Nbins=20, batch_size=None, wp_cut=1.21):
    '''
    mean and std of efficiencies. Efficiency histogram is calculated K times using
    a MC dropout enabled model.
    efficiecy_mean_std(test_data, model, N_forward)
    arg:
       test_data - collection of jet features as inputs.
       model - trained model, dropout enabled for test.
       N_forward - number evaluation with CM dropout.
       Nbins - number of bins for binning.
       
    return: mean, std
            mean - numpy array, content of each bin.
            std - numpy array, std of each bin.
    '''
    dtaset_size = test_data.shape[0]
    score_mean=[]
    jet_batch = dtaset_size//100000 + 1
    for i in range(jet_batch):
        score_mean.append(get_mean_score(test_data[100000*i:100000*(i+1)], model, N_foward=N_forward, batch_size=batch_size) )
        gc.collect()
    score_mean = np.concatenate(score_mean, axis=0)
    
    sc_filter = (score_mean[:,0]>wp_cut)
    
    data_pd = pd.DataFrame(data={'pT':momentum_space(test_data[sc_filter][:,1]),
                             'score':score_mean[sc_filter][:,0], 'std':score_mean[sc_filter][:,1]})
    
    #binning
    _Hpretag_jetPt,  _bins = np.histogram( momentum_space(test_data[:,1]), bins=Nbins)
    _Htagged_jetPt, _ = np.histogram( momentum_space(test_data[sc_filter][:,1]), bins=_bins)
    
    data_pd['bins'] = pd.cut(data_pd.pT, bins=_bins, labels=range(len(_bins)-1))
    
    def _variance(bin=0):
        filters = data_pd['bins']==bin
        stds = data_pd[filters]['std'].to_numpy()
        return np.sqrt(np.sum(stds**2))/_Hpretag_jetPt[bin]
        
    vars = np.array([_variance(bin=i) for i in range(len(_bins)-1)] )
        
    del score_mean, data_pd
    gc.collect()
    
    return _Htagged_jetPt/_Hpretag_jetPt, vars.flatten()
            
