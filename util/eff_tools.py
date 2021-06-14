
import numpy as np
from util import DL1_score

def momentum_space(X):
    '''
    momentum_space(X)
    convert normalized momentum to normal momentum space.
    inputs:
        X - normalized pT. X can be numpy array with argbitrary
            batch size.
        
    return:
        Momentum in GeV unit.
    '''
    shift = -159402.65625
    scaling = 4.1431180575664244e-06
    Momentum_space = X/scaling - shift
    return Momentum_space/1000. #GeV unit

def efficiency_hist(test_data, model, Nbins=20, batch_size=None):
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
        _re = model.predict(X_data, batch_size=batch_size)
    else:
        _re = model(X_data, training=False).numpy()
    _score = DL1_score(_re[:,2],_re[:,1], _re[:,0])
    #btagging
    _Htagged_jetPt, _ = np.histogram( momentum_space(test_data[(_score>1.45)][:,1]), bins=_bins)
    
    return _Htagged_jetPt/_jetPt #efficiency

def efficiecy_mean_std(test_data, model, N_forward=30, Nbins=20, batch_size=None):
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
        _hist_effs.append(efficiency_hist(test_data, model, Nbins=Nbins, batch_size=batch_size) )
    
    return np.mean(_hist_effs, axis=0).flatten(), np.std(_hist_effs, axis=0).flatten()
            
