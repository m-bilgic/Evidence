'''
Created on Nov 24, 2014

@author: mbilgic
'''

import numpy as np
from scipy.sparse import diags

def pos_neg_evi_nn_sparse_data(X, coef):
    """Return positive and negative evidence for sparse matrices with non-negative values"""
    
    num_feat = X.shape[1]    
    
    coef_gt0 = coef > 0
    coef_lt0 = coef < 0
    
    
    pos_coef = np.zeros(num_feat)
    neg_coef = np.zeros(num_feat)
    
    pos_coef[coef_gt0] = coef[coef_gt0]
    neg_coef[coef_lt0] = coef[coef_lt0]
    
    
    pos_coef_diags = diags(pos_coef,0)
    neg_coef_diags = diags(neg_coef,0)
    
    
    pm = X*pos_coef_diags
    nm = X*neg_coef_diags
    
    pos_evi = pm.sum(axis=1).A1
    neg_evi = nm.sum(axis=1).A1
    
    return pos_evi, neg_evi

def pos_neg_evi(X, coef):
    """Return positive and negative evidence for non-sparse matrices"""
    
    num_inst, num_feat = X.shape
    
    coef_diags = diags(coef,0)
    
    dm = X*coef_diags
    
    pos_evi = np.zeros(num_inst)
    neg_evi = np.zeros(num_inst)
    
    for i in range(num_inst):
        for j in range(num_feat):
            evi = dm[i,j]
            if evi > 0:
                pos_evi[i] += evi
            else:
                neg_evi[i] += evi
    
    return pos_evi, neg_evi

