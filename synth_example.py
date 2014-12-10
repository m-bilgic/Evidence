'''
Created on Nov 17, 2014

@author: mbilgic
'''

from scipy.stats import beta
import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':
    f1_t_c_t = beta.rvs(20, 30, size=1e6)
    f1_f_c_t = 1 - f1_t_c_t
    
    f1_t_c_f = beta.rvs(5, 45, size=1e6)
    f1_f_c_f = 1 - f1_t_c_f
    
    f2_t_c_t = beta.rvs(5, 45, size=1e6)
    f2_f_c_t = 1 - f2_t_c_t
    
    f2_t_c_f = beta.rvs(20, 30, size=1e6)
    f2_f_c_f = 1 - f2_t_c_f
    
    c_t = beta.rvs(50, 50, size=1e6)
    c_f = 1 - c_t
    
    #T, T
    j_f1t_f2t_ct = f1_t_c_t*f2_t_c_t*c_t
    j_f1t_f2t_cf = f1_t_c_f*f2_t_c_f*c_f    
    c_ct_f1t_f2t = j_f1t_f2t_ct / (j_f1t_f2t_ct + j_f1t_f2t_cf)
    c_cf_f1t_f2t = j_f1t_f2t_cf / (j_f1t_f2t_ct + j_f1t_f2t_cf)
    
    #T, F
    j_f1t_f2f_ct = f1_t_c_t*f2_f_c_t*c_t
    j_f1t_f2f_cf = f1_t_c_f*f2_f_c_f*c_f    
    c_ct_f1t_f2f = j_f1t_f2f_ct / (j_f1t_f2f_ct + j_f1t_f2f_cf)
    c_cf_f1t_f2f = j_f1t_f2f_cf / (j_f1t_f2f_ct + j_f1t_f2f_cf)
    
    #F, T
    j_f1f_f2t_ct = f1_f_c_t*f2_t_c_t*c_t
    j_f1f_f2t_cf = f1_f_c_f*f2_t_c_f*c_f    
    c_ct_f1f_f2t = j_f1f_f2t_ct / (j_f1f_f2t_ct + j_f1f_f2t_cf)
    c_cf_f1f_f2t = j_f1f_f2t_cf / (j_f1f_f2t_ct + j_f1f_f2t_cf)
    
    
    #F, F
    j_f1f_f2f_ct = f1_f_c_t*f2_f_c_t*c_t
    j_f1f_f2f_cf = f1_f_c_f*f2_f_c_f*c_f    
    c_ct_f1f_f2f = j_f1f_f2f_ct / (j_f1f_f2f_ct + j_f1f_f2f_cf)
    c_cf_f1f_f2f = j_f1f_f2f_cf / (j_f1f_f2f_ct + j_f1f_f2f_cf)
    
    print np.mean(c_ct_f1t_f2t), np.var(c_ct_f1t_f2t)
    print np.mean(c_ct_f1t_f2f), np.var(c_ct_f1t_f2f)
    print np.mean(c_ct_f1f_f2t), np.var(c_ct_f1f_f2t)    
    print np.mean(c_ct_f1f_f2f), np.var(c_ct_f1f_f2f)
    
    plt.hist(c_ct_f1t_f2t, 100, normed=1, facecolor='green', alpha=0.75)
    plt.hist(c_ct_f1t_f2f, 100, normed=1, facecolor='blue', alpha=0.75)
    plt.hist(c_ct_f1f_f2t, 100, normed=1, facecolor='black', alpha=0.75)
    plt.hist(c_ct_f1f_f2f, 100, normed=1, facecolor='red', alpha=0.75)
    plt.show()
    
    
    
    
    
    