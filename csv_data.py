'''
Created on Nov 24, 2014

Loads a uci dataset, assuming in csv format.
Assumes the features are already binary/numeric, except the class.
Assumes no missing features.
'''

import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn import cross_validation
from sklearn.preprocessing import scale
from sklearn import metrics

from utils import pos_neg_evi

if __name__ == '__main__':
    path = "C:\\Users\\mbilgic\\Desktop\\uci\\uci-tar\\nominal\\"
    
    # breast-w
    dataset = "breast-w.csv"
    class_index = 9
    num_cols = 10
    classes = ['benign', 'malignant']
    
    # car
    dataset = "car.csv"
    class_index = 21
    num_cols = 22
    classes = ['acc', 'unacc']
    
    #cmc
    dataset = "cmc.csv"
    class_index = 21
    num_cols = 22
    classes= ['1', '2']
    
    #credit-g
    dataset = "credit-g.csv"
    class_index = 61
    num_cols = 62
    classes= ['good', 'bad']
    
    #diabetes
    dataset = "diabetes.csv"
    class_index = 8
    num_cols = 9
    classes= ['tested_negative', 'tested_positive']
    
    #heart-c
    dataset = "heart-c.csv"
    class_index = 22
    num_cols = 23
    classes= ['<50', '>50_1']
    
    
    #heart-statlog
    dataset = "heart-statlog.csv"
    class_index = 13
    num_cols = 14
    classes= ['absent', 'present']
    
    
    #heart-statlog
    dataset = "hepatitis.csv"
    class_index = 19
    num_cols = 20
    classes= ['DIE', 'LIVE']
    
    
    
    read_cols = [i for i in range(num_cols) if i != class_index]
    
    
    print "Loading the data..."
    
    X = np.loadtxt(path+dataset, dtype=float, delimiter=",", skiprows=1, \
                   usecols=read_cols)
    y = np.loadtxt(path+dataset, dtype=int, delimiter=",", skiprows=1, \
                   usecols=(class_index,), converters={class_index: lambda x: classes.index(x)})
    
    print "Loaded data with shape (%d, %d)"  %X.shape
    
    clf = LogisticRegression()
    
    #z-score scaling
    print "Performing z-score scaling..."
    scale(X, copy=False)
    print "Done."
    
    print "Performing 10 fold cross validation..."
    scores = cross_validation.cross_val_score(clf, X, y, cv=10)
    print "Done."
    
    print "Cross-validation accuracy: %0.4f" %np.mean(scores)
    
    exit(0)
    
    kf = cross_validation.KFold(X.shape[0], shuffle=True, n_folds=10, random_state=42)
    
    for train, test in kf:
        clf = LogisticRegression()
        clf.fit(X[train], y[train])
        y_pred = clf.predict(X[test])
        y_true = y[test]        
        pe, ne = pos_neg_evi(X[test], clf.coef_[0])
        print pe
        print ne
        for i in range(len(y_pred)):
            print "%d\t%d\t%d\t%r\t%0.4f\t%0.4f\t%0.4f" %(i, y_true[i], y_pred[i], y_true[i]==y_pred[i], pe[i], ne[i], clf.intercept_) 
        #print y_pred
        #print pe+ne+clf.intercept_
        #print "accu %0.4f" % metrics.accuracy_score(y[test], y_pred)
        exit(0)
        
    
    
    