'''
Created on Nov 11, 2014

@author: mbilgic
'''

import numpy as np
 
from sklearn import linear_model
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from time import time

import glob

from scipy.sparse import diags

from sklearn.linear_model import LogisticRegression

def load_imdb(path, shuffle = True, random_state=42, vectorizer = CountVectorizer(min_df=2, max_df=1.0, binary=False)):
    
    print "Loading the imdb reviews data"
    
    train_neg_files = glob.glob(path+"\\train\\neg\\*.txt")
    train_pos_files = glob.glob(path+"\\train\\pos\\*.txt")
    
    train_corpus = []
    
    y_train = []
    
    for tnf in train_neg_files:
        f = open(tnf, 'r')
        line = f.read()
        train_corpus.append(line)
        y_train.append(0)
        f.close()
    
    for tpf in train_pos_files:
        f = open(tpf, 'r')
        line = f.read()
        train_corpus.append(line)
        y_train.append(1)
        f.close()
    
    test_neg_files = glob.glob(path+"\\test\\neg\\*.txt")
    test_pos_files = glob.glob(path+"\\test\\pos\\*.txt")
    
    test_corpus = []
    
    y_test = []
    
    for tnf in test_neg_files:
        f = open(tnf, 'r')
        test_corpus.append(f.read())
        y_test.append(0)
        f.close()
    
    for tpf in test_pos_files:
        f = open(tpf, 'r')
        test_corpus.append(f.read())
        y_test.append(1)
        f.close()
        
    print "Data loaded."
    
    print "Extracting features from the training dataset using a sparse vectorizer"
    print "Feature extraction technique is %s." % vectorizer
    t0 = time()
    
    X_train = vectorizer.fit_transform(train_corpus)
    
    duration = time() - t0
    print("done in %fs" % (duration))
    print "n_samples: %d, n_features: %d" % X_train.shape
    print
        
    print "Extracting features from the test dataset using the same vectorizer"
    t0 = time()
        
    X_test = vectorizer.transform(test_corpus)
    
    duration = time() - t0
    print("done in %fs" % (duration))
    print "n_samples: %d, n_features: %d" % X_test.shape
    print
    
    y_train = np.array(y_train)
    y_test = np.array(y_test)
    
    if shuffle:
        np.random.seed(random_state)
        indices = np.random.permutation(len(y_train))        
        
        X_train = X_train.tocsr()
        X_train = X_train[indices]
        y_train = y_train[indices]
        train_corpus_shuffled = [train_corpus[i] for i in indices]
        
        
        indices = np.random.permutation(len(y_test))
        
        X_test = X_test.tocsr()
        X_test = X_test[indices]
        y_test = y_test[indices]
        test_corpus_shuffled = [test_corpus[i] for i in indices]
         
    return X_train, y_train, X_test, y_test, train_corpus_shuffled, test_corpus_shuffled


if __name__ == '__main__':
    # Load the data
    
    print "Loading the data"
    
    t0 = time()
    
    #vect = CountVectorizer(min_df=1, max_df=1.0, binary=True, ngram_range=(1, 1))  # , tokenizer=StemTokenizer())
    vect = TfidfVectorizer(min_df=1, max_df=1.0, binary=True, ngram_range=(1, 1))  # , tokenizer=StemTokenizer())
    
    X_train, y_train, X_test, y_test, train_corpus, test_corpus = load_imdb("C:\\Users\\mbilgic\\Desktop\\aclImdb", shuffle=True, vectorizer=vect)
    
    feature_names = vect.get_feature_names()
    
    duration = time() - t0

    print
    print "Loading took %0.2fs." % duration
    print
    
    # Fit a classifier
    
    clf = LogisticRegression()
    clf.fit(X_train, y_train)
    
    coef = clf.coef_[0]
    
    decisions = X_test*coef.T
    
    num_inst, num_feat = X_test.shape
    
    
    coef_gt0 = coef > 0
    coef_lt0 = coef < 0
    
    
    pos_coef = np.zeros(num_feat)
    neg_coef = np.zeros(num_feat)
    
    pos_coef[coef_gt0] = coef[coef_gt0]
    neg_coef[coef_lt0] = coef[coef_lt0]
    
    
    pos_coef_diags = diags(pos_coef,0)
    neg_coef_diags = diags(neg_coef,0)
    
    
    pm = X_test*pos_coef_diags
    nm = X_test*neg_coef_diags
    
    pos_evi = pm.sum(axis=1).A1
    neg_evi = nm.sum(axis=1).A1
    
    sorted_indices = np.argsort(decisions)
    
    probs = clf.predict_proba(X_test)
    
    print
    most_conflicted = 0
    least_conflicted = 0
    max_conflict = 0
    min_conflict = np.Inf
    for i in sorted_indices[:1000]:
        conflict = pos_evi[i]
        if max_conflict < conflict:
            max_conflict = conflict
            most_conflicted = i
        if min_conflict > conflict:
            min_conflict = conflict
            least_conflicted = i
            
    i = sorted_indices[0]
    print decisions[i], probs[i][0], pos_evi[i], neg_evi[i]
    i = most_conflicted
    print decisions[i], probs[i][0], pos_evi[i], neg_evi[i]
    print test_corpus[i]
    i = least_conflicted
    print decisions[i], probs[i][0], pos_evi[i], neg_evi[i]
    print test_corpus[i]
    
    print
    most_conflicted = 0
    least_conflicted = 0
    max_conflict = 0
    min_conflict = np.Inf
    for i in sorted_indices[-1000:]:
        conflict = abs(neg_evi[i])
        if max_conflict < conflict:
            max_conflict = conflict
            most_conflicted = i
        if min_conflict > conflict:
            min_conflict = conflict
            least_conflicted = i
            
    i = sorted_indices[-1]
    print decisions[i], probs[i][0], pos_evi[i], neg_evi[i]
    i = most_conflicted
    print decisions[i], probs[i][0], pos_evi[i], neg_evi[i]
    print test_corpus[i]
    i = least_conflicted
    print decisions[i], probs[i][0], pos_evi[i], neg_evi[i]
    print test_corpus[i]
    
        
    
    # For the most confident instances
    #    Compute precision@k using
    #        Confidence ranking
    #        Least-conflict ranking
    # Print the instance that has the most conflict
    # Print the instance that has the least conflict
    