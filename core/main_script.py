# -*- coding: utf-8 -*-
"""
Created on Mon Jun 15 00:51:07 2015

@author: viethoangcr
"""

from RandomSamplingSVM import RandomSamplingSVM
import time
from sklearn import datasets
from sklearn import feature_extraction
from sklearn import svm


def single_svm(train_file, test_file, kernel):
    print("Single SVM")
    xTrain, yTrain = datasets.load_svmlight_file(train_file)
    xTest, yTest = datasets.load_svmlight_file(test_file)

    start_time = time.time()
    X_train, y_train = datasets.load_svmlight_file(train_file)
    svc = svm.SVC(**kernel)
    svc.fit(X_train,y_train)
    
    print("Remain SVs: " + str(svc.n_support_))
    print("Training time: %s" % (time.time() - start_time))
    X_test, y_test = datasets.load_svmlight_file(test_file)
    ratio = svc.score(X_test,y_test)
    print("Accuracy %f" % ratio)
    print("Total time: %s" % (time.time() - start_time))
    
def rs_svm(train_file, test_file, kernel):
    nTestingCore = [1, 2, 4, 8, 16]
    start_time = time.time()
    RS_SVM = RandomSamplingSVM(kernel)

    for iTestingCore in range(5):
        start_time = time.time()
        
        model = RS_SVM.train(xTrain, yTrain, beta=0.01, g=1, nCore=nTestingCore[iTestingCore])

        print("Remain SVs: " + str(model.n_support_), flush=True)
        print("Training time: %s" % (time.time() - start_time), flush=True)
        
        testRatio = model.score(xTest, yTest)

        print("Accuracy %f" % testRatio, flush=True)
        print("Total time: %s" % (time.time() - start_time), flush=True)
        print(flush=True)

def rs_svm_ratio(train_file, test_file, kernel):
    print("Ramdom Sampling SVM with Ratio", flush=True)
    xTrain, yTrain = datasets.load_svmlight_file(train_file)
    xTest, yTest = datasets.load_svmlight_file(test_file)
    nTestingCore = [1, 2, 4, 8, 16]
    RS_SVM = RandomSamplingSVM(kernel)

    for iTestingCore in range(5):
        start_time = time.time()
        model = RS_SVM.trainWithRatio(0.5, xTrain, yTrain, beta=0.01, nCore=nTestingCore[iTestingCore])

        print("Remain SVs: " + str(model.n_support_), flush=True)
        print("Training time: %s" % (time.time() - start_time), flush=True)
        
        testRatio = model.score(xTest, yTest)

        print("Accuracy %f" % testRatio, flush=True)
        print("Total time: %s" % (time.time() - start_time), flush=True)
        print(flush=True)
    
def rs_svm_large_file(train_file, test_file, kernel):
    print("Ramdom Sampling SVM for large dataset")
    start_time = time.time()
    RS_SVM = RandomSamplingSVM(kernel)
    model = RS_SVM.train_large_file(train_file, debug=True, temp_folder='./')
    print("Remain SVs: " + str(model.n_support_))
    print("Training time: %s" % (time.time() - start_time))
    X_test, y_test = datasets.load_svmlight_file(test_file)
    ratio = model.score(X_test,y_test)
    print("Accuracy %f" % ratio)
    print("Total time: %s" % (time.time() - start_time))

def signle_svm(train_file, test_file, kernel):
    print("Single SVM", flush=True)
    start_time = time.time()
    RS_SVM = RandomSamplingSVM(kernel)
    model = RS_SVM.train_single(train_file)
    print("Remain SVs: " + str(model.n_support_), flush=True)
    print("Training time: %s" % (time.time() - start_time), flush=True)
    X_test, y_test = datasets.load_svmlight_file(test_file)
    ratio = model.score(X_test,y_test)
    print("Accuracy %f" % ratio, flush=True)
    print("Total time: %s" % (time.time() - start_time), flush=True)  

def test(train_file, test_file, kernel):
    print(flush=True)
    print("DATASET: " + train_file, flush=True)
    
    # Single
#    single_svm(train_file, test_file, kernel)
    
    # RS_SVM using RAM
    rs_svm(train_file, test_file, kernel)
    #print('-----', flush=True)
    rs_svm_ratio(train_file, test_file, kernel)

    
    #RS_SVM using disk as cache
    #rs_svm_large_file(train_file, test_file, kernel)

#isequal = lambda x, y : True if (x-y).nnz == 0 else False
#
#X_train, y_train = datasets.load_svmlight_file(r'./w8a')
#handler = SvmlightFileHandler(r'./w8a')

#start_time = time.time()

svm_para = {'C': 10.0, 'kernel': 'rbf', 'gamma': 1.667, 'verbose': False}
test(r'./dataset/mnist_train_576_rbf_8vr.dat', r'./dataset/mnist_test_576_rbf_8vr.dat', svm_para)

test(r'./dataset/mnist_train_784_poly_8vr.dat', r'./dataset/mnist_test_784_poly_8vr.dat', svm_para)

svm_para = {'C': 10.0, 'kernel': 'rbf', 'gamma': 0.00002, 'tol': 0.01, 'verbose': False}
test(r'./dataset/covtype_tr_2vr.data', r'./dataset/covtype_tst_2vr.data', svm_para)

svm_para = {'C': 1.0, 'kernel': 'rbf', 'gamma': 1/199532, 'verbose': False}
test(r'./dataset/census.train', r'./dataset/census.test', svm_para)
